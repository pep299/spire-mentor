//! REPL-driving human agent. Reads stdin lines, parses combat-screen
//! commands, handles non-action commands in-band (printing to `out`),
//! returns [`AgentChoice`] for action / quit / continue.

use std::io::{self, BufRead, Write};

use smallvec::SmallVec;
use sts2_core::{apply as core_apply, Action, CombatState};
use sts2_mcts::{par_search, Mcts, MctsConfig};

use crate::agent::{Agent, AgentChoice};
use crate::sim::combat::legal_actions;
use crate::sim::parse::{parse, Assertion, CmpOp, Command};
use crate::sim::print;

/// MCTS knobs surfaced on the CLI for the `hint` verb and the
/// auto-hint flag. Cliext G1 (bead `sts2-rs-s3tb`).
#[derive(Clone, Copy, Debug)]
pub struct HintConfig {
    /// When `Some(sims)`, run a `sims`-simulation MCTS hint at the top
    /// of every player decision and print top-3 to `out`.
    pub auto_sims: Option<u32>,
    /// UCT exploration constant (default `sqrt(2)`).
    pub c_puct: f32,
    /// Hard rollout-depth cap.
    pub rollout_depth: u32,
    /// Number of rayon workers for root-parallel MCTS. `1` runs the
    /// single-threaded path with no thread-pool overhead. Defaults to `8`
    /// (root parallelization is ~3.0×/4w, ~5.6×/8w at a fixed sim budget;
    /// see `docs/perf/2026-06-08-mcts-perf-d1-item1-parallelism.md`). The
    /// result is deterministic in `(seed, threads, sims)` regardless of
    /// physical core count.
    pub threads: u32,
}

impl Default for HintConfig {
    fn default() -> Self {
        let d = MctsConfig::default();
        Self {
            auto_sims: None,
            c_puct: d.c_puct,
            rollout_depth: d.max_rollout_depth,
            threads: 8,
        }
    }
}

pub struct HumanAgent<R: BufRead> {
    pub input: R,
    /// Echo each input line to `out` before processing (useful for scripted
    /// replay where stdin and stdout interleave in the golden file).
    /// Echoed lines are prefixed with `<< ` so the input/output stream is
    /// distinguishable in goldens.
    pub echo_input: bool,
    /// Print `> ` prompts before reading.
    pub print_prompts: bool,
    /// Render the combined state + action-menu block at the top of every
    /// `next_action` call (lightspeed-parity auto-prompt).
    pub show_action_menu: bool,
    /// MCTS hint configuration (cliext G1).
    pub hint: HintConfig,
    /// Cache of the last-rendered legal action menu — populated whenever
    /// the auto-prompt block prints, used to resolve `Command::ActionIndex`
    /// (bare-int dispatch).
    last_actions: SmallVec<[Action; 16]>,
}

impl<R: BufRead> HumanAgent<R> {
    pub fn new(input: R) -> Self {
        Self {
            input,
            echo_input: false,
            print_prompts: true,
            show_action_menu: true,
            hint: HintConfig::default(),
            last_actions: SmallVec::new(),
        }
    }
}

impl<R: BufRead> Agent for HumanAgent<R> {
    fn next_action<W: Write>(
        &mut self,
        state: &CombatState,
        out: &mut W,
    ) -> io::Result<AgentChoice> {
        // Lightspeed-parity auto-prompt: render the combined state +
        // action-menu block at the top of every `next_action` call.
        // This fires once per agent decision; in-loop `state`/`hand`/
        // `actions` re-renders are user-requested and don't refresh
        // `last_actions`.
        if self.show_action_menu {
            let actions = legal_actions(state);
            print::print_combat_prompt(out, state, &actions)?;
            self.last_actions = SmallVec::from_vec(actions);
        }
        if let Some(sims) = self.hint.auto_sims {
            // Auto-hint runs the main-search-only path; fair-eval is
            // on-demand via the `hint --fair` REPL verb.
            run_hint(
                out,
                state,
                sims,
                self.hint.c_puct,
                self.hint.rollout_depth,
                self.hint.threads,
                None,
                false,
            )?;
        }
        loop {
            if self.print_prompts {
                write!(out, "> ")?;
                out.flush()?;
            }
            let mut line = String::new();
            let n = self.input.read_line(&mut line)?;
            if n == 0 {
                return Ok(AgentChoice::QuitRepl); // EOF
            }
            if self.echo_input {
                writeln!(out, "<< {}", line.trim_end())?;
            }
            match parse(&line) {
                Ok(Command::Blank | Command::Comment(_)) => {}
                Ok(Command::Action(a)) => return Ok(AgentChoice::Action(a)),
                Ok(Command::ActionIndex(i)) => {
                    let idx = usize::from(i);
                    if let Some(action) = self.last_actions.get(idx).cloned() {
                        return Ok(AgentChoice::Action(action));
                    }
                    writeln!(
                        out,
                        "  no action at menu index {i} (menu size {})",
                        self.last_actions.len()
                    )?;
                }
                Ok(Command::Hand) => print::print_hand(out, state)?,
                Ok(Command::State) => print::print_combat_state(out, state)?,
                Ok(Command::Actions) => print::print_legal_actions(out, state)?,
                Ok(Command::Info(i)) => print::print_card_info(out, state, i)?,
                Ok(Command::Help) => print::print_help(out)?,
                Ok(Command::Quit) => return Ok(AgentChoice::QuitRepl),
                Ok(Command::Assert(a)) => {
                    if let Some(msg) = check_assertion(state, &a) {
                        return Ok(AgentChoice::AssertFailed(msg));
                    }
                    writeln!(out, "  assert ok: {a:?}")?;
                }
                Ok(Command::Hint {
                    sims,
                    fair_per_action,
                    fair_all,
                }) => {
                    // Explicit `hint <N>` overrides; otherwise inherit
                    // the auto-hint budget (`--mcts-hint-default`) so
                    // bare `hint` and the auto-prompt produce the same
                    // numbers. Final fallback is the legacy 1000.
                    let resolved_sims = sims.or(self.hint.auto_sims).unwrap_or(1000);
                    // `--fair[-all]` without `=N` inherits the resolved
                    // main `sims` so `hint 20000 --fair-all` runs the
                    // per-action fair-eval at 20000 too. `=N` keeps the
                    // user's explicit override.
                    let resolved_fair = fair_per_action.map(|inner| inner.unwrap_or(resolved_sims));
                    run_hint(
                        out,
                        state,
                        resolved_sims,
                        self.hint.c_puct,
                        self.hint.rollout_depth,
                        self.hint.threads,
                        resolved_fair,
                        fair_all,
                    )?;
                }
                Ok(Command::Eval { action_idx, sims }) => {
                    let idx = usize::from(action_idx);
                    let Some(action) = self.last_actions.get(idx).cloned() else {
                        writeln!(
                            out,
                            "  eval: no action at menu index {action_idx} (menu size {})",
                            self.last_actions.len(),
                        )?;
                        continue;
                    };
                    run_action_eval(
                        out,
                        state,
                        &action,
                        sims,
                        self.hint.c_puct,
                        self.hint.rollout_depth,
                        self.hint.threads,
                    )?;
                }
                Err(e) => writeln!(out, "  parse error: {e}")?,
            }
        }
    }
}

/// Threshold for "low-confidence" annotation: any root child with
/// visits < this fraction of the most-visited child is flagged. UCB
/// stops sending visits to branches that early rollouts judge weak, so
/// their Q (and the derived win%) is biased downward and reflects mostly
/// shallow random play — not "win% if I commit to this action and play
/// well." See bead `sts2-rs-60y1`.
const LOW_CONFIDENCE_VISIT_FRACTION: f64 = 0.05;

/// Number of top-by-visits actions to evaluate independently in
/// fair-eval mode (the `hint --fair` REPL verb). Default per-action
/// sim budget lives in the parser (`parse_hint`'s
/// `DEFAULT_FAIR_PER_ACTION`).
const FAIR_EVAL_TOP_K: usize = 3;

/// Run an MCTS hint search on `state` for `sims` iterations and print
/// the top-3 root actions to `out`. Does NOT mutate `state` (the
/// engine operates on a clone). Bead `sts2-rs-s3tb` (cliext G1).
///
/// When `fair_per_action` is `Some(n)`, additionally run an independent
/// MCTS from the post-action state for each of the top-K root actions
/// (K = `FAIR_EVAL_TOP_K`) with `n` sims each, and report their own
/// search's win/tie/loss rates. These numbers are NOT subject to the
/// main search's UCB selection bias and answer "win% if I commit to
/// this action and play well from there." Bead `sts2-rs-60y1`.
#[allow(clippy::too_many_arguments)]
fn run_hint<W: Write>(
    out: &mut W,
    state: &CombatState,
    sims: u32,
    c_puct: f32,
    rollout_depth: u32,
    threads: u32,
    fair_per_action: Option<u32>,
    fair_all: bool,
) -> io::Result<()> {
    let actions = legal_actions(state);
    if actions.is_empty() {
        writeln!(out, "  hint: no legal actions (terminal state)")?;
        return Ok(());
    }
    let config = MctsConfig {
        c_puct,
        max_simulations: sims,
        max_rollout_depth: rollout_depth,
        seed: 0,
        ..MctsConfig::default()
    };
    let (mut outcomes, pick) = if threads <= 1 {
        let mut mcts = Mcts::new(state.clone(), config);
        let _best = mcts.search_n(sims);
        let pick = mcts.explain_root_pick().ok();
        (mcts.root_outcomes(), pick)
    } else {
        let r = par_search(state.clone(), config, threads);
        (r.outcomes, Some(r.pick))
    };
    let pick_action = pick.as_ref().map(|p| p.action.clone());
    let band: Vec<Action> = pick.as_ref().map_or_else(Vec::new, |p| {
        p.band.iter().map(|&i| outcomes[i].action.clone()).collect()
    });
    // Primary sort: Q descending (the actual "best" signal). Secondary
    // sort: visits descending (tiebreaker — more visits = more
    // confidence). Q-first ranking matters on saturated states where
    // every rollout wins, the visit distribution drifts toward uniform
    // via PUCT's exploration term and visit-rank becomes near-noise.
    outcomes.sort_by(|a, b| {
        b.mean_value
            .partial_cmp(&a.mean_value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.visits.cmp(&a.visits))
    });
    // The move the search itself plays (`Mcts::explain_root_pick`) leads,
    // so `#1` is the run agent's move, not a Q re-rank that disagrees with
    // it on ties and guards (GH #1022).
    if let Some(pos) = pick_action
        .as_ref()
        .and_then(|a| outcomes.iter().position(|o| o.action == *a))
    {
        outcomes[..=pos].rotate_right(1);
    }
    let total: u32 = outcomes.iter().map(|o| o.visits).sum();
    let denom = f64::from(total.max(1));
    // Q-saturation detection: if Q-spread across root children is
    // below this epsilon, the rollouts can't distinguish actions and
    // the ranking is essentially noise. Surface a warning rather than
    // pretending the top entry is meaningfully "best".
    let q_min = outcomes
        .iter()
        .map(|o| o.mean_value)
        .fold(f32::INFINITY, f32::min);
    let q_max = outcomes
        .iter()
        .map(|o| o.mean_value)
        .fold(f32::NEG_INFINITY, f32::max);
    let q_saturated = outcomes.len() > 1 && (q_max - q_min).abs() < 1e-3;
    writeln!(
        out,
        "  hint ({sims} sims, {threads} threads, {total} root visits, {n} actions):",
        n = outcomes.len(),
    )?;
    if q_saturated {
        writeln!(
            out,
            "    note: Q saturated at {q_max:+.3} — every rollout reaches the same \
             terminal value, so MCTS has no signal to rank actions on this state. \
             Visit counts drift toward uniform via PUCT exploration. Try a harder \
             encounter or deeper rollout cap (--mcts-rollout-depth)."
        )?;
    }
    // Low-confidence threshold is relative to the *most-visited* row
    // (which after the Q/visit sort is usually outcomes[0], but Q-sort
    // can put a lightly-visited high-Q row first — so we take the max
    // across all outcomes).
    let max_visits = outcomes.iter().map(|o| o.visits).max().unwrap_or(0);
    // Round up to the nearest u32, clamped to [1, u32::MAX]. The
    // input `max_visits * fraction` is bounded above by `max_visits`
    // so the cast is always representable as u32.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let low_visit_threshold =
        ((f64::from(max_visits) * LOW_CONFIDENCE_VISIT_FRACTION).ceil() as u32).max(1);
    let mut any_low_confidence = false;
    for (rank, o) in outcomes.iter().enumerate() {
        let share = 100.0 * f64::from(o.visits) / denom;
        let vdenom = f64::from(o.visits.max(1));
        let pw = 100.0 * f64::from(o.wins) / vdenom;
        let low_conf = o.visits < low_visit_threshold;
        if low_conf {
            any_low_confidence = true;
        }
        // ~ prefix on Q and win/tie/loss for low-visit rows so the
        // numbers don't look authoritative. Bead `sts2-rs-60y1`.
        let q_pre = if low_conf { "≈" } else { "=" };
        let pct_pre = if low_conf { "≈" } else { "=" };
        let tag = if low_conf { "  (low-confidence)" } else { "" };
        // `[M]` is the dispatchable action-menu index — typing it
        // plays that action. `#R` is just the rank within this hint
        // table. Decoupling them lets the user type the bracketed
        // number from any row to play it.
        let menu_idx = actions.iter().position(|a| *a == o.action);
        let menu_tag = menu_idx.map_or_else(|| "[--]".to_string(), |m| format!("[{m}]"));
        let action_desc = print::format_action_for_menu(state, &o.action);
        writeln!(
            out,
            "    #{rank} → {menu_tag} {action_desc} — {v} visits ({share:.1}%), \
             Q{q_pre}{q:+.3}, win{pct_pre}{pw:.1}%{tag}",
            rank = rank + 1,
            v = o.visits,
            q = o.mean_value,
        )?;
    }
    if let Some(p) = pick.as_ref().filter(|_| band.len() > 1) {
        let others: Vec<String> = band
            .iter()
            .filter(|a| **a != p.action)
            .map(|a| print::format_action_for_menu(state, a))
            .collect();
        writeln!(
            out,
            "    pick: {} — {} (about equal: {})",
            print::format_action_for_menu(state, &p.action),
            p.rule.reason(),
            others.join(", "),
        )?;
    }
    if any_low_confidence {
        writeln!(
            out,
            "    note: low-confidence rows had <{thr:.0}% of the top action's visits — \
             UCB stopped exploring them once early rollouts looked weak, so their Q is \
             biased by shallow random play, not \"win% if I commit and play well.\" \
             Use `hint --fair` to evaluate alternatives independently.",
            thr = LOW_CONFIDENCE_VISIT_FRACTION * 100.0,
        )?;
    }

    if let Some(per_action_sims) = fair_per_action {
        run_fair_eval(
            out,
            state,
            &outcomes,
            per_action_sims,
            &config,
            threads,
            fair_all,
        )?;
    }
    Ok(())
}

// Run an independent MCTS search for each of the top-K root actions
// (ranked by visit count from the main search) and print each one's
// own-search win / tie / loss. Per-action searches are completely
// fresh (no warm prior from the main tree), so the reported win% is
// free of the main-tree UCB selection bias and answers
// "win% if I commit to this action and play well from there."
//
// Bead `sts2-rs-60y1`.
fn run_fair_eval<W: Write>(
    out: &mut W,
    state: &CombatState,
    main_outcomes: &[sts2_mcts::RootOutcome],
    per_action_sims: u32,
    base_config: &MctsConfig,
    threads: u32,
    fair_all: bool,
) -> io::Result<()> {
    // Pick top-K by VISIT count (not Q), because that's what the user
    // sees as "rank-1, rank-2, ..." in the main table and what they're
    // most likely to consider as alternatives. Visit-rank is more
    // robust than Q-rank to the saturation case. When `fair_all` is
    // set, evaluate every action instead (request: `--fair-all`).
    let mut by_visits: Vec<&sts2_mcts::RootOutcome> = main_outcomes.iter().collect();
    by_visits.sort_by_key(|o| std::cmp::Reverse(o.visits));
    let take_k = if fair_all {
        by_visits.len()
    } else {
        FAIR_EVAL_TOP_K
    };
    let top: Vec<&sts2_mcts::RootOutcome> = by_visits.into_iter().take(take_k).collect();

    let scope = if fair_all {
        format!("all {n}", n = top.len())
    } else {
        format!("top-{k} by visits", k = top.len())
    };
    writeln!(
        out,
        "  fair-eval ({per_action_sims} sims per action, {scope}) — \
         independent searches; numbers reflect own-search root-Q, not main-tree visits:",
    )?;

    let cfg = MctsConfig {
        max_simulations: per_action_sims,
        ..*base_config
    };
    for (i, o) in top.iter().enumerate() {
        let mut post = state.clone();
        let _ = core_apply(&mut post, o.action.clone());
        // After applying, the resulting state may already be terminal
        // (e.g. a killing Strike on a 1-HP enemy). In that case
        // `legal_actions` returns empty and we'd panic on `search()`,
        // so resolve the value directly via terminal_value.
        let post_actions = legal_actions(&post);
        let (visits, wins, losses, _ties, source) = if post_actions.is_empty() {
            // GH#296: must match the scale the sibling actions are scored on
            // (a `cfg.terminal_eval_mode` search). Raw `terminal_value` is the
            // Binary scale and put immediate wins below stalls under RichBoth.
            let v = sts2_mcts::terminal_state_value(
                &post,
                cfg.terminal_eval_mode,
                &sts2_mcts::LeafEval::default(),
                cfg.loss_gradient,
            );
            // Encode the terminal outcome as a single deterministic
            // sample: 1 visit, bucket selected by v.
            let (w, l, t) = if v > 0.5 {
                (1, 0, 0)
            } else if v < -0.5 {
                (0, 1, 0)
            } else {
                (0, 0, 1)
            };
            (1u32, w, l, t, "terminal")
        } else {
            let (n, w, l, t) = if threads <= 1 {
                let mut m = Mcts::new(post, cfg);
                let _ = m.search();
                aggregate_outcomes(&m.root_outcomes())
            } else {
                let r = par_search(post, cfg, threads);
                aggregate_outcomes(&r.outcomes)
            };
            (n, w, l, t, "mcts")
        };
        let vdenom = f64::from(visits.max(1));
        let pw = 100.0 * f64::from(wins) / vdenom;
        #[allow(clippy::cast_precision_loss)]
        let q = if visits == 0 {
            0.0
        } else {
            (wins as f32 - losses as f32) / visits as f32
        };
        let menu_idx = main_outcomes
            .iter()
            .position(|root| root.action == o.action) // rank-by-visits inputs preserve `.action`
            .and_then(|_| {
                // Resolve back to the live legal_actions order so the
                // bracketed index matches what's typeable in the menu.
                let acts = legal_actions(state);
                acts.iter().position(|a| *a == o.action)
            });
        let menu_tag = menu_idx.map_or_else(|| "[--]".to_string(), |m| format!("[{m}]"));
        let action_desc = print::format_action_for_menu(state, &o.action);
        writeln!(
            out,
            "    #{rank} → {menu_tag} {action_desc} — Q={q:+.3}, win={pw:.1}%  ({source}, {visits} sims)",
            rank = i + 1,
        )?;
    }
    Ok(())
}

/// Run an independent MCTS from the state that results from applying
/// `action`, then print its win/tie/loss. This is the single-action
/// version of `hint --fair`: instead of evaluating the top-K by
/// visits, the user picks exactly which action they want diagnosed.
/// Useful when you want to ask "what happens if I commit to *this*
/// specific card play?" and trust the answer regardless of how the
/// main-search UCB would have ranked it.
///
/// Output mirrors a fair-eval row. Bead `sts2-rs-60y1` (extension).
#[allow(clippy::too_many_arguments)]
fn run_action_eval<W: Write>(
    out: &mut W,
    state: &CombatState,
    action: &Action,
    sims: u32,
    c_puct: f32,
    rollout_depth: u32,
    threads: u32,
) -> io::Result<()> {
    let mut post = state.clone();
    let _ = core_apply(&mut post, action.clone());
    let post_actions = legal_actions(&post);
    let (visits, wins, losses, _ties, source) = if post_actions.is_empty() {
        let (_, v) = sts2_core::terminal_value(&post);
        let (w, l, t) = if v > 0.5 {
            (1, 0, 0)
        } else if v < -0.5 {
            (0, 1, 0)
        } else {
            (0, 0, 1)
        };
        (1u32, w, l, t, "terminal")
    } else {
        let cfg = MctsConfig {
            c_puct,
            max_simulations: sims,
            max_rollout_depth: rollout_depth,
            seed: 0,
            ..MctsConfig::default()
        };
        let (n, w, l, t) = if threads <= 1 {
            let mut m = Mcts::new(post, cfg);
            let _ = m.search();
            aggregate_outcomes(&m.root_outcomes())
        } else {
            let r = par_search(post, cfg, threads);
            aggregate_outcomes(&r.outcomes)
        };
        (n, w, l, t, "mcts")
    };
    let vdenom = f64::from(visits.max(1));
    let pw = 100.0 * f64::from(wins) / vdenom;
    #[allow(clippy::cast_precision_loss)]
    let q = if visits == 0 {
        0.0
    } else {
        (wins as f32 - losses as f32) / visits as f32
    };
    writeln!(
        out,
        "  action-eval ({sims} sims, {threads} threads) for {action:?}:\n    \
         Q={q:+.3}, win={pw:.1}%  ({source}, {visits} sims)",
    )?;
    Ok(())
}

// Aggregate (visits, wins, losses, ties) across all root children.
// For a single-threaded `Mcts::search`, the sum equals the root node's
// own counters because every simulation backpropagates through the
// root.
fn aggregate_outcomes(outcomes: &[sts2_mcts::RootOutcome]) -> (u32, u32, u32, u32) {
    let mut w = 0u32;
    let mut l = 0u32;
    let mut t = 0u32;
    let mut n = 0u32;
    for o in outcomes {
        w += o.wins;
        l += o.losses;
        t += o.ties;
        n += o.visits;
    }
    (n, w, l, t)
}

fn check_assertion(state: &CombatState, a: &Assertion) -> Option<String> {
    match a {
        Assertion::PlayerHp(op, rhs) => {
            let lhs = state.player.current_hp;
            if op.check(&lhs, rhs) {
                None
            } else {
                Some(format!(
                    "player_hp {lhs} not {op:?} {rhs}",
                    op = pretty_op(*op)
                ))
            }
        }
        Assertion::Phase(name) => {
            let actual = format!("{:?}", state.phase);
            if actual.eq_ignore_ascii_case(name) {
                None
            } else {
                Some(format!("phase {actual} != expected {name}"))
            }
        }
        Assertion::MonstersAlive(n) => {
            let alive = state.creatures.iter().filter(|c| c.current_hp > 0).count();
            if alive == *n {
                None
            } else {
                Some(format!("monsters_alive {alive} != expected {n}"))
            }
        }
    }
}

const fn pretty_op(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Eq => "==",
        CmpOp::Ge => ">=",
        CmpOp::Gt => ">",
        CmpOp::Ne => "!=",
    }
}

#[cfg(test)]
#[allow(deprecated)] // exercises the deprecated EncounterKind shim
mod hint_tests {
    use super::*;
    use crate::sim::{build_starter_state, EncounterKind, PlayerClass};

    // Smoke test the hint runner on a known-good fixture
    // (Ironclad vs Cultist, seed=0). The starter state already has
    // legal actions on turn 1 so the run must emit at least one
    // ranked action line.
    #[test]
    fn run_hint_smoke_emits_top_actions() {
        let state = build_starter_state(0, EncounterKind::Nibbit, PlayerClass::Ironclad, None);
        let mut buf: Vec<u8> = Vec::new();
        // 100 sims keeps the test under 100ms on the dev box; the
        // smoke test only cares that the output shape is sound.
        run_hint(
            &mut buf,
            &state,
            100,
            std::f32::consts::SQRT_2,
            200,
            1,
            None,
            false,
        )
        .unwrap();
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("hint ("), "expected header line, got: {text}");
        assert!(
            text.contains("#1"),
            "expected at least one ranked action (rank `#1`), got: {text}"
        );
    }
}
