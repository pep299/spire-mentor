//! `combat_asc_eval` — paired closed-loop A8–A10 combat evaluator
//! (Phase-0 instrument of the 2026-08-15 high-ascension campaign;
//! spec: docs/superpowers/specs/2026-08-15-mcts-highasc-optimization-design.md).
//!
//! Plays every corpus fixture to terminal in the DEPLOYED honest
//! dialect; the 19 knobs listed below are overridable as `k=v` argv
//! (lightspeed `tune_optuna.py` contract) — this is the arm-E tuning
//! surface, NOT every `MctsConfig` field (no `rollout_policy`,
//! `use_quiescence`, `fpu_reduction`, `chance_prove_wins`, SH, …).
//! Fixture i × replica r maps
//! to a fixed search-seed stream identical across arms (common random
//! numbers across arms, NOT within a tree), so two invocations with
//! different knobs are paired per (fixture, replica).
//!
//! ```text
//! cargo run --release -p sts2-bench --example combat_asc_eval -- \
//!     --corpus <corpus.json> [--replicas 3] [--seed-salt 0] \
//!     [--jobs 0=rayon-default] [--out fights.jsonl] \
//!     [knob=value ...]
//! knobs: sims (default 1000), depth (200), c_puct (2.5),
//!     chance_widening_c/alpha, draw_widening_c (2.0)/alpha,
//!     endturn_widening_c/alpha, tie_eps (0.02),
//!     loss_gradient (1), loss_w_removed (0.6), loss_w_hp (0.25),
//!     loss_w_turns (0.15), loss_turns_half_life (8.0), loss_span (0.8),
//!     loot_weight (1.0), root_guards (1), tie_rules (block+free+junk),
//!     dominance_pruning (0)
//! ```
//!
//! Output: one JSONL line per fight, `--out.partial` streamed +
//! flushed as each fight completes (a stuck run is visible without
//! waiting for the whole corpus), then the FINAL `--out` is written
//! once at the end sorted by `(state_id, replica)` so two identical
//! invocations diff byte-identical (mod the per-row `wall_ms` timing
//! field) regardless of which rayon worker happened to finish which
//! fight first — the streamed `.partial` is progress-only and is
//! removed on a clean exit. Finally: `SCORE mean winrate avg_hp n` on
//! stdout, cap-hit-rate on stderr.

#[path = "common/mod.rs"]
mod common;

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sts2_cli::scenario_to_combat_state_with_ascension;
use sts2_cli::sim::scenario::toml_file::load_scenario_file;
use sts2_core::{apply, is_terminal, legal_actions, CombatState};
use sts2_mcts::{HiddenInfoMode, Mcts, MctsConfig, TerminalEvalMode};

const POTION_HP_EQUIV: f32 = 8.0; // matches VICTORY_POTION_HP_EQUIV
const LOSS_SCORE: f32 = -200.0;
const MAX_ACTIONS_PER_FIGHT: u32 = 1000;
/// Raw per-fight applied-action ceiling, counted on EVERY `apply()` call
/// (forced single-legal-action moves included, not just searched
/// decisions). `MAX_ACTIONS_PER_FIGHT` alone does not bound a run of
/// consecutive forced moves — `decision` only increments on a searched
/// decision — so a stuck forced-move chain would spin forever with no
/// independent ceiling. The engine has a known failure class here: the
/// `fire_late_effects` infinite-loop hang that forced `run_arena` to add
/// per-seed timeouts (see the multi-act probe findings). 20k is far above
/// any real fight (typical fights finish in tens of decisions / a few
/// hundred applied actions including forced ones).
const MAX_STEPS_PER_FIGHT: u32 = 20_000;

/// Reader-only mirror of the `RealisticCorpus`/`onmanifold` shape
/// `corpus_gen --runmanager-toml-export` writes (see
/// `mcts_c_puct_sweep.rs:70-100` for the sibling pattern). Extra
/// top-level fields (`version`, `git_sha`, `generator`, `bucket_globs`)
/// are ignored by serde's default "unknown fields OK" behavior — this
/// reader only needs `entries`.
#[derive(Debug, Clone, Deserialize)]
struct CorpusFile {
    entries: Vec<CorpusEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct CorpusEntry {
    state_id: u32,
    #[allow(dead_code)]
    tier: String,
    fixture_path: String,
    #[allow(dead_code)]
    floor: u16,
    seed: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct FightRow {
    state_id: u32,
    replica: u32,
    won: bool,
    hp: f32,
    potions: u32,
    score: f32,
    decisions: u32,
    rollouts: u64,
    cap_hits: u64,
    wall_ms: u64,
    /// True iff this fight hit `MAX_ACTIONS_PER_FIGHT` or
    /// `MAX_STEPS_PER_FIGHT` without reaching `is_terminal` — i.e. it was
    /// forcibly stopped rather than actually decided. Scored as a loss
    /// (`won=false`, `score=LOSS_SCORE`) regardless of the engine state at
    /// cutoff. Downstream analysis should count/exclude these rows: a
    /// nonzero rate signals a stuck fight (the same failure class as the
    /// `fire_late_effects` infinite-loop hang), not a search quality
    /// finding.
    timed_out: bool,
}

/// Every `MctsConfig` knob overridable from argv, defaults = the
/// deployed honest dialect (BeliefDraw, RichBoth, depth 200, c_puct
/// 2.5, draw_widening_c 2.0 — see `deployed_config`).
#[derive(Debug, Clone)]
struct Knobs {
    sims: u32,
    depth: u32,
    c_puct: f32,
    chance_widening_c: f32,
    chance_widening_alpha: f32,
    draw_widening_c: f32,
    draw_widening_alpha: f32,
    endturn_widening_c: f32,
    endturn_widening_alpha: f32,
    tie_eps: f32,
    loss_gradient: bool,
    loss_w_removed: f32,
    loss_w_hp: f32,
    loss_w_turns: f32,
    loss_turns_half_life: f32,
    loss_span: f32,
    loot_weight: f32,
    root_guards: bool,
    /// `MctsConfig::tie_rules` (GH #1022), a `+`-joined rule list.
    tie_rules: sts2_mcts::TieRules,
    /// `MctsConfig::dominance_pruning` (default off, as deployed).
    dominance_pruning: bool,
}

impl Default for Knobs {
    fn default() -> Self {
        // Anything not called out explicitly in the CLI contract
        // (chance_widening_*, endturn_widening_*) inherits
        // `MctsConfig::default()` rather than a hand-picked constant,
        // so this struct can't silently drift from the engine default.
        let d = MctsConfig::default();
        Self {
            sims: 1000,
            depth: 200,
            c_puct: 2.5,
            chance_widening_c: d.chance_widening_c,
            chance_widening_alpha: d.chance_widening_alpha,
            draw_widening_c: 2.0,
            draw_widening_alpha: d.draw_widening_alpha,
            endturn_widening_c: d.endturn_widening_c,
            endturn_widening_alpha: d.endturn_widening_alpha,
            tie_eps: 0.02,
            loss_gradient: true,
            loss_w_removed: d.loss_w_removed,
            loss_w_hp: d.loss_w_hp,
            loss_w_turns: d.loss_w_turns,
            loss_turns_half_life: d.loss_turns_half_life,
            loss_span: d.loss_span,
            loot_weight: d.loot_weight,
            root_guards: true,
            tie_rules: d.tie_rules,
            dominance_pruning: d.dominance_pruning,
        }
    }
}

impl Knobs {
    /// Apply one `key=value` argv token. Panics on an unrecognized
    /// `key` — a typo in a knob name must not silently no-op (it would
    /// otherwise look like the knob had no effect, corrupting a tuning
    /// sweep instead of failing loudly).
    fn apply(&mut self, key: &str, val: &str) {
        fn f32_of(key: &str, val: &str) -> f32 {
            val.parse()
                .unwrap_or_else(|e| panic!("knob {key}: invalid f32 {val:?} ({e})"))
        }
        fn u32_of(key: &str, val: &str) -> u32 {
            val.parse()
                .unwrap_or_else(|e| panic!("knob {key}: invalid u32 {val:?} ({e})"))
        }
        fn bool_of(key: &str, val: &str) -> bool {
            match val {
                "1" | "true" => true,
                "0" | "false" => false,
                other => panic!("knob {key}: invalid bool {other:?} (expected 0/1/true/false)"),
            }
        }
        match key {
            "sims" => self.sims = u32_of(key, val),
            "depth" => self.depth = u32_of(key, val),
            "c_puct" => self.c_puct = f32_of(key, val),
            "chance_widening_c" => self.chance_widening_c = f32_of(key, val),
            "chance_widening_alpha" => self.chance_widening_alpha = f32_of(key, val),
            "draw_widening_c" => self.draw_widening_c = f32_of(key, val),
            "draw_widening_alpha" => self.draw_widening_alpha = f32_of(key, val),
            "endturn_widening_c" => self.endturn_widening_c = f32_of(key, val),
            "endturn_widening_alpha" => self.endturn_widening_alpha = f32_of(key, val),
            "tie_eps" => self.tie_eps = f32_of(key, val),
            "loss_gradient" => self.loss_gradient = bool_of(key, val),
            "loss_w_removed" => self.loss_w_removed = f32_of(key, val),
            "loss_w_hp" => self.loss_w_hp = f32_of(key, val),
            "loss_w_turns" => self.loss_w_turns = f32_of(key, val),
            "loss_turns_half_life" => self.loss_turns_half_life = f32_of(key, val),
            "loss_span" => self.loss_span = f32_of(key, val),
            "loot_weight" => self.loot_weight = f32_of(key, val),
            "root_guards" => self.root_guards = bool_of(key, val),
            "tie_rules" => {
                self.tie_rules =
                    sts2_mcts::TieRules::parse(val).unwrap_or_else(|e| panic!("knob {key}: {e}"));
            }
            "dominance_pruning" => self.dominance_pruning = bool_of(key, val),
            other => panic!(
                "unknown knob {other:?} — typo? known knobs: sims, depth, c_puct, \
                 chance_widening_c, chance_widening_alpha, draw_widening_c, \
                 draw_widening_alpha, endturn_widening_c, endturn_widening_alpha, \
                 tie_eps, loss_gradient, loss_w_removed, loss_w_hp, \
                 loss_w_turns, loss_turns_half_life, loss_span, loot_weight, \
                 root_guards, tie_rules, dominance_pruning"
            ),
        }
    }
}

struct Args {
    corpus: PathBuf,
    replicas: u32,
    seed_salt: u64,
    jobs: usize,
    out: PathBuf,
    knobs: Knobs,
}

const USAGE: &str = "Usage: combat_asc_eval --corpus <corpus.json> [OPTIONS] [knob=value ...]\n\n\
Options:\n  \
--corpus PATH      Corpus JSON (RealisticCorpus/onmanifold shape) — required\n  \
--replicas N       Replicas per fixture (default 3)\n  \
--seed-salt N      Arm-invariant pairing salt (default 0)\n  \
--jobs N           rayon worker threads, 0 = rayon default (default 0)\n  \
--out PATH         JSONL output path (default fights.jsonl)\n\n\
Knobs (bare k=v tokens): sims (1000), depth (200), c_puct (2.5), \
chance_widening_c/alpha, draw_widening_c (2.0)/alpha, endturn_widening_c/alpha, \
tie_eps (0.02), loss_gradient (1), loss_w_removed (0.6), \
loss_w_hp (0.25), loss_w_turns (0.15), loss_turns_half_life (8.0), \
loss_span (0.8), loot_weight (1.0), root_guards (1), tie_rules (block+free+junk), \
dominance_pruning (0)";

fn parse_args() -> Args {
    let mut corpus: Option<PathBuf> = None;
    let mut replicas: u32 = 3;
    let mut seed_salt: u64 = 0;
    let mut jobs: usize = 0;
    let mut out = PathBuf::from("fights.jsonl");
    let mut knobs = Knobs::default();

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        if arg == "--help" || arg == "-h" {
            eprintln!("{USAGE}");
            std::process::exit(0);
        } else if let Some(flag) = arg.strip_prefix("--") {
            let value = argv
                .get(i + 1)
                .unwrap_or_else(|| panic!("flag --{flag} requires a value"));
            match flag {
                "corpus" => corpus = Some(PathBuf::from(value)),
                "replicas" => {
                    replicas = value
                        .parse()
                        .unwrap_or_else(|e| panic!("--replicas: invalid u32 {value:?} ({e})"));
                }
                "seed-salt" => {
                    seed_salt = value
                        .parse()
                        .unwrap_or_else(|e| panic!("--seed-salt: invalid u64 {value:?} ({e})"));
                }
                "jobs" => {
                    jobs = value
                        .parse()
                        .unwrap_or_else(|e| panic!("--jobs: invalid usize {value:?} ({e})"));
                }
                "out" => out = PathBuf::from(value),
                other => panic!("unknown flag --{other} (see USAGE: {USAGE})"),
            }
            i += 2;
        } else if let Some(eq) = arg.find('=') {
            knobs.apply(&arg[..eq], &arg[eq + 1..]);
            i += 1;
        } else {
            panic!("unrecognized argument {arg:?} (see USAGE: {USAGE})");
        }
    }

    let corpus = corpus.unwrap_or_else(|| panic!("--corpus is required (see USAGE: {USAGE})"));
    Args {
        corpus,
        replicas,
        seed_salt,
        jobs,
        out,
        knobs,
    }
}

/// Deployed-dialect `MctsConfig`: BeliefDraw hidden info, RichBoth
/// terminal eval, every other knob from `k` (defaults = the deployed
/// honest dialect per campaign spec anchor). Leaf eval is whatever
/// `Mcts` uses by default (TexelWide) — not overridden here.
fn deployed_config(k: &Knobs, seed: u64) -> MctsConfig {
    let mut c = MctsConfig {
        max_simulations: k.sims,
        max_rollout_depth: k.depth,
        seed,
        terminal_eval_mode: TerminalEvalMode::RichBoth,
        hidden_info: HiddenInfoMode::BeliefDraw,
        ..MctsConfig::default()
    };
    c.c_puct = k.c_puct; // 2.5 default
    c.draw_widening_c = k.draw_widening_c; // 2.0 default
    c.chance_widening_c = k.chance_widening_c;
    c.chance_widening_alpha = k.chance_widening_alpha;
    c.draw_widening_alpha = k.draw_widening_alpha;
    c.endturn_widening_c = k.endturn_widening_c;
    c.endturn_widening_alpha = k.endturn_widening_alpha;
    c.tie_eps = k.tie_eps;
    c.loss_gradient = k.loss_gradient;
    c.loss_w_removed = k.loss_w_removed;
    c.loss_w_hp = k.loss_w_hp;
    c.loss_w_turns = k.loss_w_turns;
    c.loss_turns_half_life = k.loss_turns_half_life;
    c.loss_span = k.loss_span;
    c.loot_weight = k.loot_weight;
    c.root_guards = k.root_guards;
    c.tie_rules = k.tie_rules;
    c.dominance_pruning = k.dominance_pruning;
    c
}

/// splitmix64 — the arm-invariant pairing seed. Fixture i × replica r
/// × decision d maps to the SAME seed stream across arms (only the
/// knobs argv changes), so two invocations differing only in a knob
/// value are paired per (fixture, replica, decision) — common random
/// numbers across arms, not within a tree.
fn pair_seed(state_id: u32, replica: u32, decision: u32, salt: u64) -> u64 {
    let mut z = (u64::from(state_id) << 40)
        ^ (u64::from(replica) << 20)
        ^ u64::from(decision)
        ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Play one (fixture, replica) to terminal under the deployed dialect.
/// `HiddenInfoMode::BeliefDraw` requires a fresh `Mcts` per decision
/// (reroot asserts) — do not try to reuse a tree across decisions here.
fn play_one(
    template: &CombatState,
    knobs: &Knobs,
    state_id: u32,
    replica: u32,
    salt: u64,
) -> FightRow {
    let mut s = template.clone();
    let mut decision: u32 = 0;
    // Raw applied-action counter, incremented on EVERY `apply()` call
    // (forced moves included) — the independent ceiling `decision` alone
    // can't provide, since forced-move chains never touch `decision`.
    let mut steps: u32 = 0;
    let mut rollouts: u64 = 0;
    let mut cap_hits: u64 = 0;
    let mut timed_out = false;
    let t0 = std::time::Instant::now();
    while !is_terminal(&s) {
        if decision >= MAX_ACTIONS_PER_FIGHT || steps >= MAX_STEPS_PER_FIGHT {
            timed_out = true;
            break;
        }
        let acts = legal_actions(&s);
        if acts.len() == 1 {
            let a = acts[0].clone();
            let _ = apply(&mut s, a);
            steps += 1;
            continue; // forced moves don't consume search budget
        }
        let cfg = deployed_config(knobs, pair_seed(state_id, replica, decision, salt));
        // BeliefDraw REQUIRES a fresh Mcts per decision (reroot asserts).
        let mut mcts = Mcts::new(s.clone(), cfg);
        let rec = mcts.search();
        let (r, c) = mcts.rollout_stats();
        rollouts += r;
        cap_hits += c;
        let _ = apply(&mut s, rec);
        steps += 1;
        decision += 1;
    }
    let hp = s.player.current_hp.max(0) as f32;
    let potions = s.player.potions.len() as f32;
    let mut won = s.is_over && s.player_won;
    let mut score = if won {
        hp + POTION_HP_EQUIV * potions
    } else {
        LOSS_SCORE
    };
    if timed_out {
        won = false;
        score = LOSS_SCORE;
        eprintln!(
            "[asc-eval] WARNING state_id={state_id} replica={replica} timed out \
             (decision={decision} steps={steps}) — scored as a loss"
        );
    }
    FightRow {
        state_id,
        replica,
        won,
        hp,
        potions: potions as u32,
        score,
        decisions: decision,
        rollouts,
        cap_hits,
        wall_ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
        timed_out,
    }
}

/// Load + realize one corpus entry to the first `PlayerPlayPhase`
/// decision point. `fixture_path` may be absolute (what
/// `corpus_gen --runmanager-toml-export` writes today, since its
/// `out_dir` argument is typically absolute) or relative (resolved
/// against `corpus_dir`, the corpus.json's parent).
fn realize_entry(entry: &CorpusEntry, corpus_dir: &Path) -> Result<CombatState, String> {
    let raw = PathBuf::from(&entry.fixture_path);
    let path = if raw.is_absolute() {
        raw
    } else {
        corpus_dir.join(&raw)
    };
    let scenario = load_scenario_file(&path)
        .map_err(|e| format!("load_scenario_file({}): {e}", path.display()))?;
    // The loader does NOT read [player].ascension itself (Task 2) — the
    // evaluator reads it from the fixture and passes it through
    // explicitly, defaulting to 0 (A0) when the fixture omits it (the
    // TOML writer only emits the field when ascension > 0).
    let ascension = scenario
        .player
        .as_ref()
        .and_then(|p| p.ascension)
        .unwrap_or(0) as u8;
    let mut state = scenario_to_combat_state_with_ascension(&scenario, entry.seed, ascension)
        .map_err(|e| format!("scenario_to_combat_state_with_ascension: {e}"))?;
    // Mirrors combat_position_book::realize / closed_loop_harness::play_one_fight:
    // scenario_to_combat_state* hands back a pre-start_combat (Setup) state.
    let _ = sts2_core::combat_loop::start_combat(&mut state);
    let _ = sts2_core::combat_loop::start_player_turn(&mut state);
    if is_terminal(&state) || legal_actions(&state).is_empty() {
        return Err("degenerate after realize (terminal or no legal actions)".to_string());
    }
    Ok(state)
}

fn main() -> std::io::Result<()> {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();

    let args = parse_args();

    if args.jobs != 0 {
        // Ignore the error: a second ThreadPoolBuilder::build_global()
        // call (e.g. re-running main() in a test harness) fails because
        // the global pool is already installed, which is fine — the
        // existing pool stays in effect.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global();
    }

    eprintln!(
        "[asc-eval] corpus={} replicas={} seed_salt={} jobs={} out={}",
        args.corpus.display(),
        args.replicas,
        args.seed_salt,
        args.jobs,
        args.out.display(),
    );

    let bytes = std::fs::read(&args.corpus)
        .unwrap_or_else(|e| panic!("failed to read corpus {}: {e}", args.corpus.display()));
    let corpus: CorpusFile = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "corpus JSON at {} did not parse: {e}",
            args.corpus.display()
        )
    });
    let corpus_dir = match args.corpus.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };

    std::panic::set_hook(Box::new(|_| {}));
    let mut templates: Vec<(u32, CombatState)> = Vec::with_capacity(corpus.entries.len());
    for entry in &corpus.entries {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            realize_entry(entry, &corpus_dir)
        }));
        match outcome {
            Ok(Ok(state)) => templates.push((entry.state_id, state)),
            Ok(Err(e)) => eprintln!("[asc-eval] skip state_id={} ({e})", entry.state_id),
            Err(_) => eprintln!(
                "[asc-eval] skip state_id={} (panicked realizing fixture — likely a content-port gap)",
                entry.state_id
            ),
        }
    }
    let _ = std::panic::take_hook();
    eprintln!(
        "[asc-eval] {} realized templates (of {} corpus entries)",
        templates.len(),
        corpus.entries.len(),
    );

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    if templates.is_empty() {
        eprintln!(
            "[asc-eval] ERROR: 0 of {} corpus entries realized — no fixtures to \
             evaluate. This usually means fixture_path resolved to nothing (see \
             the corpus_dir-relative-path GOTCHA in README.md \"Step 1\"). Exiting \
             nonzero instead of emitting a SCORE line so callers (e.g. \
             tune_mcts_optuna.py) don't silently score this trial as 0.0.",
            corpus.entries.len(),
        );
        std::process::exit(1);
    }

    // Streamed `.partial` sibling of `--out`: one line per fight,
    // flushed as it completes, so a stuck run is visible mid-flight.
    // Rows land here in COMPLETION order (racy across rayon workers),
    // so this file is progress-only — it is NOT the determinism
    // artifact and is removed on a clean exit. The canonical `--out`
    // is written once at the end, sorted by (state_id, replica), which
    // is what makes two identical invocations diff byte-identical
    // (mod `wall_ms`) without any post-processing at the call site.
    let mut partial_name = args.out.clone().into_os_string();
    partial_name.push(".partial");
    let partial_path = PathBuf::from(partial_name);
    let partial_writer = Mutex::new(std::io::BufWriter::new(std::fs::File::create(
        &partial_path,
    )?));

    let tasks: Vec<(usize, u32)> = (0..templates.len())
        .flat_map(|i| (0..args.replicas).map(move |r| (i, r)))
        .collect();

    let knobs = &args.knobs;
    let salt = args.seed_salt;
    let rows: Vec<FightRow> = tasks
        .par_iter()
        .map(|&(i, replica)| {
            let (state_id, template) = &templates[i];
            let row = play_one(template, knobs, *state_id, replica, salt);
            {
                let mut w = partial_writer
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let _ = serde_json::to_writer(&mut *w, &row);
                let _ = w.write_all(b"\n");
                let _ = w.flush();
            }
            row
        })
        .collect();
    drop(partial_writer);
    let _ = std::fs::remove_file(&partial_path);

    let mut rows = rows;
    rows.sort_by_key(|r| (r.state_id, r.replica));

    let mut out_writer = std::io::BufWriter::new(std::fs::File::create(&args.out)?);
    for row in &rows {
        serde_json::to_writer(&mut out_writer, row)?;
        out_writer.write_all(b"\n")?;
    }
    out_writer.flush()?;

    let n = rows.len();
    let wins = rows.iter().filter(|r| r.won).count();
    let mean_score = rows.iter().map(|r| f64::from(r.score)).sum::<f64>() / n as f64;
    let win_rate = wins as f64 / n as f64;
    let avg_hp = rows.iter().map(|r| f64::from(r.hp)).sum::<f64>() / n as f64;
    let total_rollouts: u64 = rows.iter().map(|r| r.rollouts).sum();
    let total_cap_hits: u64 = rows.iter().map(|r| r.cap_hits).sum();

    println!("SCORE {mean_score:.4} {win_rate:.4} {avg_hp:.2} {n}");
    eprintln!(
        "[asc-eval] cap-hit rate {:.4} ({total_cap_hits} / {total_rollouts} rollouts)",
        total_cap_hits as f64 / total_rollouts.max(1) as f64,
    );

    Ok(())
}
