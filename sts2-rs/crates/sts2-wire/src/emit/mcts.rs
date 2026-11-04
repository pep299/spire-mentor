//! `sts2_rs.mcts_recommend` / `sts2_rs.mcts_search_tree` — the search result
//! dicts, built once here for both the `PyO3` module and the browser
//! workbench (`sts2-workbench::advisors`).
//!
//! The Python hint adapter (`sts2_hint.recommender.RecommendConfig`) and the
//! workbench's `reasoning.py` feed these from a loose `config` dict; the
//! browser feeds the same dict straight from the SPA. [`RecommendParams`]
//! is the one parser (same defaults, unknown keys dropped, the same
//! rejections for a bad `rollout_policy` / `hidden_info`), so a config
//! produces the same [`MctsConfig`] on every host.
//!
//! Everything here is single-process; the `threads > 1` root-parallel path
//! is `sts2_mcts::par_search_with_policy` (rayon) and callers that cannot
//! spawn threads — the wasm build — clamp `threads` to 1 before calling.

use serde_json::{json, Map, Value};
use sts2_core::{apply, legal_actions, Action, CombatState};
use sts2_mcts::{
    dpw_outcome_cap, hint, par_search_with_policy, root_pick_order, EdgeAction, HiddenInfoMode,
    LeafEval, Mcts, MctsConfig, NodeKind, RolloutPolicyKind, RootOutcome, RootPick, SharedPolicy,
};

use super::combat::combat_action_to_value;

/// Q-saturation epsilon: if the spread of root-child mean values is below
/// this, the rollout policy can't distinguish actions and any ranking is
/// essentially noise. Matches the threshold used by the CLI hint formatter
/// (`sts2-cli/src/agent/human.rs`, `q_saturated`).
pub const Q_SATURATION_EPSILON: f32 = 1e-3;

/// Low-confidence visit-fraction threshold: any root child with visits
/// below `LOW_CONFIDENCE_VISIT_FRACTION * max_visits` is flagged. Mirrors
/// `LOW_CONFIDENCE_VISIT_FRACTION` in `sts2-cli/src/agent/human.rs`.
pub const LOW_CONFIDENCE_VISIT_FRACTION: f64 = 0.05;

/// Hard cap on the number of actions reported in the principal variation
/// returned by [`recommend_value`]. Matches the spec ("depth ≤ 5").
pub const PV_MAX_DEPTH: usize = 5;

/// The search's own pick for one result, aligned with the
/// [`RecommendScore`]s it explains (GH #1022): the rule's [`RootPick`], with
/// indices into those scores in their search order, and the
/// [`root_pick_order`] over them.
#[derive(Clone, Debug)]
pub struct EnginePick {
    pub pick: RootPick,
    pub order: Vec<usize>,
    /// The band width the rule ran with ([`MctsConfig::tie_eps`]).
    pub tie_eps: f32,
}

impl EnginePick {
    /// Explain a finished single-tree search over `m.root_outcomes()`.
    fn from_tree<P: sts2_mcts::Policy>(m: &mut Mcts<P>, outs: &[RootOutcome]) -> Option<Self> {
        let pick = m.explain_root_pick().ok()?;
        let order = root_pick_order(&m.tree.root_state, &m.config, outs, pick.idx);
        Some(Self {
            pick,
            order,
            tie_eps: m.config.tie_eps,
        })
    }
}

/// `RecommendConfig` (`sts2_hint/recommender.py`): the knobs of one
/// recommend / search-tree call, with that class's defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecommendParams {
    pub sims: u32,
    pub depth: u32,
    pub c_puct: f32,
    pub seed: u64,
    /// Root-parallel workers (`> 1` needs a thread pool).
    pub threads: u32,
    /// Per-action independent searches instead of one root search.
    pub fair: bool,
    /// Explicit playout policy. `None` keeps what the mode deploys
    /// ([`MctsConfig::apply_mode_dialect`]: the playout net under
    /// BeliefDraw, the heuristic otherwise); a named policy always wins.
    pub rollout_policy: Option<RolloutPolicyKind>,
    pub hidden_info: HiddenInfoMode,
    /// `None` keeps the mode default (`MctsConfig::default().draw_widening_c`).
    pub draw_widening_c: Option<f32>,
    /// Explicit EndTurn DPW coefficient (an ablation knob). `None` keeps what
    /// the mode deploys ([`MctsConfig::apply_mode_dialect`]).
    pub endturn_widening_c: Option<f32>,
    /// Explicit EndTurn DPW exponent; `None` keeps the mode's.
    pub endturn_widening_alpha: Option<f32>,
    pub common_random_numbers: bool,
    /// Playouts per search leaf ([`MctsConfig::leaf_playouts`], GH #830).
    /// `None` keeps the engine default (one).
    pub leaf_playouts: Option<u32>,
    /// [`MctsConfig::net_expand_order`] (GH #830 rung 4, an ablation knob,
    /// default off). When on, [`search_tree_value`] also reports what it
    /// did under `expand_order`.
    pub net_expand_order: bool,
    /// Explicit [`MctsConfig::fpu_reduction`]; `None` keeps the engine's
    /// `+inf` first-play urgency.
    pub fpu_reduction: Option<f32>,
    /// [`MctsConfig::dominance_pruning`]: drop follow-up card plays whose
    /// reverse order the engine proves at least as good.
    pub dominance_pruning: bool,
    /// Boss win value by what the fight carries forward
    /// ([`MctsConfig::win_hp_carry`] / [`MctsConfig::win_potion_carry`],
    /// GH #771) — what the run agent sets from
    /// [`sts2_mcts::boss_win_carry`]. `None` keeps the default value.
    pub win_hp_carry: Option<f32>,
    pub win_potion_carry: Option<f32>,
}

impl Default for RecommendParams {
    fn default() -> Self {
        Self {
            sims: 300,
            depth: 200,
            c_puct: sts2_mcts::belief_draw_c_puct(300),
            seed: 0,
            threads: 1,
            fair: false,
            rollout_policy: None,
            // The DEPLOYED honest search, not the leaky one. These three are
            // one setting: see `resolve_mode_constants`.
            hidden_info: HiddenInfoMode::BeliefDraw,
            draw_widening_c: Some(sts2_mcts::BELIEF_DRAW_DRAW_WIDENING_C),
            endturn_widening_c: None,
            endturn_widening_alpha: None,
            common_random_numbers: false,
            leaf_playouts: None,
            net_expand_order: false,
            fpu_reduction: None,
            dominance_pruning: false,
            win_hp_carry: None,
            win_potion_carry: None,
        }
    }
}

/// `rollout_policy` string → kind; the same spellings `mcts_recommend`
/// accepts, the same rejection for anything else.
pub fn parse_rollout_policy(s: &str, fn_name: &str) -> Result<RolloutPolicyKind, String> {
    match s {
        "heuristic" => Ok(RolloutPolicyKind::Heuristic),
        "uniform" => Ok(RolloutPolicyKind::Uniform),
        "net" => Ok(RolloutPolicyKind::Net),
        other => Err(format!(
            "{fn_name}: rollout_policy must be 'heuristic', 'uniform' or 'net', got {other:?}"
        )),
    }
}

/// `hidden_info` string → mode. Rejects an unrecognized value rather than
/// falling back: a typo in an A/B arm's config must fail loudly, not
/// silently run the wrong search.
pub fn parse_hidden_info(s: &str, fn_name: &str) -> Result<HiddenInfoMode, String> {
    match s {
        "clairvoyant" => Ok(HiddenInfoMode::Clairvoyant),
        "beliefdraw" => Ok(HiddenInfoMode::BeliefDraw),
        "drawchance" => Ok(HiddenInfoMode::DrawChance),
        "rootscramble" => Ok(HiddenInfoMode::RootScramble),
        other => Err(format!(
            "{fn_name}: hidden_info must be one of \
             'clairvoyant'|'beliefdraw'|'drawchance'|'rootscramble', got {other:?}"
        )),
    }
}

fn num_u32(v: &Value) -> Option<u32> {
    v.as_u64().and_then(|n| u32::try_from(n).ok()).or_else(|| {
        v.as_f64()
            .filter(|f| f.is_finite() && *f >= 0.0)
            .map(|f| f as u32)
    })
}

fn num_u64(v: &Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_i64().map(|n| n as u64))
        .or_else(|| v.as_f64().filter(|f| f.is_finite()).map(|f| f as u64))
}

fn num_f32(v: &Value) -> Option<f32> {
    v.as_f64().map(|f| f as f32)
}

impl RecommendParams {
    /// `RecommendConfig.from_dict`: read the known keys of a loose config
    /// dict, keep the defaults for the rest, ignore everything unknown
    /// (older frontends still send `chance_nodes`, `log_path`, …).
    /// `fn_name` labels the error for a bad `rollout_policy` / `hidden_info`.
    pub fn from_config(config: &Value, fn_name: &str) -> Result<Self, String> {
        let mut p = Self::default();
        let Some(obj) = config.as_object() else {
            return Ok(p);
        };
        if let Some(v) = obj.get("sims").and_then(num_u32) {
            p.sims = v;
        }
        if let Some(v) = obj.get("depth").and_then(num_u32) {
            p.depth = v;
        }
        if let Some(v) = obj.get("c_puct").and_then(num_f32) {
            p.c_puct = v;
        }
        if let Some(v) = obj.get("seed").and_then(num_u64) {
            p.seed = v;
        }
        if let Some(v) = obj.get("threads").and_then(num_u32) {
            p.threads = v;
        }
        if let Some(v) = obj.get("fair").and_then(Value::as_bool) {
            p.fair = v;
        }
        if let Some(v) = obj.get("rollout_policy").and_then(Value::as_str) {
            p.rollout_policy = Some(parse_rollout_policy(v, fn_name)?);
        }
        if let Some(v) = obj.get("hidden_info").and_then(Value::as_str) {
            p.hidden_info = parse_hidden_info(v, fn_name)?;
        }
        let explicit_draw_widening = matches!(obj.get("draw_widening_c"), Some(v) if !v.is_null());
        match obj.get("draw_widening_c") {
            Some(Value::Null) | None => {}
            Some(v) => p.draw_widening_c = num_f32(v),
        }
        if let Some(v) = obj.get("endturn_widening_c").and_then(num_f32) {
            p.endturn_widening_c = Some(v);
        }
        if let Some(v) = obj.get("endturn_widening_alpha").and_then(num_f32) {
            p.endturn_widening_alpha = Some(v);
        }
        if let Some(v) = obj.get("common_random_numbers").and_then(Value::as_bool) {
            p.common_random_numbers = v;
        }
        if let Some(v) = obj.get("leaf_playouts").and_then(num_u32) {
            p.leaf_playouts = Some(v);
        }
        if let Some(v) = obj.get("dominance_pruning").and_then(Value::as_bool) {
            p.dominance_pruning = v;
        }
        if let Some(v) = obj.get("win_hp_carry").and_then(num_f32) {
            p.win_hp_carry = Some(v);
        }
        if let Some(v) = obj.get("win_potion_carry").and_then(num_f32) {
            p.win_potion_carry = Some(v);
        }
        p.resolve_mode_constants(
            obj.get("c_puct").and_then(num_f32).is_some(),
            explicit_draw_widening,
        );
        Ok(p)
    }

    /// Re-derive the constants that TRAVEL WITH `hidden_info`, unless the
    /// caller pinned them.
    ///
    /// `c_puct` and `draw_widening_c` are not free knobs: the deployed
    /// BeliefDraw honest search sets them from the budget
    /// ([`MctsConfig::apply_mode_dialect`] — the F2 retune, bead
    /// `sts2-rs-09x5n`). Pairing BeliefDraw with the engine's `SQRT_2` /
    /// no-widening numbers is a MISTUNED search, not a neutral one, and the
    /// constant depends on `sims`, which callers set independently. So a
    /// config that names a mode or a budget without naming the constants gets
    /// the constants that mode deploys at that budget. (The EndTurn pair
    /// needs no resolving here: `None` defers to the dialect in
    /// [`Self::mcts_config`].)
    ///
    /// Every other mode keeps the historical 1.4 / no widening byte-identical.
    fn resolve_mode_constants(&mut self, keep_c_puct: bool, keep_draw_widening: bool) {
        let honest = self.hidden_info == HiddenInfoMode::BeliefDraw;
        let mut dialect = MctsConfig {
            hidden_info: self.hidden_info,
            max_simulations: self.sims,
            ..MctsConfig::default()
        };
        dialect.apply_mode_dialect();
        if !keep_c_puct {
            self.c_puct = if honest { dialect.c_puct } else { 1.4 };
        }
        if !keep_draw_widening {
            self.draw_widening_c = honest.then_some(dialect.draw_widening_c);
        }
    }

    /// The search config `mcts_recommend` / `mcts_search_tree` build: the
    /// mode's deployed dialect ([`MctsConfig::apply_mode_dialect`], the same
    /// call the run agent's `sts2-runai::policy::mcts_config_for` makes), the
    /// deployed-search knobs pinned (no transposition table, regularized π
    /// temperature 0.1, uniform PUCT prior), then this call's parameters —
    /// an explicit `c_puct` / widening value / rollout policy wins over the
    /// dialect.
    #[must_use]
    pub fn mcts_config(&self) -> MctsConfig {
        let mut cfg = MctsConfig {
            max_simulations: self.sims,
            max_rollout_depth: self.depth,
            seed: self.seed,
            use_transposition_table: false,
            regularized_pi_temperature: 0.1,
            use_puct_prior: false,
            hidden_info: self.hidden_info,
            common_random_numbers: self.common_random_numbers,
            dominance_pruning: self.dominance_pruning,
            ..MctsConfig::default()
        };
        cfg.apply_mode_dialect();
        if let Some(p) = self.rollout_policy {
            cfg.rollout_policy = p;
        }
        cfg.c_puct = self.c_puct;
        cfg.draw_widening_c = self
            .draw_widening_c
            .unwrap_or(MctsConfig::default().draw_widening_c);
        if let Some(c) = self.endturn_widening_c {
            cfg.endturn_widening_c = c;
        }
        if let Some(alpha) = self.endturn_widening_alpha {
            cfg.endturn_widening_alpha = alpha;
        }
        if let Some(k) = self.leaf_playouts {
            cfg.leaf_playouts = k;
        }
        cfg.net_expand_order = self.net_expand_order;
        if self.fpu_reduction.is_some() {
            cfg.fpu_reduction = self.fpu_reduction;
        }
        cfg.win_hp_carry = self.win_hp_carry;
        cfg.win_potion_carry = self.win_potion_carry;
        cfg
    }
}

/// Snapshot of one root child outcome used by both the `fair=False`
/// (single search) and `fair=True` (per-action independent search) paths
/// of [`recommend_value`]. Decoupled from [`RootOutcome`] so the fair-eval
/// branch can synthesize terminal outcomes (e.g. a lethal playable already
/// kills the only enemy) as if they had been rolled out once.
#[derive(Clone, Debug)]
pub struct RecommendScore {
    pub action: Action,
    pub visits: u32,
    pub wins: u32,
    pub losses: u32,
    pub ties: u32,
    /// Sum of continuous leaf values; `value_sum / visits` is the Q used
    /// for action ranking.
    pub value_sum: f32,
    /// Sum of the player's final HP over the resolved playouts through this
    /// action (a death reads 0), and how many playouts that is
    /// (`hp_n <= visits`). `hp_sum / hp_n` is the mean final HP.
    pub hp_sum: f32,
    /// `Σ hp²` over the same playouts (a standard error on the mean, GH #888).
    pub hp_sq_sum: f32,
    pub hp_n: u32,
    /// This action ends the combat in a WIN immediately (fair-eval only).
    /// Ranked above a merely equal-valued continuation (GH#296).
    pub terminal_win: bool,
}

impl From<RootOutcome> for RecommendScore {
    fn from(o: RootOutcome) -> Self {
        Self {
            action: o.action,
            visits: o.visits,
            wins: o.wins,
            losses: o.losses,
            ties: o.ties,
            value_sum: o.value_sum,
            hp_sum: o.hp_sum,
            hp_sq_sum: o.hp_sq_sum,
            hp_n: o.hp_n,
            terminal_win: false,
        }
    }
}

/// Q = `value_sum / visits` from the player's perspective (0 for an
/// unvisited action).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn q_of(s: &RecommendScore) -> f32 {
    if s.visits == 0 {
        0.0
    } else {
        s.value_sum / s.visits as f32
    }
}

/// Counters summed across every root child of a post-action search.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RootAggregate {
    pub visits: u32,
    pub wins: u32,
    pub losses: u32,
    pub ties: u32,
    pub value_sum: f32,
    pub hp_sum: f32,
    pub hp_sq_sum: f32,
    pub hp_n: u32,
}

/// Sum the counters across every root child of a post-action search. The
/// per-child counters sum to the root node's own counters because every
/// simulation backpropagates through the root.
#[must_use]
pub fn aggregate_root(outs: &[RootOutcome]) -> RootAggregate {
    let mut a = RootAggregate::default();
    for o in outs {
        a.visits += o.visits;
        a.wins += o.wins;
        a.losses += o.losses;
        a.ties += o.ties;
        a.value_sum += o.value_sum;
        a.hp_sum += o.hp_sum;
        a.hp_sq_sum += o.hp_sq_sum;
        a.hp_n += o.hp_n;
    }
    a
}

/// One per-action score dict (`mcts_recommend()["actions"][i]`).
#[must_use]
#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
pub fn action_score_value(
    action: &Action,
    visits: u32,
    wins: u32,
    losses: u32,
    ties: u32,
    value_sum: f32,
    hp_sum: f32,
    hp_sq_sum: f32,
    hp_n: u32,
    low_confidence: bool,
    q_saturated: bool,
    is_best: bool,
) -> Value {
    let vdenom = f64::from(visits.max(1));
    let p_win = f64::from(wins) / vdenom;
    let p_tie = f64::from(ties) / vdenom;
    let p_loss = f64::from(losses) / vdenom;
    // Legacy `value` field — bucket-derived mean, kept for the CLI's
    // `LIKELY-WIN n%` display + parity with pre-v2oh consumers.
    let q_bucket = if visits == 0 {
        0.0_f32
    } else {
        (wins as f32 - losses as f32) / visits as f32
    };
    // Continuous Q from the lightspeed-style reward: what action ranking
    // actually consumes.
    let q_cont = if visits == 0 {
        0.0_f32
    } else {
        value_sum / visits as f32
    };
    #[allow(clippy::cast_possible_truncation)]
    let bucket = hint::win_prob_bucket(p_win as f32);
    // Mean player HP at the end of the resolved playouts; null when none
    // resolved (never 0 — a death is a real 0, no sample is not).
    let mean_final_hp = (hp_n > 0).then(|| f64::from(hp_sum) / f64::from(hp_n));
    // Population SD of the same samples (GH #888): with `hp_n` it puts a
    // standard error on `mean_final_hp`, which is what lets the coach tell a
    // real HP gap between two moves from rollout noise. Null exactly when
    // `mean_final_hp` is.
    let hp_sd = mean_final_hp.map(|m| {
        (f64::from(hp_sq_sum) / f64::from(hp_n) - m * m)
            .max(0.0)
            .sqrt()
    });
    json!({
        "action": combat_action_to_value(action),
        "visits": visits,
        "value": q_bucket,
        "value_sum": value_sum,
        "q": q_cont,
        "mean_final_hp": mean_final_hp,
        "hp_sd": hp_sd,
        "hp_n": hp_n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_win": p_win,
        "p_tie": p_tie,
        "p_loss": p_loss,
        "bucket": bucket,
        "low_confidence": low_confidence,
        "q_saturated": q_saturated,
        "is_best": is_best,
    })
}

/// Rank `scores` and build the `mcts_recommend` result dict:
/// `{actions, pv, pick, q_saturated, fair, sims, threads,
/// intent_damage_known}`.
///
/// Sort: Q desc, then a decisive WIN, then visits desc (matches
/// `human.rs` plus the GH#296 terminal-win tie-break). `engine` — the
/// search's own pick ([`sts2_mcts::Mcts::explain_root_pick`], the rule the
/// run agent plays by) — is then moved to the top, so the `is_best` row is
/// the move the Rust AI would make, not a second rule that disagrees with
/// it on ties and guards. `None` (the fair mode, which has no single tree)
/// keeps the plain Q order. The top row is `is_best`; rows below 5% of the
/// max visits are `low_confidence`; `q_saturated` when the Q spread across
/// >1 rows is under [`Q_SATURATION_EPSILON`].
///
/// The pick is explained so no consumer has to re-derive it (GH #1022):
/// each row carries `in_band` (the engine cannot tell it from its pick) and
/// `pick_rank` (the engine's preference order, 0 = the pick; a consumer
/// that must drop rows plays the lowest surviving rank), and `pick` is
/// `{rule, reason, tie_eps, gap_q, gap_z, gap_hp}` — the deciding
/// [`sts2_mcts::PickRule`] and its words, and the pick's lead over the
/// runner-up (the best-scored other row) in root Q, in its standard error
/// and in mean final HP. `pick` is null in the fair mode, whose rows rank by
/// row order and are never in a band.
#[must_use]
pub fn recommend_output_value(
    scores: Vec<RecommendScore>,
    pv: &[Action],
    engine: Option<&EnginePick>,
    fair: bool,
    sims: u32,
    threads: u32,
    intent_damage_known: bool,
) -> Value {
    let n = scores.len();
    // Row order as indices into `scores`, which keep the search's order so
    // the engine's indices stay valid.
    let mut rows: Vec<usize> = (0..n).collect();
    rows.sort_by(|&a, &b| {
        let (a, b) = (&scores[a], &scores[b]);
        q_of(b)
            .partial_cmp(&q_of(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.terminal_win.cmp(&a.terminal_win))
            .then(b.visits.cmp(&a.visits))
    });
    let engine = engine.filter(|e| e.pick.idx.is_some_and(|i| i < n) && e.order.len() == n);
    if let Some(pos) = engine.and_then(|e| rows.iter().position(|&r| Some(r) == e.pick.idx)) {
        rows[..=pos].rotate_right(1);
    }
    let mut rank = vec![0_usize; n];
    for (k, &i) in engine.map_or(&rows, |e| &e.order).iter().enumerate() {
        rank[i] = k;
    }
    let q_saturated = if n > 1 {
        let q_min = scores.iter().map(q_of).fold(f32::INFINITY, f32::min);
        let q_max = scores.iter().map(q_of).fold(f32::NEG_INFINITY, f32::max);
        (q_max - q_min).abs() < Q_SATURATION_EPSILON
    } else {
        false
    };
    let max_visits = scores.iter().map(|s| s.visits).max().unwrap_or(0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let low_visit_threshold =
        ((f64::from(max_visits) * LOW_CONFIDENCE_VISIT_FRACTION).ceil() as u32).max(1);
    let actions: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(pos, &i)| {
            let s = &scores[i];
            let mut row = action_score_value(
                &s.action,
                s.visits,
                s.wins,
                s.losses,
                s.ties,
                s.value_sum,
                s.hp_sum,
                s.hp_sq_sum,
                s.hp_n,
                s.visits < low_visit_threshold,
                q_saturated,
                pos == 0,
            );
            row["pick_rank"] = json!(rank[i]);
            row["in_band"] = json!(engine.is_some_and(|e| e.pick.band.contains(&i)));
            row
        })
        .collect();
    let mean_hp = |i: usize| {
        let s = &scores[i];
        (s.hp_n > 0).then(|| f64::from(s.hp_sum) / f64::from(s.hp_n))
    };
    let pick = engine.map(|e| {
        let p = &e.pick;
        let gap_hp = p
            .idx
            .zip(p.runner_up)
            .and_then(|(a, b)| Some(mean_hp(a)? - mean_hp(b)?));
        json!({
            "rule": p.rule.as_str(),
            "reason": p.rule.reason(),
            "tie_eps": e.tie_eps,
            "gap_q": p.gap_q,
            "gap_z": p.gap_z,
            "gap_hp": gap_hp,
        })
    });
    let pv: Vec<Value> = pv.iter().map(combat_action_to_value).collect();
    json!({
        "actions": actions,
        "pv": pv,
        "pick": pick,
        "q_saturated": q_saturated,
        "fair": fair,
        "sims": sims,
        "threads": threads,
        "intent_damage_known": intent_damage_known,
    })
}

/// Score ONE action the way one iteration of the fair branch of
/// [`recommend_value`] does.
///
/// Under an honest hidden-information mode (anything but `Clairvoyant`) this
/// is a forced-root search ([`Mcts::search_forced`]) rooted at `state`, the
/// position the action is chosen FROM: the search's own root conversion
/// (belief canonicalization / blindfold, seeded from the search seed alone)
/// runs BEFORE the action, and a drawing or rolling action passes a chance
/// node that re-deals per simulation. Applying the action to the raw `state`
/// and searching only what follows — what this did before, and still does
/// under `Clairvoyant` — prices the realized outcome: a root draw took the
/// true next card and a root roll the state's real RNG cursors, so a fair row
/// moved with information the player cannot have (`tests/fair_root.rs`;
/// reported against the manoj fork, `257fc6817` / `1edb23cf0`). It also
/// gives an action that ends the combat its whole budget instead of one
/// synthesized visit.
fn score_action_raw(
    state: &CombatState,
    params: &RecommendParams,
    action: &Action,
) -> RecommendScore {
    if params.hidden_info != HiddenInfoMode::Clairvoyant {
        return score_action_forced_raw(state, params, action);
    }
    let cfg = params.mcts_config();
    let policy = SharedPolicy::uniform();
    let mut post = state.clone();
    let _ = apply(&mut post, action.clone());
    if legal_actions(&post).is_empty() {
        // Already terminal — encode as a single-sample outcome bucketed by
        // terminal value, on the SAME scale as the sibling searches
        // (`cfg.terminal_eval_mode`, GH#296).
        let v = sts2_mcts::terminal_state_value(
            &post,
            cfg.terminal_eval_mode,
            &LeafEval::default(),
            cfg.loss_gradient,
        );
        let (w, l, t) = if v > 0.5 {
            (1u32, 0u32, 0u32)
        } else if v < -0.5 {
            (0, 1, 0)
        } else {
            (0, 0, 1)
        };
        #[allow(clippy::cast_precision_loss)]
        let hp = post.player.current_hp.max(0) as f32;
        return RecommendScore {
            action: action.clone(),
            visits: 1,
            wins: w,
            losses: l,
            ties: t,
            value_sum: v,
            hp_sum: hp,
            hp_sq_sum: hp * hp,
            hp_n: 1,
            terminal_win: post.player_won,
        };
    }
    let agg = if params.threads <= 1 {
        let mut m = Mcts::with_policy(post, cfg, policy);
        let _ = m.search();
        aggregate_root(&m.root_outcomes())
    } else {
        let r = par_search_with_policy(post, cfg, params.threads, policy, LeafEval::default());
        aggregate_root(&r.outcomes)
    };
    RecommendScore {
        action: action.clone(),
        visits: agg.visits,
        wins: agg.wins,
        losses: agg.losses,
        ties: agg.ties,
        value_sum: agg.value_sum,
        hp_sum: agg.hp_sum,
        hp_sq_sum: agg.hp_sq_sum,
        hp_n: agg.hp_n,
        terminal_win: false,
    }
}

/// The honest fair row for `action`: a single-threaded forced-root search
/// from `state` (see [`score_action_raw`]). `terminal_win` (GH#296: an
/// immediate win ranks above an equal-valued continuation) is read from the
/// search's own view of the root, never from the raw state: every forced
/// simulation was a win and the action ends the combat on the
/// seed-determined belief root.
fn score_action_forced_raw(
    state: &CombatState,
    params: &RecommendParams,
    action: &Action,
) -> RecommendScore {
    let mut m = Mcts::with_policy(state.clone(), params.mcts_config(), SharedPolicy::uniform());
    let Some(o) = m.search_forced(action, params.sims.max(1)) else {
        return RecommendScore {
            action: action.clone(),
            visits: 0,
            wins: 0,
            losses: 0,
            ties: 0,
            value_sum: 0.0,
            hp_sum: 0.0,
            hp_sq_sum: 0.0,
            hp_n: 0,
            terminal_win: false,
        };
    };
    let terminal_win = o.visits > 0 && o.wins == o.visits && {
        let mut post = m.tree.root_state.clone();
        let _ = apply(&mut post, action.clone());
        legal_actions(&post).is_empty() && post.player_won
    };
    RecommendScore {
        terminal_win,
        ..RecommendScore::from(o)
    }
}

/// One action dict, same shape as a `recommend_value` `actions[i]` row
/// (`is_best`, `low_confidence`, `q_saturated` all false), computed by the
/// exact code the fair branch runs per action.
#[must_use]
pub fn score_action(state: &CombatState, params: &RecommendParams, action: &Action) -> Value {
    let s = score_action_raw(state, params, action);
    action_score_value(
        &s.action,
        s.visits,
        s.wins,
        s.losses,
        s.ties,
        s.value_sum,
        s.hp_sum,
        s.hp_sq_sum,
        s.hp_n,
        false,
        false,
        false,
    )
}

/// Price ONE move from the position it is chosen FROM: a single-threaded
/// search rooted at `state` under `params` (same config and hidden-info mode
/// as [`recommend_value`]) with every simulation pinned to `action`
/// ([`Mcts::search_forced`]). Unlike [`score_action`], which searches the
/// post-state reached with `state`'s real RNG, a drawing / turn-ending move
/// passes a chance node that re-deals per simulation, so the row never
/// prices the realized draw. Same row shape as [`score_action`]; `None` when
/// `action` is not a root action of `state` (or the root is terminal).
#[must_use]
pub fn score_action_forced(
    state: &CombatState,
    params: &RecommendParams,
    action: &Action,
) -> Option<Value> {
    let mut m = Mcts::with_policy(state.clone(), params.mcts_config(), SharedPolicy::uniform());
    let o = m.search_forced(action, params.sims)?;
    Some(action_score_value(
        action,
        o.visits,
        o.wins,
        o.losses,
        o.ties,
        o.value_sum,
        o.hp_sum,
        o.hp_sq_sum,
        o.hp_n,
        false,
        false,
        false,
    ))
}

/// The fair branch of [`recommend_value`]: one [`score_action_raw`] row per
/// legal action, in `legal` order.
///
/// An honest mode's forced-root search is single-threaded, so `threads > 1`
/// spreads the ACTIONS over that many scoped threads instead of splitting
/// each search root-parallel; every row is computed exactly as with one
/// thread, so the result does not depend on `threads`. `Clairvoyant` keeps
/// its root-parallel post-state searches. Callers that cannot spawn threads
/// (the wasm build) clamp `threads` to 1 first, as for every other path.
fn fair_scores(
    state: &CombatState,
    params: &RecommendParams,
    legal: &[Action],
) -> Vec<RecommendScore> {
    let workers = usize::try_from(params.threads)
        .unwrap_or(usize::MAX)
        .min(legal.len());
    if workers <= 1 || params.hidden_info == HiddenInfoMode::Clairvoyant {
        return legal
            .iter()
            .map(|action| score_action_raw(state, params, action))
            .collect();
    }
    let mut slots: Vec<Option<RecommendScore>> = (0..legal.len()).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                scope.spawn(move || {
                    legal
                        .iter()
                        .enumerate()
                        .skip(w)
                        .step_by(workers)
                        .map(|(i, action)| (i, score_action_raw(state, params, action)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            for (i, row) in h.join().expect("fair-eval worker panicked") {
                slots[i] = Some(row);
            }
        }
    });
    slots
        .into_iter()
        .map(|r| r.expect("every action scored"))
        .collect()
}

/// Run the search `mcts_recommend` runs and return its result dict.
///
/// `fair=false` (default): ONE search from `state`, root child counters per
/// action, PV from the live tree (single-threaded) or just the best action
/// (root-parallel). `fair=true`: per legal action, apply it and run an
/// INDEPENDENT search from the post-action state; an already-terminal
/// post-action state is scored as one synthesized outcome. `threads > 1`
/// uses the rayon root-parallel search — never on wasm.
///
/// `intent_damage_known` is the wire-side trust flag the `PyO3` state
/// carries (whether the enemy intents' damage was known when the state was
/// built); a native engine state passes `true`.
#[must_use]
pub fn recommend_value(
    state: &CombatState,
    params: &RecommendParams,
    intent_damage_known: bool,
) -> Value {
    let cfg = params.mcts_config();
    let policy = SharedPolicy::uniform();
    let legal = legal_actions(state);
    let (scores, pv, engine): (Vec<RecommendScore>, Vec<Action>, Option<EnginePick>) =
        if legal.is_empty() {
            (Vec::new(), Vec::new(), None)
        } else if params.fair {
            // No PV for the fair case — each action has its own tree.
            (fair_scores(state, params, &legal), Vec::new(), None)
        } else if params.threads <= 1 {
            let mut m = Mcts::with_policy(state.clone(), cfg, policy);
            let played = m.search();
            let outs = m.root_outcomes();
            let engine = EnginePick::from_tree(&mut m, &outs);
            debug_assert!(engine.as_ref().is_none_or(|e| e.pick.action == played));
            let pv: Vec<Action> = m.get_pv().into_iter().take(PV_MAX_DEPTH).collect();
            (
                outs.into_iter().map(RecommendScore::from).collect(),
                pv,
                engine,
            )
        } else {
            let r = par_search_with_policy(
                state.clone(),
                cfg,
                params.threads,
                policy,
                LeafEval::default(),
            );
            // The merge picked over the raw root (`par_search_with_policy`),
            // so the fallback order does too.
            let engine = EnginePick {
                order: root_pick_order(state, &cfg, &r.outcomes, r.pick.idx),
                pick: r.pick,
                tie_eps: cfg.tie_eps,
            };
            let scores = r.outcomes.into_iter().map(RecommendScore::from).collect();
            // Merged counters but no shared tree — the PV is just the best action.
            (scores, vec![r.best_action], Some(engine))
        };
    recommend_output_value(
        scores,
        &pv,
        engine.as_ref(),
        params.fair,
        params.sims,
        params.threads,
        intent_damage_known,
    )
}

/// Terse description of the action on a tree edge, derived purely from the
/// stable [`EdgeAction`] (no per-node state needed). Returns
/// `(kind, label, target_index)`. `PlayCard` labels by card identity (the
/// `CardId` debug name + `+` when upgraded); `Other` variants map to the same
/// `kind` strings [`combat_action_to_value`] uses, with a compact human label.
#[must_use]
pub fn edge_descr(edge: &EdgeAction) -> (String, String, Option<u8>) {
    match edge {
        EdgeAction::PlayCard {
            card_id,
            upgraded,
            target,
            ..
        } => {
            let mark = if *upgraded { "+" } else { "" };
            ("play".to_string(), format!("{card_id:?}{mark}"), *target)
        }
        // A teammate is no enemy slot: name it in the label, not in
        // `target_index`.
        EdgeAction::PlayCardAtAlly {
            card_id,
            upgraded,
            ally,
            ..
        } => {
            let mark = if *upgraded { "+" } else { "" };
            (
                "play".to_string(),
                format!("{card_id:?}{mark} → ally {ally}"),
                None,
            )
        }
        EdgeAction::Other(a) => match a {
            Action::EndTurn => ("end_turn".to_string(), "End turn".to_string(), None),
            Action::PlayCard { hand_index, target } => {
                ("play".to_string(), format!("card[{hand_index}]"), *target)
            }
            Action::PlayCardAtAlly { hand_index, ally } => (
                "play".to_string(),
                format!("card[{hand_index}] → ally {ally}"),
                None,
            ),
            Action::UsePotion { slot, target } => {
                ("potion".to_string(), format!("Potion s{slot}"), *target)
            }
            Action::DrinkPotion {
                potion_slot,
                target,
            } => (
                "drink_potion".to_string(),
                format!("Potion s{potion_slot}"),
                *target,
            ),
            Action::DrinkPotionAtAlly { potion_slot, ally } => (
                "drink_potion".to_string(),
                format!("Potion s{potion_slot} → ally {ally}"),
                None,
            ),
            Action::Channel { orb, target } => {
                ("channel".to_string(), format!("Channel {orb:?}"), *target)
            }
            Action::Evoke { times } => ("evoke".to_string(), format!("Evoke x{times}"), None),
            Action::ResolveChoice { selection } => (
                "choose".to_string(),
                format!("Choose {}", selection.first().copied().unwrap_or(0)),
                None,
            ),
        },
    }
}

/// Run a single (non-fair, single-tree) search and serialize the **entire
/// search tree** for the inspector: `{nodes, root: 0, pv, sims, c_puct}`.
///
/// Each node carries `id, parent, visits, q (= value_sum/visits),
/// value_sum, wins/losses/ties, is_terminal, proven_win, proven_loss,
/// children[], edges[]` (one `{child, kind, label, target_index,
/// outcome_ordinal, visits}` per child, in `children` order — the per-edge
/// action and traffic; the graph is a DAG, so these are NOT properties of the
/// child), `the incoming edge from parent (kind/label/target_index)`, the chance-node
/// metadata (`node_kind ∈ {"decision","chance"}`, `chance_action`,
/// `outcome_cap`, `chance_draws`, `chance_closed`, `outcome_ordinal`) and a
/// `summary` of the state after the node's action. For a chance node
/// `children` are its sampled outcomes (probability = `child.visits /
/// node.visits`) and `q` is the expectimax value. `threads` / `fair` in
/// `params` are ignored: the tree is single by construction.
#[must_use]
pub fn search_tree_value(state: &CombatState, params: &RecommendParams) -> Value {
    let cfg = params.mcts_config();
    let mut nodes: Vec<Value> = Vec::new();
    let mut pv: Vec<Action> = Vec::new();
    let mut expand_order = sts2_mcts::ExpandOrderStats::default();
    if !legal_actions(state).is_empty() {
        let mut m = Mcts::with_policy(state.clone(), cfg, SharedPolicy::uniform());
        let _ = m.search();
        expand_order = m.expand_order_stats();
        pv = m.get_pv().into_iter().take(PV_MAX_DEPTH).collect();
        let arena = &m.tree.arena;
        // The transposition table is always on, so the search graph is a DAG:
        // one node can hang under several parents, reached by a different
        // action from each (Zap→BeamCell and BeamCell→Zap merge). An edge's
        // label and visit count therefore belong to the (parent, child) pair,
        // not to the child — serialize them per edge, parallel to `children`.
        // Decision edges carry `child_visits[i]`, the parent's own traffic
        // through that edge (the child's `visits` sums every parent's);
        // outcome edges carry the outcome child's `visits`, the statistic the
        // engine itself samples outcomes by.
        let edge_lists: Vec<Vec<Value>> = arena
            .iter()
            .map(|n| {
                if matches!(n.kind, NodeKind::Chance) {
                    n.chance_outcomes
                        .iter()
                        .map(|o| {
                            json!({
                                "child": o.child.0,
                                "kind": "outcome",
                                "label": format!("outcome #{}", o.n),
                                "target_index": Value::Null,
                                "outcome_ordinal": o.n,
                                "visits": arena[o.child.0 as usize].visits,
                            })
                        })
                        .collect()
                } else {
                    n.children
                        .iter()
                        .enumerate()
                        .map(|(k, (edge, cid))| {
                            let (kind, label, tgt) = edge_descr(edge);
                            json!({
                                "child": cid.0,
                                "kind": kind,
                                "label": label,
                                "target_index": tgt,
                                "outcome_ordinal": Value::Null,
                                "visits": n.child_visits.get(k).copied().unwrap_or(0),
                            })
                        })
                        .collect()
                }
            })
            .collect();
        // The node-level incoming edge (kind/label/target_index/
        // outcome_ordinal) is the one from the node's allocating `parent`,
        // so it agrees with the `parent` field. Under a shared node it is
        // only ONE of its incoming edges; readers that walk from the root
        // should use the parent's `edges` instead.
        let mut incoming: Vec<Option<&Value>> = vec![None; arena.len()];
        for (pid, list) in edge_lists.iter().enumerate() {
            for e in list {
                let cid = e["child"].as_u64().unwrap_or(0) as usize;
                if arena[cid].parent.map(|p| p.0 as usize) == Some(pid) {
                    incoming[cid] = Some(e);
                }
            }
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        for (i, n) in arena.iter().enumerate() {
            let is_chance = matches!(n.kind, NodeKind::Chance);
            // A chance node's subtree lives in `chance_outcomes`, NOT
            // `children` (empty for chance nodes); flatten both so the
            // serialized tree stays connected.
            let children: Vec<u32> = edge_lists[i]
                .iter()
                .map(|e| e["child"].as_u64().unwrap_or(0) as u32)
                .collect();
            // DPW outcome cap at the node's current visits — the engine's
            // own class selection (EndTurn > Draw > Default) and formula.
            let outcome_cap = if is_chance {
                dpw_outcome_cap(&cfg, n)
            } else {
                0
            };
            let q = if n.visits > 0 {
                n.value_sum / n.visits as f32
            } else {
                0.0
            };
            let field = |k: &str| incoming[i].map_or(Value::Null, |e| e[k].clone());
            let (kind, label, target_index, outcome_ordinal) = (
                field("kind"),
                field("label"),
                field("target_index"),
                field("outcome_ordinal"),
            );
            let enemies: Vec<Value> = n
                .snapshot
                .enemies
                .iter()
                .map(|e| json!({"uid": e.uid, "hp": e.current_hp, "max_hp": e.max_hp}))
                .collect();
            let mut d = Map::new();
            d.insert("id".into(), json!(i as u32));
            d.insert("parent".into(), json!(n.parent.map(|p| p.0)));
            d.insert("visits".into(), json!(n.visits));
            d.insert("q".into(), json!(q));
            d.insert("value_sum".into(), json!(n.value_sum));
            d.insert("wins".into(), json!(n.wins));
            d.insert("losses".into(), json!(n.losses));
            d.insert("ties".into(), json!(n.ties));
            d.insert("is_terminal".into(), json!(n.is_terminal));
            d.insert("proven_win".into(), json!(n.proven_win));
            d.insert("proven_loss".into(), json!(n.proven_loss));
            d.insert("children".into(), json!(children));
            d.insert("edges".into(), json!(edge_lists[i]));
            d.insert("kind".into(), json!(kind));
            d.insert("label".into(), json!(label));
            d.insert("target_index".into(), json!(target_index));
            d.insert(
                "node_kind".into(),
                json!(if is_chance { "chance" } else { "decision" }),
            );
            d.insert(
                "chance_action".into(),
                json!(n.chance_action.as_ref().map(|a| edge_descr(a).1)),
            );
            d.insert("outcome_cap".into(), json!(outcome_cap));
            d.insert("chance_draws".into(), json!(n.chance_draws));
            d.insert("chance_closed".into(), json!(n.chance_closed));
            d.insert("outcome_ordinal".into(), json!(outcome_ordinal));
            d.insert(
                "summary".into(),
                json!({
                    "player_hp": n.snapshot.player_hp,
                    "player_max_hp": n.snapshot.player_max_hp,
                    "player_block": n.snapshot.player_block,
                    "energy": n.snapshot.energy,
                    "round": n.snapshot.round,
                    "enemies": enemies,
                }),
            );
            nodes.push(Value::Object(d));
        }
    }
    let pv: Vec<Value> = pv.iter().map(combat_action_to_value).collect();
    let mut out = json!({
        "nodes": nodes,
        "root": 0u32,
        "pv": pv,
        "sims": params.sims,
        "c_puct": params.c_puct,
    });
    // Only with the knob on, so the default body (the workbench inspector's,
    // hashed by its golden replays) is unchanged.
    if params.net_expand_order {
        out["expand_order"] = json!({
            "considered": expand_order.considered,
            "scored": expand_order.scored,
            "reordered": expand_order.reordered,
            "first_changed": expand_order.first_changed,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_config_keeps_defaults_and_drops_unknown_keys() {
        let p = RecommendParams::from_config(
            &json!({"chance_nodes": true, "log_path": "/x", "sims": 64}),
            "t",
        )
        .unwrap();
        assert_eq!(p.sims, 64);
        assert_eq!(p.depth, 200);
        assert_eq!(p.threads, 1);
        assert!(!p.fair);
        // The DEPLOYED honest search is the default, and its two paired
        // constants follow the mode and the budget rather than being pinned:
        // 64 sims sits below the 500 step, so `c_puct` is 0.8, not 2.5.
        assert_eq!(p.hidden_info, HiddenInfoMode::BeliefDraw);
        assert!((p.c_puct - sts2_mcts::belief_draw_c_puct(64)).abs() < 1e-6);
        assert_eq!(p.draw_widening_c, Some(2.0));
        // A non-object config is the defaults.
        assert_eq!(
            RecommendParams::from_config(&Value::Null, "t").unwrap(),
            RecommendParams::default()
        );
    }

    #[test]
    fn from_config_reads_the_deployed_honest_triple() {
        let p = RecommendParams::from_config(
            &json!({"sims": 2000, "c_puct": 2.5, "hidden_info": "beliefdraw",
                    "draw_widening_c": 2.0, "threads": 1, "seed": 7}),
            "t",
        )
        .unwrap();
        assert_eq!(p.hidden_info, HiddenInfoMode::BeliefDraw);
        assert_eq!(p.draw_widening_c, Some(2.0));
        assert_eq!(p.seed, 7);
        let cfg = p.mcts_config();
        assert_eq!(cfg.max_simulations, 2000);
        assert!((cfg.draw_widening_c - 2.0).abs() < 1e-6);
        assert!(!cfg.use_puct_prior);
        assert!(!cfg.use_transposition_table);
        // `null` keeps the MODE default, which is now 2.0 because the default
        // mode is BeliefDraw and the widening constant travels with it.
        let p = RecommendParams::from_config(&json!({"draw_widening_c": null}), "t").unwrap();
        assert_eq!(p.draw_widening_c, Some(2.0));
        // Naming a non-honest mode without the constants restores the engine
        // defaults byte-identically — the F2 retune is BeliefDraw-only.
        let p = RecommendParams::from_config(&json!({"hidden_info": "clairvoyant"}), "t").unwrap();
        assert_eq!(p.draw_widening_c, None);
        assert!((p.c_puct - 1.4).abs() < 1e-6);
        assert!(
            (p.mcts_config().draw_widening_c - MctsConfig::default().draw_widening_c).abs() < 1e-6
        );
    }

    /// GH #758: the EndTurn pair follows the mode's deployed dialect (the run
    /// agent's PR #467 retune under BeliefDraw), an explicit value wins, and
    /// every other mode keeps the engine's 0.5 / 0.4.
    #[test]
    fn endturn_widening_follows_the_mode_unless_pinned() {
        let (dc, dalpha) = sts2_mcts::BELIEF_DRAW_ENDTURN_WIDENING;
        let cfg = RecommendParams::default().mcts_config();
        assert_eq!(cfg.endturn_widening_c.to_bits(), dc.to_bits());
        assert_eq!(cfg.endturn_widening_alpha.to_bits(), dalpha.to_bits());

        let p = RecommendParams::from_config(
            &json!({"endturn_widening_c": 0.5, "endturn_widening_alpha": 0.4}),
            "t",
        )
        .unwrap();
        assert_eq!(p.endturn_widening_c, Some(0.5));
        let cfg = p.mcts_config();
        assert!((cfg.endturn_widening_c - 0.5).abs() < 1e-6);
        assert!((cfg.endturn_widening_alpha - 0.4).abs() < 1e-6);
        // Pinning the EndTurn pair leaves the rest of the dialect alone.
        assert!((cfg.draw_widening_c - 2.0).abs() < 1e-6);

        let cfg = RecommendParams::from_config(&json!({"hidden_info": "clairvoyant"}), "t")
            .unwrap()
            .mcts_config();
        let engine = MctsConfig::default();
        assert_eq!(
            cfg.endturn_widening_c.to_bits(),
            engine.endturn_widening_c.to_bits()
        );
        assert_eq!(
            cfg.endturn_widening_alpha.to_bits(),
            engine.endturn_widening_alpha.to_bits()
        );
    }

    /// A small real combat: Ironclad starter-ish deck vs a weak encounter.
    fn fixture() -> CombatState {
        let deck: Vec<String> = [
            "STRIKE_IRONCLAD",
            "STRIKE_IRONCLAD",
            "DEFEND_IRONCLAD",
            "BASH",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        crate::scenario::build_scenario_state(
            7,
            "Ironclad",
            &deck,
            60,
            80,
            &[],
            &[],
            "NIBBITS_WEAK",
            0,
            None,
            None,
        )
        .expect("scenario builds")
    }

    fn small_params(fair: bool) -> RecommendParams {
        RecommendParams {
            sims: 24,
            threads: 1,
            fair,
            seed: 3,
            ..RecommendParams::default()
        }
    }

    /// The default tree body has no `expand_order` key (the workbench
    /// inspector's body is hashed by its golden replays); with the GH #830
    /// rung-4 knob on it reports the reorder counters, in their order.
    #[test]
    fn search_tree_reports_expand_order_only_with_the_knob() {
        let st = fixture();
        let p = RecommendParams {
            sims: 64,
            ..small_params(false)
        };
        let off = search_tree_value(&st, &p);
        assert!(
            off.get("expand_order").is_none(),
            "{:?}",
            off.get("expand_order")
        );
        assert!(!p.mcts_config().net_expand_order);
        let on_params = RecommendParams {
            net_expand_order: true,
            ..p
        };
        assert!(on_params.mcts_config().net_expand_order);
        let on = search_tree_value(&st, &on_params);
        let eo = &on["expand_order"];
        let n = |k: &str| eo[k].as_u64().unwrap_or_else(|| panic!("{k} in {eo}"));
        assert!(n("considered") > 0, "{eo}");
        assert!(n("considered") >= n("scored"));
        assert!(n("scored") >= n("reordered"));
        assert!(n("reordered") >= n("first_changed"));
    }

    #[test]
    fn action_rows_carry_mean_final_hp() {
        let st = fixture();
        for fair in [false, true] {
            let out = recommend_value(&st, &small_params(fair), true);
            let rows = out["actions"].as_array().unwrap();
            assert!(!rows.is_empty());
            assert!(
                rows.iter()
                    .any(|r| r["hp_n"].as_u64().unwrap() > 0 && r["mean_final_hp"].is_f64()),
                "fair={fair}: {rows:?}"
            );
            for r in rows {
                assert!(r["hp_n"].as_u64().unwrap() <= r["visits"].as_u64().unwrap());
            }
        }
    }

    #[test]
    fn mean_final_hp_null_without_samples() {
        let v = action_score_value(
            &Action::EndTurn,
            10,
            5,
            5,
            0,
            1.0,
            0.0,
            0.0,
            0,
            false,
            false,
            false,
        );
        assert!(v["mean_final_hp"].is_null());
        assert!(v["hp_sd"].is_null());
        assert_eq!(v["hp_n"], json!(0));
        let v = action_score_value(
            &Action::EndTurn,
            10,
            5,
            5,
            0,
            1.0,
            30.0,
            230.0, // 6, 7, 8, 9
            4,
            false,
            false,
            false,
        );
        assert_eq!(v["mean_final_hp"], json!(7.5));
        let sd = v["hp_sd"].as_f64().unwrap();
        assert!((sd - 1.25_f64.sqrt()).abs() < 1e-9, "{sd}");
    }

    #[test]
    fn score_action_matches_fair_row() {
        let st = fixture();
        let p = small_params(true);
        let out = recommend_value(&st, &p, true);
        for row in out["actions"].as_array().unwrap() {
            let mut expect = row.clone();
            // The engine's ranking of the row among its siblings, which a
            // lone row cannot carry (GH #1022).
            for k in ["pick_rank", "in_band"] {
                expect.as_object_mut().unwrap().remove(k);
            }
            let a = legal_actions(&st)
                .into_iter()
                .find(|a| combat_action_to_value(a) == row["action"])
                .expect("row action is legal");
            for k in ["is_best", "low_confidence", "q_saturated"] {
                expect[k] = json!(false);
            }
            let mut got = score_action(&st, &p, &a);
            for k in ["is_best", "low_confidence", "q_saturated"] {
                got[k] = json!(false);
            }
            assert_eq!(got, expect);
        }
    }

    #[test]
    fn bad_modes_are_rejected_loudly() {
        let e = RecommendParams::from_config(&json!({"hidden_info": "psychic"}), "mcts_recommend")
            .unwrap_err();
        assert!(e.starts_with("mcts_recommend: hidden_info"), "{e}");
        let e =
            RecommendParams::from_config(&json!({"rollout_policy": "greedy"}), "x").unwrap_err();
        assert!(e.contains("rollout_policy"), "{e}");
    }

    #[test]
    fn an_explicit_rollout_policy_wins_over_the_dialect() {
        let honest = |extra: Value| {
            let mut c = json!({"hidden_info": "beliefdraw"});
            c.as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            RecommendParams::from_config(&c, "x")
                .expect("parses")
                .mcts_config()
                .rollout_policy
        };
        assert_eq!(honest(json!({})), RolloutPolicyKind::Net, "mode default");
        for (name, kind) in [
            ("heuristic", RolloutPolicyKind::Heuristic),
            ("uniform", RolloutPolicyKind::Uniform),
            ("net", RolloutPolicyKind::Net),
        ] {
            assert_eq!(honest(json!({"rollout_policy": name})), kind, "{name}");
        }
        let leaky = RecommendParams::from_config(&json!({"hidden_info": "clairvoyant"}), "x")
            .expect("parses")
            .mcts_config();
        assert_eq!(leaky.rollout_policy, RolloutPolicyKind::Heuristic);
    }

    #[test]
    fn output_ranks_by_q_then_terminal_win_then_visits() {
        let s =
            |visits: u32, wins: u32, value_sum: f32, terminal_win: bool, idx: u8| RecommendScore {
                action: Action::PlayCard {
                    hand_index: idx,
                    target: None,
                },
                visits,
                wins,
                losses: visits - wins,
                ties: 0,
                value_sum,
                hp_sum: 0.0,
                hp_sq_sum: 0.0,
                hp_n: 0,
                terminal_win,
            };
        let out = recommend_output_value(
            vec![
                s(100, 50, 50.0, false, 0), // q 0.5
                s(1, 1, 1.0, true, 1),      // q 1.0, terminal win
                s(10, 10, 10.0, false, 2),  // q 1.0, more visits but no terminal win
                s(2, 0, 0.0, false, 3),     // low confidence (below 5% of 100)
            ],
            &[Action::EndTurn],
            None,
            false,
            113,
            1,
            true,
        );
        let acts = out["actions"].as_array().unwrap();
        let idx = |i: usize| acts[i]["action"]["card_index"].as_u64().unwrap();
        assert_eq!([idx(0), idx(1), idx(2), idx(3)], [1, 2, 0, 3]);
        assert_eq!(acts[0]["is_best"], json!(true));
        assert_eq!(acts[1]["is_best"], json!(false));
        assert_eq!(acts[3]["low_confidence"], json!(true));
        assert_eq!(acts[0]["low_confidence"], json!(true)); // 1 visit < ceil(5)
        assert_eq!(out["q_saturated"], json!(false));
        assert_eq!(out["pv"][0]["action"], json!("end_turn"));
        assert_eq!(out["sims"], json!(113));
        assert_eq!(acts[0]["bucket"], json!(hint::win_prob_bucket(1.0)));
    }

    /// The boss win-value carry (GH #771) travels from a config dict to the
    /// search config; absent, the default win value stays.
    #[test]
    fn win_carry_reaches_the_search_config() {
        let none = RecommendParams::default().mcts_config();
        assert_eq!((none.win_hp_carry, none.win_potion_carry), (None, None));
        let p = RecommendParams::from_config(
            &json!({"win_hp_carry": 0.0, "win_potion_carry": 0.0}),
            "t",
        )
        .unwrap();
        let cfg = p.mcts_config();
        assert_eq!(
            (cfg.win_hp_carry, cfg.win_potion_carry),
            (Some(0.0), Some(0.0))
        );
    }

    /// The `is_best` row is the move the search itself picked
    /// (`pick_root_action`), even when a tie-band / guard rule put it below
    /// the Q argmax; the rest keep the Q order.
    #[test]
    fn engine_pick_is_the_best_row() {
        let s = |visits: u32, value_sum: f32, idx: u8| RecommendScore {
            action: Action::PlayCard {
                hand_index: idx,
                target: None,
            },
            visits,
            wins: 0,
            losses: visits,
            ties: 0,
            value_sum,
            hp_sum: 0.0,
            hp_sq_sum: 0.0,
            hp_n: 0,
            terminal_win: false,
        };
        let engine = EnginePick {
            pick: RootPick {
                action: Action::PlayCard {
                    hand_index: 2,
                    target: None,
                },
                idx: Some(2),
                rule: sts2_mcts::PickRule::BlockGuard,
                band: [0, 1, 2].into_iter().collect(),
                runner_up: Some(0),
                gap_q: -0.015,
                gap_z: None,
            },
            // The fallback after the pick is NOT the Q order.
            order: vec![2, 1, 0],
            tie_eps: 0.02,
        };
        let mut scores = vec![s(100, 50.0, 0), s(100, 49.0, 1), s(100, 48.5, 2)];
        scores[0].hp_sum = 300.0;
        scores[0].hp_n = 10;
        scores[2].hp_sum = 280.0;
        scores[2].hp_n = 10;
        let out = recommend_output_value(scores, &[], Some(&engine), false, 300, 1, true);
        let acts = out["actions"].as_array().unwrap();
        let idx = |i: usize| acts[i]["action"]["card_index"].as_u64().unwrap();
        assert_eq!([idx(0), idx(1), idx(2)], [2, 0, 1]);
        assert_eq!(acts[0]["is_best"], json!(true));
        assert_eq!(acts[1]["is_best"], json!(false));
        let rank = |i: usize| acts[i]["pick_rank"].as_u64().unwrap();
        assert_eq!([rank(0), rank(1), rank(2)], [0, 2, 1]);
        assert!(acts.iter().all(|a| a["in_band"] == json!(true)));
        let pick = &out["pick"];
        assert_eq!(pick["rule"], json!("block_guard"));
        assert_eq!(
            pick["reason"],
            json!(sts2_mcts::PickRule::BlockGuard.reason())
        );
        // The runner-up (score 0) ended 2 HP above the pick.
        assert!((pick["gap_hp"].as_f64().unwrap() + 2.0).abs() < 1e-9);
        assert_eq!(pick["gap_z"], Value::Null);
    }

    /// A real search explains itself: `pick` names a rule, the `is_best`
    /// row is rank 0 and in its own band, and the ranks are a permutation.
    /// The fair mode has no single pick: `pick` is null and rows rank in
    /// row order, outside any band.
    #[test]
    fn recommend_explains_its_pick() {
        let st = fixture();
        for (fair, threads) in [(false, 1), (false, 2), (true, 1)] {
            let out = recommend_value(
                &st,
                &RecommendParams {
                    threads,
                    ..small_params(fair)
                },
                true,
            );
            let acts = out["actions"].as_array().unwrap();
            assert!(acts.len() > 1, "fixture needs a choice");
            let mut ranks: Vec<u64> = acts
                .iter()
                .map(|a| a["pick_rank"].as_u64().unwrap())
                .collect();
            if fair {
                assert_eq!(out["pick"], Value::Null);
                assert_eq!(ranks, (0..acts.len() as u64).collect::<Vec<_>>());
                assert!(acts.iter().all(|a| a["in_band"] == json!(false)));
                continue;
            }
            let rule = out["pick"]["rule"].as_str().expect("a rule");
            assert_eq!(acts[0]["pick_rank"], json!(0));
            // Only an exact rule decides outside the statistics' band.
            let exact = ["lost_position", "proven_win", "illegal_fallback"].contains(&rule);
            assert_eq!(
                acts[0]["in_band"],
                json!(!exact),
                "threads={threads}: {out}"
            );
            ranks.sort_unstable();
            assert_eq!(ranks, (0..acts.len() as u64).collect::<Vec<_>>());
        }
    }

    #[test]
    fn empty_scores_are_a_valid_result() {
        let out = recommend_output_value(Vec::new(), &[], None, false, 0, 1, false);
        assert_eq!(out["actions"], json!([]));
        assert_eq!(out["q_saturated"], json!(false));
        assert_eq!(out["intent_damage_known"], json!(false));
    }

    /// GH #830: `leaf_playouts` travels from a config dict to the search
    /// config; absent, the engine's single playout stays. And it reaches the
    /// search: four playouts per leaf move the recommendation's values.
    #[test]
    fn leaf_playouts_reaches_the_search() {
        assert_eq!(RecommendParams::default().leaf_playouts, None);
        assert_eq!(RecommendParams::default().mcts_config().leaf_playouts, 1);
        let p = RecommendParams::from_config(&json!({"leaf_playouts": 4}), "t").unwrap();
        assert_eq!(p.leaf_playouts, Some(4));
        assert_eq!(p.mcts_config().leaf_playouts, 4);

        let st = fixture();
        let one = recommend_value(&st, &small_params(false), true);
        let four = recommend_value(
            &st,
            &RecommendParams {
                leaf_playouts: Some(4),
                ..small_params(false)
            },
            true,
        );
        let q = |v: &Value| -> Vec<f64> {
            v["actions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| a["q"].as_f64().unwrap())
                .collect()
        };
        assert_ne!(q(&one), q(&four), "leaf_playouts=4 left every Q unchanged");
    }
}
