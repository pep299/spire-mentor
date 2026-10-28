//! `MctsCombatAgent` — a net-free stepwise combat agent for PPO
//! net-play (GH #220).
//!
//! In PPO net-play mode (`ppo.env.PpoRunEnv` with `--policy net` /
//! `net_greedy` / `random`) the agent is consulted **only** for
//! `Phase::Combat` decisions: every multi-option non-combat surface is
//! yielded to the PPO net, and single-option surfaces auto-resolve. The
//! combat search itself is a heuristic-leaf MCTS (see
//! `sts2_runai::policy::MctsCombatPolicy::mcts_pick`).
//!
//! It wraps the `MctsCombatPolicy` combat search under a
//! [`HeuristicPolicy`] meta, so no net weights are needed. Because the
//! meta is only ever reached on a combat MCTS→meta fallback (never for a
//! yielded PPO decision), a heuristic meta keeps the fallback graceful
//! (first-option) instead of panicking.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sts2_core::full_run::RunPolicy;
use sts2_core::run_manager::Phase;
use sts2_core::run_rng_set::splitmix64;
use sts2_core::Side;
use sts2_mcts::rollout_net::{heuristic_mirror_net, net_for};
use sts2_mcts::turn_plan::{enumerate_plans, locate_plan, PlanCap, PlanEnd};
use sts2_mcts::turn_plan_features::{
    anchors, deck_context, plan_cards, plan_features, plan_logprob,
};
use sts2_mcts::RootOutcome;
use sts2_runai::policy::{deployed_hidden_info, CombatBudget, HeuristicPolicy, MctsCombatPolicy};
use sts2_runai::turn_plan_teacher::price_plans;

use crate::run_manager::{phase_str, run_action_to_pydict, PyRunManager};

/// Python-facing stepwise agent that plays a net-free
/// `MctsCombatPolicy<HeuristicPolicy>` on a live `RunManager`, emitting one
/// per-decision record per `step()` call, for PPO net-play (combat MCTS +
/// PPO meta).
///
/// Defaults reproduce `run_arena`'s combat search exactly:
/// `sims=1000, boss_sims=4000, threads=1` (eval-cheap).
#[pyclass(name = "MctsCombatAgent", module = "sts2_rs")]
#[derive(Debug)]
pub struct PyMctsCombatAgent {
    policy: MctsCombatPolicy<HeuristicPolicy>,
    /// The constructor seed, which [`Self::turn_plan_labels`] mixes into
    /// its per-position seed.
    seed: i32,
}

/// Salt that keeps turn-plan label seeds off the decision seeds
/// `mcts_pick` derives from the same `(seed, counter)` pair.
const TURN_PLAN_SEED_SALT: u64 = 0x7E41_C0DE_5EED_0830;

#[pymethods]
impl PyMctsCombatAgent {
    #[new]
    #[pyo3(signature = (seed = 0, sims = 1000, boss_sims = 4000, threads = 1))]
    fn new(seed: i32, sims: u32, boss_sims: u32, threads: usize) -> Self {
        // `threads` is reserved for future parallelism; eval-cheap is
        // single-threaded by design (throughput parallelises across seeds).
        // Kept in the signature so `MctsCombatAgent` is a drop-in for
        // `HeuristicRunAgent`.
        let _ = threads;

        let mut budget = CombatBudget::eval_cheap();
        budget.sims = sims;
        budget.boss_sims = boss_sims;

        let policy = MctsCombatPolicy::new(HeuristicPolicy::new(), budget, seed);
        Self { policy, seed }
    }

    /// Apply one decision to `rm` and return a step record dict, or `None`
    /// if the policy has no legal action to take.
    ///
    /// Identical shape/semantics to `HeuristicRunAgent.step` — see that
    /// method's doc comment for the key list. The `combat_search` record is
    /// non-`None` only on a combat decision where the MCTS ran (i.e. no
    /// fallback to the meta).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        rm: &mut PyRunManager,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        // Bind a pending event room so the policy sees its real options,
        // as `drive_full_run` does (GH #841).
        rm.inner.bind_pending_event();
        let actions = rm.inner.get_available_actions();
        if actions.is_empty() {
            return Ok(None);
        }

        let phase = rm.inner.phase();
        let is_combat = phase == Phase::Combat;

        let Some(idx) = self.policy.choose(&rm.inner, &actions) else {
            return Ok(None);
        };
        let action = actions[idx];
        let retained_visits = if is_combat {
            self.policy.last_retained_visits()
        } else {
            None
        };

        // Capture the combat search record AFTER `choose` (which runs the
        // MCTS and caches it in `last_combat_search`) but BEFORE
        // `take_action` (which advances the state). Only meaningful when the
        // MCTS ran, i.e. `is_combat` AND the policy didn't fall back to the
        // meta — same rationale as `HeuristicRunAgent::step`.
        let combat_search: Option<Bound<'py, PyDict>> = if is_combat {
            self.policy
                .last_combat_search()
                .map(|rec| -> PyResult<Bound<'py, PyDict>> {
                    let d = PyDict::new(py);

                    // Build the per-action list. Compute the best index once
                    // so exactly one entry carries `is_best = true` even if
                    // the same action appears more than once in root_outcomes.
                    let best_idx = rec
                        .outcomes
                        .iter()
                        .position(|o| o.action == rec.chosen)
                        .unwrap_or(0);
                    let list = PyList::empty(py);
                    for (i, o) in rec.outcomes.iter().enumerate() {
                        let is_best = i == best_idx;
                        let entry = root_outcome_to_pydict(py, o, is_best)?;
                        list.append(entry)?;
                    }
                    d.set_item("actions", list)?;

                    // The recommended action (chosen by Q-then-visits).
                    d.set_item("recommended", crate::action_to_pydict(py, &rec.chosen)?)?;
                    // What `net_expand_order` (GH #830 rung 4) did in this
                    // search; all zero with the knob off or on a forced move.
                    let eo = PyDict::new(py);
                    eo.set_item("considered", rec.expand_order.considered)?;
                    eo.set_item("scored", rec.expand_order.scored)?;
                    eo.set_item("reordered", rec.expand_order.reordered)?;
                    eo.set_item("first_changed", rec.expand_order.first_changed)?;
                    d.set_item("expand_order", eo)?;
                    Ok(d)
                })
                .transpose()?
        } else {
            None
        };

        // Turn-plan root filter (GH #830); `None` with the knob off.
        let root_filter: Option<Bound<'py, PyDict>> = if is_combat {
            self.policy
                .last_root_filter()
                .map(|f| -> PyResult<Bound<'py, PyDict>> {
                    let d = PyDict::new(py);
                    d.set_item("applied", f.applied)?;
                    d.set_item("fallback", f.fallback)?;
                    d.set_item("legal_n", f.legal_n)?;
                    d.set_item("plans_n", f.plans_n)?;
                    d.set_item("kept_n", f.kept_n)?;
                    d.set_item("allowed_n", f.allowed_n)?;
                    d.set_item("chosen_via", f.chosen_via)?;
                    d.set_item("filter_us", f.filter_us)?;
                    Ok(d)
                })
                .transpose()?
        } else {
            None
        };

        // Apply the chosen action.
        rm.ensure_play_log_flag();
        let result = rm.inner.take_action(action);

        let over = rm.inner.is_over();
        let won = rm.inner.run_state.player_won;

        let out = PyDict::new(py);
        out.set_item("phase", phase_str(phase))?;
        out.set_item("is_combat", is_combat)?;
        out.set_item("over", over)?;
        out.set_item("won", won)?;
        out.set_item("action", run_action_to_pydict(py, &action)?)?;
        out.set_item("combat_search", combat_search)?;
        // `tree_reuse` (GH #830): root visits this decision's search kept
        // from the one before (`None`: fresh tree, no search, or knob off).
        out.set_item("retained_visits", retained_visits)?;
        out.set_item("root_filter", root_filter)?;
        out.set_item("result", format!("{result:?}"))?;
        Ok(Some(out))
    }

    /// Teacher labels for every deterministic plan of the current player
    /// turn (GH #830), or `None` when `rm` is not on the player's turn of a
    /// live combat. Does not advance `rm`, and leaves the decision counter
    /// (hence every later `step()`) untouched.
    ///
    /// Plans are enumerated under the deployed search's hidden-info mode
    /// and priced by `price_plans` through this agent's own policy, so the
    /// env/override bundle is the deployed one. The seed mixes the agent
    /// seed with its decision counter through splitmix64: `price_plans`
    /// seeds plan `i` with `seed ^ i`, so adjacent raw position seeds would
    /// share plan seeds.
    ///
    /// Returns `{"cap_hit", "n_plans", "deck": {"cards", "relics",
    /// "powers": [[id, amount]], "ctx"}, "plans": [{"actions" (the
    /// `combat_search` action dicts of `step()`), "end": "end_turn" |
    /// "chance" | "terminal", "features", "cards", "logp_heur", "logp_rl2"
    /// (None without a playout net), "anchors": {"lethal", "max_block",
    /// "max_damage", "heuristic", "rl2"}, "q", "visits", "q_var",
    /// "final_hp"}]}`.
    ///
    /// `sims_per_plan == 0` skips the pricing: `q`, `visits`, `q_var` and
    /// `final_hp` are `None`. With `locate` (a played turn, as bridge action
    /// dicts against the current state) the result also holds
    /// `"locate": {"index", "used", "end", "failure"}` from
    /// `turn_plan::locate_plan`: the plan the played turn is, if listed.
    #[pyo3(signature = (rm, sims_per_plan, max_plans = 512, locate = None))]
    fn turn_plan_labels<'py>(
        &mut self,
        py: Python<'py>,
        rm: &PyRunManager,
        sims_per_plan: u32,
        max_plans: usize,
        locate: Option<Vec<Bound<'py, PyDict>>>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        if rm.inner.phase() != Phase::Combat {
            return Ok(None);
        }
        let Some(combat) = rm.inner.combat.as_ref() else {
            return Ok(None);
        };
        if combat.is_over || combat.side != Side::Player {
            return Ok(None);
        }

        let cap = PlanCap {
            max_plans,
            ..PlanCap::default()
        };
        let set = enumerate_plans(combat, deployed_hidden_info(), cap);
        let flags = anchors(combat, &set);
        let heur = heuristic_mirror_net();
        let rl2 = net_for(combat);
        let seed_bits = u64::from(u32::from_ne_bytes(self.seed.to_ne_bytes()));
        let seed = splitmix64(
            splitmix64(seed_bits ^ TURN_PLAN_SEED_SALT) ^ self.policy.combat_decisions(),
        );
        let prices = if sims_per_plan == 0 {
            vec![None; set.plans.len()]
        } else {
            price_plans(
                &mut self.policy,
                &rm.inner,
                combat,
                &set.plans,
                sims_per_plan,
                seed,
            )
            .into_iter()
            .map(Some)
            .collect()
        };

        let ctx = deck_context(combat);
        let deck = PyDict::new(py);
        deck.set_item("cards", &ctx.cards)?;
        deck.set_item("relics", &ctx.relics)?;
        let powers = PyList::empty(py);
        for (id, amount) in &ctx.powers {
            let pair = PyList::empty(py);
            pair.append(id)?;
            pair.append(amount)?;
            powers.append(pair)?;
        }
        deck.set_item("powers", powers)?;
        deck.set_item("ctx", ctx.ctx.to_vec())?;

        let plans = PyList::empty(py);
        for ((plan, flag), price) in set.plans.iter().zip(&flags).zip(&prices) {
            let d = PyDict::new(py);
            let actions = PyList::empty(py);
            for a in &plan.actions {
                actions.append(crate::action_to_pydict(py, a)?)?;
            }
            d.set_item("actions", actions)?;
            d.set_item(
                "end",
                match plan.end {
                    PlanEnd::EndTurn => "end_turn",
                    PlanEnd::Chance => "chance",
                    PlanEnd::Terminal => "terminal",
                },
            )?;
            d.set_item("features", plan_features(combat, plan).to_vec())?;
            d.set_item("cards", plan_cards(combat, plan))?;
            d.set_item("logp_heur", plan_logprob(&heur, combat, plan))?;
            d.set_item("logp_rl2", rl2.map(|net| plan_logprob(net, combat, plan)))?;
            let a = PyDict::new(py);
            a.set_item("lethal", flag.lethal)?;
            a.set_item("max_block", flag.max_block)?;
            a.set_item("max_damage", flag.max_damage)?;
            a.set_item("heuristic", flag.heuristic)?;
            a.set_item("rl2", flag.rl2)?;
            d.set_item("anchors", a)?;
            d.set_item("q", price.as_ref().map(|p| p.mean_value))?;
            d.set_item("visits", price.as_ref().map(|p| p.visits))?;
            d.set_item("q_var", price.as_ref().map(|p| p.value_variance))?;
            d.set_item("final_hp", price.as_ref().and_then(|p| p.mean_final_hp))?;
            plans.append(d)?;
        }

        let out = PyDict::new(py);
        out.set_item("cap_hit", set.cap_hit)?;
        out.set_item("n_plans", set.plans.len())?;
        out.set_item("deck", deck)?;
        out.set_item("plans", plans)?;
        if let Some(played) = locate {
            let actions = played
                .iter()
                .map(crate::parse_bridge_action)
                .collect::<PyResult<Vec<_>>>()?;
            let loc = locate_plan(combat, deployed_hidden_info(), &set, &actions);
            let d = PyDict::new(py);
            d.set_item("index", loc.index)?;
            d.set_item("used", loc.used)?;
            d.set_item(
                "end",
                loc.end.map(|e| match e {
                    PlanEnd::EndTurn => "end_turn",
                    PlanEnd::Chance => "chance",
                    PlanEnd::Terminal => "terminal",
                }),
            )?;
            d.set_item("failure", loc.failure)?;
            out.set_item("locate", d)?;
        }
        Ok(Some(out))
    }

    /// `(elite decisions, boss decisions, motif-tagged common decisions,
    /// simulations requested)` over this agent's lifetime — the compute
    /// readout for benchmark arms. Forced moves count as decisions but
    /// request no simulations.
    fn combat_budget_spent(&self) -> (u64, u64, u64, u64) {
        self.policy.combat_budget_spent()
    }

    /// What `tree_reuse` (`STS2_RUNAI_MCTS_OVERRIDES=tree_reuse=1`, GH
    /// #830) did over this agent's lifetime, as a dict of counts (all zero
    /// with the knob off). `searched` = searched decisions; `reused` of them
    /// started from the previous decision's tree (`reused_via_chance`
    /// through a chance edge); each `fresh_*` counts a fresh tree by reason.
    /// Reuse rate = `reused / searched`; carried-over share of root visits =
    /// `retained_visits / (retained_visits + new_sims)`.
    fn tree_reuse_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let c = self.policy.tree_reuse_counts();
        let d = PyDict::new(py);
        d.set_item("searched", c.searched)?;
        d.set_item("reused", c.reused)?;
        d.set_item("reused_via_chance", c.reused_via_chance)?;
        d.set_item("fresh_no_tree", c.fresh_no_tree)?;
        d.set_item("fresh_not_expanded", c.fresh_not_expanded)?;
        d.set_item("fresh_mismatch", c.fresh_mismatch)?;
        d.set_item("fresh_outcome_unseen", c.fresh_outcome_unseen)?;
        d.set_item("fresh_edge_set", c.fresh_edge_set)?;
        d.set_item("fresh_other", c.fresh_other)?;
        d.set_item("retained_visits", c.retained_visits)?;
        d.set_item("new_sims", c.new_sims)?;
        d.set_item("forced_carried", c.forced_carried)?;
        d.set_item("forced_dropped", c.forced_dropped)?;
        Ok(d)
    }
}

/// Serialise one [`RootOutcome`] into the same per-action dict shape that
/// `mcts_recommend` emits, reusing `action_score_pydict` — byte-copy of
/// `heuristic_agent.rs`'s helper.
fn root_outcome_to_pydict<'py>(
    py: Python<'py>,
    o: &RootOutcome,
    is_best: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = crate::action_score_pydict(
        py,
        &o.action,
        o.visits,
        o.wins,
        o.losses,
        o.ties,
        o.value_sum,
        o.hp_sum,
        o.hp_sq_sum,
        o.hp_n,
        /* low_confidence */ false,
        /* q_saturated */ false,
        is_best,
    )?;
    // The child's return variance, so a caller can put the top-two gap in
    // standard errors (`ppo.pilot_human_decks --escalate`, GH #777). Kept off
    // the shared wire shape (`action_score_value`) that the workbench pins.
    d.set_item("value_var", o.value_variance())?;
    Ok(d)
}

/// The turn-plan root filter (GH #830) outside the search, for the offline
/// flip-rate measurement: which root actions the filtered search could
/// pick. `TurnPlanRootFilter(weights_path)` loads the exported scorer once.
#[pyclass(name = "TurnPlanRootFilter", module = "sts2_rs")]
#[derive(Debug)]
pub struct PyTurnPlanRootFilter {
    scorer: sts2_mcts::turn_plan_scorer::PlanScorer,
}

#[pymethods]
impl PyTurnPlanRootFilter {
    #[new]
    fn new(weights_path: &str) -> PyResult<Self> {
        let text = std::fs::read_to_string(weights_path)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{weights_path}: {e}")))?;
        let scorer = sts2_mcts::turn_plan_scorer::PlanScorer::from_json(&text)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{weights_path}: {e}")))?;
        Ok(Self { scorer })
    }

    /// `turn_plan_scorer::root_filter` at `rm`'s combat with top-`k`, or
    /// `None` off the player's turn of a live combat: `{"fallback": None |
    /// "forced" | "cap" | "empty", "legal_n", "plans_n", "kept_n",
    /// "allowed": [(action dict, via_topk)], "filter_us"}`.
    #[pyo3(signature = (rm, k, max_plans = 512))]
    fn filter<'py>(
        &self,
        py: Python<'py>,
        rm: &PyRunManager,
        k: usize,
        max_plans: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        if rm.inner.phase() != Phase::Combat {
            return Ok(None);
        }
        let Some(combat) = rm.inner.combat.as_ref() else {
            return Ok(None);
        };
        if combat.is_over || combat.side != Side::Player {
            return Ok(None);
        }
        let t0 = std::time::Instant::now();
        let cap = PlanCap {
            max_plans,
            ..PlanCap::default()
        };
        let f = sts2_mcts::turn_plan_scorer::root_filter(combat, &self.scorer, k, cap);
        let us = u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX);
        let d = PyDict::new(py);
        d.set_item(
            "fallback",
            f.fallback
                .map(sts2_mcts::turn_plan_scorer::FilterFallback::as_str),
        )?;
        d.set_item("legal_n", f.legal_n)?;
        d.set_item("plans_n", f.plans_n)?;
        d.set_item("kept_n", f.kept_n)?;
        let allowed = PyList::empty(py);
        for (a, top) in &f.allowed {
            let pair = PyList::empty(py);
            pair.append(crate::action_to_pydict(py, a)?)?;
            pair.append(*top)?;
            allowed.append(pair)?;
        }
        d.set_item("allowed", allowed)?;
        d.set_item("filter_us", us)?;
        Ok(Some(d))
    }
}
