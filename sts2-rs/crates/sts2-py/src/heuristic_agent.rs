//! `HeuristicRunAgent` — the heuristic-meta stepwise run agent.
//!
//! Drives the deployed combat cell `MctsCombatPolicy<HeuristicPolicy>` one
//! decision at a time on a live `PyRunManager`, with the tier-1
//! all-surfaces [`HeuristicPolicy`] making the card-shaped picks. Combat
//! is the frozen MCTS.
//!
//! This is the character-agnostic expert used to bootstrap a PPO campaign:
//! the heuristic `argmax ΔV` covers card rewards / removes / upgrades, the
//! optionality heuristic covers pathing/shop/rest/events, and combat stays
//! MCTS. Requires no net weights.
//!
//! Exposed as `sts2_rs.HeuristicRunAgent` and registered in `lib.rs`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use sts2_core::full_run::RunPolicy;
use sts2_core::run_manager::Phase;
use sts2_mcts::RootOutcome;
use sts2_runai::policy::{CombatBudget, HeuristicPolicy, MctsCombatPolicy};

use crate::run_manager::{phase_str, run_action_to_pydict, PyRunManager};

/// Python-facing stepwise agent that plays the tier-1 [`HeuristicPolicy`]
/// meta over MCTS combat on a live `RunManager`, emitting one per-decision
/// record per `step()` call — the same record shape as
/// [`crate::ppo_agent::PyPpoRunAgent`], so the PPO env drives either
/// interchangeably.
///
/// Combat budget defaults reproduce `--combat eval-cheap`:
/// `sims=1000, boss_sims=4000, threads=1`. Loads no net.
///
/// **`threads` is accepted and ignored** — this agent's combat search is
/// always single-threaded. See `new` for why, and do not present it to a
/// user as a live knob.
#[pyclass(name = "HeuristicRunAgent", module = "sts2_rs")]
#[derive(Debug)]
pub struct PyHeuristicRunAgent {
    policy: MctsCombatPolicy<HeuristicPolicy>,
}

#[pymethods]
impl PyHeuristicRunAgent {
    #[new]
    #[pyo3(signature = (seed = 0, sims = 1000, boss_sims = 4000, threads = 1))]
    fn new(seed: i32, sims: u32, boss_sims: u32, threads: usize) -> Self {
        // ACCEPTED AND IGNORED. Not "reserved" — nothing reads it, so a
        // caller passing threads=8 gets a single-threaded search and no
        // warning. eval-cheap is single-threaded by design (throughput
        // parallelises across seeds, not within one tree), and the sweep
        // lane stays single-threaded so arms are byte-comparable.
        //
        // Wiring `par_search` in here is a real option, but it SPLITS the
        // budget across workers, so it is not free: at 1000 sims / 8
        // workers it costs ~5.8pp of deep-oracle agreement, at 4000 / 8
        // ~2.5pp (`docs/experiments/2026-08-10-root-parallel-decision-
        // quality.md`). Doing it right means raising the sim budget to
        // keep sims-per-tree up, which is a separate call from "make the
        // knob do something".
        let _ = threads;

        let mut budget = CombatBudget::eval_cheap();
        budget.sims = sims;
        budget.boss_sims = boss_sims;

        let policy = MctsCombatPolicy::new(HeuristicPolicy::new(), budget, seed);
        Self { policy }
    }

    /// Retarget the combat search budget mid-run.
    ///
    /// The agent's MCTS RNG and its per-run decision counter are NOT
    /// reset, so a paused run resumes searching harder from exactly where
    /// it stopped — the workbench's "stop, raise the sims, resume" flow.
    /// Constructing a fresh agent instead would rewind the decision
    /// counter and replay per-decision seeds the run already spent.
    #[pyo3(signature = (sims = None, boss_sims = None))]
    fn set_budget(&mut self, sims: Option<u32>, boss_sims: Option<u32>) -> PyResult<()> {
        let mut budget = self.policy.budget();
        if let Some(v) = sims {
            if v == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "sims must be positive",
                ));
            }
            budget.sims = v;
        }
        if let Some(v) = boss_sims {
            if v == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "boss_sims must be positive",
                ));
            }
            budget.boss_sims = v;
        }
        self.policy.set_budget(budget);
        Ok(())
    }

    /// The combat search budget in force right now, as
    /// `{"sims": int, "boss_sims": int}` — so a caller can tell whether a
    /// :meth:`set_budget` actually landed on the live agent.
    fn budget<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let b = self.policy.budget();
        let d = PyDict::new(py);
        d.set_item("sims", b.sims)?;
        d.set_item("boss_sims", b.boss_sims)?;
        Ok(d)
    }

    /// Apply one decision to `rm` and return a step record dict, or `None`
    /// if the policy has no legal action to take. Identical keys/semantics
    /// to [`crate::ppo_agent::PyPpoRunAgent::step`].
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

        // Capture the combat search AFTER `choose` (runs MCTS, caches it)
        // but BEFORE `take_action` (advances state). Only when MCTS ran.
        let combat_search: Option<Bound<'py, PyDict>> = if is_combat {
            self.policy
                .last_combat_search()
                .map(|rec| -> PyResult<Bound<'py, PyDict>> {
                    let d = PyDict::new(py);
                    let best_idx = rec
                        .outcomes
                        .iter()
                        .position(|o| o.action == rec.chosen)
                        .unwrap_or(0);
                    let list = PyList::empty(py);
                    for (i, o) in rec.outcomes.iter().enumerate() {
                        let entry = root_outcome_to_pydict(py, o, i == best_idx)?;
                        list.append(entry)?;
                    }
                    d.set_item("actions", list)?;
                    d.set_item("recommended", crate::action_to_pydict(py, &rec.chosen)?)?;
                    Ok(d)
                })
                .transpose()?
        } else {
            None
        };

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
        out.set_item("result", format!("{result:?}"))?;
        Ok(Some(out))
    }
}

/// Serialise one [`RootOutcome`] into the same per-action dict shape that
/// `mcts_recommend` emits, reusing `action_score_pydict`.
pub(crate) fn root_outcome_to_pydict<'py>(
    py: Python<'py>,
    o: &RootOutcome,
    is_best: bool,
) -> PyResult<Bound<'py, PyDict>> {
    crate::action_score_pydict(
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
    )
}
