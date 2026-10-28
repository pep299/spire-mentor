//! Turn-plan teacher pricing (GH #830, turn-plan search G0).
//!
//! Prices each enumerated [`TurnPlan`] with the deployed combat search
//! forced onto the plan's last action. These prices are the teacher Q a
//! plan scorer trains on, so the search is built by
//! [`MctsCombatPolicy::build_search`] — the same builder the deployed pick
//! uses — never by a second config path.
//!
//! The plan's prefix is applied to the caller's RAW combat; the search then
//! applies its own hidden-info root conversion. Plans are enumerated on a
//! belief-view clone, so this replay is valid only when the belief view and
//! the raw state agree on every prefix action. That does NOT hold on the
//! pre-#785 engine for Havoc and Cascade: they pop the top draw card without
//! consuming RNG, so the enumerator extends plans past them with the belief
//! view's top card while this replay plays the true one (and panics when that
//! card opens a choice). G1 must replay on the enumerator's root view
//! (#912).

use sts2_core::full_run::RunPolicy;
use sts2_core::run_manager::RunManager;
use sts2_core::{apply, CombatState};
use sts2_mcts::terminal_state_value_with;
use sts2_mcts::turn_plan::{PlanEnd, TurnPlan};

use crate::policy::MctsCombatPolicy;

/// Teacher price of one plan, on the search's value scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanPrice {
    /// Mean backed-up value of the plan's last action.
    pub mean_value: f32,
    /// Simulations that reached the last action's child; 0 for a
    /// `Terminal` plan, which is valued without a search.
    pub visits: u32,
    /// Population variance of the backed-up values (0 for `Terminal`).
    pub value_variance: f32,
    /// Mean final player HP over the simulations that reached a decided
    /// end, `None` when none did.
    pub mean_final_hp: Option<f32>,
}

/// Price every plan in `plans` (enumerated from `combat`), in order.
///
/// Plan `i` is searched with seed `seed ^ i`. A `Terminal` plan is valued
/// with [`terminal_state_value_with`] under the config and leaf evaluator
/// `build_search` constructs for its prefix state, so it sits on the same
/// scale as the searched plans.
///
/// # Panics
///
/// When a plan is empty, or when the forced search cannot price a
/// non-terminal plan's last action (`search_forced` returned `None`):
/// a missing label is a bug upstream, never a default price.
pub fn price_plans<M: RunPolicy>(
    policy: &mut MctsCombatPolicy<M>,
    rm: &RunManager,
    combat: &CombatState,
    plans: &[TurnPlan],
    sims_per_plan: u32,
    seed: u64,
) -> Vec<PlanPrice> {
    plans
        .iter()
        .enumerate()
        .map(|(i, plan)| {
            let (last, prefix) = plan
                .actions
                .split_last()
                .unwrap_or_else(|| panic!("empty turn plan {i}"));
            let mut state = combat.clone();
            for a in prefix {
                let _ = apply(&mut state, a.clone());
            }
            let mut search = policy.build_search(rm, &state, sims_per_plan, seed ^ i as u64);
            if plan.end == PlanEnd::Terminal {
                let _ = apply(&mut state, last.clone());
                // The search's own final-HP bookkeeping (`final_hp_of`).
                #[allow(clippy::cast_precision_loss)]
                let hp = state.player.current_hp.max(0) as f32;
                return PlanPrice {
                    mean_value: terminal_state_value_with(
                        &state,
                        &search.config,
                        search.leaf_eval(),
                    ),
                    visits: 0,
                    value_variance: 0.0,
                    mean_final_hp: Some(hp),
                };
            }
            let outcome = search
                .search_forced(last, sims_per_plan)
                .unwrap_or_else(|| panic!("search_forced could not price plan {i}: {plan:?}"));
            PlanPrice {
                mean_value: outcome.mean_value,
                visits: outcome.visits,
                value_variance: outcome.value_variance(),
                mean_final_hp: outcome.mean_final_hp(),
            }
        })
        .collect()
}
