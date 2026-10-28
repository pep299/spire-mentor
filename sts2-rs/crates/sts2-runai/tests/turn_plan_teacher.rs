//! Teacher pricing gates (GH #830, turn-plan G0 Task 3).
//!
//! `price_plans` prices every enumerated plan with the deployed combat
//! search forced onto the plan's last action; the search comes from
//! `MctsCombatPolicy::build_search`, the builder the deployed pick uses.
//! The byte-identity of that extraction is pinned by
//! `policy::tests::build_search_refactor_is_byte_identical` (it needs the
//! crate-private `mcts_pick`).

#[path = "../../sts2-mcts/tests/common/mod.rs"]
mod common;

use common::{build_vertical_slice, fixture};
use sts2_content::run_callbacks::new_run_manager_for_character;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::RunManager;
use sts2_core::{apply, is_terminal, Action, CombatState};
use sts2_mcts::terminal_state_value_with;
use sts2_mcts::turn_plan::{enumerate_plans, PlanCap, PlanEnd, TurnPlan};
use sts2_mcts::HiddenInfoMode;
use sts2_runai::policy::{CombatBudget, FirstOptionPolicy, LeafKind, MctsCombatPolicy};
use sts2_runai::turn_plan_teacher::{price_plans, PlanPrice};

const SEED: u64 = 0x5EED;

fn policy(sims: u32) -> MctsCombatPolicy<FirstOptionPolicy> {
    let budget = CombatBudget {
        sims,
        boss_sims: sims,
        elite_sims: None,
        motif_sims: None,
        leaf: LeafKind::TexelWide,
    };
    MctsCombatPolicy::new(FirstOptionPolicy, budget, 7)
}

fn rm() -> RunManager {
    new_run_manager_for_character(7, CharacterId::Ironclad)
}

fn plans(state: &CombatState) -> Vec<TurnPlan> {
    let set = enumerate_plans(state, HiddenInfoMode::BeliefDraw, PlanCap::default());
    assert!(!set.cap_hit);
    set.plans
}

fn plan_index(plans: &[TurnPlan], end: PlanEnd) -> usize {
    plans
        .iter()
        .position(|p| p.end == end)
        .unwrap_or_else(|| panic!("no {end:?} plan in {plans:?}"))
}

#[test]
fn lethal_plan_prices_above_end_turn() {
    let state = fixture(&[CardId::StrikeIronclad], 6, &[], 3);
    let plans = plans(&state);
    let strike = plans
        .iter()
        .position(|p| matches!(p.actions[..], [Action::PlayCard { hand_index: 0, .. }]))
        .unwrap_or_else(|| panic!("no [Strike] plan in {plans:?}"));
    let end_turn = plans
        .iter()
        .position(|p| p.actions == [Action::EndTurn])
        .unwrap_or_else(|| panic!("no [EndTurn] plan in {plans:?}"));
    let prices = price_plans(&mut policy(200), &rm(), &state, &plans, 200, SEED);
    assert_eq!(prices.len(), plans.len());
    assert!(
        prices[strike].mean_value > prices[end_turn].mean_value,
        "strike {:?} vs end turn {:?}",
        prices[strike],
        prices[end_turn]
    );
    assert!(prices[end_turn].visits > 0, "EndTurn is searched");
}

#[test]
#[allow(clippy::float_cmp)] // same function, same inputs: bit-equal by construction
fn terminal_plan_priced_without_search() {
    let state = fixture(&[CardId::StrikeIronclad], 6, &[], 3);
    let plans = plans(&state);
    let i = plan_index(&plans, PlanEnd::Terminal);
    let plan = &plans[i];
    let sims = 200;
    let prices = price_plans(&mut policy(sims), &rm(), &state, &plans, sims, SEED);
    let price = prices[i];
    assert_eq!(price.visits, 0, "a terminal plan runs no search");
    assert_eq!(price.value_variance, 0.0);

    // The expected value on the scale the searched plans use: the config
    // and leaf `build_search` constructs for this plan's prefix state.
    let (last, prefix) = plan.actions.split_last().expect("non-empty plan");
    let mut s = state.clone();
    for a in prefix {
        let _ = apply(&mut s, a.clone());
    }
    let search = policy(sims).build_search(&rm(), &s, sims, SEED ^ i as u64);
    let _ = apply(&mut s, last.clone());
    assert!(is_terminal(&s));
    let expected = terminal_state_value_with(&s, &search.config, search.leaf_eval());
    assert_eq!(price.mean_value, expected);
    #[allow(clippy::cast_precision_loss)]
    let hp = s.player.current_hp.max(0) as f32;
    assert_eq!(price.mean_final_hp, Some(hp));
}

#[test]
fn pricing_is_deterministic() {
    let state = build_vertical_slice(7);
    let mut plans = plans(&state);
    // A handful of plans keeps the debug-build run short.
    plans.truncate(6);
    let sims = 60;
    let a = price_plans(&mut policy(sims), &rm(), &state, &plans, sims, SEED);
    let b = price_plans(&mut policy(sims), &rm(), &state, &plans, sims, SEED);
    let bits = |ps: &[PlanPrice]| -> Vec<(u32, u32, u32, Option<u32>)> {
        ps.iter()
            .map(|p| {
                (
                    p.mean_value.to_bits(),
                    p.visits,
                    p.value_variance.to_bits(),
                    p.mean_final_hp.map(f32::to_bits),
                )
            })
            .collect()
    };
    assert_eq!(bits(&a), bits(&b));
    assert!(a.iter().all(|p| p.visits > 0 && p.visits <= sims), "{a:?}");
}
