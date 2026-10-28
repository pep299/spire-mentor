//! Turn-plan features, deck context, anchors and plan log-probabilities
//! (GH #830, turn-plan G0 Task 2).

mod common;

use common::{fixture, new_cards, push_draw};
use sts2_core::cards::CardId;
use sts2_core::{apply, legal_actions, tt_hash_v2, Action, CombatState};
use sts2_mcts::expert::END_TURN_PRIORITY;
use sts2_mcts::rollout_net::heuristic_mirror_net;
use sts2_mcts::turn_plan::{enumerate_plans, PlanCap, PlanEnd, PlanSet, TurnPlan};
use sts2_mcts::turn_plan_features::{
    anchors, deck_context, plan_cards, plan_features, plan_logprob, PLAN_DIM,
};
use sts2_mcts::HiddenInfoMode;

fn plans(state: &CombatState) -> PlanSet {
    let set = enumerate_plans(state, HiddenInfoMode::BeliefDraw, PlanCap::default());
    assert!(!set.cap_hit);
    set
}

/// The state just before `plan`'s last action, replayed on `state`.
fn before_last(state: &CombatState, plan: &TurnPlan) -> CombatState {
    let mut s = state.clone();
    for a in &plan.actions[..plan.actions.len() - 1] {
        let _ = apply(&mut s, a.clone());
    }
    s
}

#[test]
fn plan_dim_is_stable() {
    assert_eq!(PLAN_DIM, 18);
}

#[test]
fn bash_plan_records_vulnerable() {
    let state = fixture(&[CardId::Bash], 40, &[], 3);
    let set = plans(&state);
    let bash = set
        .plans
        .iter()
        .find(|p| p.actions.len() == 2 && p.end == PlanEnd::EndTurn)
        .expect("a [Bash, EndTurn] plan");
    let f = plan_features(&state, bash);
    assert!(f[8] > 0.0, "Vulnerable not recorded: {f:?}");
    assert!(f[3] > 0.0, "Bash removed no HP: {f:?}");
    assert!((f[10] - 1.0).abs() < 1e-6, "cards played: {f:?}");
    assert_eq!(&f[15..18], &[1.0, 0.0, 0.0], "end one-hot: {f:?}");
    assert_eq!(plan_cards(&state, bash), vec!["Bash".to_owned()]);

    let pass = set.plans.iter().find(|p| p.actions.len() == 1).unwrap();
    assert!(plan_features(&state, pass)[8].abs() < 1e-6);
    assert!(plan_cards(&state, pass).is_empty());
}

#[test]
fn end_turn_plan_features_exclude_enemy_turn() {
    let mut state = fixture(&[CardId::DefendIronclad], 40, &[], 3);
    state.creatures[0].intent = Some(sts2_core::Intent::Attack { damage: 100 });
    let set = plans(&state);
    let defend = set
        .plans
        .iter()
        .find(|p| p.actions.len() == 2)
        .expect("[Defend, EndTurn]");
    let f = plan_features(&state, defend);
    assert!(
        f[1].abs() < 1e-6,
        "HP lost must exclude the enemy turn: {f:?}"
    );
    assert!(f[0] > 0.0, "block gained: {f:?}");
    assert!(
        (f[14] - (100.0 - f[0])).abs() < 1e-6,
        "telegraphed minus block: {f:?}"
    );
}

#[test]
fn deck_context_counts_all_piles() {
    let mut state = fixture(
        &[CardId::StrikeIronclad, CardId::DefendIronclad],
        40,
        &[],
        3,
    );
    push_draw(
        &mut state,
        &[CardId::Bash, CardId::Anger, CardId::StrikeIronclad],
    );
    let discard = new_cards(&mut state, &[CardId::Inflame]);
    state.player.discard.extend(discard);
    let exhaust = new_cards(&mut state, &[CardId::PommelStrike]);
    state.player.exhaust.extend(exhaust);

    let ctx = deck_context(&state);
    assert_eq!(ctx.cards.len(), 7, "{:?}", ctx.cards);
    for name in ["Inflame", "PommelStrike", "Bash"] {
        assert!(ctx.cards.iter().any(|c| c == name), "{name} missing");
    }
    assert!((ctx.ctx[1] - 1.0).abs() < 1e-6, "full HP: {:?}", ctx.ctx);
    assert!((ctx.ctx[2] - 3.0).abs() < 1e-6, "energy: {:?}", ctx.ctx);
    assert!((ctx.ctx[5] - 1.0).abs() < 1e-6, "one enemy: {:?}", ctx.ctx);
}

#[test]
fn anchors_mark_exactly_one_max_block() {
    let state = fixture(
        &[
            CardId::StrikeIronclad,
            CardId::DefendIronclad,
            CardId::DefendIronclad,
        ],
        40,
        &[],
        3,
    );
    let set = plans(&state);
    let flags = anchors(&state, &set);
    assert_eq!(flags.len(), set.plans.len());
    let marked: Vec<usize> = (0..flags.len()).filter(|&i| flags[i].max_block).collect();
    assert_eq!(marked.len(), 1, "{flags:?}");
    let best = plan_features(&state, &set.plans[marked[0]])[0];
    for p in set.plans.iter().filter(|p| p.end == PlanEnd::EndTurn) {
        assert!(plan_features(&state, p)[0] <= best);
    }
    assert_eq!(
        flags.iter().filter(|f| f.max_damage).count(),
        1,
        "{flags:?}"
    );
}

#[test]
fn lethal_anchor_set_when_kill_exists() {
    let state = fixture(&[CardId::StrikeIronclad], 6, &[], 3);
    let set = plans(&state);
    let flags = anchors(&state, &set);
    let lethal: Vec<usize> = (0..flags.len()).filter(|&i| flags[i].lethal).collect();
    assert_eq!(lethal.len(), 1, "{flags:?}");
    assert_eq!(set.plans[lethal[0]].end, PlanEnd::Terminal);

    let state = fixture(&[CardId::StrikeIronclad], 40, &[], 3);
    let set = plans(&state);
    assert!(anchors(&state, &set).iter().all(|f| !f.lethal));
}

#[test]
fn lethal_anchor_covers_a_random_killing_plan() {
    let mut state = fixture(&[CardId::PommelStrike], 6, &[], 3);
    push_draw(
        &mut state,
        &[CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash],
    );
    let set = plans(&state);
    let flags = anchors(&state, &set);
    let pommel = set
        .plans
        .iter()
        .position(|p| matches!(p.actions.first(), Some(Action::PlayCard { .. })))
        .expect("a Pommel Strike plan");
    assert_eq!(set.plans[pommel].end, PlanEnd::Chance, "{set:?}");
    assert!(flags[pommel].lethal, "{flags:?}");
    assert_eq!(flags.iter().filter(|f| f.lethal).count(), 1, "{flags:?}");
}

#[test]
fn deck_context_ignores_draw_order() {
    let mut a = fixture(&[CardId::StrikeIronclad], 40, &[], 3);
    push_draw(
        &mut a,
        &[CardId::Bash, CardId::Anger, CardId::DefendIronclad],
    );
    let mut b = a.clone();
    b.player.draw.reverse();
    assert_ne!(a.player.draw, b.player.draw);
    assert_eq!(deck_context(&a).cards, deck_context(&b).cards);
}

/// The mirror net's greedy line, replayed on `state` until `EndTurn`;
/// returns the state it ends the turn from.
fn greedy_pre_end_turn(state: &CombatState) -> CombatState {
    let net = heuristic_mirror_net();
    let mut s = state.clone();
    loop {
        let legal = legal_actions(&s);
        let logits = net.logits(&s, &legal, END_TURN_PRIORITY);
        let mut best = 0;
        for (i, &l) in logits.iter().enumerate() {
            if l > logits[best] {
                best = i;
            }
        }
        if legal[best] == Action::EndTurn {
            return s;
        }
        let _ = apply(&mut s, legal[best].clone());
    }
}

#[test]
fn heuristic_anchor_matches_greedy_mirror() {
    let state = fixture(
        &[
            CardId::StrikeIronclad,
            CardId::DefendIronclad,
            CardId::Bash,
            CardId::StrikeIronclad,
        ],
        60,
        &[],
        4,
    );
    let set = plans(&state);
    let flags = anchors(&state, &set);
    let marked: Vec<usize> = (0..flags.len()).filter(|&i| flags[i].heuristic).collect();
    assert_eq!(marked.len(), 1, "{flags:?}");
    let plan = &set.plans[marked[0]];
    assert_eq!(plan.end, PlanEnd::EndTurn);
    assert_eq!(
        tt_hash_v2(&before_last(&state, plan)),
        tt_hash_v2(&greedy_pre_end_turn(&state)),
        "flagged plan {plan:?} does not end where the greedy line does"
    );
    // No built-in net for a character-less fixture: no rl2 anchor.
    assert!(flags.iter().all(|f| !f.rl2));
}

/// Needs the built-in Ironclad net (compiled in when its file exists).
#[cfg(rollout_net_ironclad)]
#[test]
fn rl2_anchor_marks_one_plan_for_ironclad() {
    if std::env::var_os("STS2_ROLLOUT_NET").is_some() {
        return;
    }
    let mut state = fixture(
        &[CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash],
        60,
        &[],
        3,
    );
    state.player.character = Some(sts2_core::CharacterId::Ironclad);
    let set = plans(&state);
    let flags = anchors(&state, &set);
    assert_eq!(flags.iter().filter(|f| f.rl2).count(), 1, "{flags:?}");
}

#[test]
fn logprob_of_greedy_plan_is_highest_under_its_net() {
    let state = fixture(
        &[
            CardId::StrikeIronclad,
            CardId::DefendIronclad,
            CardId::Bash,
            CardId::StrikeIronclad,
        ],
        60,
        &[],
        4,
    );
    let set = plans(&state);
    let flags = anchors(&state, &set);
    let net = heuristic_mirror_net();
    let greedy = flags.iter().position(|f| f.heuristic).expect("greedy plan");
    let lp_greedy = plan_logprob(&net, &state, &set.plans[greedy]);
    assert!(lp_greedy.is_finite() && lp_greedy <= 0.0, "{lp_greedy}");
    // Ties are real (two orders of equal-priority cards), so the greedy
    // plan is *a* maximum; most plans (an early EndTurn) fall far below it.
    let mut below = 0;
    for p in &set.plans {
        let lp = plan_logprob(&net, &state, p);
        assert!(
            lp <= lp_greedy + 1e-5,
            "plan {p:?}: {lp} > greedy {lp_greedy}"
        );
        below += usize::from(lp < lp_greedy - 1.0);
    }
    assert!(
        below * 2 > set.plans.len(),
        "{below} of {}",
        set.plans.len()
    );
}
