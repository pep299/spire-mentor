//! Final-player-HP bookkeeping on the MCTS (coach plan 1, task 1).
//!
//! `RootOutcome::{hp_sum, hp_n, mean_final_hp}` record the player's HP at the
//! end of each decided rollout. They are bookkeeping only: no selection or
//! backup rule reads them, so the search itself must be unchanged.

mod common;

use common::build_handful_ex;
use sts2_core::cards::{CardId, TargetType};
use sts2_core::{Action, Intent};
use sts2_mcts::{Mcts, MctsConfig, RootOutcome};

const PLAYER_HP: i32 = 30;

/// 12-HP enemy that will hit for 15; hand = Defend + Strike, 2 energy.
fn searched_outcomes() -> Vec<RootOutcome> {
    let hand = [
        (CardId::DefendIronclad, 0, 5, 1, TargetType::Self_),
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
    ];
    let state = build_handful_ex(
        0,
        &hand,
        /*cultist_hp=*/ 12,
        PLAYER_HP,
        Intent::Attack { damage: 15 },
        &[],
        /*energy=*/ 2,
    );
    let cfg = MctsConfig {
        max_simulations: 1500,
        seed: 7,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::new(state, cfg);
    let _ = mcts.search();
    mcts.root_outcomes()
}

fn find(outs: &[RootOutcome], pred: impl Fn(&Action) -> bool) -> &RootOutcome {
    outs.iter()
        .find(|o| pred(&o.action))
        .expect("action at root")
}

#[test]
fn final_hp_prefers_blocking_over_ending_turn() {
    let outs = searched_outcomes();
    let defend = find(&outs, |a| {
        matches!(
            a,
            Action::PlayCard {
                hand_index: 0,
                target: None
            }
        )
    });
    let end = find(&outs, |a| matches!(a, Action::EndTurn));
    let d = defend.mean_final_hp().expect("defend has HP samples");
    let e = end.mean_final_hp().expect("end turn has HP samples");
    assert!(d > e, "defend mean HP {d} should beat end-turn mean HP {e}");
}

#[test]
fn final_hp_counts_are_bounded() {
    let outs = searched_outcomes();
    for o in &outs {
        assert!(
            o.hp_n <= o.visits,
            "{:?}: hp_n {} > visits {}",
            o.action,
            o.hp_n,
            o.visits
        );
        #[allow(clippy::cast_precision_loss)]
        let cap = o.hp_n as f32 * PLAYER_HP as f32;
        assert!(
            o.hp_sum >= 0.0 && o.hp_sum <= cap,
            "{:?}: hp_sum {} out of [0,{cap}]",
            o.action,
            o.hp_sum
        );
    }
    let end = find(&outs, |a| matches!(a, Action::EndTurn));
    assert_eq!(
        RootOutcome {
            hp_n: 0,
            hp_sum: 0.0,
            hp_sq_sum: 0.0,
            ..end.clone()
        }
        .mean_final_hp(),
        None
    );
}

/// `(action debug, visits, wins, losses, ties, value_sum)` recorded on the
/// tree BEFORE the HP bookkeeping existed. The search must not move.
const SNAPSHOT: [(&str, u32, u32, u32, u32, f32); 3] = [
    (
        "PlayCard { hand_index: 1, target: Some(0) }",
        556,
        553,
        3,
        0,
        578.06055,
    ),
    (
        "PlayCard { hand_index: 0, target: None }",
        606,
        604,
        2,
        0,
        634.15186,
    ),
    ("EndTurn", 338, 332, 6, 0, 335.919),
];

#[test]
fn final_hp_does_not_change_the_search() {
    let got: Vec<_> = searched_outcomes()
        .iter()
        .map(|o| {
            (
                format!("{:?}", o.action),
                o.visits,
                o.wins,
                o.losses,
                o.ties,
                o.value_sum,
            )
        })
        .collect();
    let want: Vec<_> = SNAPSHOT
        .iter()
        .map(|&(a, v, w, l, t, s)| (a.to_string(), v, w, l, t, s))
        .collect();
    assert_eq!(got, want);
}

/// `search_forced` spends the budget on the one root action it is given and
/// reports that child's outcome (with HP samples), leaving siblings with at
/// most the warm-up visits.
#[test]
fn search_forced_spends_the_budget_on_one_action() {
    let hand = [
        (CardId::DefendIronclad, 0, 5, 1, TargetType::Self_),
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
    ];
    let state = build_handful_ex(
        0,
        &hand,
        /*cultist_hp=*/ 12,
        PLAYER_HP,
        Intent::Attack { damage: 15 },
        &[],
        /*energy=*/ 2,
    );
    let cfg = MctsConfig {
        max_simulations: 200,
        seed: 7,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::new(state.clone(), cfg);
    let out = mcts
        .search_forced(&Action::EndTurn, 200)
        .expect("end turn is a root action");
    assert_eq!(out.action, Action::EndTurn);
    assert!(out.visits >= 198, "{out:?}");
    assert!(out.mean_final_hp().is_some());
    let others: u32 = mcts
        .root_outcomes()
        .iter()
        .filter(|o| o.action != Action::EndTurn)
        .map(|o| o.visits)
        .sum();
    assert!(others <= 2, "warm-up only: {others}");
    // Deterministic for a seed.
    let again = Mcts::new(state.clone(), cfg)
        .search_forced(&Action::EndTurn, 200)
        .unwrap();
    assert_eq!(
        (again.visits, again.wins, again.hp_n),
        (out.visits, out.wins, out.hp_n)
    );
    // Not a root action → None.
    let bogus = Action::PlayCard {
        hand_index: 9,
        target: None,
    };
    assert!(Mcts::new(state, cfg).search_forced(&bogus, 50).is_none());
}
