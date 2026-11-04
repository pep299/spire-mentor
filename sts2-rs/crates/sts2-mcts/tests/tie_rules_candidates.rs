//! The phase-4 candidate rules of GH #1022, all off by default: `lasting`
//! (the free-play guard for plays whose effect outlasts the turn), `order`
//! (order dominance at the root) and `keep_potion`. Each test rigs root
//! statistics so the move under test sits inside the `tie_eps` band behind
//! a band winner, then reads `pick_root` with and without the rule.
//!
//! The last test is #963's negative fixture: with every rule on, Rocket
//! Punch first (the real −1.5 HP mistake measured there) is never promoted.

mod common;

use common::build_handful_ex;
use sts2_core::cards::{CardId, TargetType};
use sts2_core::{legal_actions, Action, CombatState, Intent};
use sts2_mcts::{pick_root, MctsConfig, PickRule, RootOutcome, TieRules};

fn outcome(action: Action, visits: u32, q: f32) -> RootOutcome {
    RootOutcome {
        action,
        visits,
        wins: 0,
        losses: 0,
        ties: 0,
        #[allow(clippy::cast_precision_loss)]
        value_sum: q * visits as f32,
        value_sq_sum: 0.0,
        hp_sum: 0.0,
        hp_sq_sum: 0.0,
        hp_n: 0,
        mean_value: q,
    }
}

fn rules(names: &str) -> MctsConfig {
    MctsConfig {
        tie_rules: TieRules::parse(names).expect("known rules"),
        ..MctsConfig::default()
    }
}

/// `(action, visits, q)` → the rule's pick and its deciding rule.
fn pick(
    state: &CombatState,
    config: &MctsConfig,
    stats: &[(Action, u32, f32)],
) -> (Action, PickRule) {
    let children: Vec<RootOutcome> = stats
        .iter()
        .map(|(a, v, q)| outcome(a.clone(), *v, *q))
        .collect();
    let p = pick_root(state, config, &children).expect("non-empty");
    (p.action, p.rule)
}

fn play(hand_index: u8, target: Option<u8>) -> Action {
    Action::PlayCard { hand_index, target }
}

fn single_enemy(hand: &[(CardId, i32, i32, i8, TargetType)], energy: i8) -> CombatState {
    build_handful_ex(
        0,
        hand,
        /*cultist_hp=*/ 40,
        /*player_hp=*/ 60,
        Intent::Attack { damage: 5 },
        &[],
        energy,
    )
}

#[test]
fn lasting_skips_a_free_play_that_costs_hp() {
    // Bloodletting (0 energy, 3 HP) and Anger (0 energy, 6 damage), block
    // standing above the incoming 5 so the block guard stays out of it.
    let mut s = single_enemy(
        &[
            (CardId::Bloodletting, 0, 0, 0, TargetType::Self_),
            (CardId::Anger, 6, 0, 0, TargetType::Enemy),
        ],
        1,
    );
    s.player.block = 10;
    let legal = legal_actions(&s);
    let bloodletting = play(0, None);
    let anger = play(1, Some(0));
    assert!(
        legal.contains(&bloodletting) && legal.contains(&anger),
        "{legal:?}"
    );

    // EndTurn wins the band; only Bloodletting is in it.
    let stats = [
        (Action::EndTurn, 500, 0.500),
        (bloodletting.clone(), 100, 0.495),
        (anger.clone(), 50, 0.300),
    ];
    assert_eq!(
        pick(&s, &rules("block+free+junk"), &stats),
        (bloodletting, PickRule::FreePlay),
        "the deployed guard plays the free card"
    );
    assert_eq!(
        pick(&s, &rules("block+lasting+junk"), &stats),
        (Action::EndTurn, PickRule::BandVisits),
        "Bloodletting leaves nothing past the turn and costs HP"
    );

    // A free attack in the band is played under either guard.
    let stats = [
        (Action::EndTurn, 500, 0.500),
        (play(0, None), 50, 0.300),
        (anger.clone(), 100, 0.495),
    ];
    assert_eq!(pick(&s, &rules("block+free+junk"), &stats).0, anger);
    assert_eq!(
        pick(&s, &rules("block+lasting+junk"), &stats),
        (anger, PickRule::FreeLasting)
    );
}

#[test]
fn order_plays_the_proven_first_card_first() {
    // GH #846's pair: Bash (8 + 2 Vulnerable) before Strike deals 17,
    // Strike first 14; neither kills a 40-HP enemy. The enemy is re-keyed
    // off the Nibbit, whose ally-dependent AI disables the proof.
    let mut s = single_enemy(
        &[
            (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
            (CardId::Bash, 8, 0, 2, TargetType::Enemy),
        ],
        3,
    );
    let e = s.enemy_uids[0].0;
    let c = s.creatures.iter_mut().find(|c| c.id == e).expect("enemy");
    c.monster_id = sts2_core::MonsterId("FuzzyWurmCrawler");
    c.species = Some(sts2_core::MonsterKind::FuzzyWurmCrawler);
    let (strike, bash) = (play(0, Some(0)), play(1, Some(0)));
    let in_band = [
        (strike.clone(), 500, 0.500),
        (bash.clone(), 100, 0.495),
        (Action::EndTurn, 50, 0.300),
    ];
    assert_eq!(
        pick(&s, &rules("block+free+junk"), &in_band),
        (strike.clone(), PickRule::BandVisits)
    );
    assert_eq!(
        pick(&s, &rules("block+free+junk+order"), &in_band),
        (bash, PickRule::Order)
    );

    // Out of the band the proof does not matter: a resolved Q gap stands.
    let resolved = [
        (strike.clone(), 500, 0.500),
        (play(1, Some(0)), 100, 0.400),
        (Action::EndTurn, 50, 0.300),
    ];
    assert_eq!(
        pick(&s, &rules("block+free+junk+order"), &resolved).0,
        strike
    );
}

#[test]
fn keep_potion_keeps_a_tied_drink() {
    let s = single_enemy(&[(CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy)], 3);
    let drink = Action::DrinkPotion {
        potion_slot: 0,
        target: None,
    };
    let strike = play(0, Some(0));
    let stats = [
        (drink.clone(), 500, 0.500),
        (strike.clone(), 100, 0.495),
        (Action::EndTurn, 200, 0.490),
    ];
    assert_eq!(pick(&s, &rules("block+free+junk"), &stats).0, drink);
    assert_eq!(
        pick(&s, &rules("block+free+junk+keep_potion"), &stats),
        (Action::EndTurn, PickRule::KeepPotion),
        "the most-visited in-band move that keeps the potion"
    );
    // A drink the band does not tie stands.
    let resolved = [
        (drink.clone(), 500, 0.500),
        (strike, 100, 0.400),
        (Action::EndTurn, 200, 0.300),
    ];
    assert_eq!(
        pick(&s, &rules("block+free+junk+keep_potion"), &resolved).0,
        drink
    );
}

/// #963: Defect holding 2× Boost Away, Rocket Punch, Strike and Defend.
/// Rocket Punch first is the measured mistake (−1.52 ± 0.15 HP), so no rule
/// may promote it over a band-winning Boost Away, whatever else is on.
#[test]
fn no_rule_promotes_rocket_punch_first() {
    let s = single_enemy(
        &[
            (CardId::BoostAway, 0, 6, 0, TargetType::Self_),
            (CardId::BoostAway, 0, 6, 0, TargetType::Self_),
            (CardId::RocketPunch, 13, 0, 2, TargetType::Enemy),
            (CardId::StrikeDefect, 6, 0, 1, TargetType::Enemy),
            (CardId::DefendDefect, 0, 5, 1, TargetType::Self_),
        ],
        3,
    );
    let legal = legal_actions(&s);
    let boost = play(0, None);
    let rocket = play(2, Some(0));
    assert!(
        legal.contains(&boost) && legal.contains(&rocket),
        "{legal:?}"
    );
    let everything = rules("block+free+junk+lasting+spend+order+order_block+upgraded+keep_potion");
    let stats: Vec<(Action, u32, f32)> = legal
        .iter()
        .map(|a| {
            let visits = if *a == boost { 500 } else { 100 };
            (a.clone(), visits, 0.5)
        })
        .collect();
    let (picked, _) = pick(&s, &everything, &stats);
    assert_ne!(picked, rocket, "Rocket Punch first promoted");
    assert_eq!(picked, boost);
}

fn give_relic(s: &mut CombatState, kind: sts2_core::RelicKind) {
    s.relics.push(sts2_core::Relic {
        id: std::num::NonZeroU32::new(77).expect("non-zero"),
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

/// "End turn with 1 energy left and a Strike in hand": the energy is lost
/// at the end of the turn, so `spend` plays the Strike where the free-play
/// guards (0-cost cards only) do not — unless the card would exhaust out of
/// the fight, or Ice Cream keeps the energy.
#[test]
fn spend_uses_the_energy_a_turn_would_lose() {
    let mut s = single_enemy(
        &[
            (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
            (CardId::Impervious, 0, 30, 1, TargetType::Self_),
        ],
        1,
    );
    s.player.block = 10;
    let (strike, impervious) = (play(0, Some(0)), play(1, None));
    let stats = [
        (Action::EndTurn, 500, 0.500),
        (strike.clone(), 100, 0.495),
        (impervious.clone(), 200, 0.495),
    ];
    assert_eq!(
        pick(&s, &rules("block+free+junk"), &stats),
        (Action::EndTurn, PickRule::BandVisits)
    );
    // Impervious has more visits, but it exhausts; the Strike goes to the
    // discard pile either way.
    assert_eq!(
        pick(&s, &rules("block+free+spend+junk"), &stats),
        (strike, PickRule::Spend)
    );
    give_relic(&mut s, sts2_core::RelicKind::IceCream);
    assert_eq!(
        pick(&s, &rules("block+free+spend+junk"), &stats).0,
        Action::EndTurn,
        "Ice Cream keeps the energy for next turn"
    );
}

/// "Rage first is a no-brainer": Rage then Strike gains block, Strike then
/// Rage does not, and nothing else differs. `order` requires equal block
/// and stays out; `order_block` counts the block — unless Orichalcum makes
/// a little more block worse.
#[test]
fn order_block_plays_rage_before_the_attack() {
    let mut s = single_enemy(
        &[
            (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
            (CardId::RageCard, 0, 0, 0, TargetType::Self_),
        ],
        3,
    );
    let e = s.enemy_uids[0].0;
    let c = s.creatures.iter_mut().find(|c| c.id == e).expect("enemy");
    c.monster_id = sts2_core::MonsterId("FuzzyWurmCrawler");
    c.species = Some(sts2_core::MonsterKind::FuzzyWurmCrawler);
    let (strike, rage) = (play(0, Some(0)), play(1, None));
    let stats = [
        (strike.clone(), 500, 0.500),
        (rage.clone(), 100, 0.495),
        (Action::EndTurn, 50, 0.300),
    ];
    assert_eq!(
        pick(&s, &rules("block+free+junk+order"), &stats),
        (strike.clone(), PickRule::BandVisits)
    );
    assert_eq!(
        pick(&s, &rules("block+free+junk+order_block"), &stats),
        (rage, PickRule::OrderBlock)
    );
    give_relic(&mut s, sts2_core::RelicKind::Orichalcum);
    assert_eq!(
        pick(&s, &rules("block+free+junk+order_block"), &stats).0,
        strike
    );
}

/// "Shrug It Off ranked above Shrug It Off+": the upgraded copy at the same
/// cost takes the pick from its base copy, and nothing else does.
#[test]
fn upgraded_plays_the_better_copy() {
    let mut s = single_enemy(
        &[
            (CardId::ShrugItOff, 0, 8, 1, TargetType::Self_),
            (CardId::ShrugItOff, 0, 11, 1, TargetType::Self_),
            (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
        ],
        3,
    );
    let plus = s.player.hand[1];
    let card = s.cards.iter_mut().find(|c| c.uid == plus).expect("card");
    card.upgraded = true;
    let (shrug, shrug_plus, strike) = (play(0, None), play(1, None), play(2, Some(0)));
    let stats = [
        (shrug.clone(), 500, 0.500),
        (shrug_plus.clone(), 100, 0.495),
        (strike.clone(), 200, 0.495),
    ];
    assert_eq!(pick(&s, &rules("block+free+junk"), &stats).0, shrug);
    assert_eq!(
        pick(&s, &rules("block+free+junk+upgraded"), &stats),
        (shrug_plus, PickRule::Upgraded)
    );
    // A different card in the band is no upgrade of the pick.
    let stats = [
        (strike.clone(), 500, 0.500),
        (shrug, 100, 0.495),
        (play(1, None), 100, 0.495),
    ];
    assert_eq!(
        pick(&s, &rules("block+free+junk+upgraded"), &stats).0,
        strike
    );
}

/// "Offering+ and Feel No Pain+ in hand, but it plays Offering": Feel No
/// Pain first turns Offering's exhaust into block.
#[test]
fn order_block_plays_feel_no_pain_before_offering() {
    let mut s = single_enemy(
        &[
            (CardId::Offering, 0, 0, 0, TargetType::Self_),
            (CardId::FeelNoPainCard, 0, 0, 1, TargetType::Self_),
        ],
        3,
    );
    let e = s.enemy_uids[0].0;
    let c = s.creatures.iter_mut().find(|c| c.id == e).expect("enemy");
    c.monster_id = sts2_core::MonsterId("FuzzyWurmCrawler");
    c.species = Some(sts2_core::MonsterKind::FuzzyWurmCrawler);
    // `build_handful` types every self-target card a Skill; Feel No Pain is a
    // Power, which leaves play instead of being drawn back by Offering.
    let fnp_uid = s.player.hand[1];
    let card = s.cards.iter_mut().find(|c| c.uid == fnp_uid).expect("card");
    card.card_type = sts2_core::cards::CardType::Power;
    let (offering, fnp) = (play(0, None), play(1, None));
    let stats = [
        (offering.clone(), 500, 0.500),
        (fnp.clone(), 100, 0.495),
        (Action::EndTurn, 50, 0.300),
    ];
    assert_eq!(
        pick(&s, &rules("block+free+junk+order"), &stats).0,
        offering
    );
    assert_eq!(
        pick(&s, &rules("block+free+junk+order_block"), &stats),
        (fnp, PickRule::OrderBlock)
    );
}
