#![allow(dead_code)]
//! Shared helpers for the boss-fidelity tests (#830 / #832): a real encounter
//! built with the C#-oracle scenario builder and driven through `apply`. Only
//! a boss's HP is set by hand, to reach a kill quickly.

use sts2_core::{apply, Action, CombatState, Creature, CreatureUid, PowerKind, StepResult};
use sts2_wire::scenario::build_scenario_state;

pub fn names(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|s| (*s).to_string()).collect()
}

/// A fresh fight against `encounter` with the scenario builder (player turn 1,
/// opening hand drawn).
pub fn fight(
    character: &str,
    deck: &[&str],
    hp: i32,
    relics: &[&str],
    potions: &[&str],
    encounter: &str,
    ascension: u8,
) -> CombatState {
    build_scenario_state(
        7,
        character,
        &names(deck),
        hp,
        hp,
        &names(relics),
        &names(potions),
        encounter,
        ascension,
        None,
        None,
    )
    .expect("scenario builds")
}

pub fn boss_uid(state: &CombatState) -> CreatureUid {
    state.enemy_uids[0]
}

pub fn boss(state: &CombatState) -> &Creature {
    let uid = boss_uid(state);
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("boss present")
}

pub fn boss_mut(state: &mut CombatState) -> &mut Creature {
    let uid = boss_uid(state);
    state
        .creatures
        .iter_mut()
        .find(|c| c.id == uid.0)
        .expect("boss present")
}

pub fn player(state: &CombatState) -> &Creature {
    let uid = state.player.creature_uid.expect("player mirror creature");
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("player creature present")
}

pub fn power_kinds(state: &CombatState, c: &Creature) -> Vec<PowerKind> {
    c.powers
        .iter()
        .filter_map(|pu| state.powers.get_by_id(pu.0))
        .map(|p| p.kind)
        .collect()
}

pub fn power_amount(state: &CombatState, c: &Creature, kind: PowerKind) -> i32 {
    c.powers
        .iter()
        .filter_map(|pu| state.powers.get_by_id(pu.0))
        .filter(|p| p.kind == kind)
        .map(|p| p.amount)
        .sum()
}

pub fn move_id(c: &Creature) -> &'static str {
    c.ai.as_ref()
        .map(sts2_core::MonsterAi::current_state_id)
        .expect("boss has an AI")
}

pub fn end_turn(state: &mut CombatState) -> StepResult {
    apply(state, Action::EndTurn)
}

/// Play the first card in hand whose id is `card`, at the boss.
pub fn play(state: &mut CombatState, card: sts2_core::cards::CardId) -> StepResult {
    let idx = state
        .player
        .hand
        .iter()
        .position(|u| state.cards[u.0.get() as usize - 1].card_id == card)
        .unwrap_or_else(|| panic!("{card:?} in hand"));
    apply(
        state,
        Action::PlayCard {
            hand_index: u8::try_from(idx).expect("hand index fits"),
            target: Some(0),
        },
    )
}

/// The giant after one enemy turn: PRESSURIZE has put SteamEruption 20 on it
/// (A9+, `WaterfallGiant.cs:74`), and it is now player turn 2.
pub fn pressurized_giant(deck: &[&str], relics: &[&str], potions: &[&str]) -> CombatState {
    let mut state = fight(
        "Ironclad",
        deck,
        90,
        relics,
        potions,
        "WATERFALL_GIANT_BOSS",
        10,
    );
    assert_eq!(move_id(boss(&state)), "PRESSURIZE");
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    let giant = boss(&state);
    assert_eq!(power_amount(&state, giant, PowerKind::SteamEruption), 20);
    assert_eq!(move_id(giant), "STOMP");
    state
}
