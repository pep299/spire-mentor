//! GH #1119: a combat `PlayCard` aimed at a dead or absent enemy must come
//! back as `ActionResult::InvalidTarget`, not `Advanced`. The combat layer
//! already rejected it (`StepResult::InvalidTarget`, state untouched), but
//! `RunManager` dropped that result, so a workbench driver aiming at a dead
//! enemy livelocked with no error. GH #1122 does the same for the other
//! refusals (too little energy, a bad hand index): `IllegalPlay`.

use sts2_content::run_callbacks::{
    initialize_run_setup, new_run_manager_for_character_with_ascension,
};
use sts2_core::cards::TargetType;
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{ActionResult, Phase, RunAction, RunManager};

fn rm_in_first_combat() -> RunManager {
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::cards::dispatch::ensure_installed();
    let mut rm = new_run_manager_for_character_with_ascension(11, CharacterId::Ironclad, 0);
    initialize_run_setup(&mut rm);
    for _ in 0..200 {
        if rm.phase() == Phase::Combat {
            break;
        }
        let a: RunAction = rm.get_available_actions()[0];
        rm.take_action(a);
    }
    assert_eq!(rm.phase(), Phase::Combat);
    rm
}

/// Hand index of a single-enemy-targeted card (a starter Strike/Bash).
fn enemy_card_index(rm: &RunManager) -> usize {
    let combat = rm.combat.as_ref().unwrap();
    combat
        .player
        .hand
        .iter()
        .position(|u| combat.cards[u.0.get() as usize - 1].target_type == TargetType::Enemy)
        .expect("a starter hand holds an enemy-targeted card")
}

fn hand_and_energy(rm: &RunManager) -> (usize, i32) {
    let c = rm.combat.as_ref().unwrap();
    (c.player.hand.len(), c.player.energy)
}

#[test]
fn play_card_at_an_absent_enemy_slot_is_rejected() {
    let mut rm = rm_in_first_combat();
    let idx = enemy_card_index(&rm);
    let before = hand_and_energy(&rm);
    let r = rm.take_action(RunAction::PlayCard {
        hand_idx: idx,
        target: Some(9),
    });
    assert_eq!(r, ActionResult::InvalidTarget);
    assert_eq!(
        hand_and_energy(&rm),
        before,
        "a rejected play mutates nothing"
    );
}

#[test]
fn play_card_at_a_dead_enemy_is_rejected() {
    let mut rm = rm_in_first_combat();
    let idx = enemy_card_index(&rm);
    {
        let c = rm.combat.as_mut().unwrap();
        let first = c.enemy_uids[0];
        c.creatures
            .iter_mut()
            .find(|cr| cr.id == first.0)
            .unwrap()
            .current_hp = 0;
    }
    let before = hand_and_energy(&rm);
    let r = rm.take_action(RunAction::PlayCard {
        hand_idx: idx,
        target: Some(0),
    });
    assert_eq!(r, ActionResult::InvalidTarget);
    assert_eq!(hand_and_energy(&rm), before);
}

#[test]
fn play_card_at_a_live_enemy_still_advances() {
    let mut rm = rm_in_first_combat();
    let idx = enemy_card_index(&rm);
    let r = rm.take_action(RunAction::PlayCard {
        hand_idx: idx,
        target: Some(0),
    });
    assert_eq!(r, ActionResult::Advanced);
}

#[test]
fn play_card_without_the_energy_is_rejected() {
    let mut rm = rm_in_first_combat();
    let idx = enemy_card_index(&rm);
    rm.combat.as_mut().unwrap().player.energy = 0;
    let before = hand_and_energy(&rm);
    let r = rm.take_action(RunAction::PlayCard {
        hand_idx: idx,
        target: Some(0),
    });
    assert_eq!(r, ActionResult::IllegalPlay);
    assert_eq!(hand_and_energy(&rm), before);
}

#[test]
fn play_card_past_the_end_of_the_hand_is_rejected() {
    let mut rm = rm_in_first_combat();
    let before = hand_and_energy(&rm);
    let r = rm.take_action(RunAction::PlayCard {
        hand_idx: before.0 + 3,
        target: Some(0),
    });
    assert_eq!(r, ActionResult::IllegalPlay);
    assert_eq!(hand_and_energy(&rm), before);
}
