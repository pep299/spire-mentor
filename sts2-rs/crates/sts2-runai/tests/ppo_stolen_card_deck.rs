//! The PPO observation's deck while a Thieving Hopper holds a card that was
//! upgraded only in this fight: the stolen card is out of it, found by its
//! deck link (C# `DeckVersion`), not by its combat state. Ported from
//! manojnuthakki/spire-mentor 2c16a3498.

use sts2_content::run_callbacks::{
    initialize_run_setup, new_run_manager_for_character_with_ascension,
};
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{Phase, RunAction};

#[test]
fn a_held_card_upgraded_in_the_fight_is_out_of_the_ppo_observed_deck() {
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::cards::dispatch::ensure_installed();
    let mut rm = new_run_manager_for_character_with_ascension(11, CharacterId::Ironclad, 0);
    initialize_run_setup(&mut rm);
    rm.run_state
        .master_deck
        .push(sts2_content::cards::spec_index::card_pod_from_id(CardId::Uppercut, false).unwrap());
    for _ in 0..200 {
        if rm.phase() == Phase::Combat {
            break;
        }
        let a: RunAction = rm.get_available_actions()[0];
        rm.take_action(a);
    }
    assert_eq!(rm.phase(), Phase::Combat);
    let deck_size = rm.run_state.master_deck.len();
    let combat = rm.combat.as_mut().unwrap();
    let uid = combat
        .player
        .draw
        .iter()
        .chain(combat.player.hand.iter())
        .chain(combat.player.discard.iter())
        .copied()
        .find(|u| combat.cards[u.get() as usize - 1].card_id == CardId::Uppercut)
        .expect("Uppercut in the fight");
    assert!(sts2_core::card_effects::upgrade_card_uid(combat, uid));
    let _ = sts2_core::card_effects::remove_card_from_any_pile(combat, uid);
    let thief = combat.enemy_uids[0];
    combat.record_swipe_steal_of(thief, uid);

    let obs = sts2_runai::ppo::policy::build_obs(&rm, "combat", None);
    assert_eq!(obs.deck.len(), deck_size - 1);
    assert!(obs.deck.iter().all(|(id, _)| id != "Uppercut"));
}
