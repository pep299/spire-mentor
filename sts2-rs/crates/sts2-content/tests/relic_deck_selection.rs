//! GH #28: obtaining Claws / Tri-Boomerang / Pael's Growth must yield
//! actual deck decisions in the agent mode, not silently pick deck front.

use sts2_core::cards::{Card, CardId, CardType, EnchantmentId};
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{ActionResult, RunAction, RunManager};
use sts2_core::{GameVersion, RelicId};

fn run(ids: &[CardId]) -> RunManager {
    sts2_content::relics::dispatch::ensure_installed();
    let mut rm = RunManager::new(28028, CharacterId::Ironclad)
        .with_content_callbacks(sts2_content::run_callbacks::ironclad_content_callbacks());
    rm.run_state.game_version = GameVersion::V0_111_0;
    rm.run_state.interactive_card_transform = true;
    rm.run_state.master_deck = ids
        .iter()
        .map(|&card_id| Card {
            card_id,
            card_type: if card_id == CardId::DefendIronclad {
                CardType::Skill
            } else {
                CardType::Attack
            },
            ..Card::pod_defaults()
        })
        .collect();
    rm
}

fn take(rm: &mut RunManager, idx: usize) {
    assert_eq!(
        rm.take_action(RunAction::TransformDeckCard { idx }),
        ActionResult::Advanced
    );
}

#[test]
fn claws_zero_cards_is_a_real_choice_and_does_not_change_the_deck() {
    let mut rm = run(&[CardId::StrikeIronclad, CardId::Bash]);
    let original = rm.run_state.master_deck.clone();
    rm.run_state.obtain_relic(RelicId::Claws);
    assert_eq!(rm.run_state.master_deck, original);
    assert_eq!(rm.get_available_actions().len(), 3);
    take(&mut rm, 2); // confirm 0 of 2
    assert_eq!(rm.run_state.master_deck, original);
    assert!(rm.run_state.pending_claws_indices.is_none());
}

#[test]
fn claws_one_target_is_not_transformed_until_confirmed() {
    let mut rm = run(&[
        CardId::StrikeIronclad,
        CardId::StrikeIronclad,
        CardId::StrikeIronclad,
        CardId::Bash,
    ]);
    rm.run_state.obtain_relic(RelicId::Claws);
    take(&mut rm, 3); // select Bash, not the first Strike
    assert_eq!(rm.run_state.master_deck[3].card_id, CardId::Bash);
    assert_eq!(rm.get_available_actions().len(), 4); // 3 remaining + confirm
    take(&mut rm, 3);
    let ids: Vec<CardId> = rm.run_state.master_deck.iter().map(|c| c.card_id).collect();
    assert_eq!(
        ids,
        [
            CardId::StrikeIronclad,
            CardId::StrikeIronclad,
            CardId::StrikeIronclad,
            CardId::Maul,
        ]
    );
    assert!(rm.run_state.pending_claws_indices.is_none());
}

#[test]
fn claws_six_targets_and_saved_actions_replay_identically() {
    let actions = [0, 0, 0, 0, 0, 0, 0]; // six cards, then confirm at idx 0
    let make = || {
        let mut rm = run(&[CardId::StrikeIronclad; 8]);
        rm.run_state.obtain_relic(RelicId::Claws);
        rm
    };
    let mut first = make();
    let mut second = make();
    for (step, idx) in actions.into_iter().enumerate() {
        assert_eq!(
            first.get_available_actions(),
            second.get_available_actions(),
            "step {step}"
        );
        take(&mut first, idx);
        take(&mut second, idx);
        assert_eq!(first.run_state, second.run_state, "step {step}");
    }
    assert_eq!(
        first
            .run_state
            .master_deck
            .iter()
            .filter(|c| c.card_id == CardId::Maul)
            .count(),
        6
    );
    assert_eq!(
        first.run_state.master_deck[0].card_id,
        CardId::StrikeIronclad
    );
    assert_eq!(
        first.run_state.master_deck[1].card_id,
        CardId::StrikeIronclad
    );
}

#[test]
fn tri_boomerang_offers_three_attack_picks_not_defends() {
    let mut rm = run(&[
        CardId::StrikeIronclad,
        CardId::DefendIronclad,
        CardId::Bash,
        CardId::Anger,
        CardId::StrikeIronclad,
        CardId::StrikeIronclad,
    ]);
    rm.run_state.obtain_relic(RelicId::TriBoomerang);
    assert_eq!(rm.run_state.pending_deck_enchants.len(), 3);
    assert_eq!(rm.enchant_eligible_deck_indices(), vec![0, 2, 3, 4, 5]);
    assert!(rm
        .run_state
        .master_deck
        .iter()
        .all(|c| c.enchantment(EnchantmentId::Instinct) == 0));
    for idx in [4, 0, 0] {
        assert_eq!(
            rm.take_action(RunAction::EnchantDeckCard { idx }),
            ActionResult::Advanced
        );
    }
    assert_eq!(rm.run_state.pending_deck_enchants.len(), 0);
    assert_eq!(
        rm.run_state
            .master_deck
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Instinct) == 1)
            .count(),
        3
    );
    assert_eq!(
        rm.run_state.master_deck[1].enchantment(EnchantmentId::Instinct),
        0
    );
}

#[test]
fn tri_boomerang_auto_resolves_when_only_two_attacks_exist() {
    let mut rm = run(&[CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash]);
    rm.run_state.obtain_relic(RelicId::TriBoomerang);
    assert!(rm.run_state.pending_deck_enchants.is_empty());
    assert_eq!(
        rm.run_state
            .master_deck
            .iter()
            .filter(|c| c.enchantment(EnchantmentId::Instinct) == 1)
            .count(),
        2
    );
}

#[test]
fn paels_growth_enchants_the_selected_card_with_clone() {
    let mut rm = run(&[CardId::StrikeIronclad, CardId::Bash, CardId::DefendIronclad]);
    rm.run_state.obtain_relic(RelicId::PaelsGrowth);
    assert_eq!(rm.run_state.pending_deck_enchants.len(), 1);
    assert_eq!(
        rm.take_action(RunAction::EnchantDeckCard { idx: 2 }),
        ActionResult::Advanced
    );
    assert_eq!(
        rm.run_state.master_deck[2].enchantment(EnchantmentId::Clone),
        4
    );
    assert_eq!(
        rm.run_state.master_deck[0].enchantment(EnchantmentId::Clone),
        0
    );
    assert_eq!(
        rm.run_state.master_deck[1].enchantment(EnchantmentId::Clone),
        0
    );
}
