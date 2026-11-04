//! Relics that read or gate a card play, driven through the live play path
//! (`play_card` + `legal_actions`) with the real content registry installed.
//!
//! GH #742, the Act-3 ancient relics VelvetChoker and SpikedGauntlets. C#
//! (v0.107.1):
//! - `VelvetChoker.ShouldPlay` vetoes a play once `_cardsPlayedThisTurn >= 6`,
//!   and `VelvetChoker.ModifyMaxEnergy` adds `EnergyVar(1)`.
//! - `SpikedGauntlets.TryModifyEnergyCostInCombat` raises the owner's Power
//!   cards' COST by 1, so `CanPlay` (affordability) and `SpendResources`
//!   (payment) both see `cost + 1`.
//!
//! GH #741 (same root cause as RightHandHand): `IvoryTile.AfterCardPlayed`
//! reads `cardPlay.Resources.EnergyValue`, which C# sets for every play, so a
//! 3-cost non-X card refunds 1 energy.

use std::num::NonZeroU32;

use sts2_core::cards::CardId;
use sts2_core::combat_loop::play_card;
use sts2_core::potions::CharacterId;
use sts2_core::{legal_actions, Action, CardUid, CombatState, MonsterKind, Relic, RelicKind};

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_content::relics::dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
}

/// Mid-turn Ironclad state against one 300-HP Nibbit, holding `relic`, with
/// `hand` in hand and `energy` to spend.
fn state_with(relic: RelicKind, hand: &[CardId], energy: i8) -> CombatState {
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(7),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(7),
        ..Default::default()
    };
    state.player.character = Some(CharacterId::Ironclad);
    state.player.current_hp = 70;
    state.player.max_hp = 70;
    state.player.max_energy = 3;
    state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = 300;
        c.current_hp = 300;
    }
    state.relics.push(Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: relic,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    for &id in hand {
        let mut card = sts2_core::lookup_card_factory()
            .and_then(|f| f(id, false))
            .expect("card factory has the card");
        let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
        card.uid = uid;
        state.cards.push(card);
        state.player.hand.push(uid);
    }
    state.player.energy = i32::from(energy);
    state.phase = sts2_core::CombatPhase::PlayerPlayPhase;
    state
}

fn any_card_play_legal(state: &CombatState) -> bool {
    legal_actions(state)
        .iter()
        .any(|a| matches!(a, Action::PlayCard { .. }))
}

/// VelvetChoker allows six card plays a turn; the seventh is vetoed both by
/// `legal_actions` and by `play_card` (which leaves hand and energy alone).
#[test]
fn velvet_choker_blocks_the_seventh_card_play() {
    install();
    let mut state = state_with(RelicKind::VelvetChoker, &[CardId::DefendIronclad; 7], 99);
    for played in 0..6 {
        assert!(any_card_play_legal(&state), "play {played} should be legal");
        let before = state.player.hand.len();
        let _ = play_card(&mut state, 0, None);
        assert_eq!(
            state.player.hand.len(),
            before - 1,
            "play {played} resolved"
        );
    }
    assert!(
        !any_card_play_legal(&state),
        "the seventh play must not be offered: {:?}",
        legal_actions(&state)
    );
    let (hand, energy) = (state.player.hand.clone(), state.player.energy);
    let _ = play_card(&mut state, 0, None);
    assert_eq!(state.player.hand, hand, "the seventh play must be rejected");
    assert_eq!(state.player.energy, energy);
}

/// VelvetChoker's counter resets at the player's turn start, so the cap is per
/// turn: after a new turn the hand is playable again.
#[test]
fn velvet_choker_cap_resets_next_turn() {
    install();
    let mut state = state_with(RelicKind::VelvetChoker, &[CardId::DefendIronclad; 6], 99);
    for _ in 0..6 {
        let _ = play_card(&mut state, 0, None);
    }
    assert!(!any_card_play_legal(&state));
    let _ = sts2_core::apply(&mut state, Action::EndTurn);
    assert_eq!(state.phase, sts2_core::CombatPhase::PlayerPlayPhase);
    assert!(!state.player.hand.is_empty(), "turn start drew cards");
    assert!(any_card_play_legal(&state));
}

/// VelvetChoker grants +1 max energy (`VelvetChoker.ModifyMaxEnergy`).
#[test]
fn velvet_choker_adds_one_max_energy() {
    install();
    let state = state_with(RelicKind::VelvetChoker, &[], 0);
    let ctx = sts2_core::hooks::MaxEnergyCtx::default();
    assert_eq!(
        sts2_core::hooks::dispatch_modify_max_energy(&state, &ctx),
        1
    );
}

/// SpikedGauntlets makes a 1-cost Power card cost 2: unaffordable at 1 energy
/// (not offered, and `play_card` rejects it without spending anything).
#[test]
fn spiked_gauntlets_power_card_unaffordable_at_printed_cost() {
    install();
    let mut state = state_with(RelicKind::SpikedGauntlets, &[CardId::Inflame], 1);
    let inflame = state.player.hand[0];
    assert_eq!(sts2_core::hooks::effective_card_cost(&state, inflame), 2);
    assert!(!any_card_play_legal(&state), "{:?}", legal_actions(&state));
    let _ = play_card(&mut state, 0, None);
    assert_eq!(state.player.hand.as_slice(), &[inflame]);
    assert_eq!(state.player.energy, 1);
}

/// SpikedGauntlets: at 2 energy the 1-cost Power card is playable and costs 2.
#[test]
fn spiked_gauntlets_power_card_pays_cost_plus_one() {
    install();
    let mut state = state_with(RelicKind::SpikedGauntlets, &[CardId::Inflame], 2);
    assert!(any_card_play_legal(&state));
    let _ = play_card(&mut state, 0, None);
    assert!(state.player.hand.is_empty(), "Inflame was played");
    assert_eq!(state.player.energy, 0);
}

/// SpikedGauntlets leaves non-Power cards at their printed cost.
#[test]
fn spiked_gauntlets_leaves_attacks_alone() {
    install();
    let mut state = state_with(RelicKind::SpikedGauntlets, &[CardId::StrikeIronclad], 1);
    let strike = state.player.hand[0];
    assert_eq!(sts2_core::hooks::effective_card_cost(&state, strike), 1);
    let _ = play_card(&mut state, 0, Some(0));
    assert!(state.player.hand.is_empty());
    assert_eq!(state.player.energy, 0);
}

/// GH #741: IvoryTile refunds 1 energy after a non-X play that spends 3+
/// energy (Bludgeon, cost 3), and not after a 1-cost Strike.
#[test]
fn ivory_tile_refunds_after_a_three_cost_non_x_play() {
    install();
    for (card, energy_after) in [(CardId::Bludgeon, 1), (CardId::StrikeIronclad, 2)] {
        let mut state = state_with(RelicKind::IvoryTile, &[card], 3);
        let _ = play_card(&mut state, 0, Some(0));
        assert!(state.player.hand.is_empty(), "{card:?} was played");
        assert_eq!(state.player.energy, energy_after, "{card:?}");
    }
}

/// IvoryTile reads the raw energy spent (`EnergyValue`), not the ChemicalX-
/// boosted X: a 1-energy Whirlwind with ChemicalX (X = 3) refunds nothing.
#[test]
fn ivory_tile_ignores_chemical_x_bonus() {
    install();
    let mut state = state_with(RelicKind::IvoryTile, &[CardId::Whirlwind], 1);
    state.relics.push(Relic {
        id: NonZeroU32::new(2).expect("nz"),
        kind: RelicKind::ChemicalX,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    let _ = play_card(&mut state, 0, None);
    assert!(state.player.hand.is_empty(), "Whirlwind was played");
    assert_eq!(state.player.energy, 0, "no IvoryTile refund for 1 energy");
}

/// ChemicalX still adds +2 to the X an X-cost card resolves: a 1-energy
/// Whirlwind (5 damage per hit) with ChemicalX hits 3 times.
#[test]
fn chemical_x_still_boosts_the_x_value() {
    install();
    let mut state = state_with(RelicKind::ChemicalX, &[CardId::Whirlwind], 1);
    let hp_before =
        state.creatures.iter().map(|c| c.current_hp).sum::<i32>() - state.player.current_hp;
    let _ = play_card(&mut state, 0, None);
    let hp_after =
        state.creatures.iter().map(|c| c.current_hp).sum::<i32>() - state.player.current_hp;
    assert_eq!(hp_before - hp_after, 15, "3 hits of 5");
}
