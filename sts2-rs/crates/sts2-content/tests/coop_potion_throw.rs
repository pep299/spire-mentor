//! Throwing a potion at a teammate in co-op (GH #1017).
//!
//! C# lets a player throw any `AnyPlayer` potion at another living player
//! instead of drinking it (`PotionModel.CanThrowAtAlly`: `AnyPlayer`, more than
//! one player, combat in progress; `IsValidTarget`: a living player). The
//! potion's `OnUse(target)` then acts on the teammate. The engine offers one
//! `Action::DrinkPotionAtAlly` per living teammate beside the drink.

use sts2_core::potions::PotionKind;
use sts2_core::{
    Action, CombatPhase, CombatState, MonsterKind, PlayerCombatState, PowerKind, StepResult,
};

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
    sts2_content::potions::dispatch::install_default_registry();
}

/// Us holding `potion`, and `mates` teammate seats, in a fight with a Nibbit,
/// our play phase open, 3 energy each.
fn fight(mates: usize, potion: PotionKind) -> CombatState {
    install();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        game_version: sts2_core::GameVersion::V0_111_0,
        ..Default::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.potions.push(potion);
    state.init_player_creature();
    for _ in 0..mates {
        state.allies.push(PlayerCombatState {
            current_hp: 70,
            max_hp: 70,
            energy: 3,
            ..PlayerCombatState::default()
        });
    }
    sts2_core::coop::open_seats(&mut state);
    state.coop.full = true;
    let _ = state.spawn_enemy(MonsterKind::Nibbit);
    state.round = 1;
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

/// The legal uses of the potion in slot 0, by teammate (`None`: drunk).
fn uses(state: &CombatState) -> Vec<Option<u8>> {
    sts2_core::legal_actions(state)
        .into_iter()
        .filter(|a| a.potion_slot() == Some(0))
        .map(|a| a.potion_ally())
        .collect()
}

fn throw(state: &mut CombatState, ally: u8) -> StepResult {
    sts2_core::apply(
        state,
        Action::DrinkPotionAtAlly {
            potion_slot: 0,
            ally,
        },
    )
}

/// Teammate `ally`'s `kind`, waiting in their seat.
fn mate_power(state: &CombatState, ally: usize, kind: PowerKind) -> i32 {
    state.coop.stash[ally]
        .powers
        .iter()
        .filter_map(|u| state.powers.get_by_id(u.0))
        .filter(|p| p.kind == kind)
        .map(|p| p.amount)
        .sum()
}

fn our_power(state: &CombatState, kind: PowerKind) -> i32 {
    state
        .player_power_uids()
        .iter()
        .filter_map(|u| state.powers.get_by_id(u.0))
        .filter(|p| p.kind == kind)
        .map(|p| p.amount)
        .sum()
}

#[test]
fn an_any_player_potion_can_be_drunk_or_thrown_at_each_living_teammate() {
    assert_eq!(
        uses(&fight(2, PotionKind::BlockPotion)),
        [None, Some(0), Some(1)]
    );
    // Two players: the one teammate is a choice too (self or them).
    assert_eq!(uses(&fight(1, PotionKind::BlockPotion)), [None, Some(0)]);
}

#[test]
fn solo_and_other_potions_offer_no_throw() {
    assert_eq!(uses(&fight(0, PotionKind::BlockPotion)), [None]);
    // FirePotion is `AnyEnemy`, Gambler's Brew `Self`: never thrown at a player.
    assert_eq!(uses(&fight(2, PotionKind::FirePotion)), [None]);
    assert_eq!(uses(&fight(2, PotionKind::GamblersBrew)), [None]);
}

#[test]
fn a_thrown_block_potion_blocks_the_teammate_and_leaves_the_belt() {
    for ally in 0..2u8 {
        let mut state = fight(2, PotionKind::BlockPotion);
        assert_eq!(throw(&mut state, ally), StepResult::Advanced);
        let blocks: Vec<i32> = state.allies.iter().map(|a| a.block).collect();
        let mut want = vec![0, 0];
        want[usize::from(ally)] = 12;
        assert_eq!(blocks, want, "thrown at {ally}");
        assert_eq!(state.player.block, 0, "not us");
        assert!(state
            .player
            .potions
            .iter()
            .all(|p| *p != PotionKind::BlockPotion));
        assert_eq!(state.coop.seated, 0, "we are seated again");
    }
}

#[test]
fn a_thrown_strength_potion_gives_the_teammate_strength() {
    let mut state = fight(2, PotionKind::StrengthPotion);
    assert_eq!(throw(&mut state, 1), StepResult::Advanced);
    assert_eq!(mate_power(&state, 1, PowerKind::Strength), 2);
    assert_eq!(mate_power(&state, 0, PowerKind::Strength), 0);
    assert_eq!(our_power(&state, PowerKind::Strength), 0);
}

#[test]
fn a_thrown_energy_potion_is_the_teammates_energy() {
    let mut state = fight(2, PotionKind::EnergyPotion);
    assert_eq!(throw(&mut state, 0), StepResult::Advanced);
    assert_eq!(state.allies[0].energy, 5);
    assert_eq!(state.allies[1].energy, 3);
    assert_eq!(state.player.energy, 3);
}

#[test]
fn a_throw_at_a_dead_teammate_or_of_the_wrong_potion_changes_nothing() {
    let mut dead = fight(2, PotionKind::BlockPotion);
    dead.allies[0].current_hp = 0;
    assert_eq!(uses(&dead), [None, Some(1)]);
    assert_eq!(throw(&mut dead, 0), StepResult::InvalidTarget);
    assert_eq!(dead.player.potions.len(), 1, "still on the belt");
    let mut fire = fight(2, PotionKind::FirePotion);
    assert_eq!(throw(&mut fire, 0), StepResult::InvalidTarget);
    assert_eq!(fire.player.potions.len(), 1);
}
