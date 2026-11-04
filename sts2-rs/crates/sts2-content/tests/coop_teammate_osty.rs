//! GH #1017: a Necrobinder teammate's Osty fights for them as ours does.
//!
//! C# `CreatureCmd.Damage` (v0.111.0 `CreatureCmd.cs:285-310`): the owner's
//! block absorbs first, then `Hook.ModifyUnblockedDamageTarget` hands the
//! unblocked part of a powered hit on a player to their living Osty
//! (`DieForYouPower.cs:15-30`, which checks `target == Owner.PetOwner`: each
//! Osty covers only its own player), and the Osty's overkill goes on to the
//! player. Osty attacks (`Unleash`) are dealt `FromOsty(Owner.Osty)`: the
//! card owner's Osty, so a teammate's Unleash hits with their Osty's HP.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    Action, CardUid, CombatPhase, CombatState, Intent, MonsterKind, PlayerCombatState,
};

fn install() {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

/// Our Ironclad and a Necrobinder teammate (`mate_block` block) whose Osty
/// has `osty_hp`, against a 200-HP Nibbit telegraphing a 7-damage attack.
fn fight(osty_hp: i32, mate_block: i32) -> CombatState {
    install();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(1017_i32),
        ..Default::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.character = Some(CharacterId::Ironclad);
    state.init_player_creature();
    state.allies.push(PlayerCombatState {
        current_hp: 70,
        max_hp: 70,
        block: mate_block,
        energy: 3,
        max_energy: 3,
        character: Some(CharacterId::Necrobinder),
        ..PlayerCombatState::default()
    });
    sts2_core::coop::open_seats(&mut state);
    let nibbit = state.spawn_enemy(MonsterKind::Nibbit);
    let c = state
        .creatures
        .iter_mut()
        .find(|c| c.id == nibbit.0)
        .expect("nibbit");
    c.max_hp = 200;
    c.current_hp = 200;
    c.intent = Some(Intent::Attack { damage: 7 });
    c.ai = None;
    state.restore_ally_osty(0, osty_hp, osty_hp.max(1), 0);
    state.round = 1;
    state.coop.threat_round = Some(1);
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn mate_osty_hp(state: &CombatState) -> i32 {
    let uid = state.allies[0].pet_uid.expect("their Osty");
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("in the arena")
        .current_hp
}

/// "NIBBIT dealt 1 damage to OSTY / 6 damage to NECROBINDER" (the 4-player
/// recording of GH #1017): a 1-HP Osty soaks 1 of a 7 hit, its overkill
/// goes on to its owner. We have no Osty and take all 7.
#[test]
fn a_hit_on_a_teammate_goes_through_their_osty_first() {
    let mut state = fight(1, 0);
    state.phase = CombatPhase::EnemyTurnExec;
    let _ = sts2_core::combat_loop::enemy_turn_exec(&mut state);
    assert_eq!(mate_osty_hp(&state), 0);
    assert_eq!(state.allies[0].current_hp, 70 - 6);
    assert_eq!(state.player.current_hp, 80 - 7);
}

/// Their own block first, then their Osty, then them.
#[test]
fn a_teammates_block_soaks_before_their_osty() {
    let mut state = fight(3, 2);
    state.phase = CombatPhase::EnemyTurnExec;
    let _ = sts2_core::combat_loop::enemy_turn_exec(&mut state);
    assert_eq!(state.allies[0].block, 0);
    assert_eq!(mate_osty_hp(&state), 0);
    assert_eq!(state.allies[0].current_hp, 70 - 2);
}

/// A big Osty takes the whole unblocked hit.
#[test]
fn a_big_teammate_osty_takes_the_whole_hit() {
    let mut state = fight(10, 0);
    state.phase = CombatPhase::EnemyTurnExec;
    let _ = sts2_core::combat_loop::enemy_turn_exec(&mut state);
    assert_eq!(mate_osty_hp(&state), 3);
    assert_eq!(state.allies[0].current_hp, 70);
}

/// Their Unleash deals `6 + their Osty's HP` (C# `Unleash.cs`), from their
/// Osty.
#[test]
fn a_teammates_unleash_hits_with_their_ostys_damage() {
    let mut state = fight(5, 0);
    let factory = sts2_core::lookup_card_factory().expect("factory");
    let mut card = factory(CardId::Unleash, false).expect("card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    state.allies[0].hand.push(uid);
    let nibbit = state.enemy_uids[0];
    let hp = |s: &CombatState| {
        s.creatures
            .iter()
            .find(|c| c.id == nibbit.0)
            .expect("nibbit")
            .current_hp
    };
    let before = hp(&state);
    let _ = sts2_core::coop::apply_as(
        &mut state,
        0,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    assert_eq!(before - hp(&state), 6 + 5);
    assert_eq!(state.allies[0].energy, 2, "they paid for it");
    assert_eq!(state.player.pet_uid, None, "we still have none");
}
