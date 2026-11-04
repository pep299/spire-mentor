//! Gang Up counts a teammate's hits (GH #989). C# `GangUp` deals
//! `5 + Extra × n`, where `n` counts the target's powered
//! `DamageReceivedEntry`s this turn from a same-side dealer that is not the
//! card's owner: in co-op, every teammate.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    Action, CardUid, CombatPhase, CombatState, GameVersion, MonsterKind, PlayerCombatState,
};

fn install() {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

/// Our Ironclad and a Silent teammate against a Nibbit, our play phase open.
fn fight() -> CombatState {
    install();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(1234_i32),
        game_version: GameVersion::V0_111_0,
        ..Default::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.player.character = Some(CharacterId::Ironclad);
    state.init_player_creature();
    state.allies.push(PlayerCombatState {
        current_hp: 70,
        max_hp: 70,
        character: Some(CharacterId::Silent),
        ..PlayerCombatState::default()
    });
    sts2_core::coop::open_seats(&mut state);
    state.coop.full = true;
    let _ = state.spawn_enemy(MonsterKind::Nibbit);
    state.round = 1;
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

/// The Nibbit's HP lost to our Gang Up.
fn gang_up_damage(state: &mut CombatState) -> i32 {
    let factory = sts2_core::lookup_card_factory().expect("factory");
    let mut card = factory(CardId::GangUp, false).expect("card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    state.player.hand.push(uid);
    let hand_index = u8::try_from(state.player.hand.len() - 1).expect("fits");
    let nibbit = state.enemy_uids[0];
    let hp = |s: &CombatState| {
        s.creatures
            .iter()
            .find(|c| c.id == nibbit.0)
            .expect("nibbit")
            .current_hp
    };
    let before = hp(state);
    let _ = sts2_core::apply(
        state,
        Action::PlayCard {
            hand_index,
            target: Some(0),
        },
    );
    before - hp(state)
}

#[test]
fn gang_up_counts_a_teammates_hits_on_the_target() {
    let mut state = fight();
    let nibbit = state.enemy_uids[0];
    state.allies[0]
        .powered_hits_by_target_this_turn
        .push((nibbit, 3));
    assert_eq!(gang_up_damage(&mut state), 5 + 5 * 3);
}

#[test]
fn gang_up_without_teammate_hits_deals_its_base() {
    let mut state = fight();
    assert_eq!(gang_up_damage(&mut state), 5);
}

#[test]
fn our_own_hits_do_not_count() {
    let mut state = fight();
    let nibbit = state.enemy_uids[0];
    state
        .player
        .powered_hits_by_target_this_turn
        .push((nibbit, 3));
    assert_eq!(gang_up_damage(&mut state), 5);
}
