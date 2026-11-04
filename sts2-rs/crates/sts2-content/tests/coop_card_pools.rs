//! A co-op fight generates cards from C#'s co-op pools (GH #991): the same
//! `GenerateAllCards()` lists with the `MultiplayerOnly` cards kept
//! (`CardFactory.FilterForPlayerCount`), so the same generation roll lands on
//! the card the game makes.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::pool_order::{coop_combat_pool, Pool};
use sts2_core::cards::{CardId, CardType};
use sts2_core::potions::CharacterId;
use sts2_core::{
    Action, CardUid, CombatPhase, CombatState, GameVersion, MonsterKind, PlayerCombatState,
};

fn install() {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

/// An Ironclad in a v0.111.0 fight with a Nibbit, our play phase open, and
/// one teammate seat when `coop`.
fn fight(coop: bool) -> CombatState {
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
    if coop {
        state.allies.push(PlayerCombatState {
            current_hp: 70,
            max_hp: 70,
            character: Some(CharacterId::Silent),
            ..PlayerCombatState::default()
        });
        sts2_core::coop::open_seats(&mut state);
        state.coop.full = true;
    }
    let _ = state.spawn_enemy(MonsterKind::Nibbit);
    state.round = 1;
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

/// Play `id` and return the cards it put in our hand.
fn play(state: &mut CombatState, id: CardId) -> Vec<CardId> {
    let factory = sts2_core::lookup_card_factory().expect("factory");
    let mut card = factory(id, false).expect("card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    state.player.hand.push(uid);
    let hand_index = u8::try_from(state.player.hand.len() - 1).expect("fits");
    let target = state.enemy_uids.first().map(|_| 0);
    let _ = sts2_core::apply(state, Action::PlayCard { hand_index, target });
    state
        .player
        .hand
        .iter()
        .map(|u| state.cards[u.0.get() as usize - 1].card_id)
        .collect()
}

/// The first card of `pool` after C#'s `TakeRandom` (an `UnstableShuffle`)
/// on a copy of the fight's generation stream.
fn first_after_shuffle(state: &CombatState, mut pool: Vec<CardId>) -> CardId {
    let mut rng = state.run_rng.combat_card_generation.clone();
    rng.shuffle(pool.as_mut_slice());
    pool[0]
}

#[test]
fn jack_of_all_trades_draws_from_the_coop_colorless_pool() {
    let mut state = fight(true);
    let pool: Vec<CardId> = coop_combat_pool(Pool::Colorless, GameVersion::V0_111_0)
        .iter()
        .map(|s| s.id)
        .filter(|&id| id != CardId::JackOfAllTrades)
        .collect();
    assert!(pool.contains(&CardId::BelieveInYou));
    let expected = first_after_shuffle(&state, pool);
    assert_eq!(play(&mut state, CardId::JackOfAllTrades), [expected]);
}

#[test]
fn infernal_blade_draws_from_the_coop_ironclad_attacks() {
    let mut state = fight(true);
    let pool: Vec<CardId> = coop_combat_pool(
        Pool::Character(CharacterId::Ironclad),
        GameVersion::V0_111_0,
    )
    .iter()
    .filter(|s| s.card_type == CardType::Attack)
    .map(|s| s.id)
    .collect();
    assert!(pool.contains(&CardId::Outrage), "a multiplayer-only attack");
    let expected = first_after_shuffle(&state, pool);
    assert_eq!(play(&mut state, CardId::InfernalBlade), [expected]);
}

/// Solo, the same rolls keep drawing from the single-player pools.
#[test]
fn a_solo_fight_keeps_the_single_player_pools() {
    let mut state = fight(false);
    let made = play(&mut state, CardId::JackOfAllTrades);
    assert_eq!(made.len(), 1);
    assert!(!sts2_content::cards::spec_index::is_multiplayer_only(
        made[0]
    ));
}
