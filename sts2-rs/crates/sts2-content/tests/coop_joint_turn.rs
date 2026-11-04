//! The joint co-op search's seat hand-off (`CoopState::joint`, GH #1046): our
//! `EndTurn` seats the teammate, who then plays move by move through `apply`;
//! their `EndTurn` seats us again and the end of turn runs on.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::{Action, CardUid, CombatPhase, CombatState, MonsterKind, PlayerCombatState};

fn mint(state: &mut CombatState, id: CardId) -> CardUid {
    let factory = sts2_core::lookup_card_factory().expect("card factory installed");
    let mut card = factory(id, false).expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    uid
}

/// Us (Ironclad, ten Defends) and a Silent teammate (ten of `mate_card`), in
/// the play phase of round 1 against one Nibbit at `enemy_hp`, joint search on.
fn joint_fight(mate_card: CardId, enemy_hp: i32) -> CombatState {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        ..Default::default()
    };
    state.player.character = Some(sts2_core::potions::CharacterId::Ironclad);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = enemy_hp;
        c.current_hp = enemy_hp;
    }
    for _ in 0..10 {
        let uid = mint(&mut state, CardId::DefendIronclad);
        state.player.draw.push(uid);
    }
    let mut mate = PlayerCombatState {
        character: Some(sts2_core::potions::CharacterId::Silent),
        current_hp: 70,
        max_hp: 70,
        ..PlayerCombatState::default()
    };
    for _ in 0..10 {
        mate.draw.push(mint(&mut state, mate_card));
    }
    state.allies.push(mate);
    sts2_core::coop::open_seats(&mut state);
    state.coop.full = true;
    state.round = 1;
    state.phase = CombatPhase::PlayerTurnStart;
    let _ = sts2_core::combat_loop::start_player_turn(&mut state);
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert_eq!(state.coop.to_play, 1);
    state.coop.joint = true;
    state
}

#[test]
fn our_end_turn_seats_the_teammate_and_theirs_ends_the_round() {
    let mut state = joint_fight(CardId::DefendSilent, 200);

    let _ = sts2_core::apply(&mut state, Action::EndTurn);

    assert_eq!(state.coop.joint_acting, Some(0), "the teammate is to move");
    assert_eq!(state.coop.seated, 1);
    assert_eq!(state.coop.to_play, 0);
    assert_eq!(state.round, 1, "the end of turn waits for them");
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert_eq!(state.player.max_hp, 70, "their seat");
    let actions = sts2_core::legal_actions(&state);
    assert!(actions.iter().any(|a| matches!(a, Action::PlayCard { .. })));
    assert!(
        !actions
            .iter()
            .any(|a| matches!(a, Action::DrinkPotion { .. })),
        "a joint teammate plays cards only"
    );

    let _ = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(
        state.player.discard.len(),
        1,
        "the Defend went to their discard"
    );
    let _ = sts2_core::apply(&mut state, Action::EndTurn);

    assert_eq!(state.coop.joint_acting, None);
    assert_eq!(state.coop.seated, 0, "we sit in our seat again");
    assert_eq!(state.player.max_hp, 80);
    assert_eq!(state.round, 2, "the enemy turn ran and round 2 began");
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    assert_eq!(state.coop.to_play, 1, "they have round 2 ahead of them");
    assert_eq!(
        state.allies[0].hand.len(),
        5,
        "they drew their round-2 hand"
    );
}

#[test]
fn a_teammate_who_wins_the_fight_hands_the_seat_back() {
    let mut state = joint_fight(CardId::StrikeSilent, 6);

    let _ = sts2_core::apply(&mut state, Action::EndTurn);
    assert_eq!(state.coop.joint_acting, Some(0));
    let target = state
        .creatures
        .iter()
        .position(|c| c.species.is_some())
        .and_then(|i| u8::try_from(i).ok());
    let actions = sts2_core::legal_actions(&state);
    let strike = actions
        .iter()
        .find(|a| {
            matches!(
                a,
                Action::PlayCard {
                    target: Some(_),
                    ..
                }
            )
        })
        .cloned()
        .unwrap_or(Action::PlayCard {
            hand_index: 0,
            target,
        });
    let _ = sts2_core::apply(&mut state, strike);

    assert!(state.is_over && state.player_won);
    assert_eq!(state.coop.joint_acting, None);
    assert_eq!(state.coop.seated, 0, "a finished state reads from our seat");
    assert_eq!(state.player.max_hp, 80);
}

#[test]
fn without_joint_our_end_turn_plays_the_teammate_at_once() {
    let mut state = joint_fight(CardId::DefendSilent, 200);
    state.coop.joint = false;

    let _ = sts2_core::apply(&mut state, Action::EndTurn);

    assert_eq!(state.coop.joint_acting, None);
    assert_eq!(state.round, 2);
}
