//! The `AnyAlly` cards in a co-op fight: each lands on the teammate it targets
//! (C# `cardPlay.Target`), whose energy, piles, orbs and powers are simulated.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::hooks::{modify_damage, CreatureRef, DamageCtx, ValueProps};
use sts2_core::{
    Action, CardUid, CombatPhase, CombatState, CreatureUid, MonsterKind, PlayerCombatState,
    PowerKind,
};

fn install() {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

fn mint(state: &mut CombatState, id: CardId, upgraded: bool) -> CardUid {
    let factory = sts2_core::lookup_card_factory().expect("card factory installed");
    let mut card = factory(id, upgraded).expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    uid
}

/// Us and one teammate seat in a v0.111.0 fight with a Nibbit, our play phase
/// open, 5 energy each.
fn fight() -> (CombatState, CreatureUid) {
    install();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        game_version: sts2_core::GameVersion::V0_111_0,
        ..Default::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.init_player_creature();
    state.allies.push(PlayerCombatState {
        current_hp: 70,
        max_hp: 70,
        energy: 5,
        ..PlayerCombatState::default()
    });
    sts2_core::coop::open_seats(&mut state);
    state.coop.full = true;
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    state.round = 1;
    state.phase = CombatPhase::PlayerPlayPhase;
    (state, enemy)
}

/// Play `id` from our hand at the teammate.
fn play(state: &mut CombatState, id: CardId, upgraded: bool) {
    let uid = mint(state, id, upgraded);
    state.player.hand.push(uid);
    let hand_index = u8::try_from(state.player.hand.len() - 1).expect("fits");
    let _ = sts2_core::apply(
        state,
        Action::PlayCard {
            hand_index,
            target: None,
        },
    );
    assert!(
        !state.player.hand.contains(&uid),
        "{id:?} was played (phase {:?})",
        state.phase
    );
}

/// The teammate's `kind`, waiting in their seat.
fn mate_power(state: &CombatState, kind: PowerKind) -> i32 {
    state.coop.stash[0]
        .powers
        .iter()
        .filter_map(|u| state.powers.get_by_id(u.0))
        .filter(|p| p.kind == kind)
        .map(|p| p.amount)
        .sum()
}

#[test]
fn believe_in_you_gives_the_teammate_energy() {
    let (mut state, _) = fight();
    play(&mut state, CardId::BelieveInYou, false);
    assert_eq!(state.allies[0].energy, 7);
    assert_eq!(state.player.energy, 5);
    play(&mut state, CardId::BelieveInYou, true);
    assert_eq!(state.allies[0].energy, 10);
}

#[test]
fn blaze_gives_the_teammate_strength() {
    let (mut state, _) = fight();
    play(&mut state, CardId::Blaze, false);
    assert_eq!(mate_power(&state, PowerKind::Strength), 5);
    assert_eq!(state.player_power_amount(PowerKind::Strength), 0);
}

#[test]
fn coordinate_and_fade_lend_the_teammate_strength_and_dexterity_for_a_turn() {
    let (mut state, _) = fight();
    play(&mut state, CardId::CoordinateCard, false);
    play(&mut state, CardId::FadeCard, true);
    assert_eq!(mate_power(&state, PowerKind::Strength), 5);
    assert_eq!(mate_power(&state, PowerKind::Coordinate), 5);
    assert_eq!(mate_power(&state, PowerKind::Dexterity), 9);
    assert_eq!(mate_power(&state, PowerKind::Fade), 9);
    assert_eq!(state.player_power_amount(PowerKind::Strength), 0);
}

#[test]
fn concoct_and_soulbound_go_on_the_teammate() {
    let (mut state, _) = fight();
    play(&mut state, CardId::ConcoctCard, false);
    play(&mut state, CardId::SoulboundCard, false);
    assert_eq!(mate_power(&state, PowerKind::Concoct), 3);
    assert_eq!(mate_power(&state, PowerKind::Soulbound), 1);
}

/// A Soul we make gives the teammate we Soulbound one of their own, shuffled
/// into their draw pile; theirs does not come back to us.
#[test]
fn soulbound_hands_the_teammate_a_soul_for_each_of_ours() {
    let (mut state, _) = fight();
    play(&mut state, CardId::SoulboundCard, false);
    let factory = sts2_core::lookup_card_factory().expect("factory");
    let soul = factory(CardId::Soul, false).expect("soul");
    let decoy = mint(&mut state, CardId::StrikeIronclad, false);
    state.player.play.push(decoy); // mid-play: the Soul is ours
    let _ = sts2_core::card_effects::spawn_card_into_pile(
        &mut state,
        soul,
        sts2_core::card_effects::PileKind::Hand,
    );
    let souls = |pile: &[CardUid]| {
        pile.iter()
            .filter(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::Soul)
            .count()
    };
    assert_eq!(souls(&state.allies[0].draw), 1);
    assert_eq!(souls(&state.player.draw), 0);
}

#[test]
fn ignition_channels_plasma_for_the_teammate() {
    let (mut state, _) = fight();
    state.allies[0].orbs.capacity = 3;
    play(&mut state, CardId::Ignition, false);
    assert!(state.allies[0]
        .orbs
        .slots
        .contains(&sts2_core::orbs::Orb::plasma()));
    assert!(state.player.orbs.slots.is_empty());
}

#[test]
fn constellation_draws_energises_and_blocks_the_teammate() {
    let (mut state, _) = fight();
    let card = mint(&mut state, CardId::DefendSilent, false);
    state.allies[0].draw.push(card);
    play(&mut state, CardId::Constellation, false);
    assert_eq!(state.allies[0].hand.as_slice(), [card]);
    assert_eq!(state.allies[0].energy, 6);
    assert_eq!(state.allies[0].block, 9);
    assert_eq!(state.player.block, 0);
}

#[test]
fn largesse_puts_a_colorless_card_in_the_teammates_hand() {
    let (mut state, _) = fight();
    play(&mut state, CardId::Largesse, true);
    assert_eq!(state.allies[0].hand.len(), 1);
    let card = &state.cards[state.allies[0].hand[0].0.get() as usize - 1];
    assert!(card.upgraded, "Largesse+ upgrades it");
    assert!(state.player.hand.is_empty());
}

/// The teammate's pick comes from the chooser when one answers, from their
/// own draw pile into their own hand.
#[test]
fn tutor_lets_the_teammate_fetch_a_card_from_their_draw_pile() {
    let (mut state, _) = fight();
    let strike = mint(&mut state, CardId::StrikeSilent, false);
    let defend = mint(&mut state, CardId::DefendSilent, false);
    state.allies[0].draw.extend([strike, defend]);
    let asked = std::rc::Rc::new(std::cell::Cell::new(None));
    let seen = std::rc::Rc::clone(&asked);
    sts2_core::coop::with_seat_chooser(
        Box::new(move |_, ally, options, _| {
            seen.set(Some(ally));
            let i = options
                .iter()
                .position(|o| o.card_id == CardId::DefendSilent)?;
            Some(Action::ResolveChoice {
                selection: smallvec::smallvec![u8::try_from(i).ok()?],
            })
        }),
        || play(&mut state, CardId::Tutor, false),
    );
    assert_eq!(asked.get(), Some(0), "the teammate picks");
    assert_eq!(state.allies[0].hand.as_slice(), [defend]);
    assert_eq!(state.allies[0].draw.as_slice(), [strike]);
}

/// Intercept: our block, the teammate Covered (a powered hit on them does
/// nothing) and our Intercept doubling ours, both until the enemy turn ends.
#[test]
fn intercept_covers_the_teammate_until_the_enemy_turn_ends() {
    let (mut state, enemy) = fight();
    play(&mut state, CardId::InterceptCard, false);
    assert_eq!(state.player.block, 9);
    assert_eq!(mate_power(&state, PowerKind::Covered), 1);
    assert_eq!(state.player_power_amount(PowerKind::Intercept), 1);
    let hit = |target| DamageCtx {
        card_source: None,
        dealer: CreatureRef::Enemy(enemy),
        target,
        props: ValueProps::MOVE,
    };
    assert_eq!(modify_damage(&state, 6, &hit(CreatureRef::Player)), 12);
    sts2_core::coop::with_seat(&mut state, 1, |state| {
        assert_eq!(modify_damage(state, 6, &hit(CreatureRef::Player)), 0);
    });

    let _ = sts2_core::apply(&mut state, Action::EndTurn);
    assert_eq!(state.round, 2);
    assert_eq!(mate_power(&state, PowerKind::Covered), 0);
    assert_eq!(state.player_power_amount(PowerKind::Intercept), 0);
}
