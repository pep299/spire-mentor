//! Imitation Learning in a co-op fight (GH #992): its power sits on the card's
//! owner, bound to the teammate the card targeted. When that teammate plays a
//! Power card, the owner auto-plays a clone of it and the power counts down
//! (v0.111.0 `ImitationLearningPower.BeforeCardPlayed` / `AfterCardPlayed`).

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::{Action, CardUid, CombatPhase, CombatState, PlayerCombatState, PowerKind};

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
/// open, 9 energy each.
fn fight() -> CombatState {
    install();
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(4242_i32),
        game_version: sts2_core::GameVersion::V0_111_0,
        ..Default::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 9;
    state.init_player_creature();
    state.allies.push(PlayerCombatState {
        current_hp: 70,
        max_hp: 70,
        energy: 9,
        ..PlayerCombatState::default()
    });
    sts2_core::coop::open_seats(&mut state);
    state.coop.full = true;
    let _ = state.spawn_enemy(sts2_core::MonsterKind::Nibbit);
    state.round = 1;
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

/// Play `id` from our hand (an `AnyAlly` card lands on the teammate).
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
    assert!(!state.player.hand.contains(&uid), "{id:?} was played");
}

/// The teammate plays `id` from their hand (`coop::apply_as`).
fn mate_plays(state: &mut CombatState, id: CardId, target: Option<u8>) {
    let uid = mint(state, id, false);
    state.allies[0].hand.push(uid);
    let hand_index = u8::try_from(state.allies[0].hand.len() - 1).expect("fits");
    let _ = sts2_core::coop::apply_as(state, 0, Action::PlayCard { hand_index, target });
    assert!(
        !state.allies[0].hand.contains(&uid),
        "the teammate played {id:?}"
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

fn our_imitation(state: &CombatState) -> Vec<i32> {
    state
        .player_power_uids()
        .iter()
        .filter_map(|u| state.powers.get_by_id(u.0))
        .filter(|p| p.kind == PowerKind::ImitationLearning)
        .map(|p| p.amount)
        .collect()
}

#[test]
fn the_teammates_power_cards_are_copied_until_the_power_runs_out() {
    let mut state = fight();
    play(&mut state, CardId::ImitationLearningCard, false);
    assert_eq!(our_imitation(&state), vec![2]);

    mate_plays(&mut state, CardId::Footwork, None);
    assert_eq!(mate_power(&state, PowerKind::Dexterity), 2);
    assert_eq!(
        state.player_power_amount(PowerKind::Dexterity),
        2,
        "our clone"
    );
    assert_eq!(our_imitation(&state), vec![1]);

    mate_plays(&mut state, CardId::Inflame, None);
    assert_eq!(mate_power(&state, PowerKind::Strength), 2);
    assert_eq!(
        state.player_power_amount(PowerKind::Strength),
        2,
        "our clone"
    );
    assert!(
        our_imitation(&state).is_empty(),
        "decremented to 0, removed"
    );

    mate_plays(&mut state, CardId::Inflame, None);
    assert_eq!(mate_power(&state, PowerKind::Strength), 4);
    assert_eq!(state.player_power_amount(PowerKind::Strength), 2, "spent");

    // A Power clone leaves no card behind on our side, and the teammate's
    // energy paid for their cards only.
    assert!(state.player.play.is_empty());
    assert!(state.player.discard.is_empty());
    assert_eq!(state.player.exhaust.len(), 1, "only Imitation Learning");
    assert_eq!(state.player.energy, 8, "the clone costs us nothing");
}

#[test]
fn a_second_imitation_learning_raises_the_one_instance() {
    let mut state = fight();
    play(&mut state, CardId::ImitationLearningCard, false);
    play(&mut state, CardId::ImitationLearningCard, true);
    assert_eq!(our_imitation(&state), vec![5]);
}

#[test]
fn the_teammates_attacks_and_skills_are_not_copied() {
    let mut state = fight();
    play(&mut state, CardId::ImitationLearningCard, false);
    mate_plays(&mut state, CardId::StrikeSilent, Some(0));
    mate_plays(&mut state, CardId::DefendSilent, None);
    assert_eq!(our_imitation(&state), vec![2]);
    assert_eq!(state.player.block, 0);
}

/// `cardPlay.Card.Owner != _playerTarget`: the owner's own Power cards are
/// not the teammate's.
#[test]
fn our_own_power_cards_are_not_copied() {
    let mut state = fight();
    play(&mut state, CardId::ImitationLearningCard, false);
    play(&mut state, CardId::Inflame, false);
    assert_eq!(state.player_power_amount(PowerKind::Strength), 2);
    assert_eq!(mate_power(&state, PowerKind::Strength), 0);
    assert_eq!(our_imitation(&state), vec![2]);
}

/// The teammate's Imitation Learning, bound to us, copies our Power cards
/// onto them.
#[test]
fn a_teammates_imitation_learning_copies_our_power_cards() {
    let mut state = fight();
    mate_plays(&mut state, CardId::ImitationLearningCard, None);
    assert_eq!(mate_power(&state, PowerKind::ImitationLearning), 2);
    play(&mut state, CardId::Inflame, false);
    assert_eq!(state.player_power_amount(PowerKind::Strength), 2);
    assert_eq!(mate_power(&state, PowerKind::Strength), 2, "their clone");
    assert_eq!(mate_power(&state, PowerKind::ImitationLearning), 1);
}

/// Both players imitate each other: our clone of their Power is our Power
/// card, which their Imitation Learning copies back, and so on until one runs
/// out. C# fires each power once per first-in-series play.
#[test]
fn two_imitators_copy_back_and_forth_until_one_runs_out() {
    let mut state = fight();
    play(&mut state, CardId::ImitationLearningCard, false);
    mate_plays(&mut state, CardId::ImitationLearningCard, None);
    mate_plays(&mut state, CardId::Inflame, None);
    // Their Inflame → our clone (ours 2→1) → their clone (theirs 2→1) → our
    // clone (ours 1→0, gone) → their clone (theirs 1→0, gone).
    assert_eq!(mate_power(&state, PowerKind::Strength), 6);
    assert_eq!(state.player_power_amount(PowerKind::Strength), 4);
    assert!(our_imitation(&state).is_empty());
    assert_eq!(mate_power(&state, PowerKind::ImitationLearning), 0);
}

/// `IsFirstInSeries`: a Power card the teammate plays twice (Echo Form) is
/// copied once, after its first play.
#[test]
fn an_echoed_power_card_is_copied_once() {
    let mut state = fight();
    play(&mut state, CardId::ImitationLearningCard, false);
    let raw = state.powers.iter().map(|p| p.id.get()).max().unwrap_or(0) + 1;
    let id = std::num::NonZeroU32::new(raw).expect("nz");
    state.powers.push(sts2_core::Power {
        id,
        kind: PowerKind::EchoForm,
        amount: 1,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.coop.stash[0].powers.push(sts2_core::PowerUid(id));
    mate_plays(&mut state, CardId::Inflame, None);
    assert_eq!(mate_power(&state, PowerKind::Strength), 4, "echoed");
    assert_eq!(state.player_power_amount(PowerKind::Strength), 2);
    assert_eq!(our_imitation(&state), vec![1]);
}

/// The arena's next power id, which a fresh power takes (`max + 1`).
fn next_power_id(state: &CombatState) -> std::num::NonZeroU32 {
    let raw = state.powers.iter().map(|p| p.id.get()).max().unwrap_or(0) + 1;
    std::num::NonZeroU32::new(raw).expect("nz")
}

/// Power ids are reused, so a seat left behind by a removed power (here one
/// naming our own seat, as a spent Flanking of ours would) must not bind the
/// fresh Imitation Learning that takes its id.
#[test]
fn a_stale_power_seat_does_not_bind_a_fresh_imitation_learning() {
    let mut state = fight();
    let stale = next_power_id(&state);
    state.coop.power_seat.push((stale, 0));
    play(&mut state, CardId::ImitationLearningCard, false);
    let il = state
        .player_power_uids()
        .into_iter()
        .find(|u| {
            state
                .powers
                .get_by_id(u.0)
                .is_some_and(|p| p.kind == PowerKind::ImitationLearning)
        })
        .expect("our Imitation Learning");
    assert_eq!(il.0, stale, "it took the reused id");
    assert_eq!(
        sts2_core::coop::imitation_learning_target(&state, il.0, 0),
        1
    );
    play(&mut state, CardId::ImitationLearningCard, false);
    assert_eq!(our_imitation(&state), vec![4], "the second card stacks");
    mate_plays(&mut state, CardId::Inflame, None);
    assert_eq!(state.player_power_amount(PowerKind::Strength), 2);
}

/// The same for Tank's Guarded: a stale seat on the reused id must not hide
/// the Tank owner, or the Guarded would not leave with them.
#[test]
fn a_stale_power_seat_does_not_hide_a_guardeds_tank() {
    let mut state = fight();
    let stale = next_power_id(&state);
    state.coop.power_seat.push((stale, 1));
    sts2_core::coop::tank_guards_teammates(&mut state, 1);
    assert_eq!(mate_power(&state, PowerKind::Guarded), 1);
    assert_eq!(sts2_core::coop::seat_of_power(&state, stale), 0);
    sts2_core::coop::drop_guarded_of(&mut state, 0);
    assert_eq!(mate_power(&state, PowerKind::Guarded), 0);
}
