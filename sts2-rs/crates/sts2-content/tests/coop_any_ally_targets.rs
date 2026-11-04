//! Choosing an `AnyAlly` card's teammate with three or four players (GH #1017).
//!
//! C# lets the player aim these cards at any living teammate
//! (`NTargetManager.cs:524-533`: a living player who is not us). The engine
//! offers one `Action::PlayCardAtAlly` per living teammate once there are two
//! or more; with one there is nothing to choose and the play stays an
//! untargeted `PlayCard`, as it always was.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::{
    Action, CardUid, CombatPhase, CombatState, MonsterKind, PlayerCombatState, PowerKind,
    StepResult,
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

/// Us and `mates` teammate seats in a v0.111.0 fight with a Nibbit, our play
/// phase open, 5 energy each.
fn fight(mates: usize) -> CombatState {
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
    for _ in 0..mates {
        state.allies.push(PlayerCombatState {
            current_hp: 70,
            max_hp: 70,
            energy: 5,
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

/// Put `id` in our hand; its hand index.
fn hold(state: &mut CombatState, id: CardId) -> u8 {
    let uid = mint(state, id, false);
    state.player.hand.push(uid);
    u8::try_from(state.player.hand.len() - 1).expect("fits")
}

/// The play of hand card `hand_index` at teammate `ally` (`None`: untargeted).
fn play_action(hand_index: u8, ally: Option<u8>) -> Action {
    match ally {
        Some(ally) => Action::PlayCardAtAlly { hand_index, ally },
        None => Action::PlayCard {
            hand_index,
            target: None,
        },
    }
}

/// The legal plays of hand card `hand_index`, by teammate (`None`: an
/// untargeted `PlayCard`).
fn plays_of(state: &CombatState, hand_index: u8) -> Vec<Option<u8>> {
    sts2_core::legal_actions(state)
        .into_iter()
        .filter_map(|a| match a {
            Action::PlayCard {
                hand_index: h,
                target: None,
            } if h == hand_index => Some(None),
            Action::PlayCardAtAlly {
                hand_index: h,
                ally,
            } if h == hand_index => Some(Some(ally)),
            _ => None,
        })
        .collect()
}

/// Play `id` from our hand at teammate `target`.
fn play_at(state: &mut CombatState, id: CardId, target: Option<u8>) {
    let hand_index = hold(state, id);
    assert!(
        plays_of(state, hand_index).contains(&target),
        "{id:?} at {target:?} is offered"
    );
    let r = sts2_core::apply(state, play_action(hand_index, target));
    assert_eq!(r, StepResult::Advanced, "{id:?} at {target:?}");
    assert!(state.player.hand.is_empty(), "{id:?} was played");
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

#[test]
fn an_any_ally_card_offers_one_play_per_living_teammate() {
    let mut state = fight(3);
    let hi = hold(&mut state, CardId::BelieveInYou);
    assert_eq!(plays_of(&state, hi), [Some(0), Some(1), Some(2)]);
    // A card with no teammate in it is untouched.
    let strike = hold(&mut state, CardId::DefendIronclad);
    assert_eq!(plays_of(&state, strike), [None]);
}

#[test]
fn with_one_teammate_the_play_stays_untargeted() {
    let mut state = fight(1);
    let hi = hold(&mut state, CardId::BelieveInYou);
    assert_eq!(plays_of(&state, hi), [None]);
}

#[test]
fn a_dead_teammate_is_not_offered() {
    let mut state = fight(3);
    state.allies[1].current_hp = 0;
    let hi = hold(&mut state, CardId::Blaze);
    assert_eq!(plays_of(&state, hi), [Some(0), Some(2)]);
    // Aiming at them anyway is refused and changes nothing (C#
    // `PlayCardAction` cancels a play at an invalid target).
    let energy = state.player.energy;
    let r = sts2_core::apply(&mut state, play_action(hi, Some(1)));
    assert_eq!(r, StepResult::InvalidTarget);
    assert_eq!(state.player.hand.len(), 1);
    assert_eq!(state.player.energy, energy);
}

#[test]
fn with_one_teammate_left_alive_the_play_is_untargeted_and_lands_on_them() {
    let mut state = fight(2);
    state.allies[0].current_hp = 0;
    let hi = hold(&mut state, CardId::BelieveInYou);
    assert_eq!(plays_of(&state, hi), [None]);
    state.player.hand.clear();
    play_at(&mut state, CardId::BelieveInYou, None);
    assert_eq!(state.allies[1].energy, 7);
    assert_eq!(state.allies[0].energy, 5);
}

#[test]
fn believe_in_you_energises_the_chosen_teammate() {
    let mut state = fight(2);
    play_at(&mut state, CardId::BelieveInYou, Some(1));
    assert_eq!(state.allies[1].energy, 7);
    assert_eq!(state.allies[0].energy, 5);
    assert_eq!(state.player.energy, 5);
}

#[test]
fn blaze_strengthens_the_chosen_teammate() {
    let mut state = fight(3);
    play_at(&mut state, CardId::Blaze, Some(2));
    assert_eq!(mate_power(&state, 2, PowerKind::Strength), 5);
    assert_eq!(mate_power(&state, 0, PowerKind::Strength), 0);
    assert_eq!(mate_power(&state, 1, PowerKind::Strength), 0);
    play_at(&mut state, CardId::Blaze, Some(0));
    assert_eq!(mate_power(&state, 0, PowerKind::Strength), 5);
    assert_eq!(mate_power(&state, 2, PowerKind::Strength), 5);
}

#[test]
fn intercept_covers_the_chosen_teammate() {
    let mut state = fight(2);
    play_at(&mut state, CardId::InterceptCard, Some(1));
    assert_eq!(mate_power(&state, 1, PowerKind::Covered), 1);
    assert_eq!(mate_power(&state, 0, PowerKind::Covered), 0);
    assert_eq!(state.player_power_amount(PowerKind::Intercept), 1);
}

#[test]
fn lift_blocks_the_chosen_teammate() {
    let mut state = fight(2);
    play_at(&mut state, CardId::Lift, Some(1));
    assert!(state.allies[1].block > 0);
    assert_eq!(state.allies[0].block, 0);
}

/// The chosen teammate picks, from their own draw pile.
#[test]
fn tutor_lets_the_chosen_teammate_fetch_a_card() {
    let mut state = fight(2);
    let first = mint(&mut state, CardId::StrikeSilent, false);
    let second = mint(&mut state, CardId::DefendSilent, false);
    state.allies[0].draw.push(first);
    state.allies[1].draw.push(second);
    let asked = std::rc::Rc::new(std::cell::Cell::new(None));
    let seen = std::rc::Rc::clone(&asked);
    sts2_core::coop::with_seat_chooser(
        Box::new(move |_, ally, _, _| {
            seen.set(Some(ally));
            Some(Action::ResolveChoice {
                selection: smallvec::smallvec![0],
            })
        }),
        || play_at(&mut state, CardId::Tutor, Some(1)),
    );
    assert_eq!(asked.get(), Some(1), "the second teammate picks");
    assert_eq!(state.allies[1].hand.as_slice(), [second]);
    assert!(state.allies[0].hand.is_empty());
    assert_eq!(state.allies[0].draw.as_slice(), [first]);
}

/// A teammate's own `AnyAlly` play names its target in THEIR frame: seated,
/// `allies[ally]` is us.
#[test]
fn a_teammate_can_aim_at_us() {
    let mut state = fight(2);
    let uid = mint(&mut state, CardId::BelieveInYou, false);
    state.allies[1].hand.push(uid);
    // Seated as teammate 1, we sit in their `allies[1]`.
    let _ = sts2_core::coop::apply_as(&mut state, 1, play_action(0, Some(1)));
    assert!(state.allies[1].hand.is_empty(), "played");
    assert_eq!(state.player.energy, 7);
    assert_eq!(state.allies[0].energy, 5);
}

/// C# `CardCmd.AutoPlay` rolls an `AnyAlly` card's teammate off
/// `CombatTargets` (`CardCmd.cs:86-97`), over the living ones in run order.
#[test]
fn an_auto_played_any_ally_card_rolls_its_teammate() {
    let mut state = fight(2);
    let mut rng = state.run_rng.combat_targets.clone();
    let pick = usize::try_from(rng.next_int_exclusive(0, 2)).expect("non-negative");
    let uid = mint(&mut state, CardId::BelieveInYou, false);
    state.player.draw.push(uid);
    assert!(sts2_core::combat_loop::auto_play_card_from_draw_top(
        &mut state, false
    ));
    assert_eq!(
        state.run_rng.combat_targets.counter(),
        rng.counter(),
        "one roll"
    );
    assert_eq!(state.allies[pick].energy, 7);
    assert_eq!(state.allies[1 - pick].energy, 5);
    assert_eq!(state.coop.card_target, None, "cleared after the play");
}

/// An enemy slot on a plain `PlayCard` means nothing to an `AnyAlly` card:
/// it lands on the first living teammate, as before GH #1017.
#[test]
fn a_plain_play_of_an_any_ally_card_ignores_its_enemy_slot() {
    let mut state = fight(2);
    let hand_index = hold(&mut state, CardId::BelieveInYou);
    let r = sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index,
            target: Some(1),
        },
    );
    assert_eq!(r, StepResult::Advanced);
    assert_eq!(state.allies[0].energy, 7);
    assert_eq!(state.allies[1].energy, 5);
}

/// A card that is not `AnyAlly` has no teammate to aim at.
#[test]
fn an_ordinary_card_cannot_be_aimed_at_a_teammate() {
    let mut state = fight(2);
    let hand_index = hold(&mut state, CardId::DefendIronclad);
    let r = sts2_core::apply(&mut state, play_action(hand_index, Some(1)));
    assert_eq!(r, StepResult::InvalidTarget);
    assert_eq!(state.player.hand.len(), 1);
}

static ASKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Believe in You's effect, plus a hand pick the first time it resolves, so
/// the play parks on the choice and its Echo Form replay runs on resume.
fn believe_then_pick(
    state: &mut CombatState,
    _t: Option<sts2_core::CreatureUid>,
) -> sts2_core::card_effects::EffectResult {
    sts2_core::coop::with_card_target(state, |state| state.gain_energy(2));
    if !ASKED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        let options = state
            .player
            .hand
            .iter()
            .map(|&uid| sts2_core::ChoiceOption {
                card_id: state.cards[uid.0.get() as usize - 1].card_id,
                upgraded: false,
                card_uid: Some(uid),
                source_pile: Some(sts2_core::cards::PileKind::Hand),
            })
            .collect();
        state.open_hand_choice(sts2_core::PendingChoice {
            kind: sts2_core::ChoiceKind::ExhaustFromHand,
            options,
            min_select: 1,
            max_select: 1,
            can_skip: false,
            post_effect: None,
            picked: smallvec::SmallVec::new(),
        });
    }
    sts2_core::card_effects::EffectResult::Resolved
}

/// C# keeps `cardPlay.Target` for the whole play: a play parked on a choice
/// resumes, replays included, at the teammate it chose.
#[test]
fn a_play_parked_on_a_choice_keeps_its_teammate() {
    let mut state = fight(2);
    sts2_core::card_effects::register(CardId::BelieveInYou, believe_then_pick);
    let row = state.player.creature_uid.expect("our row");
    sts2_core::card_effects::apply_power_to_creature(&mut state, row, PowerKind::EchoForm, 1);
    // Two cards to pick from, or the pick resolves itself.
    let _ = hold(&mut state, CardId::DefendIronclad);
    let _ = hold(&mut state, CardId::DefendIronclad);
    let hand_index = hold(&mut state, CardId::BelieveInYou);
    let r = sts2_core::apply(&mut state, play_action(hand_index, Some(1)));
    assert_eq!(r, StepResult::PendingChoice);
    assert_eq!(state.allies[1].energy, 7, "the first play");
    let _ = sts2_core::apply(
        &mut state,
        Action::ResolveChoice {
            selection: smallvec::smallvec![0],
        },
    );
    assert!(state.pending_choice.is_none());
    assert_eq!(
        state.allies[1].energy, 9,
        "the echo lands on the same teammate"
    );
    assert_eq!(state.allies[0].energy, 5);
    assert_eq!(state.coop.card_target, None, "cleared after the play");
}

/// C# `InterceptPower`: a powered hit on its owner is multiplied by
/// `coveredCreatures.Count + 1`, so covering both teammates triples it.
#[test]
fn intercept_on_two_teammates_triples_the_hits_on_us() {
    use sts2_core::hooks::{modify_damage, CreatureRef, DamageCtx, ValueProps};
    let mut state = fight(2);
    let enemy = state.enemy_uids[0];
    let hit = |state: &CombatState| {
        modify_damage(
            state,
            6,
            &DamageCtx {
                card_source: None,
                dealer: CreatureRef::Enemy(enemy),
                target: CreatureRef::Player,
                props: ValueProps::MOVE,
            },
        )
    };
    play_at(&mut state, CardId::InterceptCard, Some(0));
    assert_eq!(hit(&state), 12);
    play_at(&mut state, CardId::InterceptCard, Some(1));
    assert_eq!(mate_power(&state, 1, PowerKind::Covered), 1);
    assert_eq!(state.player_power_amount(PowerKind::Intercept), 1);
    assert_eq!(hit(&state), 18);
    // Covering the same teammate again still counts them once.
    play_at(&mut state, CardId::InterceptCard, Some(1));
    assert_eq!(hit(&state), 18);
}
