//! GH #1054 (Ironclad 2164 act 2, v0.111.0): Stampede auto-plays a Headbutt
//! during end turn and its put-back select surfaces. In C# the select is awaited
//! inside Headbutt's `OnPlay`; once it is answered `OnPlayWrapper` finishes the
//! card (its replays, then the move from Play to the result pile and
//! `CheckForEmptyHand`, v0.111.0 CardModel.cs:1985-2006) before returning to
//! `CardCmd.AutoPlay` and the Stampede loop. Only then does the turn go on to
//! the end-of-turn hand flush, so Headbutt lands in discard BEFORE the flushed
//! hand cards.
//!
//! The trigger is a Headbutt that REPLAYS: its second put-back select opens
//! while the resume of the first one is finishing the card, and drains inline
//! (C# `AutoResolveInline`). That nested resolve took the end-turn resume arm
//! meant for a top-level resolve and ran the whole end-turn chain, hand flush
//! included, from inside Headbutt's replay loop. Same pile contents, different
//! order, which then feeds the next StableShuffle.

use smallvec::SmallVec;

use sts2_core::cards::CardId;
use sts2_core::combat_loop::{start_combat, start_player_turn};
use sts2_core::{
    apply, legal_actions, Action, CardUid, CharacterId, CombatState, MonsterKind, PowerKind,
    StepResult,
};

fn mint(state: &mut CombatState, id: CardId) -> CardUid {
    let mut card = sts2_core::lookup_card_factory()
        .and_then(|f| f(id, false))
        .expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn answer_first(state: &mut CombatState) -> StepResult {
    let pick = legal_actions(state)
        .into_iter()
        .find_map(|a| match a {
            Action::ResolveChoice { selection } if !selection.is_empty() => Some(selection),
            _ => None,
        })
        .expect("a non-empty ResolveChoice is legal while a select is surfaced");
    apply(state, Action::ResolveChoice { selection: pick })
}

/// Hand = Headbutt (plus Stomp when `with_stomp`) and two Slimed, three Defends
/// in discard (so every Headbutt put-back has a real choice), eight Defends in
/// the draw pile, and Stampede(`stampede`). Headbutt replays `replays` times.
fn run(stampede: i32, with_stomp: bool, replays: i32) -> (CombatState, CardUid, [CardUid; 2]) {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(7),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(7),
        ..CombatState::default()
    };
    state.player.character = Some(CharacterId::Ironclad);
    state.player.current_hp = 400;
    state.player.max_hp = 400;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    let enemy = state.spawn_enemy(MonsterKind::Nibbit);
    if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
        c.max_hp = 400;
        c.current_hp = 400;
    }
    let draw: SmallVec<[CardUid; 32]> = (0..8)
        .map(|_| mint(&mut state, CardId::DefendIronclad))
        .collect();
    state.player.draw = draw.clone();
    let _ = start_combat(&mut state);
    let puid = state.player.creature_uid.expect("player creature");
    sts2_core::card_effects::apply_power_to_creature(
        &mut state,
        puid,
        PowerKind::Stampede,
        stampede,
    );
    let _ = start_player_turn(&mut state);

    let headbutt = mint(&mut state, CardId::Headbutt);
    state.cards[headbutt.0.get() as usize - 1].base_replay_count = replays;
    let stomp = with_stomp.then(|| mint(&mut state, CardId::Stomp));
    let slimed = [
        mint(&mut state, CardId::Slimed),
        mint(&mut state, CardId::Slimed),
    ];
    let discard = [
        mint(&mut state, CardId::DefendIronclad),
        mint(&mut state, CardId::DefendIronclad),
        mint(&mut state, CardId::DefendIronclad),
    ];
    state.player.hand.clear();
    state.player.discard.clear();
    state.player.exhaust.clear();
    state.player.draw = draw;
    state.player.hand.push(headbutt);
    if let Some(s) = stomp {
        state.player.hand.push(s);
    }
    state.player.hand.extend(slimed);
    state.player.discard.extend(discard);
    state.player.energy = 3;
    (state, headbutt, slimed)
}

fn assert_headbutt_before_flush(stampede: i32, with_stomp: bool, replays: i32) {
    let (mut state, headbutt, slimed) = run(stampede, with_stomp, replays);
    let mut r = apply(&mut state, Action::EndTurn);
    let mut guard = 0;
    while r == StepResult::PendingChoice {
        r = answer_first(&mut state);
        guard += 1;
        assert!(guard < 8, "too many selects");
    }
    let pos = |u: CardUid| {
        state
            .player
            .discard
            .iter()
            .position(|&d| d == u)
            .unwrap_or_else(|| panic!("{u:?} in discard {:?}", state.player.discard))
    };
    let h = pos(headbutt);
    assert!(
        h < pos(slimed[0]) && h < pos(slimed[1]),
        "Headbutt reaches discard before the end-of-turn hand flush \
         (stampede {stampede}, stomp {with_stomp}, replays {replays}): discard {:?}",
        state.player.discard
    );
}

#[test]
#[serial_test::serial(global_registry)]
fn stampede_headbutt_select_resolves_before_hand_flush() {
    assert_headbutt_before_flush(1, false, 0);
    assert_headbutt_before_flush(2, true, 0);
}

/// Ironclad 2164 act 2: the Stampede-played Headbutt replays once.
#[test]
#[serial_test::serial(global_registry)]
fn stampede_replayed_headbutt_reaches_discard_before_hand_flush() {
    assert_headbutt_before_flush(1, false, 1);
    assert_headbutt_before_flush(2, true, 1);
}
