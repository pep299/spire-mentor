//! GH #1131: a reshuffle select inside a draw-pile auto-play prompts the player.
//!
//! C# `CardPileCmd.AutoPlayFromDrawPile` calls `ShuffleIfNecessary` before each
//! `drawPile.First()`, and the reshuffle awaits `Hook.AfterShuffle`, so
//! Stratagem's grid select prompts before the top card is taken
//! (`CardPileCmd.cs:857`, `:939`). Havoc, Cascade and Distilled Chaos all go
//! through it.
//!
//! Rust used to drain that select inline with option 0 because the collect
//! loop had nowhere to wait. It now parks the collect on
//! `pending_collect_resume`, and the resolver runs it first, before any
//! auto-play tail parked around it.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    ChoiceKind, CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory,
    MonsterId, MonsterKind, Power, PowerKind, PowerUid,
};

fn fresh() -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: GameVersion::V0_107_1,
        ..CombatState::default()
    };
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: 500,
        max_hp: 500,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(id));
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    state.player.character = Some(CharacterId::Ironclad);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    let pid = NonZeroU32::new(50).expect("nz");
    state.powers.push(Power::new(pid, PowerKind::Stratagem, 1));
    state.player.powers.push(PowerUid(pid));
    state
}

fn new_card(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, false, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn resolve(state: &mut CombatState, idx: u8) {
    let _ = sts2_core::apply(
        state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![idx],
        },
    );
}

fn assert_at_rest(state: &CombatState) {
    assert!(state.pending_choice.is_none());
    assert!(state.pending_collect_resume.is_none());
    assert!(state.pending_autoplay_resume.is_none());
    assert!(state.pending_play.is_none());
    assert!(
        state.player.play.is_empty(),
        "play: {:?}",
        state.player.play
    );
}

/// Hand-played Havoc on an empty draw pile: the reshuffle's Stratagem grid
/// prompts before Havoc takes the top card.
#[test]
fn havoc_reshuffle_select_prompts() {
    let mut state = fresh();
    for id in [CardId::StrikeIronclad, CardId::DefendIronclad, CardId::Bash] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let havoc = new_card(&mut state, CardId::Havoc);
    state.player.hand.push(havoc);

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("Stratagem grid prompts");
    assert_eq!(pc.kind, ChoiceKind::AddFromDrawToHand);
    assert_eq!(pc.options.len(), 3);
    assert!(state.pending_collect_resume.is_some());
    assert!(state.player.exhaust.is_empty(), "nothing auto-played yet");

    resolve(&mut state, 0);
    assert_at_rest(&state);
    assert_eq!(state.player.hand.len(), 1, "the Stratagem pick");
    assert_eq!(
        state.player.exhaust.len(),
        1,
        "Havoc's force-exhausted card"
    );
    assert_eq!(state.player.draw.len(), 1);
    assert!(state.player.discard.contains(&havoc));
}

/// Cascade collects one card, reshuffles and prompts, then collects the rest
/// after the pick and plays all three in order.
#[test]
fn cascade_reshuffle_select_prompts_mid_collect() {
    let mut state = fresh();
    let first = new_card(&mut state, CardId::StrikeIronclad);
    state.player.draw.push(first);
    for id in [CardId::DefendIronclad, CardId::Bash, CardId::IronWave] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let cascade = new_card(&mut state, CardId::Cascade);
    state.player.hand.push(cascade);

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("Stratagem grid prompts");
    assert_eq!(pc.kind, ChoiceKind::AddFromDrawToHand);
    assert!(
        state.player.play.contains(&first),
        "collected before the shuffle"
    );

    resolve(&mut state, 0);
    assert_at_rest(&state);
    assert_eq!(state.player.hand.len(), 1, "the Stratagem pick");
    assert!(
        state.player.draw.is_empty(),
        "two more collected after the pick"
    );
    assert!(state.player.discard.contains(&first));
    assert!(state.player.discard.contains(&cascade));
    assert_eq!(state.player.discard.len(), 4);
}

/// Havoc auto-plays a second Havoc, whose reshuffle prompts. After the pick it
/// auto-plays an Armaments whose upgrade select prompts too. The tails then
/// finish inner to outer: Armaments, the inner Havoc, the outer Havoc.
#[test]
fn nested_havoc_reshuffle_then_inner_select() {
    let mut state = fresh();
    for _ in 0..2 {
        let uid = new_card(&mut state, CardId::Armaments);
        state.player.discard.push(uid);
    }
    let inner = new_card(&mut state, CardId::Havoc);
    state.player.draw.push(inner);
    let outer = new_card(&mut state, CardId::Havoc);
    state.player.hand.push(outer);
    for _ in 0..2 {
        let uid = new_card(&mut state, CardId::DefendIronclad);
        state.player.hand.push(uid);
    }

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("Stratagem grid prompts");
    assert_eq!(pc.kind, ChoiceKind::AddFromDrawToHand);

    resolve(&mut state, 0);
    let pc = state.pending_choice.as_ref().expect("Armaments prompts");
    assert_eq!(pc.kind, ChoiceKind::UpgradeInHand);
    assert_eq!(pc.options.len(), 3, "two Defends and the picked Armaments");

    resolve(&mut state, 0);
    assert_at_rest(&state);
    let auto_played: Vec<_> = state
        .player
        .exhaust
        .iter()
        .map(|u| state.cards[u.0.get() as usize - 1].card_id)
        .collect();
    assert_eq!(auto_played, vec![CardId::Armaments, CardId::Havoc]);
    assert!(state.player.discard.contains(&outer));
    assert_eq!(state.cards.iter().filter(|c| c.upgraded).count(), 1);
}

/// Distilled Chaos on an empty draw pile prompts the same way, and its
/// `AfterPotionUsed` waits for the three auto-plays.
#[test]
fn distilled_chaos_reshuffle_select_prompts() {
    sts2_content::potions::dispatch::ensure_installed();
    let mut state = fresh();
    for id in [
        CardId::StrikeIronclad,
        CardId::DefendIronclad,
        CardId::Bash,
        CardId::IronWave,
    ] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    state
        .player
        .potions
        .push(sts2_core::potions::PotionKind::DistilledChaos);
    state.player.potion_slots = 3;

    let _ = sts2_core::apply(
        &mut state,
        sts2_core::Action::DrinkPotion {
            potion_slot: 0,
            target: None,
        },
    );
    let pc = state
        .pending_choice
        .as_ref()
        .expect("Stratagem grid prompts");
    assert_eq!(pc.kind, ChoiceKind::AddFromDrawToHand);
    assert!(state.pending_potion_hook_resume.is_some(), "the hook waits");

    resolve(&mut state, 0);
    assert_at_rest(&state);
    assert!(state.pending_potion_hook_resume.is_none());
    assert_eq!(state.player.hand.len(), 1, "the Stratagem pick");
    assert!(state.player.draw.is_empty(), "three auto-played");
    assert_eq!(state.player.discard.len(), 3);
}

/// A hand-played Havoc has no outer auto-play resume. Its reshuffle prompts;
/// after the pick it auto-plays an Armaments whose upgrade select prompts and
/// parks that Armaments' tail. `resume_collect` used to overwrite the fresh
/// resume with the (absent) outer one, so the Armaments never finished: it
/// stayed on Play, un-exhausted. Random-loadout Ironclad v0.111.0 seed 1165.
#[test]
fn hand_havoc_reshuffle_then_collected_select_keeps_its_resume() {
    let mut state = fresh();
    for _ in 0..2 {
        let uid = new_card(&mut state, CardId::Armaments);
        state.player.discard.push(uid);
    }
    let havoc = new_card(&mut state, CardId::Havoc);
    state.player.hand.push(havoc);
    for _ in 0..2 {
        let uid = new_card(&mut state, CardId::DefendIronclad);
        state.player.hand.push(uid);
    }

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("Stratagem grid prompts");
    assert_eq!(pc.kind, ChoiceKind::AddFromDrawToHand);

    resolve(&mut state, 0);
    let pc = state.pending_choice.as_ref().expect("Armaments prompts");
    assert_eq!(pc.kind, ChoiceKind::UpgradeInHand);
    assert!(
        state.pending_autoplay_resume.is_some(),
        "the Armaments' tail is parked"
    );

    resolve(&mut state, 0);
    assert_at_rest(&state);
    let exhausted: Vec<_> = state
        .player
        .exhaust
        .iter()
        .map(|u| state.cards[u.0.get() as usize - 1].card_id)
        .collect();
    assert_eq!(exhausted, vec![CardId::Armaments], "Havoc's card finished");
    assert!(state.player.discard.contains(&havoc));
    assert_eq!(state.cards.iter().filter(|c| c.upgraded).count(), 1);
}
