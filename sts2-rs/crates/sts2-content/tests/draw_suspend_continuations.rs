//! Card effects that keep going after a draw that suspended on a Stratagem
//! reshuffle grid-select, or whose drawn-card report is read after nested
//! on-draw riders ran.
//!
//! C# awaits `CardPileCmd.Draw` (and the `StratagemPower.AfterShuffle` select
//! inside it) before the rest of the card's `OnPlay` runs. Rust suspends the
//! draw on `pending_draw_resume`; whatever the card does after the draw must
//! wait for the resume.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid,
};

const V107: GameVersion = GameVersion::V0_107_1;
const V111: GameVersion = GameVersion::V0_111_0;

fn fresh(version: GameVersion) -> (CombatState, CreatureUid) {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
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
    state.player.energy = 5;
    state.player.max_energy = 3;
    state.player.character = Some(CharacterId::Ironclad);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    (state, CreatureUid(id))
}

fn new_card(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, false, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn push_hand(state: &mut CombatState, id: CardId) -> u8 {
    let uid = new_card(state, id);
    state.player.hand.push(uid);
    u8::try_from(state.player.hand.len() - 1).expect("u8")
}

fn add_player_power(state: &mut CombatState, raw: u32, kind: PowerKind, amount: i32) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(Power::new(pid, kind, amount));
    state.player.powers.push(PowerUid(pid));
}

fn resolve(state: &mut CombatState, idx: u8) {
    let _ = sts2_core::apply(
        state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![idx],
        },
    );
}

// ── Burning Pact after a suspended Dark Embrace draw ─────────────────────

/// Burning Pact with Havoc as the only other hand card: the forced exhaust
/// resolves inline, Dark Embrace draws 1 into an empty draw pile, and the
/// reshuffle opens Stratagem's grid. C# `BurningPact.OnPlay` (v0.111.0
/// BurningPact.cs:26-32) awaits the exhaust, and with it
/// `DarkEmbracePower.AfterCardExhausted`'s `Draw` (DarkEmbracePower.cs:47)
/// and the select inside it, before its own `Draw(2)`. So nothing is drawn
/// before the pick, and the pick, Dark Embrace's card and Burning Pact's two
/// cards all land: four cards (act23 Ironclad v0.111.0 seed 2076, step 649).
fn burning_pact_after_suspended_dark_embrace(version: GameVersion) {
    let (mut state, _) = fresh(version);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    add_player_power(&mut state, 51, PowerKind::DarkEmbrace, 1);
    for id in [
        CardId::StrikeIronclad,
        CardId::DefendIronclad,
        CardId::Bash,
        CardId::Anger,
        CardId::IronWave,
        CardId::TwinStrike,
    ] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let _ = push_hand(&mut state, CardId::Havoc);
    let bp = push_hand(&mut state, CardId::BurningPact);
    let _ = sts2_core::combat_loop::play_card(&mut state, bp, None);

    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert!(
        state.player.hand.is_empty(),
        "nothing is drawn before the Stratagem pick: {:?}",
        state.player.hand
    );

    resolve(&mut state, 0);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_draw_resume.is_none());
    assert_eq!(
        state.player.hand.len(),
        4,
        "Stratagem pick + Dark Embrace 1 + Burning Pact 2"
    );
    assert_eq!(state.player.draw.len(), 2, "6 reshuffled - pick - 3 drawn");
}

#[test]
fn burning_pact_draws_after_suspended_dark_embrace_draw_v0111() {
    burning_pact_after_suspended_dark_embrace(V111);
}

#[test]
fn burning_pact_draws_after_suspended_dark_embrace_draw_v0107() {
    burning_pact_after_suspended_dark_embrace(V107);
}

// ── Pillage reads the card ITS draw returned ─────────────────────────────

fn push_draw(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let uid = new_card(state, id);
    state.player.draw.push(uid);
    uid
}

/// Pillage under Hellraiser draws Pommel Strike, which Hellraiser auto-plays;
/// Pommel's own draw takes Battle Trance (a Skill). C# `Pillage.OnPlay` tests
/// the card ITS `CardPileCmd.Draw` returned (Pillage.cs:31-33) — Pommel, an
/// Attack — so it keeps drawing: Twin Strike (auto-played), then Defend.
/// Rust read `last_drawn_card` after the nested draw had overwritten it with
/// Battle Trance and stopped (act23 Ironclad v0.111.0 seed 2166, Mecha
/// Knight left at 1 hp).
fn pillage_reads_its_own_draw_through_nested_draw(version: GameVersion) {
    let (mut state, _enemy) = fresh(version);
    add_player_power(&mut state, 50, PowerKind::Hellraiser, 1);
    let _pommel = push_draw(&mut state, CardId::PommelStrike);
    let trance = push_draw(&mut state, CardId::BattleTrance);
    let _twin = push_draw(&mut state, CardId::TwinStrike);
    let defend = push_draw(&mut state, CardId::DefendIronclad);
    let p = push_hand(&mut state, CardId::Pillage);
    let _ = sts2_core::combat_loop::play_card(&mut state, p, Some(0));

    assert!(state.pending_choice.is_none());
    assert!(
        state.player.hand.contains(&trance),
        "Pommel drew Battle Trance"
    );
    assert!(
        state.player.hand.contains(&defend),
        "Pillage kept drawing past the nested draw to Defend: hand {:?}, draw {:?}",
        state.player.hand,
        state.player.draw
    );
    assert!(state.player.draw.is_empty());
}

#[test]
fn pillage_reads_its_own_draw_through_nested_draw_v0111() {
    pillage_reads_its_own_draw_through_nested_draw(V111);
}

#[test]
fn pillage_reads_its_own_draw_through_nested_draw_v0107() {
    pillage_reads_its_own_draw_through_nested_draw(V107);
}

// ── Pillage stays in its loop across a suspended draw ────────────────────

/// Pillage draws Iron Wave, then its next draw reshuffles an all-Attack
/// discard into a Stratagem select. C# is still inside the awaited `Draw`
/// in Pillage's do/while (Pillage.cs:28-33): after the pick that draw takes
/// an Attack and the loop goes on until the piles run dry. Rust broke out
/// of the loop on the suspend (no card reported yet) and drew only the one
/// resumed card (act23 Ironclad v0.111.0 seed 2114, step 1003).
fn pillage_loops_across_stratagem_suspend(version: GameVersion) {
    let (mut state, _enemy) = fresh(version);
    add_player_power(&mut state, 50, PowerKind::Stratagem, 1);
    let iron_wave = push_draw(&mut state, CardId::IronWave);
    for id in [CardId::Anger, CardId::TwinStrike, CardId::Bash] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let p = push_hand(&mut state, CardId::Pillage);
    let _ = sts2_core::combat_loop::play_card(&mut state, p, Some(0));
    let pc = state.pending_choice.as_ref().expect("Stratagem grid opens");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert_eq!(state.player.hand.as_slice(), &[iron_wave]);

    resolve(&mut state, 0);
    assert!(state.pending_choice.is_none());
    assert!(state.pending_draw_resume.is_none());
    assert_eq!(
        state.player.hand.len(),
        4,
        "Iron Wave + the pick + both remaining Attacks: hand {:?}",
        state.player.hand
    );
    assert!(state.player.draw.is_empty());
}

#[test]
fn pillage_loops_across_stratagem_suspend_v0111() {
    pillage_loops_across_stratagem_suspend(V111);
}

#[test]
fn pillage_loops_across_stratagem_suspend_v0107() {
    pillage_loops_across_stratagem_suspend(V107);
}
