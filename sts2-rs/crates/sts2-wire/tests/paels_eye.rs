//! Pael's Eye, end to end from a wire state.
//!
//! C# `PaelsEye.cs` (v0.111.0, same body in v0.107.1): `AnyCardsPlayedThisTurn`
//! is `CardPlaysFinished.Any(Actor == Owner && HappenedThisTurn &&
//! !CardPlay.IsAutoPlay)`, except that on `TurnNumber == 1` an owned
//! `WhisperingEarring` (melted or not) makes it true. `BeforeSideTurnEndEarly`
//! exhausts the hand and `ShouldTakeExtraTurn` grants the extra turn only when
//! it is false. Our `Relic.counter` bit 1 is that "any cards played" flag.

use sts2_wire::state_from_json_str;

fn install() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
}

fn card(id: &str, i: usize) -> serde_json::Value {
    serde_json::json!({"id": id, "stable_idx": i, "kind": "card", "upgraded": false})
}

fn fight(
    hand: &[&str],
    draw: &[&str],
    relics: serde_json::Value,
    powers: serde_json::Value,
) -> sts2_core::CombatState {
    let value = serde_json::json!({
        "wire_version": 1, "phase": "PlayerPlayPhase", "round": 1, "side": "Player",
        "is_elite": false, "turn": 1, "combat_id": 0, "rng": {"seed": 1, "ascension": 0},
        "player": {"class": "ironclad", "hp": 80, "max_hp": 80, "block": 0, "energy": 3,
                   "max_energy": 3, "energy_regen_per_turn": 3, "gold": 0, "relics": relics,
                   "powers": powers, "potions": []},
        "deck": {"hand": hand.iter().enumerate().map(|(i, id)| card(id, i + 1)).collect::<Vec<_>>(),
                 "draw_pile": draw.iter().enumerate().map(|(i, id)| card(id, 20 + i)).collect::<Vec<_>>(),
                 "discard_pile": [], "exhaust_pile": []},
        "encounter": [{"kind": "Nibbit", "uid": 1, "hp": 200, "max_hp": 200, "block": 0,
                       "powers": [], "intents": [{"kind": "Attack", "amount": 1, "times": 1}],
                       "intent_history": [], "is_dead": false, "is_escaped": false}],
    });
    state_from_json_str(&value.to_string()).unwrap()
}

fn eye(state: &sts2_core::CombatState) -> i32 {
    state
        .relics
        .iter()
        .find(|r| r.kind == sts2_core::RelicKind::PaelsEye)
        .expect("Pael's Eye")
        .counter
}

/// A hand play counts.
#[test]
fn paels_eye_hand_play_counts_as_played() {
    install();
    let mut state = fight(
        &["DEFEND_IRONCLAD", "DEFEND_IRONCLAD"],
        &["DEFEND_IRONCLAD"; 6],
        serde_json::json!([{"id": "PAELS_EYE", "counter": 0}]),
        serde_json::json!([]),
    );
    let _ = sts2_core::apply(
        &mut state,
        sts2_core::Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(eye(&state) & 0b10, 0b10, "a hand play sets the played bit");
}

/// Hellraiser auto-plays the Strikes drawn at turn start; those are
/// `IsAutoPlay`, so the turn still counts as one with no cards played and
/// Pael's Eye exhausts the hand and grants the extra turn (act23 Ironclad
/// seeds 2142 / 2019).
#[test]
fn paels_eye_ignores_auto_plays() {
    install();
    // Turn 1 already had a hand play (bit 1 set), so it ends normally.
    let mut state = fight(
        &["DEFEND_IRONCLAD"],
        &[
            "STRIKE_IRONCLAD",
            "DEFEND_IRONCLAD",
            "DEFEND_IRONCLAD",
            "DEFEND_IRONCLAD",
            "DEFEND_IRONCLAD",
            "DEFEND_IRONCLAD",
            "DEFEND_IRONCLAD",
        ],
        serde_json::json!([{"id": "PAELS_EYE", "counter": 0b10}]),
        serde_json::json!([{"name": "HELLRAISER", "amount": 1}]),
    );
    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert_eq!(state.turn_number(), 2, "on the next player turn");
    assert!(
        state.player.discard.len() + state.player.exhaust.len() >= 2,
        "Hellraiser auto-played the drawn Strike"
    );
    assert_eq!(eye(&state) & 0b10, 0, "an auto-play does not count");

    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert_eq!(
        state.combat_log.extra_turns_taken_this_combat, 1,
        "a turn of only auto-plays earns the extra turn"
    );
}

/// `AnyCardsPlayedThisTurn` is true on turn 1 when the owner has a Whispering
/// Earring (its turn-1 plays are all auto-plays): no exhaust, no extra turn.
#[test]
fn paels_eye_whispering_earring_turn_one_counts_as_played() {
    install();
    let mut state = fight(
        &["DEFEND_IRONCLAD", "DEFEND_IRONCLAD"],
        &["DEFEND_IRONCLAD"; 6],
        serde_json::json!([
            {"id": "PAELS_EYE", "counter": 0},
            {"id": "WHISPERING_EARRING", "counter": 0}
        ]),
        serde_json::json!([]),
    );
    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert!(state.player.exhaust.is_empty(), "hand not exhausted");
    assert_eq!(state.combat_log.extra_turns_taken_this_combat, 0);
}
