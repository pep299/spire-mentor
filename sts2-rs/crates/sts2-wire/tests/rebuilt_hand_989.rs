//! GH #989: a board rebuilt mid-turn keeps what C# keeps out of sight.
//!
//! Juggling counts this turn's Attacks in private data and copies the 3rd
//! into the hand; the wire carries the count as the Attack bucket of
//! `cards_played_this_turn`. A card Choices Paradox gave Retain arrives as
//! the `retain` keyword flag and stays in hand at turn end.

use serde_json::json;
use sts2_core::cards::CardId;
use sts2_core::{apply, Action, CombatState};
use sts2_wire::state_from_json;

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

/// The fixture's Ironclad (2 energy) against a Nibbit, holding a Strike with
/// `flags` and nothing else, over a draw pile of Defends (so the next hand
/// is drawn without a reshuffle).
fn board(flags: &serde_json::Value) -> serde_json::Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    let mut strike = v["deck"]["hand"][0].clone();
    strike["flags"] = flags.clone();
    v["deck"]["hand"] = json!([strike]);
    v["deck"]["draw_pile"] = (10..20)
        .map(|i| json!({"id": "DEFEND_IRONCLAD", "stable_idx": i, "kind": "card", "upgraded": false}))
        .collect();
    v["deck"]["discard_pile"] = json!([]);
    v["deck"]["pinned_top"] = json!([]);
    v
}

fn strikes_in_hand(st: &CombatState) -> usize {
    st.player
        .hand
        .iter()
        .filter(|u| st.cards[u.0.get() as usize - 1].card_id == CardId::StrikeIronclad)
        .count()
}

/// Play the hand's Strike after `attacks` Attacks this turn, with Juggling.
fn juggled_strike(attacks: u32) -> CombatState {
    install();
    let mut v = board(&json!([]));
    v["player"]["cards_played_this_turn"] = json!([attacks, 0, 0, 0, 0]);
    v["player"]["powers"] = json!([{"name": "JUGGLING", "amount": 1}]);
    let mut st = state_from_json(&v).expect("builds");
    let _ = apply(
        &mut st,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    st
}

#[test]
fn juggling_copies_the_third_attack_of_a_rebuilt_turn() {
    assert_eq!(strikes_in_hand(&juggled_strike(2)), 1, "the 3rd Attack");
    assert_eq!(strikes_in_hand(&juggled_strike(1)), 0, "only the 2nd");
}

/// Whether the hand's Strike is still in hand after the turn ends.
fn kept_through_turn_end(flags: &serde_json::Value) -> bool {
    install();
    let mut st = state_from_json(&board(flags)).expect("builds");
    let strike = st.player.hand[0];
    let _ = apply(&mut st, Action::EndTurn);
    st.player.hand.contains(&strike)
}

#[test]
fn a_retain_keyword_on_the_wire_keeps_the_card_in_hand() {
    assert!(kept_through_turn_end(&json!(["retain"])));
    assert!(!kept_through_turn_end(&json!([])));
}
