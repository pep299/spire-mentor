//! GH #989: a seat's combat-history counters survive the wire.
//!
//! C# reads some card values from `CombatManager.History`, which a board
//! rebuilt mid-fight does not have: Tear Asunder hits once more for every
//! hit that took its owner's HP this combat, and Gang Up deals 5 more for
//! every powered hit a teammate landed on the target this turn. Without the
//! counters a recorded co-op fight's Tear Asunder swung once instead of four
//! times, and its Gang Up dealt 7 instead of 30.

use serde_json::json;
use sts2_wire::emit::combat_state_to_value;
use sts2_wire::state_from_json;

fn fixture() -> serde_json::Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    serde_json::from_str(&text).expect("fixture parses")
}

/// The fixture with one whole Silent teammate seat carrying `history`.
fn with_teammate(history: &serde_json::Value) -> serde_json::Value {
    let mut v = fixture();
    let mut ally = json!({
        "class": "silent", "hp": 50, "max_hp": 70, "block": 0,
        "energy": 3, "max_energy": 3, "powers": [], "hand": [],
    });
    for (k, val) in history.as_object().expect("object") {
        ally[k] = val.clone();
    }
    v["allies"] = json!([ally]);
    v
}

#[test]
fn our_counters_reach_the_fight_and_round_trip() {
    let mut v = fixture();
    v["player"]["unblocked_hits_this_combat"] = json!(3);
    v["player"]["powered_hits_this_turn"] = json!([{"enemy": 0, "hits": 2}]);
    let st = state_from_json(&v).expect("builds");
    assert_eq!(st.combat_log.unblocked_damage_received_this_combat, 3);
    let target = st.enemy_uids[0];
    assert_eq!(
        st.player.powered_hits_by_target_this_turn.as_slice(),
        &[(target, 2)]
    );
    let out = combat_state_to_value(&st).expect("emits");
    assert_eq!(out["player"]["unblocked_hits_this_combat"], 3);
    assert_eq!(
        out["player"]["powered_hits_this_turn"],
        json!([{"enemy": 0, "hits": 2}])
    );
}

#[test]
fn a_teammates_counters_reach_their_seat_and_round_trip() {
    let v = with_teammate(&json!({
        "unblocked_hits_this_combat": 2,
        "powered_hits_this_turn": [{"enemy": 0, "hits": 3}],
    }));
    let st = state_from_json(&v).expect("builds");
    assert_eq!(
        st.coop.stash[0]
            .combat_log
            .unblocked_damage_received_this_combat,
        2
    );
    assert_eq!(
        st.allies[0].powered_hits_by_target_this_turn.as_slice(),
        &[(st.enemy_uids[0], 3)]
    );
    let out = combat_state_to_value(&st).expect("emits");
    assert_eq!(out["allies"][0]["unblocked_hits_this_combat"], 2);
    assert_eq!(
        out["allies"][0]["powered_hits_this_turn"],
        json!([{"enemy": 0, "hits": 3}])
    );
}

#[test]
fn absent_counters_are_zero_and_not_emitted() {
    let st = state_from_json(&fixture()).expect("builds");
    assert_eq!(st.combat_log.unblocked_damage_received_this_combat, 0);
    assert!(st.player.powered_hits_by_target_this_turn.is_empty());
    let out = combat_state_to_value(&st).expect("emits");
    assert!(out["player"].get("unblocked_hits_this_combat").is_none());
    assert!(out["player"].get("powered_hits_this_turn").is_none());
}

#[test]
fn bad_counters_are_schema_errors() {
    for (key, bad) in [
        ("unblocked_hits_this_combat", json!(-1)),
        ("powered_hits_this_turn", json!([{"enemy": 0}])),
        ("powered_hits_this_turn", json!({"enemy": 0, "hits": 1})),
    ] {
        let mut v = fixture();
        v["player"][key] = bad;
        assert!(state_from_json(&v).is_err(), "{key}");
    }
}

/// GH #812: the extra-turn count survives the wire, so a board rebuilt right
/// after a PaelsEye / Ambergris extra turn keeps C#'s `TurnNumber` (round + 1)
/// instead of falling back to turn = round.
#[test]
fn extra_turns_round_trip_and_keep_the_turn_number_ahead_of_the_round() {
    let mut v = fixture();
    v["player"]["extra_turns_taken_this_combat"] = json!(1);
    let st = state_from_json(&v).expect("builds");
    assert_eq!(st.combat_log.extra_turns_taken_this_combat, 1);
    assert_eq!(st.turn_number(), st.round + 1);
    let out = combat_state_to_value(&st).expect("emits");
    assert_eq!(out["player"]["extra_turns_taken_this_combat"], 1);
    let again = state_from_json(&out).expect("rebuilds");
    assert_eq!(again.turn_number(), st.turn_number());

    let v = with_teammate(&json!({"extra_turns_taken_this_combat": 2}));
    let st = state_from_json(&v).expect("builds");
    assert_eq!(st.coop.stash[0].combat_log.extra_turns_taken_this_combat, 2);
    let out = combat_state_to_value(&st).expect("emits");
    assert_eq!(out["allies"][0]["extra_turns_taken_this_combat"], 2);
}

#[test]
fn absent_extra_turns_are_zero_and_not_emitted() {
    let st = state_from_json(&fixture()).expect("builds");
    assert_eq!(st.combat_log.extra_turns_taken_this_combat, 0);
    assert_eq!(st.turn_number(), st.round);
    let out = combat_state_to_value(&st).expect("emits");
    assert!(out["player"].get("extra_turns_taken_this_combat").is_none());
}

#[test]
fn bad_extra_turns_are_schema_errors() {
    for bad in [json!(-1), json!(70_000), json!("1")] {
        let mut v = fixture();
        v["player"]["extra_turns_taken_this_combat"] = bad.clone();
        assert!(state_from_json(&v).is_err(), "{bad}");
    }
}
