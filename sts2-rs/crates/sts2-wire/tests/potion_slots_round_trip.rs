//! GH #893 item 4: the potion-belt capacity survives the wire.
//!
//! `player.potions` is NOT the slot count on every producer: the Rust emitter
//! lists only filled slots and the C# headless `PlayerSummary` filters empty
//! ones, while the live mod sends one entry per slot. So the capacity rides
//! in an explicit `player.potion_slots` key (C# `Player.MaxPotionCount`).
//! Before it, `state_from_dict` never set `player.potion_slots`, so a rebuilt
//! A4+ board (Tight Belt: 2 slots) filled potions as if it had 5.

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

#[test]
fn an_explicit_potion_slots_key_sets_the_belt_capacity() {
    let mut v = fixture();
    // An A7 board (Tight Belt): two slots, both empty.
    v["player"]["potions"] = serde_json::json!([null, null]);
    v["player"]["potion_slots"] = serde_json::json!(2);
    let st = state_from_json(&v).expect("builds");
    assert_eq!(st.player.potion_slots, 2);
    assert_eq!(st.player.effective_potion_slots(), 2);
}

#[test]
fn potion_slots_round_trip_through_the_emitter() {
    for slots in [2u8, 3, 5] {
        let mut st = state_from_json(&fixture()).expect("builds");
        st.player.potion_slots = slots;
        let v = combat_state_to_value(&st).expect("emits");
        assert_eq!(v["player"]["potion_slots"], i64::from(slots), "emitted");
        let back = state_from_json(&v).expect("re-parses");
        assert_eq!(back.player.potion_slots, slots, "round trip");
        assert_eq!(back.player.effective_potion_slots(), usize::from(slots));
    }
}

#[test]
fn an_absent_key_keeps_the_legacy_unset_sentinel() {
    let mut v = fixture();
    if let Some(p) = v["player"].as_object_mut() {
        p.remove("potion_slots");
    }
    let st = state_from_json(&v).expect("builds");
    assert_eq!(
        st.player.potion_slots, 0,
        "unset, as before the key existed"
    );
}

#[test]
fn a_bad_potion_slots_value_is_a_schema_error() {
    let mut v = fixture();
    v["player"]["potion_slots"] = serde_json::json!("two");
    assert!(state_from_json(&v).is_err());
    v["player"]["potion_slots"] = serde_json::json!(-1);
    assert!(state_from_json(&v).is_err());
}
