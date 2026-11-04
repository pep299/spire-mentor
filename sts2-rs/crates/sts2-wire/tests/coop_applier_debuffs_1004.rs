//! GH #1004: enemy debuffs that answer only the players who did NOT apply
//! them (Flanking, Knockdown, Tag Team). The wire names the applier's seat
//! (`applier_seat`), so a teammate's debuff helps our hits.

use serde_json::{json, Value};
use sts2_core::{apply, Action};
use sts2_wire::state_from_json;

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

fn card(id: &str, i: u32) -> Value {
    json!({"id": id, "stable_idx": i, "kind": "card", "upgraded": false})
}

/// The fixture's co-op board with `debuff` on the first enemy, our hand only
/// `id`. Returns the board before and after playing it at that enemy.
fn play_into(id: &str, debuff: Value) -> (Value, Value) {
    install();
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: Value = serde_json::from_str(&text).expect("fixture parses");
    v["game_version"] = json!("0.111.0");
    v["player"]["energy"] = json!(3);
    v["player"]["powers"] = json!([]);
    v["player"]["relics"] = json!([]);
    v["deck"]["hand"] = json!([card(id, 1)]);
    v["deck"]["draw_pile"] = (10..20).map(|i| card("DEFEND_IRONCLAD", i)).collect();
    v["deck"]["discard_pile"] = json!([]);
    v["deck"]["pinned_top"] = json!([]);
    v["encounter"][0]["powers"] = json!([debuff]);
    v["encounter"][0]["block"] = json!(0);
    v["encounter"][0]["hp"] = json!(40);
    v["encounter"][0]["max_hp"] = json!(40);
    v["coop_full"] = json!(true);
    v["allies"] = json!([{
        "class": "silent", "hp": 60, "max_hp": 70, "block": 0,
        "energy": 0, "max_energy": 3, "powers": [], "ended_turn": true,
        "hand": [],
        "draw_pile": (40..50).map(|i| card("DEFEND_SILENT", i)).collect::<Vec<_>>(),
        "discard_pile": [], "exhaust_pile": [], "play_pile": [],
        "relics": [],
    }]);
    let mut st = state_from_json(&v).expect("builds");
    let before = sts2_wire::emit::combat_state_to_value(&st).expect("emits");
    let _ = apply(
        &mut st,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    let after = sts2_wire::emit::combat_state_to_value(&st).expect("emits");
    (before, after)
}

fn lost(id: &str, debuff: Value) -> i64 {
    let (_, after) = play_into(id, debuff);
    40 - after["encounter"][0]["hp"].as_i64().expect("hp")
}

#[test]
fn a_teammates_flanking_doubles_our_hit() {
    let flanking = json!({"name": "FLANKING", "amount": 2, "applier_seat": 1});
    assert_eq!(lost("STRIKE_IRONCLAD", flanking), 12);
}

#[test]
fn our_own_flanking_leaves_our_hit_alone() {
    let flanking = json!({"name": "FLANKING", "amount": 2});
    assert_eq!(lost("STRIKE_IRONCLAD", flanking), 6);
}

#[test]
fn a_teammates_knockdown_multiplies_our_hit() {
    let knockdown = json!({"name": "KNOCKDOWN", "amount": 3, "applier_seat": 1});
    assert_eq!(lost("STRIKE_IRONCLAD", knockdown), 18);
}

#[test]
fn a_teammates_tag_team_plays_our_attack_again_once() {
    let tag_team = json!({"name": "TAG_TEAM", "amount": 1, "applier_seat": 1});
    let (_, after) = play_into("STRIKE_IRONCLAD", tag_team);
    assert_eq!(40 - after["encounter"][0]["hp"].as_i64().expect("hp"), 12);
    let powers = after["encounter"][0]["powers"].to_string();
    assert!(!powers.contains("TAG_TEAM"), "spent: {powers}");
}

#[test]
fn our_own_tag_team_does_not_replay_our_attack() {
    let tag_team = json!({"name": "TAG_TEAM", "amount": 1});
    assert_eq!(lost("STRIKE_IRONCLAD", tag_team), 6);
}

#[test]
fn the_applier_seat_survives_a_round_trip() {
    let flanking = json!({"name": "FLANKING", "amount": 2, "applier_seat": 1});
    let (before, _) = play_into("STRIKE_IRONCLAD", flanking);
    assert_eq!(
        before["encounter"][0]["powers"][0]["applier_seat"],
        json!(1)
    );
}

#[test]
fn flanking_a_teammate_plays_in_the_engine_doubles_our_next_hit() {
    install();
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: Value = serde_json::from_str(&text).expect("fixture parses");
    v["game_version"] = json!("0.111.0");
    v["player"]["energy"] = json!(3);
    v["player"]["powers"] = json!([]);
    v["player"]["relics"] = json!([]);
    v["deck"]["hand"] = json!([card("STRIKE_IRONCLAD", 1)]);
    v["deck"]["draw_pile"] = (10..20).map(|i| card("DEFEND_IRONCLAD", i)).collect();
    v["deck"]["discard_pile"] = json!([]);
    v["deck"]["pinned_top"] = json!([]);
    v["encounter"][0]["powers"] = json!([]);
    v["encounter"][0]["block"] = json!(0);
    v["encounter"][0]["hp"] = json!(40);
    v["encounter"][0]["max_hp"] = json!(40);
    v["coop_full"] = json!(true);
    v["allies"] = json!([{
        "class": "silent", "hp": 60, "max_hp": 70, "block": 0,
        "energy": 3, "max_energy": 3, "powers": [], "ended_turn": false,
        "hand": [card("FLANKING", 30)],
        "draw_pile": (40..50).map(|i| card("DEFEND_SILENT", i)).collect::<Vec<_>>(),
        "discard_pile": [], "exhaust_pile": [], "play_pile": [],
        "relics": [],
    }]);
    let mut st = state_from_json(&v).expect("builds");
    let _ = sts2_core::coop::apply_as(
        &mut st,
        0,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    let _ = apply(
        &mut st,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    let after = sts2_wire::emit::combat_state_to_value(&st).expect("emits");
    let flanking = &after["encounter"][0]["powers"][0];
    assert_eq!(flanking["applier_seat"], json!(1), "{flanking}");
    assert_eq!(40 - after["encounter"][0]["hp"].as_i64().expect("hp"), 12);
}
