//! GH #1004: player powers that answer what ANOTHER player does in a co-op
//! fight. The teammate's powers wait in their seat, so each of these needed
//! its own co-op hook.

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

fn power(name: &str, amount: i32) -> Value {
    json!({"name": name, "amount": amount})
}

/// A whole v0.111.0 co-op fight on the fixture board: our Ironclad holding
/// only `id` and carrying `ours`, an Ironclad teammate carrying `theirs` and
/// holding `their_hand`. Plays `id` (at the first enemy when `target`) and
/// returns the board as emitted.
fn after_playing(id: &str, target: bool, ours: Value, theirs: Value, their_hand: &[&str]) -> Value {
    let st = played(id, target, ours, theirs, their_hand);
    sts2_wire::emit::combat_state_to_value(&st).expect("emits")
}

/// [`after_playing`]'s fight itself.
fn played(
    id: &str,
    target: bool,
    ours: Value,
    theirs: Value,
    their_hand: &[&str],
) -> sts2_core::CombatState {
    install();
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: Value = serde_json::from_str(&text).expect("fixture parses");
    v["game_version"] = json!("0.111.0");
    v["player"]["block"] = json!(0);
    v["player"]["energy"] = json!(3);
    v["player"]["powers"] = ours;
    v["player"]["relics"] = json!([]);
    v["deck"]["hand"] = json!([card(id, 1)]);
    v["deck"]["draw_pile"] = (10..20).map(|i| card("DEFEND_IRONCLAD", i)).collect();
    v["deck"]["discard_pile"] = json!([]);
    v["deck"]["pinned_top"] = json!([]);
    v["encounter"][0]["powers"] = json!([]);
    v["encounter"][0]["block"] = json!(0);
    v["coop_full"] = json!(true);
    v["allies"] = json!([{
        "class": "ironclad", "hp": 60, "max_hp": 70, "block": 0,
        "energy": 1, "max_energy": 3, "powers": theirs, "ended_turn": true,
        "hand": their_hand.iter().zip(30..).map(|(id, i)| card(id, i)).collect::<Vec<_>>(),
        "draw_pile": (40..50).map(|i| card("DEFEND_IRONCLAD", i)).collect::<Vec<_>>(),
        "discard_pile": [], "exhaust_pile": [], "play_pile": [],
        "relics": [],
    }]);
    let mut st = state_from_json(&v).expect("builds");
    let _ = apply(
        &mut st,
        Action::PlayCard {
            hand_index: 0,
            target: target.then_some(0),
        },
    );
    st
}

fn power_amount(powers: &Value, name: &str) -> Option<i64> {
    powers
        .as_array()?
        .iter()
        .find(|p| p["name"] == name || p["id"] == name)
        .and_then(|p| p["amount"].as_i64())
}

#[test]
fn beacon_of_hope_gives_the_teammate_half_our_block() {
    let out = after_playing(
        "DEFEND_IRONCLAD",
        false,
        json!([power("BEACON_OF_HOPE", 1)]),
        json!([]),
        &[],
    );
    assert_eq!(out["player"]["block"], json!(5));
    assert_eq!(out["allies"][0]["block"], json!(2), "half of 5, truncated");
}

#[test]
fn two_beacons_send_block_back_only_once() {
    // C#: we gain 5 and give 2.5 (stored 2); their Beacon gives 1.25 back
    // (stored 1), and ours, still giving, does not answer that.
    let out = after_playing(
        "DEFEND_IRONCLAD",
        false,
        json!([power("BEACON_OF_HOPE", 1)]),
        json!([power("BEACON_OF_HOPE", 1)]),
        &[],
    );
    assert_eq!(out["player"]["block"], json!(6));
    assert_eq!(out["allies"][0]["block"], json!(2));
}

#[test]
fn sneaky_blocks_when_someone_else_attacks() {
    let out = after_playing(
        "STRIKE_IRONCLAD",
        true,
        json!([]),
        json!([power("SNEAKY", 2)]),
        &[],
    );
    assert_eq!(out["allies"][0]["block"], json!(2));
}

#[test]
fn sneaky_ignores_its_owners_own_attacks() {
    let out = after_playing(
        "STRIKE_IRONCLAD",
        true,
        json!([power("SNEAKY", 2)]),
        json!([]),
        &[],
    );
    assert_eq!(out["player"]["block"], json!(0));
}

#[test]
fn a_teammates_underworld_dooms_what_we_hit() {
    let out = after_playing(
        "STRIKE_IRONCLAD",
        true,
        json!([]),
        json!([power("UNDERWORLD", 1)]),
        &[],
    );
    assert_eq!(
        power_amount(&out["encounter"][0]["powers"], "DOOM"),
        Some(6)
    );
}

#[test]
fn our_own_underworld_does_not_doom_our_hits() {
    let out = after_playing(
        "STRIKE_IRONCLAD",
        true,
        json!([power("UNDERWORLD", 1)]),
        json!([]),
        &[],
    );
    assert_eq!(power_amount(&out["encounter"][0]["powers"], "DOOM"), None);
}

#[test]
fn hammer_time_makes_the_teammate_forge_too() {
    let out = after_playing(
        "BULWARK",
        false,
        json!([power("HAMMER_TIME", 1)]),
        json!([]),
        &[],
    );
    let hand = out["allies"][0]["hand"].to_string();
    assert!(hand.contains("SOVEREIGN_BLADE"), "{hand}");
}

#[test]
fn tank_guards_the_teammate() {
    let out = after_playing("TANK_CARD", false, json!([]), json!([]), &[]);
    assert_eq!(
        power_amount(&out["allies"][0]["powers"], "GUARDED"),
        Some(1)
    );
}

#[test]
fn a_second_tank_guards_no_one_again() {
    let out = after_playing(
        "TANK_CARD",
        false,
        json!([power("TANK", 1)]),
        json!([]),
        &[],
    );
    assert_eq!(power_amount(&out["allies"][0]["powers"], "GUARDED"), None);
}

#[test]
fn our_exhaust_makes_the_teammates_midnight_cheaper() {
    let out = after_playing("IMPERVIOUS", false, json!([]), json!([]), &["MIDNIGHT"]);
    let midnight = &out["allies"][0]["hand"][0];
    assert_eq!(midnight["id"], json!("MIDNIGHT"));
    assert_eq!(midnight["cost_override"], json!(11), "{midnight}");
}

#[test]
fn guarded_leaves_when_the_tank_who_gave_it_dies() {
    // C# `GuardedPower.AfterDeath`: `creature == Applier` removes it.
    let mut st = played("TANK_CARD", false, json!([]), json!([]), &[]);
    let tank = st.coop.seated;
    sts2_core::coop::drop_guarded_of(&mut st, tank);
    let out = sts2_wire::emit::combat_state_to_value(&st).expect("emits");
    assert_eq!(power_amount(&out["allies"][0]["powers"], "GUARDED"), None);
}
