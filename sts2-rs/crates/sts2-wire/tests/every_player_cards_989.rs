//! GH #989: cards whose C# loops over `GetTeammatesOf(Owner.Creature)` reach
//! every living player, the teammate as well as the owner.

use serde_json::json;
use sts2_core::{apply, Action};
use sts2_wire::state_from_json;

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

fn card(id: &str, i: u32) -> serde_json::Value {
    json!({"id": id, "stable_idx": i, "kind": "card", "upgraded": false})
}

/// The fixture's Ironclad holding only `id`, with an Ironclad teammate seat,
/// as a whole co-op fight. Returns the board as emitted after playing it.
fn after_playing(id: &str) -> serde_json::Value {
    install();
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    v["player"]["block"] = json!(0);
    v["player"]["powers"] = json!([]);
    v["player"]["relics"] = json!([]);
    v["deck"]["hand"] = json!([card(id, 1)]);
    v["deck"]["draw_pile"] = (10..20).map(|i| card("DEFEND_IRONCLAD", i)).collect();
    v["deck"]["discard_pile"] = json!([]);
    v["deck"]["pinned_top"] = json!([]);
    v["encounter"][0]["powers"] = json!([]);
    v["coop_full"] = json!(true);
    v["allies"] = json!([{
        "class": "ironclad", "hp": 60, "max_hp": 70, "block": 0,
        "energy": 1, "max_energy": 3, "powers": [], "ended_turn": true,
        "hand": [],
        "draw_pile": (40..50).map(|i| card("DEFEND_IRONCLAD", i)).collect::<Vec<_>>(),
        "discard_pile": [], "exhaust_pile": [], "play_pile": [],
        "relics": [],
    }]);
    let mut st = state_from_json(&v).expect("builds");
    let target = (id == "OUTRAGE").then_some(0);
    let _ = apply(
        &mut st,
        Action::PlayCard {
            hand_index: 0,
            target,
        },
    );
    sts2_wire::emit::combat_state_to_value(&st).expect("emits")
}

fn ids(pile: &serde_json::Value) -> Vec<String> {
    pile.as_array()
        .expect("a pile")
        .iter()
        .map(|c| c["id"].as_str().unwrap_or_default().to_owned())
        .collect()
}

#[test]
fn outrage_clones_itself_into_every_players_discard() {
    let out = after_playing("OUTRAGE");
    assert_eq!(ids(&out["deck"]["discard_pile"]), ["OUTRAGE", "OUTRAGE"]);
    assert_eq!(ids(&out["allies"][0]["discard_pile"]), ["OUTRAGE"]);
}

#[test]
fn huddle_up_draws_for_the_teammate_too() {
    let out = after_playing("HUDDLE_UP");
    assert_eq!(ids(&out["deck"]["hand"]).len(), 2);
    assert_eq!(ids(&out["allies"][0]["hand"]).len(), 2);
}

#[test]
fn blade_symphony_gives_the_teammate_shivs() {
    let out = after_playing("BLADE_SYMPHONY");
    assert_eq!(ids(&out["deck"]["hand"]), ["SHIV", "SHIV"]);
    assert_eq!(ids(&out["allies"][0]["hand"]), ["SHIV", "SHIV"]);
}

#[test]
fn rally_blocks_the_teammate_too() {
    let out = after_playing("RALLY");
    assert_eq!(out["player"]["block"], json!(12));
    assert_eq!(out["allies"][0]["block"], json!(12));
}

#[test]
fn energy_surge_energizes_the_teammate_too() {
    let out = after_playing("ENERGY_SURGE");
    assert_eq!(out["allies"][0]["energy"], json!(3));
}

#[test]
fn plot_gives_the_teammate_draw_next_turn() {
    let out = after_playing("PLOT");
    let powers = out["allies"][0]["powers"].to_string();
    assert!(powers.contains("DRAW_CARDS_NEXT_TURN"), "{powers}");
}
