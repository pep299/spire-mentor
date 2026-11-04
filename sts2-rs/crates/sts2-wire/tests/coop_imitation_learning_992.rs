//! GH #992: Imitation Learning on a rebuilt co-op board. The wire names the
//! teammate the power copies by seat (`applier_seat`, C# `PlayerTarget`), so a
//! board read mid-fight keeps copying that teammate's Power cards.

use serde_json::{json, Value};
use sts2_core::Action;
use sts2_wire::state_from_json;

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

fn card(id: &str, i: u32) -> Value {
    json!({"id": id, "stable_idx": i, "kind": "card", "upgraded": false})
}

/// The fixture's co-op board with `power` on us; the teammate holds Inflame.
fn board(power: Value) -> Value {
    install();
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: Value = serde_json::from_str(&text).expect("fixture parses");
    v["game_version"] = json!("0.111.0");
    v["player"]["energy"] = json!(3);
    v["player"]["powers"] = json!([power]);
    v["player"]["relics"] = json!([]);
    v["deck"]["hand"] = json!([]);
    v["deck"]["draw_pile"] = (10..20).map(|i| card("DEFEND_IRONCLAD", i)).collect();
    v["deck"]["discard_pile"] = json!([]);
    v["deck"]["pinned_top"] = json!([]);
    v["coop_full"] = json!(true);
    v["allies"] = json!([{
        "class": "ironclad", "hp": 60, "max_hp": 70, "block": 0,
        "energy": 3, "max_energy": 3, "powers": [], "ended_turn": false,
        "hand": [card("INFLAME", 30)],
        "draw_pile": (40..50).map(|i| card("DEFEND_IRONCLAD", i)).collect::<Vec<_>>(),
        "discard_pile": [], "exhaust_pile": [], "play_pile": [],
        "relics": [],
    }]);
    v
}

fn power<'a>(powers: &'a Value, name: &str) -> Option<&'a Value> {
    powers
        .as_array()
        .expect("powers")
        .iter()
        .find(|p| p["name"] == json!(name))
}

#[test]
fn a_rebuilt_imitation_learning_copies_the_teammates_power_card() {
    let v = board(json!({"name": "IMITATION_LEARNING", "amount": 2, "applier_seat": 1}));
    let mut st = state_from_json(&v).expect("builds");
    let before = sts2_wire::emit::combat_state_to_value(&st).expect("emits");
    let il = power(&before["player"]["powers"], "IMITATION_LEARNING").expect("ours");
    assert_eq!(il["applier_seat"], json!(1), "the bound seat round-trips");

    let _ = sts2_core::coop::apply_as(
        &mut st,
        0,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    let after = sts2_wire::emit::combat_state_to_value(&st).expect("emits");
    let ours = &after["player"]["powers"];
    assert_eq!(
        power(ours, "STRENGTH").expect("our clone")["amount"],
        json!(2)
    );
    let il = power(ours, "IMITATION_LEARNING").expect("still up");
    assert_eq!(il["amount"], json!(1));
    assert_eq!(il["applier_seat"], json!(1));
}
