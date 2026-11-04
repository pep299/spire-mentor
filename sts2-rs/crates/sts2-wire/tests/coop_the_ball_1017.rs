//! GH #1017: The Ball's grown damage rides the card's `base_damage` across a
//! rebuilt co-op board, whoever holds it, and the card leaves for the
//! teammate after its play.
//!
//! The mod sends every card's `stats` (the DynamicVars' `BaseValue`s), and
//! The Ball's `damage` is the grown `DamageVar.BaseValue` (`TheBall.cs:55`);
//! the adapters map it onto `base_damage`.

use serde_json::json;
use sts2_core::{apply, Action, CombatState};
use sts2_wire::state_from_json;

fn install() {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    sts2_core::relic_effects::install_default_registry();
}

fn card(id: &str, i: u32) -> serde_json::Value {
    json!({"id": id, "stable_idx": i, "kind": "card", "upgraded": false})
}

fn ball(i: u32, damage: i32) -> serde_json::Value {
    json!({"id": "THE_BALL", "stable_idx": i, "kind": "card", "upgraded": false, "base_damage": damage})
}

/// A whole v0.111.0 co-op fight on the fixture board: our Ironclad holding a
/// Ball grown to 30, an Ironclad teammate with a Ball grown to 40 on top of
/// their draw pile.
fn board() -> CombatState {
    install();
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    v["game_version"] = json!("0.111.0");
    v["player"]["block"] = json!(0);
    v["player"]["powers"] = json!([]);
    v["player"]["relics"] = json!([]);
    v["deck"]["hand"] = json!([ball(1, 30)]);
    v["deck"]["draw_pile"] = (10..20).map(|i| card("DEFEND_IRONCLAD", i)).collect();
    v["deck"]["discard_pile"] = json!([]);
    v["deck"]["pinned_top"] = json!([]);
    v["encounter"][0]["powers"] = json!([]);
    v["encounter"][0]["block"] = json!(0);
    v["encounter"][0]["hp"] = json!(100);
    v["encounter"][0]["max_hp"] = json!(100);
    v["coop_full"] = json!(true);
    let mut mate_draw: Vec<_> = vec![ball(40, 40)];
    mate_draw.extend((41..50).map(|i| card("DEFEND_IRONCLAD", i)));
    v["allies"] = json!([{
        "class": "ironclad", "hp": 60, "max_hp": 70, "block": 0,
        "energy": 1, "max_energy": 3, "powers": [], "ended_turn": true,
        "hand": [],
        "draw_pile": mate_draw,
        "discard_pile": [], "exhaust_pile": [], "play_pile": [],
        "relics": [],
    }]);
    state_from_json(&v).expect("builds")
}

fn balls(pile: &serde_json::Value) -> Vec<i64> {
    pile.as_array()
        .expect("a pile")
        .iter()
        .filter(|c| c["id"] == "THE_BALL")
        .map(|c| c["base_damage"].as_i64().expect("base_damage"))
        .collect()
}

#[test]
fn a_grown_ball_round_trips_in_every_seat() {
    let st = board();
    let out = sts2_wire::emit::combat_state_to_value(&st).expect("emits");
    assert_eq!(balls(&out["deck"]["hand"]), [30]);
    assert_eq!(balls(&out["allies"][0]["draw_pile"]), [40]);
    let again = state_from_json(&out).expect("rebuilds");
    let out2 = sts2_wire::emit::combat_state_to_value(&again).expect("emits");
    assert_eq!(balls(&out2["deck"]["hand"]), [30]);
    assert_eq!(balls(&out2["allies"][0]["draw_pile"]), [40]);
}

#[test]
fn a_rebuilt_ball_hits_at_its_grown_damage_and_goes_to_the_teammate() {
    let mut st = board();
    let hp = st
        .creatures
        .iter()
        .find(|c| c.species.is_some())
        .expect("enemy")
        .current_hp;
    let _ = apply(
        &mut st,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    let after = st
        .creatures
        .iter()
        .find(|c| c.species.is_some())
        .expect("enemy");
    assert_eq!(hp - after.current_hp, 30);
    let out = sts2_wire::emit::combat_state_to_value(&st).expect("emits");
    assert!(balls(&out["deck"]["discard_pile"]).is_empty());
    let mut theirs = balls(&out["allies"][0]["draw_pile"]);
    theirs.sort_unstable();
    assert_eq!(theirs, [40, 40], "ours, grown to 40, joined theirs");
}
