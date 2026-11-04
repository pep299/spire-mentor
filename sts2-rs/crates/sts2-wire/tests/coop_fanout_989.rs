//! GH #989: in a whole co-op fight a teammate hears what C# tells every
//! player. Their relics hear an enemy die (Gremlin Horn draws them a card),
//! and a Wriggler's WRIGGLE gives each player an Infection.

use serde_json::json;
use sts2_core::cards::CardId;
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

/// The fixture's Ironclad holding a Strike against `enemy`, with an Ironclad
/// teammate seat (`relics`, five Defends to draw), as a whole co-op fight.
fn coop_board(enemy: &serde_json::Value, relics: &[&str]) -> serde_json::Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    v["deck"]["hand"] = json!([card("STRIKE_IRONCLAD", 4)]);
    v["deck"]["draw_pile"] = (10..20).map(|i| card("DEFEND_IRONCLAD", i)).collect();
    v["deck"]["discard_pile"] = json!([]);
    v["deck"]["pinned_top"] = json!([]);
    v["encounter"] = json!([enemy]);
    v["coop_full"] = json!(true);
    v["allies"] = json!([{
        "class": "ironclad", "hp": 60, "max_hp": 70, "block": 0,
        "energy": 3, "max_energy": 3, "powers": [], "ended_turn": true,
        "hand": [card("STRIKE_IRONCLAD", 30)],
        "draw_pile": (40..45).map(|i| card("DEFEND_IRONCLAD", i)).collect::<Vec<_>>(),
        "discard_pile": [], "exhaust_pile": [], "play_pile": [],
        "relics": relics.iter().map(|r| json!({"id": r, "counter": null})).collect::<Vec<_>>(),
    }]);
    v
}

fn ally_cards(st: &CombatState, id: CardId) -> usize {
    let a = &st.allies[0];
    a.hand
        .iter()
        .chain(&a.draw)
        .chain(&a.discard)
        .filter(|u| st.cards[u.0.get() as usize - 1].card_id == id)
        .count()
}

/// The teammate's hand size and energy after our Strike kills a 1-HP Nibbit
/// beside a healthy one.
fn after_the_kill(relics: &[&str]) -> (usize, i32) {
    install();
    let nibbit = |uid: u32, hp: i32| {
        json!({"kind": "Nibbit", "uid": uid, "hp": hp, "max_hp": 32, "block": 0,
               "powers": [], "intents": [{"kind": "Attack", "amount": 9, "times": 1}],
               "intent_history": [0, 1], "is_dead": false, "is_escaped": false})
    };
    let mut v = coop_board(&nibbit(2, 1), relics);
    v["encounter"] = json!([nibbit(2, 1), nibbit(3, 30)]);
    let mut st = state_from_json(&v).expect("builds");
    let _ = apply(
        &mut st,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    (st.allies[0].hand.len(), st.allies[0].energy)
}

#[test]
fn a_teammates_gremlin_horn_hears_an_enemy_die() {
    assert_eq!(after_the_kill(&["GREMLIN_HORN"]), (2, 4));
    assert_eq!(after_the_kill(&[]), (1, 3));
}

#[test]
fn a_wriggle_gives_the_teammate_an_infection_too() {
    install();
    let wriggler = json!({
        "kind": "Wriggler", "uid": 2, "hp": 20, "max_hp": 20, "block": 0,
        "powers": [], "move_state": "WRIGGLE_MOVE", "intent_history": [],
        "intents": [{"kind": "Buff"}, {"kind": "StatusCard"}],
        "is_dead": false, "is_escaped": false,
    });
    let mut st = state_from_json(&coop_board(&wriggler, &[])).expect("builds");
    let _ = apply(&mut st, Action::EndTurn);
    assert_eq!(ally_cards(&st, CardId::Infection), 1);
}
