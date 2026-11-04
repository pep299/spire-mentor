//! The fair evaluation (`mcts_recommend(fair=True)`) under an honest
//! hidden-information mode must not see hidden information through the ROOT
//! action either: a draw the action makes is not the true next card, and a
//! target or card it rolls does not follow the state's real RNG cursors.
//!
//! Before the fix each action was applied to the raw state and only the
//! search AFTER it was honest, so a fair row priced the realized draw.
//! Reported against the manoj fork (`257fc6817`, `1edb23cf0`, review F3 of
//! its A4d step); fixed here with the engine's own forced-root search.

use serde_json::{json, Value};
use sts2_core::CombatState;
use sts2_wire::emit::mcts::{recommend_value, RecommendParams};
use sts2_wire::state_from_json_str;

/// Shrug It Off and a Defend in hand, Bludgeon and Wound to draw (in that
/// order: a concrete, not a belief, pile), an enemy attacking for 25 at 20 HP.
fn fixture() -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/fair_root_draw.json"),
        )
        .expect("fixture readable"),
    )
    .expect("fixture parses")
}

fn params(sims: u32, hidden: &str) -> RecommendParams {
    RecommendParams::from_config(
        &json!({"sims": sims, "seed": 5, "fair": true, "threads": 1, "hidden_info": hidden}),
        "test",
    )
    .expect("params")
}

/// `(action, q, visits)` per fair row, sorted by action.
fn rows(v: &Value, sims: u32, hidden: &str) -> Vec<(String, f64, u64)> {
    let state = state_from_json_str(&v.to_string()).expect("rebuilds");
    state_rows(&state, sims, hidden)
}

fn state_rows(state: &CombatState, sims: u32, hidden: &str) -> Vec<(String, f64, u64)> {
    let out = recommend_value(state, &params(sims, hidden), true);
    let mut rows: Vec<(String, f64, u64)> = out["actions"]
        .as_array()
        .expect("actions")
        .iter()
        .map(|a| {
            (
                a["action"].to_string(),
                a["q"].as_f64().expect("q"),
                a["visits"].as_u64().expect("visits"),
            )
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows
}

/// The fixture's state, and the same state with its two unseen draw-pile
/// cards in the other physical order (the SAME card instances, so only the
/// hidden order differs, not the uid labelling the belief canonicalizes by).
fn both_orders() -> (CombatState, CombatState) {
    let a = state_from_json_str(&fixture().to_string()).expect("rebuilds");
    let mut b = a.clone();
    b.player.draw.reverse();
    (a, b)
}

/// Shrug It Off draws one of two unseen cards. Under BeliefDraw its fair row
/// must be the same whichever order the state holds them in; before the fix
/// it drew the true top card (Bludgeon, or Wound when reversed) and its Q
/// moved by ~0.05.
#[test]
fn fair_rows_do_not_depend_on_the_hidden_draw_order() {
    let (a, b) = both_orders();
    let ra = state_rows(&a, 300, "beliefdraw");
    assert!(ra.iter().all(|r| r.2 > 0), "every action searched: {ra:?}");
    assert_eq!(ra, state_rows(&b, 300, "beliefdraw"));
}

/// An action that ends the combat is priced on the same scale as the others:
/// the same visits as every other row (the forced search's budget, less the
/// simulations that materialize the root), not one synthesized visit. (End
/// Turn here loses: the enemy hits for 25 at 20 HP.)
#[test]
fn a_combat_ending_action_gets_the_whole_budget() {
    let (a, _) = both_orders();
    let r = state_rows(&a, 300, "beliefdraw");
    let end = r
        .iter()
        .find(|r| r.0.contains("end_turn"))
        .expect("end turn row");
    assert!(r.iter().all(|x| x.2 == end.2), "{r:?}");
    assert!(end.2 >= 290, "{r:?}");
    assert!(end.1 < -0.5, "a loss: {r:?}");
}

/// Sword Boomerang's three hits land at random among two enemies whose HP
/// differs, so where they land changes what follows. Under BeliefDraw the
/// fair rows must not follow the state's real combat-stream cursors.
#[test]
fn fair_rows_do_not_depend_on_the_states_dice() {
    let mut a = fixture();
    a["deck"]["hand"] = json!([
        {"id": "SWORD_BOOMERANG", "kind": "card", "upgraded": false, "stable_idx": 1, "type": "Attack"},
        {"id": "DEFEND_IRONCLAD", "kind": "card", "upgraded": false, "stable_idx": 2, "type": "Skill"}]);
    let mut first = a["encounter"][0].clone();
    first["max_hp"] = json!(40);
    first["hp"] = json!(20);
    let mut second = first.clone();
    second["uid"] = json!(3);
    second["hp"] = json!(26);
    a["encounter"] = json!([first, second]);
    a["player"]["hp"] = json!(60);
    let base = rows(&a, 300, "beliefdraw");
    for n in 1..=6 {
        let mut b = a.clone();
        b["rng"]["counters"] =
            json!({"shuffle": n, "monster_ai": n, "combat_targets": n, "niche": n});
        assert_eq!(base, rows(&b, 300, "beliefdraw"), "stream cursors at {n}");
    }
}

/// The clairvoyant mode is the declared exception: it sees the true pile, so
/// its fair rows may (and here do) depend on the hidden order.
#[test]
fn clairvoyant_fair_rows_still_see_the_true_pile() {
    let (a, b) = both_orders();
    assert_ne!(
        state_rows(&a, 300, "clairvoyant"),
        state_rows(&b, 300, "clairvoyant")
    );
}

/// `threads > 1` spreads the honest fair searches over actions; each row is
/// the one-thread row, so the dict does not depend on the thread count.
#[test]
fn honest_fair_rows_do_not_depend_on_threads() {
    let (a, _) = both_orders();
    let p = |threads: u32| {
        RecommendParams::from_config(
            &json!({"sims": 200, "seed": 5, "fair": true, "threads": threads,
                    "hidden_info": "beliefdraw"}),
            "test",
        )
        .expect("params")
    };
    assert_eq!(
        recommend_value(&a, &p(1), true)["actions"],
        recommend_value(&a, &p(4), true)["actions"]
    );
}
