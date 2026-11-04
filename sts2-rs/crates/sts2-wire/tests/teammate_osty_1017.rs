//! GH #1017: a Necrobinder teammate's Osty survives the wire.
//!
//! Only our own Osty travelled (`$.player.osty`), so a rebuilt teammate had
//! none: a monster hit their Osty would have soaked (C# `DieForYouPower`)
//! landed on them, and their Unleash dealt nothing. A teammate's Osty rides
//! in `$.allies[i].osty`, in our `osty` shape, and is built into their seat
//! (`PlayerCombatState::pet_uid`, which `coop::swap_seat` swaps).

use serde_json::json;
use sts2_core::{CombatState, PowerKind};
use sts2_wire::emit::combat_state_to_value;
use sts2_wire::state_from_json;

fn with_teammate(extra: &serde_json::Value) -> serde_json::Value {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combat_action_ironclad.json"),
    )
    .expect("fixture");
    let mut v: serde_json::Value = serde_json::from_str(&text).expect("fixture parses");
    let mut ally = json!({
        "class": "necrobinder", "hp": 50, "max_hp": 70, "block": 0,
        "energy": 3, "max_energy": 3, "powers": [], "hand": [],
    });
    for (k, val) in extra.as_object().expect("object") {
        ally[k] = val.clone();
    }
    v["allies"] = json!([ally]);
    v
}

fn osty_of(st: &CombatState, ally: usize) -> &sts2_core::Creature {
    let uid = st.allies[ally].pet_uid.expect("their pet");
    st.creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("in the arena")
}

fn has_die_for_you(st: &CombatState, c: &sts2_core::Creature) -> bool {
    c.powers.iter().any(|u| {
        st.powers
            .iter()
            .any(|p| p.id == u.0 && p.kind == PowerKind::DieForYou)
    })
}

#[test]
fn a_teammates_osty_joins_their_seat_and_round_trips() {
    let osty = json!({"alive": true, "hp": 5, "max_hp": 9, "block": 0});
    let st = state_from_json(&with_teammate(&json!({ "osty": osty }))).expect("builds");
    assert_eq!(st.player.pet_uid, None, "ours untouched");
    let c = osty_of(&st, 0);
    assert_eq!((c.current_hp, c.max_hp), (5, 9));
    assert!(has_die_for_you(&st, c), "their hits redirect to it");
    assert!(
        !st.enemy_uids.contains(&st.allies[0].pet_uid.expect("pet")),
        "a pet, not an enemy"
    );

    let out = combat_state_to_value(&st).expect("emits");
    assert_eq!(out["allies"][0]["osty"], osty);
    assert!(out["player"].get("osty").is_none());
    let back = state_from_json(&out).expect("re-parses");
    let c = osty_of(&back, 0);
    assert_eq!((c.current_hp, c.max_hp), (5, 9));
}

/// A dead Osty stays (C# never removes it from `Pets`), keeping the max HP
/// its next summon revives to.
#[test]
fn a_teammates_dead_osty_keeps_its_max_hp() {
    let osty = json!({"alive": false, "hp": 0, "max_hp": 7, "block": 0});
    let st = state_from_json(&with_teammate(&json!({ "osty": osty }))).expect("builds");
    let c = osty_of(&st, 0);
    assert_eq!((c.current_hp, c.max_hp), (0, 7));
    let out = combat_state_to_value(&st).expect("emits");
    assert_eq!(out["allies"][0]["osty"], osty);
}

/// Both Ostys at once: ours stays ours, theirs theirs.
#[test]
fn our_osty_and_a_teammates_are_two_creatures() {
    let mut v = with_teammate(&json!({"osty": {"alive": true, "hp": 4, "max_hp": 4, "block": 0}}));
    v["player"]["osty"] = json!({"alive": true, "hp": 8, "max_hp": 8, "block": 0});
    let st = state_from_json(&v).expect("builds");
    let ours = st.player.pet_uid.expect("ours");
    let theirs = st.allies[0].pet_uid.expect("theirs");
    assert_ne!(ours, theirs);
    assert_eq!(osty_of(&st, 0).current_hp, 4);
    let out = combat_state_to_value(&st).expect("emits");
    assert_eq!(out["player"]["osty"]["hp"], 8);
    assert_eq!(out["allies"][0]["osty"]["hp"], 4);
}

#[test]
fn a_teammate_without_an_osty_emits_none() {
    let st = state_from_json(&with_teammate(&json!({}))).expect("builds");
    assert_eq!(st.allies[0].pet_uid, None);
    let out = combat_state_to_value(&st).expect("emits");
    assert!(out["allies"][0].get("osty").is_none());
}
