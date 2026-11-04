//! The read-only `CombatState` getters of GH #1097 (`telegraph`,
//! `peek_intents`, `true_draw_top`) on real fights built with the scenario
//! builder: what each returns, that the forecast names the moves the fight
//! then telegraphs, and that none of them moves the state.

mod common;

use std::collections::BTreeMap;

use common::{boss, boss_mut, end_turn, fight, move_id};
use serde_json::{json, Value};
use sts2_core::{CombatState, Intent, StepResult};
use sts2_wire::emit::combat::{
    combat_state_to_value, enemy_intent_forecast, enemy_telegraphs, telegraph_intents,
    true_draw_top, MAX_FORECAST_STEPS,
};

/// The Terror Eel at A10 against a 20-Defend deck: its pattern is the fixed
/// CRASH_MOVE ↔ ThrashMove cycle (`TerrorEel.cs`), and ThrashMove telegraphs
/// two intents (a multi-attack and a Vigor buff).
fn eel() -> CombatState {
    let state = fight(
        "Ironclad",
        &["DEFEND_IRONCLAD"; 20],
        400,
        &[],
        &[],
        "TERROR_EEL_ELITE",
        10,
    );
    assert_eq!(move_id(boss(&state)), "CRASH_MOVE");
    state
}

/// `{kind, amount, times}` of an intent dict: the move-table part, without
/// the board-baked `effective_amount`.
fn bare(intent: &Value) -> Value {
    json!([intent["kind"], intent["amount"], intent["times"]])
}

fn ids(steps: &Value) -> Vec<&str> {
    steps
        .as_array()
        .expect("forecast list")
        .iter()
        .map(|s| s["move_id"].as_str().expect("move id"))
        .collect()
}

/// Everything a getter could disturb: the wire dict and the three hashes.
fn fingerprint(state: &CombatState) -> (Value, u64, u64, u64) {
    (
        combat_state_to_value(state).expect("serializes"),
        sts2_core::state_hash(state),
        sts2_core::search_hash(state),
        sts2_core::chance_outcome_hash(state),
    )
}

#[test]
fn the_telegraph_head_is_the_wire_head_and_the_tail_is_the_move() {
    let mut state = eel();
    // A one-intent move: the telegraph is the wire's list.
    let wire = combat_state_to_value(&state).expect("serializes");
    let tele = enemy_telegraphs(&state).expect("telegraph");
    assert_eq!(tele[0], wire["encounter"][0]["intents"]);
    assert_eq!(tele[0].as_array().unwrap().len(), 1);

    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    assert_eq!(move_id(boss(&state)), "ThrashMove");
    let wire = combat_state_to_value(&state).expect("serializes");
    let tele = enemy_telegraphs(&state).expect("telegraph");
    let list = tele[0].as_array().expect("list");
    assert_eq!(list.len(), 2, "{tele}");
    assert_eq!(list[0], wire["encounter"][0]["intents"][0]);
    assert_eq!(list[0]["kind"], "Attack");
    assert_eq!(list[0]["times"], 3);
    assert_eq!(list[1]["kind"], "Buff");
    // The wire itself still carries the head only.
    assert_eq!(wire["encounter"][0]["intents"].as_array().unwrap().len(), 1);
}

#[test]
fn a_head_reprimed_away_from_the_move_stands_alone() {
    let mut state = eel();
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    assert_eq!(telegraph_intents(boss(&state)).len(), 2);

    boss_mut(&mut state).intent = Some(Intent::Heal);
    assert_eq!(telegraph_intents(boss(&state)).as_slice(), &[Intent::Heal]);
    let wire = combat_state_to_value(&state).expect("serializes");
    assert_eq!(
        enemy_telegraphs(&state).unwrap()[0],
        wire["encounter"][0]["intents"]
    );

    boss_mut(&mut state).intent = None;
    assert!(telegraph_intents(boss(&state)).is_empty());
    assert_eq!(enemy_telegraphs(&state).unwrap(), json!([[]]));
}

/// Step 1 is the move after the telegraph both before and after the first
/// move; `peek_intents` would repeat the telegraph before it.
#[test]
fn the_forecast_names_the_moves_the_fight_then_telegraphs() {
    let mut state = eel();
    let opening = enemy_intent_forecast(&state, 3).expect("forecast");
    assert_eq!(ids(&opening[0]), ["ThrashMove", "CRASH_MOVE", "ThrashMove"]);
    for (i, step) in opening[0].as_array().unwrap().iter().enumerate() {
        assert_eq!(step["step"], i + 1);
        assert_eq!(step["certainty"], 1.0);
    }
    assert_eq!(opening[0][0]["intents"].as_array().unwrap().len(), 2);

    let mut seen = Vec::new();
    for _ in 0..4 {
        let forecast = enemy_intent_forecast(&state, 3).expect("forecast");
        assert_eq!(end_turn(&mut state), StepResult::Advanced);
        let now = move_id(boss(&state));
        assert_eq!(forecast[0][0]["move_id"], now);
        // The forecast's intents are the telegraph the fight then shows.
        let tele = enemy_telegraphs(&state).expect("telegraph");
        let shown: Vec<Value> = tele[0].as_array().unwrap().iter().map(bare).collect();
        let predicted: Vec<Value> = forecast[0][0]["intents"]
            .as_array()
            .unwrap()
            .iter()
            .map(bare)
            .collect();
        assert_eq!(shown, predicted);
        seen.push(now);
    }
    assert_eq!(
        seen,
        ["ThrashMove", "CRASH_MOVE", "ThrashMove", "CRASH_MOVE"]
    );
}

#[test]
fn the_forecast_takes_k_and_refuses_a_step_count_it_cannot_number() {
    let state = eel();
    assert_eq!(enemy_intent_forecast(&state, 0).unwrap(), json!([[]]));
    let long = enemy_intent_forecast(&state, MAX_FORECAST_STEPS).unwrap();
    assert_eq!(long[0].as_array().unwrap().len(), MAX_FORECAST_STEPS);
    assert_eq!(long[0][MAX_FORECAST_STEPS - 1]["step"], MAX_FORECAST_STEPS);
    assert!(enemy_intent_forecast(&state, MAX_FORECAST_STEPS + 1).is_err());
}

#[test]
fn true_draw_top_is_the_order_the_next_draws_take() {
    let mut state = eel();
    let pile: Vec<u32> = state.player.draw.iter().map(|u| u.0.get()).collect();
    assert_eq!(pile.len(), 15, "20 cards less the opening hand");
    let top = true_draw_top(&state, 5).expect("top");
    let top_uids: Vec<u64> = top
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["stable_idx"].as_u64().unwrap())
        .collect();
    assert_eq!(
        top_uids,
        pile[..5].iter().map(|&u| u64::from(u)).collect::<Vec<_>>()
    );
    // The same cards, in the same order, as the wire's draw pile.
    let wire = combat_state_to_value(&state).expect("serializes");
    assert_eq!(
        top.as_array().unwrap().as_slice(),
        &wire["deck"]["draw_pile"].as_array().unwrap()[..5]
    );
    assert_eq!(
        true_draw_top(&state, 99).unwrap(),
        wire["deck"]["draw_pile"],
        "k past the pile returns the whole pile"
    );
    assert_eq!(true_draw_top(&state, 0).unwrap(), json!([]));

    // The next turn's hand is exactly those five, in that order.
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    let hand: Vec<u64> = state
        .player
        .hand
        .iter()
        .map(|u| u64::from(u.0.get()))
        .collect();
    assert_eq!(hand, top_uids);
}

#[test]
fn true_draw_top_refuses_a_belief_view_and_reads_a_sampled_world() {
    let state = eel();
    let mut belief = state.clone();
    sts2_core::blindfold::belief_root(&mut belief, 11);
    let err = true_draw_top(&belief, 5).expect_err("a belief view has no draw order");
    assert!(err.message().contains("deferred_draw_order"), "{err}");

    let mut world = state.clone();
    sts2_core::blindfold::blindfold_root(&mut world, 11);
    let wire = combat_state_to_value(&world).expect("serializes");
    assert_eq!(
        true_draw_top(&world, 5)
            .unwrap()
            .as_array()
            .unwrap()
            .as_slice(),
        &wire["deck"]["draw_pile"].as_array().unwrap()[..5]
    );
}

/// Monsters whose next move is decided by something the coming turn changes
/// besides the move machine, so a forecast made against the current board can
/// name another move with certainty 1.0. Each was read off this test's output.
const BOARD_DRIVEN: &[(&str, &str)] = &[
    // The Slumber counter ticks down at turn end and the wake re-points the
    // telegraph at ROLL_OUT_MOVE.
    ("SLUMBERING_BEETLE_NORMAL", "SlumberingBeetle"),
    // The telegraphed move's effect decrements `TurnsUntilSummonable` (the
    // AI counter), which opens CALL_FOR_BACKUP at the next roll.
    ("TWO_TAILED_RATS_NORMAL", "TwoTailedRat"),
    // Asleep for a fixed number of turns, then it wakes into SLASH.
    ("LAGAVULIN_MATRIARCH_BOSS", "LagavulinMatriarch"),
];

/// Every encounter of the four act tables at A10, on both builds, played
/// passively for six enemy turns: the telegraph's head is the wire's at every
/// turn, no getter moves the state, and every step-1 forecast made with
/// certainty 1.0 names the move the enemy telegraphs after the coming enemy
/// turn, except for the [`BOARD_DRIVEN`] monsters.
#[test]
fn across_every_encounter_the_getters_agree_with_the_fight() {
    for version in sts2_core::GameVersion::ALL {
        let mut built = 0usize;
        let mut checked = 0usize;
        let mut multi = 0usize;
        let mut misses: Vec<String> = Vec::new();
        let mut excused = 0usize;
        for idx in 0..sts2_content::encounters::n_encounters() {
            let def = sts2_content::encounters::by_index(idx).expect("in range");
            let mut state = sts2_wire::scenario::build_scenario_state_for_version(
                version,
                7,
                "Ironclad",
                &vec!["DEFEND_IRONCLAD".to_owned(); 20],
                999,
                999,
                &[],
                &[],
                def.id,
                10,
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("{version:?} {}: {e}", def.id));
            built += 1;
            for turn in 0..6 {
                let before = fingerprint(&state);
                let tele = enemy_telegraphs(&state).expect("telegraph");
                let forecast = enemy_intent_forecast(&state, 3).expect("forecast");
                let _ = true_draw_top(&state, 10).expect("draw top");
                assert_eq!(fingerprint(&state), before, "{} turn {turn}", def.id);

                for (i, wire_enemy) in before.0["encounter"].as_array().unwrap().iter().enumerate()
                {
                    let head = wire_enemy["intents"].get(0).cloned().unwrap_or(Value::Null);
                    assert_eq!(
                        tele[i].get(0).cloned().unwrap_or(Value::Null),
                        head,
                        "{version:?} {} turn {turn} enemy {i}",
                        def.id
                    );
                    if tele[i].as_array().unwrap().len() > 1 {
                        multi += 1;
                    }
                }

                // Step 1 of each enemy, keyed by uid (enemies can spawn or die).
                let predicted: BTreeMap<u32, (String, f64)> = state
                    .enemy_uids
                    .iter()
                    .zip(forecast.as_array().unwrap())
                    .filter_map(|(uid, steps)| {
                        let s = steps.get(0)?;
                        Some((
                            uid.0.get(),
                            (s["move_id"].as_str()?.to_owned(), s["certainty"].as_f64()?),
                        ))
                    })
                    .collect();
                if end_turn(&mut state) != StepResult::Advanced {
                    break;
                }
                for (uid, (want, certainty)) in &predicted {
                    if *certainty < 1.0 {
                        continue;
                    }
                    let Some(c) = state
                        .creatures
                        .iter()
                        .find(|c| c.id.get() == *uid && c.current_hp > 0)
                    else {
                        continue;
                    };
                    let Some(ai) = c.ai.as_ref() else { continue };
                    checked += 1;
                    if ai.current_state_id() == want {
                        continue;
                    }
                    let species = c.species.map(|k| format!("{k:?}")).unwrap_or_default();
                    if BOARD_DRIVEN
                        .iter()
                        .any(|(enc, kind)| *enc == def.id && *kind == species)
                    {
                        excused += 1;
                        continue;
                    }
                    misses.push(format!(
                        "{} turn {turn} {species} (uid {uid}): forecast {want}, then {}",
                        def.id,
                        ai.current_state_id()
                    ));
                }
            }
        }
        eprintln!(
            "{version:?}: built {built} encounters; {checked} certain step-1 forecasts \
             checked, {excused} board-driven, {} missed; {multi} multi-intent telegraphs",
            misses.len()
        );
        for m in &misses {
            eprintln!("  {m}");
        }
        assert_eq!(built, sts2_content::encounters::n_encounters());
        assert!(
            checked > 500,
            "{version:?}: only {checked} forecasts checked"
        );
        assert!(multi > 0, "{version:?}: no multi-intent telegraph seen");
        assert!(
            misses.is_empty(),
            "{version:?}: {} missed forecasts",
            misses.len()
        );
    }
}
