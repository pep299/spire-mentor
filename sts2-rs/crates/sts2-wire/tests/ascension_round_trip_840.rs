//! GH #840: a reconstructed monster keeps its ascension move table.
//!
//! C# reads every monster attack magnitude through
//! `AscensionHelper.GetValueIfAscension(DeadlyEnemies, deadly, base)` for the
//! whole fight (`SkulkingColony.cs:41-45`, `Entomancer.cs:27-31`,
//! `Vantom.cs:68-72`), and the engine bakes those values into the move table
//! once, when the AI is built (`sts2_content::ascension_scaling`).
//! `state_from_dict` rebuilt every monster's AI with the A0 `build_ai`, so the
//! current telegraph (which rides the wire) was right, and every LATER move hit
//! for the A0 number: the PIMC lockstep probe saw 375 of 463 A10 states diverge
//! at the next telegraph.
//!
//! Each test rebuilds an A10 fight from its own wire dict at every player
//! turn, carries the monster AI's instance state the way `transplant_hidden`
//! does (the wire cannot carry it), steps both through `EndTurn`, and requires
//! the same next telegraph.

mod common;

use common::{end_turn, fight};
use sts2_core::monsters::Intent;
use sts2_core::{CombatState, StepResult};

fn rebuild(state: &CombatState) -> CombatState {
    let wire = sts2_wire::emit::combat_state_to_value(state).expect("state emits");
    let mut rebuilt = sts2_wire::state_from_json(&wire).expect("wire rebuilds");
    assert_eq!(rebuilt.ascension, state.ascension);
    assert_eq!(rebuilt.enemy_uids.len(), state.enemy_uids.len());
    // `transplant_hidden`'s AI carry (sts2-py `lib.rs`): the move machine's
    // per-instance history has no wire representation.
    for (src_uid, dst_uid) in state.enemy_uids.iter().zip(rebuilt.enemy_uids.clone()) {
        let src = state
            .creatures
            .iter()
            .find(|c| c.id == src_uid.0)
            .expect("source enemy");
        let (history, src_ai) = (src.intent_history, src.ai.clone());
        let dst = rebuilt
            .creatures
            .iter_mut()
            .find(|c| c.id == dst_uid.0)
            .expect("rebuilt enemy");
        dst.intent_history = history;
        if let (Some(dst_ai), Some(src_ai)) = (dst.ai.as_mut(), src_ai.as_ref()) {
            dst_ai.transplant_instance_state_from(src_ai);
        }
    }
    rebuilt
}

/// Each enemy's (move state, telegraph).
fn telegraphs(state: &CombatState) -> Vec<(&'static str, Option<Intent>)> {
    state
        .enemy_uids
        .iter()
        .map(|uid| {
            let c = state
                .creatures
                .iter()
                .find(|c| c.id == uid.0)
                .expect("enemy present");
            (
                c.ai.as_ref()
                    .map_or("<no ai>", sts2_core::MonsterAi::current_state_id),
                c.intent,
            )
        })
        .collect()
}

/// [`telegraphs`] without the Buff / Debuff magnitude, which the wire drops
/// on the round trip (a known, separate artifact: the move resolves from the
/// move table, so the next telegraph still matches).
fn wire_telegraphs(state: &CombatState) -> Vec<(&'static str, Option<Intent>)> {
    telegraphs(state)
        .into_iter()
        .map(|(m, i)| {
            let i = i.map(|i| match i {
                Intent::Buff { kind, .. } => Intent::Buff { kind, amount: 0 },
                Intent::Debuff { kind, .. } => Intent::Debuff { kind, amount: 0 },
                other => other,
            });
            (m, i)
        })
        .collect()
}

/// Lockstep the true fight against a fresh rebuild for `turns` enemy turns
/// and return every telegraph the true fight showed after an `EndTurn`.
fn lockstep(encounter: &str, turns: usize) -> Vec<(&'static str, Option<Intent>)> {
    let mut truth = fight(
        "Ironclad",
        &["DEFEND_IRONCLAD"; 12],
        400,
        &[],
        &[],
        encounter,
        10,
    );
    assert_eq!(truth.ascension, 10);
    let mut seen = Vec::new();
    for turn in 0..turns {
        let mut world = rebuild(&truth);
        assert_eq!(
            wire_telegraphs(&world),
            wire_telegraphs(&truth),
            "{encounter} turn {turn}: the current telegraph rides the wire"
        );
        let r_truth = end_turn(&mut truth);
        let r_world = end_turn(&mut world);
        assert_eq!(r_world, r_truth, "{encounter} turn {turn}: step result");
        assert_eq!(
            telegraphs(&world),
            telegraphs(&truth),
            "{encounter} turn {turn}: the NEXT telegraph must come off the \
             A10 move table, not the A0 one"
        );
        seen.extend(telegraphs(&truth));
        if r_truth != StepResult::Advanced {
            break;
        }
    }
    seen
}

/// The (damage, hits) of every `move_id` telegraph in `seen`.
fn attacks(seen: &[(&'static str, Option<Intent>)], move_id: &str) -> Vec<(i32, u8)> {
    seen.iter()
        .filter(|(m, _)| *m == move_id)
        .filter_map(|(_, i)| match i {
            Some(Intent::Attack { damage }) => Some((*damage, 1)),
            Some(Intent::MultiAttack { damage, hits }) => Some((*damage, *hits)),
            _ => None,
        })
        .collect()
}

/// Every A10 value asserted below differs from its A0 one, so the lockstep
/// would have failed on it with an A0 rebuild.
#[test]
fn skulking_colony_rebuilt_at_a10_telegraphs_a10_damage() {
    let seen = lockstep("SKULKING_COLONY_ELITE", 5);
    // `SkulkingColony.cs:41,43,45`: Inertia 11/9, Zoom 16/14, PiercingStabs 8/7.
    assert!(attacks(&seen, "ZOOM_MOVE").contains(&(16, 1)), "{seen:?}");
    assert!(
        attacks(&seen, "INERTIA_MOVE").contains(&(11, 1)),
        "{seen:?}"
    );
    assert!(
        attacks(&seen, "PIERCING_STABS_MOVE").contains(&(8, 2)),
        "{seen:?}"
    );
}

#[test]
fn entomancer_rebuilt_at_a10_telegraphs_a10_damage() {
    let seen = lockstep("ENTOMANCER_ELITE", 5);
    // `Entomancer.cs:27,29`: Spear 20/18; Bees hit count 8/7 (3 per hit).
    assert!(attacks(&seen, "SPEAR").contains(&(20, 1)), "{seen:?}");
    assert!(attacks(&seen, "BEES").contains(&(3, 8)), "{seen:?}");
}

#[test]
fn vantom_rebuilt_at_a10_telegraphs_a10_damage() {
    let seen = lockstep("VANTOM_BOSS", 5);
    // `Vantom.cs:68,70,72`: InkBlot 8/7, InkyLance 7/6 (x2), Dismember 30/26.
    assert!(attacks(&seen, "INK_BLOT").contains(&(8, 1)), "{seen:?}");
    assert!(attacks(&seen, "INKY_LANCE").contains(&(7, 2)), "{seen:?}");
    assert!(attacks(&seen, "DISMEMBER").contains(&(30, 1)), "{seen:?}");
}

/// The other #840 reader: the KaiserCrab Surrounded facing is inferred by
/// forward-modelling each arm's BASE intent plain and ×1.5 against the wire
/// telegraph, and C# telegraphs the A9+ number (`Crusher.cs:81` THRASH 14/12,
/// `Rocket.cs:61` TARGETING_RETICLE 4/3). Facing Left, the A10 wire shows
/// Crusher plain 14 and Rocket boosted ⌊4·3/2⌋ = 6. Against an A0 table
/// (12 / 3) neither arm matched either candidate, and the facing fell back to
/// the spawn default, Right — the wrong arm took the ×1.5 for the rest of the
/// rollout.
#[test]
fn kaiser_crab_facing_is_inferred_from_a10_telegraphs() {
    let mut d: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/combat_action_ironclad.json"),
        )
        .expect("fixture readable"),
    )
    .expect("valid JSON");
    d["rng"]["ascension"] = serde_json::json!(10);
    d["intent_amounts"] = serde_json::json!("effective");
    d["player"]["powers"] = serde_json::json!([{ "name": "SURROUNDED", "amount": 1 }]);
    let arm = |kind: &str, uid: i64, back_attack: &str, move_state: &str, wire: i32| {
        serde_json::json!({
            "kind": kind,
            "uid": uid,
            "hp": 150,
            "max_hp": 209,
            "block": 0,
            "powers": [
                { "name": back_attack, "amount": 1 },
                { "name": "CRAB_RAGE", "amount": 1 }
            ],
            "intents": [{ "kind": "Attack", "amount": wire, "times": 1 }],
            "intent_history": [],
            "is_dead": false,
            "is_escaped": false,
            "move_state": move_state
        })
    };
    d["encounter"] = serde_json::json!([
        arm("Crusher", 2, "BACK_ATTACK_LEFT", "THRASH_MOVE", 14),
        arm(
            "Rocket",
            3,
            "BACK_ATTACK_RIGHT",
            "TARGETING_RETICLE_MOVE",
            6
        ),
    ]);

    let state = sts2_wire::state_from_json(&d).expect("wire rebuilds");
    let surrounded = state
        .powers
        .iter()
        .find(|p| p.kind == sts2_core::PowerKind::Surrounded)
        .expect("player Surrounded reconstructed");
    assert_eq!(
        surrounded.counter,
        sts2_core::hooks::SURROUNDED_FACING_LEFT,
        "plain A10 THRASH on the BackAttackLeft arm ⇒ facing Left"
    );
    let intent_of = |id: &str| {
        state
            .creatures
            .iter()
            .find(|c| c.monster_id.0 == id)
            .and_then(|c| c.intent)
    };
    assert_eq!(intent_of("Crusher"), Some(Intent::Attack { damage: 14 }));
    assert_eq!(
        intent_of("Rocket"),
        Some(Intent::Attack { damage: 4 }),
        "the back arm's boosted 6 unbakes to its A10 base 4"
    );
}
