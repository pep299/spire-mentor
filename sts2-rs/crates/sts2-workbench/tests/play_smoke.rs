//! End-to-end smoke: drive a run through the route dispatcher with a
//! scripted policy and exercise seek / save / restore / persistence.
//!
//! Not a parity test (that is `golden_parity.rs`); this catches panics and
//! shape regressions in the port without needing the Python oracle.

use serde_json::{json, Value};
use sts2_workbench::api::{Config, Workbench};

fn new_wb() -> Workbench {
    Workbench::new(Config {
        assets_base: "https://assets.test".to_string(),
        ..Config::default()
    })
}

fn call(wb: &mut Workbench, method: &str, route: &str, body: Value) -> Value {
    match wb.call(method, route, &body) {
        Ok(v) => v,
        Err(e) => panic!("{method} {route} -> {} {}", e.status, e.detail),
    }
}

/// The scripted policy of the golden dumper, in miniature: returns the next
/// `(action, args)` for a step, or `None` at game over.
#[cfg(workbench_game_data)] // only the game-data-gated tests use it
fn policy(step: &Value, shop_done: &mut bool) -> Option<(String, Value)> {
    let state = &step["state"];
    let decision = step["decision"].as_str().unwrap_or("");
    let options = state["options"].as_array().cloned().unwrap_or_default();
    let first_enabled = |opts: &[Value]| {
        opts.iter()
            .find(|o| o["enabled"].as_bool().unwrap_or(true))
            .map(|o| (o["action"].as_str().unwrap().to_string(), o["args"].clone()))
    };
    match decision {
        "game_over" => None,
        "combat_play" => {
            let hand = state["hand"].as_array().cloned().unwrap_or_default();
            let enemy0 = state["enemies"][0]["index"].clone();
            for card in &hand {
                if card["can_play"].as_bool() == Some(true) {
                    let mut args = json!({"card_index": card["index"]});
                    if card["target_type"] == json!("AnyEnemy") {
                        args["target_index"] = enemy0.clone();
                    }
                    return Some(("play_card".to_string(), args));
                }
            }
            Some(("end_turn".to_string(), json!({})))
        }
        "map_select" => {
            let prefer = [
                "Shop", "RestSite", "Treasure", "Elite", "Event", "Monster", "Ancient", "Boss",
            ];
            for p in prefer {
                if let Some(o) = options
                    .iter()
                    .find(|o| o["label"].as_str().is_some_and(|l| l.starts_with(p)))
                {
                    return Some((o["action"].as_str().unwrap().to_string(), o["args"].clone()));
                }
            }
            first_enabled(&options)
        }
        "shop" => {
            if !*shop_done {
                *shop_done = true;
                if let Some(o) = options.iter().find(|o| o["action"] == json!("buy_card")) {
                    return Some(("buy_card".to_string(), o["args"].clone()));
                }
            }
            Some(("leave_room".to_string(), json!({})))
        }
        "card_select" => {
            let min = state["min_select"].as_u64().unwrap_or(1).max(1);
            let indices: Vec<String> = (0..min).map(|i| i.to_string()).collect();
            Some((
                "select_cards".to_string(),
                json!({"indices": indices.join(",")}),
            ))
        }
        _ => first_enabled(&options).or_else(|| Some(("proceed".to_string(), json!({})))),
    }
}

#[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
#[test]
fn ironclad_plays_through_act_one_start() {
    let mut wb = new_wb();
    let caps = call(&mut wb, "GET", "/api/capabilities", Value::Null);
    assert_eq!(caps["backends"], json!(["rust"]));
    assert!(
        wb.call("GET", "/api/live/state", &Value::Null).is_err(),
        "no run yet → 400"
    );

    let mut step = call(
        &mut wb,
        "POST",
        "/api/live/start",
        json!({"character": "Ironclad", "seed": "SMOKE1", "ascension": 0, "act1": "overgrowth"}),
    );
    assert_eq!(step["decision"], json!("event_choice"));
    assert_eq!(step["state"]["event_name"], json!("Neow"));
    assert_eq!(
        step["state"]["assets_base"].as_str(),
        None,
        "assets_base sits on the step"
    );
    assert_eq!(step["assets_base"], json!("https://assets.test"));
    assert!(step["state"]["event_options"]
        .as_array()
        .is_some_and(|o| !o.is_empty()));
    assert!(step["state"]["player"]["relics"][0]["image"].is_string());

    let mut shop_done = false;
    let mut seen = std::collections::BTreeSet::new();
    let mut steps = 0;
    let mut reached_combat = false;
    while let Some((action, args)) = policy(&step, &mut shop_done) {
        let resp = call(
            &mut wb,
            "POST",
            "/api/live/action",
            json!({"action": action, "args": args}),
        );
        assert!(
            resp["error"].is_null(),
            "action {action} {args} rejected: {}",
            resp["error"]
        );
        assert!(resp["label"].is_string());
        step = resp["step"].clone();
        let d = step["decision"].as_str().unwrap_or("").to_string();
        if d == "combat_play" {
            if !reached_combat {
                // The phone-speed probe works over the live combat.
                let bench = call(
                    &mut wb,
                    "POST",
                    "/api/bench",
                    json!({"sims": 40, "playouts": 2}),
                );
                assert_eq!(bench["sims"], json!(40));
                assert!(bench["best_action"].is_object(), "{bench}");
                assert_eq!(bench["playouts"]["n"], json!(2));
            }
            reached_combat = true;
            let hand = step["state"]["hand"].as_array().unwrap();
            for c in hand {
                assert!(c["image"].is_string() || c["image"].is_null());
                assert!(c["description"].is_string());
                assert!(c["color"].is_string());
            }
            assert!(step["state"]["energy_image"].is_string());
            assert!(step["meta"]["turn"].is_number());
        }
        seen.insert(d);
        steps += 1;
        if steps >= 120 {
            break;
        }
    }
    assert!(
        reached_combat,
        "never reached a combat; decisions seen: {seen:?}"
    );
    assert!(seen.contains("map_select"), "{seen:?}");

    // Timeline + seek back 3 nodes + fork.
    let tl = call(&mut wb, "GET", "/api/live/timeline", Value::Null);
    let path = tl["active_path"].as_array().unwrap().clone();
    assert!(path.len() > 4);
    let target = path[path.len() - 4].clone();
    let sought = call(
        &mut wb,
        "POST",
        "/api/live/seek",
        json!({"node_id": target}),
    );
    assert!(sought["step"]["decision"].is_string());
    let tl2 = call(&mut wb, "GET", "/api/live/timeline", Value::Null);
    assert_eq!(tl2["active_leaf"], target);
    assert!(wb
        .call("POST", "/api/live/seek", &json!({"node_id": 99999}))
        .is_err_and(|e| e.status == 404));

    // Live map at any decision.
    let map = call(&mut wb, "GET", "/api/live/map", Value::Null);
    assert!(map["graphs"].is_array());

    // Save → restore round trip lands on the same decision.
    let before = call(&mut wb, "GET", "/api/live/state", Value::Null);
    let payload = wb
        .build_save("smoke save", "2026-01-01T00:00:00Z")
        .expect("save");
    assert_eq!(payload["schema_version"], json!(1));
    assert_eq!(
        payload["step_count"].as_u64().unwrap() as usize,
        path.len() - 4
    );
    let header = Workbench::save_header(&payload);
    assert!(header.get("line").is_none());
    let restored = wb
        .restore_save(&payload, sts2_workbench::runlog::DEFAULT_NOW_ISO)
        .expect("restore");
    assert_eq!(restored["decision"], before["decision"]);
    assert_eq!(
        restored["state"]["player"]["hp"],
        before["state"]["player"]["hp"]
    );

    // Persist → import round trip keeps the whole tree and the active leaf.
    let persisted = wb.export_persist();
    assert!(persisted["nodes"]
        .as_array()
        .is_some_and(|n| n.len() == path.len() - 3));
    let mut wb2 = new_wb();
    let mut ticks = 0;
    let imported = wb2
        .import_persist(&persisted, |_done, _total| ticks += 1)
        .expect("import");
    assert!(ticks > 0);
    assert_eq!(imported["decision"], restored["decision"]);
    let tl3 = call(&mut wb2, "GET", "/api/live/timeline", Value::Null);
    assert_eq!(tl3["active_leaf"], persisted["active_leaf"]);

    // Quit clears the slot and names the run log it wrote (M4).
    let out = call(&mut wb, "POST", "/api/live/quit", Value::Null);
    assert_eq!(out["ok"], json!(true));
    assert!(
        out["log_path"]
            .as_str()
            .is_some_and(|p| std::path::Path::new(p)
                .extension()
                .is_some_and(|e| e == "jsonl")),
        "{out}"
    );
    assert!(wb.take_run_log().is_some());
    assert!(wb.call("GET", "/api/live/state", &Value::Null).is_err());
}

#[test]
fn every_character_starts_and_rolls_act_one() {
    for character in ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"] {
        let mut wb = new_wb();
        let step = call(
            &mut wb,
            "POST",
            "/api/live/start",
            json!({"character": character, "seed": "ROLL2", "ascension": 10}),
        );
        assert_eq!(step["state"]["player"]["name"], json!(character));
        let act1 = step["state"]["act1"].as_str().unwrap();
        assert!(
            act1 == "overgrowth" || act1 == "underdocks",
            "{character}: {act1}"
        );
        assert_eq!(step["state"]["ascension"], json!(10));
        assert!(step["state"]["player"]["deck"]
            .as_array()
            .is_some_and(|d| !d.is_empty()));
    }
}

#[test]
fn bad_requests_get_python_status_codes() {
    let mut wb = new_wb();
    let e = wb
        .call("POST", "/api/live/start", &json!({"character": "Nobody"}))
        .unwrap_err();
    assert_eq!(e.status, 400);
    let e = wb
        .call("POST", "/api/live/start", &json!({"backend": "csharp"}))
        .unwrap_err();
    assert_eq!(e.status, 400);
    call(&mut wb, "POST", "/api/live/start", json!({"seed": "X"}));
    let e = wb
        .call(
            "POST",
            "/api/live/action",
            &json!({"action": "teleport", "args": {}}),
        )
        .unwrap_err();
    assert_eq!(e.status, 400);
    // Advisors are served (M2): off combat the hint is a 409, like Python;
    // a route the browser build does not have is a 501.
    let e = wb.call("POST", "/api/live/hint", &json!({})).unwrap_err();
    assert_eq!(e.status, 409);
    let e = wb
        .call("POST", "/api/live/deck-analysis", &json!({}))
        .unwrap_err();
    assert_eq!(e.status, 501);
    assert!(wb.build_save("   ", "t").is_err_and(|e| e.status == 400));
    assert!(wb
        .restore_save(
            &json!({"schema_version": 7}),
            sts2_workbench::runlog::DEFAULT_NOW_ISO
        )
        .is_err_and(|e| e.status == 404));
}
