//! The replay viewer (M4) over the API: a run played by the agent, quit to a
//! run log, loaded back, and every replay route answering — plus the deck
//! analysis routes on the live run and "play from here".

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

fn err(wb: &mut Workbench, method: &str, route: &str, body: Value) -> (u16, String) {
    match wb.call(method, route, &body) {
        Ok(v) => panic!("{method} {route} unexpectedly ok: {v}"),
        Err(e) => (e.status, e.detail),
    }
}

/// A short agent-played run, ended: returns the written log.
fn play_and_quit(wb: &mut Workbench, seed: &str, steps: usize) -> sts2_workbench::runlog::RunLog {
    call(
        wb,
        "POST",
        "/api/live/start-faithful",
        json!({"character": "Ironclad", "seed": seed, "sims": 4, "boss_sims": 8}),
    );
    for _ in 0..steps {
        let out = call(wb, "POST", "/api/live/run-mode-step", json!({}));
        if out["over"] == json!(true) || out["done"] == json!(true) {
            break;
        }
    }
    let out = call(
        wb,
        "POST",
        "/api/live/quit",
        json!({"now": "2026-09-03T04:22:57.5Z"}),
    );
    assert_eq!(out["ok"], json!(true));
    let log = wb.take_run_log().expect("a log was written");
    assert_eq!(out["log_path"], json!(log.filename));
    assert!(wb.take_run_log().is_none(), "collected once");
    log
}

/// A run that a new start replaces is quit, not dropped: its log waits for
/// the host stamped with the start's `now`, as Python's `LiveRuns.replace`
/// writes it. Both logs of a quit-after-restart come out, oldest first.
#[test]
fn a_replaced_run_leaves_its_log() {
    let mut wb = new_wb();
    let start = |wb: &mut Workbench, now: &str| {
        call(
            wb,
            "POST",
            "/api/live/start-faithful",
            json!({"character": "Ironclad", "seed": "run_11", "sims": 4, "boss_sims": 8, "now": now}),
        );
        call(wb, "POST", "/api/live/run-mode-step", json!({}));
    };
    start(&mut wb, "2026-09-03T04:22:57Z");
    assert!(wb.take_run_log().is_none(), "nothing was live to replace");
    start(&mut wb, "2026-09-03T04:23:10Z");
    call(
        &mut wb,
        "POST",
        "/api/live/quit",
        json!({"now": "2026-09-03T04:24:00Z"}),
    );
    let names: Vec<String> = std::iter::from_fn(|| wb.take_run_log())
        .map(|l| l.filename)
        .collect();
    assert_eq!(
        names,
        vec![
            "20260903_042310_Ironclad_11.jsonl".to_string(),
            "20260903_042400_Ironclad_11.jsonl".to_string(),
        ]
    );
}

#[test]
fn quit_writes_the_active_path_as_a_run_log() {
    let mut wb = new_wb();
    let e = err(&mut wb, "POST", "/api/live/quit", json!({}));
    assert_eq!(e.0, 400);
    let log = play_and_quit(&mut wb, "run_11", 25);
    assert_eq!(log.filename, "20260903_042257_Ironclad_11.jsonl");
    let lines: Vec<Value> = log
        .text
        .lines()
        .map(|l| serde_json::from_str(l).expect("jsonl"))
        .collect();
    assert_eq!(lines[0]["type"], json!("meta"));
    assert_eq!(lines[0]["data"]["run_mode"], json!(true));
    assert_eq!(lines[0]["data"]["character"], json!("Ironclad"));
    assert_eq!(lines[1]["type"], json!("state"));
    assert_eq!(lines[1]["step"], json!(1));
    let actions: Vec<&Value> = lines
        .iter()
        .filter(|l| l["type"] == json!("action"))
        .collect();
    assert!(actions.len() >= 20);
    // Native-agent steps carry their label and replayable form.
    let a = actions[0];
    assert_eq!(a["data"]["action"], json!("__agent_native__"));
    assert!(a["data"]["label"].as_str().is_some_and(|l| !l.is_empty()));
    assert_eq!(a["data"]["replay"]["action"], json!("__agent_native__"));
    assert!(
        a["data"]["args"]["action"].is_string(),
        "native args keep their kind"
    );
    let last = lines.last().unwrap();
    assert_eq!(last["type"], json!("meta"));
    assert_eq!(last["data"]["summary"], json!(true));
    assert!(last["data"]["final_floor"].is_number());
    assert!(lines.iter().any(|l| l["type"] == json!("combat_log")));
}

#[test]
fn replay_routes_over_a_loaded_log() {
    let mut wb = new_wb();
    for route in [
        "/api/timeline",
        "/api/map",
        "/api/step/0",
        "/api/deck/trend",
    ] {
        assert_eq!(
            err(&mut wb, "GET", route, Value::Null),
            (400, "no run loaded".to_string())
        );
    }
    let log = play_and_quit(&mut wb, "run_12", 40);
    // `/api/load` and the log listings are host-owned: the worker hands the
    // text to `load_run` and lists IndexedDB itself.
    for (m, route) in [
        ("POST", "/api/load"),
        ("GET", "/api/runs"),
        ("GET", "/api/mcts-runs"),
    ] {
        assert_eq!(err(&mut wb, m, route, json!({"path": log.filename})).0, 501);
    }
}

#[test]
fn load_run_and_the_replay_projections() {
    let mut wb = new_wb();
    let log = play_and_quit(&mut wb, "run_12", 40);
    assert!(wb
        .load_run("bad.jsonl", "{not json")
        .is_err_and(|e| e.status == 422));
    let loaded = wb.load_run(&log.filename, &log.text).expect("load");
    let count = loaded["count"].as_u64().unwrap();
    assert!(count >= 20);
    assert_eq!(loaded["resume"], json!({"kind": "run", "reason": null}));

    let tl = call(&mut wb, "GET", "/api/timeline", Value::Null);
    let steps = tl["steps"].as_array().unwrap();
    assert_eq!(steps.len() as u64, count);
    for s in steps.iter().take(steps.len() - 1) {
        assert!(s["label"].as_str().is_some_and(|l| !l.is_empty()), "{s}");
        assert!(s["details"].is_array());
    }
    assert_eq!(steps[0]["floor"], json!(1), "Neow is the game's floor 1");
    assert!(steps
        .iter()
        .any(|s| s["combat_log"].as_array().is_some_and(|c| !c.is_empty())));

    let map = call(&mut wb, "GET", "/api/map", Value::Null);
    let rows = map["rows"].as_array().unwrap();
    assert!(!rows.is_empty());
    // An agent-played `move` records native args (`{action, col}`, no
    // `row`), so `chosen` never matches for it — the Python route has the
    // same blind spot; a hand-played `select_map_node` marks its node.
    assert!(rows[0]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .all(|n| n["chosen"].is_boolean()));
    assert!(rows[0]["nodes"][0]["icon"].is_string() || rows[0]["nodes"][0]["icon"].is_null());
    assert!(!map["graphs"].as_array().unwrap().is_empty());

    let step = call(&mut wb, "GET", "/api/step/0", Value::Null);
    assert_eq!(step["idx"], json!(0));
    assert_eq!(step["source"], json!("log"));
    assert_eq!(step["assets_base"], json!("https://assets.test"));
    assert!(step["action_label"].is_string());
    let e = err(&mut wb, "GET", &format!("/api/step/{count}"), Value::Null);
    assert_eq!(
        e,
        (404, format!("step {count} out of range (0..{})", count - 1))
    );
    assert_eq!(err(&mut wb, "GET", "/api/step/x", Value::Null).0, 422);

    let da = call(&mut wb, "GET", "/api/step/0/deck-analysis", Value::Null);
    assert_eq!(da["deck_size"], json!(10));
    assert!(da["scores"]["volume"].is_number());
    let trend = call(&mut wb, "GET", "/api/deck/trend", Value::Null);
    assert!(!trend["points"].as_array().unwrap().is_empty());
    assert_eq!(trend["points"][0]["idx"], json!(0));

    // No net was loaded: the WPA review has nothing.
    let wpa = call(&mut wb, "POST", "/api/wpa-review", json!({"config": {}}));
    assert_eq!(wpa["available"], json!(false));
    assert!(wpa["reason"]
        .as_str()
        .unwrap()
        .contains("no recorded value data"));

    // The combat routes rebuild the run to the step and search it.
    let combat_idx = steps
        .iter()
        .position(|s| s["decision"] == json!("combat_play"))
        .expect("a fight") as i64;
    let rec = call(
        &mut wb,
        "POST",
        "/api/recommend",
        json!({"idx": combat_idx, "config": {"sims": 16, "seed": 1}}),
    );
    assert!(
        rec["scores"].as_array().is_some_and(|s| !s.is_empty()),
        "{rec}"
    );
    assert_eq!(rec["config"], json!({"sims": 16, "seed": 1}));
    // Cached back onto the step.
    let step = call(
        &mut wb,
        "GET",
        &format!("/api/step/{combat_idx}"),
        Value::Null,
    );
    assert!(!step["scores"].as_array().unwrap().is_empty());
    let rec0 = call(
        &mut wb,
        "POST",
        "/api/recommend",
        json!({"idx": 0, "config": {}}),
    );
    assert!(
        rec0["error"].is_string(),
        "off combat: an error payload, {rec0}"
    );
    let e = err(
        &mut wb,
        "POST",
        "/api/search_tree",
        json!({"idx": 0, "config": {}}),
    );
    assert_eq!(
        e,
        (
            409,
            "search tree only available for combat decisions".to_string()
        )
    );
    let tree = call(
        &mut wb,
        "POST",
        "/api/search_tree",
        json!({"idx": combat_idx, "config": {"sims": 16}}),
    );
    assert!(tree["nodes"].as_array().is_some_and(|n| !n.is_empty()));
    assert_eq!(
        err(&mut wb, "POST", "/api/recommend", json!({"idx": "x"})).0,
        422
    );

    let review = call(
        &mut wb,
        "POST",
        "/api/regret-review",
        json!({"config": {"sims": 8, "top_n": 1}}),
    );
    assert!(
        review["decisions"]
            .as_array()
            .is_some_and(|d| !d.is_empty()),
        "{review}"
    );
    assert_eq!(review["config"], json!({"blunder_pct": 8.0, "top_n": 1}));

    // Replay path tops up the played move like the live coach: at this tiny
    // budget (64 sims) a re-searched, hand-played row has < PLAYED_MIN_VISITS visits, and
    // without the top-up it grades "uncertain" and vanishes from the review.
    let recomputed = call(
        &mut wb,
        "POST",
        "/api/regret-review",
        json!({"config": {"sims": 64, "recompute": true}}),
    );
    let ds = recomputed["decisions"].as_array().expect("decisions");
    assert!(!ds.is_empty(), "{recomputed}");
    assert!(
        ds.iter().all(|d| d["verdict"] != json!("uncertain")),
        "replay review left an ungraded decision: {recomputed}"
    );

    // MCTS Runs over the host's logs.
    let logs = vec![(log.filename.clone(), log.text.clone())];
    let runs = Workbench::mcts_runs(&logs);
    assert_eq!(runs["runs"][0]["path"], json!(log.filename));
    assert_eq!(
        runs["runs"][0]["stem"],
        json!(log.filename.trim_end_matches(".jsonl"))
    );
    let agg = Workbench::aggregate_runs(&logs);
    assert_eq!(agg["n_runs"], json!(1));

    // Play from here: a live run again, at the recorded step.
    let e = err(
        &mut wb,
        "POST",
        "/api/live/from_replay",
        json!({"idx": 999}),
    );
    assert_eq!(e.0, 404);
    let resumed = call(
        &mut wb,
        "POST",
        "/api/live/from_replay",
        json!({"idx": combat_idx}),
    );
    assert_eq!(resumed["decision"], json!("combat_play"));
    let live_tl = call(&mut wb, "GET", "/api/live/timeline", Value::Null);
    let live_labels: Vec<&str> = live_tl["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["label"].as_str().unwrap_or(""))
        .collect();
    let log_labels: Vec<&str> = steps
        .iter()
        .take(combat_idx as usize + 1)
        .map(|s| s["label"].as_str().unwrap_or(""))
        .collect();
    assert_eq!(
        live_labels.len() as i64,
        combat_idx + 1,
        "live {live_labels:?} vs log {log_labels:?}"
    );
    // The live deck analysis and the reward preview work on it.
    let live_da = call(&mut wb, "GET", "/api/live/deck-analysis", Value::Null);
    let step_da = call(
        &mut wb,
        "GET",
        &format!("/api/step/{combat_idx}/deck-analysis"),
        Value::Null,
    );
    assert_eq!(
        live_da, step_da,
        "the resumed run grades the recorded deck identically"
    );
    let preview = call(
        &mut wb,
        "POST",
        "/api/deck/reward-preview",
        json!({"deck": [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH"}], "candidates": [{"index": 0, "id": "INFLAME"}, {"index": 1, "id": "CLEAVE", "upgraded": true}]}),
    );
    assert_eq!(preview["candidates"].as_array().unwrap().len(), 2);
    assert_eq!(preview["candidates"][1]["index"], json!(1));
    assert!(preview["base"]["damage"].is_number());
}

#[test]
fn an_unresumable_log_says_so() {
    let mut wb = new_wb();
    // A log without the identity head or replayable actions.
    let text = [
        r#"{"step":1,"ts":"t","type":"state","data":{"decision":"map_select","floor":0,"act":1,"player":{"deck":[{"id":"STRIKE_IRONCLAD"}]}}}"#,
        r#"{"step":1,"ts":"t","type":"action","data":{"cmd":"action","action":"select_map_node","args":{"col":0,"row":1},"label":"go"}}"#,
        r#"{"step":2,"ts":"t","type":"state","data":{"decision":"game_over","victory":false}}"#,
        "",
    ]
    .join("\n");
    let loaded = wb.load_run("foreign.jsonl", &text).unwrap();
    assert_eq!(loaded["count"], json!(2));
    assert_eq!(loaded["resume"]["kind"], Value::Null);
    let e = err(&mut wb, "POST", "/api/live/from_replay", json!({"idx": 0}));
    assert_eq!(e.0, 409);
    assert!(e.1.contains("cannot be resumed"));
    let rec = call(&mut wb, "POST", "/api/recommend", json!({"idx": 0}));
    assert!(rec["error"].as_str().unwrap().contains("cannot be resumed"));
    let wpa = call(&mut wb, "POST", "/api/wpa-review", json!({}));
    assert_eq!(wpa["available"], json!(false));
    let tl = call(&mut wb, "GET", "/api/timeline", Value::Null);
    assert_eq!(tl["steps"][0]["label"], json!("go"));
}

/// Rewrites a recorded run so every combat step carries ONE synthetic score
/// block: the played move at `played_visits` (not best) beside a best row.
fn with_synthetic_rows(text: &str, played_visits: i64) -> String {
    let lines: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).expect("jsonl"))
        .filter(|l: &Value| l["type"] != json!("mcts"))
        .collect();
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        out.push(l.to_string());
        if l["type"] != json!("state") || l["data"]["decision"] != json!("combat_play") {
            continue;
        }
        let Some(act) = lines[i + 1..].iter().find(|x| x["type"] == json!("action")) else {
            continue;
        };
        let (Some(idx), _) = sts2_workbench::advisors::action_stable_key(&act["data"]["args"])
        else {
            continue;
        };
        let rows = json!([
            {"stable_idx": idx, "target_index": act["data"]["args"]["target_index"],
             "action": "played", "win_prob": 0.2, "visits": played_visits, "q": 0.0,
             "is_best": false},
            {"stable_idx": 9999, "target_index": null, "action": "best",
             "win_prob": 0.9, "visits": 5000, "q": 0.5, "is_best": true},
        ]);
        out.push(
            json!({"step": l["step"], "ts": "t", "type": "mcts",
                   "data": {"source": "agent", "root_outcomes": rows}})
            .to_string(),
        );
    }
    out.join("\n") + "\n"
}

fn review_of(text: &str) -> Value {
    let mut wb = new_wb();
    wb.load_run("x.jsonl", text).expect("load");
    call(
        &mut wb,
        "POST",
        "/api/regret-review",
        json!({"config": {"sims": 64}}),
    )
}

#[test]
fn recorded_thin_played_rows_are_topped_up() {
    let mut wb = new_wb();
    let log = play_and_quit(&mut wb, "run_12", 40);
    let min = i64::from(u8::try_from(sts2_workbench::advisors::PLAYED_MIN_VISITS).unwrap());

    // Thin played rows: the review re-searches them, so none stay "uncertain".
    let topped = review_of(&with_synthetic_rows(&log.text, 2));
    let ds = topped["decisions"].as_array().expect("decisions");
    assert!(!ds.is_empty(), "{topped}");
    for d in ds {
        assert_ne!(d["verdict"], json!("uncertain"), "{d}");
    }

    // Rows already at the floor are graded as recorded (no rebuild, no search).
    let kept = review_of(&with_synthetic_rows(&log.text, min));
    let ds = kept["decisions"].as_array().expect("decisions");
    assert!(!ds.is_empty(), "{kept}");
    assert!(
        ds.iter().all(|d| d["verdict"] != json!("uncertain")),
        "{kept}"
    );
}
