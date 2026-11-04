//! The search-tree inspector over a transposed search graph.
//!
//! The transposition table is always on, so one node can hang under several
//! parents, reached by a different action from each. The serialized tree
//! used to stamp ONE incoming label per node (whichever parent was walked
//! last), so the inspector showed `Zap → Zap` where the second edge was
//! really Beam Cell (bug report: the `use_vuln_first` save below). Each
//! node now carries `edges[]` parallel to `children`; this pins that every
//! edge's label matches what the edge did to the state.

use serde_json::{json, Value};
use sts2_workbench::api::{Config, Workbench};

/// Defect A10 seed 3, floor 2, turn 2: hand Strike x3, Beam Cell, Zap with
/// 3 energy vs one enemy. Zap→BeamCell and BeamCell→Zap reach one state.
fn save() -> Value {
    let p = |a: &str, args: Value| json!({"action": a, "args": args});
    let card = |i: u32| p("play_card", json!({"card_index": i}));
    let hit = |i: u32| p("play_card", json!({"card_index": i, "target_index": 0}));
    let end = || p("end_turn", json!({}));
    let line = vec![
        p("choose_option", json!({"option_index": 1})),
        p("select_map_node", json!({"col": 0, "row": 1})),
        hit(0),
        hit(3),
        card(2),
        end(),
        card(1),
        card(1),
        hit(2),
        end(),
        card(4),
        hit(3),
        p("select_card_reward", json!({"card_index": 0})),
        p("select_map_node", json!({"col": 0, "row": 2})),
        hit(2),
        card(0),
        card(0),
        end(),
    ];
    json!({
        "schema_version": 1, "name": "use_vuln_first", "saved_at": "2026-10-18T12:58:08Z",
        "character": "Defect", "seed": "3", "ascension": 10, "backend": "rust",
        "run_mode": false, "act1": "roll", "step_count": line.len(),
        "progress": {"act": 1, "floor": 2, "hp": 52, "max_hp": 75},
        "line": line, "node_values": [],
    })
}

fn summary(n: &Value) -> (i64, i64) {
    let s = &n["summary"];
    (
        s["energy"].as_i64().unwrap(),
        s["enemies"][0]["hp"].as_i64().unwrap(),
    )
}

#[test]
fn shared_nodes_are_labelled_per_edge() {
    let mut wb = Workbench::new(Config::default());
    wb.restore_save(&save(), sts2_workbench::runlog::DEFAULT_NOW_ISO)
        .unwrap_or_else(|e| panic!("restore: {} {}", e.status, e.detail));
    let tree = wb
        .call(
            "POST",
            "/api/live/search_tree",
            &json!({"config": {"sims": 2000, "seed": 1}}),
        )
        .unwrap_or_else(|e| panic!("search_tree: {} {}", e.status, e.detail));
    let nodes = tree["nodes"].as_array().unwrap();
    let node = |id: &Value| &nodes[usize::try_from(id.as_u64().unwrap()).unwrap()];

    let mut parents = vec![0_u32; nodes.len()];
    let (mut zap, mut beam) = (0, 0);
    for n in nodes {
        let edges = n["edges"].as_array().unwrap();
        assert_eq!(edges.len(), n["children"].as_array().unwrap().len());
        let mut through = 0;
        for (e, c) in edges.iter().zip(n["children"].as_array().unwrap()) {
            assert_eq!(&e["child"], c, "edges[] must be parallel to children[]");
            let child = node(c);
            parents[usize::try_from(c.as_u64().unwrap()).unwrap()] += 1;
            // An edge's traffic is part of the child's (which sums all parents).
            assert!(e["visits"].as_u64().unwrap() <= child["visits"].as_u64().unwrap());
            through += e["visits"].as_u64().unwrap();
            if n["node_kind"] != "decision" || child["node_kind"] != "decision" {
                continue;
            }
            // Same turn only: an edge into a decision node is a card play.
            let ((e0, hp0), (e1, hp1)) = (summary(n), summary(child));
            match e["label"].as_str().unwrap() {
                "Zap" => {
                    zap += 1;
                    assert_eq!(
                        (e1, hp1),
                        (e0 - 1, hp0),
                        "a Zap edge spends 1 energy, deals 0"
                    );
                }
                "BeamCell" => {
                    beam += 1;
                    assert_eq!(e1, e0, "a Beam Cell edge is free");
                    assert!(hp1 < hp0, "a Beam Cell edge deals damage");
                }
                _ => {}
            }
        }
        if n["node_kind"] == "decision" {
            assert!(through <= n["visits"].as_u64().unwrap());
        }
    }
    assert!(zap > 0 && beam > 0, "fixture lost its Zap/Beam Cell edges");
    assert!(
        parents.iter().any(|&p| p > 1),
        "fixture no longer transposes; the test would pass vacuously"
    );
}
