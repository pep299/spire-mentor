//! `dominance_pruning` on the position that reported the defect.
//!
//! Defect A10 seed 3, floor 2, turn 2: Beam Cell (0 cost, 3 damage, 1
//! Vulnerable) and Strike vs one enemy. Beam Cell first proves at least as
//! good as Strike first — every Strike-first line that plays Beam Cell later
//! is matched by the same line with Beam Cell moved to the front, for 1-3
//! more damage and an otherwise identical state. The mean backup still
//! ranked Strike above Beam Cell on every seed (GH #492), because Strike's
//! Q averaged over its own dominated `Strike, …, Beam Cell` lines. With
//! those lines pruned, Beam Cell must come out ahead.

use serde_json::{json, Value};
use sts2_workbench::api::{Config, Workbench};

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

/// Root Q of the Beam Cell and Strike children.
fn root_q(wb: &mut Workbench, seed: u64, pruning: bool) -> (f64, f64) {
    let tree = wb
        .call(
            "POST",
            "/api/live/search_tree",
            &json!({"config": {"sims": 2000, "seed": seed, "dominance_pruning": pruning}}),
        )
        .unwrap_or_else(|e| panic!("search_tree: {} {}", e.status, e.detail));
    let nodes = tree["nodes"].as_array().unwrap();
    let q = |label: &str| {
        nodes[0]["children"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| &nodes[usize::try_from(c.as_u64().unwrap()).unwrap()])
            .find(|n| n["label"] == label)
            .unwrap_or_else(|| panic!("no root child {label}"))["q"]
            .as_f64()
            .unwrap()
    };
    (q("BeamCell"), q("StrikeDefect"))
}

#[test]
fn vulnerable_first_outranks_strike_first() {
    let mut wb = Workbench::new(Config::default());
    wb.restore_save(&save(), sts2_workbench::runlog::DEFAULT_NOW_ISO)
        .unwrap_or_else(|e| panic!("restore: {} {}", e.status, e.detail));
    for seed in [1, 2] {
        let (beam, strike) = root_q(&mut wb, seed, true);
        assert!(
            beam > strike,
            "seed {seed}: Q(BeamCell) {beam} <= Q(Strike) {strike} with pruning on"
        );
    }
}
