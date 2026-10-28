import json

from game_log import GameLogger


def test_log_mcts_writes_typed_record(tmp_path):
    lg = GameLogger("ironclad", "SEED", log_dir=str(tmp_path))
    lg.log_state({"decision": "combat_play", "round": 1})
    lg.log_mcts(
        {
            "decision_idx": 1,
            "config": {"sims": 10, "c_puct": 1.0, "depth": 20, "seed": 1},
            "total_visits": 10,
            "recommended_stable_idx": -1,
            "recommended_label": "End turn",
            "pv": ["End turn"],
            "root_outcomes": [
                {
                    "stable_idx": -1,
                    "action_label": "End turn",
                    "target_index": None,
                    "visits": 10,
                    "q": -1.0,
                    "wins": 0,
                    "losses": 10,
                    "ties": 0,
                    "is_best": True,
                },
            ],
        }
    )
    lg.close()
    with open(lg.path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    mcts = [e for e in lines if e["type"] == "mcts"]
    assert len(mcts) == 1
    assert mcts[0]["step"] == 1  # shares the preceding state's step
    assert mcts[0]["data"]["root_outcomes"][0]["losses"] == 10


def test_log_meta_extra_writes_provenance(tmp_path):
    lg = GameLogger("ironclad", "SEED", log_dir=str(tmp_path))
    lg.log_meta_extra(
        {
            "fixture_stem": "floor017_vantom_boss",
            "build_id": "v0.99.1",
            "encounter": "VANTOM_BOSS",
            "sims": 3000,
        }
    )
    lg.close()
    with open(lg.path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    metas = [e for e in lines if e["type"] == "meta"]
    # head (constructor) + the provenance record
    extra = next(e for e in metas if e["data"].get("fixture_stem"))
    assert extra["data"]["build_id"] == "v0.99.1"
    assert extra["data"]["sims"] == 3000


def test_log_summary_writes_summary_meta(tmp_path):
    lg = GameLogger("ironclad", "SEED", log_dir=str(tmp_path))
    lg.log_summary(
        {"final_won": True, "final_turn": 6, "final_player_hp": 41, "final_player_max_hp": 70}
    )
    lg.close()
    with open(lg.path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    summary = next(e["data"] for e in lines if e["type"] == "meta" and e["data"].get("summary"))
    assert summary["final_won"] is True
    assert summary["final_turn"] == 6
    assert summary["final_player_hp"] == 41
