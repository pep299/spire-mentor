"""The turn-plan root filter in `combat_bench` (GH #830 root-filter screen).

Pins the per-cell `root_filter` record (`run.RootFilterLog`), its absence for
an agent without the knob, and, with the wheel, that a bench cell played
with `STS2_RUNAI_TURNPLAN_ROOT_FILTER` reports a filtered root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ppo.combat_bench import agents, run

BENCH = Path(__file__).resolve().parents[2] / "data" / "combat_bench"


def _rec(applied: bool, fallback: str | None, via: str | None) -> dict:
    return {
        "root_filter": {
            "applied": applied,
            "fallback": fallback,
            "legal_n": 5,
            "plans_n": 40,
            "kept_n": 9 if applied else 0,
            "allowed_n": 3 if applied else 5,
            "chosen_via": via,
            "filter_us": 1500,
        }
    }


def test_root_filter_log_counts_and_rows() -> None:
    log = run.RootFilterLog()
    log.step(_rec(True, None, "topk"), 0.25)
    log.step(_rec(True, None, "anchor_only"), 0.5)
    log.step(_rec(False, "cap", None), 0.1)
    log.step(_rec(False, "forced", None), 0.0)
    log.step({"root_filter": None}, 0.1)  # knob off: no row
    log.step(None, 0.1)
    s = log.summary()
    assert s["decisions"] == 4
    assert s["applied"] == 2
    assert (s["fallback_forced"], s["fallback_cap"], s["fallback_empty"]) == (1, 1, 0)
    assert (s["chosen_topk"], s["chosen_anchor_only"]) == (1, 1)
    assert (s["legal_n"], s["allowed_n"]) == (10, 6)  # applied decisions only
    assert s["filter_ms"] == pytest.approx(6.0)
    assert s["decide_ms"] == pytest.approx(850.0)
    assert s["rows"][0] == [1, "", 5, 40, 9, 3, "t", 1.5, 250.0]
    assert s["rows"][2] == [0, "c", 5, 40, 0, 5, "", 1.5, 100.0]
    json.dumps(s)


def test_an_empty_log_adds_no_field() -> None:
    assert run.RootFilterLog().rows == []


def _weak_fight() -> dict:
    return next(
        f
        for f in run.load_set(str(BENCH / "ironclad_devpool_v1.jsonl"))
        if f["encounter"].endswith("_WEAK")
    )


def _wheel_has_root_filter() -> bool:
    sts2_rs = pytest.importorskip("sts2_rs")
    rm = run.build_run(_weak_fight(), 1)
    rec = sts2_rs.MctsCombatAgent(seed=0, sims=10, boss_sims=10).step(rm)
    return rec is not None and "root_filter" in rec


def test_a_cell_without_the_knob_has_no_root_filter_field() -> None:
    if not _wheel_has_root_filter():
        pytest.skip("wheel predates the root filter")
    spec = agents.mcts_spec(20, 20, agents.build_env({}))
    agents.apply_env(spec["env"])
    rec = run.play_cell(_weak_fight(), 0, spec, run.Caps())
    assert rec["end"] in ("won", "died")
    assert "root_filter" not in rec


def test_a_cell_with_the_knob_reports_a_filtered_root(tmp_path: Path) -> None:
    if not _wheel_has_root_filter():
        pytest.skip("wheel predates the root filter")
    weights = (
        Path(__file__).resolve().parents[3]
        / "sts2-rs/crates/sts2-mcts/tests/data/turnplan_root_filter/root_filter.json"
    )
    env = agents.build_env({"STS2_RUNAI_TURNPLAN_ROOT_FILTER": f"{weights}:1"})
    spec = agents.mcts_spec(20, 20, env)
    agents.apply_env(spec["env"])
    try:
        rec = run.play_cell(_weak_fight(), 0, spec, run.Caps())
    finally:
        agents.apply_env({})
    assert rec["end"] in ("won", "died"), rec
    f = rec["root_filter"]
    assert f["decisions"] == len(f["rows"]) > 0
    assert f["applied"] > 0
    assert f["allowed_n"] < f["legal_n"]
    assert f["chosen_topk"] + f["chosen_anchor_only"] == f["applied"]


def test_book_classifies_a_decision() -> None:
    from ppo.turnplan import root_flip

    a = {"action": "play", "card_index": 0, "target_index": 0}
    b = {"action": "end_turn"}
    filt = {"fallback": None, "allowed": [[a, True]]}
    assert root_flip.book(filt, a) == "kept"
    assert root_flip.book(filt, b) == "pruned"
    assert root_flip.book({"fallback": "cap", "allowed": []}, a) == "fallback:cap"
    assert root_flip.book(None, a) == "fallback:offturn"


def test_summarize_flip_rates_by_room_and_entry() -> None:
    from ppo.turnplan import root_flip

    def row(b10: str, b6: str) -> dict:
        return {"b0_ms": 100.0, "filter_ms": {"10": 5.0, "6": 5.0}, "book": {"10": b10, "6": b6}}

    lines = [
        {"room_type": "boss", "rows": [row("kept", "pruned"), row("kept", "kept")]},
        {"room_type": "elite", "rows": [row("fallback:cap", "fallback:cap")]},
        {"room_type": "boss", "error": "boom"},
    ]
    s = root_flip.summarize(lines, [10, 6])
    assert s["fights"] == 3 and s["errors"] == 1
    assert s["K=6"]["all"]["decisions"] == 3
    assert s["K=6"]["all"]["pruned"] == 1
    assert s["K=6"]["entry"]["pruned"] == 1  # the first decision of a fight
    assert s["K=6"]["boss"]["flip_rate_of_filtered"] == pytest.approx(0.5)
    assert s["K=10"]["elite"]["fallback:cap"] == 1
    assert s["wall"]["wall_ratio_K=10"] == pytest.approx(1.05)
