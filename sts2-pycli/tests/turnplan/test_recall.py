"""`ppo.turnplan.recall`: the G0 recall report and verdict (GH #830).

Pins the recall@K / anchor-union / regret arithmetic on hand-made positions,
the SE-based `resolved` subset, that capped positions are counted but never
measured, the verdict rule on both sides of each clause, and that the CLI
writes both report files from two trained scorers. Every record is synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ppo.turnplan import data, recall, train
from test_train import _synthetic, _write_dir


def _pos(
    q: list[float],
    *,
    file: str = "a.run.gz",
    room: str = "boss",
    cap_hit: bool = False,
    visits: list[int] | None = None,
    q_var: list[float] | None = None,
    anchors: list[int] = (),
) -> data.Position:
    """A position with terminal plans (SE 0) unless `visits` / `q_var` say otherwise."""
    n = len(q)
    anc = np.zeros((n, len(data.ANCHORS)), bool)
    for i in anchors:
        anc[i, 0] = True
    return data.Position(
        key=f"{file}:1",
        rep=0,
        file=file,
        room_type=room,
        cap_hit=cap_hit,
        deck_cards=[],
        relics=[],
        powers=[],
        ctx=np.zeros(data.CTX_DIM, np.float32),
        plan_feats=np.zeros((n, data.PLAN_DIM), np.float32),
        plan_cards=[[] for _ in range(n)],
        q=np.asarray(q, np.float32),
        q_var=np.asarray(q_var if q_var is not None else [0.0] * n, np.float32),
        visits=np.asarray(visits if visits is not None else [0] * n, np.int64),
        logp_heur=np.zeros(n, np.float32),
        logp_rl2=np.zeros(n, np.float32),
        anchors=anc,
    )


def _ranked(best_rank: int, n: int = 12) -> tuple[list[float], np.ndarray]:
    """q with a unique best at plan 0, and scores putting it at 1-based `best_rank`."""
    q = [1.0] + [0.5 - 0.01 * i for i in range(n - 1)]
    order = list(range(1, n))
    order.insert(best_rank - 1, 0)  # order[r] = the plan at rank r
    scores = np.zeros(n, np.float32)
    for r, plan in enumerate(order):
        scores[plan] = n - r
    return q, scores


def _same(scores: list[np.ndarray]) -> dict[str, list[np.ndarray]]:
    """The same scores for every filter."""
    return {f: scores for f in recall.FILTERS}


def test_recall_at_k_handmade() -> None:
    positions, scores = [], []
    for i, rank in enumerate((1, 3, 7)):
        q, s = _ranked(rank)
        positions.append(_pos(q, file=f"f{i}.run.gz"))
        scores.append(s)
    rep = recall.report(positions, _same(scores), n_boot=50)

    cell = rep["subsets"]["all"]["filters"]["scorer_deck"]
    assert [cell[str(k)]["recall"]["est"] for k in recall.KS] == pytest.approx(
        [1 / 3, 2 / 3, 2 / 3, 1.0]
    )
    # Regret at K = 1: 0 for the rank-1 position, 1.0 - 0.5 for the other two.
    assert cell["1"]["regret"]["mean"] == pytest.approx(2 * 0.5 / 3)
    assert cell["10"]["regret"]["mean"] == pytest.approx(0.0)
    assert rep["subsets"]["all"]["n_positions"] == 3
    assert rep["subsets"]["all"]["n_files"] == 3
    # Same scores for every filter: every paired difference is exactly 0.
    d = rep["subsets"]["all"]["diffs"]["scorer_deck-rl2"]["6"]["recall"]
    assert (d["est"], d["lo"], d["hi"]) == (0.0, 0.0, 0.0)


def test_tie_at_the_top_is_a_hit_and_ranking_ties_break_by_index() -> None:
    # Plans 1 and 3 share the max q; a top-1 keeping plan 3 hits.
    q = [0.0, 1.0, 0.2, 1.0]
    kept = recall.kept_mask(np.array([0.0, 0.0, 0.0, 5.0]), 1)
    assert kept.tolist() == [False, False, False, True]
    assert recall.hit(np.asarray(q), kept) and recall.regret(np.asarray(q), kept) == pytest.approx(
        0.0
    )
    # Equal scores: the lower plan index wins; -inf (no rl2 prior) ranks last.
    assert recall.kept_mask(np.array([1.0, 1.0, 1.0, 1.0]), 2).tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert recall.kept_mask(np.array([-np.inf, 0.0, -np.inf]), 2).tolist() == [True, True, False]


def test_anchor_union_counts_anchor_hits() -> None:
    q, s = _ranked(12)
    p = _pos(q, anchors=[0])  # the best plan is an anchor but ranked last
    rep = recall.report([p], _same([s]), n_boot=20)
    cell = rep["subsets"]["all"]["filters"]["heur"]["1"]
    assert cell["recall"]["est"] == pytest.approx(0.0)
    assert cell["recall_anchors"]["est"] == pytest.approx(1.0)
    assert cell["regret"]["mean"] == pytest.approx(0.5)
    assert cell["regret_anchors"]["mean"] == pytest.approx(0.0)
    assert cell["kept_anchors_mean"] == pytest.approx(2.0)  # top-1 plus one anchor

    # An anchor inside the top K adds no width; one outside adds itself.
    s4 = np.array([3.0, 2.0, 1.0, 0.0])
    inside = np.array([[True], [False], [False], [False]])
    assert recall.kept_mask(s4, 2, inside).tolist() == [True, True, False, False]
    outside = np.array([[False], [False], [False], [True]])
    assert recall.kept_mask(s4, 2, outside).tolist() == [True, True, False, True]


def test_resolved_filter_uses_se() -> None:
    # Gap 0.2, SE of each sqrt(0.01 / 100) = 0.01 -> 2 * sqrt(2) * 0.01 = 0.028 < 0.2.
    assert recall.is_resolved(_pos([0.6, 0.4], visits=[100, 100], q_var=[0.01, 0.01]))
    # Same gap, q_var 4.0 -> SE 0.2 each -> threshold 0.566 > 0.2.
    assert not recall.is_resolved(_pos([0.6, 0.4], visits=[100, 100], q_var=[4.0, 4.0]))
    # Fewer visits widen the SE: q_var 0.01 at 1 visit -> SE 0.1, threshold 0.283.
    assert not recall.is_resolved(_pos([0.6, 0.4], visits=[1, 1], q_var=[0.01, 0.01]))
    # Terminal plans (visits 0) have SE 0: any strict gap resolves.
    assert recall.is_resolved(_pos([0.6, 0.59], visits=[0, 0], q_var=[0.0, 0.0]))
    # A tie at the top: the second-best is the best q strictly below the max.
    assert recall.is_resolved(_pos([0.6, 0.6, 0.1], visits=[0, 0, 0]))
    # All plans equal (or one plan): nothing to resolve.
    assert not recall.is_resolved(_pos([0.3, 0.3]))
    assert not recall.is_resolved(_pos([0.3]))

    resolved = _pos([0.6, 0.4], file="r.run.gz")
    loose = _pos([0.6, 0.4], file="l.run.gz", visits=[1, 1], q_var=[4.0, 4.0])
    s = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]
    rep = recall.report([resolved, loose], _same(s), n_boot=20)
    assert rep["subsets"]["resolved"]["n_positions"] == 1
    assert rep["subsets"]["resolved"]["filters"]["rl2"]["1"]["recall"]["est"] == pytest.approx(0.0)
    assert rep["subsets"]["all"]["filters"]["rl2"]["1"]["recall"]["est"] == pytest.approx(0.5)


def test_report_counts_cap_hit_positions_separately() -> None:
    q, good = _ranked(1)
    _, bad = _ranked(12)
    positions = [
        _pos(q, file="a.run.gz", room="boss"),
        _pos(q, file="b.run.gz", room="elite", cap_hit=True),
        _pos(q, file="c.run.gz", room="boss", cap_hit=True),
        _pos(q, file="d.run.gz", room="monster"),
    ]
    rep = recall.report(positions, _same([good, bad, bad, good]), n_boot=20)
    assert rep["n"]["cap_hit"] == {"boss": 1, "elite": 1}
    assert rep["n"]["positions"] == 4 and rep["n"]["measured"] == 2
    assert rep["subsets"]["all"]["n_positions"] == 2
    assert rep["subsets"]["all"]["filters"]["rl2"]["1"]["recall"]["est"] == pytest.approx(1.0)
    by_room = rep["subsets"]["all"]["by_room"]
    assert by_room["boss"]["n_positions"] == 1 and by_room["monster"]["n_positions"] == 1
    assert "elite" not in by_room


def test_bootstrap_is_by_run_file_and_deterministic() -> None:
    # Ten positions in ONE file: a file-cluster bootstrap cannot vary.
    q, s_hit = _ranked(1)
    _, s_miss = _ranked(12)
    positions = [_pos(q, file="one.run.gz") for _ in range(10)]
    scores = [s_hit if i % 2 else s_miss for i in range(10)]
    rep = recall.report(positions, _same(scores), n_boot=200)
    ci = rep["subsets"]["all"]["filters"]["scorer_deck"]["1"]["recall"]
    assert (ci["est"], ci["lo"], ci["hi"]) == pytest.approx((0.5, 0.5, 0.5))

    positions = [_pos(q, file=f"{i}.run.gz") for i in range(10)]
    a = recall.report(positions, _same(scores), n_boot=200, seed=7)
    b = recall.report(positions, _same(scores), n_boot=200, seed=7)
    ci = a["subsets"]["all"]["filters"]["scorer_deck"]["1"]["recall"]
    assert ci["lo"] < 0.5 < ci["hi"]
    assert a == b


def _verdict_inputs(r6a: float, lo: float, hi: float) -> dict:
    return {
        "subsets": {
            "resolved": {
                "n_positions": 100,
                "filters": {"scorer_deck": {"6": {"recall_anchors": {"est": r6a}}}},
                "diffs": {"scorer_deck-rl2": {"6": {"recall": {"est": 0.0, "lo": lo, "hi": hi}}}},
            }
        }
    }


def test_verdict_pass_and_kill() -> None:
    v = recall.verdict(_verdict_inputs(0.96, 0.01, 0.05))
    assert v["verdict"] == "PASS" and v["failing"] == []

    v = recall.verdict(_verdict_inputs(0.95, 0.001, 0.05))  # the bound is inclusive
    assert v["verdict"] == "PASS"

    v = recall.verdict(_verdict_inputs(0.94, 0.01, 0.05))
    assert v["verdict"] == "KILL" and len(v["failing"]) == 1
    assert "recall@(6 ∪ anchors)" in v["failing"][0]

    for lo, hi in ((0.0, 0.05), (-0.01, 0.05), (-0.05, -0.01)):
        v = recall.verdict(_verdict_inputs(0.99, lo, hi))
        assert v["verdict"] == "KILL" and len(v["failing"]) == 1, (lo, hi)
        assert "scorer_deck − rl2" in v["failing"][0]

    v = recall.verdict(_verdict_inputs(0.5, -0.1, 0.1))
    assert v["verdict"] == "KILL" and len(v["failing"]) == 2
    assert v["line"].startswith("KILL")

    empty = _verdict_inputs(0.99, 0.01, 0.05)
    empty["subsets"]["resolved"]["n_positions"] = 0
    assert recall.verdict(empty)["verdict"] == "KILL"


def test_cli_writes_json_and_md(tmp_path: Path) -> None:
    _write_dir(tmp_path / "d", _synthetic(120, seed=4))
    (tmp_path / "d" / "header.json").write_text(json.dumps({"sims_per_plan": 77}))
    d = str(tmp_path / "d")
    for name, extra in (("deck", []), ("nodeck", ["--no-deck"])):
        assert (
            train.main(["--data", d, "--out", str(tmp_path / name), "--epochs", "1", *extra]) == 0
        )

    out = tmp_path / "rep"
    args = ["--data", d, "--model", str(tmp_path / "deck"), "--model-nodeck"]
    args += [str(tmp_path / "nodeck"), "--out", str(out), "--n-boot", "50"]
    assert recall.main(args) == 0

    rep = json.loads((out / "recall.json").read_text(encoding="utf-8"))
    n_holdout = len(data.load([d], "holdout"))
    assert rep["n"]["positions"] == n_holdout > 0
    assert rep["qualifiers"]["sims_per_plan"] == [77]
    assert set(rep["subsets"]["all"]["filters"]) == set(recall.FILTERS)
    assert rep["verdict"]["verdict"] in ("PASS", "KILL")
    md = (out / "recall.md").read_text(encoding="utf-8")
    assert rep["verdict"]["line"] in md and "Ironclad A10" in md
