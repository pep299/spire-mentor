"""`ppo.turnplan.label`: the G0 turn-plan label generator (GH #830).

The generator plays the deployed pilot through dev-pool fights and records
every enumerated plan with its teacher price at each turn start. These tests
pin the train/holdout split, the dev_v1 exclusion, one label per turn start,
resume and the shard reader. The fights are small hallway fights played at a
tiny pilot budget, so a test takes seconds, not the deployed minutes.
"""

from __future__ import annotations

import gzip
import itertools
import json
import zlib
from pathlib import Path
from typing import Any

import pytest
from ppo.combat_bench.run import load_set

pytest.importorskip("sts2_rs")

from ppo.turnplan import label

BENCH = Path(__file__).resolve().parents[2] / "data" / "combat_bench"


def _weak_fights(n: int) -> list[dict[str, Any]]:
    """The first `n` devpool `_WEAK` hallway fights outside dev_v1."""
    screened = {f["key"] for f in load_set(str(BENCH / "ironclad_dev_v1.jsonl"))}
    pool = load_set(str(BENCH / "ironclad_devpool_v1.jsonl"))
    weak = [f for f in pool if f["key"] not in screened and f["encounter"].endswith("_WEAK")]
    return weak[:n]


def _write_set(path: Path, fights: list[dict[str, Any]]) -> str:
    path.write_text("".join(json.dumps(f) + "\n" for f in fights), encoding="utf-8")
    return str(path)


def _run(tmp: Path, set_path: str, *extra: str, exclude: str | None = None) -> int:
    argv = [
        "--set",
        set_path,
        "--exclude",
        exclude or _write_set(tmp / "none.jsonl", []),
        "--reps",
        "1",
        "--sims-per-plan",
        "20",
        "--hallway-p",
        "1.0",
        "--workers",
        "2",
        "--pilot-sims",
        "40",
        "--pilot-boss-sims",
        "40",
        "--out",
        str(tmp / "out"),
        *extra,
    ]
    return label.main(argv)


def _raw_lines(out: Path) -> list[dict[str, Any]]:
    rows = []
    for shard in sorted(out.glob("shard-*.jsonl.gz")):
        rows += list(label.iter_shard(shard))
    return rows


def test_split_is_by_file_and_deterministic() -> None:
    files = [f"{i:016x}.run.gz" for i in range(200)]
    splits = [label.split_of(f) for f in files]
    assert splits == [label.split_of(f) for f in files]
    for f, s in zip(files, splits, strict=True):
        assert s == ("holdout" if zlib.crc32(f.encode()) % 5 == 0 else "train")
    assert {"train", "holdout"} == set(splits)


def test_excluded_keys_never_labelled(tmp_path: Path) -> None:
    fights = _weak_fights(3)
    set_path = _write_set(tmp_path / "set.jsonl", fights)
    exclude = _write_set(tmp_path / "exclude.jsonl", [fights[1]])

    assert _run(tmp_path, set_path, exclude=exclude) == 0

    rows = list(label.read_positions(tmp_path / "out"))
    keys = {r["key"] for r in rows}
    assert fights[1]["key"] not in keys
    assert keys == {fights[0]["key"], fights[2]["key"]}
    assert fights[1]["key"] not in {r["key"] for r in _raw_lines(tmp_path / "out")}


def test_one_label_per_turn_start(tmp_path: Path) -> None:
    fight = _weak_fights(1)[0]
    set_path = _write_set(tmp_path / "set.jsonl", [fight])

    assert _run(tmp_path, set_path) == 0

    rows = _raw_lines(tmp_path / "out")
    assert rows, "a hallway fight at hallway_p 1.0 labels at least its first turn"
    turns = [r["turn"] for r in rows]
    assert turns[0] == 1
    assert all(b > a for a, b in itertools.pairwise(turns)), turns
    for r in rows:
        assert (r["key"], r["rep"], r["file"]) == (fight["key"], 0, fight["file"])
        assert r["room_type"] == fight["room_type"]
        assert r["split"] == label.split_of(fight["file"])
        assert r["n_plans"] == len(r["plans"]) >= 1
        assert isinstance(r["cap_hit"], bool)
    (cell,) = label.read_cells(tmp_path / "out").values()
    assert cell["positions"] == len(rows)
    assert (tmp_path / "out" / "header.json").is_file()
    assert "positions/s" in (tmp_path / "out" / "progress.log").read_text()


def test_hallway_p_zero_labels_no_hallway_turn(tmp_path: Path) -> None:
    fight = _weak_fights(1)[0]
    set_path = _write_set(tmp_path / "set.jsonl", [fight])

    assert _run(tmp_path, set_path, "--hallway-p", "0.0") == 0

    assert _raw_lines(tmp_path / "out") == []
    (cell,) = label.read_cells(tmp_path / "out").values()
    assert cell["positions"] == 0 and cell["end"] in {"won", "died"}


def test_resume_skips_done_cells(tmp_path: Path) -> None:
    fights = _weak_fights(2)
    set_path = _write_set(tmp_path / "set.jsonl", fights)
    assert _run(tmp_path, set_path, "--limit", "1") == 0
    first = _raw_lines(tmp_path / "out")
    assert {r["key"] for r in first} == {fights[0]["key"]}

    assert _run(tmp_path, set_path) == 0

    rows = _raw_lines(tmp_path / "out")
    # The done cell is not replayed: its lines appear once, the new cell's too.
    assert [r for r in rows if r["key"] == fights[0]["key"]] == first
    assert {r["key"] for r in rows} == {fights[0]["key"], fights[1]["key"]}
    assert set(label.read_cells(tmp_path / "out")) == {(f["key"], 0) for f in fights}
    # A third run has nothing to play and writes nothing.
    assert _run(tmp_path, set_path) == 0
    assert _raw_lines(tmp_path / "out") == rows


def test_resume_refuses_a_changed_config(tmp_path: Path) -> None:
    fights = _weak_fights(1)
    set_path = _write_set(tmp_path / "set.jsonl", fights)
    assert _run(tmp_path, set_path) == 0
    assert _run(tmp_path, set_path, "--sims-per-plan", "30") == 2


def test_reader_reads_each_cell_from_its_ledger_shard(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    a1, a2 = {"key": "a", "rep": 0, "turn": 1}, {"key": "a", "rep": 0, "turn": 2}
    b1 = {"key": "b", "rep": 0, "turn": 1}

    def shard(name: str, rows: list[dict[str, Any]], cut: int = 0) -> None:
        data = gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode())
        # A killed cell leaves a stream with no gzip trailer.
        (out / name).write_bytes(data[: len(data) - cut])

    shard("shard-0-0.jsonl.gz", [a1, a2], cut=6)  # the ledger's shard for a, cut
    shard("shard-0-1.jsonl.gz", [a1, a2, b1])  # a cut-off attempt no ledger line names
    shard("shard-1-0.jsonl.gz", [b1])  # b's replay
    (out / "cells.jsonl").write_text(
        json.dumps({"key": "a", "rep": 0, "end": "won", "shard": "shard-0-0.jsonl.gz"})
        + '\n{"key": "b", "re\n'  # a torn line in the middle is skipped
        + json.dumps({"key": "b", "rep": 0, "end": "watchdog", "shard": "shard-0-1.jsonl.gz"})
        + "\n"
        + json.dumps({"key": "b", "rep": 0, "end": "won", "shard": "shard-1-0.jsonl.gz"})
        + "\n"
        + json.dumps({"key": "c", "rep": 0, "end": "won", "shard": None})
        + "\n"
    )

    # The cut stream yields what it has (`a1`, maybe `a2`); b comes only from its last shard.
    got = list(label.read_positions(out))
    assert got[-1] == b1 and got.count(b1) == 1
    assert got[:-1] in ([a1], [a1, a2])


def test_a_cell_killed_mid_run_is_replayed_and_read_once(tmp_path: Path) -> None:
    fight = _weak_fights(1)[0]
    set_path = _write_set(tmp_path / "set.jsonl", [fight])
    out = tmp_path / "out"
    assert _run(tmp_path, set_path) == 0
    first = list(label.read_positions(out))
    assert first
    # Simulate a parent killed before it booked the cell: the ledger line is
    # torn, so the cell is not done, and its shard is an orphan.
    ledger = (out / "cells.jsonl").read_text()
    (out / "cells.jsonl").write_text(ledger[: len(ledger) // 2])

    assert _run(tmp_path, set_path) == 0

    assert len(list(out.glob("shard-*.jsonl.gz"))) == 2
    (cell,) = label.read_cells(out).values()
    assert cell["shard"].startswith("shard-1-")
    again = list(label.read_positions(out))
    turns = [r["turn"] for r in again]
    assert len(turns) == len(set(turns)) == len(first)


def test_a_cell_over_the_timeout_is_booked_as_watchdog(tmp_path: Path) -> None:
    fights = _weak_fights(2)
    set_path = _write_set(tmp_path / "set.jsonl", fights)

    assert _run(tmp_path, set_path, "--cell-timeout-s", "0.01") == 0

    cells = label.read_cells(tmp_path / "out")
    assert set(cells) == {(f["key"], 0) for f in fights}
    assert {c["end"] for c in cells.values()} == {"watchdog"}
    assert "watchdog" in (tmp_path / "out" / "progress.log").read_text()
    # A resume skips them unless asked to retry them.
    assert _run(tmp_path, set_path) == 0
    assert {c["end"] for c in label.read_cells(tmp_path / "out").values()} == {"watchdog"}
    assert _run(tmp_path, set_path, "--retry-ends", "watchdog") == 0
    assert {c["end"] for c in label.read_cells(tmp_path / "out").values()} <= {"won", "died"}
