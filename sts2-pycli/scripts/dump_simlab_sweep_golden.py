#!/usr/bin/env python3
"""Record the SimLab sweep golden for the Rust port (GH #707).

The Python sweep runner (``simlab.runner.run_sweep`` over the ``sts2_rs``
engine) is the ORACLE for the browser-only port in
``sts2-rs/crates/sts2-workbench/src/simlab_sweep.rs``. This script runs two
small sweeps single-process — an axis sweep (encounters × ascensions × decks ×
every strategy) and a start-state sweep from a dealt board — and records the
telemetry rows plus the ``/api/sweeps/{id}/results`` payload (plain and with a
baseline) exactly as the desktop server answers them. A Rust nextest test
(``tests/sweep_parity.rs``) replays the same specs through the port and
compares.

Usage (from ``sts2-pycli/``)::

    PYTHONPATH=python .venv/bin/python scripts/dump_simlab_sweep_golden.py
    PYTHONPATH=python .venv/bin/python scripts/dump_simlab_sweep_golden.py --out /tmp/x.json

Recording format (THE CONTRACT with the Rust test)
==================================================

One JSON document, ``sweeps: [{name, spec, rows, results, results_baseline,
baseline}]``:

* ``spec`` — the ``SweepSpec.to_json()`` the Rust side parses.
* ``rows`` — ``play_combat``'s rows in dispatch order, with ``wall_ms``
  removed (a timing; the only host-dependent field).
* ``results`` / ``results_baseline`` — the results endpoint's body, without
  the per-bucket ``throughput`` block (derived from ``wall_ms``), floats
  rounded to 4 decimals (Python ``round``).

Everything else is compared exactly by the Rust test except the KDE curve,
which it compares within a tolerance (the two sides sum floats in a different
order).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import sts2_rs
from fastapi.testclient import TestClient
from simlab.runner import run_sweep
from simlab.specs import SweepSpec
from simlab.web.app import create_app

DEFAULT_OUT = (
    Path(__file__).resolve().parents[2]
    / "sts2-rs"
    / "crates"
    / "sts2-workbench"
    / "tests"
    / "golden"
    / "simlab_sweep.json"
)

AXIS_SPEC: dict[str, Any] = {
    "name": "axis",
    "encounters": ["FUZZY_WURM_CRAWLER_WEAK", "monsters:Nibbit,Nibbit"],
    "ascensions": [0, 10],
    "strategies": ["random", "block_first", "attack_first", "min_overblock", "mcts"],
    "n_seeds": 6,
    "decks": [
        ["ironclad_starter", ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]],
        [
            "aggro_rush",
            ["STRIKE_IRONCLAD"] * 4
            + ["ANGER"] * 2
            + ["TWIN_STRIKE", "UPPERCUT", "POMMEL_STRIKE", "THUNDERCLAP", "HEMOKINESIS"]
            + ["DEFEND_IRONCLAD"] * 2,
        ],
    ],
    "mcts_sims": 24,
    "mcts_depth": 30,
}


def _normalize(obj: Any) -> Any:
    """Sort keys, round floats to 4 decimals, drop the timing-derived blocks."""
    if isinstance(obj, dict):
        return {
            k: _normalize(v) for k, v in sorted(obj.items()) if k not in ("throughput", "wall_ms")
        }
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def _start_state_spec() -> dict[str, Any]:
    """A start-state sweep from the same dealt board the axis sweep starts on."""
    board = sts2_rs.build_scenario_state(  # ty: ignore[unresolved-attribute]
        seed=1,
        character="Ironclad",
        deck=["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"],
        hp=80,
        max_hp=80,
        relics=[],
        encounter="MAWLER_NORMAL",
        ascension=3,
        potions=[],
    )
    wire = json.dumps(sts2_rs.state_to_dict(board), sort_keys=True)  # ty: ignore[unresolved-attribute]
    return {
        "name": "position",
        "encounters": [],
        "ascensions": [],
        "strategies": ["random", "min_overblock", "mcts"],
        "n_seeds": 5,
        "decks": [],
        "mcts_sims": 16,
        "mcts_depth": 30,
        "start_states": [["as is", wire]],
        "positions": [],
    }


def _record(spec_json: dict[str, Any], baseline: str) -> dict[str, Any]:
    sweep = SweepSpec.from_json(spec_json)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_sweep(sweep, out_dir=root / sweep.name, jobs=1)
        rows = [
            json.loads(line)
            for line in (root / sweep.name / "rows.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        client = TestClient(create_app(results_root=root, jobs=1, serve_spa=False))
        plain = client.get(f"/api/sweeps/{sweep.name}/results")
        plain.raise_for_status()
        with_baseline = client.get(
            f"/api/sweeps/{sweep.name}/results", params={"baseline": baseline}
        )
        with_baseline.raise_for_status()
    return {
        "name": sweep.name,
        "spec": sweep.to_json(),
        "rows": _normalize(rows),
        "results": _normalize(plain.json()),
        "baseline": baseline,
        "results_baseline": _normalize(with_baseline.json()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Record the SimLab sweep golden for the Rust port.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    doc = {
        "engine": {"version": sts2_rs.version(), "src_hash": sts2_rs.src_hash()},  # ty: ignore[unresolved-attribute]
        "sweeps": [_record(AXIS_SPEC, "random"), _record(_start_state_spec(), "random")],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    n = sum(len(s["rows"]) for s in doc["sweeps"])
    print(f"wrote {args.out} ({n} rows)")


if __name__ == "__main__":
    main()
