#!/usr/bin/env python3
"""Record the state-import COMBAT-LANE golden for the Rust port (GH #708).

The desktop workbench turns a SpireMentor fight paste into an engine wire dict
through ``realgame_adapter.to_headless_combat_play`` + ``sts2_hint.adapter.translate``
(``workbench.state_import.parse_pasted_state``). The browser port
(``sts2-rs/crates/sts2-workbench/src/combat_import.rs``) re-implements that
translation in Rust; this script records the Python side's answer for every
real capture under ``tests/fixtures/realgame_adapter/`` so a Rust nextest test
(``tests/state_import_combat_parity.rs``) can compare wire dicts field by field.

Usage (from ``sts2-pycli/``)::

    PYTHONPATH=python .venv/bin/python scripts/dump_state_import_golden.py

Recording format: ``{"engine": {...}, "cases": [{name, paste, has_combat,
combat_wire, warnings, exact}]}`` — ``paste`` is the fixture's ``state`` as the
JSON text the user would paste, ``combat_wire`` the engine wire dict (or null),
``warnings`` / ``exact`` the parser's fidelity claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sts2_rs
from workbench.state_import import parse_pasted_state

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "realgame_adapter"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[2]
    / "sts2-rs"
    / "crates"
    / "sts2-workbench"
    / "tests"
    / "golden"
    / "state_import_combat.json"
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Record the state-import combat-lane golden.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    cases = []
    for path in sorted(FIXTURES.glob("*.json")):
        state = json.load(path.open(encoding="utf-8"))["state"]
        paste = json.dumps(state, sort_keys=True)
        imported = parse_pasted_state(paste)
        fidelity = imported.fidelity()
        cases.append(
            {
                "name": path.stem,
                "paste": paste,
                "surface": imported.surface,
                "has_combat": imported.has_combat,
                "combat_wire": imported.combat_wire,
                "warnings": list(imported.warnings),
                "exact": fidelity["exact"],
            }
        )
    doc = {
        "engine": {"version": sts2_rs.version(), "src_hash": sts2_rs.src_hash()},  # ty: ignore[unresolved-attribute]
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} ({len(cases)} cases, {sum(c['has_combat'] for c in cases)} with a fight)"
    )


if __name__ == "__main__":
    main()
