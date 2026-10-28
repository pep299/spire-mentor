"""Leakage guard for scorers trained on pro replays (GH #830 root-filter screen).

No training turn may come from a run that supplies a fight to a measurement
panel: `dev_v1` (the screen), the `devpool_v1` holdout (G0's recall split,
`label.split_of`) or `pro_v1` (the test set). A replay is excluded when its
run hash equals a panel fight's run file stem, or when its (player, seed)
equals a panel run's (`player_id(username)`, the run file's `seed`).

The panels come from v0.107.1 run files and the replays are v0.111.0, so the
expected overlap is zero; the guard asserts it and reports the counts.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ppo.bc_human.replay import scraper_dir
from ppo.turnplan.label import split_of
from ppo.turnplan.pro import player_id

BENCH = Path(__file__).resolve().parents[3] / "data" / "combat_bench"
RUN_ROOT = scraper_dir("files-spire-codex")


def panel_rows(bench: Path = BENCH) -> list[dict[str, Any]]:
    """The fights of every measurement panel, each tagged with its `panel`."""
    rows = []
    for name, keep in (
        ("ironclad_dev_v1.jsonl", lambda r: True),
        ("ironclad_devpool_v1.jsonl", lambda r: split_of(r["file"]) == "holdout"),
        ("ironclad_pro_v1.jsonl", lambda r: True),
    ):
        for line in (bench / name).open():
            r = json.loads(line)
            if keep(r):
                rows.append({**r, "panel": name.removesuffix(".jsonl")})
    return rows


def run_seed(file: str, root: str = RUN_ROOT) -> str | None:
    """The `seed` of spire-codex run file `file` (searched under `root`)."""
    for path in Path(root).glob(f"*/*/{file}"):
        with gzip.open(path, "rt") as fh:
            return json.load(fh).get("seed")
    return None


def panel_keys(rows: Iterable[dict[str, Any]], seeds: dict[str, str | None]) -> dict[str, set]:
    """`{"run_hash": {...}, "player_seed": {(player, seed), ...}}` of the panel rows."""
    hashes, pairs = set(), set()
    for r in rows:
        hashes.add(r["file"].split(".")[0])
        seed = seeds.get(r["file"])
        if seed:
            pairs.add((r["player"], seed))
    return {"run_hash": hashes, "player_seed": pairs}


def leaks(replays: Iterable[dict[str, Any]], keys: dict[str, set]) -> dict[str, list[str]]:
    """Replay files that overlap a panel, by rule.

    Args:
        replays: `{"run_hash", "username", "seed"}` per replay file.
        keys: `panel_keys` output.

    Returns:
        `{"run_hash": [...], "player_seed": [...]}`: the run hashes of the
        overlapping replays under each rule.
    """
    out: dict[str, list[str]] = {"run_hash": [], "player_seed": []}
    for r in replays:
        if r["run_hash"] in keys["run_hash"]:
            out["run_hash"].append(r["run_hash"])
        if r.get("username") and r.get("seed"):
            if (player_id(r["username"]), r["seed"]) in keys["player_seed"]:
                out["player_seed"].append(r["run_hash"])
    return out
