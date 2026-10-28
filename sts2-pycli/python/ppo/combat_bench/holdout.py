"""The combat bench's held-out runs, kept out of every training pool (GH #830).

A playout or policy net trained on a bench run has seen the exact decks and
fights it is later scored on, so its bench numbers would overstate it.
`build` lists every run file that has a fight in a bench set in its MANIFEST
(`held_out_runs`): `MANIFEST.json` for v1, `MANIFEST_<version>.json` for each
later version. The holdout is the union over every MANIFEST in the bench data
dir; `rollout_learn gen` and `rl` subtract it by default.
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable, Iterable
from typing import Any

MANIFEST_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "combat_bench", "MANIFEST.json"
    )
)


def manifest_paths() -> list[str]:
    """Every bench MANIFEST next to `MANIFEST_PATH`: v1's and each `MANIFEST_<version>.json`."""
    data = os.path.dirname(MANIFEST_PATH)
    found = glob.glob(os.path.join(data, "MANIFEST.json"))
    return found + sorted(glob.glob(os.path.join(data, "MANIFEST_v*.json")))


def holdout_runs(manifest_path: str | None = None) -> set[str]:
    """Run file names (`<hash>.run.gz`) that have a fight in a bench set.

    Args:
        manifest_path: One bench MANIFEST; default the union of every one in
            `manifest_paths()`.

    Raises:
        FileNotFoundError: No MANIFEST at `manifest_path`, or none at all
            (the public snapshot ships without the bench data).
    """
    paths = [manifest_path] if manifest_path else manifest_paths()
    if not paths:
        raise FileNotFoundError(f"no combat bench MANIFEST next to {MANIFEST_PATH}")
    runs: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            runs |= set(json.load(f)["held_out_runs"])
    return runs


def subtract_bench_runs[T](
    pool: Iterable[T],
    file_of: Callable[[T], str],
    allow_overlap: bool = False,
    manifest_path: str | None = None,
    tag: str = "[combat_bench]",
) -> tuple[list[T], dict[str, Any]]:
    """Drop pool items whose run is a bench run, unless `allow_overlap`.

    Args:
        pool: Training candidates (fight rows, or tuples holding one).
        file_of: Maps an item to its run file name.
        allow_overlap: Keep bench runs (prints a warning) instead of dropping.
        manifest_path: One bench MANIFEST; default every one (`holdout_runs`).
        tag: Log prefix.

    Returns:
        `(pool, meta)`. `meta` records what happened for the trained net's
        metadata: `manifest` (the MANIFEST file names read), `bench_runs`,
        `overlap_fights` and `overlap_runs` (bench items found in the pool),
        and `subtracted`.
    """
    items = list(pool)
    paths = [manifest_path] if manifest_path else manifest_paths()
    try:
        bench = holdout_runs(manifest_path)
    except FileNotFoundError:
        where = manifest_path or os.path.dirname(MANIFEST_PATH)
        print(f"{tag} WARNING: no combat bench MANIFEST at {where}; nothing held out")
        return items, {"manifest": None, "subtracted": False}
    hits = [x for x in items if file_of(x) in bench]
    meta = {
        "manifest": [os.path.basename(p) for p in paths],
        "bench_runs": len(bench),
        "overlap_fights": len(hits),
        "overlap_runs": len({file_of(x) for x in hits}),
        "subtracted": not allow_overlap,
    }
    if allow_overlap:
        if hits:
            print(
                f"{tag} WARNING: --allow-bench-overlap: training on {len(hits)} fights from "
                f"{meta['overlap_runs']} combat-bench runs; this net's bench scores are not "
                "held out",
                flush=True,
            )
        return items, meta
    if hits:
        print(f"{tag} held out {len(hits)} fights from {meta['overlap_runs']} combat-bench runs")
    return [x for x in items if file_of(x) not in bench], meta
