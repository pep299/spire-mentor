"""Telemetry rows, JSONL persistence, and summary aggregation."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def bucket_key(row: dict[str, Any]) -> str:
    # "default" covers legacy pre-deck-axis rows so old sweep dirs stay browsable.
    return f"{row['encounter']}/{row['ascension']}/{row.get('deck', 'default')}/{row['strategy']}"


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def throughput(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-core throughput aggregated over `rows`.

    Rates are core-time based: they divide work (combats/actions/turns) by the
    summed per-combat wall time (`sum(wall_ms)`), i.e. the CPU-seconds one core
    spent. This is the single-core speed, independent of how many workers ran
    the sweep. The sweep-wide wall-clock rates (which reflect parallelism) are
    injected separately by `run_sweep`, which alone knows elapsed time and the
    worker count.

    Rows missing the timing/count fields (e.g. error rows, or hand-built test
    rows) contribute 0 and never raise.

    Args:
        rows: Telemetry row dicts (error rows are ignored).

    Returns:
        Dict with combats, actions, turns, core_seconds, and the three
        per-core rates (0.0 when no core time was recorded).
    """
    ok = [r for r in rows if "error" not in r]
    combats = len(ok)
    actions = sum(r.get("actions", 0) or 0 for r in ok)
    turns = sum(r.get("turns", 0) or 0 for r in ok)
    core_s = sum(r.get("wall_ms", 0) or 0 for r in ok) / 1000.0
    rate = (lambda n: round(n / core_s, 2)) if core_s > 0 else (lambda _n: 0.0)
    return {
        "combats": combats,
        "actions": actions,
        "turns": turns,
        "core_seconds": round(core_s, 3),
        "combats_per_core_s": rate(combats),
        "actions_per_core_s": rate(actions),
        "turns_per_core_s": rate(turns),
    }


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return float(sorted_vals[idx])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rows into per-(encounter, ascension, deck, strategy) buckets.

    Args:
        rows: List of telemetry row dicts from play_combat.

    Returns:
        Dict with a "buckets" key mapping bucket_key strings to stats dicts
        and a top-level "throughput" block (per-core rates aggregated over all
        rows; see `throughput`). Each bucket carries n, mean_dmg, median_dmg,
        p95_dmg, max_dmg, win_rate, mean_turns, errors plus the simthespire-style
        stats mode_dmg, worst, worst_p, a box dict (q1/median/q3 with p1/p99
        whiskers lo/hi and outliers beyond them), and a per-bucket "throughput"
        block (so e.g. mcts-vs-heuristic per-core speed is visible per cell).
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault(bucket_key(r), []).append(r)
    out: dict[str, Any] = {}
    for key, rs in buckets.items():
        # Error rows have no valid numeric fields; aggregate over ok rows only.
        ok = [r for r in rs if "error" not in r]
        errors = [r for r in rs if "error" in r]
        if not ok:
            out[key] = {
                "n": 0,
                "mean_dmg": 0.0,
                "median_dmg": 0.0,
                "p95_dmg": 0.0,
                "max_dmg": 0.0,
                "win_rate": 0.0,
                "mean_turns": 0.0,
                "errors": len(errors),
                "mode_dmg": 0.0,
                "worst": 0.0,
                "worst_p": 0.0,
                "box": {
                    "q1": 0.0,
                    "median": 0.0,
                    "q3": 0.0,
                    "lo": 0.0,
                    "hi": 0.0,
                    "outliers": [],
                },
                "throughput": throughput([]),
            }
            continue
        dmg = sorted(r["dmg_taken"] for r in ok)
        counts = Counter(dmg)
        # Mode: most frequent value, ties broken by smallest value (dmg is sorted).
        mode_dmg = float(max(counts, key=lambda v: (counts[v], -v)))
        worst = dmg[-1]
        lo = quantile(dmg, 0.01)
        hi = quantile(dmg, 0.99)
        out[key] = {
            "n": len(ok),
            "mean_dmg": round(statistics.fmean(dmg), 3),
            "median_dmg": quantile(dmg, 0.5),
            "p95_dmg": quantile(dmg, 0.95),
            "max_dmg": float(worst),
            "win_rate": round(sum(1 for r in ok if r["victory"]) / len(ok), 4),
            "mean_turns": round(statistics.fmean(r["turns"] for r in ok), 2),
            "errors": len(errors),
            "mode_dmg": mode_dmg,
            "worst": float(worst),
            "worst_p": round(counts[worst] / len(ok), 6),
            "box": {
                "q1": quantile(dmg, 0.25),
                "median": quantile(dmg, 0.5),
                "q3": quantile(dmg, 0.75),
                "lo": lo,
                "hi": hi,
                # unique outlier values for fence annotation; use dmg_values for per-point detail
                "outliers": sorted({v for v in dmg if v < lo or v > hi}),
            },
            "throughput": throughput(rs),
        }
    return {"buckets": out, "throughput": throughput(rows)}
