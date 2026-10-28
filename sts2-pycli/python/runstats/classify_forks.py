"""Classify fork-fatal campaign output into causal buckets (spec Phase 1).

Buckets, in precedence order (thresholds from
docs/superpowers/specs/2026-07-17-boss-loss-diagnosis-design.md):

1. VARIANCE       — the deployed config itself wins >= half its replays.
2. UNWINNABLE     — even clairvoyant 10x wins < 1/4 of replays.
3. INFO_LIMITED   — clairvoyant-cheap lifts win prob >= 3/8 over deployed.
4. BUDGET_LIMITED — honest 10x lifts win prob >= 3/8 over deployed.
5. EVAL_LIMITED   — winnable, but neither information nor budget alone
                    flips it (the GH#124 evaluator-thesis bucket).

Usage:
    PYTHONPATH=python uv run --no-sync python -m runstats.classify_forks \
        ../sts2-rs/experiments/boss-loss-diagnosis/fork.*.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

BUCKETS = ("VARIANCE", "UNWINNABLE", "INFO_LIMITED", "BUDGET_LIMITED", "EVAL_LIMITED")
ARMS = ("deployed", "honest10x", "clair", "clair10x")


def classify_seed(p: dict[str, float]) -> str:
    """Spec thresholds, precedence order. ``p`` maps arm token -> win prob."""
    if p["deployed"] >= 0.5:
        return "VARIANCE"
    if p["clair10x"] < 0.25:
        return "UNWINNABLE"
    if p["clair"] - p["deployed"] >= 0.375:
        return "INFO_LIMITED"
    if p["honest10x"] - p["deployed"] >= 0.375:
        return "BUDGET_LIMITED"
    return "EVAL_LIMITED"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--json", help="write per-seed classification JSON here")
    args = ap.parse_args()
    by_seed: dict[int, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    enc: dict[int, str] = {}
    act: dict[int, int] = {}
    for path in args.jsonl:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") != "rep":
                    continue
                by_seed[r["seed"]][r["arm"]].append(bool(r["survived"]))
                enc[r["seed"]] = r["encounter"]
                act[r["seed"]] = r["fatal_act"]

    complete = {
        s: {a: sum(v) / len(v) for a, v in arms.items()}
        for s, arms in by_seed.items()
        if all(arms.get(a) for a in ARMS)
    }
    skipped = len(by_seed) - len(complete)
    buckets: dict[str, list[int]] = defaultdict(list)
    by_enc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s, p in complete.items():
        b = classify_seed(p)
        buckets[b].append(s)
        by_enc[enc[s]][b] += 1

    n = len(complete)
    print(f"classified {n} seeds ({skipped} incomplete/skipped)")
    for b in BUCKETS:
        c = len(buckets[b])
        pct = f"({c / n:.1%})" if n else ""
        print(f"  {b:<15} {c:>4}  {pct}")
    print("\nper-encounter (top 10 by classified deaths):")
    for e, bs in sorted(by_enc.items(), key=lambda kv: -sum(kv[1].values()))[:10]:
        total = sum(bs.values())
        parts = " ".join(f"{b}={bs.get(b, 0)}" for b in BUCKETS)
        print(f"  {e:<40} n={total:<4} {parts}")

    if args.json:
        out = {
            "n_classified": n,
            "n_incomplete": skipped,
            "buckets": {b: sorted(buckets[b]) for b in BUCKETS},
            "per_seed": {
                str(s): {"bucket": classify_seed(p), "encounter": enc[s], "act": act[s], **p}
                for s, p in complete.items()
            },
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
