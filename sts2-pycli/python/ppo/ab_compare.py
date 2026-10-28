"""Paired A/B gate over two ``ppo.probe`` JSONL files (GH #452 teacher gate).

    python -m ppo.ab_compare baseline.jsonl treatment.jsonl

Joins rows on seed, drops pairs where either arm errored/timed out, and runs
the exact McNemar test on discordant win/loss pairs. The D1 teacher gate:
treatment (search) WR up AND p below the pre-registered threshold — see
docs/experiments/2026-08-18-d1-teacher-gate/PREREGISTRATION.md.
"""

from __future__ import annotations

import argparse
import json
import sys

_BAD_OUTCOMES = ("error", "timeout")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over discordant pair counts
    (b = baseline-only wins, c = treatment-only wins)."""
    from scipy.stats import binomtest

    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def _load(path: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[int(r["seed"])] = r
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("treatment")
    args = ap.parse_args(argv)

    a, b_rows = _load(args.baseline), _load(args.treatment)
    seeds = sorted(set(a) & set(b_rows))
    usable = [
        s
        for s in seeds
        if a[s].get("outcome") not in _BAD_OUTCOMES
        and b_rows[s].get("outcome") not in _BAD_OUTCOMES
    ]
    dropped = len(set(a) | set(b_rows)) - len(usable)
    wa = sum(1 for s in usable if a[s].get("won"))
    wb = sum(1 for s in usable if b_rows[s].get("won"))
    disc_b = sum(1 for s in usable if a[s].get("won") and not b_rows[s].get("won"))
    disc_c = sum(1 for s in usable if b_rows[s].get("won") and not a[s].get("won"))
    p = mcnemar_exact(disc_b, disc_c)

    n = len(usable)
    print(f"paired seeds: {n} (dropped {dropped} error/timeout/unmatched)")
    print(f"baseline WR: {wa}/{n} = {wa / max(1, n):.3f}")
    print(f"treatment WR: {wb}/{n} = {wb / max(1, n):.3f}")
    print(f"delta: {(wb - wa) / max(1, n) * 100:+.2f}pp")
    print(f"discordant: b={disc_b} c={disc_c} (b = baseline-only wins)")
    print(f"McNemar exact p = {p:.5f}")
    verdict = "PASS" if (wb > wa and p < 0.05) else "no significant improvement"
    print(f"gate (treatment WR up AND p<0.05): {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
