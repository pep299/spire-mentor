#!/usr/bin/env python3
"""Summarize issue #40 sweep JSONs into a comparison table."""

from __future__ import annotations

import json
import sys
from collections import Counter


def summ(path: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    res = d["results"]
    n = len(res)
    acts = Counter(r.get("act") for r in res)
    tos = sum(1 for r in res if r.get("timeout"))
    return {
        "reward_sims": d["reward_sims"],
        "n": n,
        "wins": d["wins"],
        "win_rate": d["wins"] / n,
        "act1_surv": d["reached_act2"],
        "act1_surv_rate": d["reached_act2"] / n,
        "act3_reach": d["reached_act3"],
        "died_act1": acts.get(1, 0),
        "died_act2": acts.get(2, 0),
        "act3": acts.get(3, 0),
        "timeouts": tos,
        "wall_s": round(d["wall_s"]),
    }


def main() -> int:
    rows = [summ(p) for p in sys.argv[1:]]
    rows.sort(key=lambda r: r["reward_sims"])
    hdr = (
        f"{'reward-sims':>11} {'n':>3} {'wins':>5} {'win%':>6} "
        f"{'act1-surv':>9} {'surv%':>6} {'act3':>4} {'TO':>3} {'wall_s':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['reward_sims']:>11} {r['n']:>3} {r['wins']:>5} "
            f"{r['win_rate'] * 100:>5.1f}% {r['act1_surv']:>9} "
            f"{r['act1_surv_rate'] * 100:>5.1f}% {r['act3_reach']:>4} "
            f"{r['timeouts']:>3} {r['wall_s']:>7}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
