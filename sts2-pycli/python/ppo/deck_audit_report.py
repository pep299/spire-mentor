"""Report for the deckbuilding audit (GH #778): bot vs human card-reward behaviour.

Reads `ppo.deck_audit` (bot) and `ppo.deck_audit_human` (spire-codex) JSONL and
prints, per ascension, split by run outcome:

  * skip rate at combat card rewards, by act;
  * skip rate by deck-size bucket (deck size BEFORE the pick);
  * deck size at the first card reward of acts 2 and 3 (= the deck carried out
    of the previous act; conditional on reaching it) and the final deck of wins.

Every comparison here is UNPAIRED and conditional on survival — a run that dies
in act 1 never contributes an act-2 row — so read the split-by-outcome columns,
not the pooled ones, before attributing a gap to policy.

    python -m ppo.deck_audit_report --bot a10.jsonl a5.jsonl --human human.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from collections.abc import Sequence
from typing import Any

BUCKETS = ((0, 15), (16, 20), (21, 25), (26, 30), (31, 999))


def load(paths: list[str], source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rewards, runs = [], []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["source"] = source
                (rewards if r["kind"] == "card_reward" else runs).append(r)
    runs = [r for r in runs if "error" not in r]
    if source == "bot":
        won = {(r["ascension"], r["seed"]): bool(r["won"]) for r in runs}
        rewards = [
            dict(r, won=won[r["ascension"], r["seed"]])
            for r in rewards
            if (r["ascension"], r["seed"]) in won
        ]
    return rewards, runs


def _rate(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "      —"
    k = sum(r["skipped"] for r in rows)
    return f"{k / len(rows):5.1%} ({len(rows)})"


def _mean(xs: Sequence[float]) -> str:
    return f"{statistics.mean(xs):5.1f} ({len(xs)})" if xs else "      —"


def _first_deck(rewards: list[dict[str, Any]], act: int) -> dict[Any, int]:
    """Deck size at each run's first card reward in `act`, keyed by run."""
    out: dict[Any, tuple[int, int]] = {}
    for r in rewards:
        if r["act_index"] != act:
            continue
        key = r.get("seed", r.get("file")), r["ascension"], r["source"]
        if key not in out or r["total_floor"] < out[key][0]:
            out[key] = (r["total_floor"], r["deck_size"])
    return {k: v[1] for k, v in out.items()}


def report(rewards: list[dict[str, Any]], runs: list[dict[str, Any]], asc: int) -> None:
    groups = collections.OrderedDict()
    for source in ("bot", "human"):
        for won in (True, False, None):
            label = f"{source}:{'win' if won else 'loss' if won is False else 'all'}"
            sel = [
                r
                for r in rewards
                if r["source"] == source
                and r["ascension"] == asc
                and (won is None or r["won"] == won)
            ]
            sel_runs = [
                r
                for r in runs
                if r["source"] == source
                and r["ascension"] == asc
                and (won is None or r["won"] == won)
            ]
            if sel_runs:
                groups[label] = (sel, sel_runs)
    if not groups:
        return
    print(f"\n=== A{asc} ===")
    print(
        f"{'group':12s} {'runs':>5s} " + " ".join(f"{'skip act' + str(a):>14s}" for a in (1, 2, 3))
    )
    for label, (sel, sel_runs) in groups.items():
        print(
            f"{label:12s} {len(sel_runs):5d} "
            + " ".join(f"{_rate([r for r in sel if r['act_index'] == a]):>14s}" for a in (1, 2, 3))
        )
    print(
        f"\n{'group':12s} "
        + " ".join(f"{f'deck {lo}-{hi}' if hi < 999 else f'deck {lo}+':>14s}" for lo, hi in BUCKETS)
    )
    for label, (sel, _) in groups.items():
        print(
            f"{label:12s} "
            + " ".join(
                f"{_rate([r for r in sel if lo <= r['deck_size'] <= hi]):>14s}"
                for lo, hi in BUCKETS
            )
        )
    print(
        f"\n{'group':12s} {'deck@act2':>14s} {'deck@act3':>14s} {'final(win)':>14s} {'rewards/run':>12s}"
    )
    for label, (sel, sel_runs) in groups.items():
        d2 = list(_first_deck(sel, 2).values())
        d3 = list(_first_deck(sel, 3).values())
        finals = [
            (r["final"]["deck_size"] if r["source"] == "bot" else r["final_deck"])
            for r in sel_runs
            if r["won"]
        ]
        print(
            f"{label:12s} {_mean(d2):>14s} {_mean(d3):>14s} {_mean(finals):>14s} "
            f"{len(sel) / len(sel_runs):12.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bot", nargs="+", required=True)
    ap.add_argument("--human", nargs="+", required=True)
    ap.add_argument("--ascension", type=int, nargs="+", default=None)
    args = ap.parse_args(argv)
    br, bruns = load(args.bot, "bot")
    hr, hruns = load(args.human, "human")
    rewards, runs = br + hr, bruns + hruns
    ascs = args.ascension or sorted({r["ascension"] for r in bruns})
    for a in ascs:
        report(rewards, runs, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
