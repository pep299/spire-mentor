"""Mine fork-fatal --trace output: what does clairvoyant do differently?

Input: ``run_arena fork-fatal --trace`` JSONL for the deployed + clair arms
on the INFO_LIMITED seeds (spec
docs/superpowers/specs/2026-07-17-info-fight-trace-mining-design.md).

For every rep the per-decision trace is folded into per-turn features
(blocking vs visible incoming attack damage, potion timing, tempo, energy
waste), then the probe contrast is **clair WINNING reps vs deployed LOSING
reps** per seed, aggregated per encounter with sign-consistency counts.

Incoming damage is the sum of visible ATTACK/MULTI_ATTACK/DEATH_BLOW intent
payloads — base values, not strength/weak-adjusted, so over/underblock are
estimates; hp deltas between turns are exact.

Usage:
    PYTHONPATH=python uv run --no-sync python -m runstats.trace_mine \
        ../sts2-rs/experiments/boss-loss-diagnosis/trace.deployed.jsonl \
        ../sts2-rs/experiments/boss-loss-diagnosis/trace.clair.jsonl \
        --json mine.json --examples 3
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import pairwise
from statistics import mean
from typing import Any


def _incoming(step: dict[str, Any]) -> int:
    """Visible attack damage aimed at the player this enemy phase."""
    total = 0
    for e in step["enemies"]:
        parts = e.split(":")  # MONSTER.X : hp+Bb : INTENT[[:d]|[:dxh]]
        intent = parts[2:]
        if not intent:
            continue
        kind = intent[0]
        if kind in ("ATTACK", "DEATH_BLOW") and len(intent) > 1:
            total += int(intent[1])
        elif kind == "MULTI_ATTACK" and len(intent) > 1:
            d, h = intent[1].split("x")
            total += int(d) * int(h)
    return total


def _enemy_hp(step: dict[str, Any]) -> int:
    return sum(int(e.split(":")[1].split("+")[0]) for e in step["enemies"])


def turn_features(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold decision steps into one feature row per combat round."""
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in trace:
        by_round[int(s["round"])].append(s)
    turns = []
    for rnd in sorted(by_round):
        steps = by_round[rnd]
        end = next((s for s in steps if s["action"] == "EndTurn"), None)
        potions = [s["action"] for s in steps if s["action"].startswith("Potion:")]
        turns.append(
            {
                "round": rnd,
                "hp": int(steps[0]["hp"]),
                "enemy_hp": _enemy_hp(steps[0]),
                "n_enemies": len(steps[0]["enemies"]),
                "cards_played": sum(1 for s in steps if s["action"].startswith("Play:")),
                "potions": potions,
                # None when the fight resolved mid-turn (kill or death).
                "block_end": None if end is None else int(end["block"]),
                "incoming": None if end is None else _incoming(end),
                "energy_left": None if end is None else int(end["energy"]),
            }
        )
    return turns


def rep_features(trace: list[dict[str, Any]]) -> dict[str, float]:
    """Fight-level feature summary of one traced rep."""
    turns = turn_features(trace)
    ended = [t for t in turns if t["block_end"] is not None]
    over = [max(0, t["block_end"] - t["incoming"]) for t in ended]
    under = [max(0, t["incoming"] - t["block_end"]) for t in ended]
    potion_rounds = [t["round"] for t in turns for _ in t["potions"]]
    first_kill = next(
        (t["round"] for prev, t in pairwise(turns) if t["n_enemies"] < prev["n_enemies"]),
        None,
    )
    dmg_dealt = [max(0, prev["enemy_hp"] - t["enemy_hp"]) for prev, t in pairwise(turns)]
    dmg_taken = [max(0, prev["hp"] - t["hp"]) for prev, t in pairwise(turns)]
    n = len(turns)
    return {
        "rounds": float(n),
        "potions_used": float(len(potion_rounds)),
        "first_potion_round": float(potion_rounds[0]) if potion_rounds else float("nan"),
        "overblock_per_turn": mean(over) if over else 0.0,
        "underblock_per_turn": mean(under) if under else 0.0,
        "energy_left_per_turn": mean(t["energy_left"] for t in ended) if ended else 0.0,
        "cards_per_turn": mean(t["cards_played"] for t in turns) if turns else 0.0,
        "dmg_dealt_per_turn": mean(dmg_dealt) if dmg_dealt else 0.0,
        "dmg_taken_per_turn": mean(dmg_taken) if dmg_taken else 0.0,
        "first_kill_round": float(first_kill) if first_kill else float("nan"),
    }


FEATURES = (
    "rounds",
    "potions_used",
    "first_potion_round",
    "overblock_per_turn",
    "underblock_per_turn",
    "energy_left_per_turn",
    "cards_per_turn",
    "dmg_dealt_per_turn",
    "dmg_taken_per_turn",
    "first_kill_round",
)


def _nanmean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    return mean(xs) if xs else float("nan")


def load_reps(paths: list[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") == "rep" and r.get("trace"):
                    rows.append(r)
    return rows


def contrast(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-seed clair-win minus deployed-loss feature means, aggregated."""
    by_seed: dict[int, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    enc: dict[int, str] = {}
    winrate: dict[int, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        seed = r["seed"]
        enc[seed] = r["encounter"]
        winrate[seed][r["arm"]].append(bool(r["survived"]))
        group = None
        if r["arm"] == "clair" and r["survived"]:
            group = "clair_win"
        elif r["arm"] == "deployed" and not r["survived"]:
            group = "deployed_loss"
        if group:
            by_seed[seed][group].append(rep_features(r["trace"]))

    per_seed = {}
    for seed, groups in sorted(by_seed.items()):
        if not groups.get("clair_win") or not groups.get("deployed_loss"):
            continue  # needs at least one rep on each side to contrast
        diff = {}
        for feat in FEATURES:
            cw = _nanmean([g[feat] for g in groups["clair_win"]])
            dl = _nanmean([g[feat] for g in groups["deployed_loss"]])
            diff[feat] = {"clair_win": cw, "deployed_loss": dl, "diff": cw - dl}
        per_seed[seed] = {
            "encounter": enc[seed],
            "clair_p": mean(winrate[seed]["clair"]),
            "deployed_p": mean(winrate[seed]["deployed"]),
            "features": diff,
        }

    def aggregate(seeds: list[int]) -> dict[str, Any]:
        out = {}
        for feat in FEATURES:
            diffs = [per_seed[s]["features"][feat]["diff"] for s in seeds]
            diffs = [d for d in diffs if d == d]
            if not diffs:
                out[feat] = None
                continue
            out[feat] = {
                "mean_diff": mean(diffs),
                "n": len(diffs),
                "pos": sum(1 for d in diffs if d > 0),
                "neg": sum(1 for d in diffs if d < 0),
                "clair_win_mean": _nanmean(
                    [per_seed[s]["features"][feat]["clair_win"] for s in seeds]
                ),
                "deployed_loss_mean": _nanmean(
                    [per_seed[s]["features"][feat]["deployed_loss"] for s in seeds]
                ),
            }
        return out

    by_enc: dict[str, list[int]] = defaultdict(list)
    for s, v in per_seed.items():
        by_enc[v["encounter"]].append(s)
    return {
        "n_seeds": len(per_seed),
        "overall": aggregate(list(per_seed)),
        "per_encounter": {
            e: aggregate(ss) for e, ss in sorted(by_enc.items(), key=lambda kv: -len(kv[1]))
        },
        "per_seed": per_seed,
    }


def _fmt_turn(t: dict[str, Any]) -> str:
    blk = "-" if t["block_end"] is None else f"{t['block_end']}b/{t['incoming']}in"
    pots = f" {len(t['potions'])}pot" if t["potions"] else ""
    return (
        f"    r{t['round']:>2} hp={t['hp']:>3} foes={t['n_enemies']}({t['enemy_hp']:>3}) "
        f"cards={t['cards_played']} {blk}{pots}"
    )


def print_examples(rows: list[dict[str, Any]], rep: dict[str, Any], n: int) -> None:
    """Side-by-side turn summaries for the n largest clair-vs-deployed gaps."""
    ranked = sorted(rep["per_seed"].items(), key=lambda kv: kv[1]["deployed_p"] - kv[1]["clair_p"])[
        :n
    ]
    by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[r["seed"], r["arm"]].append(r)
    for seed, v in ranked:
        print(
            f"\n== seed {seed} {v['encounter']} deployed_p={v['deployed_p']:.2f} clair_p={v['clair_p']:.2f}"
        )
        loss = next(r for r in by_key[seed, "deployed"] if not r["survived"])
        win = next(r for r in by_key[seed, "clair"] if r["survived"])
        for label, r in (("deployed LOSS", loss), ("clair WIN", win)):
            print(f"  {label} (rep {r['rep']}, end_hp {r['end_hp']}):")
            for t in turn_features(r["trace"]):
                print(_fmt_turn(t))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--json", help="write the full contrast report JSON here")
    ap.add_argument("--examples", type=int, default=0)
    args = ap.parse_args()
    rows = load_reps(args.jsonl)
    rep = contrast(rows)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)

    print(f"seeds contrasted: {rep['n_seeds']} (traced reps: {len(rows)})")
    print(f"{'feature':<22}{'clairWIN':>10}{'deplLOSS':>10}{'diff':>9}  sign(+/-/n)")
    for feat in FEATURES:
        a = rep["overall"][feat]
        if a is None:
            continue
        print(
            f"{feat:<22}{a['clair_win_mean']:>10.2f}{a['deployed_loss_mean']:>10.2f}"
            f"{a['mean_diff']:>9.2f}  {a['pos']}/{a['neg']}/{a['n']}"
        )
    for e, agg in rep["per_encounter"].items():
        n = next((v["n"] for v in agg.values() if v), 0)
        if n < 5:
            continue
        print(f"\n[{e}] n={n}")
        for feat in FEATURES:
            a = agg[feat]
            if a is None:
                continue
            print(
                f"{feat:<22}{a['clair_win_mean']:>10.2f}{a['deployed_loss_mean']:>10.2f}"
                f"{a['mean_diff']:>9.2f}  {a['pos']}/{a['neg']}/{a['n']}"
            )
    if args.examples:
        print_examples(rows, rep, args.examples)


if __name__ == "__main__":
    main()
