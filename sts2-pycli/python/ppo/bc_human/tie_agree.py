"""Do the in-band root rules pick what humans pick? `python -m ppo.bc_human.tie_agree`.

GH #1022 phase 3. Every human decision that `sync.replay` keeps in sync (v0.111
SpireCodex replays with RNG counters) is searched once by the deployed pilot
(`sts2_rs.tie_rule_picks`, the run agent's search at the eval-cheap 1000 / 4000
budget), and its pick is re-made under each arm's rule set over that same tree.
A row holds each arm's move, the pilot's and the human's, all as move keys, so
a human who played the other copy of a card agrees.

The read is on the decisions where an arm and the pilot pick different moves.
Everywhere else both agree with the human or both do not, which says nothing
about the rule. `report` pairs the arm against the pilot on those decisions and
clusters the standard error by run, since one player's decisions are not
independent.

    python -m ppo.bc_human.tie_agree run --character IRONCLAD --runs 300 --out rows.jsonl
    python -m ppo.bc_human.tie_agree report rows.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
import zlib
from collections import defaultdict
from collections.abc import Iterable
from multiprocessing import Pool
from typing import Any

from ppo.bc_human import replay, sync
from ppo.bc_human.sync_census import ROOT, eligible

DEFAULT_ARMS = "none,block,free,junk,block+free+junk"


def tier_of(encounter: str) -> str:
    """The pilot's budget tier for an encounter id."""
    if encounter.endswith("_BOSS"):
        return "boss"
    if encounter.endswith("_ELITE"):
        return "elite"
    return "common"


def run_file(job: tuple[str, list[str], int, int]) -> dict[str, Any]:
    """Every synced decision of one replay, searched and re-picked per arm."""
    import sts2_rs

    path, arms, sims, boss_sims = job
    name = os.path.basename(path)
    out: dict[str, Any] = {"file": name, "rows": [], "fights": 0, "skipped": 0}
    try:
        for fi, rec in enumerate(replay.fights(path)):
            if rec["result"] not in replay.FINISHED or not rec["rng_state"]:
                continue
            out["fights"] += 1
            tier = tier_of(rec["encounter"] or "")
            budget = boss_sims if tier == "boss" else sims
            for k, (st, action) in enumerate(sync.replay(rec)["pairs"]):
                if len(st.legal_actions()) < 2:
                    continue
                seed = zlib.crc32(f"{name}:{fi}:{k}".encode())
                try:
                    res = sts2_rs.tie_rule_picks(st, tier, budget, seed, arms, human=action)
                except (ValueError, RuntimeError):
                    out["skipped"] += 1
                    continue
                out["rows"].append(
                    {
                        "file": name,
                        "fight": fi,
                        "d": k,
                        "asc": rec.get("ascension"),
                        "floor": rec.get("floor"),
                        "enc": rec.get("encounter"),
                        "tier": tier,
                        "legal": res["legal"],
                        "human": res["human"],
                        "pick": res["pick"]["key"],
                        "rule": res["pick"]["rule"],
                        "band": res["pick"]["band"],
                        "arms": [a["key"] for a in res["arms"]],
                    }
                )
    except Exception as e:  # one bad replay must not stop the sweep
        out["error"] = repr(e)[:300]
    return out


def cmd_run(args: argparse.Namespace) -> None:
    arms = [a for a in args.arms.split(",") if a]
    files = sorted(glob.glob(f"{ROOT}/{args.character.lower()}/*/*.gz"))
    random.Random(0).shuffle(files)
    with Pool(args.procs) as p:
        ok = p.starmap(eligible, [(f, args.character) for f in files], chunksize=64)
    files = [f for f, e in zip(files, ok, strict=True) if e]
    if args.runs:
        files = files[: args.runs]
    print(f"[tie_agree] {len(files)} runs, arms {arms}", flush=True)
    jobs = [(f, arms, args.sims, args.boss_sims) for f in files]
    t0 = time.time()
    n_rows = 0
    with Pool(args.procs) as p, open(args.out, "w", encoding="utf-8") as fh:
        head = {"k": "h", "arms": arms, "character": args.character}
        head |= {"sims": args.sims, "boss_sims": args.boss_sims, "runs": len(files)}
        fh.write(json.dumps(head) + "\n")
        for n, res in enumerate(p.imap_unordered(run_file, jobs), 1):
            for row in res["rows"]:
                fh.write(json.dumps(row) + "\n")
            n_rows += len(res["rows"])
            if "error" in res:
                print(f"[tie_agree] {res['file']}: {res['error']}", flush=True)
            if n % 10 == 0 or n == len(jobs):
                fh.flush()
                print(
                    f"[tie_agree] {n}/{len(jobs)} runs, {n_rows} decisions, {time.time() - t0:.0f}s",
                    flush=True,
                )


def paired(rows: Iterable[dict[str, Any]], arm: int) -> dict[str, Any]:
    """Arm vs pilot agreement with the human where their picks differ.

    Returns the flip count, each side's agreement there, and the mean
    difference (arm minus pilot) with a run-clustered standard error.
    """
    by_file: dict[str, list[int]] = defaultdict(list)
    n = arm_hits = pilot_hits = 0
    for r in rows:
        if r["arms"][arm] == r["pick"]:
            continue
        a = int(r["arms"][arm] == r["human"])
        p = int(r["pick"] == r["human"])
        n += 1
        arm_hits += a
        pilot_hits += p
        by_file[r["file"]].append(a - p)
    if n == 0:
        return {"flips": 0, "arm": 0, "pilot": 0, "diff": None, "se": None}
    mean = (arm_hits - pilot_hits) / n
    clusters = len(by_file)
    resid = sum((sum(d) - mean * len(d)) ** 2 for d in by_file.values())
    se = math.sqrt(resid * clusters / max(clusters - 1, 1)) / n
    return {"flips": n, "arm": arm_hits, "pilot": pilot_hits, "diff": mean, "se": se}


def cmd_report(args: argparse.Namespace) -> None:
    with open(args.rows, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    head, rows = lines[0], [r for r in lines[1:] if r.get("human") is not None]
    if args.min_asc is not None:
        rows = [r for r in rows if (r.get("asc") or 0) >= args.min_asc]
    n = len(rows)
    agree = sum(r["pick"] == r["human"] for r in rows)
    banded = sum(r["band"] > 1 for r in rows)
    print(f"{head['character']}: {n} decisions from {len({r['file'] for r in rows})} runs")
    print(
        f"  pilot agrees with the human on {100 * agree / max(n, 1):.1f}%; band > 1 on {100 * banded / max(n, 1):.1f}%"
    )
    rules: dict[str, int] = defaultdict(int)
    for r in rows:
        rules[r["rule"]] += 1
    print(
        "  pilot pick rule: "
        + ", ".join(f"{k} {100 * v / n:.2f}%" for k, v in sorted(rules.items()))
    )
    print(
        f"  {'arm':<20} {'flips':>6} {'flip%':>6} {'arm=human':>10} {'pilot=human':>12} {'diff':>7} {'±se':>6}"
    )
    for i, name in enumerate(head["arms"]):
        s = paired(rows, i)
        diff = "-" if s["diff"] is None else f"{100 * s['diff']:+.1f}"
        se = "-" if s["se"] is None else f"{100 * s['se']:.1f}"
        print(
            f"  {name:<20} {s['flips']:>6} {100 * s['flips'] / max(n, 1):>6.2f} "
            f"{s['arm']:>10} {s['pilot']:>12} {diff:>7} {se:>6}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--character", default="IRONCLAD")
    r.add_argument("--out", required=True)
    r.add_argument("--runs", type=int, default=0, help="0 = every eligible run")
    r.add_argument("--arms", default=DEFAULT_ARMS)
    r.add_argument("--sims", type=int, default=1000)
    r.add_argument("--boss-sims", type=int, default=4000)
    r.add_argument("--procs", type=int, default=24)
    rep = sub.add_parser("report")
    rep.add_argument("rows")
    rep.add_argument("--min-asc", type=int, default=None)
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
