"""Deckbuilding audit driver (GH #778): what the run policy does at card rewards.

Plays full runs through `ppo.workers.play_batch` with `record_decisions` on and
keeps only what the audit reads, so a 600-run sweep is megabytes, not the
gigabytes the raw recorded observations (full map per decision) would be:

  * one `card_reward` row per card-reward decision — act, floor, deck size,
    HP, the offered ids, the pick (or `null` for a skip);
  * one `run` row per run — outcome, final floor, and the deck size seen at
    each floor (last decision on that floor), so deck size entering each boss
    can be read back.

Rows stream to `--out` (JSONL) one chunk at a time, so a long sweep shows
progress and survives being killed (`--resume` skips seeds already written).

    PYTHONPATH=python python -m ppo.deck_audit --ckpt data/ppo/<champion>.pt \
        --ascension 10 --seeds 200 --act1 roll --out /tmp/audit.a10.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any


def reduce_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact one `play_one` row into audit rows (card rewards + one run row)."""
    from ppo.encoder import fine_surface

    seed = row["seed"]
    asc = row.get("ascension")
    if row.get("outcome") == "error":
        return [{"kind": "run", "seed": seed, "ascension": asc, "error": row.get("error")}]
    out: list[dict[str, Any]] = []
    deck_by_floor: dict[int, int] = {}
    for d in row.get("decisions") or []:
        m = d["metrics"]
        deck_by_floor[int(m["total_floor"])] = int(m["deck_size"])
        actions = d["actions"]
        if fine_surface(actions) != "card-reward":
            continue
        obs = d["obs"]
        offered = list(obs.get("card_reward_option_ids") or [])
        chosen = actions[d["chosen"]]
        idx = chosen.get("idx") if chosen.get("action") == "pick_card_reward" else None
        out.append(
            {
                "kind": "card_reward",
                "seed": seed,
                "ascension": asc,
                "act1": row.get("act1"),
                "act_index": m["act_index"],
                "act_floor": m["act_floor"],
                "total_floor": m["total_floor"],
                "deck_size": m["deck_size"],
                "hp": m["hp"],
                "max_hp": m["max_hp"],
                "gold": m["gold"],
                "offered": offered,
                "can_skip": any(
                    a.get("action") == "pick_card_reward" and a.get("idx") is None for a in actions
                ),
                "chosen_action": chosen.get("action"),
                "picked": offered[int(idx)]
                if idx is not None and int(idx) < len(offered)
                else None,
                "skipped": chosen.get("action") == "pick_card_reward" and idx is None,
            }
        )
    out.append(
        {
            "kind": "run",
            "seed": seed,
            "ascension": asc,
            "act1": row.get("act1"),
            "outcome": row.get("outcome"),
            "won": row.get("won"),
            "final": row.get("final"),
            "deck_by_floor": deck_by_floor,
            "wall_s": row.get("wall_s"),
            "wheel": row.get("wheel"),
        }
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--policy", default="net_greedy", choices=("net", "net_greedy"))
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--act1", default="roll", choices=("overgrowth", "underdocks", "roll"))
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=48, help="seeds per play_batch call")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="skip seeds with a run row in --out")
    args = ap.parse_args(argv)

    from ppo.workers import play_batch

    done: set[int] = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "run" and "error" not in r:
                    done.add(r["seed"])
    seeds = [s for s in range(args.seed_start, args.seed_start + args.seeds) if s not in done]
    t0 = time.time()
    n_runs = n_won = 0
    with open(args.out, "a" if args.resume else "w") as sink:
        for i in range(0, len(seeds), args.chunk):
            rows = play_batch(
                seeds[i : i + args.chunk],
                policy=args.policy,
                ascension=args.ascension,
                character=args.character,
                procs=args.procs,
                record_decisions=True,
                ckpt_path=args.ckpt,
                act1=args.act1,
            )
            for row in rows:
                for r in reduce_row(row):
                    sink.write(json.dumps(r) + "\n")
                n_runs += 1
                n_won += bool(row.get("won"))
            sink.flush()
            print(
                f"[deck_audit] asc={args.ascension} {n_runs}/{len(seeds)} runs, "
                f"{n_won} won, {time.time() - t0:.0f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
