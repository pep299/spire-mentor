"""Human side of the deckbuilding audit (GH #778): card-reward choices in spire-codex.

Scans the spire-codex run corpus for one character and emits the same
`card_reward` / `run` row shapes as `ppo.deck_audit`, so one report reads both:

  * filter: `--build` (default v0.107.1), `game_mode == standard`, single
    player, not abandoned, the requested character;
  * a card reward is a `card_choices` list of >= 2 cards at a combat room
    (`rooms[0].room_type` in monster / elite / boss). Events and ancients
    offer cards through the same field and are excluded. A skip is a reward
    where no card has `was_picked`;
  * deck size is reconstructed BACKWARDS from the final deck: the deck entering
    floor f is `len(final deck) - gained(>= f) + removed(>= f)`. Transforms are
    size-neutral. It is approximate (a few cards gained by relic effects are not
    in `cards_gained`), and the report says so.

    python -m ppo.deck_audit_human --ascension 10 --out /tmp/human.a10.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from multiprocessing import Pool
from typing import Any

DEFAULT_ROOT = "../../sts2_run_scraper/data/files-spire-codex"
COMBAT_ROOMS = frozenset({"monster", "elite", "boss"})


def extract(path: str, build: str, character: str, ascensions: set[int]) -> list[dict[str, Any]]:
    try:
        with gzip.open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return []
    if (
        d.get("build_id") != build
        or d.get("game_mode") != "standard"
        or d.get("was_abandoned")
        or len(d.get("players") or []) != 1
        or d.get("ascension") not in ascensions
    ):
        return []
    player = d["players"][0]
    if player.get("character") != f"CHARACTER.{character.upper()}":
        return []
    asc, won = d["ascension"], bool(d.get("win"))
    pts = [(act_i, pt) for act_i, act in enumerate(d["map_point_history"]) for pt in act]
    # Per-floor deck deltas, then the backwards reconstruction.
    deltas = []
    for _, pt in pts:
        ps = pt["player_stats"][0]
        deltas.append(len(ps.get("cards_gained") or []) - len(ps.get("cards_removed") or []))
    deck = len(player.get("deck") or [])
    entering = [0] * len(pts)
    for i in range(len(pts) - 1, -1, -1):
        deck -= deltas[i]
        entering[i] = deck
    rows: list[dict[str, Any]] = []
    deck_by_floor: dict[int, int] = {}
    act_floor = 0
    prev_act = -1
    for i, (act_i, pt) in enumerate(pts):
        if act_i != prev_act:
            act_floor, prev_act = 0, act_i
        act_floor += 1
        floor = i + 1
        deck_by_floor[floor] = entering[i]
        rooms = pt.get("rooms") or []
        room_type = rooms[0].get("room_type") if rooms else None
        ps = pt["player_stats"][0]
        choices = ps.get("card_choices") or []
        if room_type not in COMBAT_ROOMS or len(choices) < 2:
            continue
        picked = [c["card"]["id"] for c in choices if c.get("was_picked")]
        rows.append(
            {
                "kind": "card_reward",
                "source": "human",
                "file": os.path.basename(path),
                "ascension": asc,
                "won": won,
                "act_index": act_i + 1,
                "act_floor": act_floor,
                "total_floor": floor,
                "room_type": room_type,
                "deck_size": entering[i],
                "hp": ps.get("current_hp"),
                "max_hp": ps.get("max_hp"),
                "offered": [c["card"]["id"] for c in choices],
                "picked": picked[0] if picked else None,
                "skipped": not picked,
            }
        )
    rows.append(
        {
            "kind": "run",
            "source": "human",
            "file": os.path.basename(path),
            "ascension": asc,
            "won": won,
            "floors": len(pts),
            "final_deck": len(player.get("deck") or []),
            "deck_by_floor": deck_by_floor,
            "boss_floors": [
                i + 1 for i, (_, pt) in enumerate(pts) if pt["map_point_type"] == "boss"
            ],
            "killed_by": d.get("killed_by_encounter"),
        }
    )
    return rows


def _job(args: tuple[str, str, str, frozenset[int]]) -> list[dict[str, Any]]:
    return extract(args[0], args[1], args[2], set(args[3]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--build", default="v0.107.1")
    ap.add_argument("--ascension", type=int, nargs="+", default=list(range(11)))
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    base = os.path.join(args.root, args.character.lower())
    paths = [
        os.path.join(base, month, f)
        for month in sorted(os.listdir(base))
        for f in sorted(os.listdir(os.path.join(base, month)))
    ]
    jobs = [(p, args.build, args.character, frozenset(args.ascension)) for p in paths]
    n_runs = 0
    with Pool(args.procs) as pool, open(args.out, "w", encoding="utf-8") as sink:
        for rows in pool.imap_unordered(_job, jobs, chunksize=64):
            for r in rows:
                sink.write(json.dumps(r) + "\n")
                n_runs += r["kind"] == "run"
    print(f"[deck_audit_human] {len(paths)} files scanned, {n_runs} runs kept -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
