#!/usr/bin/env python3
"""C#↔Rust map-generation parity for ALL THREE ACTS (the all-acts analogue of
the Act-1-only `run_setup_parity.py`).

`run_setup_parity.py` can only read each engine's LIVE `current_act_map`, which
is Act 1 on a fresh run — reaching acts 2/3 there would require winning the
bosses. But an act's map graph is a pure function of `(seed, act)`: both engines
seed it `Rng(seed, "act_{N}_map")` (C# `StandardActMap.CreateFor`; Rust
`RunRngSet::map_rng`), independent of run progress. So acts 2/3 are fully
determined at run START and can be generated combat-free:

  * C#   — the `dump_maps` headless command reconstructs each act's
           `StandardActMap` (per-act named RNG + `ActModel`) and dumps its
           node-type/edge graph.
  * Rust — the `RunManager.act_map_for(act)` accessor generates the map for an
           arbitrary act without advancing the run.

Like `run_setup_parity`, we compare the coordinate-independent invariant — the
SET of root→boss paths as canonical room-type sequences (the Rust `ancient`
prefix stripped) — per act. Identical path sets ⇒ the per-act map-gen RNG +
graph topology agree.

Act 1 here MUST reproduce `run_setup_parity`'s verdict (both compare the Act-1
map); a divergence at act 1 means the new `dump_maps`/`act_map_for` oracles are
unfaithful, not that the engines disagree.

Usage:
    scripts/wheel-venv.sh run -- python python/act23_map_parity.py -n 100
    scripts/wheel-venv.sh run -- python python/act23_map_parity.py --all-characters -n 50
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "python")

import sts2_client
import sts2_rs  # type: ignore[import-not-found]
from parity_audit import assert_wheel_fresh
from run_setup_parity import _csharp_paths, _rust_paths, _seed_for
from sts2_client import Game, add_game_version_arg, apply_game_version_arg

ALL_CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]
NUM_ACTS = 3


def _cs_act_maps(character: str, seed: str, ascension: int) -> list[dict]:
    """Per-act C# map graphs (``{rows, boss}``) via the `dump_maps` command."""
    g = Game()
    try:
        g.start(character=character, seed=seed, ascension=ascension)
        resp = g.send({"cmd": "dump_maps"})
    finally:
        g.close()
    if resp.get("type") == "error":
        raise RuntimeError(f"csharp dump_maps: {resp.get('message')}")
    return list(resp.get("acts", []))


def _rust_act_maps(
    character: str, seed: str, ascension: int, game_version: str | None = None
) -> list[list[dict]]:
    """Per-act Rust map graphs via `act_map_for(act)` (1-based).

    Built for ``game_version`` — by default the build the C# oracle runs
    (``STS2_GAME_VERSION`` / ``--game-version``), so both sides compare the
    same game. Without it the Rust side was always v0.107.1 (GH #893).
    """
    rm = sts2_rs.RunManager(
        string_seed=seed,
        character=character,
        ascension_level=ascension,
        neow_start=True,
        game_version=game_version or sts2_client.GAME_VERSION,
    )
    out: list[list[dict]] = []
    for act in range(1, NUM_ACTS + 1):
        am = rm.act_map_for(act)
        out.append(list(am) if am is not None else [])
    return out


def sweep(characters: list[str], ascension: int, n: int, start: int) -> int:
    """Diff per-act path sets across a seed range. Returns the failure count."""
    ok = [0, 0, 0]
    total = [0, 0, 0]
    fails = 0
    for character in characters:
        for i in range(start, start + n):
            seed = _seed_for(i)
            cs = _cs_act_maps(character, seed, ascension)
            rs = _rust_act_maps(character, seed, ascension)
            for ai in range(min(len(cs), len(rs), NUM_ACTS)):
                cs_paths = _csharp_paths(cs[ai])
                rs_paths = _rust_paths(rs[ai])
                total[ai] += 1
                if cs_paths == rs_paths:
                    ok[ai] += 1
                    continue
                fails += 1
                only_cs = sorted(cs_paths - rs_paths)
                only_rs = sorted(rs_paths - cs_paths)
                print(
                    f"[{character} {seed} act{ai + 1}] MAP DIFF (cs={len(cs_paths)} rs={len(rs_paths)} paths)"
                )
                for p in only_cs[:4]:
                    print(f"    only_cs: {' '.join(p)}")
                for p in only_rs[:4]:
                    print(f"    only_rs: {' '.join(p)}")
    for ai in range(NUM_ACTS):
        print(f"act{ai + 1} map parity: {ok[ai]}/{total[ai]} seeds path-set identical")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--all-characters", action="store_true")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("-n", "--num", type=int, default=100)
    ap.add_argument("--start", type=int, default=0, help="first seed index")
    add_game_version_arg(ap)
    args = ap.parse_args()
    apply_game_version_arg(args)

    assert_wheel_fresh()
    characters = ALL_CHARACTERS if args.all_characters else [args.character]
    fails = sweep(characters, args.ascension, args.num, args.start)
    print("\n✓ ALL-ACTS MAP PARITY CLEAN" if fails == 0 else f"\n✗ {fails} MAP PARITY FAILURE(S)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
