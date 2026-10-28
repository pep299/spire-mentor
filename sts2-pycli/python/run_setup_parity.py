#!/usr/bin/env python3
"""item#5: run-level (map-generation) parity sweep over ARBITRARY seeds.

The combat sweep (parity_audit.py) starts *inside* a synthetic combat, so it
never exercises the run-setup RNG cascade — the string-seed hash + map
generator that lay out the act. The existing r3_map_parity covers map structure
too, but only for the seeds present in the scraped `.run` corpus. This driver
fills the gap: it sweeps an arbitrary range of seeds, builds the act map on BOTH
the live C# headless (StartRun(seed) → get_map) and the Rust engine
(RunManager(string_seed).act_map()), and asserts the two agree.

Because the two map dumps use different coordinate origins and node inclusion
(C# excludes the Neow/ancient start node and emits the boss separately;
Rust includes both inline), a node-coordinate diff is spurious. Instead — like
r3_map_parity — we compare the coordinate-independent invariant: the SET of
root→boss paths, each as a canonical room-type sequence (the Rust 'ancient'
prefix stripped so both start at the row-1 monsters). Identical path sets ⇒ the
map-gen RNG + graph topology agree.

Usage:
    scripts/wheel-venv.sh run -- python python/run_setup_parity.py -n 100
    scripts/wheel-venv.sh run -- python python/run_setup_parity.py --character Silent -n 50

Exit non-zero if any swept seed's map diverges.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import sts2_client
import sts2_rs
from sts2_client import Game, add_game_version_arg, apply_game_version_arg

#: C# PointType.ToString() (PascalCase) ⇄ Rust point_type (snake_case).
_TYPE_CANON = {
    "monster": "monster",
    "elite": "elite",
    "restsite": "rest_site",
    "rest_site": "rest_site",
    "shop": "shop",
    "treasure": "treasure",
    "unknown": "unknown",
    "event": "event",
    "boss": "boss",
    "ancient": "ancient",
}


def _canon_type(t: str) -> str:
    return _TYPE_CANON.get(t.lower().replace("_", ""), t.lower())


def _enumerate_paths(
    nodes: dict[tuple[int, int], dict[str, Any]],
    starts: list[tuple[int, int]],
    boss_coords: set[tuple[int, int]],
) -> set[tuple[str, ...]]:
    """All start→boss paths as canonical room-type sequences (depth-first)."""
    paths: set[tuple[str, ...]] = set()

    def walk(coord: tuple[int, int], acc: list[str]) -> None:
        acc = [*acc, nodes[coord]["type"]]
        if coord in boss_coords:
            paths.add(tuple(acc))
            return
        children = [c for c in nodes[coord]["children"] if c in nodes]
        if not children:
            paths.add(tuple(acc))  # dead-end path (shouldn't reach boss)
            return
        for child in children:
            walk(child, acc)

    for s in starts:
        walk(s, [])
    return paths


def _csharp_paths(m: dict) -> set[tuple[str, ...]]:
    nodes: dict[tuple[int, int], dict[str, Any]] = {}
    for row in m.get("rows") or []:
        for n in row:
            nodes[n["col"], n["row"]] = {
                "type": _canon_type(n["type"]),
                "children": [(c["col"], c["row"]) for c in (n.get("children") or [])],
            }
    b = m["boss"]
    bcoord = (b["col"], b["row"])
    nodes[bcoord] = {"type": _canon_type(b["type"]), "children": []}
    min_row = min(r for _, r in nodes)
    starts = [c for c in nodes if c[1] == min_row]
    return _enumerate_paths(nodes, starts, {bcoord})


def _rust_paths(am: list[dict]) -> set[tuple[str, ...]]:
    nodes = {
        (p["col"], p["row"]): {
            "type": _canon_type(p["point_type"]),
            "children": [(c, r) for (c, r) in (p.get("children") or [])],
        }
        for p in am
    }
    max_row = max(r for _, r in nodes)
    boss = {c for c in nodes if c[1] == max_row and nodes[c]["type"] == "boss"} or {
        c for c in nodes if c[1] == max_row
    }
    min_row = min(r for _, r in nodes)
    starts = [c for c in nodes if c[1] == min_row]
    paths = _enumerate_paths(nodes, starts, boss)
    # Strip the leading Neow/ancient node so both engines start at row-1 monsters.
    return {p[1:] if p and p[0] == "ancient" else p for p in paths}


def _seed_for(i: int) -> str:
    """Deterministic synthetic string seed for sweep index i (base36-ish)."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    n, out = i, ""
    while True:
        n, r = divmod(n, len(alphabet))
        out = alphabet[r] + out
        if n == 0:
            break
    return f"PAR{out}"


def _rust_act1_map(
    character: str, seed: str, ascension: int, game_version: str | None = None
) -> list[dict] | None:
    """The Rust Act-1 map for ``seed``, built for ``game_version`` (default:
    the build the C# oracle runs, ``STS2_GAME_VERSION`` / ``--game-version``)
    so both sides compare the same game (GH #893)."""
    rm = sts2_rs.RunManager(
        string_seed=seed,
        character=character,
        ascension_level=ascension,
        game_version=game_version or sts2_client.GAME_VERSION,
    )
    return rm.act_map()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--character", default="Ironclad")
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("-n", "--num", type=int, default=100, help="seeds to sweep")
    p.add_argument("--start", type=int, default=0, help="first sweep index")
    p.add_argument("--show-mismatches", type=int, default=10)
    add_game_version_arg(p)
    args = p.parse_args(argv)
    apply_game_version_arg(args)

    clean = 0
    mismatches: list[tuple[str, int, int]] = []
    for i in range(args.start, args.start + args.num):
        seed = _seed_for(i)
        # A fresh headless per seed: StartRun refuses a second run on a live
        # RunState ("State is already set") and `quit` tears the process down,
        # so reusing one Game is not an option. Subprocess startup dominates the
        # cost — keep -n modest, or parallelize externally by seed range.
        g = Game()
        try:
            r = g.start(character=args.character, seed=seed, ascension=args.ascension)
            if r.get("type") == "error":
                mismatches.append((seed, -1, -1))
                print(f"  [C#-ERROR] {seed}: {r.get('message')}", file=sys.stderr)
                continue
            cs = _csharp_paths(g.get_map())
        except Exception as exc:
            print(f"  [HARNESS] {seed}: {type(exc).__name__}: {exc}", file=sys.stderr)
            mismatches.append((seed, -1, -1))
            continue
        finally:
            g.close()
        am = _rust_act1_map(args.character, seed, args.ascension)
        if am is None:
            mismatches.append((seed, len(cs), -1))
            continue
        rs = _rust_paths(am)
        if cs == rs:
            clean += 1
        else:
            mismatches.append((seed, len(cs - rs), len(rs - cs)))

    total = args.num
    print(
        f"\n=== run-setup (map) parity [{args.character} asc{args.ascension}]: "
        f"{clean}/{total} seeds map-clean, {len(mismatches)} divergent ===",
        file=sys.stderr,
    )
    for seed, cs_only, rs_only in mismatches[: args.show_mismatches]:
        print(
            f"  DIVERGENT {seed}: cs_only_paths={cs_only} rs_only_paths={rs_only}", file=sys.stderr
        )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
