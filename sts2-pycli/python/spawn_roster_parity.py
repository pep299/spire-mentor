#!/usr/bin/env python3
"""item#6: dedicated spawn-roster parity sweep (NO C# roster injection).

The combat sweep (parity_audit.py) builds the Rust roster NATIVELY first and
only falls back to injecting C#'s monster kinds / move-states when the two
disagree (sts2-rs-3010t / zdjjx) — necessary so RNG-branched GenerateMonsters
rosters don't cascade, but it also MASKS a genuine spawn bug: a wrong native
roster is silently overwritten with C#'s instead of being reported.

This tool removes the mask. For every encounter it inits both engines
independently, takes Rust's NATIVE roster (monsters=None — no injection), and
diffs the roster *shape* against C#: per slot the monster kind, max_hp (the
HP roll), and on-spawn powers, plus the count and order. A mismatch is either:

  * a real spawn bug (deterministic-roster encounter whose native Rust roster
    differs from C#) — the class this tool exists to surface; or
  * an RNG-branched encounter (FLYCONID / SLITHERING_STRANGLER / SLIMES_*)
    whose roster is rolled off a run-setup stream the synthetic builder can't
    replicate — expected, and allowlisted so it doesn't drown the signal.

Usage:
    scripts/wheel-venv.sh run -- python python/spawn_roster_parity.py
    scripts/wheel-venv.sh run -- python python/spawn_roster_parity.py --character Ironclad
    scripts/wheel-venv.sh run -- python python/spawn_roster_parity.py --show-allowlisted

Exit non-zero if any non-allowlisted encounter has a roster divergence.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from parity_fuzz import _csharp_init, _rust_init
from parity_fuzz_scenario import (
    _STARTER_DECK,
    ALL_ENCOUNTERS,
    Scenario,
)
from parity_fuzz_state import normalize_csharp, normalize_rust
from sts2_client import Game

#: Encounters whose C# GenerateMonsters is RNG-branched off a run-setup stream
#: the synthetic Rust scenario builder cannot replicate, so a native roster
#: divergence here is EXPECTED (the combat sweep injects C#'s roster for exactly
#: these). Keep this list short and bead-referenced — anything NOT here that
#: diverges is a real bug. (sts2-rs-3010t / zdjjx.)
RNG_BRANCHED_ALLOWLIST: frozenset[str] = frozenset(
    {
        "FLYCONID_NORMAL",
        "SLITHERING_STRANGLER_NORMAL",
        "SLIMES_NORMAL",
        "SLIMES_LARGE",
        "SLIMES_WEAK",  # same slime-split family as SLIMES_NORMAL/LARGE
        # BowlbugsNormal.cs:39-57 draws slots 1-2 via base.Rng.NextItem over
        # [Egg, Silk, Nectar] without replacement (two RNG reads off the
        # encounter stream). Rust's randomize_bowlbugs_normal port is faithful,
        # but the synthetic builder has no run-RNG stream so it returns the
        # static canonical fallback [Rock, Egg, Silk] — slot 2 lands Silk(40)
        # vs C#'s rolled Nectar(35). RNG-branched, not an engine bug.
        "BOWLBUGS_NORMAL",
        "BOWLBUGS_WEAK",  # same base.Rng.NextItem worker draw (BowlbugsWeak.cs:27-34)
        # RubyRaidersNormal.cs:38-49 draws 3 of 5 raiders via base.Rng.NextItem
        # without replacement; randomize_ruby_raiders_normal ports it 1:1 (same
        # in v0.111.0). The synthetic builder can't replay the encounter stream.
        "RUBY_RAIDERS_NORMAL",
    }
)


def _roster_sig(monsters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Roster shape we hold both engines to: per slot kind / max_hp / powers.

    Block is excluded (C# clears enemy block before player turn 1, so it is not
    observable here); hp == max_hp at spawn so only max_hp is kept.
    """
    return [
        {
            "kind": m.get("kind"),
            "max_hp": m.get("max_hp"),
            "powers": [{"kind": p["kind"], "amount": p["amount"]} for p in (m.get("powers") or [])],
        }
        for m in monsters
    ]


def _diff_roster(cs: list[dict], rs: list[dict]) -> list[str]:
    """Human-readable per-slot roster differences (empty == identical)."""
    out: list[str] = []
    if len(cs) != len(rs):
        out.append(f"count cs={len(cs)} rs={len(rs)}")
    for i in range(max(len(cs), len(rs))):
        c = cs[i] if i < len(cs) else None
        r = rs[i] if i < len(rs) else None
        if c == r:
            continue
        if c is None or r is None:
            out.append(f"slot[{i}] cs={c} rs={r}")
            continue
        if c["kind"] != r["kind"]:
            out.append(f"slot[{i}].kind cs={c['kind']} rs={r['kind']}")
        if c["max_hp"] != r["max_hp"]:
            out.append(f"slot[{i}].max_hp cs={c['max_hp']} rs={r['max_hp']}")
        if c["powers"] != r["powers"]:
            out.append(f"slot[{i}].powers cs={c['powers']} rs={r['powers']}")
    return out


def _check_encounter(character: str, encounter: str) -> list[str]:
    sc = Scenario(
        character=character,
        deck=_STARTER_DECK,
        hp=80,
        max_hp=80,
        relics=(),
        encounter=encounter,
        potions=(),
    )
    g = Game()
    try:
        cs_raw = _csharp_init(g, sc)
    finally:
        g.close()
    rs_state = _rust_init(sc, monsters=None)  # NATIVE roster — the whole point
    cs = normalize_csharp(cs_raw)["monsters"]
    rs = normalize_rust(rs_state)["monsters"]
    return _diff_roster(_roster_sig(cs), _roster_sig(rs))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--character", default="Ironclad")
    p.add_argument(
        "--show-allowlisted",
        action="store_true",
        help="also print allowlisted RNG-branched encounters that diverged",
    )
    p.add_argument(
        "--only", default=None, help="restrict to encounters whose id contains this substring"
    )
    args = p.parse_args(argv)

    encounters = [e for e in ALL_ENCOUNTERS if not args.only or args.only in e]
    real_bugs: list[tuple[str, list[str]]] = []
    allow_div: list[str] = []
    clean = 0
    for enc in encounters:
        try:
            diffs = _check_encounter(args.character, enc)
        except Exception as exc:
            real_bugs.append((enc, [f"HARNESS ERROR: {type(exc).__name__}: {exc}"]))
            continue
        if not diffs:
            clean += 1
        elif enc in RNG_BRANCHED_ALLOWLIST:
            allow_div.append(enc)
            if args.show_allowlisted:
                print(f"[allowlisted] {enc}: {'; '.join(diffs)}", file=sys.stderr)
        else:
            real_bugs.append((enc, diffs))

    print(
        f"\n=== spawn-roster parity ({args.character}): "
        f"{len(encounters)} encounters, {clean} native-clean, "
        f"{len(allow_div)} allowlisted-divergent, {len(real_bugs)} UNTRIAGED divergence(s) ===",
        file=sys.stderr,
    )
    for enc, diffs in real_bugs:
        print(f"  DIVERGENT {enc}", file=sys.stderr)
        for d in diffs[:8]:
            print(f"      {d}", file=sys.stderr)
    if real_bugs:
        print(
            "\nUNTRIAGED = native Rust roster differs from C# and the encounter is "
            "not allowlisted. Each is either (a) an RNG-branched roster the synthetic "
            "builder can't replicate → verify vs the C# GenerateMonsters decomp and "
            "move to RNG_BRANCHED_ALLOWLIST with a bead ref, or (b) a real Rust spawn "
            "bug (wrong kind/count/HP-roll/on-spawn-power) → fix. The combat sweep "
            "masks both by injecting C#'s roster; this tool is the only place they "
            "surface.",
            file=sys.stderr,
        )
    return 1 if real_bugs else 0


if __name__ == "__main__":
    raise SystemExit(main())
