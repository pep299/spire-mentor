"""Live (NOCACHE) single-trial reproduction for parity triage.

Usage:
  python python/repro_bucket.py <examples.json> "<BUCKET>"   # by bucket key
  python python/repro_bucket.py <examples.json> --seed <N>   # by seed

Reconstructs the Scenario embedded in examples.json and re-runs the trial
live (max_turns=6, the v2 cap), printing the full divergence list plus the
enriched cs/rs monster+player state at the diverging step.
"""

import json
import sys

import parity_audit as pa
from parity_fuzz_scenario import Scenario


def _resolve_via_assemble(char: str, seed: int) -> Scenario:
    """Resolve the exact Scenario for a v2-full trial index (`seed`)."""
    trials = pa.assemble_v2_trials(
        tier="full",
        axes=["upgraded", "arenas", "fuzz", "pairwise"],
        character=char,
    )
    for idx, sc, _ in trials:
        if idx == seed:
            return sc
    raise SystemExit(f"seed {seed} out of range for {char} (n={len(trials)})")


def _run(sc: Scenario, seed: int) -> None:
    print(f"seed={seed} enc={sc.encounter}")
    print(f"deck={sc.deck}")
    print(f"relics={sc.relics} potions={sc.potions} hp={sc.hp}/{sc.max_hp}")
    pa._CACHE = None  # force live C#
    out = pa.run_trial(seed=seed, max_turns=6, scenario=sc)
    print("category:", out.get("category"))
    print("turn:", out.get("turn"), "action:", out.get("action"))
    print("first_path:", out.get("first_path"))
    print("divergences:", json.dumps(out.get("divergences"), indent=1))


def main() -> int:
    if sys.argv[1] == "--by-index":
        # repro_bucket.py --by-index <Char> <seed>
        char, seed = sys.argv[2], int(sys.argv[3])
        _run(_resolve_via_assemble(char, seed), seed)
        return 0
    examples_path = sys.argv[1]
    d = json.load(open(examples_path, encoding="utf-8"))
    if sys.argv[2] == "--seed":
        seed = int(sys.argv[3])
        ex = None
        for v in d.values():
            vv = v if isinstance(v, list) else [v]
            for e in vv:
                if e["seed"] == seed:
                    ex = e
                    break
        if ex is None:
            print(f"seed {seed} not found in {examples_path}")
            return 1
    else:
        bucket = sys.argv[2]
        ex = d[bucket]
        ex = ex[0] if isinstance(ex, list) else ex

    sc_d = ex["scenario"]
    sc = Scenario(
        character=sc_d["character"],
        deck=tuple(sc_d["deck"]),
        hp=sc_d["hp"],
        max_hp=sc_d["max_hp"],
        relics=tuple(sc_d["relics"]),
        encounter=sc_d["encounter"],
        potions=tuple(sc_d.get("potions") or ()),
        draw_order=tuple(sc_d["draw_order"]) if sc_d.get("draw_order") else None,
        coverage_subject=sc_d.get("coverage_subject"),
    )
    print(f"seed={ex['seed']} enc={sc.encounter} deck={sc.deck}")
    print(f"relics={sc.relics} potions={sc.potions} hp={sc.hp}/{sc.max_hp}")
    pa._CACHE = None  # force live C#
    out = pa.run_trial(seed=ex["seed"], max_turns=6, scenario=sc)
    print("category:", out.get("category"))
    print("turn:", out.get("turn"), "action:", out.get("action"))
    print("first_path:", out.get("first_path"))
    print("divergences:", json.dumps(out.get("divergences"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
