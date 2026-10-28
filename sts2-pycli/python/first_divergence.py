"""First-divergence triage for the parity sweep.

`run_trial` already returns the FIRST diverging *action* (it snapshots and
diffs both engines after every action). What the flat manifest lacks is
*attribution*: which card / monster / power surface owns each divergence, and
which surfaces recur across many seeds (the high-yield engine roots).

This tool re-runs every divergent `seed:N` from the latest sweep manifest,
extracts (category, turn, action, first diverging field), derives a heuristic
root-cause SURFACE, and prints a triage grouped by that surface so the
biggest recurring root floats to the top — the per-step bisection the
remaining tail needs, without a two-sided event-trace harness.

Usage (from sts2-pycli):
  WHEEL_VENV_NO_BUILD=1 ./scripts/wheel-venv.sh run -- python python/first_divergence.py
  POS=1777,1835 ... python python/first_divergence.py        # explicit subset
  MANIFEST=<path> ... python python/first_divergence.py       # pin a manifest
"""

import glob
import json
import os
import sys
from collections import defaultdict

sys.argv = ["x"]
import parity_audit as pa  # noqa: E402


def latest_manifest() -> str:
    if os.environ.get("MANIFEST"):
        return os.environ["MANIFEST"]
    root = os.path.join(
        os.path.dirname(__file__), "..", "..", "sts2-rs", "docs", "audits", "sweeps"
    )
    cands = sorted(glob.glob(os.path.join(root, "*-full-manifest.json")), key=os.path.getmtime)
    return cands[-1]


def divergent_positions(manifest: str) -> list[int]:
    subj = json.load(open(manifest, encoding="utf-8"))["subjects"]
    pos = []
    for k, v in subj.items():
        if v.get("verdict") != "clean" and k.startswith("seed:"):
            pos.append(int(k.split(":", 1)[1]))
    return sorted(pos)


def first_substantive(divs) -> dict | None:
    """First diverging entry that is NOT a pure terminal/teardown plumbing field."""
    if not isinstance(divs, list):
        return None
    for d in divs:
        p = d.get("path", "")
        if p in ("combat_terminal", "player_won"):
            continue
        return d
    return divs[0] if divs else None


def surface(out: dict) -> str:
    """Heuristic root-cause surface for grouping.

    The action that *surfaced* the divergence is usually the owner for
    play_card (a card's own resolution); for end_turn the owner is the enemy
    turn, so we key on the lead monster species instead so enemy-AI / attack
    divergences cluster by monster.
    """
    act = out.get("action") or {}
    cat = out.get("category")
    d = first_substantive(out.get("divergences"))
    path = (d or {}).get("path", "")
    mons = out.get("_mons") or "?"

    if act.get("action") == "play_card":
        return f"card:{act.get('_card_id') or '?'}"
    if act.get("action") == "use_potion":
        return f"potion:{act.get('_card_id') or '?'}"
    if act.get("action") == "end_turn":
        # enemy-turn resolution — attribute to the monster roster + field class
        cls = (
            "hp"
            if ".hp" in path
            else ("pile" if cat in ("PILE_COUNT", "DRAW_ORDER") else path.split("[")[0] or cat)
        )
        return f"enemy_turn:{mons}:{cls}"
    return f"{cat}:{mons}"


def main() -> None:
    # This tool drives run_trial directly (not the sweep's main()), so the
    # freshness gate would otherwise be skipped — and a stale wheel makes every
    # "fix" look like a no-op here too. Run the same gate the sweep does.
    pa.assert_wheel_fresh()
    manifest = latest_manifest()
    trials = pa.assemble_v2_trials("full", list(pa.V2_AXES))
    if os.environ.get("POS"):
        positions = [int(x) for x in os.environ["POS"].split(",")]
    else:
        positions = divergent_positions(manifest)
    print(f"manifest: {manifest}")
    print(f"divergent positions: {len(positions)}")

    by_surface: dict[str, list[tuple]] = defaultdict(list)
    rows = []
    for p in positions:
        seed, sc, mt = trials[p]
        try:
            out = pa.run_trial(seed, mt, scenario=sc)
        except Exception as exc:
            rows.append((p, "?", "RUN_ERR", repr(exc)[:60]))
            continue
        scd = out.get("scenario") or {}
        mons = scd.get("monsters") or scd.get("enemy_kinds") or scd.get("encounter") or "?"
        out["_mons"] = mons if isinstance(mons, str) else ",".join(map(str, mons))
        if out.get("category") == "clean":
            # the manifest said divergent but a re-run is clean (flaky / fixed) —
            # surface it so we don't chase ghosts
            by_surface["NOW_CLEAN"].append((p, out["_mons"], "clean", ""))
            continue
        d = first_substantive(out.get("divergences"))
        fld = (d or {}).get("path", out.get("first_path") or "")
        s = surface(out)
        by_surface[s].append(
            (
                p,
                out["_mons"],
                out.get("category"),
                f"{fld} cs={(d or {}).get('csharp')!r} rs={(d or {}).get('rust')!r}",
            )
        )

    print("\n=== TRIAGE BY ROOT-CAUSE SURFACE (most recurrent first) ===")
    for s, items in sorted(by_surface.items(), key=lambda kv: -len(kv[1])):
        print(f"\n[{len(items):2d}] {s}")
        for p, _m, cat, detail in items[:6]:
            print(f"      seed:{p:<5} {cat:<13} {detail}")
        if len(items) > 6:
            print(f"      ... +{len(items) - 6} more: {[x[0] for x in items[6:]]}")


if __name__ == "__main__":
    main()
