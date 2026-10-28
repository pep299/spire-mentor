"""Select the Phase-1 fork-replay seed sample from the anatomy JSON.

Rule (spec, sampling frame set by the Phase-0 loss mass): ALL losses of the
top-N killer encounters + a random sample of the remaining losses, up to
--max seeds total. Only seeds whose run actually died in combat (present in
fatal_by_seed) are eligible.

Usage:
    PYTHONPATH=python uv run --no-sync python -m runstats.select_fork_seeds \
        anatomy.json --top 3 --max 250 --out seeds.txt
"""

from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("anatomy_json")
    ap.add_argument("--top", type=int, default=2, help="killers to include fully")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--out", default="seeds.txt")
    args = ap.parse_args()
    with open(args.anatomy_json, encoding="utf-8") as f:
        rep = json.load(f)
    fatal = rep["fatal_by_seed"]  # seed -> {encounter, ...}
    # Exclude every pseudo-encounter bucket (angle-bracketed:
    # <no-combat-death>, <empty_actions>, <livelock>, ...) — they never
    # match fatal_by_seed, so letting one rank into the top-N would only
    # displace a real killer from the include-all-losses frame.
    top = {k["encounter"] for k in rep["killers"][: args.top] if not k["encounter"].startswith("<")}
    must = sorted(s for s, ff in fatal.items() if ff["encounter"] in top)
    rest = sorted(s for s, ff in fatal.items() if ff["encounter"] not in top)
    budget = max(0, args.max - len(must))
    rng = random.Random(20260717)
    sample = must + (rest if len(rest) <= budget else rng.sample(rest, budget))
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(f"{s}\n" for s in sorted(sample, key=int))
    print(
        f"selected {len(sample)} seeds ({len(must)} from top-{args.top} killers "
        f"{sorted(top)}, {len(sample) - len(must)} sampled from {len(rest)} others)"
    )


if __name__ == "__main__":
    main()
