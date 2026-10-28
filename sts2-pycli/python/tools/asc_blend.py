"""Per-ascension win-rate breakdown + harder-tilted blended scalar (plan Global Constraints)."""

import json
import sys
from collections import defaultdict

BLEND_WEIGHTS = {a: (a + 1) for a in range(10)}  # A0..A9 -> 1..10
BLEND_WEIGHTS[10] = 5  # A10 = half of A9 (A10 is A9 + DoubleBoss only)


def per_level_winrate(rows):
    won = defaultdict(int)
    tot = defaultdict(int)
    for r in rows:
        a = int(r["ascension"])
        tot[a] += 1
        won[a] += 1 if r["won"] else 0
    return {a: (won[a], tot[a], 100.0 * won[a] / tot[a]) for a in sorted(tot)}


def blended(per_level):
    num = sum(BLEND_WEIGHTS[a] * rate for a, rate in per_level.items())
    den = sum(BLEND_WEIGHTS[a] for a in per_level)
    return num / den if den else 0.0


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    rows = load_rows(sys.argv[1])
    pl = per_level_winrate(rows)
    print(f"{'asc':>3} {'won':>5} {'tot':>5} {'rate%':>7} {'wt':>4}")
    for a, (w, t, rate) in pl.items():
        print(f"{a:>3} {w:>5} {t:>5} {rate:>7.2f} {BLEND_WEIGHTS[a]:>4}")
    print(f"blended (tilted) = {blended({a: r for a, (_, _, r) in pl.items()}):.3f}")


if __name__ == "__main__":
    main()
