# sts2-pycli/python/tools/asc_ab.py
"""Per-ascension paired A/B (McNemar) + harder-tilted blended delta across init seeds."""

import importlib.util
import json
import pathlib
import sys
from collections import defaultdict

from scipy import stats  # sts2-pycli/.venv has scipy

_spec = importlib.util.spec_from_file_location(
    "asc_blend", pathlib.Path(__file__).with_name("asc_blend.py")
)
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def _load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def per_level_delta(base, new):
    def key(r):
        return (r["seed"], r["character"], r["ascension"])

    bmap = {key(r): r for r in base}
    by_asc = defaultdict(list)
    for rn in new:
        rb = bmap.get(key(rn))
        if rb is not None:
            by_asc[rn["ascension"]].append((1 if rb["won"] else 0, 1 if rn["won"] else 0))
    out = {}
    for a, pairs in sorted(by_asc.items()):
        bw = sum(b for b, _ in pairs)
        nw = sum(n for _, n in pairs)
        tot = len(pairs)
        n01 = sum(1 for b, n in pairs if b == 0 and n == 1)  # new wins, base lost
        n10 = sum(1 for b, n in pairs if b == 1 and n == 0)
        disc = n01 + n10
        p = 1.0 if disc == 0 else stats.binomtest(n01, disc, 0.5).pvalue
        out[a] = (100.0 * bw / tot, 100.0 * nw / tot, 100.0 * (nw - bw) / tot, p)
    return out


def blended_from_delta(d, idx):  # idx: 0=base_pct,1=new_pct
    return ab.blended({a: vals[idx] for a, vals in d.items()})


def main():
    base = _load(sys.argv[1])
    print(f"{'asc':>3} {'base%':>7} {'new%':>7} {'Δpp':>7} {'mcN_p':>8}")
    blended_deltas = []
    for new_path in sys.argv[2:]:
        d = per_level_delta(base, _load(new_path))
        print(f"--- {new_path} ---")
        for a, (bp, npc, dp, p) in d.items():
            print(f"{a:>3} {bp:>7.2f} {npc:>7.2f} {dp:>+7.2f} {p:>8.4g}")
        bb = blended_from_delta(d, 0)
        bn = blended_from_delta(d, 1)
        print(f"blended base={bb:.3f} new={bn:.3f} Δ={bn - bb:+.3f}")
        blended_deltas.append(bn - bb)
    signs = {d > 0 for d in blended_deltas}
    print(f"\nblended Δ per init seed: {[round(x, 3) for x in blended_deltas]}")
    print(f"consistent sign: {len(signs) == 1}  (bless requires all same-sign & positive)")


if __name__ == "__main__":
    main()
