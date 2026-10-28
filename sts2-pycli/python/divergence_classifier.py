"""Free cascade/mislabel pre-filter for parity-sweep divergences.

Partitions a sweep's examples.json into {mislabel, pile_cascade,
likely_real} using rules over the example record alone — no agent tokens.
Only the likely_real bucket warrants investigation.

Usage:
  python python/divergence_classifier.py <tmp_dir_with_examples.json>
"""

import json
import re
import sys
from collections import Counter


def base_id(cid: str) -> str:
    return cid.split("+")[0] if cid else cid


def strip_idx(p: str) -> str:
    return re.sub(r"\[\d+\]", "[]", p or "?")


PILE_PREFIXES = ("draw_", "hand_", "discard_", "exhaust_", "pile")


def classify(ex: dict) -> tuple[str, str]:
    """Return (class, first_path)."""
    d = ex.get("divergences")
    d = d[0] if isinstance(d, list) and d else (d if isinstance(d, dict) else {})
    path = d.get("path", "") if isinstance(d, dict) else ""
    hand = [base_id(h) for h in ((ex.get("hand_before") or {}).get("csharp") or [])]
    deck = [base_id(c) for c in (ex.get("scenario", {}).get("deck") or [])]
    cid = base_id((ex.get("action") or {}).get("_card_id"))

    # Mislabel: the bucket's named card isn't even available this run.
    if cid and cid not in hand and cid not in deck:
        return "mislabel", path
    # Pile cascade: first-divergence is a draw/hand/discard count or order
    # field — almost always an upstream draw/RNG root, not the named card.
    if path.startswith(PILE_PREFIXES) or "draw_ids" in path or "hand_ids" in path:
        return "pile_cascade", path
    return "likely_real", path


def main() -> None:
    tmp = sys.argv[1]
    ex = json.load(open(f"{tmp}/examples.json", encoding="utf-8"))
    klass = Counter()
    real = []
    for k, v in ex.items():
        fam, _, move = k.partition("|")
        if move == "end_turn":
            klass["end_turn_cascade"] += 1
            continue
        c, path = classify(v)
        klass[c] += 1
        if c == "likely_real":
            dvs = v.get("divergences")
            dv = dvs[0] if isinstance(dvs, list) and dvs else dvs
            real.append(
                {
                    "bucket": k,
                    "seed": v.get("seed"),
                    "turn": v.get("turn"),
                    "path": strip_idx(path),
                    "div": dv,
                    "encounter": v.get("scenario", {}).get("encounter"),
                    "card": (v.get("action") or {}).get("_card_id"),
                }
            )

    print("== classification of distinct divergent buckets ==")
    for c, n in klass.most_common():
        print(f"  {n:4d}  {c}")
    print(f"\n== LIKELY-REAL shortlist ({len(real)}) — the only buckets worth agent tokens ==")
    # group likely-real by (stripped path) to reveal shared roots
    by_path = Counter(r["path"] for r in real)
    print("  -- by first-divergence field --")
    for p, n in by_path.most_common():
        print(f"    {n:3d}  {p}")
    print("  -- individual likely-real buckets --")
    for r in sorted(real, key=lambda r: r["path"]):
        print(
            f"    {r['bucket']:34} seed={r['seed']} t{r['turn']} {r['path']:22} {r['div']} enc={r['encounter']}"
        )


if __name__ == "__main__":
    main()
