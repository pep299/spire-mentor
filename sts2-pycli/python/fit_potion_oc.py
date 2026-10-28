"""Causal potion opportunity-cost fit from a randomized-drink corpus (option B).

GH #447 follow-up, plan Task 6. Input: `jitter-sweep.sh` output — Ironclad
A8 runs recorded at full detail, driven with `STS2_RUNAI_DRINK_JITTER=p:salt`
(a deterministic per-fight coin masks `DrinkPotion` at the search root, so
belt-at-fight-entry varies exogenously). This script:

1. parses each run into one row per fight entry: run outcome, act, tier,
   hp fraction, and the held belt bucketed into potion classes;
2. recomputes the mask coin offline (the seam's documented splitmix64
   formula) as a manipulation check — masked fights must show zero drinks;
3. fits an unpenalized logistic regression
   ``win ~ held[class × act] + hp_frac + floor + act + tier`` and converts
   each class coefficient to **HP-equivalents through the model's own HP
   coefficient** (``oc = coef_class / (coef_hp_frac / mean_max_hp)``) —
   which is exactly what "HP-equivalent" means;
4. writes an OC table in the `STS2_RUNAI_POTION_OC` v1 format
   (`source: "jitter_fit"`, Potion-Shaped Rock pinned to 0).

Caveats recorded in the output diagnostics: fight rows cluster within runs
(point estimates are fine, naive SEs would not be); unmasked fights' belts
remain partially policy-chosen, so exogeneity is strongest for the
masked-upstream variation the coins inject.

Usage (from sts2-pycli):
    PYTHONPATH=python uv run --frozen python python/fit_potion_oc.py \
        ../logs/2026-08-19-potion-oc/jitter --p 0.5 --salts 1,2,3,4,5 \
        --out ../logs/2026-08-19-potion-oc/oc_ironclad_v0.json
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

M64 = (1 << 64) - 1

CLASSES = ("scaling", "nuke", "heal", "other")  # + token, pinned to 0
CLASS_OF = {
    # boss ammunition per the expert analysis (50-56% of uses on boss floors)
    "POTION.POWER_POTION": "scaling",
    "POTION.STRENGTH_POTION": "scaling",
    "POTION.DEXTERITY_POTION": "scaling",
    "POTION.DUPLICATOR": "scaling",
    "POTION.TOUCH_OF_INSANITY": "scaling",
    "POTION.FYSH_OIL": "scaling",
    "POTION.POWDERED_DEMISE": "scaling",
    # flat damage, burned in trash by experts
    "POTION.FIRE_POTION": "nuke",
    "POTION.EXPLOSIVE_AMPOULE": "nuke",
    # healing / sustain
    "POTION.BLOOD_POTION": "heal",
    "POTION.FRUIT_JUICE": "heal",
    "POTION.FAIRY_IN_A_BOTTLE": "heal",
    # free relic token — no opportunity cost by construction
    "POTION.POTION_SHAPED_ROCK": "token",
}


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return (x ^ (x >> 31)) & M64


def coin_masks(run_seed: int, total_floor: int, salt: int, p: float) -> bool:
    """Exact replica of `potion_oc.rs::DrinkJitter::masks`."""
    material = (run_seed & 0xFFFFFFFF) ^ (total_floor & 0xFFFFFFFF) ^ ((salt << 32) & M64)
    return (splitmix64(material) >> 11) / float(1 << 53) < p


def boss_floors(run: dict) -> set[int]:
    out = set()
    for act in run.get("map_point_history") or []:
        for pt in act:
            if pt.get("map_point_type") == "Boss":
                out.add(pt["floor"])
    return out


def elite_floors(run: dict) -> set[int]:
    out = set()
    for act in run.get("map_point_history") or []:
        for pt in act:
            if pt.get("map_point_type") == "Elite":
                out.add(pt["floor"])
    return out


def fight_rows(path: str, salt: int, p: float) -> tuple[list[dict], int, int]:
    """One row per fight entry + (masked fights, drinks inside them)."""
    with gzip.open(path) as fh:
        run = json.load(fh)
    seed = int(Path(path).stem.split(".")[0])
    win = bool(run["win"])
    bosses, elites = boss_floors(run), elite_floors(run)
    rows: list[dict] = []
    masked_fights = masked_drinks = 0
    prev_kind = prev_floor = None
    cur_masked = False
    for d in run["sts2sim"]["decisions"]:
        off = d["offered"]
        kind, floor = off["kind"], d["floor"]
        if kind != "combat":
            prev_kind, prev_floor = kind, floor
            continue
        if prev_kind != "combat" or prev_floor != floor:
            cur_masked = coin_masks(seed, floor, salt, p)
            masked_fights += cur_masked
            belt = [str(t) for t in (d.get("potions") or [])]
            held = Counter(CLASS_OF.get(t, "other") for t in belt)
            max_hp = d.get("max_hp") or 0
            rows.append(
                {
                    "win": win,
                    "act": min(int(d["act"]), 3),
                    "floor": floor,
                    "tier": "B" if floor in bosses else ("E" if floor in elites else "N"),
                    "hp_frac": (d["hp"] / max_hp) if max_hp else 0.0,
                    "max_hp": max_hp,
                    "held": {c: held.get(c, 0) for c in CLASSES},
                    "masked": cur_masked,
                }
            )
        if off.get("action") == "potion" and cur_masked:
            masked_drinks += 1
        prev_kind, prev_floor = kind, floor
    return rows, masked_fights, masked_drinks


def design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = [f"held_{c}_act{a}" for c in CLASSES for a in (1, 2, 3)]
    names += ["hp_frac", "floor", "act2", "act3", "tierE", "tierB"]
    x = np.zeros((len(rows), len(names)))
    y = np.zeros(len(rows))
    for i, r in enumerate(rows):
        y[i] = r["win"]
        for j, c in enumerate(CLASSES):
            x[i, j * 3 + (r["act"] - 1)] = r["held"][c]
        base = len(CLASSES) * 3
        x[i, base] = r["hp_frac"]
        x[i, base + 1] = r["floor"] / 50.0
        x[i, base + 2] = r["act"] == 2
        x[i, base + 3] = r["act"] == 3
        x[i, base + 4] = r["tier"] == "E"
        x[i, base + 5] = r["tier"] == "B"
    return x, y, names


def fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unpenalized logistic fit; returns coefficients (no intercept term in
    the returned vector — sklearn keeps it separately)."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty=None, max_iter=2000)
    model.fit(x, y)
    return model.coef_[0]


def build_table(rows: list[dict], coefs: np.ndarray, names: list[str]) -> dict:
    coef = dict(zip(names, coefs, strict=True))
    mean_max_hp = float(np.mean([r["max_hp"] for r in rows if r["max_hp"]]))
    per_hp = coef["hp_frac"] / mean_max_hp
    diagnostics = {
        "rows": len(rows),
        "mean_max_hp": round(mean_max_hp, 2),
        "coef_hp_frac": round(float(coef["hp_frac"]), 5),
        "per_hp": round(float(per_hp), 7),
        "class_coefs": {
            f"{c}_act{a}": round(float(coef[f"held_{c}_act{a}"]), 5)
            for c in CLASSES
            for a in (1, 2, 3)
        },
    }
    entries = []
    for pid, cls in sorted(CLASS_OF.items()):
        for act in (1, 2, 3):
            if cls == "token":
                oc = 0.0
            elif per_hp <= 0:
                oc = None  # HP coefficient inverted — fit is uninterpretable
            else:
                oc = max(float(coef[f"held_{cls}_act{act}"]) / per_hp, 0.0)
            if oc is not None:
                entries.append({"potion": pid, "act": act, "oc": round(oc, 3)})
    # everything unlisted (class "other") prices through default_oc, set to
    # the "other" class fit at act 2 (mid-run) if interpretable, else 8.
    default = 8.0
    if per_hp > 0:
        default = max(float(coef["held_other_act2"]) / per_hp, 0.0)
    return {
        "version": 1,
        "character": "ironclad",
        "units": "hp_equiv",
        "source": "jitter_fit",
        "default_oc": round(default, 3),
        "entries": entries,
        "diagnostics": diagnostics,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", help="jitter dir holding salt_<k>/runs/*/*.run.gz")
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--salts", type=str, default="1,2,3,4,5")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    masked_fights = masked_drinks = 0
    for salt in (int(s) for s in args.salts.split(",")):
        files = sorted(glob.glob(f"{args.corpus}/salt_{salt}/runs/*/*.run.gz"))
        print(f"[salt {salt}] {len(files)} runs", flush=True)
        for f in files:
            r, mf, md = fight_rows(f, salt, args.p)
            rows.extend(r)
            masked_fights += mf
            masked_drinks += md

    total_fights = len(rows)
    share = masked_fights / total_fights if total_fights else 0.0
    print(
        f"fights {total_fights}, masked {masked_fights} ({share:.1%}), "
        f"drinks inside masked fights {masked_drinks}",
        flush=True,
    )
    # Manipulation check: the mask removes root drinks entirely.
    assert masked_drinks == 0, (
        f"{masked_drinks} drinks in masked fights — coin replication or seam broken"
    )
    assert abs(share - args.p) < 0.05, f"masked share {share:.2%} far from p={args.p}"

    x, y, names = design(rows)
    coefs = fit_logistic(x, y)
    table = build_table(rows, coefs, names)
    Path(args.out).write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(table["diagnostics"], indent=1))
    print(f"default_oc={table['default_oc']} -> wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
