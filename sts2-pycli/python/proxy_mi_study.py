"""Proxy-fidelity / mutual-information study scorer.

Design: ``../sts2-rs/docs/plans/2026-06-25-proxy-fidelity-mi-study.md``.

Reads ``proxy.jsonl`` (from ``run_arena proxy-study``): one row per
card-shaped self-play decision, carrying a battery of deck-quality proxies
(``boss_only`` / ``elite_only`` / ``boss_elite`` / ``chain_k2`` /
``random3`` / ``nextact_boss``) evaluated on the current deck, plus the
run's eventual outcome.

For each proxy P it measures the **residual** information about the
outcome Y, conditioned on cheap already-free features Z (act, hp-frac,
gold, ascension, deck-size, relic-count):

    baseline model  f(Z)    → AUC_Z,  logloss_Z
    augmented model f(Z,P)  → AUC_ZP, logloss_ZP
    ΔAUC = AUC_ZP - AUC_Z
    cMI  ≈ logloss_Z - logloss_ZP        (conditional MI in nats)
    cMI / mean(probe_sims)               ← the ranking key (info per compute)

Both models are gradient-boosted trees, fit on the SAME per-proxy row
subset (rows where that proxy is present), split by seed so no run
straddles train/test. ``boss_elite`` is the production panel — the row to
beat. This is a FIDELITY study only; a proxy winning here still has to win
an A/B win-rate sweep before adoption (Goodhart stays downstream).

    python proxy_mi_study.py --proxy proxy.jsonl --target reached2

Cheap target ``reached2`` (survived act 1) is robust when true wins are
near-zero; ``won`` / ``reached3`` / a custom ``--final-act-thresh`` also work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

# Cheap conditioning features Z — indices into ContextFeatures.scalars
# (sts2-runai eval::ranker_features::SCALAR_NAMES). Kept here as the
# already-free baseline the proxy must beat.
CHEAP_SCALARS = {
    "hp_frac": 2,
    "gold": 3,
    "act_index": 5,
    "ascension": 9,
    "deck_size": 10,
    "relic_count": 15,
}
# Proxy DeltaStats columns tested as features (the scalar-metric
# hypotheses — dmg/turn, hp-loss, velocity — live here too).
PROXY_COLS = ["win_rate", "loss_rate", "dmg_t5_mean", "hp_lost_mean", "turns_mean", "score"]


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def target_vector(rows: list[dict], target: str, thresh: int) -> np.ndarray:
    """Binary outcome label per row."""
    if target == "won":
        return np.array([int(r["won"]) for r in rows], dtype=int)
    if target == "reached2":
        return np.array([int(r["final_act"] >= 2) for r in rows], dtype=int)
    if target == "reached3":
        return np.array([int(r["final_act"] >= 3) for r in rows], dtype=int)
    if target == "final_act":
        return np.array([int(r["final_act"] >= thresh) for r in rows], dtype=int)
    raise SystemExit(f"unknown --target {target!r}")


def cheap_z(rows: list[dict]) -> np.ndarray:
    idx = list(CHEAP_SCALARS.values())
    return np.array([[r["context"]["scalars"][i] for i in idx] for r in rows], dtype=np.float64)


def proxy_lookup(rows: list[dict]) -> dict[str, list[int | None]]:
    """For each proxy id, the index into each row's `proxies` list (or None
    if that proxy is absent for the row — e.g. nextact_boss in the final act)."""
    ids: dict[str, list[int | None]] = {}
    all_ids = []
    for r in rows:
        seen = {p["id"]: k for k, p in enumerate(r["proxies"])}
        for pid in seen:
            if pid not in ids:
                ids[pid] = [None] * len(rows)
                all_ids.append(pid)
    for ri, r in enumerate(rows):
        for k, p in enumerate(r["proxies"]):
            ids[p["id"]][ri] = k
    return ids


def seed_holdout(seeds: np.ndarray, test_pct: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean test mask; whole seeds assigned to test so no run straddles."""
    uniq = np.unique(seeds)
    rng.shuffle(uniq)
    n_test = max(1, len(uniq) * test_pct // 100)
    test_seeds = set(uniq[:n_test].tolist())
    return np.array([s in test_seeds for s in seeds])


def fit_auc_ll(xtr, ytr, xte, yte, seed) -> tuple[float, float]:
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=3, random_state=seed
    )
    m.fit(xtr, ytr)
    p = m.predict_proba(xte)[:, 1]
    return roc_auc_score(yte, p), log_loss(yte, p, labels=[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proxy", required=True, type=Path, help="proxy.jsonl")
    ap.add_argument(
        "--target",
        default="won",
        choices=["won", "reached2", "reached3", "final_act"],
        help="binary outcome. `won` (default) is balanced for a strong on-policy "
        "agent; `reached2`/`reached3` are fallbacks if wins are near-0 or near-1.",
    )
    ap.add_argument("--final-act-thresh", type=int, default=3, help="for --target final_act")
    ap.add_argument("--test-pct", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.proxy)
    seeds = np.array([r["seed"] for r in rows])
    y = target_vector(rows, args.target, args.final_act_thresh)
    Z = cheap_z(rows)
    plut = proxy_lookup(rows)
    rng = np.random.default_rng(args.seed)

    base = float(y.mean())
    print(
        f"{len(rows)} rows, {len(np.unique(seeds))} seeds | target={args.target} "
        f"base-rate {base:.3f} ({int(y.sum())} pos)",
        file=sys.stderr,
    )
    if y.sum() == 0 or y.sum() == len(y):
        raise SystemExit(
            f"target {args.target!r} is degenerate (all {int(y[0])}). "
            "Pick a target with both classes — try reached2."
        )

    # Baseline f(Z) on the FULL set (reference; each proxy re-fits its own
    # baseline on its present-rows subset for an apples-to-apples Δ).
    te_full = seed_holdout(seeds, args.test_pct, rng)
    auc_z0, ll_z0 = fit_auc_ll(Z[~te_full], y[~te_full], Z[te_full], y[te_full], args.seed)
    print(f"baseline Z-only (full)   AUC {auc_z0:.3f}  logloss {ll_z0:.4f}\n", file=sys.stderr)

    results = []
    for pid, idxs in sorted(plut.items()):
        present = np.array([k is not None for k in idxs])
        if present.sum() < 50:
            print(f"  skip {pid}: only {int(present.sum())} present rows", file=sys.stderr)
            continue
        # Proxy feature matrix on present rows.
        P = np.array(
            [[rows[ri]["proxies"][idxs[ri]][c] for c in PROXY_COLS] for ri in np.where(present)[0]],
            dtype=np.float64,
        )
        Zp = Z[present]
        yp = y[present]
        sp = seeds[present]
        cost = float(
            np.mean([rows[ri]["proxies"][idxs[ri]]["probe_sims"] for ri in np.where(present)[0]])
        )
        if yp.sum() == 0 or yp.sum() == len(yp):
            print(f"  skip {pid}: degenerate target on present rows", file=sys.stderr)
            continue
        te = seed_holdout(sp, args.test_pct, np.random.default_rng(args.seed))
        if yp[te].sum() == 0 or yp[te].sum() == len(yp[te]):
            print(f"  skip {pid}: single-class test split", file=sys.stderr)
            continue
        auc_z, ll_z = fit_auc_ll(Zp[~te], yp[~te], Zp[te], yp[te], args.seed)
        ZP = np.hstack([Zp, P])
        auc_zp, ll_zp = fit_auc_ll(ZP[~te], yp[~te], ZP[te], yp[te], args.seed)
        results.append(
            {
                "proxy": pid,
                "n": int(present.sum()),
                "d_auc": auc_zp - auc_z,
                "cmi": ll_z - ll_zp,
                "mi_per_sim": (ll_z - ll_zp) / cost if cost > 0 else float("nan"),
                "auc_zp": auc_zp,
                "cost": cost,
            }
        )

    if not results:
        raise SystemExit("no proxy had enough present rows to score")
    results.sort(key=lambda r: r["mi_per_sim"], reverse=True)

    print(
        f"\n{'proxy':14}{'n':>7}{'ΔAUC':>9}{'cMI(nats)':>11}"
        f"{'cMI/sim':>12}{'AUC(Z+P)':>10}{'cost':>9}"
    )
    print("-" * 72)
    for r in results:
        star = "  ← baseline panel" if r["proxy"] == "boss_elite" else ""
        print(
            f"{r['proxy']:14}{r['n']:>7}{r['d_auc']:>+9.3f}{r['cmi']:>11.4f}"
            f"{r['mi_per_sim']:>12.2e}{r['auc_zp']:>10.3f}{r['cost']:>9.0f}{star}"
        )
    print(
        "\nRanked by cMI/sim (residual info per probe-sim). A proxy below "
        "boss_elite on cMI but far cheaper may still win on cMI/sim.\n"
        "Reminder: fidelity ≠ win-rate. A/B the winner before adopting.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
