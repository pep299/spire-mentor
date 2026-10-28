"""Learned **combat surrogate**: ``(pre-combat state, encounter) → outcome``.

Trains a model that maps a pre-combat setup (deck / HP / relics / potions /
gold / floor / ascension) **plus the encounter being entered** to that
combat's outcome — win probability and HP lost — in one forward pass, so the
outer-loop run-AI can score "how does this deck do in *this* fight" without
running a combat MCTS search.

Data comes from the Rust labeler ``combat_surrogate_data`` (one JSONL row per
real grounded combat probe). This script densifies the sparse context with the
sidecar ``*.schema.json``, appends the encounter one-hot, then fits two
gradient-boosted models:

* **win head** — regresses the probe ``win_rate`` (∈ [0,1]); reported both as a
  probability (Brier / log-loss / AUC) and thresholded win/loss accuracy.
* **hp head** — regresses ``hp_frac_lost`` (HP lost / max HP), the scale-free
  target; also reported back in absolute HP.

Evaluation is on a **held-out set of run seeds** (group split — no seed appears
in both train and test) and is compared against two baselines: a global-mean
predictor and a per-(encounter, ascension) mean predictor (the strongest
"lookup table" baseline a surrogate has to beat).

    python train_combat_surrogate.py \\
        --labels  ../sts2-rs/data/runai/surrogate/combat_labels.jsonl \\
        --schema  ../sts2-rs/data/runai/surrogate/combat_labels.jsonl.schema.json \\
        --out-dir ../sts2-rs/data/runai/surrogate

sklearn HistGradientBoosting is the prototype choice (best-in-class on tabular,
no tuning, CPU-fast). The productionization path — a torch MLP with a
hand-rolled-forward-plus-JSON-weights in-Rust inference seam — is noted in
the writeup.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)


def load(labels_path: Path, schema_path: Path):
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    n_cards = schema["n_cards"]
    n_relics = schema["n_relics"]
    n_enc = len(schema["encounters"])
    ctx_dim = schema["context_dim"]
    # ctx_dim = 2*n_cards + n_relics + n_scalars ; x = ctx ++ enc one-hot
    width = ctx_dim + n_enc
    relic_base = 2 * n_cards
    scalar_base = 2 * n_cards + n_relics

    rows = [
        json.loads(l) for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    n = len(rows)
    X = np.zeros((n, width), dtype=np.float32)
    y_win = np.zeros(n, dtype=np.float64)
    y_hpfrac = np.zeros(n, dtype=np.float64)
    max_hp = np.zeros(n, dtype=np.float64)
    seeds = np.zeros(n, dtype=np.int64)
    enc_idx = np.zeros(n, dtype=np.int64)
    asc = np.zeros(n, dtype=np.int64)

    for i, r in enumerate(rows):
        for idx, count, upg in r["deck"]:
            X[i, 2 * idx] = count
            X[i, 2 * idx + 1] = upg
        for idx, count in r["relics"]:
            X[i, relic_base + idx] = count
        sc = r["scalars"]
        X[i, scalar_base : scalar_base + len(sc)] = sc
        ei = r["enc_idx"]
        X[i, ctx_dim + ei] = 1.0
        enc_idx[i] = ei
        y_win[i] = r["win_rate"]
        y_hpfrac[i] = r["hp_frac_lost"]
        max_hp[i] = r["max_hp"]
        seeds[i] = r["seed"]
        # ascension is scalar index 9 in SCALAR_NAMES
        asc[i] = round(sc[9])

    return dict(
        X=X,
        y_win=y_win,
        y_hpfrac=y_hpfrac,
        max_hp=max_hp,
        seeds=seeds,
        enc_idx=enc_idx,
        asc=asc,
        schema=schema,
        ctx_dim=ctx_dim,
        n_enc=n_enc,
    )


def seed_split(seeds: np.ndarray, holdout_frac: float, rng: np.random.Generator):
    uniq = np.unique(seeds)
    rng.shuffle(uniq)
    n_test = max(1, round(len(uniq) * holdout_frac))
    test_seeds = set(uniq[:n_test].tolist())
    test = np.array([s in test_seeds for s in seeds])
    return ~test, test


def group_mean_baseline(key_tr, y_tr, key_te, global_mean):
    """Predict each test row's target as the train mean of its group key."""
    table: dict = {}
    sums: dict = {}
    counts: dict = {}
    for k, y in zip(key_tr, y_tr):
        sums[k] = sums.get(k, 0.0) + y
        counts[k] = counts.get(k, 0) + 1
    for k in sums:
        table[k] = sums[k] / counts[k]
    return np.array([table.get(k, global_mean) for k in key_te])


def regression_report(name, y_true, y_pred, unit=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:<28} MAE={mae:.4f}{unit}  RMSE={rmse:.4f}{unit}  R2={r2:+.3f}")
    return dict(mae=mae, rmse=rmse, r2=r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = load(args.labels, args.schema)
    X, seeds = d["X"], d["seeds"]
    n, width = X.shape
    print(
        f"loaded {n} rows × {width} features "
        f"({len(np.unique(seeds))} seeds, {d['n_enc']} encounters)"
    )

    tr, te = seed_split(seeds, args.holdout_frac, rng)
    print(
        f"split: {tr.sum()} train / {te.sum()} test rows "
        f"(grouped by seed, holdout={args.holdout_frac})"
    )

    # Drop columns constant across the TRAIN split (mostly the card-histogram
    # slots for cards never in this character's pool) — huge speedup, no info
    # loss. Mask is fit on train only (no test leakage).
    col_min = X[tr].min(axis=0)
    col_max = X[tr].max(axis=0)
    keep = col_max > col_min
    print(f"pruned {(~keep).sum()} constant columns → {keep.sum()} active features")
    X = X[:, keep]
    width = X.shape[1]

    y_win, y_hpf = d["y_win"], d["y_hpfrac"]
    max_hp = d["max_hp"]
    win_true_bin = (y_win[te] >= 0.5).astype(int)

    metrics = {"n_rows": n, "n_features": width, "n_train": int(tr.sum()), "n_test": int(te.sum())}

    # ───────────────────────── WIN head ─────────────────────────
    print("\n=== WIN (probe win_rate ∈ [0,1]) ===")
    t0 = time.time()
    win_model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=args.seed,
    )
    win_model.fit(X[tr], y_win[tr])
    win_pred = np.clip(win_model.predict(X[te]), 0.0, 1.0)
    print(f"  (fit {time.time() - t0:.1f}s, {win_model.n_iter_} trees)")

    # baselines
    gm = y_win[tr].mean()
    win_base_global = np.full(te.sum(), gm)
    key_tr = list(zip(d["enc_idx"][tr].tolist(), d["asc"][tr].tolist()))
    key_te = list(zip(d["enc_idx"][te].tolist(), d["asc"][te].tolist()))
    win_base_lookup = group_mean_baseline(key_tr, y_win[tr], key_te, gm)

    print("  regression on win_rate:")
    metrics["win_reg_global_mean"] = regression_report(
        "baseline: global mean", y_win[te], win_base_global
    )
    metrics["win_reg_lookup"] = regression_report(
        "baseline: (enc,asc) mean", y_win[te], win_base_lookup
    )
    metrics["win_reg_model"] = regression_report("SURROGATE", y_win[te], win_pred)

    # classification framing (threshold 0.5)
    def clf_metrics(pred):
        p = np.clip(pred, 1e-6, 1 - 1e-6)
        acc = float(((p >= 0.5).astype(int) == win_true_bin).mean())
        brier = float(np.mean((p - win_true_bin) ** 2))
        try:
            auc = float(roc_auc_score(win_true_bin, p))
        except ValueError:
            auc = float("nan")
        ll = float(log_loss(win_true_bin, p, labels=[0, 1]))
        return dict(accuracy=acc, auc=auc, brier=brier, log_loss=ll)

    base_rate = float(win_true_bin.mean())
    print(f"  classification (win = win_rate≥0.5; test base-rate {base_rate:.3f}):")
    for nm, pred in [
        ("baseline: global mean", win_base_global),
        ("baseline: (enc,asc) mean", win_base_lookup),
        ("SURROGATE", win_pred),
    ]:
        m = clf_metrics(pred)
        print(
            f"    {nm:<26} acc={m['accuracy']:.3f}  AUC={m['auc']:.3f}  "
            f"Brier={m['brier']:.4f}  logloss={m['log_loss']:.4f}"
        )
        metrics[f"win_clf_{'model' if nm == 'SURROGATE' else nm.split(':')[1].strip()}"] = m

    # ───────────────────────── HP head ─────────────────────────
    print("\n=== HP LOST (fraction of max HP) ===")
    t0 = time.time()
    hp_model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=args.seed,
    )
    hp_model.fit(X[tr], y_hpf[tr])
    hp_pred = np.clip(hp_model.predict(X[te]), 0.0, None)
    print(f"  (fit {time.time() - t0:.1f}s, {hp_model.n_iter_} trees)")

    gmh = y_hpf[tr].mean()
    hp_base_global = np.full(te.sum(), gmh)
    hp_base_lookup = group_mean_baseline(key_tr, y_hpf[tr], key_te, gmh)

    print("  regression on hp_frac_lost:")
    metrics["hp_reg_global_mean"] = regression_report(
        "baseline: global mean", y_hpf[te], hp_base_global
    )
    metrics["hp_reg_lookup"] = regression_report(
        "baseline: (enc,asc) mean", y_hpf[te], hp_base_lookup
    )
    metrics["hp_reg_model"] = regression_report("SURROGATE", y_hpf[te], hp_pred)

    print("  back in absolute HP (pred_frac × max_hp):")
    hp_abs_true = y_hpf[te] * max_hp[te]
    regression_report("baseline: global mean", hp_abs_true, hp_base_global * max_hp[te], unit=" hp")
    regression_report(
        "baseline: (enc,asc) mean", hp_abs_true, hp_base_lookup * max_hp[te], unit=" hp"
    )
    metrics["hp_abs_model"] = regression_report(
        "SURROGATE", hp_abs_true, hp_pred * max_hp[te], unit=" hp"
    )

    # ───────────────────────── persist ─────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "surrogate_metrics.json").write_text(json.dumps(metrics, indent=2))
    try:
        import joblib

        joblib.dump(
            {"win": win_model, "hp": hp_model, "schema": d["schema"], "keep_columns": keep},
            args.out_dir / "combat_surrogate.joblib",
        )
        print(f"\nsaved models → {args.out_dir / 'combat_surrogate.joblib'}")
    except Exception as e:  # joblib optional
        print(f"\n(model not pickled: {e})")
    print(f"metrics → {args.out_dir / 'surrogate_metrics.json'}")


if __name__ == "__main__":
    main()
