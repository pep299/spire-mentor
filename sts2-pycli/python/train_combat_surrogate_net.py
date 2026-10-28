"""Torch two-head **combat surrogate** trainer + JSON export (AlphaZero
outer-loop plan, Milestone 1, Task 3).

Supersedes the sklearn prototype (`train_combat_surrogate.py`, kept in
place for reference/offline-metric comparisons) for productionization:
this script trains a small torch MLP and exports its weights in the
*exact* JSON schema the Rust inference seam
(`sts2-rs/crates/sts2-runai/src/eval/surrogate.rs::SurrogateNet`) loads —
the same hand-rolled-forward-plus-JSON-weights pattern as `ValueNet`.

Model
-----
``enc_emb = nn.Embedding(n_encounters + 1, enc_dim)`` (row ``n_encounters``
is the reserved out-of-distribution fallback, left at its default
zero-init — the loader clamps any ``encounter_idx >= n_encounters`` to
this row). The trunk is a single hidden layer over the concatenation of
the standardized dense context and the encounter embedding:

    h   = relu(W1^T @ [context_standardized ++ enc_emb[idx]] + b1)
    win = sigmoid(w_win @ h + b_win)
    hp  = clamp(w_hp @ h + b_hp, 0, 1)

matching `SurrogateNet::predict` bit-for-bit in structure (the loader
does the same context-standardize -> concat -> W1 -> relu -> two linear
heads forward). Loss is ``BCEWithLogits(win_logit, win_rate) +
lambda * SmoothL1Loss(hp_pred, hp_frac_lost)`` with ``lambda = 1.0``.

Data comes from the Rust labeler ``combat_surrogate_data`` (JSONL rows +
a sidecar ``*.schema.json`` pinning the dense context layout and the
encounter index space) — see `train_combat_surrogate.py::load` for the
prototype this trainer's `load()` is ported from (sparse-row densify).

    python train_combat_surrogate_net.py \\
        --labels       ../../sts2-rs/data/runai/surrogate/combat_labels_v107_realistic.jsonl \\
        --labels-extra ../../sts2-rs/data/runai/surrogate/combat_labels_v107.jsonl \\
        --schema       ../../sts2-rs/data/runai/surrogate/combat_labels_v107_realistic.jsonl.schema.json \\
        --out          ../../sts2-rs/data/runai/surrogate/combat_surrogate_net.json \\
        --epochs 200 --hidden 128 --enc-dim 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from torch import nn

# Minimum held-out REALISTIC rows (with both win/loss present) required to
# fit a `win_calib` — below this, calibration is fit on too little signal to
# trust, so we emit `null` (identity) rather than overfit a noisy mapping.
MIN_CALIB_ROWS = 20


def load(labels_paths: list[Path], schema_path: Path) -> dict[str, Any]:
    """Densify sparse JSONL label rows against the `.schema.json` sidecar.

    Ported from `train_combat_surrogate.py::load` (same sparse->dense
    layout: `deck(count,upg)` per card idx, then relic counts, then the
    dense scalar block) but returns the *context-only* dense matrix `X`
    (width `context_dim`, no encounter one-hot appended) plus a separate
    `enc_idx` column — this trainer's model consumes the encounter as an
    embedding lookup, not a one-hot input column, so it must stay a
    distinct tensor rather than folded into `X` the way the sklearn
    prototype does.

    `is_realistic` is derived PER ROW from that row's own `source` field
    (`combat_surrogate_data.rs` tags every harvested row `source:
    "realistic"|"synthetic"`): `r["source"] == "realistic"`. Rows with no
    `source` key at all (e.g. hand-written pre-`source` fixtures) fall
    back to `path_idx == 0` (the primary `--labels` file counts as
    realistic, `--labels-extra` doesn't) — this is a compatibility shim,
    not the primary signal; every current harvester output carries
    `source`. `is_realistic` feeds both the win-head calibrator (Task 5)
    and the gate (`eval_combat_surrogate_net.py`), both of which must
    score/fit on real self-play states rather than the synthetic/extra
    rows.
    """
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    n_cards = schema["n_cards"]
    n_relics = schema["n_relics"]
    ctx_dim = schema["context_dim"]
    relic_base = 2 * n_cards
    scalar_base = 2 * n_cards + n_relics

    rows: list[dict[str, Any]] = []
    is_realistic_list: list[bool] = []
    for path_idx, p in enumerate(labels_paths):
        file_rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows.extend(file_rows)
        for r in file_rows:
            src = r.get("source")
            is_realistic_list.append(src == "realistic" if src is not None else path_idx == 0)
    n = len(rows)
    if n == 0:
        raise ValueError(f"no label rows found in {labels_paths}")

    x = np.zeros((n, ctx_dim), dtype=np.float32)
    y_win = np.zeros(n, dtype=np.float32)
    y_hpfrac = np.zeros(n, dtype=np.float32)
    max_hp = np.zeros(n, dtype=np.float64)
    seeds = np.zeros(n, dtype=np.int64)
    enc_idx = np.zeros(n, dtype=np.int64)

    for i, r in enumerate(rows):
        for idx, count, upg in r["deck"]:
            x[i, 2 * idx] = count
            x[i, 2 * idx + 1] = upg
        for idx, count in r["relics"]:
            x[i, relic_base + idx] = count
        sc = r["scalars"]
        x[i, scalar_base : scalar_base + len(sc)] = sc
        enc_idx[i] = r["enc_idx"]
        y_win[i] = r["win_rate"]
        y_hpfrac[i] = r["hp_frac_lost"]
        max_hp[i] = r["max_hp"]
        seeds[i] = r["seed"]

    return dict(
        X=x,
        y_win=y_win,
        y_hpfrac=y_hpfrac,
        max_hp=max_hp,
        seeds=seeds,
        enc_idx=enc_idx,
        is_realistic=np.array(is_realistic_list, dtype=bool),
        schema=schema,
        ctx_dim=ctx_dim,
        n_enc=schema["n_encounters"],
        # Raw row dicts, verbatim, same order as every array above — needed
        # so `main` can re-serialize the held-out test split as-is (Fix #1,
        # `<out>.heldout.jsonl`) without reconstructing rows from the dense
        # arrays.
        rows=rows,
    )


def fit_win_calibrator(raw_p: np.ndarray, y_soft: np.ndarray, kind: str) -> dict[str, Any] | None:
    """Fit a `win_calib` mapping raw sigmoid `win_prob` -> empirical win
    rate on held-out REALISTIC rows.

    Fits against the **soft** `win_rate ∈ [0, 1]` target (the empirical
    win fraction over each label row's probe replays), NOT a binarized
    win/loss call — isotonic regression on the soft target yields
    `g(pred) = E[win | pred]` directly, which is both the Brier-optimal
    quantity for a value net and non-circular against the gate's binary
    Brier check (`eval_combat_surrogate_net.py`), which scores a
    different, thresholded target. `apply_win_calib` and the isotonic
    wire format (`x`/`y` breakpoints) are unchanged by this — only what
    `y` is fit against changed.

    The **Platt** branch is the exception: `LogisticRegression` fits
    discrete classes, not a continuous `[0, 1]` target, so it binarizes
    `y_soft` internally as a documented limitation of that path —
    isotonic (the default) is the correct soft-fit path; Platt is kept
    for comparison, not pretended to be soft-fit.

    Returns `None` (identity, i.e. `win_calib: null`) if there isn't
    enough held-out realistic signal to fit safely — too few rows, or
    the soft target has no spread (constant `win_rate`, which is
    degenerate for isotonic and would binarize to a single class for
    Platt) — rather than crash or emit a garbage calibrator.
    """
    if len(raw_p) < MIN_CALIB_ROWS or (float(y_soft.max()) - float(y_soft.min())) < 1e-6:
        return None
    p = np.clip(raw_p, 1e-6, 1 - 1e-6).astype(np.float64)
    if kind == "platt":
        # LogisticRegression cannot fit a soft [0, 1] target directly —
        # binarize here as a documented limitation (isotonic, the
        # default, is the correct soft-fit path; see docstring).
        y_bin = (y_soft >= 0.5).astype(int)
        if len(np.unique(y_bin)) < 2:
            # Soft target has spread but it's all on one side of 0.5
            # (e.g. win_rate entirely in [0.5, 0.7]) — binarizing would
            # collapse to a single class, which LogisticRegression can't
            # fit. Fall back to identity rather than crash.
            return None
        logit = np.log(p / (1 - p))
        clf = LogisticRegression()
        clf.fit(logit.reshape(-1, 1), y_bin)
        return {"kind": "platt", "a": float(clf.coef_[0, 0]), "b": float(clf.intercept_[0])}
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y_soft)
        return {
            "kind": "isotonic",
            "x": iso.X_thresholds_.tolist(),
            "y": iso.y_thresholds_.tolist(),
        }
    raise ValueError(f"unknown --calibration kind {kind!r}")


def apply_win_calib(raw_p: np.ndarray, calib: dict[str, Any] | None) -> np.ndarray:
    """Mirror `WinCalib::apply` in
    `sts2-rs/crates/sts2-runai/src/eval/surrogate.rs` — used so the
    `--emit-parity-fixture` torch forward applies the SAME calibration to
    its emitted `win_prob` as the Rust `forward` will, keeping the
    Rust<->torch parity fixture meaningful once `win_calib` is non-null.
    """
    if calib is None:
        return raw_p
    kind = calib["kind"]
    if kind == "platt":
        p = np.clip(raw_p, 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        z = calib["a"] * logit + calib["b"]
        out = 1.0 / (1.0 + np.exp(-z))
    elif kind == "isotonic":
        # np.interp clamps outside [x[0], x[-1]] to y[0]/y[-1] and does
        # piecewise-linear interpolation in between — exactly the Rust
        # `isotonic_interp` semantics, given non-decreasing breakpoints.
        out = np.interp(raw_p, calib["x"], calib["y"])
    else:
        raise ValueError(f"unknown win_calib kind {kind!r}")
    return np.clip(out, 0.0, 1.0)


def seed_split(
    seeds: np.ndarray, holdout_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Group-split by run seed — no seed appears in both train and test."""
    uniq = np.unique(seeds)
    rng.shuffle(uniq)
    n_test = max(1, round(len(uniq) * holdout_frac)) if len(uniq) > 1 else 0
    test_seeds = set(uniq[:n_test].tolist())
    test = np.array([s in test_seeds for s in seeds])
    train = ~test
    if not train.any():
        # Degenerate (all-one-seed) fixtures: fall back to training on
        # everything and evaluating on the same rows rather than crashing.
        train = np.ones_like(train)
    return train, test


class SurrogateMLP(nn.Module):
    """Mirrors `SurrogateNet::predict`'s forward exactly: one hidden layer
    over `[standardized context ++ encounter embedding]`, two linear
    heads (win-logit, hp)."""

    def __init__(self, context_dim: int, n_encounters: int, enc_dim: int, hidden: int) -> None:
        super().__init__()
        self.enc_emb = nn.Embedding(n_encounters + 1, enc_dim)
        # Row `n_encounters` is the reserved OOD fallback (Rust loader
        # clamps any encounter_idx >= n_encounters to it) — zero-init so
        # an untrained/rarely-seen OOD row contributes nothing extra.
        with torch.no_grad():
            self.enc_emb.weight[n_encounters].zero_()
        self.trunk = nn.Linear(context_dim + enc_dim, hidden)
        self.relu = nn.ReLU()
        self.win_head = nn.Linear(hidden, 1)
        self.hp_head = nn.Linear(hidden, 1)

    def forward(
        self, ctx: torch.Tensor, enc_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.enc_emb(enc_idx)
        h = self.relu(self.trunk(torch.cat([ctx, emb], dim=-1)))
        win_logit = self.win_head(h).squeeze(-1)
        hp = self.hp_head(h).squeeze(-1)
        return win_logit, hp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--labels-extra", type=Path, default=None, help="optional 2nd JSONL merged in")
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--enc-dim", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--hp-loss-weight", type=float, default=1.0, help="lambda on the HP SmoothL1 term"
    )
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--emit-parity-fixture",
        type=Path,
        default=None,
        help=(
            "write a Rust<->torch round-trip parity fixture "
            "({net, cases:[{ctx_dense (RAW, pre-standardization), encounter_idx, "
            "win_prob, hp_frac_lost}]}) for N held-out rows to this path "
            "(AlphaZero Milestone 1, Task 4)"
        ),
    )
    ap.add_argument(
        "--parity-cases",
        type=int,
        default=8,
        help="number of held-out rows to include in --emit-parity-fixture",
    )
    ap.add_argument(
        "--calibration",
        choices=("isotonic", "platt"),
        default="isotonic",
        help=(
            "win_calib fit kind, on the held-out REALISTIC split only "
            "(rows from the primary --labels file, not --labels-extra); "
            "falls back to null (identity) below MIN_CALIB_ROWS or if the "
            "split is single-class"
        ),
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    label_paths = [args.labels] + ([args.labels_extra] if args.labels_extra else [])
    d = load(label_paths, args.schema)
    x_all, seeds, enc_idx_all = d["X"], d["seeds"], d["enc_idx"]
    y_win_all, y_hpf_all = d["y_win"], d["y_hpfrac"]
    is_realistic_all = d["is_realistic"]
    ctx_dim, n_enc = d["ctx_dim"], d["n_enc"]
    n = x_all.shape[0]
    print(
        f"loaded {n} rows x {ctx_dim} context features "
        f"({len(np.unique(seeds))} seeds, {n_enc} encounters)"
    )

    tr, te = seed_split(seeds, args.holdout_frac, rng)
    print(f"split: {tr.sum()} train / {te.sum()} test rows (grouped by seed)")

    # Standardize context with TRAIN mean/std -> feat_mu/feat_sd (exported
    # verbatim; the Rust loader standardizes with these at inference time).
    feat_mu = x_all[tr].mean(axis=0)
    feat_sd = x_all[tr].std(axis=0)
    feat_sd = np.where(feat_sd > 1e-6, feat_sd, 1.0).astype(np.float32)  # loader requires sd > 0
    feat_mu = feat_mu.astype(np.float32)

    x_std = (x_all - feat_mu) / feat_sd

    ctx_t = torch.from_numpy(x_std.astype(np.float32))
    enc_t = torch.from_numpy(enc_idx_all)
    win_t = torch.from_numpy(y_win_all)
    hpf_t = torch.from_numpy(y_hpf_all)

    tr_idx = np.nonzero(tr)[0]
    te_idx = np.nonzero(te)[0]

    model = SurrogateMLP(
        context_dim=ctx_dim, n_encounters=n_enc, enc_dim=args.enc_dim, hidden=args.hidden
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss()

    n_train = len(tr_idx)
    batch_size = max(1, min(args.batch_size, n_train))
    for epoch in range(args.epochs):
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            batch = tr_idx[perm[start : start + batch_size]]
            b_ctx = ctx_t[batch]
            b_enc = enc_t[batch]
            b_win = win_t[batch]
            b_hpf = hpf_t[batch]

            opt.zero_grad()
            win_logit, hp_pred = model(b_ctx, b_enc)
            loss = bce(win_logit, b_win) + args.hp_loss_weight * huber(hp_pred, b_hpf)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(batch)
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch + 1 == args.epochs:
            print(f"  epoch {epoch + 1}/{args.epochs}  train loss={epoch_loss / n_train:.4f}")

    # ───────────────────────── offline eval ─────────────────────────
    model.eval()
    with torch.no_grad():
        win_logit_te, hp_pred_te = model(ctx_t[te_idx], enc_t[te_idx])
        win_prob_te = torch.sigmoid(win_logit_te).numpy()
        hp_pred_te_np = hp_pred_te.clamp(0.0, 1.0).numpy()

    y_win_te = y_win_all[te_idx]
    y_hpf_te = y_hpf_all[te_idx]
    win_true_bin = (y_win_te >= 0.5).astype(int)

    print("\n=== held-out metrics ===")
    try:
        auc = (
            roc_auc_score(win_true_bin, win_prob_te)
            if len(np.unique(win_true_bin)) > 1
            else float("nan")
        )
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(win_true_bin, np.clip(win_prob_te, 1e-6, 1 - 1e-6))
    print(f"  WIN   AUC={auc:.3f}  Brier={brier:.4f}")
    hp_mae = mean_absolute_error(y_hpf_te, hp_pred_te_np)
    print(f"  HP    MAE(frac)={hp_mae:.4f}")

    # ─────────────────────── win-head calibration ───────────────────────
    # Fit on the held-out REALISTIC rows only (real self-play states, not
    # --labels-extra) — search consumes win_prob as a probability, so it
    # must be calibrated on the distribution it will actually see. Fit
    # target is the SOFT win_rate (Fix #3), not the binarized outcome —
    # see `fit_win_calibrator` docstring.
    realistic_te_mask = is_realistic_all[te_idx]
    y_win_te_r = y_win_te[realistic_te_mask]
    win_calib = fit_win_calibrator(win_prob_te[realistic_te_mask], y_win_te_r, args.calibration)
    n_calib_rows = int(realistic_te_mask.sum())
    if win_calib is None:
        print(
            f"\nwin_calib: null (identity) — {n_calib_rows} held-out realistic rows "
            f"< MIN_CALIB_ROWS={MIN_CALIB_ROWS} or win_rate has no spread"
        )
    else:
        raw_r = win_prob_te[realistic_te_mask]
        calibrated = apply_win_calib(raw_r, win_calib)
        win_true_bin_r = win_true_bin[realistic_te_mask]
        # Binary Brier — kept for continuity with the gate's binary check
        # (still circular-adjacent since it's the trainer's own held-out
        # split, but comparable to prior runs).
        cal_brier = brier_score_loss(win_true_bin_r, np.clip(calibrated, 1e-6, 1 - 1e-6))
        raw_brier = brier_score_loss(win_true_bin_r, np.clip(raw_r, 1e-6, 1 - 1e-6))
        # Soft-label MSE vs the actual win_rate target the calibrator was
        # fit against (Fix #3) — the non-circular, Brier-optimal metric.
        raw_mse = float(np.mean((raw_r - y_win_te_r) ** 2))
        cal_mse = float(np.mean((calibrated - y_win_te_r) ** 2))
        print(f"\nwin_calib: {win_calib['kind']} fit on {n_calib_rows} held-out realistic rows")
        print(f"  binary Brier (win_rate>=0.5) : {raw_brier:.4f} -> {cal_brier:.4f}")
        print(f"  soft-label MSE (vs win_rate) : {raw_mse:.4f} -> {cal_mse:.4f}")

    # ───────────────────────── export ─────────────────────────
    w1_full = model.trunk.weight.detach().numpy()  # [hidden, context_dim+enc_dim]
    w1_rows = w1_full.T  # [context_dim+enc_dim, hidden] row-major, matches SurrogateNet's w1
    b1 = model.trunk.bias.detach().numpy()
    w_win = model.win_head.weight.detach().numpy().reshape(-1)
    b_win = float(model.win_head.bias.detach().item())
    w_hp = model.hp_head.weight.detach().numpy().reshape(-1)
    b_hp = float(model.hp_head.bias.detach().item())
    enc_emb = model.enc_emb.weight.detach().numpy()  # [n_encounters+1, enc_dim]

    net = {
        "schema_id": d["schema"]["schema_id"],
        "engine": d["schema"]["engine"],
        "context_dim": ctx_dim,
        "n_encounters": n_enc,
        "enc_dim": args.enc_dim,
        "hidden": args.hidden,
        "feat_mu": feat_mu.tolist(),
        "feat_sd": feat_sd.tolist(),
        "enc_emb": enc_emb.tolist(),
        "w1": w1_rows.tolist(),
        "b1": b1.tolist(),
        "w_win": w_win.tolist(),
        "b_win": b_win,
        "w_hp": w_hp.tolist(),
        "b_hp": b_hp,
        "win_calib": win_calib,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(net))
    print(f"\nwrote surrogate net -> {args.out}")

    emit_heldout_sidecar(
        out_path=args.out,
        rows=d["rows"],
        schema=d["schema"],
        te_idx=te_idx,
        is_realistic_all=is_realistic_all,
    )

    if args.emit_parity_fixture is not None:
        emit_parity_fixture(
            path=args.emit_parity_fixture,
            net=net,
            model=model,
            x_all=x_all,
            enc_idx_all=enc_idx_all,
            case_idx=(te_idx if len(te_idx) > 0 else tr_idx),
            n_cases=args.parity_cases,
        )


def emit_heldout_sidecar(
    out_path: Path,
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
    te_idx: np.ndarray,
    is_realistic_all: np.ndarray,
) -> None:
    """Write the held-out TEST split (Fix #1, train-set-contamination fix)
    as a sidecar next to the exported net: `<out-stem>.heldout.jsonl` (the
    original row dicts, verbatim, `source` intact) + `.schema.json` (a
    verbatim copy of the training schema).

    This is what `eval_combat_surrogate_net.py`'s `--labels` must point at
    for the go/no-go gate — scoring the training `--labels` file instead
    reports memorized, optimistic metrics, since the trainer trained on
    ~`1 - holdout_frac` of it. The full test split (both realistic and
    non-realistic rows, whatever `source` they carry) is emitted so the
    gate's own realism filter (`is_realistic`, derived from `source`)
    selects the realistic subset itself rather than trusting a second,
    possibly-inconsistent split decision here.
    """
    out_stem = out_path.name[: -len(".json")] if out_path.name.endswith(".json") else out_path.name
    heldout_path = out_path.parent / f"{out_stem}.heldout.jsonl"
    heldout_schema_path = out_path.parent / f"{out_stem}.heldout.jsonl.schema.json"

    heldout_rows = [rows[i] for i in te_idx.tolist()]
    n_test = len(heldout_rows)
    n_test_realistic = int(is_realistic_all[te_idx].sum())

    body = "\n".join(json.dumps(r) for r in heldout_rows)
    heldout_path.write_text(body + "\n" if body else "")
    heldout_schema_path.write_text(json.dumps(schema))

    print(
        f"wrote held-out test split -> {heldout_path} "
        f"({n_test} test rows, {n_test_realistic} realistic test rows)\n"
        f"  schema -> {heldout_schema_path}"
    )


def emit_parity_fixture(
    path: Path,
    net: dict[str, Any],
    model: SurrogateMLP,
    x_all: np.ndarray,
    enc_idx_all: np.ndarray,
    case_idx: np.ndarray,
    n_cases: int,
) -> None:
    """Write the Rust<->torch round-trip parity fixture (Task 4).

    CRITICAL convention (matches `SurrogateNet::predict_from_dense` in
    `crates/sts2-runai/src/eval/surrogate.rs`): `ctx_dense` is the RAW
    densified context — `x_all[row]`, width `context_dim`, BEFORE the
    `(x - feat_mu) / feat_sd` standardization the net's `feat_mu`/`feat_sd`
    encode. The torch forward computing `win_prob`/`hp_frac_lost` below
    standardizes internally (mirroring what the Rust loader does), so the
    Rust test can feed the same raw `ctx_dense` into `predict_from_dense`
    and land on the identical standardize-then-forward path. Emitting
    already-standardized values here would make the Rust side
    double-standardize and silently break parity.

    Task 5: `net["win_calib"]` (if non-null) is applied to `win_prob` here
    via `apply_win_calib` — the SAME calibration
    `SurrogateNet::forward`/`WinCalib::apply` applies on the Rust side —
    so the emitted `win_prob` is the fully-calibrated value the Rust
    `predict_from_dense` must reproduce within `1e-4`.
    """
    n = min(n_cases, len(case_idx))
    picked = case_idx[:n]

    feat_mu = np.asarray(net["feat_mu"], dtype=np.float32)
    feat_sd = np.asarray(net["feat_sd"], dtype=np.float32)

    ctx_raw = x_all[picked].astype(np.float32)
    ctx_std = (ctx_raw - feat_mu) / feat_sd
    enc_idx = enc_idx_all[picked]

    model.eval()
    with torch.no_grad():
        win_logit, hp_pred = model(
            torch.from_numpy(ctx_std.astype(np.float32)), torch.from_numpy(enc_idx)
        )
        win_prob = apply_win_calib(torch.sigmoid(win_logit).numpy(), net["win_calib"])
        hp_frac_lost = hp_pred.clamp(0.0, 1.0).numpy()

    cases = [
        {
            "ctx_dense": ctx_raw[i].tolist(),
            "encounter_idx": int(enc_idx[i]),
            "win_prob": float(win_prob[i]),
            "hp_frac_lost": float(hp_frac_lost[i]),
        }
        for i in range(n)
    ]
    fixture = {"net": net, "cases": cases}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote parity fixture ({n} cases) -> {path}")


if __name__ == "__main__":
    main()
