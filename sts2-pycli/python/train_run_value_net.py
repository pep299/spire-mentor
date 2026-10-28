"""Torch whole-run WIN-PROBABILITY value net trainer + JSON export
(AlphaZero outer-loop plan, Milestone 4).

`ContextFeatures -> P(win)` plus an act-reached auxiliary head, trained on
the human corpus (`human_features` binary output: `features.jsonl` +
`decisions.jsonl` + `schema.json`). Exports the *exact* JSON schema the
Rust inference seam
(`sts2-rs/crates/sts2-runai/src/eval/run_value_net.rs::RunValueNet`)
loads — same hand-rolled-forward-plus-JSON-weights pattern as
`SurrogateNet` / `ValueNet`, and this trainer mirrors
`train_combat_surrogate_net.py` structurally (argparse shape, seed-grouped
split, standardize-on-train-mean/std, isotonic win-calibration, parity
fixture) minus the encounter embedding, plus the act-reached aux head.

Model
-----
    h         = relu(W1^T @ context_standardized + b1)
    win_logit = w_val @ h + b_val          -> sigmoid -> win_prob
    act_logit = W_act^T @ h + b_act        -> softmax over 4 buckets

matching `RunValueNet::forward` bit-for-bit in structure. Loss is
``BCEWithLogitsLoss(win_logit, win) + act_loss_weight *
CrossEntropyLoss(act_logits, act_bucket)``.

Data / labels
-------------
`features.jsonl` carries one sparse `ContextFeatures` row per floor
(`{seed, step, context: {deck, relics, scalars}, ...}`); `decisions.jsonl`
carries one `{seed, act, win}` row per RUN. Every floor-row of a run is
joined (by `seed`) to that run's terminal `win` (the value target) and
`act` (the auxiliary target, bucketed to `clamp(act - 1, 0, 3)` — matches
`human_features`'s `target = acts_reached + win`, so losers land in
buckets 0-2 and winners saturate bucket 3). `densify`/`load_schema` are
imported (reused, not copied) from `train_value.py` (mirrors
`ContextFeatures::write_dense`); `seed_split`/`fit_win_calibrator`/
`apply_win_calib`/`MIN_CALIB_ROWS` are imported from
`train_combat_surrogate_net.py` (same seed-grouped-split +
isotonic-calibration contracts, no need to re-derive them).

    python train_run_value_net.py \\
        --features  /tmp/human-v/features.jsonl \\
        --decisions /tmp/human-v/decisions.jsonl \\
        --schema    schema.json \\
        --out       ../../sts2-rs/data/runai/nets/run_value_net.json \\
        --epochs 300 --hidden 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn
from train_combat_surrogate_net import (
    MIN_CALIB_ROWS,
    apply_win_calib,
    fit_win_calibrator,
    seed_split,
)
from train_value import densify, load_schema

# Number of act-reached buckets the aux head predicts over — MUST match
# `run_value_net.rs::N_ACT_BUCKETS`.
N_ACT_BUCKETS = 4


def load(features_path: Path, decisions_path: Path, schema: dict[str, Any]) -> dict[str, Any]:
    """Join `features.jsonl` (per-floor context rows) against
    `decisions.jsonl` (per-run `{seed, act, win}`) by `seed`.

    Every floor-row of a run inherits that run's terminal `win` (value
    target, already binary) and `act` (bucketed to `clamp(act - 1, 0, 3)`
    for the aux head — `human_features` writes `act = acts_reached + win`,
    so losers occupy buckets 0-2 by act reached and winners saturate
    bucket 3). Standard MC value-learning: every intermediate state of a
    trajectory gets the trajectory's final outcome as its label.
    """
    dec_by_seed: dict[int, tuple[int, int]] = {}
    for line in decisions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        seed = int(r["seed"])
        dec_by_seed[seed] = (int(r.get("win", 0)), int(r.get("act", 1)))

    rows: list[dict[str, Any]] = []
    xs: list[np.ndarray] = []
    y_win: list[float] = []
    y_act: list[int] = []
    seeds: list[int] = []
    n_dropped = 0
    for line in features_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        seed = int(r["seed"])
        dec = dec_by_seed.get(seed)
        if dec is None:
            n_dropped += 1
            continue
        win, act = dec
        xs.append(densify(r["context"], schema))
        y_win.append(float(win))
        y_act.append(max(0, min(N_ACT_BUCKETS - 1, act - 1)))
        seeds.append(seed)
        rows.append(r)

    if n_dropped > 0:
        print(
            f"dropped {n_dropped} feature rows with no matching decision seed",
            file=sys.stderr,
        )

    if not xs:
        raise ValueError(
            f"no (feature, decision) pairs after join — seed mismatch between "
            f"{features_path} and {decisions_path}?"
        )

    return dict(
        X=np.stack(xs),
        y_win=np.asarray(y_win, dtype=np.float32),
        y_act=np.asarray(y_act, dtype=np.int64),
        seeds=np.asarray(seeds, dtype=np.int64),
        ctx_dim=schema["context_dim"],
        schema=schema,
        # Raw row dicts, verbatim, same order as every array above — needed
        # so `main` can re-serialize the held-out test split as-is, same
        # convention as `train_combat_surrogate_net.py::emit_heldout_sidecar`.
        rows=rows,
    )


class RunValueMLP(nn.Module):
    """Mirrors `RunValueNet::forward` exactly: one hidden ReLU layer over
    the standardized context, a scalar win-logit head, and a 4-way
    act-bucket logit head."""

    def __init__(self, context_dim: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Linear(context_dim, hidden)
        self.relu = nn.ReLU()
        self.win_head = nn.Linear(hidden, 1)
        self.act_head = nn.Linear(hidden, N_ACT_BUCKETS)

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.relu(self.trunk(ctx))
        win_logit = self.win_head(h).squeeze(-1)
        act_logits = self.act_head(h)
        return win_logit, act_logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--act-loss-weight",
        type=float,
        default=0.3,
        help="lambda on the act-reached CrossEntropy aux term",
    )
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--emit-parity-fixture",
        type=Path,
        default=None,
        help=(
            "write a Rust<->torch round-trip parity fixture "
            "({net, cases:[{ctx_dense (RAW, pre-standardization), win_prob, "
            "act_probs}]}) for N held-out rows to this path"
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
            "win_calib fit kind, on the held-out split's win predictions vs "
            "the binary win target; falls back to null (identity) below "
            "MIN_CALIB_ROWS or if the split is single-class"
        ),
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    schema = load_schema(args.schema)
    d = load(args.features, args.decisions, schema)
    x_all, seeds = d["X"], d["seeds"]
    y_win_all, y_act_all = d["y_win"], d["y_act"]
    ctx_dim = d["ctx_dim"]
    n = x_all.shape[0]
    print(
        f"loaded {n} rows x {ctx_dim} context features "
        f"({len(np.unique(seeds))} seeds); win-rate {y_win_all.mean():.3f}"
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
    win_t = torch.from_numpy(y_win_all)
    act_t = torch.from_numpy(y_act_all)

    tr_idx = np.nonzero(tr)[0]
    te_idx = np.nonzero(te)[0]

    model = RunValueMLP(context_dim=ctx_dim, hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    n_train = len(tr_idx)
    batch_size = max(1, min(args.batch_size, n_train))
    for epoch in range(args.epochs):
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            batch = tr_idx[perm[start : start + batch_size]]
            b_ctx = ctx_t[batch]
            b_win = win_t[batch]
            b_act = act_t[batch]

            opt.zero_grad()
            win_logit, act_logits = model(b_ctx)
            loss = bce(win_logit, b_win) + args.act_loss_weight * ce(act_logits, b_act)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(batch)
        if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch + 1 == args.epochs:
            print(f"  epoch {epoch + 1}/{args.epochs}  train loss={epoch_loss / n_train:.4f}")

    # ───────────────────────── offline eval ─────────────────────────
    model.eval()
    with torch.no_grad():
        win_logit_te, act_logits_te = model(ctx_t[te_idx])
        win_prob_te = torch.sigmoid(win_logit_te).numpy()
        act_pred_te = act_logits_te.argmax(dim=-1).numpy()

    y_win_te = y_win_all[te_idx]
    y_act_te = y_act_all[te_idx]
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
    print(f"  WIN AUC={auc:.3f}  Brier={brier:.4f}")
    act_acc = float((act_pred_te == y_act_te).mean())
    print(f"  ACT bucket accuracy={act_acc:.3f} (n={len(y_act_te)})")

    # ─────────────────────── win-head calibration ───────────────────────
    # Fit on the held-out split's win predictions vs the binary win target
    # (the human-corpus `win` label is already the hard 0/1 outcome, unlike
    # the surrogate's soft probe `win_rate` — `fit_win_calibrator` handles
    # either as its `y_soft` argument, since a 0/1 array is a degenerate
    # soft target).
    win_calib = fit_win_calibrator(win_prob_te, y_win_te, args.calibration)
    n_calib_rows = len(te_idx)
    if win_calib is None:
        print(
            f"\nwin_calib: null (identity) — {n_calib_rows} held-out rows "
            f"< MIN_CALIB_ROWS={MIN_CALIB_ROWS} or win target has no spread"
        )
    else:
        calibrated = apply_win_calib(win_prob_te, win_calib)
        cal_brier = brier_score_loss(win_true_bin, np.clip(calibrated, 1e-6, 1 - 1e-6))
        raw_brier = brier_score_loss(win_true_bin, np.clip(win_prob_te, 1e-6, 1 - 1e-6))
        print(f"\nwin_calib: {win_calib['kind']} fit on {n_calib_rows} held-out rows")
        print(f"  binary Brier: {raw_brier:.4f} -> {cal_brier:.4f}")

    # ───────────────────────── export ─────────────────────────
    w1_full = model.trunk.weight.detach().numpy()  # [hidden, context_dim]
    w1_rows = w1_full.T  # [context_dim, hidden] row-major, matches RunValueNet's w1
    b1 = model.trunk.bias.detach().numpy()
    w_val = model.win_head.weight.detach().numpy().reshape(-1)  # [hidden]
    b_val = float(model.win_head.bias.detach().item())
    w_act_full = model.act_head.weight.detach().numpy()  # [4, hidden]
    w_act_rows = w_act_full.T  # [hidden, 4] row-major, matches RunValueNet's w_act
    b_act = model.act_head.bias.detach().numpy()  # [4]

    net = {
        "schema_id": schema["schema_id"],
        "engine": schema.get("engine", "unknown"),
        "context_dim": ctx_dim,
        "hidden": args.hidden,
        "feat_mu": feat_mu.tolist(),
        "feat_sd": feat_sd.tolist(),
        "w1": w1_rows.tolist(),
        "b1": b1.tolist(),
        "w_val": w_val.tolist(),
        "b_val": b_val,
        "w_act": w_act_rows.tolist(),
        "b_act": b_act.tolist(),
        "win_calib": win_calib,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(net))
    print(f"\nwrote run value net -> {args.out}")

    emit_heldout_sidecar(out_path=args.out, rows=d["rows"], schema=schema, te_idx=te_idx)

    if args.emit_parity_fixture is not None:
        emit_parity_fixture(
            path=args.emit_parity_fixture,
            net=net,
            model=model,
            x_all=x_all,
            case_idx=(te_idx if len(te_idx) > 0 else tr_idx),
            n_cases=args.parity_cases,
        )


def emit_heldout_sidecar(
    out_path: Path,
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
    te_idx: np.ndarray,
) -> None:
    """Write the held-out TEST split as a sidecar next to the exported net:
    `<out-stem>.heldout.jsonl` (the original row dicts, verbatim) +
    `.schema.json` (a verbatim copy of the training schema) — same
    train/gate-split-contamination-avoidance convention as
    `train_combat_surrogate_net.py::emit_heldout_sidecar`.
    """
    out_stem = out_path.name[: -len(".json")] if out_path.name.endswith(".json") else out_path.name
    heldout_path = out_path.parent / f"{out_stem}.heldout.jsonl"
    heldout_schema_path = out_path.parent / f"{out_stem}.heldout.jsonl.schema.json"

    heldout_rows = [rows[i] for i in te_idx.tolist()]
    body = "\n".join(json.dumps(r) for r in heldout_rows)
    heldout_path.write_text(body + "\n" if body else "")
    heldout_schema_path.write_text(json.dumps(schema))

    print(
        f"wrote held-out test split -> {heldout_path} ({len(heldout_rows)} test rows)\n"
        f"  schema -> {heldout_schema_path}"
    )


def emit_parity_fixture(
    path: Path,
    net: dict[str, Any],
    model: RunValueMLP,
    x_all: np.ndarray,
    case_idx: np.ndarray,
    n_cases: int,
) -> None:
    """Write the Rust<->torch round-trip parity fixture.

    CRITICAL convention (matches `RunValueNet::forward` in
    `crates/sts2-runai/src/eval/run_value_net.rs`): `ctx_dense` is the RAW
    densified context — `x_all[row]`, width `context_dim`, BEFORE the
    `(x - feat_mu) / feat_sd` standardization the net's `feat_mu`/`feat_sd`
    encode. The torch forward computing `win_prob`/`act_probs` below
    standardizes internally (mirroring what the Rust loader does), so the
    Rust test can feed the same raw `ctx_dense` into the loaded net and
    land on the identical standardize-then-forward path. Emitting
    already-standardized values here would make the Rust side
    double-standardize and silently break parity.

    `net["win_calib"]` (if non-null) is applied to `win_prob` here via
    `apply_win_calib` — the SAME calibration `RunValueNet::forward`/
    `WinCalib::apply` applies on the Rust side — so the emitted `win_prob`
    is the fully-calibrated value the Rust `win_prob` must reproduce
    within `1e-4`. `act_probs` is never calibrated (mirrors the Rust
    aux head, which is diagnostics-only).
    """
    n = min(n_cases, len(case_idx))
    picked = case_idx[:n]

    feat_mu = np.asarray(net["feat_mu"], dtype=np.float32)
    feat_sd = np.asarray(net["feat_sd"], dtype=np.float32)

    ctx_raw = x_all[picked].astype(np.float32)
    ctx_std = (ctx_raw - feat_mu) / feat_sd

    model.eval()
    with torch.no_grad():
        win_logit, act_logits = model(torch.from_numpy(ctx_std.astype(np.float32)))
        win_prob = apply_win_calib(torch.sigmoid(win_logit).numpy(), net["win_calib"])
        act_probs = torch.softmax(act_logits, dim=-1).numpy()

    cases = [
        {
            "ctx_dense": ctx_raw[i].tolist(),
            "win_prob": float(win_prob[i]),
            "act_probs": act_probs[i].tolist(),
        }
        for i in range(n)
    ]
    fixture = {"net": net, "cases": cases}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"wrote parity fixture ({n} cases) -> {path}")


if __name__ == "__main__":
    main()
