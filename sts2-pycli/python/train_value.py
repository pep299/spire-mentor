"""M4 value head: ``V(run_state) → graded final-act-reached``.

Full-run-AI plan §5/M4 (bead ``sts2-rs-40ikw.8``). The plan's V trains on
*campaign outcomes*; pure win/loss is unusable while the agent wins ~0, but
the **act reached** is a graded, non-myopic outcome that varies richly even at
0 wins (a run that dies in act 1 had lower-value mid-states than one that
reaches act 3). This is the deck-*scaling* signal the myopic ``fight_delta``
labels can't express — V learns to value a run-state by how far the run
eventually gets under the current policy (the Monte-Carlo return of the single
self-play trajectory).

Input: a campaign's ``features.jsonl`` (per-decision deck / relic / scalar
context) + ``decisions.jsonl`` (per-decision ``act``; the per-seed max is the
run's final act). Target per decision = its seed's final act ∈ {1, 2, 3}.
Pure numpy (no torch), split by seed, early stop on val. Reports held-out
MAE + Spearman + per-final-act mean prediction (the discrimination that
matters for ranking states).

NOTE: the ``run_arena ranker-schema`` / ``export-ranker-features`` commands
that produced these inputs were removed along with the learned pick-ranker
(PPO replaced it); this trainer only works on previously exported campaign
data.

    python train_value.py --features features.jsonl --decisions decisions.jsonl \
        --schema schema.json --out value_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_schema(path: Path) -> dict:
    schema = json.loads(path.read_text(encoding="utf-8"))
    for key in ("schema_id", "n_card", "n_relic", "n_scalar", "context_dim"):
        if key not in schema:
            raise SystemExit(f"schema missing {key!r}")
    return schema


def densify(context: dict, schema: dict) -> np.ndarray:
    """Mirror the Rust exporter's ``ContextFeatures::write_dense`` layout."""
    n_card, n_relic = schema["n_card"], schema["n_relic"]
    x = np.zeros(schema["context_dim"], dtype=np.float32)
    for idx, count, upgraded in context["deck"]:
        i = int(idx)
        if i < n_card:
            x[2 * i] = count
            x[2 * i + 1] = upgraded
    base = 2 * n_card
    for idx, count in context["relics"]:
        i = int(idx)
        if i < n_relic:
            x[base + i] = count
    sbase = base + n_relic
    scalars = np.asarray(context["scalars"], dtype=np.float32)
    x[sbase : sbase + schema["n_scalar"]] = scalars[: schema["n_scalar"]]
    return x


def per_seed_final_act(decisions_path: Path) -> dict[int, int]:
    """Run's final act = max ``act`` over that seed's decision records."""
    final: dict[int, int] = {}
    for line in decisions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        seed = int(r["seed"])
        final[seed] = max(final.get(seed, 0), int(r.get("act", 1)))
    return final


def load_xy(
    features_path: Path, schema: dict, final_act: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, seeds, steps = [], [], [], []
    for line in features_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        seed = int(r["seed"])
        if seed not in final_act:
            continue
        xs.append(densify(r["context"], schema))
        ys.append(float(final_act[seed]))
        seeds.append(seed)
        steps.append(int(r.get("step", 0)))
    if not xs:
        raise SystemExit("no (feature, final-act) pairs — seed mismatch?")
    return (
        np.stack(xs),
        np.asarray(ys, dtype=np.float64),
        np.asarray(seeds),
        np.asarray(steps, dtype=np.int64),
    )


def split_by_seed(
    seeds: np.ndarray, val_pct: int, test_pct: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign whole seeds to train/val/test so no run straddles splits."""
    uniq = np.unique(seeds)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, n * test_pct // 100)
    n_val = max(1, n * val_pct // 100)
    test_seeds = set(uniq[:n_test].tolist())
    val_seeds = set(uniq[n_test : n_test + n_val].tolist())
    tr = np.array([s not in test_seeds and s not in val_seeds for s in seeds])
    va = np.array([s in val_seeds for s in seeds])
    te = np.array([s in test_seeds for s in seeds])
    return tr, va, te


class ValueNet:
    """One hidden ReLU layer → scalar value. Standardized inputs/target."""

    def __init__(self, dim: int, hidden: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2.0 / dim), (dim, hidden)).astype(np.float64)
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, 1)).astype(np.float64)
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = x @ self.w1 + self.b1
        h = np.maximum(z, 0.0)
        return (h @ self.w2 + self.b2).ravel(), h

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC = P(score(win) > score(loss)) via the Mann-Whitney U identity
    on mid-ranks (ties handled)."""
    labels = np.asarray(labels)
    npos = int(labels.sum())
    nneg = int(len(labels) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rsum_pos = ranks[labels == 1].sum()
    return float((rsum_pos - npos * (npos + 1) / 2.0) / (npos * nneg))


def per_seed_win(decisions_path: Path) -> dict[int, int]:
    """Run won iff any decision record for that seed has win==1."""
    win: dict[int, int] = {}
    for line in decisions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        seed = int(r["seed"])
        win[seed] = max(win.get(seed, 0), int(r.get("win", 0)))
    return win


def build_next_index(seeds: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Map each row to its successor row within the same run (ordered by
    ``step``); -1 marks a terminal row (the run's deepest state). Rows of one
    run never straddle seeds, so the successor is purely a (seed, step) lookup —
    no game-state reconstruction needed."""
    order: dict[int, list[tuple[int, int]]] = {}
    for i, (s, st) in enumerate(zip(seeds.tolist(), steps.tolist())):
        order.setdefault(int(s), []).append((int(st), i))
    nxt = np.full(len(seeds), -1, dtype=np.int64)
    for lst in order.values():
        lst.sort()
        for k in range(len(lst) - 1):
            nxt[lst[k][1]] = lst[k + 1][1]
    return nxt


def fit(
    xn: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[ValueNet, float, float]:
    """Standardize targets on the train split, fit the one-hidden-layer net with
    early stopping on val, restore the best weights. Returns the net plus the
    (y_mu, y_sd) used so callers can denormalize predictions to act units."""
    ymu, ysd = y[tr].mean(), y[tr].std() + 1e-12
    yn = (y - ymu) / ysd
    net = ValueNet(xn.shape[1], args.hidden, args.init_seed)
    idx_tr = np.where(tr)[0]
    best_val = np.inf
    best = None
    since = 0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(idx_tr)
        for s in range(0, len(idx_tr), args.batch):
            bi = idx_tr[s : s + args.batch]
            xb, yb = xn[bi], yn[bi]
            pred, h = net.forward(xb)
            err = pred - yb
            n = len(bi)
            gw2 = h.T @ err.reshape(-1, 1) / n + args.l2 * net.w2
            gb2 = err.mean(keepdims=True)
            dh = (err.reshape(-1, 1) @ net.w2.T) * (h > 0)
            gw1 = xb.T @ dh / n + args.l2 * net.w1
            gb1 = dh.mean(0)
            net.w2 -= args.lr * gw2
            net.b2 -= args.lr * gb2
            net.w1 -= args.lr * gw1
            net.b1 -= args.lr * gb1
        vpred = net.predict(xn[va])
        vloss = float(((vpred - yn[va]) ** 2).mean())
        if vloss < best_val - 1e-5:
            best_val = vloss
            best = (net.w1.copy(), net.b1.copy(), net.w2.copy(), net.b2.copy())
            since = 0
        else:
            since += 1
            if since >= args.patience:
                print(f"early stop at epoch {epoch} (best val {best_val:.4f})", file=sys.stderr)
                break
    if best is not None:
        net.w1, net.b1, net.w2, net.b2 = best
    return net, float(ymu), float(ysd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", default="value_weights.json")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-pct", type=int, default=15)
    ap.add_argument("--test-pct", type=int, default=15)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument(
        "--td-bootstrap",
        action="store_true",
        help="offline TD(0) fitted-value-iteration: relabel non-terminal states "
        "with gamma*V(next) and refit (reduces early-floor label variance)",
    )
    ap.add_argument("--td-gamma", type=float, default=0.95)
    ap.add_argument("--td-iters", type=int, default=2)
    args = ap.parse_args()

    schema = load_schema(Path(args.schema))
    final_act = per_seed_final_act(Path(args.decisions))
    dist: dict[int, int] = {}
    for a in final_act.values():
        dist[a] = dist.get(a, 0) + 1
    print(f"per-seed final-act distribution: {dict(sorted(dist.items()))}", file=sys.stderr)

    x, y, seeds, steps = load_xy(Path(args.features), schema, final_act)
    rng = np.random.default_rng(args.init_seed)
    tr, va, te = split_by_seed(seeds, args.val_pct, args.test_pct, rng)
    print(
        f"split by seed: train={tr.sum()} val={va.sum()} test={te.sum()} decisions "
        f"({len(np.unique(seeds))} seeds)",
        file=sys.stderr,
    )

    # Standardize ONLY the dense scalar sub-block (last n_scalar dims).
    # The deck/relic one-hot/count block is already 0/small and sparse —
    # standardizing it divides near-constant features by ~0 std and
    # explodes the inputs (the removed train_ranker.py did the same,
    # normalizing only the scalar block). Target standardized for a
    # well-conditioned MSE.
    mu = np.zeros(x.shape[1], dtype=np.float64)
    sd = np.ones(x.shape[1], dtype=np.float64)
    sbase = 2 * schema["n_card"] + schema["n_relic"]
    mu[sbase:] = x[tr, sbase:].mean(0)
    sd[sbase:] = x[tr, sbase:].std(0) + 1e-6
    xn = (x - mu) / sd

    net, ymu, ysd = fit(xn, y, tr, va, args, rng)

    # TD(0) bootstrapping (offline fitted-value-iteration): relabel each
    # non-terminal state s_t with gamma * V(s_{t+1}) — no per-step reward exists,
    # so r_t = 0 mid-run — while pinning the terminal state at its true MC
    # outcome, then refit. This shrinks the early-floor label variance that pure
    # MC-outcome replication suffers (a floor-1 state otherwise inherits the
    # whole run's noisy final result as its target).
    if args.td_bootstrap:
        nxt = build_next_index(seeds, steps)
        safe = np.where(nxt < 0, 0, nxt)
        n_term = int((nxt < 0).sum())
        print(
            f"TD: {len(y) - n_term}/{len(y)} rows bootstrapped ({n_term} terminal), "
            f"gamma={args.td_gamma}, iters={args.td_iters}",
            file=sys.stderr,
        )
        for it in range(1, args.td_iters + 1):
            v = net.predict(xn) * ysd + ymu
            y_td = np.where(nxt < 0, y, args.td_gamma * v[safe])
            net, ymu, ysd = fit(xn, y_td, tr, va, args, rng)
            rho_it = spearman(net.predict(xn[te]) * ysd + ymu, y[te])
            print(
                f"TD iter {it}: test Spearman(V, true final-act) {rho_it:.3f}",
                file=sys.stderr,
            )

    # Held-out report (in original act units).
    tpred = net.predict(xn[te]) * ysd + ymu
    ttrue = y[te]
    mae = float(np.abs(tpred - ttrue).mean())
    rho = spearman(tpred, ttrue)
    print(f"TEST: MAE {mae:.3f} act-units, Spearman {rho:.3f} (n={te.sum()})", file=sys.stderr)
    for a in sorted(set(ttrue.astype(int))):
        m = ttrue.astype(int) == a
        print(f"  final-act {a}: mean V {tpred[m].mean():.3f} (n={int(m.sum())})", file=sys.stderr)

    # Win/loss discrimination (binary): does V rank winning runs above losing
    # ones? AUC is threshold-free; raw accuracy is near-useless under the heavy
    # class imbalance (always-predict-loss already scores 1-baserate), so we
    # also report base rate, accuracy at the base-rate quantile threshold, and
    # the lift in the top-decile of V. Per-act AUC shows how predictability
    # grows toward the end of the run (early floors are intrinsically low-signal).
    win_by_seed = per_seed_win(Path(args.decisions))
    win_arr = np.array([win_by_seed.get(int(s), 0) for s in seeds], dtype=np.int64)
    win_te = win_arr[te]
    base = float(win_te.mean())
    overall_auc = auc(tpred, win_te)
    # accuracy at the threshold that predicts the top `base` fraction as wins
    thr = np.quantile(tpred, 1.0 - base) if 0.0 < base < 1.0 else np.inf
    acc = float(((tpred >= thr).astype(int) == win_te).mean())
    maj = float(max(base, 1.0 - base))
    # top-decile lift
    d_thr = np.quantile(tpred, 0.9)
    top = win_te[tpred >= d_thr]
    top_rate = float(top.mean()) if len(top) else float("nan")
    print(
        f"WIN/LOSS: base-rate {base:.3f} | AUC {overall_auc:.3f} | "
        f"acc@q {acc:.3f} (majority {maj:.3f}) | top-decile win-rate {top_rate:.3f} "
        f"({top_rate / base:.1f}x base)",
        file=sys.stderr,
    )
    sbase_act = 2 * schema["n_card"] + schema["n_relic"] + 5  # scalar idx 5 = act_index
    act_te = x[te, sbase_act].astype(int)
    for a in sorted(set(act_te.tolist())):
        m = act_te == a
        print(
            f"  act-index {a}: win-AUC {auc(tpred[m], win_te[m]):.3f} "
            f"(n={int(m.sum())}, win-rate {win_te[m].mean():.3f})",
            file=sys.stderr,
        )

    Path(args.out).write_text(
        json.dumps(
            {
                "schema_id": schema["schema_id"],
                "kind": "value_v1_final_act",
                "context_dim": int(x.shape[1]),
                "hidden": args.hidden,
                "feat_mu": mu.tolist(),
                "feat_sd": sd.tolist(),
                "y_mu": float(ymu),
                "y_sd": float(ysd),
                "w1": net.w1.tolist(),
                "b1": net.b1.tolist(),
                "w2": net.w2.ravel().tolist(),
                "b2": float(net.b2[0]),
            }
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
