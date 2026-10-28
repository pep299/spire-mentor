"""Offline gate for PBRS shaping coefficients (GH #464).

Scores candidate `ShapingConfig`s on a recorded corpus in seconds, without
training anything. It exists because the obvious instruments are both wrong:
a `ppo.screen` arm costs an hour and cannot resolve this at all, and the
naive offline version cannot either — for a reason worth stating up front,
because it bounds what any answer to #464 can be worth.

## The shaping is inert against an ideal critic — at EVERY lambda

With gamma=1 and terminal shaping un-credited, `r_t = Phi(s_{t+1}) - Phi(s_t)`.
Take any critic of the form `V(s) = m(s) - Phi(s)` — the form the fitted
critic converges to, since `shape()` is linear in features it observes and
`base()` is a function of the floor. Then every TD residual is

    delta_t = r_t + V(s_{t+1}) - V(s_t)
            = (Phi_{t+1} - Phi_t) + (m_{t+1} - Phi_{t+1}) - (m_t - Phi_t)
            = m_{t+1} - m_t

The `Phi` terms cancel identically, so `delta_t` — and therefore `A_lam` for
*every* lambda, not just lambda=1 — carries no shaping coefficient at all.
`test_td_residuals_are_phi_invariant` asserts this to float tolerance.

This is stronger than the usual policy-invariance statement (Ng, Harada &
Russell 1999), which only fixes the optimal policy. It says the gradient
estimator itself is untouched. **So shaping in this trainer can only act
through critic misfit**: the difference between the `V` the optimizer
actually has and the `m - Phi` it is converging to. Nothing else is left.

## What this gate therefore measures

Not "which coefficients are best" in the abstract — that question is empty
at zero misfit. It measures **how much shaping can move the gradient at a
given critic quality**, so you can decide whether the knob is worth any
further effort at all:

1. Fit `m_hat(s) = E[base(T) | s]` once, k-fold by run. Config-independent,
   so the reference advantage `A_ref = base(T) - m_hat(s)` — an unbiased
   sample of the config-independent true advantage `A_pi` — is shared by
   every candidate and no candidate can win by being easier to fit.
2. Per candidate, fit a *realistic* critic: ridge on the shaped return
   `G_c`, k-fold by run, at penalty `alpha`. This is the critic that has to
   spend capacity cancelling `Phi`, and it is where misfit enters.
3. Report `adv_corr = corr(A_lam^c, A_ref)` — how much true-advantage signal
   the estimator retains — and `ev_return`, which is directly comparable to
   the `explained_variance` the trainer already logs per iteration.

`alpha` is the critic-quality knob, and `--calibrate-ev` sets it from data:
sweep `alpha` until `ev_return` matches the campaign's logged
`explained_variance` (0.34-0.65 across the committed `*.stats.jsonl`), then
read the spread there. That is the number #464 turns on:

    spread ~ 0.00x  →  shaping cannot matter; standardize the knob and close it
    spread ~ 0.0x   →  worth one screen arm, read on iterations-to-fixed-WR

## What it cannot tell you

The fitted critic here is penalized-linear; the real one is a transformer
trained by SGD, whose misfit is dominated by optimizer lag rather than by a
ridge penalty. So the *magnitude* is an honest order-of-magnitude bound on
how much room shaping has, but the *ranking* among candidates is a statement
about this critic class. Do not read a winner off this table and ship it.
This gate never promotes anything; `docs/ppo-champion-training-playbook.md`
Section 2 is unchanged by it.

Usage (from sts2-pycli), recording the corpus with the CURRENT champion so
the exchange rate reflects the policy being trained, not `HeuristicRunAgent`:

    PYTHONPATH=python uv run --frozen python -m ppo.probe \
        --policy net --ckpt data/ppo/<the .pt named in champions.toml> \
        --character Silent --seeds 256 --sims 100 --boss-sims 300 \
        --record --out /tmp/silent-corpus.jsonl

    PYTHONPATH=python uv run --frozen python -m ppo.shaping_gate \
        --corpus /tmp/silent-corpus.jsonl --calibrate-ev 0.65 \
        --json /tmp/silent-gate.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .rewards import ShapingConfig, base_value, shape_value, shaped_rewards

# Floor one-hot width. `base()` saturates at TOTAL_FLOORS=45 and the deepest
# committed campaign mean is 43.4, so 56 bins covers every observed floor with
# headroom; anything past it is clipped into the last bin rather than dropped.
N_FLOOR_BINS = 56

# Continuous baseline features, in matrix column order. `hp_frac`,
# `num_upgraded` and `num_relics` are load-bearing rather than incidental:
# `shape()` is linear in exactly these three, so this feature basis CAN
# represent `-Phi` exactly and the only thing stopping it is `alpha`. That is
# deliberate — it makes `alpha` the single knob standing for critic quality.
_CONT_SCALE: tuple[tuple[str, float], ...] = (
    ("hp_frac", 1.0),
    ("num_upgraded", 1.0),
    ("num_relics", 1.0),
    ("gold", 0.01),
    ("deck_size", 0.1),
    ("num_potions", 1.0),
    ("act_index", 1.0),
)

# Log grid for `--calibrate-ev`. Spans "critic is essentially the regression
# optimum" to "critic barely beats predicting the mean".
_ALPHA_GRID = tuple(float(10.0**e) for e in np.arange(-4.0, 6.01, 0.5))


@dataclass(frozen=True)
class Run:
    """One recorded episode reduced to what the reward needs."""

    metrics: list[dict[str, Any]]  # per-decision snapshot, pre-action
    final: dict[str, Any]
    won: bool


def load_runs(path: str, limit: int = 0) -> list[Run]:
    """Read a `ppo.probe --record --out` corpus (same format `ppo.shaping_fit` eats)."""
    runs: list[Run] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("outcome") in ("error", "timeout"):
                continue
            decisions = row.get("decisions") or []
            if not decisions:
                continue
            # Backfill the ascension `rewards.total_floors` needs, for corpora
            # recorded before `metrics_snapshot` carried it (GH #464).
            asc = int(row.get("ascension") or 0)
            runs.append(
                Run(
                    metrics=[{"ascension": asc, **d["metrics"]} for d in decisions],
                    final={"ascension": asc, **row["final"]},
                    won=bool(row.get("won")),
                )
            )
            if limit and len(runs) >= limit:
                break
    return runs


def features(metrics: dict[str, Any]) -> np.ndarray:
    """One state → baseline feature row (floor one-hot ++ scaled continuous)."""
    row = np.zeros(N_FLOOR_BINS + len(_CONT_SCALE), dtype=np.float64)
    floor = int(metrics.get("total_floor") or 0)
    row[min(max(floor, 0), N_FLOOR_BINS - 1)] = 1.0
    hp_frac = float(metrics.get("hp") or 0) / max(1, int(metrics.get("max_hp") or 1))
    for i, (key, scale) in enumerate(_CONT_SCALE):
        val = hp_frac if key == "hp_frac" else float(metrics.get(key) or 0)
        row[N_FLOOR_BINS + i] = val * scale
    return row


def design_matrix(runs: list[Run]) -> tuple[np.ndarray, np.ndarray]:
    """→ (X [n_states, d], run_id [n_states])."""
    xs: list[np.ndarray] = []
    rid: list[int] = []
    for i, run in enumerate(runs):
        for m in run.metrics:
            xs.append(features(m))
            rid.append(i)
    if not xs:
        return np.zeros((0, N_FLOOR_BINS + len(_CONT_SCALE))), np.zeros(0, dtype=np.int64)
    return np.array(xs), np.array(rid, dtype=np.int64)


def terminal_targets(runs: list[Run]) -> np.ndarray:
    """`base(T)` of the owning run, per state — the config-independent target."""
    cfg = ShapingConfig()  # only progress_weight / win_bonus are read
    out: list[float] = []
    for run in runs:
        target = base_value(run.final, run.won, cfg)
        out.extend([target] * len(run.metrics))
    return np.array(out)


def shaped_returns(runs: list[Run], cfg: ShapingConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    """→ (return-to-go per state [n_states], per-run reward arrays)."""
    rets: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    for run in runs:
        r = np.array(shaped_rewards(run.metrics, run.final, run.won, cfg)[0])
        rewards.append(r)
        rets.append(np.cumsum(r[::-1])[::-1])
    if not rets:
        return np.zeros(0), []
    return np.concatenate(rets), rewards


def phi(run: Run, cfg: ShapingConfig) -> np.ndarray:
    """Potential at each decision state (non-terminal, so shaping is credited)."""
    return np.array([base_value(m, False, cfg) + shape_value(m, cfg) for m in run.metrics])


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Ridge with an UNPENALIZED intercept (centre, solve, recover) → (beta, intercept)."""
    if X.shape[0] == 0:
        return np.zeros(X.shape[1]), 0.0
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())
    Xc = X - x_mean
    gram = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    beta = np.linalg.solve(gram, Xc.T @ (y - y_mean))
    return beta, y_mean - float(x_mean @ beta)


def kfold_predict(
    X: np.ndarray, y: np.ndarray, run_id: np.ndarray, n_runs: int, folds: int, alpha: float
) -> np.ndarray:
    """Out-of-sample ridge prediction for every state, split BY RUN.

    Splitting by state would leak: every state in a run shares that run's
    terminal outcome, so a within-run split lets the fit read the answer off a
    sibling state and reports an explained variance training never sees.
    """
    pred = np.zeros(X.shape[0], dtype=np.float64)
    folds = max(2, min(folds, n_runs)) if n_runs >= 2 else 1
    if folds == 1:  # degenerate corpus (tests); in-sample is the only option
        beta, b0 = ridge_fit(X, y, alpha)
        return X @ beta + b0
    fold_of_run = np.arange(n_runs) % folds
    for f in range(folds):
        held = fold_of_run[run_id] == f
        if held.all() or not held.any():
            continue
        beta, b0 = ridge_fit(X[~held], y[~held], alpha)
        pred[held] = X[held] @ beta + b0
    return pred


def gae(rewards: np.ndarray, values: np.ndarray, lam: float, gamma: float = 1.0) -> np.ndarray:
    """GAE(gamma, lam) over one episode with a zero terminal bootstrap.

    Mirrors `trainer.gae_from_row` exactly; kept separate so the gate never
    needs a `PPOConfig`, a checkpoint, or torch.
    """
    n = len(rewards)
    out = np.zeros(n, dtype=np.float64)
    adv = 0.0
    next_v = 0.0
    for t in range(n - 1, -1, -1):
        delta = rewards[t] + gamma * next_v - values[t]
        adv = delta + gamma * lam * adv
        out[t] = adv
        next_v = values[t]
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    sa, sb = float(a.std()), float(b.std())
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _ev(y: np.ndarray, pred: np.ndarray) -> float:
    var = float(y.var())
    return 1.0 - float(((y - pred) ** 2).mean()) / var if var > 1e-12 else float("nan")


def oracle_values(runs: list[Run], cfg: ShapingConfig, m_hat: np.ndarray) -> np.ndarray:
    """`V = m_hat - Phi`, the ideal critic. Every candidate collapses onto it."""
    parts, pos = [], 0
    for run in runs:
        n = len(run.metrics)
        parts.append(m_hat[pos : pos + n] - phi(run, cfg))
        pos += n
    return np.concatenate(parts) if parts else np.zeros(0)


def evaluate(
    runs: list[Run],
    cfg: ShapingConfig,
    a_ref: np.ndarray,
    X: np.ndarray,
    run_id: np.ndarray,
    *,
    folds: int,
    alpha: float,
    lam: float,
    gamma: float = 1.0,
    m_hat: np.ndarray | None = None,
) -> dict[str, float]:
    """Score one candidate against the shared, config-independent `a_ref`.

    `m_hat` set → oracle critic (`V = m_hat - Phi`, misfit zero by
    construction, so every candidate must score identically). Otherwise the
    realistic one: ridge on this candidate's own shaped return at `alpha`.
    """
    rets, rewards = shaped_returns(runs, cfg)
    if not rewards:
        return {}
    values = (
        oracle_values(runs, cfg, m_hat)
        if m_hat is not None
        else kfold_predict(X, rets, run_id, len(runs), folds, alpha)
    )

    a_lam: list[np.ndarray] = []
    pos = 0
    for run, r in zip(runs, rewards):
        n = len(run.metrics)
        a_lam.append(gae(r, values[pos : pos + n], lam, gamma))
        pos += n
    lam_v = np.concatenate(a_lam)
    ref_std = float(a_ref.std())
    return {
        "adv_corr": _corr(lam_v, a_ref),
        "adv_std_ratio": float(lam_v.std() / ref_std) if ref_std > 1e-12 else float("nan"),
        "ev_return": _ev(rets, values),
        "n_states": float(lam_v.size),
    }


def fit_candidate(runs: list[Run]) -> ShapingConfig:
    """Floor-controlled OLS refit on this corpus (the `ppo.shaping_fit` estimator)."""
    from .shaping_fit import floor_controlled_ols

    xs: list[list[float]] = []
    ys: list[float] = []
    floors: list[int] = []
    cfg = ShapingConfig()
    for run in runs:
        target = base_value(run.final, run.won, cfg)
        for m in run.metrics:
            hp_frac = float(m.get("hp") or 0) / max(1, int(m.get("max_hp") or 1))
            xs.append([hp_frac, float(m.get("num_upgraded") or 0), float(m.get("num_relics") or 0)])
            ys.append(target)
            floors.append(int(m.get("total_floor") or 0))
    X, y, fl = np.array(xs), np.array(ys), np.array(floors)
    hp_coef, upg_coef, relic_coef = (float(b) for b in floor_controlled_ols(X, y, fl))
    means = X.mean(axis=0)
    return ShapingConfig(
        hp_coef=hp_coef,
        upg_coef=upg_coef,
        relic_coef=relic_coef,
        offset=hp_coef * means[0] + upg_coef * means[1] + relic_coef * means[2],
    )


def parse_candidate(spec: str) -> tuple[str, ShapingConfig]:
    """`name=hp,upg,relic,offset` → (name, config)."""
    name, sep, rest = spec.partition("=")
    parts = [p.strip() for p in rest.split(",")]
    if not name or not sep or len(parts) != 4:
        raise ValueError(f"--config wants 'name=hp,upg,relic,offset', got {spec!r}")
    hp, upg, relic, offset = (float(p) for p in parts)
    return name, ShapingConfig(hp_coef=hp, upg_coef=upg, relic_coef=relic, offset=offset)


def build_candidates(
    runs: list[Run], specs: list[str], with_fit: bool
) -> list[tuple[str, ShapingConfig]]:
    """Built-ins ++ `--config` overrides, in report order."""
    out: list[tuple[str, ShapingConfig]] = [
        # The trainer defaults every campaign but Necrobinder shipped on.
        ("priors", ShapingConfig()),
        # No shaping at all — the reference the others have to beat.
        ("none", ShapingConfig(hp_coef=0.0, upg_coef=0.0, relic_coef=0.0, offset=0.0)),
        # The only refit ever shipped (Necrobinder asc_it300).
        (
            "necro_fit",
            ShapingConfig(hp_coef=0.0351, upg_coef=0.0015, relic_coef=0.0, offset=0.0314),
        ),
    ]
    if with_fit:
        out.append(("corpus_fit", fit_candidate(runs)))
    out.extend(parse_candidate(s) for s in specs)
    return out


def calibrate_alpha(
    runs: list[Run],
    cfg: ShapingConfig,
    X: np.ndarray,
    run_id: np.ndarray,
    folds: int,
    target_ev: float,
) -> tuple[float, float, float]:
    """Pick the ridge penalty whose OOS `ev_return` matches a logged critic EV.

    → (alpha, |gap|, achieved_ev). The campaign's own `explained_variance`
    (0.34-0.65 across the committed `*.stats.jsonl`) is the only measurement
    we have of how good the real critic is, so it is what sets the operating
    point. A large gap is not a failure — see the caller: it always misses
    low, which makes the reported spread a ceiling rather than an estimate.
    """
    rets, _ = shaped_returns(runs, cfg)
    best, best_gap, best_ev = _ALPHA_GRID[0], float("inf"), float("nan")
    for alpha in _ALPHA_GRID:
        ev = _ev(rets, kfold_predict(X, rets, run_id, len(runs), folds, alpha))
        gap = abs(ev - target_ev)
        if gap < best_gap:
            best, best_gap, best_ev = alpha, gap, ev
    return best, best_gap, best_ev


def _print_table(rows: list[dict[str, Any]], title: str) -> None:
    """`ev_return` is per-candidate: each row explains ITS OWN shaped return,
    whose variance the coefficients change. Only the row the trainer actually
    ran is comparable to the logged `explained_variance`; the column is a
    diagnostic across rows, not a ranking."""
    print(f"\n{title}\n")
    print(
        f"{'candidate':<12} {'hp':>8} {'upg':>8} {'relic':>7} {'offset':>8} "
        f"{'adv_corr':>9} {'std_ratio':>10} {'ev_return':>10}"
    )
    for r in sorted(rows, key=lambda r: -(r.get("adv_corr") or float("-inf"))):
        print(
            f"{r['name']:<12} {r['hp_coef']:>8.4f} {r['upg_coef']:>8.4f} "
            f"{r['relic_coef']:>7.4f} {r['offset']:>8.4f} "
            f"{r['adv_corr']:>9.4f} {r['adv_std_ratio']:>10.4f} {r['ev_return']:>10.4f}"
        )


def _spread(rows: list[dict[str, Any]], key: str = "adv_corr") -> float:
    vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return (max(vals) - min(vals)) if vals else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, help="ppo.probe --record --out JSONL")
    ap.add_argument("--limit", type=int, default=0, help="cap runs read (0 = all)")
    ap.add_argument("--lam", type=float, default=0.97, help="GAE lambda under test")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5, help="k-fold splits, BY RUN")
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge penalty = critic quality")
    ap.add_argument(
        "--calibrate-ev",
        type=float,
        default=None,
        metavar="EV",
        help="pick --alpha so ev_return matches this logged explained_variance",
    )
    ap.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="NAME=HP,UPG,RELIC,OFFSET",
        help="extra candidate (repeatable)",
    )
    ap.add_argument("--no-fit", action="store_true", help="skip the corpus refit candidate")
    ap.add_argument("--json", type=str, default=None, help="write the tables as JSON")
    args = ap.parse_args(argv)

    runs = load_runs(args.corpus, args.limit)
    if not runs:
        print(f"no usable runs in {args.corpus}")
        return 1
    n_states = sum(len(r.metrics) for r in runs)
    wins = sum(1 for r in runs if r.won)
    print(f"corpus: {len(runs)} runs / {n_states} states / win rate {wins / len(runs):.3f}")

    X, run_id = design_matrix(runs)
    y_term = terminal_targets(runs)
    m_hat = kfold_predict(X, y_term, run_id, len(runs), args.folds, args.alpha)
    a_ref = y_term - m_hat
    print(
        f"m_hat: {X.shape[1]} features, {args.folds}-fold by run, "
        f"OOS R^2 on base(T) = {_ev(y_term, m_hat):.3f}"
    )

    alpha, calib_gap, calib_ev = args.alpha, None, None
    if args.calibrate_ev is not None:
        # Calibrated on `priors`, the config every campaign but Necrobinder
        # actually trained under, so the target EV and the measured EV are
        # quoting the same world.
        alpha, calib_gap, calib_ev = calibrate_alpha(
            runs, ShapingConfig(), X, run_id, args.folds, args.calibrate_ev
        )
        print(
            f"calibrated alpha={alpha:g} → ev_return {calib_ev:.3f} "
            f"vs target {args.calibrate_ev} (gap {calib_gap:.3f})"
        )
        if calib_gap > 0.05:
            # Only ever misses LOW: a linear basis on a small corpus cannot
            # reach a transformer's EV. A worse critic has more misfit, and
            # misfit is the only channel shaping has — so the fitted spread
            # below is an UPPER BOUND on the real effect, not an estimate.
            print(
                f"  WARNING: this basis tops out below the target — the fitted critic "
                f"({calib_ev:.3f}) is WORSE than the campaign's ({args.calibrate_ev}), so "
                f"the spread below overstates the real effect. Read it as a ceiling.\n"
                f"  A bigger corpus raises the ceiling; 48 runs will not reach 0.65."
            )

    candidates = build_candidates(runs, args.config, not args.no_fit)

    def score(m: np.ndarray | None) -> list[dict[str, Any]]:
        """`m` set → the ideal critic; `None` → the fitted one."""
        return [
            {
                "name": n,
                **evaluate(
                    runs,
                    c,
                    a_ref,
                    X,
                    run_id,
                    folds=int(args.folds),
                    alpha=float(alpha),
                    lam=float(args.lam),
                    gamma=float(args.gamma),
                    m_hat=m,
                ),
                **vars(c),
            }
            for n, c in candidates
        ]

    oracle = score(m_hat)
    fitted = score(None)

    _print_table(oracle, f"ideal critic (V = m_hat - Phi), lambda={args.lam} — MUST be flat")
    o_spread = _spread(oracle)
    ok = o_spread < 1e-9
    print(
        f"\nadv_corr spread {o_spread:.2e} "
        f"[{'OK — matches the theorem' if ok else 'BROKEN — the tables below are void'}]"
    )

    _print_table(fitted, f"fitted critic (ridge alpha={alpha:g}), lambda={args.lam}")
    f_spread = _spread(fitted)
    print(f"\nadv_corr spread {f_spread:.2e} — this is the number #464 turns on")
    print(
        "  < 1e-3  shaping cannot move the gradient at this critic quality; "
        "standardize the knob\n"
        "  > 1e-2  worth ONE screen arm, read on iterations-to-fixed-win-rate"
    )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "corpus": args.corpus,
                    "n_runs": len(runs),
                    "n_states": n_states,
                    "win_rate": wins / len(runs),
                    "lam": args.lam,
                    "gamma": args.gamma,
                    "alpha": alpha,
                    "calibrate_ev": args.calibrate_ev,
                    "calibrated_ev": calib_ev,
                    "calibrated_gap": calib_gap,
                    "m_hat_r2": _ev(y_term, m_hat),
                    "oracle": {"candidates": oracle, "adv_corr_spread": o_spread},
                    "fitted": {"candidates": fitted, "adv_corr_spread": f_spread},
                },
                fh,
                indent=2,
            )
        print(f"wrote {args.json}")
    return 0 if ok else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
