"""G-D n_eq: how many playouts a value net is worth, pre-registered (GH #830).

Plan Step 2 G-D, "Metric fix (pre-registered)". Raw-MSE n_eq is INVALID: with
true centred sibling deltas d and playout SD s it gives a constant V an n_eq of
s^2 / E[d^2] (10-100 on contested states), so it is only reported, as a
demonstration (`raw_mse_neq_const`). The two metrics that decide:

* PRIMARY, regret n_eq of the estimator as the search seam would use it:
  `M_lam = lam * V + (1 - lam) * (one playout)`, `lam` in `LAMBDAS` picked on
  probe-dev. Regret of a group = `max R - R(argmax M)` (exact expectation over
  random tie-breaks). The reference curve is the mean of k playouts, k = 1..64,
  drawn as disjoint chunks of the estimator pool; k = 0 is a uniformly random
  pick. n_eq is where the curve crosses M's regret (log-linear in k).
* CO-PRIMARY, calibrated n_eq of V: on centred deltas (member minus group
  mean, groups weighted equally), the explained share
  `R2 = 1 - (MSE(dR - beta*dx) - noise) / (E[dR^2] - noise)` with `beta`
  cross-fitted on probe-dev and the reference's own noise subtracted
  (`noise` from the per-seed centred playouts, valid for independent and CRN
  seeds alike). For the optimal beta it is the noise-corrected corr^2. A
  constant scores 0. n_eq is where the k-playout curve of R2 (each k with its
  own dev-fitted beta) reaches V's.

Sanity (required before reading V; `sanity_ok`): a constant V scores
calibrated n_eq < 0.5 and regret n_eq ~ 1 (the mix then ranks by its playout);
a fresh single playout scores ~ 1 on calibrated n_eq and on its own regret n_eq
(`lam = 1`; mixed with the pool playout it is a 2-playout mean, so its
`M_lam` n_eq is ~ 2, reported too). TexelWide (the deployed depth-cut leaf,
and its atanh) is scored like V. If the regret curve moves by less than 3
bootstrap SE between k=1 and k=64 the regret metric is `uninformative` for that
stratum and the calibrated metric alone decides. CIs: fight-cluster bootstrap.
At most two looks at probe-test (`probe_test_looks.jsonl` beside the test
references).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 0.9)
KMAX = 64
SANITY_BAND = (0.67, 1.5)
MAX_TEST_LOOKS = 2
STRATA = ("contested_boss", "contested_elite", "decided_win", "afterstate")


@dataclass
class RefSet:
    """One stratum's reference groups, padded to `M` members.

    Attributes:
        R: `[G, M]` reference means (0 at padding).
        pool: `[G, M, P]` estimator-pool playouts, disjoint from `R`'s.
        fresh: `[G, M, F]` fresh playouts for the sanity checks.
        noise: `[G, M]` variance of each member's centred reference delta.
        mask: `[G, M]` real members.
        cluster: `[G]` fight-cluster ids (bootstrap unit).
        crn: `[G]` whether the group's members share seeds.
        n_r: Playouts behind each `R`.
        refvar: `[G, M]` per-playout variance of the reference playouts.
        crn_pairs: `(var of paired differences, sum of member variances)` summed
            over member pairs, from every reference column.
    """

    R: np.ndarray
    pool: np.ndarray
    fresh: np.ndarray
    noise: np.ndarray
    mask: np.ndarray
    cluster: np.ndarray
    crn: np.ndarray
    n_r: int
    refvar: np.ndarray
    crn_pairs: tuple[float, float]


def build_refset(
    groups: list[np.ndarray],
    n_r: int,
    n_pool: int,
    n_fresh: int,
    clusters: list[str],
    crn: list[bool],
) -> RefSet:
    """Pad `groups` (`[members, n_r + n_pool + n_fresh]` playout values each)."""
    g_n = len(groups)
    m = max((len(x) for x in groups), default=1)
    R = np.zeros((g_n, m))
    pool = np.zeros((g_n, m, n_pool))
    fresh = np.zeros((g_n, m, n_fresh))
    noise = np.zeros((g_n, m))
    refvar = np.zeros((g_n, m))
    mask = np.zeros((g_n, m), bool)
    pair_diff = pair_sum = 0.0
    for gi, x in enumerate(groups):
        x = np.asarray(x, np.float64)
        n = len(x)
        ref = x[:, :n_r]
        R[gi, :n] = ref.mean(1)
        pool[gi, :n] = x[:, n_r : n_r + n_pool]
        fresh[gi, :n] = x[:, n_r + n_pool : n_r + n_pool + n_fresh]
        z = ref - ref.mean(0, keepdims=True)
        noise[gi, :n] = z.var(1, ddof=1) / n_r
        refvar[gi, :n] = ref.var(1, ddof=1)
        mask[gi, :n] = True
        allv = x[:, : n_r + n_pool]
        var = allv.var(1, ddof=1)
        for a in range(n):
            for b in range(a + 1, n):
                pair_diff += float((allv[a] - allv[b]).var(ddof=1))
                pair_sum += float(var[a] + var[b])
    ids = {c: i for i, c in enumerate(sorted(set(clusters)))}
    return RefSet(
        R=R,
        pool=pool,
        fresh=fresh,
        noise=noise,
        mask=mask,
        cluster=np.array([ids[c] for c in clusters], np.int64),
        crn=np.array(crn, bool),
        n_r=n_r,
        refvar=refvar,
        crn_pairs=(pair_diff, pair_sum),
    )


def subset(rs: RefSet, idx: np.ndarray) -> RefSet:
    """The groups `idx` of `rs` (cluster ids kept)."""
    return RefSet(
        R=rs.R[idx],
        pool=rs.pool[idx],
        fresh=rs.fresh[idx],
        noise=rs.noise[idx],
        mask=rs.mask[idx],
        cluster=rs.cluster[idx],
        crn=rs.crn[idx],
        n_r=rs.n_r,
        refvar=rs.refvar[idx],
        crn_pairs=rs.crn_pairs,
    )


# ── regret ───────────────────────────────────────────────────────────────


def group_regret(R: np.ndarray, mask: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Per-group regret of picking `argmax S` (ties split evenly).

    Args:
        R: `[G, M]` reference values.
        mask: `[G, M]` real members.
        S: `[G, M]` or `[G, M, B]` scores (B draws).

    Returns:
        `[G]` or `[G, B]`.
    """
    Rm = np.where(mask, R, -np.inf)
    rmax = Rm.max(1)
    if S.ndim == 3:
        mk = mask[:, :, None]
        Rx = R[:, :, None]
        rmax = rmax[:, None]
    else:
        mk = mask
        Rx = R
    Sm = np.where(mk, S, -np.inf)
    top = Sm.max(1, keepdims=True)
    tie = (Sm >= top) & mk
    pick = (np.where(tie, Rx, 0.0)).sum(1) / tie.sum(1)
    return rmax - pick


def random_regret(rs: RefSet) -> np.ndarray:
    """`[G]` regret of a uniformly random pick (the k = 0 point)."""
    n = rs.mask.sum(1)
    mean = np.where(rs.mask, rs.R, 0.0).sum(1) / n
    return np.where(rs.mask, rs.R, -np.inf).max(1) - mean


def chunk_means(pool: np.ndarray, k: int) -> np.ndarray:
    """`[G, M, P // k]` means of disjoint k-chunks of the pool (CRN columns aligned)."""
    n = pool.shape[2] // k
    return pool[:, :, : n * k].reshape(pool.shape[0], pool.shape[1], n, k).mean(-1)


def regret_curve(rs: RefSet, kmax: int = KMAX) -> np.ndarray:
    """`[G, kmax + 1]` per-group regret of the k-playout mean, k = 0..kmax."""
    out = np.zeros((rs.R.shape[0], kmax + 1))
    out[:, 0] = random_regret(rs)
    for k in range(1, kmax + 1):
        if rs.pool.shape[2] // k < 1:
            out[:, k] = np.nan
            continue
        out[:, k] = group_regret(rs.R, rs.mask, chunk_means(rs.pool, k)).mean(1)
    return out


def mix_regret(
    rs: RefSet, v: np.ndarray, lam: float, draws: np.ndarray | None = None
) -> np.ndarray:
    """`[G]` regret of `lam * v + (1 - lam) * playout_i`, averaged over the pool draws i.

    `draws` replaces the pool playouts (`[G, M, B]`), e.g. for a fresh estimator.
    """
    pool = rs.pool if draws is None else draws
    if lam >= 1.0:
        return group_regret(rs.R, rs.mask, v)
    S = lam * v[:, :, None] + (1.0 - lam) * pool
    return group_regret(rs.R, rs.mask, S).mean(1)


# ── calibrated ───────────────────────────────────────────────────────────


def centred(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Member minus group mean (over real members); 0 at padding. Works on `[G, M, ...]`."""
    mk = mask if x.ndim == 2 else mask.reshape(mask.shape + (1,) * (x.ndim - 2))
    n = mask.sum(1).reshape((-1,) + (1,) * (x.ndim - 2))
    mean = np.where(mk, x, 0.0).sum(1) / n
    return np.where(mk, x - np.expand_dims(mean, 1), 0.0)


def _w(rs: RefSet) -> np.ndarray:
    return rs.mask / rs.mask.sum(1, keepdims=True)


def cal_parts(rs: RefSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(dR, S_dd, S_noise)`: centred references and per-group weighted sums."""
    dR = centred(rs.R, rs.mask)
    w = _w(rs)
    return dR, (w * dR**2).sum(1), (w * rs.noise).sum(1)


def fit_beta(rs: RefSet, dx: np.ndarray) -> float:
    """Least-squares slope of `dR` on `dx` (`[G, M]` or `[G, M, B]`), groups equal."""
    dR, _, _ = cal_parts(rs)
    w = _w(rs)
    if dx.ndim == 3:
        w = w[:, :, None]
        dR = dR[:, :, None]
    num = float((w * dx * dR).sum())
    den = float((w * dx * dx).sum())
    return num / den if den > 1e-18 else 0.0


def residual_sums(rs: RefSet, dx: np.ndarray, beta: float) -> np.ndarray:
    """`[G]` weighted `sum (dR - beta * dx)^2` (mean over draws for `[G, M, B]`)."""
    dR, _, _ = cal_parts(rs)
    w = _w(rs)
    if dx.ndim == 3:
        return (w[:, :, None] * (dR[:, :, None] - beta * dx) ** 2).sum(1).mean(1)
    return (w * (dR - beta * dx) ** 2).sum(1)


def r2_from_sums(
    s_res: np.ndarray, s_dd: np.ndarray, s_noise: np.ndarray, wts: np.ndarray | None = None
) -> float:
    """Noise-corrected explained share from per-group sums (optionally weighted)."""
    if wts is None:
        wts = np.ones_like(s_dd)
    num = float((wts * (s_res - s_noise)).sum())
    den = float((wts * (s_dd - s_noise)).sum())
    return 1.0 - num / den if den > 1e-18 else float("nan")


def corr2(rs: RefSet, dx: np.ndarray) -> float:
    """Noise-corrected squared correlation of `dx` with the true centred deltas."""
    dR, s_dd, s_noise = cal_parts(rs)
    w = _w(rs)
    sxy = float((w * dx * dR).sum())
    sxx = float((w * dx * dx).sum())
    syy = float((s_dd - s_noise).sum())
    return sxy * sxy / (sxx * syy) if sxx > 1e-18 and syy > 1e-18 else 0.0


def k_dx(rs: RefSet, k: int) -> np.ndarray:
    return centred(chunk_means(rs.pool, k), rs.mask)


def k_betas(rs_dev: RefSet, kmax: int = KMAX) -> np.ndarray:
    """`[kmax + 1]` dev-fitted slope per k (k = 0 unused)."""
    out = np.zeros(kmax + 1)
    for k in range(1, kmax + 1):
        if rs_dev.pool.shape[2] // k >= 1:
            out[k] = fit_beta(rs_dev, k_dx(rs_dev, k))
    return out


def cal_curve_sums(rs: RefSet, betas: np.ndarray, kmax: int = KMAX) -> np.ndarray:
    """`[G, kmax + 1]` residual sums of the k-playout estimator (k = 0: predict 0)."""
    _, s_dd, _ = cal_parts(rs)
    out = np.zeros((rs.R.shape[0], kmax + 1))
    out[:, 0] = s_dd
    for k in range(1, kmax + 1):
        if rs.pool.shape[2] // k < 1:
            out[:, k] = np.nan
            continue
        out[:, k] = residual_sums(rs, k_dx(rs, k), float(betas[k]))
    return out


# ── n_eq ─────────────────────────────────────────────────────────────────


def neq_from_curve(ys: np.ndarray, target: float, increasing: bool) -> tuple[float, bool]:
    """Where the k = 0..K curve `ys` reaches `target`; `(n_eq, censored)`.

    The curve is first made monotone (running max, or running min for a
    decreasing one). Between k = 0 and 1 the interpolation is linear in k,
    above 1 linear in log k. A target beyond the curve's end is censored at K.
    """
    ys = np.asarray(ys, np.float64)
    ks = np.arange(len(ys), dtype=np.float64)
    ok = ~np.isnan(ys)
    ks, ys = ks[ok], ys[ok]
    if not increasing:
        ys, target = -ys, -target
    ys = np.maximum.accumulate(ys)
    if not math.isfinite(target):
        return float("nan"), False
    if target <= ys[0]:
        return 0.0, False
    if target > ys[-1]:
        return float(ks[-1]), True
    i = int(np.argmax(ys >= target))
    ka, kb, ya, yb = ks[i - 1], ks[i], ys[i - 1], ys[i]
    frac = (target - ya) / (yb - ya) if yb > ya else 1.0
    if ka < 1.0:
        return float(ka + frac * (kb - ka)), False
    return float(math.exp(math.log(ka) + frac * (math.log(kb) - math.log(ka)))), False


def raw_mse_neq_const(rs: RefSet) -> float:
    """The INVALID raw-MSE n_eq of a constant V on centred deltas: sigma^2 / E[d^2].

    Reported only to show why it was ruled out (plan G-D, metric fix).
    """
    _, s_dd, s_noise = cal_parts(rs)
    w = _w(rs)
    sig2 = float((w * rs.refvar).sum())
    d2 = float((s_dd - s_noise).sum())
    return sig2 / d2 if d2 > 1e-18 else float("inf")


# ── per-stratum evaluation ───────────────────────────────────────────────


@dataclass
class Estimator:
    """What is scored: `v` on test, `v_dev` on dev, plus how to mix it."""

    name: str
    v: np.ndarray
    v_dev: np.ndarray
    fresh_index: int | None = None  # a fresh-playout pseudo-estimator


def choose_lambda(rs_dev: RefSet, v_dev: np.ndarray) -> tuple[float, dict[str, float]]:
    """The `LAMBDAS` entry with the lowest dev regret of the mix."""
    regs = {str(lam): float(mix_regret(rs_dev, v_dev, lam).mean()) for lam in LAMBDAS}
    best = min(LAMBDAS, key=lambda lam: (regs[str(lam)], lam))
    return float(best), regs


def bootstrap_weights(cluster: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """`[n_boot, G]` group weights from resampling fight clusters with replacement."""
    rng = np.random.default_rng(seed)
    n_c = int(cluster.max()) + 1 if len(cluster) else 0
    counts = (
        rng.multinomial(n_c, np.full(n_c, 1.0 / n_c), size=n_boot) if n_c else np.zeros((n_boot, 0))
    )
    return counts[:, cluster].astype(np.float64)


def _ci(xs: np.ndarray) -> list[float]:
    xs = np.asarray(xs, np.float64)
    xs = xs[np.isfinite(xs)]
    if not len(xs):
        return [float("nan"), float("nan")]
    return [float(np.percentile(xs, 5)), float(np.percentile(xs, 95))]


def score_estimator(
    rs: RefSet,
    rs_dev: RefSet,
    v: np.ndarray,
    v_dev: np.ndarray,
    reg_curve: np.ndarray,
    cal_curve: np.ndarray,
    s_dd: np.ndarray,
    s_noise: np.ndarray,
    boots: np.ndarray,
    lam: float | None = None,
    beta: float | None = None,
    draws: np.ndarray | None = None,
) -> dict[str, Any]:
    """Regret and calibrated n_eq of one estimator, with bootstrap CIs.

    `lam` / `beta` default to the dev-chosen / dev-fitted values.
    """
    lam_regs: dict[str, float] = {}
    if lam is None:
        lam, lam_regs = choose_lambda(rs_dev, v_dev)
    reg_m = mix_regret(rs, v, lam, draws)
    reg_v = mix_regret(rs, v, 1.0)
    dx = centred(v, rs.mask)
    if beta is None:
        beta = fit_beta(rs_dev, centred(v_dev, rs_dev.mask))
    res_v = residual_sums(rs, dx, beta)
    curve = reg_curve.mean(0)
    out: dict[str, Any] = {
        "lambda": lam,
        "lambda_dev_regret": lam_regs,
        "beta": beta,
        "regret_M": float(reg_m.mean()),
        "regret_V": float(reg_v.mean()),
        "corr2_V": corr2(rs, dx),
    }
    r2_curve = np.array(
        [r2_from_sums(cal_curve[:, k], s_dd, s_noise) for k in range(cal_curve.shape[1])]
    )
    out["R2_V"] = r2_from_sums(res_v, s_dd, s_noise)
    out["neq_regret_M"], out["neq_regret_M_censored"] = neq_from_curve(
        curve, out["regret_M"], False
    )
    out["neq_regret_V"], out["neq_regret_V_censored"] = neq_from_curve(
        curve, out["regret_V"], False
    )
    out["neq_cal"], out["neq_cal_censored"] = neq_from_curve(r2_curve, out["R2_V"], True)
    if len(boots):
        bm, bv, bc = [], [], []
        for w in boots:
            sw = w.sum()
            if sw <= 0:
                continue
            c = (w[:, None] * reg_curve).sum(0) / sw
            bm.append(neq_from_curve(c, float((w * reg_m).sum() / sw), False)[0])
            bv.append(neq_from_curve(c, float((w * reg_v).sum() / sw), False)[0])
            rc = np.array(
                [r2_from_sums(cal_curve[:, k], s_dd, s_noise, w) for k in range(cal_curve.shape[1])]
            )
            bc.append(neq_from_curve(rc, r2_from_sums(res_v, s_dd, s_noise, w), True)[0])
        out["neq_regret_M_ci90"] = _ci(np.array(bm))
        out["neq_regret_V_ci90"] = _ci(np.array(bv))
        out["neq_cal_ci90"] = _ci(np.array(bc))
    return out


def crn_ratio(rs: RefSet) -> dict[str, float]:
    """Variance of sibling differences under the seeds used vs independent seeds.

    `pairs`: summed over member pairs, Var(y_a - y_b) per paired seed over
    Var(y_a) + Var(y_b). `centred`: the reference noise of the centred deltas
    (as seeded) over what independent seeds would give.
    """
    pd, ps = rs.crn_pairs
    w = _w(rs)
    n = rs.mask.sum(1, keepdims=True)
    v = np.where(rs.mask, rs.refvar / rs.n_r, 0.0)
    indep = v * (1 - 2 / n) + v.sum(1, keepdims=True) / n**2
    return {
        "pairs": pd / ps if ps > 0 else float("nan"),
        "centred": float((w * rs.noise).sum() / (w * np.where(rs.mask, indep, 0.0)).sum()),
    }


def evaluate_stratum(
    rs: RefSet,
    rs_dev: RefSet,
    estimators: dict[str, tuple[np.ndarray, np.ndarray]],
    n_boot: int = 1000,
    seed: int = 830,
    kmax: int = KMAX,
) -> dict[str, Any]:
    """Every metric of one stratum: the k curves, each estimator, the sanity checks.

    Args:
        rs: The evaluated groups (probe-test, or probe-dev for selection).
        rs_dev: Probe-dev groups of the same stratum (lambda, beta).
        estimators: `{name: (v on rs, v on rs_dev)}`, `[G, M]` each.
    """
    kmax = min(kmax, rs.pool.shape[2], rs_dev.pool.shape[2])
    reg_curve = regret_curve(rs, kmax)
    betas = k_betas(rs_dev, kmax)
    cal_curve = cal_curve_sums(rs, betas, kmax)
    _, s_dd, s_noise = cal_parts(rs)
    boots = bootstrap_weights(rs.cluster, n_boot, seed)
    curve = reg_curve.mean(0)
    r2c = np.array([r2_from_sums(cal_curve[:, k], s_dd, s_noise) for k in range(kmax + 1)])
    deltas = []
    for w in boots:
        sw = w.sum()
        if sw > 0:
            deltas.append(float((w * (reg_curve[:, 1] - reg_curve[:, kmax])).sum() / sw))
    delta = float(curve[1] - curve[kmax])
    se = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
    out: dict[str, Any] = {
        "groups": int(rs.R.shape[0]),
        "members": int(rs.mask.sum()),
        "clusters": len(set(rs.cluster.tolist())),
        "crn_share": float(rs.crn.mean()) if len(rs.crn) else 0.0,
        "kmax": kmax,
        "regret_curve": {str(k): float(curve[k]) for k in range(kmax + 1)},
        "R2_curve": {str(k): float(r2c[k]) for k in range(kmax + 1)},
        "k_betas": {str(k): float(betas[k]) for k in (1, 2, 4, 8, 16, 32, 64) if k <= kmax},
        "regret_delta_1_kmax": delta,
        "regret_delta_se": se,
        "uninformative": bool(not (delta >= 3 * se)),
        "raw_mse_neq_const": raw_mse_neq_const(rs),
        "crn_variance_ratio": crn_ratio(rs),
        "estimators": {},
    }
    for name, (v, v_dev) in estimators.items():
        out["estimators"][name] = score_estimator(
            rs, rs_dev, v, v_dev, reg_curve, cal_curve, s_dd, s_noise, boots
        )
    # Sanity: a constant, and a fresh single playout (each fresh column in turn).
    zero, zero_dev = np.zeros_like(rs.R), np.zeros_like(rs_dev.R)
    const = score_estimator(
        rs, rs_dev, zero, zero_dev, reg_curve, cal_curve, s_dd, s_noise, boots[:0]
    )
    fr = []
    n_f = min(rs.fresh.shape[2], rs_dev.fresh.shape[2])
    for j in range(n_f):
        fr.append(
            score_estimator(
                rs, rs_dev, rs.fresh[:, :, j], rs_dev.fresh[:, :, j], reg_curve, cal_curve,
                s_dd, s_noise, boots[:0], lam=0.5,
            )
        )  # fmt: skip
    fresh = {
        key: float(np.mean([f[key] for f in fr])) if fr else float("nan")
        for key in ("neq_cal", "neq_regret_V", "neq_regret_M", "R2_V", "regret_V")
    }
    lo, hi = SANITY_BAND
    checks = {
        "const_cal_lt_0.5": const["neq_cal"] < 0.5,
        "const_regret_M_approx_1": lo <= const["neq_regret_M"] <= hi,
        "fresh_cal_approx_1": lo <= fresh["neq_cal"] <= hi,
        "fresh_regret_V_approx_1": lo <= fresh["neq_regret_V"] <= hi,
    }
    out["sanity"] = {
        "const": {
            k: const[k] for k in ("neq_cal", "neq_regret_M", "neq_regret_V", "lambda", "R2_V")
        },
        "fresh_playout_mean_over_columns": fresh,
        "fresh_columns": n_f,
        "checks": checks,
        "ok": bool(n_f and all(checks.values())),
    }
    return out


# ── verdict ──────────────────────────────────────────────────────────────


def verdict(report: dict[str, Any], bar: float, est: str = "V") -> dict[str, Any]:
    """Plan G-D bars on a test report: PASS / KILL / RETRY_BAND / INVALID."""
    strata = report["strata"]
    notes = []
    if not all(s["sanity"]["ok"] for s in strata.values()):
        return {"verdict": "INVALID", "notes": ["a sanity check failed; do not read V"]}
    per = {}
    kill = False
    passed = True
    for name in ("contested_boss", "contested_elite"):
        if name not in strata:
            return {"verdict": "INVALID", "notes": [f"no {name} groups"]}
        s = strata[name]
        e = s["estimators"][est]
        cal = e["neq_cal"]
        reg = e["neq_regret_M"]
        used = [cal] if s["uninformative"] else [cal, reg]
        per[name] = {"neq_cal": cal, "neq_regret_M": reg, "uninformative": s["uninformative"]}
        if min(used) < 2.0:
            kill = True
            notes.append(f"{name}: n_eq < 2")
        if min(used) < bar:
            passed = False
        tex = max(
            (
                s["estimators"][t]["neq_cal"]
                for t in ("texel", "texel_atanh")
                if t in s["estimators"]
            ),
            default=-1.0,
        )
        if cal <= tex:
            kill = True
            notes.append(f"{name}: V no better than TexelWide ({cal:.2f} <= {tex:.2f})")
    dec = strata.get("decided_win", {}).get("estimators", {}).get(est)
    lam_d = None
    if dec is not None:
        lam_d = 0.0 if min(dec["neq_cal"], dec["neq_regret_M"]) < 1.0 else dec["lambda"]
    v = "KILL" if kill else "PASS" if passed else "RETRY_BAND"
    return {"verdict": v, "bar": bar, "per_stratum": per, "lambda_d": lam_d, "notes": notes}


# ── IO ───────────────────────────────────────────────────────────────────


def load_refs(ref_dir: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(manifest, {stratum: {"groups": [...], "member": [[ids]], "fight", "crn", "room"}})`."""
    from ppo.deepbets import value_data as vd

    meta = json.load(open(os.path.join(ref_dir, "manifest.json"), encoding="utf-8"))
    out: dict[str, Any] = {}
    for p in vd.shard_paths(ref_dir, "refs-", ".npz"):
        # Read each array ONCE: `NpzFile[key]` decompresses the whole array on
        # every access, and a group's slice is a view that keeps that copy alive
        # (the full refs once took ~60 GB this way).
        with np.load(p) as zf:
            z = {k: zf[k] for k in zf.files}
        off = z["off"]
        for gi in range(len(z["group"])):
            st = str(z["stratum"][gi])
            d = out.setdefault(
                st, {"group": [], "members": [], "vals": [], "fight": [], "crn": [], "room": []}
            )
            d["group"].append(str(z["group"][gi]))
            d["members"].append([str(m) for m in z["member"][off[gi] : off[gi + 1]]])
            d["vals"].append(z["vals"][off[gi] : off[gi + 1]])
            d["fight"].append(str(z["fight"][gi]))
            d["crn"].append(bool(z["crn"][gi]))
            d["room"].append(str(z["room"][gi]))
    return meta, out


def refset_of(meta: dict[str, Any], d: dict[str, Any]) -> RefSet:
    n_r = int(meta["n_r"])
    n_pool = int(meta["n_ref"]) - n_r
    return build_refset(d["vals"], n_r, n_pool, int(meta["n_fresh"]), d["fight"], d["crn"])


def padded_values(d: dict[str, Any], value_of: dict[str, float]) -> np.ndarray:
    """`[G, M]` estimator values for each member id (0 at padding)."""
    m = max(len(x) for x in d["members"])
    out = np.zeros((len(d["members"]), m))
    for gi, ms in enumerate(d["members"]):
        for j, mid in enumerate(ms):
            out[gi, j] = value_of[mid]
    return out


def _looks(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_eval(args: argparse.Namespace) -> int:
    from ppo.deepbets import value_data as vd
    from ppo.deepbets import value_train as vt

    dmeta, dev = load_refs(args.refs_dev)
    tmeta, test = load_refs(args.refs_test) if args.refs_test else (None, None)
    ldir = args.labels
    lmeta = json.load(open(os.path.join(ldir, "manifest.json"), encoding="utf-8"))
    vd.check_same_net(lmeta["playout_net"], dmeta["playout_net"], "labels vs dev refs")
    if tmeta:
        vd.check_same_net(lmeta["playout_net"], tmeta["playout_net"], "labels vs test refs")
    nets = {}
    if args.net:
        net = vt.load_net(args.net)
        vd.check_same_net(net["meta"]["playout_net"], dmeta["playout_net"], "net vs refs")
        nets["V"] = net
    ids = sorted(
        {m for src in (dev, test or {}) for d in src.values() for ms in d["members"] for m in ms}
    )
    texel = vt.label_column(ldir, ids, "texel")
    value_of = {
        "texel": texel,
        "texel_atanh": {k: math.atanh(max(-0.999999, min(0.999999, x))) for k, x in texel.items()},
    }
    for name, net in nets.items():
        value_of[name] = vt.predict(net, ldir, ids)
    looks_path = os.path.join(args.refs_test, "probe_test_looks.jsonl") if args.refs_test else None
    if looks_path:
        prior = _looks(looks_path)
        if len(prior) >= MAX_TEST_LOOKS:
            raise SystemExit(
                f"{len(prior)} looks at probe-test already ({looks_path}); the plan allows {MAX_TEST_LOOKS}"
            )
    report: dict[str, Any] = {
        "schema": "deepbets.value_neq.v1",
        "evaluated": "test" if test else "dev",
        "net": os.path.abspath(args.net) if args.net else None,
        "net_sha256": vd.sha256_file(args.net) if args.net else None,
        "refs_dev": os.path.abspath(args.refs_dev),
        "refs_test": os.path.abspath(args.refs_test) if args.refs_test else None,
        "playout_net": dmeta["playout_net"],
        "strata": {},
    }
    t0 = time.time()
    for stratum in STRATA:
        if stratum not in dev or (test is not None and stratum not in test):
            continue
        rs_dev = refset_of(dmeta, dev[stratum])
        if test is not None:
            rs = refset_of(tmeta, test[stratum])
            src = test[stratum]
        else:
            # Dev-only (selection): score each half with lambda and beta from the other.
            rs, src = rs_dev, dev[stratum]
        ests = {}
        for name, vals in value_of.items():
            ests[name] = (padded_values(src, vals), padded_values(dev[stratum], vals))
        if test is not None:
            report["strata"][stratum] = evaluate_stratum(rs, rs_dev, ests, args.n_boot, args.seed)
        else:
            report["strata"][stratum] = evaluate_dev_crossfit(rs_dev, ests, args.n_boot, args.seed)
    report["secs"] = round(time.time() - t0, 1)
    if test is not None:
        report["verdict"] = verdict(report, args.bar)
        with open(looks_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "net_sha256": report["net_sha256"],
                        "report": os.path.abspath(args.out),
                    }
                )
                + "\n"
            )
        report["test_look"] = len(_looks(looks_path))
    vd.write_json(args.out, report)
    print(json.dumps(summarize(report), indent=1))
    return 0


def evaluate_dev_crossfit(
    rs_dev: RefSet, ests: dict[str, tuple[np.ndarray, np.ndarray]], n_boot: int, seed: int
) -> dict[str, Any]:
    """Dev-only scoring for model selection: 2-fold by fight cluster, lambda and
    beta from the other fold, n_eq averaged over the two folds."""
    fold = np.array(
        [int(hashlib.sha1(str(c).encode()).hexdigest(), 16) % 2 for c in rs_dev.cluster]
    )
    halves = []
    for f in (0, 1):
        a, b = np.where(fold == f)[0], np.where(fold != f)[0]
        if not len(a) or not len(b):
            return evaluate_stratum(
                rs_dev, rs_dev, {k: (v, v) for k, (v, _) in ests.items()}, n_boot, seed
            )
        sub = {k: (v[a], v[b]) for k, (_, v) in ests.items()}
        halves.append(evaluate_stratum(subset(rs_dev, a), subset(rs_dev, b), sub, n_boot, seed))
    out = halves[0]
    out["crossfit"] = True
    for name in out["estimators"]:
        for key in ("neq_cal", "neq_regret_M", "neq_regret_V", "R2_V", "regret_M"):
            out["estimators"][name][key] = float(
                np.mean([h["estimators"][name][key] for h in halves])
            )
    out["groups"] = int(rs_dev.R.shape[0])
    return out


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"evaluated": report["evaluated"], "verdict": report.get("verdict")}
    for st, s in report["strata"].items():
        row = {
            "groups": s["groups"],
            "uninformative": s["uninformative"],
            "sanity_ok": s["sanity"]["ok"],
            "crn_ratio": s["crn_variance_ratio"],
            "raw_mse_neq_const(INVALID)": s["raw_mse_neq_const"],
        }
        for name, e in s["estimators"].items():
            row[name] = {
                "neq_regret_M": round(e["neq_regret_M"], 2),
                "neq_regret_V": round(e["neq_regret_V"], 2),
                "neq_cal": round(e["neq_cal"], 2),
                "lambda": e["lambda"],
                "ci_cal": e.get("neq_cal_ci90"),
                "ci_regret_M": e.get("neq_regret_M_ci90"),
            }
        out[st] = row
    return out


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ppo.deepbets.value_neq")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval", help="n_eq of a value net (and TexelWide) on reference groups")
    p.add_argument(
        "--net", default=None, help="value net JSON (value_train); omit for baselines only"
    )
    p.add_argument("--labels", required=True)
    p.add_argument("--refs-dev", required=True)
    p.add_argument("--refs-test", default=None, help="probe-test refs: a LOOK (at most 2)")
    p.add_argument("--out", required=True)
    p.add_argument("--bar", type=float, default=4.0, help="PASS bar (8 per G0/G-A middle bands)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=830)
    p.set_defaults(fn=cmd_eval)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
