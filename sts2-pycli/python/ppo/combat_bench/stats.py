"""Estimators for the combat benchmark report (GH #830).

Pure numpy, no game state: every function takes plain arrays, so the tests can
check each one against a hand-computed value.

Why these estimators:

- Fights in one run share a deck lineage and are correlated, so the SE that
  treats fights as independent understates the error. `cr1` clusters by run.
- The pro set has only ~12 players, and a player's skill moves all of their
  fights together. A cluster-robust SE with 12 clusters is biased low and its
  normal CI under-covers. `wild_cluster_ci` is the wild cluster bootstrap with
  Webb's six-point weights, which MacKinnon & Webb (2017, 2018) show stays
  close to nominal with few, unbalanced clusters. The CI inverts the
  RESTRICTED (null-imposed) bootstrap test, their best-performing variant.
- Arm-vs-arm deltas are paired on the same fights and seeds, so
  `sign_flip_test` is exact under the null "the two arms are exchangeable on
  every fight", with no distributional assumption.
- Per-encounter gaps are noisy (a few fights each). `eb_shrink` pulls each
  toward its room mean by how noisy it is (normal-normal empirical Bayes).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

Z95 = 1.959963984540054
# Webb (2014) six-point weights: +-sqrt(1/2), +-1, +-sqrt(3/2), each p = 1/6.
# Mean 0, variance 1, fourth moment 7/6. With 12 clusters Rademacher weights
# only give 2^12 = 4096 distinct draws, half of them sign mirrors; six points
# give 6^12 and a smoother bootstrap distribution.
WEBB = np.array(
    [-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)], dtype=float
)


@dataclass(frozen=True)
class Estimate:
    """A mean with a standard error.

    Attributes:
        mean: Sample mean (NaN when there is no data).
        se: Standard error of the mean (NaN when it cannot be estimated).
        n: Number of observations.
        g: Number of clusters (equal to `n` for the plain SE).
    """

    mean: float
    se: float
    n: int
    g: int

    def ci(self, z: float = Z95) -> tuple[float, float]:
        """Normal-approximation CI `mean +- z * se`."""
        return self.mean - z * self.se, self.mean + z * self.se


@dataclass(frozen=True)
class WildCI:
    """Result of `wild_cluster_ci`.

    Attributes:
        mean: Sample mean (fight-weighted).
        lo: Lower CI bound (may be -inf when the data cannot bound it).
        hi: Upper CI bound (may be +inf).
        p0: Restricted wild bootstrap p-value for H0: mean == 0.
        g: Number of clusters.
        draws: Number of bootstrap weight vectors used.
        exact: True when all 6^g weight vectors were enumerated.
    """

    mean: float
    lo: float
    hi: float
    p0: float
    g: int
    draws: int
    exact: bool


@dataclass(frozen=True)
class PermResult:
    """Result of `sign_flip_test`.

    Attributes:
        p: Two-sided p-value.
        exact: True when every sign vector was enumerated.
        n_perm: Number of sign vectors evaluated.
    """

    p: float
    exact: bool
    n_perm: int


@dataclass(frozen=True)
class EBResult:
    """Result of `eb_shrink`.

    Attributes:
        mu: Prior mean the estimates are shrunk toward.
        tau2: Between-group variance (DerSimonian-Laird moment estimate, >= 0).
        shrunk: Posterior means, one per input estimate.
        post_sd: Posterior SDs (ignoring the uncertainty in `mu` and `tau2`).
        weight: Weight kept on each raw estimate, `tau2 / (tau2 + var)`.
    """

    mu: float
    tau2: float
    shrunk: list[float]
    post_sd: list[float]
    weight: list[float]


def _codes(clusters: Sequence[Hashable] | np.ndarray) -> tuple[np.ndarray, int]:
    """Integer codes 0..g-1 for arbitrary cluster labels."""
    labels = np.asarray([str(c) for c in clusters])
    _, inv = np.unique(labels, return_inverse=True)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def mean_se(x: Sequence[float] | np.ndarray) -> Estimate:
    """Mean and plain (iid) standard error.

    Args:
        x: Observations.

    Returns:
        The estimate; `se` is NaN for fewer than two observations.
    """
    v = np.asarray(x, dtype=float)
    n = len(v)
    if n == 0:
        return Estimate(math.nan, math.nan, 0, 0)
    se = float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return Estimate(float(v.mean()), se, n, n)


def cr1(x: Sequence[float] | np.ndarray, clusters: Sequence[Hashable] | np.ndarray) -> Estimate:
    """Mean and CR1 cluster-robust standard error.

    For the intercept-only regression the sandwich is
    `V = c * sum_g (sum_{i in g} (x_i - xbar))^2 / n^2` with the Stata CR1
    small-sample factor `c = g/(g-1) * (n-1)/(n-k)`, and `k = 1` makes the
    second factor 1.

    Args:
        x: Observations.
        clusters: One cluster label per observation.

    Returns:
        The estimate; `se` is NaN with fewer than two clusters.
    """
    v = np.asarray(x, dtype=float)
    n = len(v)
    if n == 0:
        return Estimate(math.nan, math.nan, 0, 0)
    codes, g = _codes(clusters)
    m = float(v.mean())
    if g < 2:
        return Estimate(m, math.nan, n, g)
    s = np.bincount(codes, weights=v - m, minlength=g)
    se = math.sqrt(g / (g - 1) * float(np.sum(s * s))) / n
    return Estimate(m, se, n, g)


def webb_weights(g: int, draws: int, seed: int) -> tuple[np.ndarray, bool]:
    """Webb weight matrix, one row per bootstrap draw and one column per cluster.

    When `6**g <= draws` every weight vector is enumerated once, so the
    bootstrap is exact and does not depend on the seed.

    Args:
        g: Number of clusters.
        draws: Number of random draws when not enumerating.
        seed: RNG seed.

    Returns:
        `(W, exact)` with `W` of shape `(rows, g)`.
    """
    if 6**g <= draws:
        idx = np.indices((6,) * g).reshape(g, -1).T
        return WEBB[idx], True
    rng = np.random.default_rng(seed)
    return WEBB[rng.integers(0, 6, size=(draws, g))], False


class _WildMoments:
    """Bootstrap sums for the intercept-only restricted wild cluster bootstrap.

    Under H0: mean == mu0, with `d = xbar - mu0`, the restricted residual sum of
    cluster g is `R_g = S_g + n_g d`, where `S_g` is the unrestricted residual
    sum. A bootstrap draw with weights `w` gives `xbar* - mu0 = w.R / n` and
    cluster residual sums `w_g R_g - n_g (xbar* - mu0)`. Everything is linear or
    quadratic in `d`, so the seven weight products below are computed once
    (in row chunks, so memory stays small with thousands of clusters) and every
    `p(mu0)` the CI inversion asks for is O(draws).
    """

    def __init__(
        self, s: np.ndarray, ng: np.ndarray, draws: int, seed: int, chunk: int = 512
    ) -> None:
        g = len(s)
        exact = 6**g <= draws
        self.exact = exact
        if exact:
            w, _ = webb_weights(g, draws, seed)
            blocks = [w]
        else:
            rng = np.random.default_rng(seed)
            blocks = (
                WEBB[rng.integers(0, 6, size=(min(chunk, draws - i), g))]
                for i in range(0, draws, chunk)
            )
        sn = s * ng
        # Two matrix products per chunk instead of seven matrix-vector ones:
        # w @ [s, n, s*n, n^2] and w^2 @ [s^2, s*n, n^2].
        v1 = np.column_stack([s, ng, sn, ng * ng])
        v2 = np.column_stack([s * s, sn, ng * ng])
        m1, m2 = [], []
        for w in blocks:
            m1.append(w @ v1)
            m2.append((w * w) @ v2)
        a, b = np.concatenate(m1), np.concatenate(m2)
        self.ws, self.wn, self.wsn, self.wn2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        self.w2s2, self.w2sn, self.w2n2 = b[:, 0], b[:, 1], b[:, 2]
        self.draws = len(self.ws)
        self.sum_n2 = float(np.sum(ng * ng))

    def t_star(self, d: float, n: int, c: float) -> np.ndarray:
        """Bootstrap t statistics for the null `xbar - mu0 == d`."""
        num = (self.ws + d * self.wn) / n
        ss = (
            self.w2s2
            + 2 * d * self.w2sn
            + d * d * self.w2n2
            - 2 * num * (self.wsn + d * self.wn2)
            + num * num * self.sum_n2
        )
        se = np.sqrt(np.maximum(c * ss, 0.0)) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            return num / se


def wild_cluster_ci(
    x: Sequence[float] | np.ndarray,
    clusters: Sequence[Hashable] | np.ndarray,
    *,
    draws: int = 9999,
    seed: int = 0,
    alpha: float = 0.05,
) -> WildCI:
    """Wild cluster bootstrap CI for a mean, by restricted-test inversion.

    For each candidate `mu0` the restricted wild cluster bootstrap (WCR, Webb
    weights, CR1 bootstrap-t) gives a symmetric two-sided p-value; the CI is the
    set of `mu0` with `p > alpha`, found by bisection on each side of the mean.
    The same weight draws serve every `mu0`, so `p(mu0)` is a fixed step
    function and the result is reproducible for a given seed.

    Args:
        x: Observations (fight-level values).
        clusters: One cluster label per observation (the player).
        draws: Bootstrap draws (all 6^g weight vectors are used instead when
            that is fewer).
        seed: RNG seed for the weight draws.
        alpha: One minus the coverage.

    Returns:
        The CI; bounds are NaN with fewer than two clusters and may be
        infinite when the clusters cannot bound the mean.
    """
    v = np.asarray(x, dtype=float)
    n = len(v)
    if n == 0:
        return WildCI(math.nan, math.nan, math.nan, math.nan, 0, 0, False)
    codes, g = _codes(clusters)
    m = float(v.mean())
    if g < 2:
        return WildCI(m, math.nan, math.nan, math.nan, g, 0, False)
    ng = np.bincount(codes, minlength=g).astype(float)
    s = np.bincount(codes, weights=v - m, minlength=g)
    c = g / (g - 1)
    se = math.sqrt(c * float(np.sum(s * s))) / n
    if se <= 0.0:
        # Every observation equals the mean: the CI is that point, and the
        # only mean the data cannot reject is the observed one.
        return WildCI(m, m, m, 0.0 if m else 1.0, g, 0, False)
    mom = _WildMoments(s, ng, draws, seed)

    def pval(mu0: float) -> float:
        d = m - mu0
        t_obs = abs(d / se)
        t = np.abs(mom.t_star(d, n, c))
        return float(np.mean(np.nan_to_num(t, nan=0.0) >= t_obs * (1 - 1e-12)))

    def bound(direction: int) -> float:
        inside = m
        k = 1.0
        outside = m + direction * k * se
        while pval(outside) > alpha:
            inside = outside
            k *= 2.0
            if k > 2.0**30:
                return direction * math.inf
            outside = m + direction * k * se
        tol = 1e-6 * se
        for _ in range(200):
            if abs(outside - inside) <= tol:
                break
            mid = 0.5 * (inside + outside)
            if pval(mid) > alpha:
                inside = mid
            else:
                outside = mid
        return 0.5 * (inside + outside)

    return WildCI(m, bound(-1), bound(+1), pval(0.0), g, mom.draws, mom.exact)


def sign_flip_test(
    d: Sequence[float] | np.ndarray, *, n_perm: int = 20000, seed: int = 0
) -> PermResult:
    """Two-sided paired sign-flip permutation test of `mean(d) == 0`.

    Under H0 each fight's delta is as likely to be `+d_f` as `-d_f`. The
    statistic is `|sum d|`. Every sign vector is enumerated when `2^n <= n_perm`
    (exact p); otherwise `n_perm` random vectors are drawn and
    `p = (1 + #extreme) / (1 + n_perm)` (Phipson & Smyth 2010), which never
    returns 0.

    Args:
        d: Per-fight paired deltas.
        n_perm: Random sign vectors when not enumerating.
        seed: RNG seed.

    Returns:
        The p-value and how it was computed.
    """
    v = np.asarray(d, dtype=float)
    n = len(v)
    if n == 0:
        return PermResult(math.nan, False, 0)
    t_obs = abs(float(v.sum()))
    tol = 1e-9 * max(1.0, float(np.abs(v).sum()))
    if 2**n <= n_perm:
        bits = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
        t = np.abs((bits * 2 - 1) @ v)
        return PermResult(float(np.mean(t >= t_obs - tol)), True, 2**n)
    rng = np.random.default_rng(seed)
    count = 0
    chunk = max(1, min(n_perm, 2_000_000 // max(n, 1)))
    done = 0
    while done < n_perm:
        k = min(chunk, n_perm - done)
        signs = rng.integers(0, 2, size=(k, n), dtype=np.int8) * 2 - 1
        count += int(np.sum(np.abs(signs @ v) >= t_obs - tol))
        done += k
    return PermResult((1 + count) / (1 + n_perm), False, n_perm)


def holm(p: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control).

    NaN p-values stay NaN and do not count toward the family size.

    Args:
        p: Raw p-values.

    Returns:
        Adjusted p-values in the input order.
    """
    idx = [i for i, x in enumerate(p) if not math.isnan(x)]
    m = len(idx)
    out = [math.nan] * len(p)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: p[i])):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def eb_shrink(est: Sequence[float], var: Sequence[float], mu: float | None = None) -> EBResult:
    """Normal-normal empirical Bayes shrinkage of group estimates.

    Model: `est_e ~ N(theta_e, var_e)`, `theta_e ~ N(mu, tau2)`. `tau2` is the
    DerSimonian-Laird moment estimate (truncated at 0) and each estimate keeps
    weight `tau2 / (tau2 + var_e)`; the rest goes to `mu`.

    Args:
        est: Raw group estimates.
        var: Their sampling variances (all > 0).
        mu: Prior mean; defaults to the precision-weighted mean of `est`.

    Returns:
        The shrunk estimates.
    """
    e = np.asarray(est, dtype=float)
    v = np.asarray(var, dtype=float)
    k = len(e)
    if k == 0:
        return EBResult(math.nan, math.nan, [], [], [])
    w = 1.0 / v
    mu_fe = float(np.sum(w * e) / np.sum(w))
    tau2 = 0.0
    if k > 1:
        q = float(np.sum(w * (e - mu_fe) ** 2))
        denom = float(np.sum(w) - np.sum(w * w) / np.sum(w))
        tau2 = max(0.0, (q - (k - 1)) / denom) if denom > 0 else 0.0
    prior = mu_fe if mu is None else float(mu)
    b = tau2 / (tau2 + v)
    shrunk = prior + b * (e - prior)
    return EBResult(
        prior,
        tau2,
        [float(x) for x in shrunk],
        [float(x) for x in np.sqrt(b * v)],
        [float(x) for x in b],
    )
