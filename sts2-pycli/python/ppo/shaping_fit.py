"""Floor-controlled OLS refit of PBRS shaping coefficients (plan §3.3).

The fork's hp/upgrade coefficients (0.1 / 0.035, offset 0.307) are STS1
priors. This script refits them on our own expert episode dumps: regress
the run's realized terminal base value on per-decision state features,
controlling for floor (Frisch–Waugh: demean target and features within
total_floor bins, then OLS on the residuals). At fixed floor, the
coefficient on hp_frac answers "how much terminal base value does one
unit of hp_frac predict?" — exactly the exchange rate Φ's shape term
should use.

Shaping is policy-invariant for ANY coefficients (telescoping, γ=1);
the refit only improves credit assignment, so run-to-run differences
here are about variance, not correctness.

Usage (from sts2-pycli):
    PYTHONPATH=python uv run --frozen python -m ppo.shaping_fit \
        --corpus /tmp/ppo-bc/expert.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from .rewards import ShapingConfig, base_value


def load_xy(path: str, limit: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (X [n,3: hp_frac, num_upgraded, num_relics], y [n], floor [n])."""
    cfg = ShapingConfig()
    xs: list[list[float]] = []
    ys: list[float] = []
    floors: list[int] = []
    n_runs = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            decisions = row.get("decisions") or []
            if not decisions or row.get("outcome") in ("error", "timeout"):
                continue
            # Ascension backfilled for corpora predating the `metrics_snapshot`
            # key `rewards.total_floors` reads (GH #464).
            y = base_value(
                {"ascension": int(row.get("ascension") or 0), **row["final"]},
                bool(row["won"]),
                cfg,
            )
            for d in decisions:
                m = d["metrics"]
                xs.append(
                    [
                        int(m.get("hp") or 0) / max(1, int(m.get("max_hp") or 1)),
                        float(m.get("num_upgraded") or 0),
                        float(m.get("num_relics") or 0),
                    ]
                )
                ys.append(y)
                floors.append(int(m.get("total_floor") or 0))
            n_runs += 1
            if limit and n_runs >= limit:
                break
    print(f"loaded {n_runs} runs / {len(xs)} decision states")
    return np.array(xs), np.array(ys), np.array(floors)


def floor_controlled_ols(X: np.ndarray, y: np.ndarray, floors: np.ndarray) -> np.ndarray:
    """Frisch–Waugh: demean X and y within each floor bin, OLS on residuals."""
    Xr = X.astype(np.float64).copy()
    yr = y.astype(np.float64).copy()
    for f in np.unique(floors):
        sel = floors == f
        Xr[sel] -= Xr[sel].mean(axis=0)
        yr[sel] -= yr[sel].mean()
    beta, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
    return beta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    X, y, floors = load_xy(args.corpus, args.limit)
    n_per_floor = collections.Counter(floors.tolist())
    print(f"floors covered: {len(n_per_floor)} (min bin {min(n_per_floor.values())})")

    beta = floor_controlled_ols(X, y, floors)
    hp_coef, upg_coef, relic_coef = (float(b) for b in beta)
    # Offset zero-centers the shape at the corpus-mean state (cosmetic for
    # the value scale only — Φ is potential-based, constants cancel).
    means = X.mean(axis=0)
    offset = hp_coef * means[0] + upg_coef * means[1]  # hp+upg only (plan)

    print("\nfloor-controlled OLS (terminal base ~ state | floor):")
    print(f"  hp_frac       {hp_coef:+.4f}   (fork prior +0.1000)")
    print(f"  num_upgraded  {upg_coef:+.4f}   (fork prior +0.0350)")
    print(f"  num_relics    {relic_coef:+.4f}   (fork prior none; plan: start hp+upg only)")
    print(f"  offset        {offset:.4f}   (fork prior 0.3070)")
    print(
        "\ntrainer flags: "
        f"--shaping-hp-coef {hp_coef:.4f} --shaping-upg-coef {upg_coef:.4f} "
        f"--shaping-relic-coef 0.0 --shaping-offset {offset:.4f}"
    )


if __name__ == "__main__":
    main()
