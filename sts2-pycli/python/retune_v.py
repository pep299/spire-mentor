"""Retune HeuristicV weights from rollout labels (full-run-AI M2).

Bead ``sts2-rs-40ikw.6``. Loads a ``labels.jsonl`` produced by
``run_arena label-campaign`` (each row carries the option's
post-candidate ``HeuristicV::feature_map`` — the exact linear features
the Rust scorer uses, exported by the labeler so Python never
re-implements ``card_stats``), fits a logistic regression with the
rollout-preferred option as the target on pairwise feature DIFFS, and
writes an updated ``heuristic_v.toml`` candidate.

Model: for each labeled decision, let ``b`` be the rollout-best option
(max ``rollout_score``). For every other option ``j`` emit a sample
``x = features(b) - features(j), y = 1`` plus the mirrored ``(-x, 0)``
(symmetric, keeps the intercept at 0 — rank-only fitting). Fit
``P(y=1) = sigmoid(w·x)`` by L2-regularized gradient descent (numpy
only, no sklearn). The learned ``w`` is scale-free; we rescale it so
its L2 norm matches the current weight vector's norm to keep V-score
magnitudes (and the [tiers] margin thresholds) comparable.

The TOML candidate is produced by section-aware line rewriting of the
existing weights file, so comments and non-refit fields ([shop],
[projection], [tiers], relic_values overrides, deck_size_target)
survive verbatim.

Run (end-to-end small campaign):

  python python/label_campaign.py run --seeds 4 --combat none \
      --probe-sims 20 --k 4 --t-turns 3 --out-labels labels.jsonl
  python python/retune_v.py labels.jsonl --out heuristic_v_candidate.toml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = (
    _HERE / ".." / ".." / "sts2-rs" / "data" / "runai" / "heuristic_v.toml"
).resolve()

# feature_map key → (TOML section or None, TOML key). Mirrors
# HeuristicV::weight_for_feature; ``_fixed_offset`` is not a weight.
FEATURE_TO_TOML: dict[str, tuple[str | None, str]] = {
    "hp_base": (None, "hp_base"),
    "hp_act": (None, "hp_act"),
    "low_hp": (None, "low_hp"),
    "gold_sqrt": (None, "gold_sqrt"),
    "frontload_per_card": (None, "frontload_per_card"),
    "block_per_card": (None, "block_per_card"),
    "aoe_count": (None, "aoe_count"),
    "scaling_count": (None, "scaling_count"),
    "upgrade_count": (None, "upgrade_count"),
    "draw_total": (None, "draw_total"),
    "energy_total": (None, "energy_total"),
    "curse_count": (None, "curse_count"),
    "deck_size_penalty": (None, "deck_size_penalty"),
    "heavy_cost_penalty": (None, "heavy_cost_penalty"),
    "relic_default": (None, "relic_default"),
    "boss.aoe_bonus": ("boss", "aoe_bonus"),
    "potions.held": ("potions", "held"),
    "potions.boss_insurance": ("potions", "boss_insurance"),
    "ascension.per_level": ("ascension", "per_level"),
}


def load_labels(path: Path) -> dict[tuple, list[dict]]:
    by_decision: dict[tuple, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ref = row["decision_ref"]
            by_decision[ref["seed"], ref["step"]].append(row)
    return {k: v for k, v in by_decision.items() if len(v) >= 2}


def feature_names(decisions: dict[tuple, list[dict]]) -> list[str]:
    names: set[str] = set()
    for opts in decisions.values():
        for row in opts:
            names.update(row["features"].keys())
    names.discard("_fixed_offset")
    return sorted(names)


def fvec(row: dict, names: list[str]) -> np.ndarray:
    f = row["features"]
    return np.array([f.get(n, 0.0) for n in names], dtype=np.float64)


def build_samples(
    decisions: dict[tuple, list[dict]], names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for opts in decisions.values():
        best = max(opts, key=lambda r: r["rollout_score"])
        # Skip decisions where the rollout can't separate the options.
        scores = {r["rollout_score"] for r in opts}
        if len(scores) < 2:
            continue
        fb = fvec(best, names)
        for row in opts:
            if row is best:
                continue
            x = fb - fvec(row, names)
            xs.append(x)
            ys.append(1.0)
            xs.append(-x)
            ys.append(0.0)
    if not xs:
        sys.exit("no separable labeled decisions — campaign too small or probes too weak")
    return np.vstack(xs), np.array(ys)


def fit_logistic(
    x: np.ndarray, y: np.ndarray, l2: float = 1e-2, lr: float = 0.05, iters: int = 4000
) -> np.ndarray:
    """Plain batch gradient descent on L2-regularized logistic loss."""
    # Column scaling for conditioning; undone at the end.
    scale = np.maximum(np.abs(x).max(axis=0), 1e-9)
    xs = x / scale
    w = np.zeros(x.shape[1])
    n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xs @ w)))
        grad = xs.T @ (p - y) / n + l2 * w
        w -= lr * grad
    return w / scale


def agreement(decisions: dict[tuple, list[dict]], names: list[str], w: np.ndarray) -> float:
    """Fraction of decisions where argmax(w·f) matches the rollout-best."""
    agree, total = 0, 0
    for opts in decisions.values():
        scores = {r["rollout_score"] for r in opts}
        if len(scores) < 2:
            continue
        best = max(opts, key=lambda r: r["rollout_score"])
        pred = max(opts, key=lambda r: float(w @ fvec(r, names)))
        total += 1
        if pred is best:
            agree += 1
    return agree / total if total else float("nan")


def read_current_weights(weights_path: Path, names: list[str]) -> np.ndarray:
    """Parse the scalar weights out of the TOML (stdlib tomllib)."""
    import tomllib

    with open(weights_path, "rb") as f:
        data = tomllib.load(f)
    out = []
    for name in names:
        section, key = FEATURE_TO_TOML.get(name, (None, name))
        node = data.get(section, {}) if section else data
        out.append(float(node.get(key, 0.0)))
    return np.array(out)


def write_candidate_toml(
    weights_path: Path, out_path: Path, names: list[str], w: np.ndarray
) -> None:
    """Section-aware line rewrite: update only the refit scalar keys."""
    updates: dict[tuple[str | None, str], float] = {}
    for name, val in zip(names, w):
        if name in FEATURE_TO_TOML:
            updates[FEATURE_TO_TOML[name]] = float(val)

    lines = weights_path.read_text(encoding="utf-8").splitlines(keepends=True)
    section: str | None = None
    out_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
        elif "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if (section, key) in updates:
                # repr(float) always carries a decimal point / exponent,
                # keeping the TOML value a float (`:.6g` would render
                # 0.0 as the integer `0`).
                line = f"{key} = {float(f'{updates[section, key]:.6g}')!r}\n"
        out_lines.append(line)
    out_path.write_text("".join(out_lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", help="labels.jsonl from `run_arena label-campaign`")
    parser.add_argument(
        "--weights",
        type=Path,
        default=_DEFAULT_WEIGHTS,
        help="current heuristic_v.toml (template + before-agreement baseline)",
    )
    parser.add_argument("--out", type=Path, default=Path("heuristic_v_candidate.toml"))
    parser.add_argument("--l2", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--iters", type=int, default=4000)
    args = parser.parse_args(argv)

    decisions = load_labels(Path(args.labels))
    if not decisions:
        sys.exit("no labeled decisions with >=2 options in the input")
    names = feature_names(decisions)
    print(f"decisions: {len(decisions)}, features: {len(names)}")

    x, y = build_samples(decisions, names)
    print(f"pairwise samples: {len(y)}")

    w_old = read_current_weights(args.weights, names)
    w_new = fit_logistic(x, y, l2=args.l2, lr=args.lr, iters=args.iters)

    # Rescale: logistic weights are scale-free; match the old weight
    # vector's norm so V magnitudes (and [tiers] margins) stay comparable.
    old_norm = float(np.linalg.norm(w_old))
    new_norm = float(np.linalg.norm(w_new))
    if old_norm > 0 and new_norm > 0:
        w_new = w_new * (old_norm / new_norm)

    # Variance guard: fight_delta labels only vary what differs BETWEEN
    # options at one decision (deck shape, mostly). Features that never
    # vary in the pairwise diffs (HP, gold, potions, ascension, ...)
    # carry zero gradient and would be L2-shrunk to 0 — keep their
    # priors instead, since the run surfaces (rest heal-vs-smith, shop
    # gold reserve, pathing HP terms) depend on them. 2026-06-12: the
    # first 500-run campaign zeroed hp_base/low_hp/gold_sqrt this way.
    informed = np.abs(x).max(axis=0) > 1e-9
    kept = [n for n, inf in zip(names, informed) if not inf]
    w_new = np.where(informed, w_new, w_old)
    if kept:
        print(
            f"variance guard: {len(kept)} feature(s) kept at prior (no signal): {', '.join(kept)}"
        )

    before = agreement(decisions, names, w_old)
    after = agreement(decisions, names, w_new)
    print(f"agreement with rollout-preferred option: before={before:.1%} after={after:.1%}")

    print("weights (old → new):")
    for name, old, new in zip(names, w_old, w_new):
        marker = "  *" if abs(new - old) > 1e-6 else "   "
        print(f"{marker} {name:<24} {old:+.4f} → {new:+.4f}")

    write_candidate_toml(args.weights, args.out, names, w_new)
    print(f"candidate weights written: {args.out}")
    if after < before:
        print("WARNING: refit underperforms the hand priors on this corpus — don't ship it")


if __name__ == "__main__":
    main()
