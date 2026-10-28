"""`python -m ppo.turnplan.recall`: the G0 recall report and verdict (GH #830).

Can a plan filter keep the teacher's best plan? On the HOLDOUT split (run
files `crc32(file) % 5 == 0`, which the trainer never reads), four filters
each rank a position's plans and keep the top K:

- `heur`: by `logp_heur` (the expert heuristic's order),
- `rl2`: by `logp_rl2` (the rl2 playout net's order; no prior ranks last),
- `scorer_nodeck` / `scorer_deck`: by the trained plan scorer without / with
  the deck context.

    python -m ppo.turnplan.recall --data <label dir>... --model <deck dir> \\
        --model-nodeck <dir> --out <report dir> [--n-boot 2000] [--seed 0]

Per filter and K in `KS`: recall@K (the kept set holds a teacher-best plan)
and recall@(K ∪ anchors) (the kept set plus every safety anchor), the teacher
regret `max q - max q over the kept set` (mean, p95) and the kept-set size.
The teacher-best set is every plan with `q == max q`, so a tie at the top is
a hit with regret 0. A filter's own score ties break by plan index.

Positions:

- `all`: every uncapped holdout position. Capped positions (a partial plan
  list) are counted per room type and never measured.
- `resolved`: positions whose best q beats the best q strictly below it by
  more than 2 x the combined SE, `SE = sqrt(q_var / visits)` (0 for a
  terminal plan, `visits == 0`). All-equal plan lists are unresolved.

CIs are a run-file cluster bootstrap (`n_boot` resamples of the files, fixed
seed; ratio estimator); the paired differences `scorer_deck - rl2` and
`scorer_deck - scorer_nodeck` are resampled on the same draws.

Verdict (the spec's G0 kill rule): `PASS` iff, on `resolved`,
recall@(6 ∪ anchors) of `scorer_deck` >= 0.95 AND the CI of
`scorer_deck - rl2` in plain recall@6 lies above 0. The plain top-K is the
comparison because the anchors (rl2's greedy plan among them) are shared by
both filters; the anchor-union difference is reported beside it. The
deck - nodeck difference is reported, not gated. Teacher Q is never
converted to HP.

Output in `<out>`: `recall.json` and `recall.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ppo.turnplan import data
from ppo.turnplan.data import Position

FILTERS = ("heur", "rl2", "scorer_nodeck", "scorer_deck")
KS = (1, 3, 6, 10)
DIFFS = (("scorer_deck", "rl2"), ("scorer_deck", "scorer_nodeck"))
GATE_K = 6
GATE_RECALL = 0.95
N_BOOT = 2000
QUALIFIERS = {
    "character": "Ironclad A10",
    "fights": "devpool_v1 minus dev_v1 (never pro_v1)",
    "split": "holdout run files: crc32(file) % 5 == 0",
    "teacher": "deployed search (BeliefDraw), forced per plan; Q never converted to HP",
}


# ── per position ─────────────────────────────────────────────────────────


def kept_mask(scores: np.ndarray, k: int, anchors: np.ndarray | None = None) -> np.ndarray:
    """The top `k` plans by `scores` (ties: lower plan index first), plus any anchor row."""
    order = np.argsort(-scores, kind="stable")
    kept = np.zeros(len(scores), bool)
    kept[order[:k]] = True
    if anchors is not None:
        kept |= anchors.any(1)
    return kept


def hit(q: np.ndarray, kept: np.ndarray) -> bool:
    """Whether the kept set holds a plan at `max q`."""
    return bool(q[kept].max() == q.max())


def regret(q: np.ndarray, kept: np.ndarray) -> float:
    """Teacher regret: `max q - max q over the kept set` (0 on a hit)."""
    return float(q.max() - q[kept].max())


def is_resolved(p: Position) -> bool:
    """Whether the teacher's best q beats the next-best distinct q by > 2 combined SE."""
    q = p.q.astype(np.float64)
    top = q.max()
    below = q < top
    if not below.any():
        return False
    visits = np.maximum(p.visits, 1)
    se = np.where(p.visits > 0, np.sqrt(np.maximum(p.q_var, 0.0) / visits), 0.0)
    best = int(np.argmax(q))  # first plan at max q
    second = int(np.flatnonzero(below)[np.argmax(q[below])])
    return bool(top - q[second] > 2.0 * np.hypot(se[best], se[second]))


def _metrics(
    positions: list[Position], scores: dict[str, list[np.ndarray]]
) -> dict[tuple[str, int, str], np.ndarray]:
    """`{(filter, K, metric): [n] values}`; metric in hit / hit_a / regret / regret_a / kept_a."""
    out: dict[tuple[str, int, str], np.ndarray] = {}
    for f in FILTERS:
        for k in KS:
            rows = []
            for p, s in zip(positions, scores[f], strict=True):
                top = kept_mask(s, k)
                union = top | p.anchors.any(1)
                rows.append(
                    (
                        hit(p.q, top),
                        hit(p.q, union),
                        regret(p.q, top),
                        regret(p.q, union),
                        int(union.sum()),
                    )
                )
            a = np.asarray(rows, np.float64).reshape(-1, 5)
            for j, m in enumerate(("hit", "hit_a", "regret", "regret_a", "kept_a")):
                out[f, k, m] = a[:, j]
    return out


# ── bootstrap ────────────────────────────────────────────────────────────


class _Boot:
    """Run-file cluster bootstrap of ratio means over one subset of positions."""

    def __init__(self, files: list[str], n_boot: int, seed: int) -> None:
        uniq = sorted(set(files))
        self.idx = np.searchsorted(uniq, files)
        self.n_files = len(uniq)
        self.count = np.bincount(self.idx, minlength=self.n_files).astype(np.float64)
        rng = np.random.default_rng(seed)
        self.w = (
            rng.multinomial(self.n_files, np.full(self.n_files, 1.0 / self.n_files), size=n_boot)
            .astype(np.float64)
            .reshape(n_boot, self.n_files)
        )

    def ci(self, values: np.ndarray) -> dict[str, float]:
        """Point estimate and 95 % percentile CI of the mean of `values`."""
        sums = np.bincount(self.idx, weights=values, minlength=self.n_files)
        est = float(values.mean())
        if len(self.w) == 0:
            return {"est": est, "lo": est, "hi": est}
        boot = (self.w @ sums) / (self.w @ self.count)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # Rounding noise of the weighted sums must not open a zero-width CI.
        lo, hi = min(float(lo), est), max(float(hi), est)
        return {"est": est, "lo": lo, "hi": hi}


def _summary(
    m: dict[tuple[str, int, str], np.ndarray],
    sel: np.ndarray,
    files: list[str],
    n_boot: int,
    seed: int,
    *,
    diffs: bool,
) -> dict[str, Any]:
    n = int(sel.sum())
    sel_files = [f for f, s in zip(files, sel, strict=True) if s]
    out: dict[str, Any] = {"n_positions": n, "n_files": len(set(sel_files)), "filters": {}}
    if n == 0:
        return out
    boot = _Boot(sel_files, n_boot, seed)
    for f in FILTERS:
        cells = {}
        for k in KS:
            hit_, hit_a, reg, reg_a, kept_a = (
                m[f, k, x][sel] for x in ("hit", "hit_a", "regret", "regret_a", "kept_a")
            )
            cells[str(k)] = {
                "recall": boot.ci(hit_),
                "recall_anchors": boot.ci(hit_a),
                "regret": {**boot.ci(reg), "mean": float(reg.mean()), "p95": _p95(reg)},
                "regret_anchors": {
                    **boot.ci(reg_a),
                    "mean": float(reg_a.mean()),
                    "p95": _p95(reg_a),
                },
                "kept_anchors_mean": float(kept_a.mean()),
            }
        out["filters"][f] = cells
    if diffs:
        out["diffs"] = {
            f"{a}-{b}": {
                str(k): {
                    "recall": boot.ci(m[a, k, "hit"][sel] - m[b, k, "hit"][sel]),
                    "recall_anchors": boot.ci(m[a, k, "hit_a"][sel] - m[b, k, "hit_a"][sel]),
                }
                for k in KS
            }
            for a, b in DIFFS
        }
    return out


def _p95(x: np.ndarray) -> float:
    return float(np.percentile(x, 95))


# ── report ───────────────────────────────────────────────────────────────


def report(
    positions: list[Position],
    scores: dict[str, list[np.ndarray]],
    *,
    n_boot: int = N_BOOT,
    seed: int = 0,
    qualifiers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The recall report over `positions` (capped ones counted, not measured).

    Args:
        positions: Holdout positions.
        scores: Per filter in `FILTERS`, one score array per position.
        n_boot: Bootstrap resamples of the run files.
        seed: Bootstrap seed.
        qualifiers: Extra provenance merged into `QUALIFIERS`.
    """
    cap: dict[str, int] = {}
    for p in positions:
        if p.cap_hit:
            cap[p.room_type] = cap.get(p.room_type, 0) + 1
    keep = [i for i, p in enumerate(positions) if not p.cap_hit]
    meas = [positions[i] for i in keep]
    m = _metrics(meas, {f: [scores[f][i] for i in keep] for f in FILTERS})
    files = [p.file for p in meas]
    rooms = np.asarray([p.room_type for p in meas])
    resolved = np.asarray([is_resolved(p) for p in meas], bool)
    subsets: dict[str, Any] = {}
    for name, sel in (("all", np.ones(len(meas), bool)), ("resolved", resolved)):
        s = _summary(m, sel, files, n_boot, seed, diffs=True)
        s["by_room"] = {
            r: _summary(m, sel & (rooms == r), files, n_boot, seed, diffs=True)
            for r in sorted(set(rooms[sel].tolist()))
        }
        subsets[name] = s
    rep = {
        "qualifiers": {**QUALIFIERS, **(qualifiers or {})},
        "n": {
            "positions": len(positions),
            "files": len({p.file for p in positions}),
            "measured": len(meas),
            "measured_files": len(set(files)),
            "resolved": int(resolved.sum()),
            "cap_hit": dict(sorted(cap.items())),
        },
        "bootstrap": {"unit": "run file", "n_boot": n_boot, "seed": seed, "ci": "95% percentile"},
        "filters": list(FILTERS),
        "ks": list(KS),
        "subsets": subsets,
    }
    rep["verdict"] = verdict(rep)
    return rep


def verdict(rep: dict[str, Any]) -> dict[str, Any]:
    """`PASS` iff both G0 clauses hold on `resolved`; else `KILL` naming each failing one."""
    res = rep["subsets"]["resolved"]
    k = str(GATE_K)
    if res["n_positions"] == 0:
        failing = ["no resolved positions to measure"]
    else:
        failing = []
        r = res["filters"]["scorer_deck"][k]["recall_anchors"]["est"]
        if not r >= GATE_RECALL:
            failing.append(
                f"recall@({GATE_K} ∪ anchors) of scorer_deck on resolved = {r:.4f} < {GATE_RECALL}"
            )
        d = res["diffs"]["scorer_deck-rl2"][k]["recall"]
        if not d["lo"] > 0.0:
            failing.append(
                f"CI of scorer_deck − rl2 recall@{GATE_K} on resolved "
                f"[{d['lo']:+.4f}, {d['hi']:+.4f}] does not exclude 0 on the positive side"
            )
    word = "KILL" if failing else "PASS"
    line = f"{word}: " + ("; ".join(failing) if failing else "both G0 clauses hold")
    return {"verdict": word, "failing": failing, "line": line}


def _ci(c: dict[str, float], pct: bool = True) -> str:
    if pct:
        return f"{100 * c['est']:.1f}% [{100 * c['lo']:.1f}, {100 * c['hi']:.1f}]"
    return f"{c['est']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"


def render_md(rep: dict[str, Any]) -> str:
    """`rep` as Markdown."""
    n, q = rep["n"], rep["qualifiers"]
    lines = [
        "# Turn-plan G0 recall (GH #830)",
        "",
        f"**{rep['verdict']['line']}**",
        "",
        f"- {q['character']}; fights {q['fights']}; {q['split']}.",
        f"- Teacher: {q['teacher']}; sims_per_plan {q.get('sims_per_plan')}.",
        f"- {n['positions']} holdout positions in {n['files']} run files; {n['measured']} "
        f"measured ({n['measured_files']} files), {n['resolved']} resolved; capped (excluded) "
        f"by room: {n['cap_hit'] or 'none'}.",
        f"- CIs: 95% run-file cluster bootstrap, {rep['bootstrap']['n_boot']} resamples, "
        f"seed {rep['bootstrap']['seed']}. Regret is in teacher Q, never HP.",
    ]
    for key in ("model_deck", "model_nodeck"):
        if key in q:
            lines.append(f"- {key}: {q[key]}")
    for name, s in rep["subsets"].items():
        lines += ["", f"## `{name}`: {s['n_positions']} positions, {s['n_files']} run files", ""]
        if not s["n_positions"]:
            lines.append("(none)")
            continue
        lines += [
            "| filter | K | recall@K | recall@(K ∪ anchors) | plans kept (K ∪ anchors) | regret mean / p95 "
            "| regret ∪ anchors mean / p95 |",
            "|---|---|---|---|---|---|---|",
        ]
        for f in rep["filters"]:
            for k in rep["ks"]:
                c = s["filters"][f][str(k)]
                lines.append(
                    f"| {f} | {k} | {_ci(c['recall'])} | {_ci(c['recall_anchors'])} "
                    f"| {c['kept_anchors_mean']:.2f} "
                    f"| {c['regret']['mean']:.4f} / {c['regret']['p95']:.4f} "
                    f"| {c['regret_anchors']['mean']:.4f} / {c['regret_anchors']['p95']:.4f} |"
                )
        lines += ["", "Paired differences (same bootstrap draws):", ""]
        lines += ["| difference | K | recall@K | recall@(K ∪ anchors) |", "|---|---|---|---|"]
        for d, cells in s["diffs"].items():
            for k in rep["ks"]:
                c = cells[str(k)]
                lines.append(
                    f"| {d} | {k} | {_ci(c['recall'], False)} | {_ci(c['recall_anchors'], False)} |"
                )
        lines += ["", f"By room type (K = {GATE_K}):", ""]
        lines += ["| room | positions | files | " + " | ".join(rep["filters"]) + " |"]
        lines += ["|---|---|---|" + "---|" * len(rep["filters"])]
        for r, rs in s["by_room"].items():
            cells = [
                f"{_ci(rs['filters'][f][str(GATE_K)]['recall'])} / "
                f"{100 * rs['filters'][f][str(GATE_K)]['recall_anchors']['est']:.1f}%∪a"
                for f in rep["filters"]
            ]
            lines.append(
                f"| {r} | {rs['n_positions']} | {rs['n_files']} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────


def filter_scores(
    dirs: list[str], deck_dir: str, nodeck_dir: str, log: Callable[[str], None]
) -> tuple[list[Position], dict[str, list[np.ndarray]]]:
    """The holdout positions and every filter's scores for them."""
    from ppo.turnplan import train  # torch only when scoring

    scores: dict[str, list[np.ndarray]] = {}
    positions: list[Position] | None = None
    for name, d in (("scorer_deck", deck_dir), ("scorer_nodeck", nodeck_dir)):
        model, vocab = train.load_model(d)
        ps = data.load(dirs, "holdout", vocab)
        if positions is None:
            positions = ps
        elif [(p.key, p.rep, p.file) for p in ps] != [(p.key, p.rep, p.file) for p in positions]:
            raise RuntimeError("holdout positions differ between the two loads")
        log(f"scoring {len(ps)} holdout positions with {name} ({d})")
        scores[name] = train.score(model, ps)
    assert positions is not None
    scores["heur"] = [p.logp_heur for p in positions]
    scores["rl2"] = [p.logp_rl2 for p in positions]
    return positions, scores


def _model_meta(d: str) -> dict[str, Any]:
    path = Path(d) / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    keep = ("best_epoch", "best_val_loss", "n_train", "n_val", "n_train_files", "n_val_files")
    return {"dir": str(d), **{k: meta[k] for k in keep if k in meta}}


def _parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m ppo.turnplan.recall",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", nargs="+", required=True, help="label generator output dirs")
    ap.add_argument("--model", required=True, help="scorer dir trained with the deck context")
    ap.add_argument("--model-nodeck", required=True, help="scorer dir trained with --no-deck")
    ap.add_argument("--out", required=True, help="report dir")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """`python -m ppo.turnplan.recall ...`; returns the exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _parse(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} [turnplan.recall] {msg}", file=sys.stderr, flush=True)

    positions, scores = filter_scores(args.data, args.model, args.model_nodeck, log)
    headers = [Path(d) / "header.json" for d in args.data]
    heads = [json.loads(h.read_text(encoding="utf-8")) for h in headers if h.exists()]
    qualifiers = {
        "data": list(args.data),
        "sims_per_plan": [h["sims_per_plan"] for h in heads if "sims_per_plan" in h],
        "model_deck": _model_meta(args.model),
        "model_nodeck": _model_meta(args.model_nodeck),
        "argv": argv,
    }
    log(f"bootstrapping {args.n_boot} run-file resamples")
    rep = report(positions, scores, n_boot=args.n_boot, seed=args.seed, qualifiers=qualifiers)
    (out / "recall.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    (out / "recall.md").write_text(render_md(rep), encoding="utf-8")
    log(f"{rep['verdict']['line']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
