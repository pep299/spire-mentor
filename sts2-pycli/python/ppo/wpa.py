"""Post-game WPA coaching review — value-curve turning points + counterfactuals.

GitHub issue #201: after a full run completes, show the turning points — the
decisions where the agent's estimated value of the run jumped or dropped the
most — so a player (or an engineer debugging the agent) can see which choices
mattered. Three layers, matching the issue's build plan:

1. **Value curve (cheap, always on).** The PPO critic's value head ``V(s)`` is
   evaluated at every yielded non-combat decision of a completed run; the
   per-decision deltas locate the candidate turning points (RUDDER-style
   return decomposition, reusing the critic we already train).

   .. warning:: The value head estimates the PPO *shaped return* Φ (graded
      progress base + PBRS shaping, ``ppo.rewards``), NOT a win probability —
      never format it as a percentage (same rule as the workbench PPO panel).
      Curve deltas rank *where* the swings were; they don't say how many
      win-percentage points a choice cost.

2. **Counterfactual rollouts (ground truth, paid per flagged point).** For a
   flagged decision, the run is replayed to that decision under the same run
   seed — the prefix is bit-identical by construction (common random numbers:
   the recorded choice indices are fed back through the same env) — then one
   arm takes the played action and the other the alternative, and both are
   rolled to completion N times with *paired* sampling seeds. The empirical
   Δwin turns "your value dropped" into "the other pick would have won ~X%
   more", in real win-rate units grounded in the simulator.

3. **Overdetermined-outcome filter (HCA honesty rule).** A flagged point whose
   paired rollouts never disagree — or whose |Δwin| falls below a threshold —
   is marked ``overdetermined``: the outcome did not hinge on that choice and
   the tool must not coach it, however big the raw curve wobble was.

Caveats stamped into every report (from the issue):

- "good/bad" is policy-relative — continuations are played by the agent
  (combat MCTS + net-sampled meta decisions), which pilots combat
  near-optimally; a pick that is weak for the bot may be fine for a human.
- the raw curve confounds choice with revealed RNG (a drop after a
  card-reward screen may be bad *offers*, not a bad *pick*); only the
  counterfactual disentangles skill from luck.
- CRN keeps the two arms aligned only near the branch; over a long horizon
  the runs genuinely diverge, so Δwin claims are scoped to near-term effect.

Usage (from ``sts2-pycli``, venv with ``sts2_rs``):

    PYTHONPATH=python uv run --frozen python -m ppo.wpa --seed 7
    PYTHONPATH=python uv run --frozen python -m ppo.wpa --seed 7 \\
        --counterfactual --cf-rollouts 6 --out /tmp/wpa-7.json
    # Score an existing `ppo.probe --record` JSONL offline (no engine):
    PYTHONPATH=python uv run --frozen python -m ppo.wpa \\
        --from-jsonl /tmp/ppo-record.jsonl --seed 7

The scoring core is pure (the net is injected), so it also scores recorded
decision rows offline; only driving/replaying runs needs the engine.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any

#: Policies whose recorded runs replay bit-identically by feeding the recorded
#: choice indices back through ``env.step`` — the counterfactual precondition.
#: Expert runs do NOT qualify: expert mode resolves yielded decisions through
#: an extra ``agent.step`` call, so an index-fed replay drifts the combat
#: agent's RNG call count and the prefix is no longer common-random-numbers.
REPLAYABLE_POLICIES = frozenset({"random", "net", "net_greedy"})

DEFAULT_TOP_K = 5
DEFAULT_CF_ROLLOUTS = 6
DEFAULT_CF_THRESHOLD = 0.10
DEFAULT_CF_POINTS = 2

CAVEATS = (
    "policy-relative: continuations are played by the agent (combat MCTS + net sampling); "
    "a pick that is weak for the bot may be fine for a human line",
    "value curve: V is the PPO shaped-return estimate Φ, NOT a win probability; raw swings "
    "confound the choice with the RNG revealed after it (e.g. what the reward screen offered)",
    "counterfactual Δwin is common-random-numbers paired but decorrelates over a long horizon; "
    "read it as the near-term effect of the branch decision",
)

_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "ppo"))
_ITER_RE = re.compile(r"ppo_fullrun_it(\d+)\.pt$")


def default_ckpt() -> str | None:
    """``STS2_WB_PPO_CKPT`` when set (None if missing), else the newest
    committed ``data/ppo/ppo_fullrun_it<N>.pt`` (mirrors the workbench)."""
    env = os.environ.get("STS2_WB_PPO_CKPT")
    if env:
        return env if os.path.exists(env) else None
    best: tuple[int, str] | None = None
    for path in glob.glob(os.path.join(_DATA_DIR, "ppo_fullrun_it*.pt")):
        m = _ITER_RE.search(path)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), path)
    return best[1] if best else None


def action_label(action: dict[str, Any]) -> str:
    """Compact human label for a raw RunAction dict: ``verb(k=v, …)``."""
    verb = str(action.get("action", "?"))
    extras = ", ".join(f"{k}={v}" for k, v in sorted(action.items()) if k != "action")
    return f"{verb}({extras})" if extras else verb


# ── Step 1: the value curve ──────────────────────────────────────────────


@dataclass
class WpaPoint:
    """One decision point on the value curve."""

    index: int  # ordinal over the run's yielded decisions (0-based)
    step: int  # env step counter at the decision
    floor: int | None
    act: int | None
    hp: int | None
    surface: str
    phase: str
    n_actions: int
    chosen: int
    chosen_label: str
    chosen_prob: float  # policy probability of the played action
    argmax: int  # policy's preferred action index
    value: float  # V(s) BEFORE acting — shaped-return Φ units, not P(win)
    delta_prev: float  # value - previous point's value (0.0 at index 0)
    probs: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_decisions(rows: list[dict[str, Any]], net: Any) -> list[WpaPoint]:
    """Score recorded decision rows into the WPA value curve.

    Args:
        rows: Per-decision records as produced by ``ppo.workers.play_one(...,
            record_decisions=True)`` (each carries ``obs`` / ``actions`` /
            ``metrics`` / ``chosen`` / ``step`` / ``surface`` / ``phase``).
        net: Anything with ``distribution(obs, actions, metrics) ->
            (probs, value)`` — normally ``ppo.net_policy.NetPolicy``.

    Returns:
        One :class:`WpaPoint` per row, in run order, with ``delta_prev``
        filled against the previous point (0.0 for the first).
    """
    points: list[WpaPoint] = []
    prev_value: float | None = None
    for i, row in enumerate(rows):
        actions = row["actions"]
        probs, value = net.distribution(row["obs"], actions, row["metrics"])
        chosen = int(row["chosen"])
        metrics = row.get("metrics") or {}
        points.append(
            WpaPoint(
                index=i,
                step=int(row.get("step", i)),
                floor=metrics.get("total_floor"),
                act=metrics.get("act_index"),
                hp=metrics.get("hp"),
                surface=str(row.get("surface", "unknown")),
                phase=str(row.get("phase", "unknown")),
                n_actions=len(actions),
                chosen=chosen,
                chosen_label=action_label(actions[chosen]),
                chosen_prob=round(float(probs[chosen]), 6),
                argmax=max(range(len(probs)), key=probs.__getitem__),
                value=float(value),
                delta_prev=0.0 if prev_value is None else float(value) - prev_value,
                probs=[round(float(p), 6) for p in probs],
            )
        )
        prev_value = float(value)
    return points


# ── Turning points ───────────────────────────────────────────────────────


@dataclass
class TurningPoint:
    """A decision flagged for its realized value swing.

    The swing attributed to decision ``index`` is ``V(index+1) − V(index)``:
    everything that happened between acting at ``index`` and facing the next
    decision (the choice itself + combats/rewards/RNG it revealed). The run's
    LAST decision has no successor V, so it is never curve-flagged — its
    consequence is the terminal outcome itself.
    """

    index: int
    swing: float  # V_after - V_before, shaped-return Φ units
    value_before: float
    value_after: float
    floor: int | None
    surface: str
    chosen_label: str
    alt_index: int | None  # suggested counterfactual action (policy's best other pick)
    alt_label: str | None
    message: str
    counterfactual: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _alt_index(point: WpaPoint) -> int | None:
    """The natural counterfactual at a decision: the policy's argmax when the
    played action differed from it, else the runner-up by probability. None
    when there is no other candidate."""
    if point.n_actions < 2:
        return None
    if point.argmax != point.chosen:
        return point.argmax
    ranked = sorted(range(len(point.probs)), key=point.probs.__getitem__, reverse=True)
    for idx in ranked:
        if idx != point.chosen:
            return idx
    return None


def turning_points(
    points: list[WpaPoint],
    rows: list[dict[str, Any]],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_swing: float = 0.0,
) -> list[TurningPoint]:
    """Rank decisions by their realized value swing ``|V(i+1) − V(i)|``.

    Args:
        points: The scored curve (:func:`score_decisions` output).
        rows: The decision rows the curve was scored from (for alt labels).
        top_k: How many turning points to keep.
        min_swing: Drop swings below this magnitude (Φ units).
    """
    tps: list[TurningPoint] = []
    for i in range(len(points) - 1):
        swing = points[i + 1].value - points[i].value
        if abs(swing) < min_swing:
            continue
        p = points[i]
        alt = _alt_index(p)
        alt_label = action_label(rows[i]["actions"][alt]) if alt is not None else None
        direction = "dropped" if swing < 0 else "rose"
        tps.append(
            TurningPoint(
                index=i,
                swing=round(swing, 6),
                value_before=round(p.value, 6),
                value_after=round(points[i + 1].value, 6),
                floor=p.floor,
                surface=p.surface,
                chosen_label=p.chosen_label,
                alt_index=alt,
                alt_label=alt_label,
                message=(
                    f"Decision {i} (floor {p.floor}, {p.surface}): played {p.chosen_label} — "
                    f"value {direction} {swing:+.3f} (Φ {p.value:.3f}→{points[i + 1].value:.3f}) "
                    f"by the next decision"
                ),
            )
        )
    tps.sort(key=lambda t: (-abs(t.swing), t.index))
    return tps[:top_k]


# ── Steps 2+3: counterfactual rollouts + overdetermined filter ───────────


def cf_sample_seed(seed: int, decision_index: int, k: int) -> int:
    """Deterministic per-rollout sampling seed, shared by both arms of pair
    ``k`` (common random numbers). Pure arithmetic — no ``hash()``, which is
    randomized per process by PYTHONHASHSEED."""
    return (seed * 1_000_003 + decision_index * 8_191 + k) & 0x7FFF_FFFF


def _cf_rollout(job: dict[str, Any]) -> dict[str, Any]:
    """One counterfactual rollout (pool worker; job is a plain dict).

    Replays ``job['prefix']`` (recorded choice indices) over a fresh env with
    the run's own seed, takes ``job['branch_action_idx']`` at the branch
    decision, then net-samples every later decision with the job's sampling
    seed. Any index that no longer fits the replayed candidate list means the
    prefix stopped being common-random-numbers — reported as
    ``replay_divergence``, never silently mis-applied.
    """
    import numpy as np

    from ppo.env import Decision, PpoRunEnv
    from ppo.workers import _net_policy

    env_kwargs: dict[str, Any] = {}
    if job.get("sims") is not None:
        env_kwargs["sims"] = job["sims"]
    if job.get("boss_sims") is not None:
        env_kwargs["boss_sims"] = job["boss_sims"]
    env = PpoRunEnv(
        seed=job["seed"],
        ascension=job["ascension"],
        character=job["character"],
        **env_kwargs,
    )
    rng = np.random.default_rng(job["sample_seed"])
    prefix: list[int] = job["prefix"]
    prefix_n_actions: list[int] = job["prefix_n_actions"]
    pol = None
    out = env.reset()
    di = 0
    while isinstance(out, Decision):
        if di < len(prefix):
            if len(out.actions) != prefix_n_actions[di]:
                return {"won": False, "outcome": "replay_divergence"}
            chosen = prefix[di]
        elif di == len(prefix):
            if len(out.actions) != job["branch_n_actions"]:
                return {"won": False, "outcome": "replay_divergence"}
            chosen = job["branch_action_idx"]
        else:
            if pol is None:
                pol = _net_policy(job.get("ckpt_path"))
            chosen, _, _, _ = pol.sample(out.obs, out.actions, out.metrics, rng, greedy=False)
        if not 0 <= chosen < len(out.actions):
            return {"won": False, "outcome": "replay_divergence"}
        di += 1
        out = env.step(chosen)
    return {"won": bool(out.won), "outcome": out.outcome}


def summarize_counterfactual(
    factual: list[dict[str, Any]],
    alternative: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_CF_THRESHOLD,
    played_label: str = "played",
    alt_label: str = "alternative",
) -> dict[str, Any]:
    """Aggregate paired rollout results into the counterfactual verdict.

    ``delta_win`` is ``P(win | alternative) − P(win | played)`` over the
    paired rollouts. The HCA suppression rule: the point is
    ``overdetermined`` when no pair disagrees, and ``suppressed`` (don't
    coach it) when overdetermined or ``|delta_win| < threshold``.
    """
    n = min(len(factual), len(alternative))
    fact, alt = factual[:n], alternative[:n]
    anomalies = sum(1 for r in fact + alt if r["outcome"] in ("replay_divergence", "error"))
    fact_wins = sum(1 for r in fact if r["won"])
    alt_wins = sum(1 for r in alt if r["won"])
    pairs_differ = sum(1 for f, a in zip(fact, alt) if f["won"] != a["won"])
    fact_rate = fact_wins / n if n else 0.0
    alt_rate = alt_wins / n if n else 0.0
    delta = alt_rate - fact_rate
    overdetermined = n > 0 and pairs_differ == 0
    suppressed = overdetermined or abs(delta) < threshold
    if overdetermined:
        message = (
            f"counterfactual (n={n} paired rollouts): outcome identical in every pair "
            f"({played_label} {fact_wins}/{n} vs {alt_label} {alt_wins}/{n}) — "
            f"overdetermined here, not coaching this decision"
        )
    else:
        pts = round(delta * 100.0)
        better = alt_label if delta > 0 else played_label
        message = (
            f"counterfactual (n={n} paired rollouts): {played_label} won {fact_wins}/{n}, "
            f"{alt_label} won {alt_wins}/{n} → Δwin {pts:+d} pts ({better} better; "
            f"near-term, policy-relative)"
        )
        if suppressed:
            message += f" — below the ±{round(threshold * 100)} pt bar, not coaching it"
    return {
        "n": n,
        "factual_wins": fact_wins,
        "alternative_wins": alt_wins,
        "factual_win_rate": round(fact_rate, 4),
        "alternative_win_rate": round(alt_rate, 4),
        "delta_win": round(delta, 4),
        "pairs_differ": pairs_differ,
        "anomalies": anomalies,
        "overdetermined": overdetermined,
        "suppressed": suppressed,
        "message": message,
    }


def run_counterfactuals(
    row: dict[str, Any],
    tps: list[TurningPoint],
    *,
    rollouts: int = DEFAULT_CF_ROLLOUTS,
    threshold: float = DEFAULT_CF_THRESHOLD,
    max_points: int = DEFAULT_CF_POINTS,
    procs: int | None = None,
    sims: int | None = None,
    boss_sims: int | None = None,
    ckpt_path: str | None = None,
    rollout_fn: Any = None,
) -> None:
    """Fill ``tp.counterfactual`` for the top ``max_points`` turning points.

    Args:
        row: The ``play_one`` episode row the curve came from (needs ``seed``
            / ``policy`` / ``character`` / ``ascension`` / ``decisions``).
        tps: Turning points from :func:`turning_points` (mutated in place).
        rollouts: Paired rollouts per point (2×rollouts runs per point).
        threshold: Suppression bar on ``|Δwin|`` (HCA filter).
        max_points: Only the first ``max_points`` turning points are paid for.
        procs: Pool size (None = cpu−2; 1 = in-process, no pool).
        sims / boss_sims: Combat budget — MUST match the base run's, or the
            prefix replay is not common-random-numbers.
        ckpt_path: Net checkpoint for sampling the post-branch continuation.
        rollout_fn: Test seam; replaces :func:`_cf_rollout` (forces procs=1).

    Raises:
        ValueError: When the row's driving policy is not index-replayable
            (see :data:`REPLAYABLE_POLICIES`).
    """
    if row.get("policy") not in REPLAYABLE_POLICIES:
        raise ValueError(
            f"counterfactuals need an index-replayable run (policy in "
            f"{sorted(REPLAYABLE_POLICIES)}), got policy={row.get('policy')!r} — "
            f"expert-mode runs perturb the combat agent's RNG call count on replay"
        )
    decisions = row["decisions"]
    choices = [int(d["chosen"]) for d in decisions]

    jobs: list[tuple[TurningPoint, str, dict[str, Any]]] = []
    for tp in tps[:max_points]:
        if tp.alt_index is None:
            continue
        base = {
            "seed": row["seed"],
            "ascension": row.get("ascension", 0),
            "character": row.get("character", "Ironclad"),
            "sims": sims,
            "boss_sims": boss_sims,
            "ckpt_path": ckpt_path,
            "prefix": choices[: tp.index],
            "prefix_n_actions": [len(d["actions"]) for d in decisions[: tp.index]],
            "branch_n_actions": len(decisions[tp.index]["actions"]),
        }
        for k in range(rollouts):
            ss = cf_sample_seed(int(row["seed"]), tp.index, k)
            jobs.append(
                (tp, "factual", {**base, "branch_action_idx": choices[tp.index], "sample_seed": ss})
            )
            jobs.append(
                (tp, "alternative", {**base, "branch_action_idx": tp.alt_index, "sample_seed": ss})
            )
    if not jobs:
        return

    fn = rollout_fn or _cf_rollout
    if rollout_fn is not None:
        procs = 1
    if procs is None:
        procs = max(1, (os.cpu_count() or 2) - 2)
    if procs <= 1:
        results = [fn(job) for _, _, job in jobs]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(procs, len(jobs))) as pool:
            results = pool.map(fn, [job for _, _, job in jobs])

    by_point: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for (tp, arm, _), res in zip(jobs, results):
        by_point.setdefault(tp.index, {"factual": [], "alternative": []})[arm].append(res)
    for tp in tps[:max_points]:
        arms = by_point.get(tp.index)
        if arms is None:
            continue
        tp.counterfactual = summarize_counterfactual(
            arms["factual"],
            arms["alternative"],
            threshold=threshold,
            played_label=tp.chosen_label,
            alt_label=tp.alt_label or "alternative",
        )


# ── Orchestration + report ───────────────────────────────────────────────


def build_report(
    row: dict[str, Any],
    net: Any,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_swing: float = 0.0,
) -> dict[str, Any]:
    """Score a recorded episode row into the WPA report (steps 1 of 3).

    Pure given ``net`` — works offline on any ``play_one``-shaped row with
    recorded decisions (e.g. a ``ppo.probe --record`` JSONL line).
    """
    rows = row.get("decisions") or []
    if not rows:
        raise ValueError(
            "row has no recorded decisions — drive with record_decisions=True "
            "(ppo.probe --record) or use `python -m ppo.wpa --seed …` to drive one"
        )
    points = score_decisions(rows, net)
    tps = turning_points(points, rows, top_k=top_k, min_swing=min_swing)
    return {
        "run": {
            "seed": row.get("seed"),
            "policy": row.get("policy"),
            "character": row.get("character", "Ironclad"),
            "ascension": row.get("ascension", 0),
            "outcome": row.get("outcome"),
            "won": bool(row.get("won")),
            "steps": row.get("steps"),
            "n_decisions": len(rows),
            "final": row.get("final"),
        },
        "curve": [p.to_dict() for p in points],
        "turning_points": [t.to_dict() for t in tps],
        "caveats": list(CAVEATS),
        "_tps": tps,  # internal (stripped before JSON emit)
    }


def render_text(report: dict[str, Any]) -> str:
    """Human-readable rendering of a WPA report."""
    run = report["run"]
    out: list[str] = []
    result = "WON" if run["won"] else f"result={run['outcome']}"
    out.append(
        f"WPA review — seed {run['seed']} {run['character']} A{run['ascension']} "
        f"(policy {run['policy']}) → {result} after {run['n_decisions']} decisions"
    )
    out.append("value curve (V = PPO critic shaped-return Φ — NOT a win probability):")
    out.append(f"  {'idx':>3} {'floor':>5} {'surface':<10} {'V':>8} {'ΔV':>8}  chosen")
    for p in report["curve"]:
        out.append(
            f"  {p['index']:>3} {p['floor'] if p['floor'] is not None else '?':>5} "
            f"{p['surface']:<10} {p['value']:>8.3f} {p['delta_prev']:>+8.3f}  "
            f"{p['chosen_label']} (p={p['chosen_prob']:.2f})"
        )
    tps = report["turning_points"]
    out.append("")
    out.append(f"turning points (top {len(tps)} by |swing|):")
    if not tps:
        out.append("  (none above the swing threshold)")
    for rank, t in enumerate(tps, 1):
        out.append(f"  {rank}. {t['message']}")
        if t.get("alt_label"):
            out.append(f"     alternative: {t['alt_label']}")
        cf = t.get("counterfactual")
        if cf:
            out.append(f"     {cf['message']}")
    out.append("")
    out.append("caveats:")
    for c in report["caveats"]:
        out.append(f"  - {c}")
    return "\n".join(out)


def _emit(report: dict[str, Any], out_path: str | None) -> None:
    payload = {k: v for k, v in report.items() if not k.startswith("_")}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[wpa] report written to {out_path}", file=sys.stderr)
    print(render_text(report))


def _load_row(path: str, seed: int | None) -> dict[str, Any]:
    """Pick the episode row from a ``ppo.probe --record`` JSONL file."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("decisions")]
    if seed is not None:
        rows = [r for r in rows if r.get("seed") == seed]
    if not rows:
        raise SystemExit(f"no row with recorded decisions{f' for seed {seed}' if seed else ''}")
    if len(rows) > 1:
        raise SystemExit(f"{len(rows)} candidate rows — disambiguate with --seed")
    return rows[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Post-game WPA coaching review (issue #201): value curve, "
        "turning points, counterfactual Δwin."
    )
    ap.add_argument("--seed", type=int, default=None, help="run seed (drive mode)")
    ap.add_argument(
        "--from-jsonl",
        type=str,
        default=None,
        help="score a recorded ppo.probe --record row instead of driving a run",
    )
    ap.add_argument(
        "--policy",
        choices=("net_greedy", "net", "random", "expert"),
        default="net_greedy",
        help="driving policy (drive mode); counterfactuals need a non-expert policy",
    )
    ap.add_argument("--ckpt", type=str, default=None, help="RunPolicyNet checkpoint (.pt)")
    ap.add_argument("--character", type=str, default="Ironclad")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--sims", type=int, default=None, help="combat MCTS sims override")
    ap.add_argument("--boss-sims", type=int, default=None, help="boss combat MCTS sims override")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--min-swing", type=float, default=0.0, help="ignore |ΔV| below this (Φ units)")
    ap.add_argument("--out", type=str, default=None, help="write the JSON report here")
    ap.add_argument(
        "--counterfactual",
        action="store_true",
        help="ground-truth the top turning points with paired rollouts (slow: "
        "2×cf-rollouts full continuations per point)",
    )
    ap.add_argument("--cf-rollouts", type=int, default=DEFAULT_CF_ROLLOUTS)
    ap.add_argument("--cf-points", type=int, default=DEFAULT_CF_POINTS)
    ap.add_argument(
        "--cf-threshold",
        type=float,
        default=DEFAULT_CF_THRESHOLD,
        help="suppress coaching when |Δwin| is below this (HCA filter)",
    )
    ap.add_argument("--procs", type=int, default=None, help="counterfactual pool size")
    args = ap.parse_args(argv)

    ckpt = args.ckpt or default_ckpt()
    if ckpt is None:
        print(
            "[wpa] no PPO checkpoint found — pass --ckpt or commit data/ppo/ppo_fullrun_it<N>.pt",
            file=sys.stderr,
        )
        return 2

    if args.from_jsonl:
        row = _load_row(args.from_jsonl, args.seed)
    else:
        if args.seed is None:
            ap.error("--seed is required in drive mode (or use --from-jsonl)")
        from ppo.workers import play_one

        print(f"[wpa] driving seed {args.seed} with policy={args.policy} …", file=sys.stderr)
        row = play_one(
            args.seed,
            policy=args.policy,
            ascension=args.ascension,
            character=args.character,
            record_decisions=True,
            ckpt_path=ckpt,
            sims=args.sims,
            boss_sims=args.boss_sims,
        )

    from ppo.net_policy import NetPolicy

    net = NetPolicy(ckpt)
    report = build_report(row, net, top_k=args.top_k, min_swing=args.min_swing)

    if args.counterfactual:
        if row.get("policy") in REPLAYABLE_POLICIES:
            n_pts = min(args.cf_points, len(report["_tps"]))
            print(
                f"[wpa] counterfactualing top {n_pts} points × {args.cf_rollouts} paired "
                f"rollouts (2×{n_pts * args.cf_rollouts} continuations) …",
                file=sys.stderr,
            )
            run_counterfactuals(
                row,
                report["_tps"],
                rollouts=args.cf_rollouts,
                threshold=args.cf_threshold,
                max_points=args.cf_points,
                procs=args.procs,
                sims=args.sims,
                boss_sims=args.boss_sims,
                ckpt_path=ckpt,
            )
            report["turning_points"] = [t.to_dict() for t in report["_tps"]]
        else:
            note = (
                f"skipped: policy={row.get('policy')!r} is not index-replayable "
                f"(need one of {sorted(REPLAYABLE_POLICIES)})"
            )
            report["counterfactual_note"] = note
            print(f"[wpa] counterfactual {note}", file=sys.stderr)

    _emit(report, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
