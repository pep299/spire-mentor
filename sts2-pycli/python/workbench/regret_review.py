"""Post-combat regret review — the learning engine.

Replays a recorded run's player decisions, diffs each combat move against the
move MCTS would have recommended (the "MCTS line"), and quantifies the
win-probability it cost:

    Turn 3: Strike instead of Defend dropped P(win) 71%→58% (−13 pts)

This module is the **pure** scoring + rendering core (no HTTP, no game engine):
it consumes the per-action score rows the rest of the workbench already
produces (``reasoning.score`` / the recorder's ``root_outcomes``, both surfaced
as a step's ``scores`` list) plus the move that was actually taken, and emits
ranked :class:`DecisionRegret` rows. The orchestration that feeds it a timeline
— and optionally re-runs MCTS for steps that lack recorded scores — lives in
:func:`review_timeline`; the HTTP surface is in ``server.py`` and the live
"was my last move a blunder?" hook is ``BranchingLiveSession.last_move_regret``.

Win-probability convention: ``wins / visits`` in ``[0, 1]`` — the same "win%"
the ``hint`` panel shows. A *regret* is the non-negative drop
``best_win_prob − played_win_prob``; a move that matches (or beats) the
recommended line has zero regret and is never a blunder.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from enemy_labels import enemy_labels

# Sentinel stable_idx values, mirrored from workbench.record /
# workbench.reasoning so this module stays self-contained.
_END_TURN = -1
_POTION_BASE = -100  # potion slot s -> -100 - s

#: Default blunder threshold: a decision is flagged when it cost at least this
#: many percentage points of win-probability versus the MCTS line. 8 pts sits
#: above MCTS sampling noise at the usual sim budget while still catching the
#: "Strike-instead-of-Defend" class of mistake.
DEFAULT_BLUNDER_DROP_PCT = 8.0

#: Severity (win-prob points, or expected HP) below which a non-best move is a
#: "tie" — within search noise.  Mirrored in the Rust advisor.
TIE_PTS = 3.0
#: Minimum resolved playouts (``hp_n``) on BOTH rows before the HP gap is used.
HP_MIN_SAMPLES = 10
#: Standard errors (of the difference of the two rows' mean final HP) an HP
#: gap must clear to count; under it the gap is rollout noise and reads as 0
#: (GH #888).  Mirrored in the Rust advisor.
HP_GAP_SE = 2.0
#: Minimum MCTS visits on the played move before it can be graded at all.
PLAYED_MIN_VISITS = 50
#: Per-action severity (win-chance points or HP) below which an action does not
#: count toward its turn's grade: summing sub-point jitter over five plays
#: would otherwise invent a slip out of noise.
TURN_ACTION_FLOOR = 1.0


def _round_half_up(x: float) -> int:
    """Round half up: both twins compute ``floor(x + 0.5)``.

    For ``x >= 0`` that is half-away-from-zero (2.5 -> 3, unlike Python's
    banker's ``round``). The Rust advisor uses the same explicit formula, not
    ``f64::round``, so the two stay bit-identical.
    """
    return math.floor(x + 0.5)


def _hp_known(row: dict[str, Any]) -> bool:
    return row.get("mean_final_hp") is not None and int(row.get("hp_n", 0) or 0) >= HP_MIN_SAMPLES


def _hp_gap(best: dict[str, Any], played: dict[str, Any]) -> float:
    """How much more mean final HP ``best`` keeps than ``played``.

    0 when the gap is within :data:`HP_GAP_SE` standard errors of the
    difference (noise) or negative. Clamping only *after* the noise test keeps
    :func:`grade_turn`'s per-turn sum from piling up one-sided jitter. Rows
    without ``hp_sd`` (logged before it existed) skip the test.
    """
    gap = float(best["mean_final_hp"]) - float(played["mean_final_hp"])
    sb, sp = best.get("hp_sd"), played.get("hp_sd")
    if sb is not None and sp is not None:
        var = float(sb) ** 2 / int(best["hp_n"]) + float(sp) ** 2 / int(played["hp_n"])
        if gap < HP_GAP_SE * math.sqrt(var):
            return 0.0
    return max(0.0, gap)


def _win_prob(row: dict[str, Any]) -> float:
    """``win_prob`` of a score row, clamped to ``[0, 1]``."""
    wp = float(row.get("win_prob", 0.0))
    return 0.0 if wp < 0.0 else 1.0 if wp > 1.0 else wp


def action_stable_key(
    args: dict[str, Any] | None, hand: list[dict[str, Any]] | None
) -> tuple[int | None, int | None]:
    """Map a native action dict to ``(stable_idx, target_index)``.

    Mirrors :func:`workbench.record._stable_idx_and_label`: a ``play`` card's
    ``card_index`` is its stable index, ``end_turn`` is ``-1``, a potion slot
    ``s`` is ``-100 - s``. ``hand`` is accepted for parity but unused (the index
    is taken straight from the action). Returns ``(None, None)`` when ``args`` is
    missing or an unrecognized verb.
    """
    if not isinstance(args, dict) or not args:
        return None, None
    kind = args.get("action")
    target = args.get("target_index")
    target = int(target) if target is not None else None
    if kind == "play":
        try:
            return int(args["card_index"]), target
        except (KeyError, TypeError, ValueError):
            return None, target
    if kind == "end_turn":
        return _END_TURN, None
    if kind in ("potion", "drink_potion"):
        try:
            return _POTION_BASE - int(args["slot"]), target
        except (KeyError, TypeError, ValueError):
            return None, target
    return None, target


def _match_row(
    scores: list[dict[str, Any]], stable_idx: int | None, target: int | None
) -> dict[str, Any] | None:
    """Find the score row for ``(stable_idx, target)``.

    Prefers an exact ``(stable_idx, target_index)`` match (multi-target moves
    expose one row per target); falls back to a ``stable_idx``-only match so a
    single-target move still resolves when targets weren't recorded.
    """
    if stable_idx is None:
        return None
    exact = [
        r for r in scores if r.get("stable_idx") == stable_idx and r.get("target_index") == target
    ]
    if exact:
        return exact[0]
    same_idx = [r for r in scores if r.get("stable_idx") == stable_idx]
    return same_idx[0] if same_idx else None


def _best_row(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """The recommended line: the ``is_best`` row, else the highest win-prob."""
    for r in scores:
        if r.get("is_best"):
            return r
    return max(scores, key=_win_prob)


def _enemy_name(enemies: list[dict[str, Any]], target: int | None) -> str | None:
    """Label of the enemy whose ``index`` is *target*, or ``None`` if unresolved.

    The label is the name plus ``" #k"`` when several enemies share it
    (:func:`enemy_labels.enemy_labels`), so a "wrong rat" blunder says which rat.
    """
    if target is None:
        return None
    for pos, e in enumerate(enemies):
        if e.get("index") == target:
            return enemy_labels(enemies)[pos] if e.get("name") else None
    return None


def _targeted_label(row: dict[str, Any], enemies: list[dict[str, Any]]) -> str:
    """Human label for a score row, with the enemy target made explicit.

    Appends ``→ <enemy>`` when the move has a resolvable ``target_index`` *and*
    the fight has more than one enemy — i.e. exactly when the target
    disambiguates which foe was hit, so a "right card, wrong target" blunder
    reads as ``Strike → Cultist instead of Strike → Jaw Worm`` rather than the
    target-blind ``Strike instead of Strike``. Single-enemy fights (where the
    target can't be wrong) keep the clean card-only label.
    """
    base = row.get("action") or "?"
    if len(enemies) <= 1:
        return base
    name = _enemy_name(enemies, row.get("target_index"))
    return f"{base} → {name}" if name else base


def _pct(p: float) -> int:
    """Whole-percent rendering of a probability in ``[0, 1]``."""
    return round(p * 100.0)


@dataclass
class DecisionRegret:
    """One combat decision diffed against the MCTS-recommended line."""

    idx: int  # step index within the run
    node_id: int | None  # live runs: the branch-tree node to seek to (stable across rewinds)
    turn: int | None  # combat round
    floor: int | None
    act: int | None
    played_label: str
    played_win_prob: float
    best_label: str
    best_win_prob: float
    regret: float  # max(0, best - played), in [0, 1]
    drop_pct: int  # round(regret * 100)
    player_rank: int  # 0 = the player took the top line
    n_actions: int
    visits: int  # recommended line's MCTS visits (a confidence proxy)
    matched_best: bool
    is_blunder: bool
    message: str
    played_mean_hp: float | None = None
    best_mean_hp: float | None = None
    hp_drop: float | None = (
        None  # >= 0, 0 within noise; None unless both rows have enough HP samples
    )
    severity: float = 0.0  # max(regret * 100, hp_drop or 0)
    verdict: str = "best"  # best | uncertain | tie | blunder | slip
    best_line: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_decision(
    step: dict[str, Any],
    *,
    blunder_pct: float = DEFAULT_BLUNDER_DROP_PCT,
    scores: list[dict[str, Any]] | None = None,
) -> DecisionRegret | None:
    """Score one combat step's regret, or ``None`` if it can't be assessed.

    Returns ``None`` when the step has no MCTS score rows, fewer than two legal
    moves (no counterfactual), or no resolvable recorded move.

    Args:
        step: A timeline :class:`~workbench.timeline.Step` (or any dict with
            ``scores``, ``chosen_action``, ``state`` and ``meta``).
        blunder_pct: Win-prob drop (percentage points) at/above which the
            decision is flagged as a blunder.
        scores: Override score rows (defaults to ``step["scores"]``). Lets the
            orchestrator inject freshly re-searched scores.
    """
    rows = scores if scores is not None else (step.get("scores") or [])
    if len(rows) < 2:
        return None

    chosen = step.get("chosen_action") or {}
    args = chosen.get("args") if isinstance(chosen, dict) else None
    hand = (step.get("state") or {}).get("hand") or []
    played_idx, played_target = action_stable_key(args, hand)
    played = _match_row(rows, played_idx, played_target)
    if played is None:
        return None

    best = _best_row(rows)
    best_wp = _win_prob(best)
    played_wp = _win_prob(played)
    regret = max(0.0, best_wp - played_wp)
    matched_best = played.get("stable_idx") == best.get("stable_idx") and played.get(
        "target_index"
    ) == best.get("target_index")
    rank = sum(1 for r in rows if _win_prob(r) > played_wp)
    drop_pct = round(regret * 100.0)
    played_hp: float | None = None
    best_hp: float | None = None
    hp_drop: float | None = None
    if _hp_known(played) and _hp_known(best):
        played_hp = float(played["mean_final_hp"])
        best_hp = float(best["mean_final_hp"])
        hp_drop = _hp_gap(best, played)
    severity = max(regret * 100.0, hp_drop or 0.0)
    if matched_best:
        verdict = "best"
    elif int(played.get("visits", 0) or 0) < PLAYED_MIN_VISITS:
        verdict = "uncertain"
    elif severity < TIE_PTS:
        verdict = "tie"
    elif severity >= blunder_pct:
        verdict = "blunder"
    else:
        verdict = "slip"
    is_blunder = verdict == "blunder"

    meta = step.get("meta") or {}
    turn = meta.get("turn")
    enemies = (step.get("state") or {}).get("enemies") or []
    played_label = _targeted_label(played, enemies)
    best_label = _targeted_label(best, enemies)
    if matched_best:
        message = f"Turn {turn}: {played_label} matched the MCTS line (P(win) {_pct(played_wp)}%)"
    else:
        message = (
            f"Turn {turn}: {played_label} instead of {best_label} "
            f"dropped P(win) {_pct(best_wp)}%→{_pct(played_wp)}% (−{drop_pct} pts)"
        )
        if hp_drop is not None and _round_half_up(hp_drop) >= 1:
            message += f", ≈{_round_half_up(hp_drop)} HP"

    node_id_raw = step.get("node_id")
    return DecisionRegret(
        idx=int(step.get("idx", meta.get("step_no", 0)) or 0),
        node_id=int(node_id_raw) if node_id_raw is not None else None,
        turn=turn,
        floor=meta.get("floor"),
        act=meta.get("act"),
        played_label=played_label,
        played_win_prob=played_wp,
        best_label=best_label,
        best_win_prob=best_wp,
        regret=regret,
        drop_pct=drop_pct,
        player_rank=rank,
        n_actions=len(rows),
        visits=int(best.get("visits", 0) or 0),
        matched_best=matched_best,
        is_blunder=is_blunder,
        message=message,
        played_mean_hp=played_hp,
        best_mean_hp=best_hp,
        hp_drop=hp_drop,
        severity=severity,
        verdict=verdict,
        best_line=[str(x) for x in (step.get("best_line") or [])],
    )


@dataclass
class TurnGrade:
    """One combat turn graded as a whole (:func:`grade_turn`)."""

    verdict: str  # best | tie | slip | blunder
    hp_drop: float | None  # None unless some counted action had an HP gap
    severity: float  # max(win-prob points, HP) summed over the counted actions
    win_before: float | None  # best / played win prob at the turn's first graded action
    win_after: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def grade_turn(regrets: list[DecisionRegret | None]) -> TurnGrade | None:
    """Grade a combat turn from its actions' regrets, or ``None`` (stay silent).

    ``regrets`` are the turn's non-forced actions in order (a forced move, one
    legal action, has nothing to compare and is left out by the caller);
    ``None`` is an action that could not be graded.

    .. code-block:: text

        any None or verdict == "uncertain"  -> None (silent)
        counted  = actions with severity >= TURN_ACTION_FLOOR
        severity = max(sum drop_pct, sum hp_drop) over counted
        verdict  = best (none counted) | tie (< TIE_PTS)
                   | blunder (>= DEFAULT_BLUNDER_DROP_PCT) | slip

    Twin of ``advisors::grade_turn`` in ``sts2-workbench``; both are pinned by
    ``tests/fixtures/coach/grade_turn_cases.json``.
    """
    if any(r is None or r.verdict == "uncertain" for r in regrets):
        return None
    graded = [r for r in regrets if r is not None]
    counted = [r for r in graded if r.severity >= TURN_ACTION_FLOOR]
    turn_hp = sum(r.hp_drop or 0.0 for r in counted)
    turn_pts = float(sum(r.drop_pct for r in counted))
    severity = max(turn_pts, turn_hp)
    if not counted:
        verdict = "best"
    elif severity < TIE_PTS:
        verdict = "tie"
    elif severity >= DEFAULT_BLUNDER_DROP_PCT:
        verdict = "blunder"
    else:
        verdict = "slip"
    first = graded[0] if graded else None
    hp_known = any(r.hp_drop is not None for r in counted)
    return TurnGrade(
        verdict=verdict,
        hp_drop=round(turn_hp, 2) if hp_known else None,
        severity=round(severity, 2),
        win_before=round(first.best_win_prob, 4) if first else None,
        win_after=round(first.played_win_prob, 4) if first else None,
    )


def summarize(decisions: list[DecisionRegret], *, top_n: int = 2) -> dict[str, Any]:
    """Aggregate per-decision regrets into the review payload.

    ``blunders`` is the ``top_n`` costliest flagged decisions (severity
    descending, ties broken by earlier turn — the mistake you could have fixed
    soonest first), matching the bead's "surface the 1–2 plays that cost the
    most win-prob".
    """
    flagged = [d for d in decisions if d.is_blunder]
    flagged.sort(key=lambda d: (-d.severity, d.turn if d.turn is not None else 1 << 30, d.idx))
    cumulative = sum(d.regret for d in decisions)
    return {
        "decisions": [d.to_dict() for d in decisions],
        "blunders": [d.to_dict() for d in flagged[:top_n]],
        "summary": {
            "total_decisions": len(decisions),
            "blunder_count": len(flagged),
            "cumulative_drop_pct": round(cumulative * 100.0),
            "cumulative_hp_drop": _round_half_up(
                sum(d.hp_drop for d in decisions if d.hp_drop is not None)
            ),
            "worst_drop_pct": flagged[0].drop_pct if flagged else 0,
        },
    }


def review_steps(
    steps: list[dict[str, Any]],
    *,
    blunder_pct: float = DEFAULT_BLUNDER_DROP_PCT,
    top_n: int = 2,
    rescore: Any = None,
) -> dict[str, Any]:
    """Assess every combat decision in ``steps`` and summarize.

    Args:
        steps: Timeline steps (dicts).
        blunder_pct: Blunder threshold in percentage points.
        top_n: How many costliest blunders to surface.
        rescore: Optional ``callable(step) -> list[score_row]`` invoked for a
            combat step that has no recorded ``scores`` (e.g. a human line the
            recorder didn't search). Returning an empty list skips the step.
    """
    decisions: list[DecisionRegret] = []
    for step in steps:
        if (step.get("state") or {}).get("decision") != "combat_play":
            continue
        rows = step.get("scores") or []
        if not rows and rescore is not None:
            try:
                rows = rescore(step) or []
            except Exception:  # a failed re-search just skips the step
                rows = []
        d = assess_decision(step, blunder_pct=blunder_pct, scores=rows or None)
        if d is not None:
            decisions.append(d)
    out = summarize(decisions, top_n=top_n)
    out["config"] = {"blunder_pct": blunder_pct, "top_n": top_n}
    return out


def review_timeline(
    store: Any,
    *,
    blunder_pct: float = DEFAULT_BLUNDER_DROP_PCT,
    top_n: int = 2,
    rescore: Any = None,
) -> dict[str, Any]:
    """Run the regret review over a loaded :class:`~workbench.timeline.TimelineStore`."""
    return review_steps(store.all(), blunder_pct=blunder_pct, top_n=top_n, rescore=rescore)
