"""Honest-corpus loss anatomy (Phase 0 of the boss-loss diagnosis).

Spec: docs/superpowers/specs/2026-07-17-boss-loss-diagnosis-design.md.
Re-derives the 2026-07-12 anatomy (done on the CLAIRVOYANT corpus) on the
honest corpus: death-site split, killer ranking, entry-state win-vs-loss
comparison, deck-size distribution.

Usage:
    PYTHONPATH=python uv run --no-sync python -m runstats.anatomy \
        ../sts2-rs/data/run-trajectories/ppo-ironclad-a0 --json anatomy.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean, median
from typing import Any

from .load import Run, load_corpus


def fatal_fight(run: Run) -> dict[str, Any] | None:
    """Entry stats of the fight the run died in (None for wins / no combat).

    The fatal fight is the trailing block of Combat decisions: the run ends
    when the player dies, so the last combat in the decision stream is the
    killer. Entry HP = hp at that block's first decision (round resets mark
    block starts, same heuristic as ``Run.combats``).
    """
    if run.won:
        return None
    # Non-completions (livelock / empty_actions, GH #270) are NOT combat
    # deaths: win=False but killed_by_encounter is a STALE last-combat lead
    # and the run often ends at healthy HP in a Shop. Only outcome == "lost"
    # is a real death.
    if run.outcome != "lost":
        return None
    blocks: list[list[dict[str, Any]]] = []
    prev_round = 10**9
    for d in run.decisions:
        o = d.get("offered", {})
        if o.get("kind") != "combat":
            # Fights are always separated by at least one non-combat decision
            # (the recorder logs even single-option MapChoice moves), so a
            # non-combat gap is a reliable fight boundary.
            prev_round = 10**9
            continue
        rnd = int(o.get("round", 0))
        if not blocks or prev_round == 10**9 or rnd < prev_round:
            blocks.append([])
        blocks[-1].append(d)
        prev_round = rnd
    if not blocks:
        return None
    fatal = blocks[-1]
    first, last = fatal[0], fatal[-1]
    return {
        "act": int(first.get("act", 0)),
        "floor": int(first.get("floor", -1)),
        "entry_hp": int(first.get("hp", -1)),
        "turns": int(last.get("offered", {}).get("round", 0)),
        "encounter": run.killed_by,
    }


def _encounters_faced(run: Run) -> list[str]:
    """Lead monster token of every combat room visited (denominator for
    per-encounter loss rates), from ``map_point_history``."""
    leads: list[str] = []
    for act in run.data.get("map_point_history", []):
        for room in act:
            ids = room.get("monster_ids") or []
            if ids:
                leads.append(str(ids[0]))
    return leads


def anatomy(runs: list[Run]) -> dict[str, Any]:
    """Aggregate loss anatomy over a corpus (see module docstring)."""
    losses = [r for r in runs if not r.won]
    deaths_by_enc: dict[str, int] = defaultdict(int)
    faced_by_enc: dict[str, int] = defaultdict(int)
    deaths_by_act: dict[int, int] = defaultdict(int)
    entry_hp_loss: dict[str, list[int]] = defaultdict(list)
    fatal_by_seed: dict[str, dict[str, Any]] = {}

    for r in runs:
        for lead in _encounters_faced(r):
            faced_by_enc[lead] += 1
    for r in losses:
        ff = fatal_fight(r)
        if ff is None:
            key = f"<{r.outcome}>" if r.outcome != "lost" else "<no-combat-death>"
            deaths_by_enc[key] += 1
            continue
        fatal_by_seed[r.seed] = ff
        deaths_by_enc[ff["encounter"]] += 1
        deaths_by_act[ff["act"]] += 1
        entry_hp_loss[ff["encounter"]].append(ff["entry_hp"])

    killers = sorted(
        (
            {
                "encounter": enc,
                "deaths": n,
                "faced": faced_by_enc.get(enc, 0),
                "loss_rate": round(n / faced_by_enc[enc], 4) if faced_by_enc.get(enc) else None,
                "entry_hp_median": median(entry_hp_loss[enc]) if entry_hp_loss[enc] else None,
            }
            for enc, n in deaths_by_enc.items()
        ),
        key=lambda k: -int(k["deaths"] or 0),
    )
    deck_sizes_w = [len(r.deck()) for r in runs if r.won]
    deck_sizes_l = [len(r.deck()) for r in losses]
    return {
        "n_runs": len(runs),
        "n_losses": len(losses),
        "win_rate": round(1 - len(losses) / len(runs), 4) if runs else None,
        "deaths_by_act": dict(sorted(deaths_by_act.items())),
        "killers": killers,
        "deck_size_mean_win": round(mean(deck_sizes_w), 1) if deck_sizes_w else None,
        "deck_size_mean_loss": round(mean(deck_sizes_l), 1) if deck_sizes_l else None,
        "fatal_by_seed": fatal_by_seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", help="run-trajectory corpus dir")
    ap.add_argument("--json", help="write full report JSON here")
    args = ap.parse_args()
    runs = load_corpus(args.corpus)
    rep = anatomy(runs)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
    print(f"runs={rep['n_runs']} losses={rep['n_losses']} wr={rep['win_rate']}")
    print(f"deaths by act: {rep['deaths_by_act']}")
    print(f"deck size win/loss: {rep['deck_size_mean_win']} / {rep['deck_size_mean_loss']}")
    print("top killers:")
    for k in rep["killers"][:15]:
        print(
            f"  {k['encounter']:<40} deaths={k['deaths']:<4} faced={k['faced']:<5}"
            f" loss_rate={k['loss_rate']} entry_hp_med={k['entry_hp_median']}"
        )


if __name__ == "__main__":
    main()
