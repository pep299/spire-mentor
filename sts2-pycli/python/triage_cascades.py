"""Batch-classify v2-fuzz cascade seeds: genuine RNG desync vs in-sync.

For the draw/reshuffle cascade family (PILE_COUNT / DRAW_ORDER), the
per-stream RNG *counters* (always in the state dump — no STS2_TRACE_RNG
needed) already tell sync-vs-desync at the first divergence:

  * GENUINE_DESYNC  — a combat-local stream's read COUNT differs at the
                      first field divergence. An engine bug: the shuffle/
                      draw OPERATION shape differs. Deep-dive with
                      ``trace_rng.py <seed>`` to see the per-read shape.
  * IN_SYNC         — every stream counter matches; the divergence is
                      DETERMINISTIC (state/draw-count), not RNG. The big
                      sub-case is the C#-headless Draw-truncation artifact
                      (SHRUG_IT_OFF / CURE_ALL …) where RUST IS CORRECT.

This reproduces each seed exactly as ``parity_audit.run_trial`` (action
rng ``Random(seed ^ 0xDEADBEEF)``, scenario = N-th ``assemble_v2_trials``
entry), with a FRESH Game process per seed — reusing one process across
``g.start`` calls leaves C# in a terminal/stale state (only the first
trial validates), so each seed gets its own short-lived headless.

Within IN_SYNC, read the cs=/rs= direction: cs<rs (C# left a card in
draw / drew fewer) is the C#-headless Draw-truncation artifact where RUST
IS CORRECT; cs>rs may instead be a real RS under-draw — confirm per-seed
with ``trace_rng.py``.

Usage:
    ./scripts/wheel-venv.sh run -- python python/triage_cascades.py --hash <detail_hash> [--limit N]
    ./scripts/wheel-venv.sh run -- python python/triage_cascades.py --seeds 1499,2738,1452
    # --manifest defaults to the newest docs/audits/sweeps/*-full-manifest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import sts2_rs
from parity_audit import (
    _apply_csharp,
    _apply_rust,
    _csharp_init,
    _csharp_legal,
    _drain_csharp_choices,
    _drain_rust_choices,
    _enrich_csharp,
    _enrich_rust,
    _rust_init,
    _rust_legal,
    assemble_v2_trials,
)
from parity_fuzz_state import diff
from sts2_client import Game

_COMBAT_STREAMS = (
    "shuffle",
    "combat_card_generation",
    "combat_card_selection",
    "combat_energy_costs",
    "combat_targets",
    "monster_ai",
    "niche",
    "combat_orbs",
    "combat_potion_generation",
)


def _counters(cs_raw, rs_state):
    cs = cs_raw.get("rng_counters") or {}
    rs = sts2_rs.state_to_dict(rs_state.inner).get("rng_counters") or {}
    return cs, rs


def _desync_streams(cs_cnt, rs_cnt):
    return [s for s in _COMBAT_STREAMS if cs_cnt.get(s) != rs_cnt.get(s)]


def classify_seed(g: Game, seed: int, sc, max_turns: int) -> dict:
    """Reproduce one trial; return the first-divergence classification."""
    rng = random.Random(seed ^ 0xDEADBEEF)
    res = {
        "seed": seed,
        "encounter": sc.encounter,
        "class": None,
        "first_path": None,
        "desync": [],
        "action_card": None,
        "detail": "",
    }
    try:
        cs_raw = _csharp_init(g, sc)
        cs_monsters = [m["kind"] for m in (cs_raw.get("monsters") or [])]
        cs_move_states = [m.get("move_state") for m in (cs_raw.get("monsters") or [])]
        rs_state = _rust_init(sc, monsters=None, move_states=cs_move_states)
        if [m.get("kind") for m in (_enrich_rust(rs_state).get("monsters") or [])] != cs_monsters:
            rs_state = _rust_init(sc, monsters=cs_monsters, move_states=cs_move_states)
        _drain_rust_choices(rs_state)
    except Exception as exc:
        res["class"] = "HARNESS_INIT"
        res["detail"] = repr(exc)[:120]
        return res

    cs = _enrich_csharp(cs_raw)
    rs = _enrich_rust(rs_state)
    cs_cnt, rs_cnt = _counters(cs_raw, rs_state)

    def finalize(divs, action):
        ds = _desync_streams(cs_cnt, rs_cnt)
        res["desync"] = ds
        if action and action.get("action") == "play_card":
            res["action_card"] = action.get("_card_id")
        elif action:
            res["action_card"] = action.get("action")
        if divs:
            res["first_path"] = divs[0]["path"]
        if ds:
            res["class"] = "GENUINE_DESYNC"
            # Surface the REAL upstream field family (the shuffle read-count
            # delta is only a downstream detector) so the genuine-desync set
            # can be re-grouped by cause: exhaust-count / monster-state / draw.
            fields = {d["path"].split(".")[0].split("[")[0] for d in (divs or [])}
            tags = [
                t for t in ("exhaust_count", "monsters", "draw_count", "hand_count") if t in fields
            ]
            res["detail"] = ",".join(f"{s}:cs{cs_cnt.get(s)}/rs{rs_cnt.get(s)}" for s in ds) + (
                " | fields=" + "+".join(tags) if tags else ""
            )
        else:
            fp = divs[0]["path"] if divs else ""
            if fp.startswith(("draw_count", "draw_ids", "hand_count", "hand_ids", "discard")):
                res["class"] = "IN_SYNC_DRAW"
                d0 = divs[0]
                res["detail"] = f"{fp} cs={d0.get('csharp')} rs={d0.get('rust')}"
            else:
                res["class"] = "IN_SYNC_OTHER"
                res["detail"] = (
                    f"{fp} cs={divs[0].get('csharp')} rs={divs[0].get('rust')}" if divs else ""
                )

    if diff(cs, rs):
        finalize(diff(cs, rs), {"action": "init"})
        return res

    turn = 0
    while turn < max_turns:
        if (
            cs.get("combat_terminal")
            and rs.get("combat_terminal")
            and cs.get("player_won") == rs.get("player_won")
        ):
            res["class"] = "CLEAN_TERMINAL"
            return res
        cs_actions = _csharp_legal(g)
        rs_actions = _rust_legal(rs_state)
        if cs_actions != rs_actions:
            cs_cnt, rs_cnt = _counters(cs_raw, rs_state)
            res["class"] = "LEGAL_ACTION"
            res["desync"] = _desync_streams(cs_cnt, rs_cnt)
            res["detail"] = (
                f"cs_only={[a.get('action') for a in cs_actions if a not in rs_actions][:3]}"
            )
            return res
        if not cs_actions:
            break
        action = rng.choice(cs_actions)
        if action["action"] == "play_card":
            ci = action.get("card_index", 0)
            hand = cs.get("hand_ids") or []
            action = dict(action)
            action["_card_id"] = hand[ci] if 0 <= ci < len(hand) else None
        try:
            cs_result = _apply_csharp(g, action)
            _drain_csharp_choices(g, cs_result)
            _apply_rust(rs_state, action)
            _drain_rust_choices(rs_state)
            cs_raw = g.get_state()
            cs = _enrich_csharp(cs_raw)
            rs = _enrich_rust(rs_state)
            cs_cnt, rs_cnt = _counters(cs_raw, rs_state)
        except Exception as exc:
            res["class"] = "HARNESS_APPLY"
            res["detail"] = repr(exc)[:120]
            return res

        divs = diff(cs, rs)
        if divs:
            # skip both-terminal-same-outcome teardown artifact
            if (
                cs.get("combat_terminal")
                and rs.get("combat_terminal")
                and cs.get("player_won") == rs.get("player_won")
            ):
                res["class"] = "CLEAN_TERMINAL"
                return res
            finalize(divs, action)
            res["turn"] = turn
            return res
        if action.get("action") == "end_turn":
            turn += 1

    res["class"] = "CLEAN_TO_MAXTURN"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", help="detail_hash cluster to triage")
    ap.add_argument("--seeds", help="explicit comma seed list")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tier", default="full")
    ap.add_argument("--axes", default="upgraded,arenas,fuzz,pairwise")
    args = ap.parse_args()

    if args.seeds:
        want = [int(s) for s in args.seeds.split(",")]
    else:
        mani = (
            args.manifest
            or sorted(
                glob.glob(
                    os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "..",
                        "sts2-rs",
                        "docs",
                        "audits",
                        "sweeps",
                        "*-full-manifest.json",
                    )
                )
            )[-1]
        )
        subs = json.load(open(mani, encoding="utf-8"))["subjects"]
        want = [
            int(k.split(":")[1])
            for k, v in subs.items()
            if k.startswith("seed:")
            and v.get("verdict") == "divergent"
            and (args.hash is None or v.get("detail_hash") == args.hash)
        ]
        want.sort()
        print(f"# manifest {os.path.basename(mani)} hash={args.hash} -> {len(want)} dirty seeds")
    if args.limit:
        want = want[: args.limit]

    trials = assemble_v2_trials(args.tier, args.axes.split(","))
    by_seed = {t[0]: (t[1], t[2]) for t in trials}

    rows = []
    for seed in want:
        if seed not in by_seed:
            print(f"  seed {seed}: NOT IN TRIALS (tier/axes mismatch)")
            continue
        sc, mt = by_seed[seed]
        # Fresh Game per seed: reusing one process across g.start() calls
        # leaves C# in a terminal/stale state (only the first trial is valid).
        g = Game()
        try:
            r = classify_seed(g, seed, sc, mt)
        finally:
            g.close()
        rows.append(r)
        print(
            f"  {seed:5} {r['class']:16} {r.get('first_path') or '':14} "
            f"{(r.get('action_card') or ''):20} {r['detail'][:70]}"
        )

    from collections import Counter

    tally = Counter(r["class"] for r in rows)
    print("\n──────── tally ────────")
    for k, n in tally.most_common():
        print(f"  {n:4}  {k}")
    genuine = [r["seed"] for r in rows if r["class"] == "GENUINE_DESYNC"]
    print(f"\nGENUINE_DESYNC seeds (engine bugs, deep-dive with trace_rng.py): {genuine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
