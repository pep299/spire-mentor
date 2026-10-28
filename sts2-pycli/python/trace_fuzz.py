"""Turn-by-turn parity trace for a FUZZ sweep seed (scenario from examples.json).

Usage:
    scripts/wheel-venv.sh run -- python python/trace_fuzz.py <examples_dir> <seed> [encounter]

Mirrors parity_audit.run_trial's loop EXACTLY (action rng = Random(seed^0xDEADBEEF),
turn counts end_turns) and prints monster max_hp/hp + per-stream rng counters at each
step, stopping at the first divergence so the *causing action* is visible.
"""

import dataclasses
import json
import random
import sys

import parity_audit as pa
import sts2_rs
from trace_subject import (
    _apply_csharp,
    _apply_rust,
    _drain_csharp_choices,
    _drain_rust_choices,
    _enrich_csharp,
    _enrich_rust,
    _rng_line,
)


def _mons(d):
    return [
        f"{m.get('kind')}:hp{m.get('hp')}/max{m.get('max_hp')}" for m in (d.get("monsters") or [])
    ]


def main() -> int:
    exdir = sys.argv[1].rstrip("/") + "/"
    seed = int(sys.argv[2])
    enc = sys.argv[3] if len(sys.argv) > 3 else None
    ex = json.load(open(exdir + "examples.json", encoding="utf-8"))
    items = ex if isinstance(ex, list) else list(ex.values())
    cands = [
        e
        for e in items
        if e.get("seed") == seed and (enc is None or e.get("scenario", {}).get("encounter") == enc)
    ]
    if not cands:
        print(f"no example for seed {seed} enc {enc}", file=sys.stderr)
        return 2
    e = cands[0]
    scd = e["scenario"]
    fields = {f.name for f in dataclasses.fields(pa.Scenario)}
    sc = pa.Scenario(
        **{k: (tuple(v) if isinstance(v, list) else v) for k, v in scd.items() if k in fields}
    )
    print(f"=== seed {seed} enc={sc.encounter} deck={scd.get('deck')}")
    print(f"    relics={scd.get('relics')} potions={scd.get('potions')}")
    sd = e.get("divergences") or []
    print(f"    stored first-div @turn{e.get('turn')}: {sd[:2] if isinstance(sd, list) else sd}")

    prev = None
    rng = random.Random(seed ^ 0xDEADBEEF)
    g = pa.Game()
    cs_raw = pa._csharp_init(g, sc)
    cs_ms = [m["kind"] for m in (cs_raw.get("monsters") or [])]
    cs_move = [m.get("move_state") for m in (cs_raw.get("monsters") or [])]
    rs_state = pa._rust_init(sc, monsters=None, move_states=cs_move)
    if [m.get("kind") for m in (_enrich_rust(rs_state).get("monsters") or [])] != cs_ms:
        rs_state = pa._rust_init(sc, monsters=cs_ms, move_states=cs_move)
    _drain_rust_choices(rs_state)

    cs = _enrich_csharp(cs_raw)
    rs = _enrich_rust(rs_state)
    print(f"[init] cs_mons={_mons(cs)}  rs_mons={_mons(rs)}")
    print(f"       rng {_rng_line(rs_state, cs_raw)}")
    divs = pa.diff(cs, rs)
    if divs:
        print(
            f"  >>> DIVERGENCE AT INIT: {[(d['path'], d['csharp'], d['rust']) for d in divs[:8]]}"
        )
        return 0

    max_turns = 6
    step = 0
    turn = 0
    while turn < max_turns:
        if (
            cs.get("combat_terminal")
            and rs.get("combat_terminal")
            and cs.get("player_won") == rs.get("player_won")
        ):
            print("  (combat over)")
            break
        cs_actions = pa._csharp_legal(g)
        rs_actions = pa._rust_legal(rs_state)
        if cs_actions != rs_actions:
            print(
                f"  !! LEGAL_ACTION mismatch @turn{turn} step{step}: cs_only={[a for a in cs_actions if a not in rs_actions]} rs_only={[a for a in rs_actions if a not in cs_actions]}"
            )
            return 0
        if not cs_actions:
            break
        action = rng.choice(cs_actions)
        cid = ""
        if action.get("action") == "play_card":
            hand = cs.get("hand_ids") or []
            ci = action.get("card_index", 0)
            cid = hand[ci] if 0 <= ci < len(hand) else "?"
        cs_result = _apply_csharp(g, action)
        _drain_csharp_choices(g, cs_result)
        _apply_rust(rs_state, action)
        _drain_rust_choices(rs_state)
        cs_raw = g.get_state()
        cs = _enrich_csharp(cs_raw)
        rs = _enrich_rust(rs_state)
        divs = pa.diff(cs, rs)
        print(
            f"--- t{turn} s{step}: {action.get('action')} {cid} idx={action.get('card_index')} tgt={action.get('target_index')}"
        )
        print(f"      cs_mons={_mons(cs)}  rs_mons={_mons(rs)}")
        print(f"      rng {_rng_line(rs_state, cs_raw)}")
        if divs:
            print(
                f"  >>> FIRST DIVERGENCE @t{turn} s{step}: {[(d['path'], d['csharp'], d['rust']) for d in divs[:8]]}"
            )
            print(
                f"      PREV draw={(prev or {}).get('draw_ids')} disc={(prev or {}).get('discard_ids')}"
            )
            for label, d in (("CS", cs), ("RS", rs)):
                print(
                    f"      {label} hand={d.get('hand_ids')}\n"
                    f"         draw={d.get('draw_ids')}\n"
                    f"         disc={d.get('discard_ids')} exh={d.get('exhaust_ids')}"
                )
            return 0
        prev = cs
        if action.get("action") == "end_turn":
            turn += 1
        step += 1
    print("(no divergence in window)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
