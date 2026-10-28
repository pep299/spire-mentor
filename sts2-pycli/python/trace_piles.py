"""Per-sub-step PILE tracer for the auto-play-from-draw divergence family.

`trace_rng.py` reveals when the two engines' RNG *streams* diverge, but the
auto-play-from-draw cards/potions (Cascade / Catastrophe / Havoc / Mayhem /
DistilledChaos) can keep the RNG aligned yet diverge in the draw-pile *pool
contents* at a sub-step boundary — same `StableShuffle` RNG, different card
picked, because a card landed in the wrong pile earlier. `RNGTRACE` can't see
that. This tool can.

Both engines, when `STS2_TRACE_PILES=1` (and `STS2_TRACE_RNG=1`, which installs
the C# Harmony hook), emit a pile snapshot on stderr at each AutoPlay and each
discard->draw Shuffle:

    PILETRACE evt=ap   shufctr=<n> card=<Id[+]> draw=[..] hand=[..] discard=[..] exhaust=[..] play=[..]
    PILETRACE evt=shuf shufctr=<n> draw=[..] discard=[..]

`shufctr` is the player's `shuffle`-stream counter — the same key `trace_rng`
aligns on — so the Nth `ap`/`shuf` event in C# pairs with the Nth in Rust.
This driver reconstructs a scenario from an examples.json record (so it works
for ANY character), drives both engines in lockstep on the same action RNG as
`parity_audit`, and prints the per-action PILETRACE events side by side,
flagging the first event whose draw/discard pile differs.

NOTE: a C# `ap` snapshot for Catastrophe still has the to-be-played card in the
draw pile (C# picks-then-plays); Rust's `ap` fires with the card already moved
to Play. So an `ap` draw that differs only by the `card=` value is benign — the
tool marks a real divergence as one where the *remaining* pools differ.

Usage:
    ./scripts/wheel-venv.sh run -- python python/trace_piles.py <examples.json> <bucket> [max_turns]
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import threading

# Enable BOTH tracers before sts2_rs import / C# spawn (env is read once).
os.environ["STS2_TRACE_RNG"] = "1"
os.environ["STS2_TRACE_PILES"] = "1"
os.environ["STS2_PARITY_NOCACHE"] = "1"

sys.path.insert(0, "python")
import parity_audit as pa
from parity_fuzz_scenario import Scenario

_KV = re.compile(r"(\w+)=(\[[^\]]*\]|\S+)")


def _parse_pile_line(ln: str) -> dict | None:
    if "PILETRACE " not in ln:
        return None
    body = ln.split("PILETRACE ", 1)[1]
    return {k: v for k, v in _KV.findall(body)}


class _CsDrain:
    """Drain the C# subprocess stderr into a line list (thread-safe)."""

    def __init__(self, stderr) -> None:
        self._lines: list[str] = []
        self._lock = threading.Lock()
        threading.Thread(target=self._drain, args=(stderr,), daemon=True).start()

    def _drain(self, stderr) -> None:
        for line in stderr:
            with self._lock:
                self._lines.append(line.rstrip())

    def mark(self) -> int:
        with self._lock:
            return len(self._lines)

    def since(self, start: int) -> list[str]:
        with self._lock:
            return list(self._lines[start:])


class _RustFd2:
    """Redirect OS fd 2 (Rust eprintln target) to a temp file; segment by marker."""

    def __init__(self) -> None:
        import tempfile

        self._tmp = tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w+", suffix=".piletrace", delete=False
        )
        self._saved = os.dup(2)
        os.dup2(self._tmp.fileno(), 2)

    def marker(self, label: str) -> None:
        os.write(2, f"##MARK {label}\n".encode())

    def close_and_segment(self) -> dict[str, list[str]]:
        os.fsync(2)
        os.dup2(self._saved, 2)
        os.close(self._saved)
        self._tmp.seek(0)
        segs: dict[str, list[str]] = {"INIT": []}
        cur = "INIT"
        for ln in self._tmp.readlines():
            ln = ln.rstrip("\n")
            if ln.startswith("##MARK "):
                cur = ln[7:]
                segs.setdefault(cur, [])
            else:
                segs[cur].append(ln)
        return segs


def _events(lines: list[str]) -> list[dict]:
    return [d for ln in lines if (d := _parse_pile_line(ln)) is not None]


def _cmp_events(label: str, cs_ev: list[dict], rs_ev: list[dict]) -> bool:
    """Print side-by-side; return True if a real pile divergence was found."""
    print(f"\n──────── {label} : {len(cs_ev)} C# / {len(rs_ev)} RS pile events ────────")
    found = False
    for i in range(max(len(cs_ev), len(rs_ev))):
        c = cs_ev[i] if i < len(cs_ev) else None
        r = rs_ev[i] if i < len(rs_ev) else None
        if c is None or r is None:
            print(f"  [{i}] EVENT-COUNT MISMATCH  cs={c} rs={r}")
            return True
        evt = c.get("evt")
        # For `ap`, C# draw still holds the to-be-played card; normalise by
        # removing the `card=` value once from each side before comparing.
        cdraw, rdraw = c.get("draw", ""), r.get("draw", "")
        cdisc, rdisc = c.get("discard", ""), r.get("discard", "")
        tag = f"evt={evt} shufctr cs={c.get('shufctr')} rs={r.get('shufctr')} card cs={c.get('card')} rs={r.get('card')}"
        draw_diff = cdraw != rdraw
        disc_diff = cdisc != rdisc
        ctr_diff = c.get("shufctr") != r.get("shufctr")
        mark = "  <<<DIFF" if (draw_diff or disc_diff or ctr_diff) else ""
        print(f"  [{i}] {tag}{mark}")
        if draw_diff:
            print(f"        draw cs={cdraw}")
            print(f"        draw rs={rdraw}")
        if disc_diff:
            print(f"        discard cs={cdisc}")
            print(f"        discard rs={rdisc}")
        if (draw_diff or disc_diff or ctr_diff) and not found:
            found = True
    return found


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    # Mode A: `seed <N> <Character> [max_turns]` — reconstruct via
    # assemble_v2_trials (parameterised by character), like trace_rng.
    # Mode B: `<examples.json> <bucket> [max_turns]` — from a sweep record.
    if sys.argv[1] == "seed":
        from parity_audit import assemble_v2_trials

        target = int(sys.argv[2])
        character = sys.argv[3]
        max_turns = int(sys.argv[4]) if len(sys.argv) > 4 else 6
        trials = assemble_v2_trials("full", ["upgraded", "arenas", "fuzz", "pairwise"], character)
        match = [t for t in trials if t[0] == target]
        if not match:
            print(f"no trial seed {target} for {character}")
            return 1
        seed, sc, _ = match[0]
        bucket = f"seed:{target}"
    else:
        ex_path, bucket = sys.argv[1], sys.argv[2]
        max_turns = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        rec = json.load(open(ex_path, encoding="utf-8"))[bucket]
        seed = rec["seed"]
        scd = rec["scenario"]
        sc = Scenario(
            character=scd["character"],
            deck=tuple(scd["deck"]),
            hp=scd["hp"],
            max_hp=scd["max_hp"],
            relics=tuple(scd["relics"]),
            encounter=scd["encounter"],
            potions=tuple(scd.get("potions") or ()),
            draw_order=tuple(scd["draw_order"]) if scd.get("draw_order") else None,
            coverage_subject=scd.get("coverage_subject"),
        )
    print(f"# seed={seed} char={sc.character} enc={sc.encounter} bucket={bucket}")
    print(f"# deck={sc.deck}")

    rng = random.Random(seed ^ 0xDEADBEEF)
    rust_fd2 = _RustFd2()
    oracle = pa._LiveCsOracle(sc)
    cs_drain = _CsDrain(oracle._g.proc.stderr)

    rust_fd2.marker("INIT")
    m0 = cs_drain.mark()
    cs_raw = oracle.init()
    cs_move_states = [m.get("move_state") for m in (cs_raw.get("monsters") or [])]
    rs_state = pa._rust_init(sc, monsters=None, move_states=cs_move_states)
    rs_native = [m.get("kind") for m in (pa._enrich_rust(rs_state).get("monsters") or [])]
    cs_monsters = [m["kind"] for m in (cs_raw.get("monsters") or [])]
    if rs_native != cs_monsters:
        rs_state = pa._rust_init(sc, monsters=cs_monsters, move_states=cs_move_states)
    pa._drain_rust_choices(rs_state)

    cs = pa._enrich_csharp(cs_raw)
    rs = pa._enrich_rust(rs_state)

    actions_log: list[tuple[str, int]] = [("INIT", m0)]
    turn = 0
    while turn < max_turns:
        if cs.get("combat_terminal") and rs.get("combat_terminal"):
            break
        cs_actions = oracle.legal()
        if not cs_actions:
            break
        action = rng.choice(cs_actions)
        label = action.get("action")
        if label == "play_card":
            hand = cs.get("hand_ids") or []
            ci = action.get("card_index", 0)
            label = f"play {hand[ci] if 0 <= ci < len(hand) else '?'}"
        elif label == "use_potion":
            pi = action.get("potion_index", 0)
            label = f"potion {sc.potions[pi] if pi < len(sc.potions) else '?'}"
        seg_label = f"t{turn}:{label}"
        rust_fd2.marker(seg_label)
        m = cs_drain.mark()
        actions_log.append((seg_label, m))
        oracle.apply(action)
        pa._apply_rust(rs_state, action)
        pa._drain_rust_choices(rs_state)
        cs = pa._enrich_csharp(oracle.snapshot())
        rs = pa._enrich_rust(rs_state)
        if action.get("action") == "end_turn":
            turn += 1

    # End markers + collect.
    cs_drain_end = cs_drain.mark()
    rust_segs = rust_fd2.close_and_segment()  # restores fd2

    # Build per-segment C# line ranges.
    bounds = [*actions_log, ("__END__", cs_drain_end)]
    first_div = None
    for idx in range(len(actions_log)):
        label, start = actions_log[idx]
        end = bounds[idx + 1][1]
        cs_lines = cs_drain.since(start)[: end - start] if False else cs_drain.since(start)
        # cs_drain.since(start) returns from start to now; clip to this seg.
        seg_len = end - start
        cs_lines = cs_drain.since(start)[:seg_len]
        rs_seg_lines = rust_segs.get(label, [])
        cs_ev = _events(cs_lines)
        rs_ev = _events(rs_seg_lines)
        cs_sel = [ln for ln in cs_lines if "SELECTTRACE " in ln]
        rs_sel = [ln for ln in rs_seg_lines if "SELECTTRACE " in ln]
        if not cs_ev and not rs_ev and not cs_sel and not rs_sel:
            continue
        if cs_sel or rs_sel:
            print(f"\n──────── {label} : card-select options ────────")
            for s in cs_sel:
                print(f"  cs {s.split('SELECTTRACE ', 1)[1]}")
            for s in rs_sel:
                print(f"  rs {s.split('SELECTTRACE ', 1)[1]}")
        div = _cmp_events(label, cs_ev, rs_ev)
        if div and first_div is None:
            first_div = label
    print(f"\n>>> first pile divergence at: {first_div}")
    return 0


if __name__ == "__main__":
    main()
