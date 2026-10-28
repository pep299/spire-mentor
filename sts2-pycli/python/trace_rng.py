"""Bidirectional per-read RNG tracer for a v2-fuzz sweep seed.

The single biggest remaining IC-parity blocker (~156 subjects) is the
PILE_COUNT / DRAW_ORDER draw/reshuffle cascade family: a different number of
RNG reads (or a different shuffle shape) earlier in a random-play combat
shifts which card lands where. Rust-side tracing alone can't reveal *what
operation C# performed* — the C# headless was a black box for the `shuffle`
stream. This tool closes that gap.

Both engines now emit, when ``STS2_TRACE_RNG`` is set, a per-read line on
stderr:

    RNGTRACE seed=<Seed> ctr=<Counter> max=<maxExclusive> r=<result>
    RNGMAP   seed=<Seed> name=<stream_name>

``seed`` is identical across engines (base + DeterministicHashCode(name)),
so the two logs align by ``(seed, ctr)``. A run of descending-``max`` reads
on one seed IS an UnstableShuffle of that length — so this captures both the
per-step read COUNT and the shuffle STRUCTURE.

This driver reproduces a ``seed:N`` v2-fuzz subject EXACTLY as
``parity_audit.run_trial`` does (action rng = ``Random(seed ^ 0xDEADBEEF)``,
scenario = the N-th entry of ``assemble_v2_trials``), captures both engines'
per-read logs segmented by action, walks to the first divergence, and prints
the C#-vs-Rust read log + per-stream counter deltas for the diverging step
(and the one before it) so the causing shuffle/draw op is visible.

Usage:
    ./scripts/wheel-venv.sh run -- python python/trace_rng.py <seed> [tier] [axes]

    <seed>  the seed:N manifest key's N (position in assemble_v2_trials)
    tier    smoke|quick|full        (default full — matches the sweep)
    axes    comma list              (default upgraded,arenas,fuzz,pairwise)
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import threading
import time

# CRITICAL: enable tracing in BOTH engines before sts2_rs is imported or the
# C# subprocess is spawned (the env is read once, cached).
os.environ["STS2_TRACE_RNG"] = "1"

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

_RNG_STREAMS = (
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


# ───────────────────────── C# stderr capture ──────────────────────────────


class CsTrace:
    """Drains the C# subprocess stderr into a line list (thread-safe)."""

    def __init__(self, stderr) -> None:
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._stderr = stderr
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        for line in self._stderr:
            with self._lock:
                self._lines.append(line.rstrip())

    def mark(self) -> int:
        with self._lock:
            return len(self._lines)

    def wait_quiet(self, idle: float = 0.10, timeout: float = 5.0) -> None:
        """Block until no new stderr line for `idle` seconds (or timeout)."""
        deadline = time.time() + timeout
        last = self.mark()
        last_change = time.time()
        while time.time() < deadline:
            time.sleep(0.02)
            n = self.mark()
            if n != last:
                last, last_change = n, time.time()
            elif time.time() - last_change >= idle:
                return

    def since(self, start: int) -> list[str]:
        with self._lock:
            return self._lines[start:]


# ───────────────────────── Rust stderr capture ────────────────────────────
# Rust `eprintln!` (in-process via PyO3) writes to OS fd 2. Redirect fd 2 to a
# temp file for the whole run and interleave our own `##` action markers so the
# Rust read log can be segmented per action. Driver output goes to stdout (fd 1).


class RustTrace:
    def __init__(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w+", suffix=".rusttrace", delete=False
        )
        self._saved_fd2 = os.dup(2)
        os.dup2(self._tmp.fileno(), 2)

    def marker(self, label: str) -> None:
        os.write(2, f"##MARK {label}\n".encode())

    def close_and_read(self) -> list[str]:
        os.fsync(2)
        os.dup2(self._saved_fd2, 2)
        os.close(self._saved_fd2)
        self._tmp.seek(0)
        return [ln.rstrip("\n") for ln in self._tmp.readlines()]


def _parse_trace(lines: list[str]) -> tuple[dict[int, str], list[tuple]]:
    """Return (seed→name map, [(seed, ctr, max, r) ...]) from RNG* lines."""
    names: dict[int, str] = {}
    reads: list[tuple] = []
    for ln in lines:
        if ln.startswith("RNGMAP "):
            parts = dict(kv.split("=", 1) for kv in ln[7:].split())
            names[int(parts["seed"])] = parts["name"]
        elif ln.startswith("RNGTRACE "):
            parts = dict(kv.split("=", 1) for kv in ln[9:].split())
            reads.append((int(parts["seed"]), int(parts["ctr"]), parts["max"], int(parts["r"])))
    return names, reads


def _segment_rust(lines: list[str]) -> dict[str, list[str]]:
    """Split rust trace by ##MARK <label>; return {label: [raw RNG* lines]}."""
    segs: dict[str, list[str]] = {}
    cur = "INIT"
    segs[cur] = []
    for ln in lines:
        if ln.startswith("##MARK "):
            cur = ln[7:]
            segs.setdefault(cur, [])
        else:
            segs[cur].append(ln)
    return segs


def _by_stream(reads: list[tuple], names: dict[int, str]) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    for seed, ctr, mx, r in reads:
        nm = names.get(seed, f"seed{seed}")
        out.setdefault(nm, []).append((ctr, mx, r))
    return out


def _fmt_reads(reads: list[tuple]) -> str:
    return " ".join(f"[{ctr}]{mx}->{r}" for ctr, mx, r in reads)


def _counters(cs_raw: dict, rs_state) -> tuple[dict, dict]:
    cs = cs_raw.get("rng_counters") or {}
    rs = sts2_rs.state_to_dict(rs_state.inner).get("rng_counters") or {}
    return cs, rs


def _print_step(label, cs_seg, rs_seg, cs_names, rs_names):
    print(f"\n──────── {label} : RNG reads ────────")
    cs_n, cs_r = _parse_trace(cs_seg)
    rs_n, rs_r = _parse_trace(rs_seg)
    cs_names.update(cs_n)
    rs_names.update(rs_n)
    cs_streams = _by_stream(cs_r, cs_names)
    rs_streams = _by_stream(rs_r, rs_names)
    for st in sorted(set(cs_streams) | set(rs_streams)):
        c = cs_streams.get(st, [])
        r = rs_streams.get(st, [])
        flag = "" if len(c) == len(r) else "  <<< READ-COUNT DIFFERS"
        print(f"  [{st}] C#({len(c)}) {_fmt_reads(c)}{flag}")
        print(f"  [{st}] RS({len(r)}) {_fmt_reads(r)}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = int(sys.argv[1])
    tier = sys.argv[2] if len(sys.argv) > 2 else "full"
    axes = (sys.argv[3] if len(sys.argv) > 3 else "upgraded,arenas,fuzz,pairwise").split(",")

    trials = assemble_v2_trials(tier, axes)
    match = [t for t in trials if t[0] == target]
    if not match:
        print(f"no trial with seed/position {target} in tier={tier} axes={axes}")
        return 1
    seed, sc, max_turns = match[0]
    subj = sc.coverage_subject or f"seed:{seed}"
    print(f"# tracing {subj}  (scenario encounter={sc.encounter}, max_turns={max_turns})")

    rust_tr = RustTrace()
    rng = random.Random(seed ^ 0xDEADBEEF)
    cs_names: dict[int, str] = {}
    rs_names: dict[int, str] = {}

    try:
        g = Game()
        cs_tr = CsTrace(g.proc.stderr)

        # ── INIT ──
        rust_tr.marker("INIT")
        m0 = cs_tr.mark()
        cs_raw = _csharp_init(g, sc)
        cs_move_states = [m.get("move_state") for m in (cs_raw.get("monsters") or [])]
        cs_monsters = [m["kind"] for m in (cs_raw.get("monsters") or [])]
        rs_state = _rust_init(sc, monsters=None, move_states=cs_move_states)
        rs_native = [m.get("kind") for m in (_enrich_rust(rs_state).get("monsters") or [])]
        if rs_native != cs_monsters:
            rs_state = _rust_init(sc, monsters=cs_monsters, move_states=cs_move_states)
        _drain_rust_choices(rs_state)
        cs_tr.wait_quiet()

        cs = _enrich_csharp(cs_raw)
        rs = _enrich_rust(rs_state)
        prev_seg = ("INIT", cs_tr.since(m0))
        divs = diff(cs, rs)
        cs_cnt, rs_cnt = _counters(cs_raw, rs_state)
        if divs:
            print(f"\n!! DIVERGENCE AT INIT: {divs[0]['path']}")
            all_rust = rust_tr.close_and_read()
            cs_names.update(_parse_trace(cs_tr.since(0))[0])
            rs_names.update(_parse_trace(all_rust)[0])
            rust_segs = _segment_rust(all_rust)
            _print_step("INIT", prev_seg[1], rust_segs.get("INIT", []), cs_names, rs_names)
            _dump_div(divs, cs_cnt, rs_cnt)
            return 0

        turn = 0
        step = 0
        while turn < max_turns:
            if (
                cs.get("combat_terminal")
                and rs.get("combat_terminal")
                and cs.get("player_won") == rs.get("player_won")
            ):
                print(f"\n# both terminal (won={cs.get('player_won')}) — clean to turn {turn}")
                break
            cs_actions = _csharp_legal(g)
            rs_actions = _rust_legal(rs_state)
            if cs_actions != rs_actions:
                print(f"\n!! LEGAL_ACTION divergence at step {step} turn {turn}")
                print("  C# only:", [a for a in cs_actions if a not in rs_actions])
                print("  RS only:", [a for a in rs_actions if a not in cs_actions])
                break
            if not cs_actions:
                break
            action = rng.choice(cs_actions)

            label = f"STEP{step}:{action.get('action')}"
            rust_tr.marker(label)
            m = cs_tr.mark()
            cs_result = _apply_csharp(g, action)
            _drain_csharp_choices(g, cs_result)
            _apply_rust(rs_state, action)
            _drain_rust_choices(rs_state)
            cs_tr.wait_quiet()

            cs_raw = g.get_state()
            cs = _enrich_csharp(cs_raw)
            rs = _enrich_rust(rs_state)
            cs_cnt, rs_cnt = _counters(cs_raw, rs_state)
            cur_seg = (label, cs_tr.since(m))
            divs = diff(cs, rs)
            ds = [s for s in _RNG_STREAMS if cs_cnt.get(s) != rs_cnt.get(s)]
            if divs or ds:
                print(f"\n!! DIVERGENCE at step {step} turn {turn}: action={action}")
                if divs:
                    print(f"   first_path={divs[0]['path']}  ({len(divs)} fields)")
                if ds:
                    print(f"   rng-counter desync streams: {ds}")
                all_rust = rust_tr.close_and_read()
                cs_names.update(_parse_trace(cs_tr.since(0))[0])
                rs_names.update(_parse_trace(all_rust)[0])
                rust_segs = _segment_rust(all_rust)
                # context: prior step then the diverging step
                _print_step(
                    prev_seg[0], prev_seg[1], rust_segs.get(prev_seg[0], []), cs_names, rs_names
                )
                _print_step(label, cur_seg[1], rust_segs.get(label, []), cs_names, rs_names)
                _dump_div(divs, cs_cnt, rs_cnt)
                return 0

            prev_seg = cur_seg
            step += 1
            if action.get("action") == "end_turn":
                turn += 1
        else:
            print(f"\n# reached max_turns={max_turns} with no divergence")

        print("\n# no divergence found (subject is clean under this trace)")
        rust_tr.close_and_read()
        return 0
    finally:
        try:
            rust_tr.close_and_read()
        except Exception:
            pass


def _dump_div(divs, cs_cnt, rs_cnt):
    print("\n──────── per-stream counter totals (cumulative) ────────")
    for s in _RNG_STREAMS:
        c, r = cs_cnt.get(s), rs_cnt.get(s)
        flag = "" if c == r else "  <<<"
        print(f"  {s:26} C#={c} RS={r}{flag}")
    if divs:
        print("\n──────── field divergences (first 12) ────────")
        for d in divs[:12]:
            print(f"  {d['path']}: C#={d.get('csharp')} RS={d.get('rust')}")


if __name__ == "__main__":
    sys.exit(main())
