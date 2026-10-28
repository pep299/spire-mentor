"""Run-setup RNG trace: capture and diff the `up_front` read sequence that C#
and Rust perform at run creation (Neow + GenerateRooms for all acts).

This is the validation backbone for M1 of the run-RNG parity plan
(`sts2-rs/docs/plans/2026-06-10-run-rng-csharp-parity.md`): C# rolls the
boss/encounters from the shared `up_front` stream inside `GenerateRooms`, Rust
currently rolls them per-node from the Python `run_manager` stream. This tool
shows, draw-by-draw, where the two `up_front` sequences diverge so the port can
be verified read-for-read (the run-scope analog of `trace_rng.py`).

Usage:
  scripts/wheel-venv.sh run -- python python/trace_run_rng.py --seed RP0
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, "python")

# Must be set before sts2_rs import / Game spawn so both engines emit RNG* lines.
os.environ["STS2_TRACE_RNG"] = "1"


def _parse(lines: list[str]) -> tuple[dict[int, str], list[tuple]]:
    names: dict[int, str] = {}
    reads: list[tuple] = []
    for ln in lines:
        if ln.startswith("RNGMAP "):
            kv = dict(p.split("=", 1) for p in ln[7:].split())
            names[int(kv["seed"])] = kv["name"]
        elif ln.startswith("RNGTRACE "):
            kv = dict(p.split("=", 1) for p in ln[9:].split())
            # C# `NextItem` calls the 2-arg `NextInt(0, n)` overload, logged as
            # `max=0:n`; Rust's `next_int_exclusive(0, n)` lowers to the 1-arg
            # `next_int(0, n-1)` form, logged as `max=n`. Same draw — normalize
            # the leading `0:` away so the two align read-for-read.
            mx = kv["max"]
            if mx.startswith("0:"):
                mx = mx[2:]
            reads.append((int(kv["seed"]), int(kv["ctr"]), mx, int(kv["r"])))
    return names, reads


def _stream_reads(lines: list[str], stream: str) -> list[tuple]:
    """Return [(ctr, max, r), ...] for the named stream, in read order."""
    names, reads = _parse(lines)
    seeds = {s for s, n in names.items() if n == stream}
    return [(c, m, r) for (s, c, m, r) in reads if s in seeds]


def _rust_reads(seed: str, character: str, ascension: int) -> list[str]:
    tmp = tempfile.NamedTemporaryFile(
        encoding="utf-8", mode="w+", suffix=".rusttrace", delete=False
    )
    saved = os.dup(2)
    os.dup2(tmp.fileno(), 2)
    try:
        import sts2_rs  # type: ignore[import-not-found]

        sts2_rs.RunManager(
            string_seed=seed,
            character=character,
            ascension_level=ascension,
            neow_start=True,
            interactive_card_removal=True,
            direct_event_relic_grant=True,
        )
    finally:
        os.fsync(2)
        os.dup2(saved, 2)
        os.close(saved)
    tmp.seek(0)
    return [ln.rstrip("\n") for ln in tmp.readlines()]


def _cs_reads(seed: str, character: str, ascension: int) -> list[str]:
    from sts2_client import Game

    g = Game()
    lines: list[str] = []
    lock = threading.Lock()

    def drain() -> None:
        assert g.proc.stderr is not None
        for line in g.proc.stderr:
            with lock:
                lines.append(line.rstrip())

    threading.Thread(target=drain, daemon=True).start()
    try:
        g.start(character=character, seed=seed, ascension=ascension)
        # let the stderr drain settle
        deadline = time.time() + 5.0
        last, last_n = time.time(), 0
        while time.time() < deadline:
            time.sleep(0.03)
            with lock:
                n = len(lines)
            if n != last_n:
                last_n, last = n, time.time()
            elif time.time() - last >= 0.15:
                break
    finally:
        g.close()
    with lock:
        return list(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run-setup up_front RNG trace diff")
    ap.add_argument("--seed", default="RP0")
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--stream", default="up_front")
    ap.add_argument("--max-rows", type=int, default=60)
    args = ap.parse_args()

    rust = _stream_reads(_rust_reads(args.seed, args.character, args.ascension), args.stream)
    cs = _stream_reads(_cs_reads(args.seed, args.character, args.ascension), args.stream)

    print(f"seed={args.seed!r} stream={args.stream!r}: cs={len(cs)} reads, rs={len(rust)} reads")
    first_div = None
    n = max(len(cs), len(rust))
    for i in range(n):
        c = cs[i] if i < len(cs) else None
        r = rust[i] if i < len(rust) else None
        same = c is not None and r is not None and c[1:] == r[1:]  # compare (max, r)
        if not same and first_div is None:
            first_div = i
        if i < args.max_rows or (first_div is not None and abs(i - first_div) <= 3):
            cstr = f"ctr={c[0]} max={c[1]} r={c[2]}" if c else "—"
            rstr = f"ctr={r[0]} max={r[1]} r={r[2]}" if r else "—"
            mark = "" if same else "  <-- DIVERGE" if i == first_div else "  x"
            print(f"  [{i:3d}] cs[{cstr:28s}] rs[{rstr:28s}]{mark}")
    if first_div is None and len(cs) == len(rust):
        print(f"\n✓ {args.stream} sequences IDENTICAL ({len(cs)} reads)")
    else:
        print(f"\n✗ first divergence at index {first_div} (cs_len={len(cs)} rs_len={len(rust)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
