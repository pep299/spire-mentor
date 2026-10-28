"""`python -m ppo.turnplan.label`: the G0 turn-plan label generator (GH #830).

The deployed pilot (`MctsCombatAgent`, 1000 / 4000 sims, deployed env) plays
every (fight, rep) cell of a fight set to its end. Before the step at the first
player decision of each turn, `turn_plan_labels` enumerates that turn's plans
and prices each one with the deployed search; the result is one labelled
POSITION. Boss and elite turns are always labelled, hallway (`monster`) turns
with probability `--hallway-p`, drawn from `random.Random(cell_seed)`.
Labelling is a pure read (Task 4 pins that), so the pilot plays exactly the
fight it would play unlabelled. That holds only with the env-gated potion
opportunity-cost table off (the G0 env is `{}`): `build_search` advances its
`charge_slots_ctx` state on every build.

    python -m ppo.turnplan.label --set data/combat_bench/ironclad_devpool_v1.jsonl \\
        --exclude data/combat_bench/ironclad_dev_v1.jsonl --reps 2 \\
        --sims-per-plan 500 --hallway-p 0.25 --workers 46 --out <dir>

Processes. Each cell runs in its own spawned process, at most `--workers` at a
time (the `combat_bench.run._CellScheduler` pattern), because a single engine
search can hang where no Python bytecode runs (a Waterfall Giant decision at
4000 sims once ran for over an hour) and a pool would lose that slot for good.
The parent kills a cell still running after `--cell-timeout-s` and books it
as `watchdog`; a process that dies without reporting (segfault, OOM kill) is
booked as `error`. Either way the run goes on.

Output in `<dir>`:

- `header.json`: the args, `sts2_rs.src_hash()`, git head and the pilot spec.
  A resume refuses a header that differs in anything but `IDENTITY_VOLATILE`.
- `shard-<session>-<n>.jsonl.gz`: the positions of ONE cell (the `n`-th cell
  started in session `session`; 0 for the first run, +1 per resume), one line
  per position: `{"key", "rep", "file", "room_type", "turn", "split",
  "label_s", ...turn_plan_labels dict}`. Every line is sync-flushed, so the
  shard of a killed cell reads cleanly up to its last position (`iter_shard`).
  A cell that labels nothing writes no shard.
- `cells.jsonl`: one line per ended cell (`end`, `positions`, `errors`,
  `wall_s`, `shard`), written by the parent. It is the ledger: resume skips the
  cells it lists (`--retry-ends` replays the listed ends), and the last line
  of a cell names the one shard its positions are read from. A shard no ledger
  line names (a cell cut off by a killed run, replayed later) is ignored, so
  `read_positions` never depends on a replay being bit-identical.
- `errors.jsonl`: one line per position whose labelling raised (a PyO3 panic
  is a `BaseException`); the fight continues unlabelled at that turn.
- `progress.log`: positions/s, cells done and running, mean plans/position,
  cap-hit rate overall and by room, ends (watchdog / error included), at
  least every `--progress-every` seconds.

The split is by run FILE (`split_of`), never by position or fight. Capped
positions (`cap_hit`, a partial plan list) are recorded as they are; the
consumers exclude and count them. A `watchdog` / `error` cell keeps the
positions it labelled before it stopped; each one is complete on its own.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
import zlib
from collections.abc import Iterator
from dataclasses import asdict
from multiprocessing import get_context
from multiprocessing.connection import wait as wait_conns
from pathlib import Path
from typing import Any

from ppo.combat_bench import agents
from ppo.combat_bench.run import Caps, build_run, cell_seed, load_set

SCHEMA_ID = "turnplan.labels.v1"
HOLDOUT_MOD = 5
ALWAYS_LABELLED = frozenset({"boss", "elite"})
ROOMS = ("boss", "elite", "monster")
# Ends `--retry-ends` may replay: they depend on the machine, not the fight.
RETRYABLE_ENDS = frozenset({"watchdog", "error"})
# Header fields a resume may change. A cell's labels depend only on (key, rep)
# and the identity fields, so a resume may widen the run (`limit`, `reps`) or
# raise the wall-clock limit.
IDENTITY_VOLATILE = frozenset(
    {
        "argv",
        "workers",
        "host",
        "cpus",
        "git_head",
        "progress_every",
        "limit",
        "n_fights",
        "reps",
        "cell_timeout_s",
    }
)
CHECK_EVERY_S = 1.0
_SHARD = re.compile(r"^shard-(\d+)-(\d+)\.jsonl\.gz$")


def split_of(file: str) -> str:
    """`"holdout"` iff `crc32(file) % 5 == 0`, else `"train"`; by run file, never by position."""
    return "holdout" if zlib.crc32(file.encode()) % HOLDOUT_MOD == 0 else "train"


# ── reading ──────────────────────────────────────────────────────────────


def iter_shard(path: str | Path) -> Iterator[dict[str, Any]]:
    """Every complete position line of one shard, stopping cleanly at a cut stream."""
    with gzip.open(path, "rb") as f:
        try:
            for line in f:
                if not line.endswith(b"\n"):
                    return  # the stream was cut mid-line
                yield json.loads(line)
        except (EOFError, zlib.error, gzip.BadGzipFile, json.JSONDecodeError):
            return


def read_cells(out: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    """The ledger: `{(key, rep): last cell record}` from `cells.jsonl`.

    A torn line (a parent killed mid-write) is skipped, and reading goes on.
    """
    path = Path(out) / "cells.jsonl"
    if not path.exists():
        return {}
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    with open(path, "rb") as f:
        for line in f:
            try:
                c = json.loads(line)
                cells[c["key"], int(c["rep"])] = c
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return cells


def read_positions(out: str | Path) -> Iterator[dict[str, Any]]:
    """Every labelled position of the ledger's cells, each (key, rep, turn) once.

    A cell's positions are read only from the shard its last ledger line
    names; shards of cut-off attempts are ignored.
    """
    out = Path(out)
    for (key, rep), cell in sorted(read_cells(out).items()):
        shard = cell.get("shard")
        if not shard or not (out / shard).exists():
            continue
        seen: set[int] = set()
        for rec in iter_shard(out / shard):
            if (rec["key"], int(rec["rep"])) == (key, rep) and rec["turn"] not in seen:
                seen.add(rec["turn"])
                yield rec


# ── one cell ─────────────────────────────────────────────────────────────


def _view(rm: Any) -> dict[str, Any] | None:
    import sts2_rs

    cs = rm.combat_state()
    return None if cs is None else sts2_rs.state_to_dict(cs)


def label_cell(
    fight: dict[str, Any],
    rep: int,
    *,
    spec: dict[str, Any],
    sims_per_plan: int,
    max_plans: int,
    hallway_p: float,
    caps: Caps,
    emit: Any,
    on_error: Any,
) -> dict[str, Any]:
    """Play cell (`fight`, `rep`) with the pilot, labelling each turn start.

    The process env must already be the spec's (`agents.apply_env`).

    Args:
        fight: A fight-set row.
        rep: The rep id.
        spec: The pilot's `agents.mcts_spec`.
        sims_per_plan: Teacher simulations per plan.
        max_plans: The enumeration cap.
        hallway_p: Probability of labelling a hallway turn.
        caps: Round / step caps; hitting one ends the cell unscored.
        emit: Called with each position record.
        on_error: Called with `{key, rep, turn, error}` when labelling raises.

    Returns:
        The cell's ledger fields: `end`, `positions`, `errors`, `turns`, `steps`.
    """
    seed = cell_seed(fight["key"], rep)
    rm = build_run(fight, seed)
    agent = agents.make_agent(spec, seed)
    rng = random.Random(seed)
    room = fight["room_type"]
    base = {
        "key": fight["key"],
        "rep": rep,
        "file": fight["file"],
        "room_type": room,
    }
    split = split_of(fight["file"])
    last_turn: int | None = None
    positions = errors = steps = 0
    end: str | None = None
    while rm.phase() == "combat" and not rm.is_over():
        view = _view(rm)
        if view is None:
            break
        turn = int(view["turn"])
        if int(view["round"]) > caps.max_rounds:
            end = "round_cap"
            break
        if steps >= caps.max_steps:
            end = "step_cap"
            break
        if view["side"] == "Player" and turn != last_turn:
            last_turn = turn
            if room in ALWAYS_LABELLED or rng.random() < hallway_p:
                t0 = time.perf_counter()
                try:
                    labels = agent.turn_plan_labels(rm, sims_per_plan, max_plans)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as e:  # a PyO3 panic is a BaseException
                    errors += 1
                    on_error({**base, "turn": turn, "error": repr(e)})
                    labels = None
                if labels is not None:
                    dt = round(time.perf_counter() - t0, 4)
                    emit({**base, "turn": turn, "split": split, "label_s": dt, **labels})
                    positions += 1
        if agent.step(rm) is None:
            end = "no_action"
            break
        steps += 1
    if end is None:
        end = "died" if rm.is_over() and not rm.player_won else "won"
    return {
        "end": end,
        "positions": positions,
        "errors": errors,
        "turns": last_turn or 0,
        "steps": steps,
    }


# ── cell processes ───────────────────────────────────────────────────────


def _cell_process(
    cfg: dict[str, Any], fight: dict[str, Any], rep: int, shard: str, conn: Any
) -> None:
    """One cell in its own process: `("pos" | "err" | "done", rec)` down `conn`."""
    # Before any `sts2_rs` search: some engine seams latch on first use.
    agents.apply_env(cfg["spec"]["env"])
    gz: Any = None

    def emit(rec: dict[str, Any]) -> None:
        nonlocal gz
        if gz is None:
            gz = gzip.open(Path(cfg["out"]) / shard, "wt")
        gz.write(json.dumps(rec) + "\n")
        gz.flush()  # Z_SYNC_FLUSH: readable even if this process is killed
        stats = {k: rec[k] for k in ("room_type", "n_plans", "cap_hit", "label_s")}
        conn.send(("pos", stats))

    try:
        rec = label_cell(
            fight,
            rep,
            spec=cfg["spec"],
            sims_per_plan=cfg["sims_per_plan"],
            max_plans=cfg["max_plans"],
            hallway_p=cfg["hallway_p"],
            caps=Caps(**cfg["caps"]),
            emit=emit,
            on_error=lambda e: conn.send(("err", e)),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # a PyO3 panic is a BaseException
        rec = {"end": "error", "error": repr(e)}
    finally:
        if gz is not None:
            gz.close()
    conn.send(("done", rec))
    conn.close()


class _Running:
    """A started cell: its process, start time, and what it reported so far."""

    def __init__(self, proc: Any, key: str, rep: int, room: str, shard: Path) -> None:
        self.proc, self.key, self.rep, self.room, self.shard = proc, key, rep, room, shard
        self.t0 = time.monotonic()
        self.positions = self.errors = 0

    def ledger(self, rec: dict[str, Any]) -> dict[str, Any]:
        out = {"key": self.key, "rep": self.rep, "room_type": self.room}
        out |= {"positions": self.positions, "errors": self.errors} | rec
        out["wall_s"] = round(time.monotonic() - self.t0, 2)
        # Named whenever it exists: a killed cell may have written a line it
        # never got to report.
        out["shard"] = self.shard.name if self.shard.exists() else None
        return out


# ── progress ─────────────────────────────────────────────────────────────


class _Progress:
    """Aggregates cell reports; a line to stderr and `progress.log`."""

    def __init__(self, path: Path, total: int, done0: int) -> None:
        self.path, self.total, self.done0 = path, total, done0
        self.t0 = self.t_last = time.time()
        self.cells = self.positions = self.plans = self.caps = self.errors = 0
        self.n_last = 0
        self.label_s = 0.0
        self.by_room: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        self.ends: collections.Counter[str] = collections.Counter()

    def emit(self, msg: str) -> None:
        line = f"[turnplan.label] {msg}"
        print(line, file=sys.stderr, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

    def position(self, p: dict[str, Any]) -> None:
        self.positions += 1
        self.plans += p["n_plans"]
        self.caps += bool(p["cap_hit"])
        self.label_s += p["label_s"]
        room = self.by_room[p["room_type"]]
        room[0] += 1
        room[1] += bool(p["cap_hit"])

    def cell(self, rec: dict[str, Any]) -> None:
        self.cells += 1
        self.ends[rec["end"]] += 1

    def line(self, running: int) -> None:
        now = time.time()
        dt, dt_last = now - self.t0, now - self.t_last
        rate = self.positions / dt if dt > 0 else 0.0
        rate_last = (self.positions - self.n_last) / dt_last if dt_last > 0 else 0.0
        self.t_last, self.n_last = now, self.positions
        n = max(self.positions, 1)
        rooms = {r: f"{c}/{k}" for r, (k, c) in sorted(self.by_room.items())}  # capped / positions
        self.emit(
            f"{dt / 60:.1f} min: {self.positions} positions ({rate:.3f} positions/s, "
            f"{rate_last:.3f} last), cells {self.done0 + self.cells}/{self.total} "
            f"({running} running), mean plans/position {self.plans / n:.1f}, "
            f"cap-hit {self.caps / n:.1%} (by room capped/positions {rooms}), "
            f"label s/position {self.label_s / n:.2f}, position errors {self.errors}, "
            f"ends {dict(sorted(self.ends.items()))}"
        )


# ── header, main ─────────────────────────────────────────────────────────


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m ppo.turnplan.label",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--set", required=True, help="fight-set JSONL (combat_bench SCHEMA)")
    ap.add_argument("--exclude", required=True, help="fight set whose keys are never labelled")
    ap.add_argument("--out", required=True, help="output dir; resumed when it has a header")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--sims-per-plan", type=int, default=500)
    ap.add_argument("--max-plans", type=int, default=512)
    ap.add_argument("--hallway-p", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N fights")
    ap.add_argument("--order-seed", type=int, default=0, help="cell start order")
    ap.add_argument("--progress-every", type=float, default=60.0, help="seconds")
    ap.add_argument(
        "--cell-timeout-s",
        type=float,
        default=3600.0,
        help="wall clock per cell; the parent kills it and books it as watchdog",
    )
    ap.add_argument(
        "--retry-ends",
        default=None,
        help=f"comma list of ledger ends to replay on resume ({','.join(sorted(RETRYABLE_ENDS))})",
    )
    ap.add_argument("--pilot-sims", type=int, default=1000, help="deployed: 1000 (tests only)")
    ap.add_argument("--pilot-boss-sims", type=int, default=4000, help="deployed: 4000")
    ap.add_argument("--max-rounds", type=int, default=Caps.max_rounds)
    ap.add_argument("--max-steps", type=int, default=Caps.max_steps)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """`python -m ppo.turnplan.label ...`; returns the exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _parse(argv)
    out = Path(args.out)
    retry = set(args.retry_ends.split(",")) if args.retry_ends else set()
    if (
        not 0.0 <= args.hallway_p <= 1.0
        or args.reps < 1
        or args.workers < 1
        or args.cell_timeout_s <= 0
        or retry - RETRYABLE_ENDS
    ):
        print(
            "[turnplan.label] need 0 <= hallway-p <= 1, reps >= 1, workers >= 1, "
            f"cell-timeout-s > 0, retry-ends within {sorted(RETRYABLE_ENDS)}",
            file=sys.stderr,
        )
        return 2
    excluded = {f["key"] for f in load_set(args.exclude)}
    fights = [f for f in load_set(args.set) if f["key"] not in excluded]
    if args.limit is not None:
        fights = fights[: args.limit]
    bad_rooms = {f["room_type"] for f in fights} - set(ROOMS)
    if bad_rooms:
        print(f"[turnplan.label] unknown room types {sorted(bad_rooms)}", file=sys.stderr)
        return 2

    import sts2_rs

    spec = agents.mcts_spec(args.pilot_sims, args.pilot_boss_sims, agents.build_env({}))
    caps = Caps(args.max_rounds, args.max_steps)
    header = json.loads(
        json.dumps(
            {
                "schema": SCHEMA_ID,
                "set": args.set,
                "set_sha256": _sha256(args.set),
                "exclude": args.exclude,
                "exclude_sha256": _sha256(args.exclude),
                "n_fights": len(fights),
                "limit": args.limit,
                "reps": args.reps,
                "sims_per_plan": args.sims_per_plan,
                "max_plans": args.max_plans,
                "hallway_p": args.hallway_p,
                "order_seed": args.order_seed,
                "caps": asdict(caps),
                "cell_timeout_s": args.cell_timeout_s,
                "spec": spec,
                "src_hash": sts2_rs.src_hash(),
                "git_head": _git_head(),
                "workers": args.workers,
                "progress_every": args.progress_every,
                "python": platform.python_version(),
                "host": socket.gethostname(),
                "cpus": os.cpu_count(),
                "argv": argv,
            }
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    head_path = out / "header.json"
    if head_path.exists():
        old = json.loads(head_path.read_text())
        diff = sorted(
            k for k in (set(old) | set(header)) - IDENTITY_VOLATILE if old.get(k) != header.get(k)
        )
        if diff:
            print(
                f"[turnplan.label] refusing to resume {out}: the header differs in {diff}",
                file=sys.stderr,
            )
            for k in diff:
                print(f"  {k}: file={old.get(k)!r}\n  {k}:  now={header.get(k)!r}", file=sys.stderr)
            return 2
    else:
        head_path.write_text(json.dumps(header, indent=2) + "\n")
    inherited = sorted(k for k in os.environ if agents.is_engine_env(k))
    if inherited:
        print(
            f"[turnplan.label] WARNING: cell processes scrub the inherited engine env {inherited}",
            file=sys.stderr,
        )

    done = {c for c, rec in read_cells(out).items() if rec.get("end") not in retry}
    cells = [(f["key"], r) for f in fights for r in range(args.reps)]
    random.Random(args.order_seed).shuffle(cells)
    by_key = {f["key"]: f for f in fights}
    pending = [(k, r) for k, r in cells if (k, r) not in done]
    sessions = [int(m[1]) for p in out.iterdir() if (m := _SHARD.match(p.name))]
    session = max(sessions, default=-1) + 1
    progress = _Progress(out / "progress.log", len(cells), len(cells) - len(pending))
    progress.emit(
        f"{len(fights)} fights x {args.reps} reps = {len(cells)} cells; "
        f"{len(cells) - len(pending)} done, {len(pending)} to play; session {session}; "
        f"sims/plan {args.sims_per_plan}, max plans {args.max_plans}, "
        f"hallway p {args.hallway_p}, cell timeout {args.cell_timeout_s:g} s"
    )
    if not pending:
        return 0

    cfg = {
        "spec": spec,
        "out": str(out),
        "sims_per_plan": args.sims_per_plan,
        "max_plans": args.max_plans,
        "hallway_p": args.hallway_p,
        "caps": asdict(caps),
    }
    ctx = get_context("spawn")
    running: dict[Any, _Running] = {}
    todo = iter(enumerate(pending))
    ledger_path = out / "cells.jsonl"
    # A parent killed mid-write leaves a torn last line; end it, so the next
    # record is not glued onto it.
    if ledger_path.exists() and not ledger_path.read_bytes().endswith(b"\n"):
        with open(ledger_path, "ab") as f:
            f.write(b"\n")

    try:
        with (
            open(ledger_path, "a", encoding="utf-8") as ledger,
            open(out / "errors.jsonl", "a", encoding="utf-8") as errors,
        ):

            def finish(conn: Any, rec: dict[str, Any]) -> None:
                cell = running.pop(conn)
                conn.close()
                cell.proc.join(timeout=10)
                row = cell.ledger(rec)
                ledger.write(json.dumps(row) + "\n")
                ledger.flush()
                progress.cell(row)

            while True:
                while len(running) < args.workers:
                    nxt = next(todo, None)
                    if nxt is None:
                        break
                    n, (key, rep) = nxt
                    shard = out / f"shard-{session}-{n}.jsonl.gz"
                    recv, send = ctx.Pipe(duplex=False)
                    proc = ctx.Process(
                        target=_cell_process,
                        args=(cfg, by_key[key], rep, shard.name, send),
                        daemon=True,
                    )
                    proc.start()
                    send.close()
                    running[recv] = _Running(proc, key, rep, by_key[key]["room_type"], shard)
                if not running:
                    break
                for conn in wait_conns(list(running), timeout=CHECK_EVERY_S):
                    cell = running[conn]
                    while True:
                        try:
                            kind, rec = conn.recv()
                        except EOFError:
                            cell.proc.join(timeout=10)
                            code = cell.proc.exitcode
                            finish(conn, {"end": "error", "error": f"cell process died ({code})"})
                            break
                        if kind == "pos":
                            cell.positions += 1
                            progress.position(rec)
                        elif kind == "err":
                            cell.errors += 1
                            progress.errors += 1
                            errors.write(json.dumps(rec) + "\n")
                            errors.flush()
                        else:
                            finish(conn, rec)
                            break
                        if not conn.poll():
                            break
                now = time.monotonic()
                for conn, cell in list(running.items()):
                    if now - cell.t0 > args.cell_timeout_s:
                        cell.proc.kill()
                        cell.proc.join()
                        finish(
                            conn,
                            {
                                "end": "watchdog",
                                "error": f"killed after {args.cell_timeout_s:g} s wall clock",
                            },
                        )
                if time.time() - progress.t_last >= args.progress_every:
                    progress.line(len(running))
    finally:
        for cell in running.values():
            cell.proc.kill()
            cell.proc.join()
    progress.line(0)
    progress.emit(f"done -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
