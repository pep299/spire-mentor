"""`python -m ppo.turnplan.root_flip`: offline flip rate of the root filter (GH #830).

The kill gate of the root-filter screen
(`docs/superpowers/specs/2026-10-22-turnplan-pro-root-filter-screen-design.md`):
B0 (the deployed `MctsCombatAgent`, 1000 / 4000 sims, env `{}`) plays every
fight of a set once (rep 0's cell seed). Before each of its decisions the
root filter is evaluated at every `--k` (`sts2_rs.TurnPlanRootFilter`, the
same Rust code the search knob runs), and the decision is booked as

- `fallback` (`forced` / `cap` / `empty`): the filtered search would search
  the full root;
- `pruned`: B0's pick is not among the allowed root actions, so the arm
  cannot play it here (a *flip*);
- `kept` otherwise.

    python -m ppo.turnplan.root_flip --set <fights.jsonl> --weights <root_filter.json> \\
        --out <dir> [--k 10 6] [--workers 44]

One process per fight (a panic or hang costs that fight; `--fight-timeout-s`),
one line per fight in `fights.jsonl` with its decision rows, progress lines in
`progress.log`; a rerun skips the fights already in `fights.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import get_context
from multiprocessing.connection import wait as wait_conns
from pathlib import Path
from typing import Any

from ppo.combat_bench import agents
from ppo.combat_bench.run import Caps, build_run, cell_seed, load_set


def book(filt: dict[str, Any] | None, chosen: dict[str, Any] | None) -> str:
    """`fallback:<reason>`, `pruned` or `kept` for one decision at one K."""
    if filt is None:
        return "fallback:offturn"
    if filt["fallback"] is not None:
        return f"fallback:{filt['fallback']}"
    allowed = [a for a, _ in filt["allowed"]]
    return "kept" if chosen in allowed else "pruned"


def fight_rows(fight: dict[str, Any], weights: str, ks: list[int], caps: Caps) -> dict[str, Any]:
    """Play `fight` with B0 and book every decision at every K."""
    import sts2_rs

    spec = agents.mcts_spec(1000, 4000, agents.build_env({}))
    agents.apply_env(spec["env"])
    seed = cell_seed(fight["key"], 0)
    rm = build_run(fight, seed)
    agent = agents.make_agent(spec, seed)
    filt = sts2_rs.TurnPlanRootFilter(weights)
    rows = []
    steps = 0
    while rm.phase() == "combat" and not rm.is_over() and steps < caps.max_steps:
        cs = rm.combat_state()
        view = sts2_rs.state_to_dict(cs) if cs is not None else None
        if view is None or int(view["round"]) > caps.max_rounds:
            break
        fs = {k: filt.filter(rm, k) for k in ks}
        t0 = time.perf_counter()
        rec = agent.step(rm)
        dt = time.perf_counter() - t0
        if rec is None:
            break
        steps += 1
        search = rec.get("combat_search") or {}
        chosen = search.get("recommended")
        f0 = fs[ks[0]]
        rows.append(
            {
                "i": steps - 1,
                "round": int(view["round"]),
                "legal_n": None if f0 is None else f0["legal_n"],
                "plans_n": None if f0 is None else f0["plans_n"],
                "b0_ms": round(dt * 1000, 1),
                "filter_ms": {str(k): round(fs[k]["filter_us"] / 1000, 2) for k in ks if fs[k]},
                "book": {str(k): book(fs[k], chosen) for k in ks},
                "allowed_n": {
                    str(k): len(fs[k]["allowed"]) for k in ks if fs[k] and fs[k]["fallback"] is None
                },
            }
        )
    return {"end": "won" if rm.player_won else ("died" if rm.is_over() else "cut"), "rows": rows}


def _proc(fight: dict[str, Any], weights: str, ks: list[int], conn: Any) -> None:
    try:
        out = fight_rows(fight, weights, ks, Caps())
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        out = {"error": repr(e)[:300]}
    conn.send(out)
    conn.close()


def summarize(lines: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    """Flip and fallback rates per K, by room and for the fight's first decision."""
    out: dict[str, Any] = {
        "fights": len(lines),
        "errors": sum("error" in ln for ln in lines),
    }
    for k in map(str, ks):
        cells: dict[str, Counter[str]] = {}
        for ln in lines:
            for j, r in enumerate(ln.get("rows", [])):
                for sub in ("all", ln["room_type"], *(["entry"] if j == 0 else [])):
                    cells.setdefault(sub, Counter())[r["book"][k]] += 1
        out[f"K={k}"] = {
            sub: {
                "decisions": sum(c.values()),
                "pruned": c["pruned"],
                "flip_rate": c["pruned"] / max(sum(c.values()), 1),
                "flip_rate_of_filtered": c["pruned"] / max(c["pruned"] + c["kept"], 1),
                **{name: n for name, n in sorted(c.items()) if name.startswith("fallback")},
            }
            for sub, c in sorted(cells.items())
        }
    b0 = [r["b0_ms"] for ln in lines for r in ln.get("rows", [])]
    fm = {
        k: [r["filter_ms"].get(str(k), 0.0) for ln in lines for r in ln.get("rows", [])] for k in ks
    }
    out["wall"] = {
        "b0_ms_per_decision": sum(b0) / max(len(b0), 1),
        **{f"filter_ms_per_decision_K={k}": sum(v) / max(len(v), 1) for k, v in fm.items()},
        **{f"wall_ratio_K={k}": (sum(b0) + sum(v)) / max(sum(b0), 1e-9) for k, v in fm.items()},
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.turnplan.root_flip", description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 6])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--fight-timeout-s", type=float, default=2400.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = str(Path(args.weights).resolve())
    fights = load_set(args.set)[: args.limit]
    ledger = out / "fights.jsonl"
    done = set()
    if ledger.exists():
        for line in ledger.open():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = iter([f for f in fights if f["key"] not in done])
    log = (out / "progress.log").open("a")

    def say(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [root_flip] {msg}"
        print(line, file=sys.stderr, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"{len(fights)} fights, {len(done)} done; K {args.k}; weights {weights}")
    ctx = get_context("spawn")
    running: dict[Any, tuple[Any, dict[str, Any], float]] = {}
    n_done = 0
    t_last = time.time()
    with ledger.open("a") as led:

        def finish(conn: Any, res: dict[str, Any]) -> None:
            nonlocal n_done
            proc, fight, start = running.pop(conn)
            conn.close()
            proc.join(timeout=10)
            line = {"key": fight["key"], "room_type": fight["room_type"]}
            led.write(json.dumps(line | {"wall_s": round(time.time() - start, 1)} | res) + "\n")
            led.flush()
            n_done += 1

        while True:
            while len(running) < args.workers:
                fight = next(todo, None)
                if fight is None:
                    break
                recv, send = ctx.Pipe(duplex=False)
                proc = ctx.Process(target=_proc, args=(fight, weights, args.k, send), daemon=True)
                proc.start()
                send.close()
                running[recv] = (proc, fight, time.time())
            if not running:
                break
            for conn in wait_conns(list(running), timeout=1.0):
                try:
                    res = conn.recv()
                except EOFError:
                    res = {"error": f"process died ({running[conn][0].exitcode})"}
                finish(conn, res)
            now = time.time()
            for conn, (proc, _, start) in list(running.items()):
                if now - start > args.fight_timeout_s:
                    proc.kill()
                    proc.join()
                    finish(conn, {"error": "watchdog"})
            if now - t_last > 60:
                t_last = now
                say(f"{n_done} fights done this session, {len(running)} running")
    lines = [json.loads(line) for line in ledger.open()]
    summary = summarize(lines, args.k)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    say(f"done: {json.dumps({k: summary[k] for k in summary if k.startswith('K=')})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
