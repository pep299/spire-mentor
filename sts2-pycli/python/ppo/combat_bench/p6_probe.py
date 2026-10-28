"""`python -m ppo.combat_bench.p6_probe`: the G-B oracle-planner probe P6-0 (GH #830).

Deep bets plan (`deepbets/plan.md`, Step 2, "G-B, planner necessary
condition"). On 300 contested pilot decisions (150 boss, 150 elite) it asks
whether the oracle turn planner (`ppo.combat_bench.oracle`) would play
differently from the deployed pilot, and whether its verdict is stable,
honest and affordable:

* the planner's **near-best class**: the first moves whose best plan is
  within one paired SE of the top (CRN seeds 1..n);
* the pilot's **modal first move** over 8 seeds (`MctsCombatAgent`, deployed
  budget 1000 / 4000 sims) and whether it lies outside that class;
* **split-half class agreement**: the class from seeds 1..n/2 against
  n/2+1..n;
* **honesty**: identical planner output on 5 `sample_world` variants of the
  first 20 positions;
* the **fallback rate** (more than `leaf_cap` playout leaves) and leaves /
  playouts per planning point (the G-C cost input).

Kill item 6 if any holds: under 10% of positions with the modal pilot move
outside the class; split-half agreement under 60% (after one rerun at
n=256); an honesty failure (fix first); fallback above 10%.

Three subcommands:

    positions  pick the position set from the dev panel (procedure below)
    probe      run P6-0 on it (one JSON line per position, resumable)
    report     aggregate a probe file against the kill bars

**Position procedure** (`positions`). The plan's design docs draw the
positions from pilot trajectories: contested decisions (p_win in
[0.05, 0.95] or SD(hp_end) >= 5 over 32 belief playouts) whose turn DAG has
depth >= 2. Here the trajectories are the deployed pilot's on the dev panel
fights:

1. Per room, the panel's fights sorted by key and shuffled with
   `random.Random(f"{order_seed}:{room}")`.
2. Each fight is rebuilt at cell seed `cell_seed(key, rep)` (default rep 49:
   outside the screening reps 0..39 and the panel's selection reps 40..47)
   and played to its end by `MctsCombatAgent(seed=cell seed, 1000, 4000)`,
   capped like the bench (30 rounds, 3000 steps).
3. Its searched decisions (the pilot ran a search: more than one option) are
   shuffled with `random.Random(f"{order_seed}:{key}")` and screened in that
   order: `det_dag_depth(belief_view) >= 2` (deterministic edges only, merged
   on `observation_hash`), then 32 `values_many(mode="belief")` playouts with
   seeds `stable_u64(key, t, "screen", j)`; contested as above. At most
   `cap` positions per fight are accepted (boss 1, elite 2).
4. Fights are consumed in their shuffled order until the room's quota is
   met; the positions are the accepted ones of that prefix, in order.

Every seed is derived from the fight key and the step, so the set is the
same on every run of the same wheel and playout net.
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from collections.abc import Iterable, Iterator
from multiprocessing import get_context
from multiprocessing.connection import wait as wait_conns
from typing import Any

from ppo.combat_bench import agents
from ppo.combat_bench.oracle import (
    K_CHANCE,
    LEAF_CAP,
    MASK63,
    N_DEFAULT,
    Planner,
    det_dag_depth,
    legal_match,
    move_key,
    near_best_class,
    rm_to_combat,
    stable_u64,
    to_view_action,
)
from ppo.combat_bench.run import Caps, _searched, build_run, cell_seed

SCHEMA_POS = "combat_bench.p6_positions.v1"
SCHEMA_PROBE = "combat_bench.p6_probe.v1"
ORDER_SEED = 830
POSITION_REP = 49
SCREEN_N = 32
PILOT_SEEDS = 8
HONESTY_POSITIONS = 20
HONESTY_WORLDS = 5
CAP_PER_FIGHT = {"boss": 1, "elite": 2}
TASK_TIMEOUT_S = 1800

# Pre-registered bars (plan.md, G-B).
BAR_DISAGREE = 0.10
BAR_SPLIT_HALF = 0.60
BAR_FALLBACK = 0.10


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(path: str | None, msg: str) -> None:
    line = f"{_now()} {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def keystr(key: tuple) -> str:
    """A move key as a JSON-friendly string."""
    if key[0] == "end_turn":
        return "end_turn"
    if key[0] == "obs":
        return f"obs:{key[1]}"
    return "id:" + hashlib.blake2b(repr(key[1]).encode(), digest_size=6).hexdigest()


def wheel_info() -> dict[str, Any]:
    import sts2_rs

    info = sts2_rs.build_info()
    return {
        "src_hash": info.get("src_hash"),
        "build_git": info.get("build_git"),
        "pgo": info.get("pgo"),
        "python": sys.version.split()[0],
    }


def load_fights(path: str) -> tuple[dict[str, dict[str, Any]], str]:
    raw = open(path, "rb").read()
    fights = {}
    for line in raw.decode().splitlines():
        if line.strip():
            f = json.loads(line)
            fights[f["key"]] = f
    return fights, hashlib.sha256(raw).hexdigest()


def rebuild(fight: dict[str, Any], pos: dict[str, Any]) -> Any:
    """The live `RunManager` at position `pos` (replayed, hash-checked)."""
    rm = build_run(fight, int(pos["traj_seed"]))
    for a in pos["actions"]:
        rm.take_action(a)
    cs = rm.combat_state()
    if cs is None or str(cs.ordered_hash()) != str(pos["ordered_hash"]):
        raise RuntimeError(f"{pos['pos_id']}: replay does not reach the recorded position")
    return rm


# ── a process per task, with a wall-clock kill ───────────────────────────


def _task_process(fn_name: str, payload: Any, env: dict[str, str], conn: Any) -> None:
    os.environ["RAYON_NUM_THREADS"] = "1"
    agents.apply_env(env)
    try:
        out = globals()[fn_name](payload)
    except BaseException as e:  # a PyO3 panic is a BaseException
        out = {"error": repr(e), "traceback": traceback.format_exc()}
    conn.send(out)
    conn.close()


class Scheduler:
    """At most `procs` task processes at once; a task over `timeout_s` is killed."""

    def __init__(self, fn_name: str, env: dict[str, str], procs: int, timeout_s: int) -> None:
        self.fn_name = fn_name
        self.env = env
        self.procs = max(1, procs)
        self.timeout_s = timeout_s
        self.ctx = get_context("spawn")
        self.running: dict[Any, tuple[Any, float, Any]] = {}

    def run(self, tasks: Iterable[tuple[Any, Any]]) -> Iterator[tuple[Any, Any]]:
        todo = iter(tasks)
        exhausted = False
        while True:
            while not exhausted and len(self.running) < self.procs:
                nxt = next(todo, None)
                if nxt is None:
                    exhausted = True
                    break
                tag, payload = nxt
                recv, send = self.ctx.Pipe(duplex=False)
                proc = self.ctx.Process(
                    target=_task_process, args=(self.fn_name, payload, self.env, send), daemon=True
                )
                proc.start()
                send.close()
                self.running[recv] = (proc, time.monotonic(), tag)
            if not self.running:
                return
            for recv in wait_conns(list(self.running), timeout=5.0):
                proc, _, tag = self.running.pop(recv)
                try:
                    out = recv.recv()
                except EOFError:
                    out = {"error": f"task process died (exit code {proc.exitcode})"}
                recv.close()
                proc.join()
                yield tag, out
            now = time.monotonic()
            for recv, (proc, t0, tag) in list(self.running.items()):
                if now - t0 > self.timeout_s:
                    proc.kill()
                    proc.join()
                    recv.close()
                    del self.running[recv]
                    yield tag, {"error": f"timeout: killed after {self.timeout_s} s"}

    def close(self) -> None:
        for recv, (proc, _, _) in list(self.running.items()):
            proc.kill()
            proc.join()
            recv.close()
        self.running.clear()


# ── positions ────────────────────────────────────────────────────────────


def fight_positions(p: dict[str, Any]) -> dict[str, Any]:
    """Play the pilot on one fight and screen its decisions (a task)."""
    import sts2_rs

    w0, c0 = time.perf_counter(), time.process_time()
    fight, rep, cap = p["fight"], p["rep"], p["cap"]
    key = fight["key"]
    caps = Caps()
    seed = cell_seed(key, rep)
    rm = build_run(fight, seed)
    agent = sts2_rs.MctsCombatAgent(seed=seed, sims=p["sims"], boss_sims=p["boss_sims"])
    states: list[Any] = []
    actions: list[dict[str, Any]] = []
    searched: list[int] = []
    end = None
    while rm.phase() == "combat" and not rm.is_over():
        cs = rm.combat_state()
        if cs is None:
            break
        if int(sts2_rs.state_to_dict(cs)["round"]) > caps.max_rounds:
            end = "round_cap"
            break
        if len(actions) >= caps.max_steps:
            end = "step_cap"
            break
        rec = agent.step(rm)
        if rec is None:
            end = "no_action"
            break
        states.append(cs)
        if _searched(rec):
            searched.append(len(actions))
        actions.append(rec["action"])
    if end is None:
        end = "died" if rm.is_over() and not rm.player_won else "won"
    traj_cpu = time.process_time() - c0
    order = list(searched)
    random.Random(f"{p['order_seed']}:{key}").shuffle(order)
    accepted: list[dict[str, Any]] = []
    screened = rej_depth = rej_contested = 0
    for t in order:
        if len(accepted) >= cap:
            break
        cs = states[t]
        screened += 1
        view = cs.belief_view(stable_u64(key, t, "screen-view") & MASK63)
        depth = det_dag_depth(view)
        if depth < 2:
            rej_depth += 1
            continue
        seeds = [stable_u64(key, t, "screen", j) for j in range(SCREEN_N)]
        vals = sts2_rs.values_many([cs] * SCREEN_N, seeds, mode="belief")
        p_win = sum(1 for v in vals if v["won"]) / SCREEN_N
        sd_hp = statistics.pstdev([v["hp"] for v in vals])
        if not (0.05 <= p_win <= 0.95 or sd_hp >= 5):
            rej_contested += 1
            continue
        d = sts2_rs.state_to_dict(cs)
        accepted.append(
            {
                "key": key,
                "room": fight["room_type"],
                "encounter": fight["encounter"],
                "rep": rep,
                "traj_seed": seed,
                "t": t,
                "actions": actions[:t],
                "ordered_hash": str(cs.ordered_hash()),
                "round": int(d["round"]),
                "n_legal": len(cs.legal_actions()),
                "pending_choice": cs.pending_choice() is not None,
                "screen": {
                    "p_win32": p_win,
                    "sd_hp32": round(sd_hp, 3),
                    "mean_v32": round(statistics.fmean(v["search_value"] for v in vals), 5),
                    "det_depth": depth,
                },
            }
        )
    return {
        "key": key,
        "room": fight["room_type"],
        "end": end,
        "steps": len(actions),
        "searched": len(searched),
        "screened": screened,
        "rej_depth": rej_depth,
        "rej_contested": rej_contested,
        "accepted": accepted,
        "traj_cpu_s": round(traj_cpu, 3),
        "cpu_s": round(time.process_time() - c0, 3),
        "wall_s": round(time.perf_counter() - w0, 3),
    }


def cmd_positions(args: argparse.Namespace) -> int:
    fights, set_sha = load_fights(args.set)
    env = agents.build_env({}, rollout_net=args.rollout_net)
    agents.apply_env(env)
    plog = args.progress_log
    quota = {"boss": args.boss, "elite": args.elite}
    by_room: dict[str, list[dict[str, Any]]] = {}
    fight_log: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for room in ("boss", "elite"):
        if quota[room] <= 0:
            by_room[room] = []
            continue
        pool = sorted(
            (f for f in fights.values() if f["room_type"] == room), key=lambda f: f["key"]
        )
        random.Random(f"{args.order_seed}:{room}").shuffle(pool)
        cap = CAP_PER_FIGHT[room]

        def tasks(pool: list[dict[str, Any]] = pool, cap: int = cap) -> Iterator[tuple[int, Any]]:
            for i, f in enumerate(pool):
                yield (
                    i,
                    {
                        "fight": f,
                        "rep": args.rep,
                        "cap": cap,
                        "order_seed": args.order_seed,
                        "sims": args.sims,
                        "boss_sims": args.boss_sims,
                    },
                )

        sched = Scheduler("fight_positions", env, args.procs, args.timeout_s)
        results: dict[int, dict[str, Any]] = {}
        nxt = 0
        accepted: list[dict[str, Any]] = []
        log(plog, f"positions {room}: quota {quota[room]}, {len(pool)} fights, cap {cap}/fight")
        try:
            for i, out in sched.run(tasks()):
                results[i] = out
                if "error" in out:
                    log(plog, f"  {room} fight #{i} {pool[i]['key']} ERROR {out['error']}")
                while nxt in results and len(accepted) < quota[room]:
                    r = results[nxt]
                    rec = {k: v for k, v in r.items() if k != "accepted"} | {"order": nxt}
                    rec["n_accepted"] = len(r.get("accepted", []))
                    fight_log.append(rec)
                    for pos in r.get("accepted", []):
                        if len(accepted) < quota[room]:
                            accepted.append(pos | {"fight_order": nxt})
                    nxt += 1
                log(
                    plog,
                    f"  {room} done={len(results)} prefix={nxt} accepted={len(accepted)}/{quota[room]} "
                    f"last: {out.get('key')} steps={out.get('steps')} acc={len(out.get('accepted', []))} "
                    f"cpu={out.get('cpu_s')}s",
                )
                if len(accepted) >= quota[room]:
                    break
        finally:
            sched.close()
        by_room[room] = accepted
        if len(accepted) < quota[room]:
            log(plog, f"  {room}: SHORTFALL {len(accepted)}/{quota[room]} after every fight")
    # Interleave boss / elite so any prefix of the file is balanced.
    out_rows: list[dict[str, Any]] = []
    b, e = by_room.get("boss", []), by_room.get("elite", [])
    for i in range(max(len(b), len(e))):
        for room, lst in (("boss", b), ("elite", e)):
            if i < len(lst):
                out_rows.append(lst[i] | {"pos_id": f"{room[0]}{i:03d}"})
    header = {
        "schema": SCHEMA_POS,
        "created": _now(),
        "set": os.path.abspath(args.set),
        "set_sha256": set_sha,
        "procedure": {
            "order_seed": args.order_seed,
            "rep": args.rep,
            "cap_per_fight": CAP_PER_FIGHT,
            "quota": quota,
            "pilot": {"sims": args.sims, "boss_sims": args.boss_sims},
            "screen": {
                "playouts": SCREEN_N,
                "mode": "belief",
                "contested": "0.05 <= p_win <= 0.95 or sd(hp_end) >= 5",
                "det_depth_min": 2,
            },
        },
        "env": env,
        "wheel": wheel_info(),
        "fights": fight_log,
        "wall_s": round(time.perf_counter() - t_start, 1),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"header": header}) + "\n")
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    played = len(fight_log)
    log(
        plog,
        f"positions written: {args.out} boss={len(b)} elite={len(e)} from {played} fights "
        f"({sum(f.get('screened', 0) for f in fight_log)} screened, "
        f"{sum(f.get('rej_depth', 0) for f in fight_log)} depth<2, "
        f"{sum(f.get('rej_contested', 0) for f in fight_log)} uncontested) "
        f"in {header['wall_s']} s",
    )
    return 0


# ── the probe ────────────────────────────────────────────────────────────


def plan_signature(plan: Any) -> Any:
    """Everything the planner outputs, for the honesty comparison."""
    if plan.declined is not None:
        return ("declined", plan.declined, plan.stats["nodes"])
    rv = plan.root_values()
    cls, stats, top = near_best_class(rv)
    return (
        plan.leaf_counts(),
        [mv.key for mv, _ in rv],
        [v.tolist() for _, v in rv],
        sorted(map(repr, cls)),
        repr(top),
        [repr(s.action) for s in plan.best_line()],
    )


def _class_summary(rv: list, cls: list, stats: dict, top: tuple) -> dict[str, Any]:
    return {
        "top": keystr(top),
        "class": sorted(keystr(k) for k in cls),
        "moves": [
            {
                "key": keystr(mv.key),
                "label": mv.label,
                "kind": mv.kind,
                "mean": round(stats[mv.key]["mean"], 6),
                "gap": round(stats[mv.key]["gap"], 6),
                "se": round(stats[mv.key]["se"], 6),
                "in_class": mv.key in cls,
            }
            for mv, _ in rv
        ],
    }


def probe_position(p: dict[str, Any]) -> dict[str, Any]:
    """P6-0 at one position (a task)."""
    import sts2_rs

    pos, fight = p["pos"], p["fight"]
    n = p["n"]
    out: dict[str, Any] = {
        "pos_id": pos["pos_id"],
        "key": pos["key"],
        "room": pos["room"],
        "encounter": pos["encounter"],
        "t": pos["t"],
        "round": pos["round"],
        "n": n,
        "load1": round(os.getloadavg()[0], 1),
    }
    rm = rebuild(fight, pos)
    cs = rm.combat_state()
    plan_seed = stable_u64(pos["key"], pos["t"], "plan") & MASK63
    planner = Planner(n=n, k_chance=p["k_chance"], leaf_cap=p["leaf_cap"])
    w0, c0 = time.perf_counter(), time.process_time()
    plan = planner.plan(cs, plan_seed)
    out["plan_wall_s"] = round(time.perf_counter() - w0, 3)
    out["plan_cpu_s"] = round(time.process_time() - c0, 3)
    out["fallback"] = plan.declined is not None
    out["decline"] = plan.declined
    out["stats"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in plan.stats.items()}
    out["leaves"] = plan.leaf_counts()
    out["playout_leaves"] = plan.playout_leaves()
    out["playouts"] = plan.stats.get("playouts", 0)
    b = plan.root_state
    cls_keys: set[tuple] = set()
    if plan.declined is None:
        rv = plan.root_values()
        cls, stats, top = near_best_class(rv)
        cls_keys = set(cls)
        out["n_moves"] = len(rv)
        out |= _class_summary(rv, cls, stats, top)
        halves = {}
        h = n // 2
        for name, cols in (("a", slice(0, h)), ("b", slice(h, n))):
            rvh = plan.root_values(cols)
            clsh, _, toph = near_best_class(rvh)
            halves[name] = (set(clsh), toph)
        (ca, ta), (cb, tb) = halves["a"], halves["b"]
        out["half"] = {
            "class_a": sorted(keystr(k) for k in ca),
            "class_b": sorted(keystr(k) for k in cb),
            "top_a": keystr(ta),
            "top_b": keystr(tb),
            "exact": ca == cb,
            "top_cross": ta in cb and tb in ca,
            "jaccard": round(len(ca & cb) / len(ca | cb), 4),
        }
    # The pilot: its first move at this position on 8 seeds.
    pw0, pc0 = time.perf_counter(), time.process_time()
    pilot = []
    for i in range(p["pilot_seeds"]):
        seed_i = int(stable_u64(pos["key"], pos["t"], "pilot", i) % (2**31 - 1))
        agent = sts2_rs.MctsCombatAgent(seed=seed_i, sims=p["sims"], boss_sims=p["boss_sims"])
        r = rm.clone()
        rec = agent.step(r)
        a_raw = legal_match(cs, rm_to_combat(rec["action"]))
        a_b = to_view_action(cs, b, a_raw)
        pilot.append(move_key(b, a_b))
    out["pilot_wall_s"] = round(time.perf_counter() - pw0, 3)
    out["pilot_cpu_s"] = round(time.process_time() - pc0, 3)
    counts = collections.Counter(pilot)
    out["pilot"] = [keystr(k) for k in pilot]
    out["pilot_counts"] = {keystr(k): c for k, c in counts.most_common()}
    if plan.declined is None:
        top_c = max(counts.values())
        modal = [k for k, c in counts.items() if c == top_c]
        # A tie counts as inside the class if any tied move is: the
        # disagreement bar then errs toward killing the line.
        out["modal"] = sorted(keystr(k) for k in modal)
        out["modal_outside"] = all(k not in cls_keys for k in modal)
        out["p_pilot_in_class"] = sum(c for k, c in counts.items() if k in cls_keys) / len(pilot)
        known = {mv.key for mv in plan.root_moves()}
        out["pilot_unknown_moves"] = sum(1 for k in pilot if k not in known)
    # Honesty: the planner on other worlds the player cannot tell apart.
    if p["honesty_worlds"]:
        hw0 = time.perf_counter()
        sig0 = plan_signature(plan)
        same, world_differs = [], []
        for r in range(p["honesty_worlds"]):
            w = cs.sample_world(stable_u64(pos["key"], pos["t"], "world", r) & MASK63)
            world_differs.append(w.ordered_hash() != cs.ordered_hash())
            same.append(plan_signature(planner.plan(w, plan_seed)) == sig0)
        out["honesty"] = {
            "worlds": len(same),
            "identical": sum(same),
            "worlds_differ": sum(world_differs),
            "wall_s": round(time.perf_counter() - hw0, 3),
        }
    out["wall_s"] = round(time.perf_counter() - w0, 3)
    return out


def _done_ids(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if "header" in row or "error" in row:
                continue
            done.add(row["pos_id"])
    return done


def cmd_probe(args: argparse.Namespace) -> int:
    fights, set_sha = load_fights(args.set)
    with open(args.positions, encoding="utf-8") as fh:
        pos_header = json.loads(fh.readline())["header"]
        positions = [json.loads(line) for line in fh if line.strip()]
    if pos_header["set_sha256"] != set_sha:
        print("refusing: the positions were drawn from another fight set", file=sys.stderr)
        return 2
    if args.limit:
        positions = positions[: args.limit]
    env = agents.build_env({}, rollout_net=args.rollout_net)
    if env != pos_header["env"]:
        print(
            f"refusing: env {env} differs from the positions' {pos_header['env']}", file=sys.stderr
        )
        return 2
    agents.apply_env(env)
    plog = args.progress_log
    header = {
        "schema": SCHEMA_PROBE,
        "created": _now(),
        "positions": os.path.abspath(args.positions),
        "positions_sha256": hashlib.sha256(open(args.positions, "rb").read()).hexdigest(),
        "set_sha256": set_sha,
        "n": args.n,
        "k_chance": args.k_chance,
        "leaf_cap": args.leaf_cap,
        "pilot": {"seeds": args.pilot_seeds, "sims": args.sims, "boss_sims": args.boss_sims},
        "honesty": {"positions": args.honesty, "worlds": HONESTY_WORLDS},
        "env": env,
        "wheel": wheel_info(),
        "procs": args.procs,
        "argv": sys.argv[1:],
    }
    done = _done_ids(args.out)
    if not os.path.exists(args.out):
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"header": header}) + "\n")
    else:
        with open(args.out, encoding="utf-8") as fh:
            old = json.loads(fh.readline())["header"]
        for k in ("positions_sha256", "n", "k_chance", "leaf_cap", "pilot", "env"):
            if old.get(k) != header[k]:
                print(f"refusing to resume: header field {k} differs", file=sys.stderr)
                return 2
    todo = [(i, p) for i, p in enumerate(positions) if p["pos_id"] not in done]
    log(
        plog,
        f"probe n={args.n}: {len(positions)} positions, {len(done)} done, {len(todo)} to run, procs={args.procs}",
    )

    def tasks() -> Iterator[tuple[str, Any]]:
        for i, pos in todo:
            yield (
                pos["pos_id"],
                {
                    "pos": pos,
                    "fight": fights[pos["key"]],
                    "n": args.n,
                    "k_chance": args.k_chance,
                    "leaf_cap": args.leaf_cap,
                    "pilot_seeds": args.pilot_seeds,
                    "sims": args.sims,
                    "boss_sims": args.boss_sims,
                    "honesty_worlds": HONESTY_WORLDS if i < args.honesty else 0,
                },
            )

    sched = Scheduler("probe_position", env, args.procs, args.timeout_s)
    t0 = time.perf_counter()
    k = 0
    try:
        for pid, out in sched.run(tasks()):
            k += 1
            if "error" in out:
                out = {"pos_id": pid} | out
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(out) + "\n")
            if "error" in out:
                log(plog, f"  {pid} ERROR {out['error']}")
                continue
            hon = out.get("honesty")
            log(
                plog,
                f"  [{k}/{len(todo)}] {pid} {out['room']} {out['encounter']} r{out['round']} "
                f"leaves={out['playout_leaves']}{' FALLBACK' if out['fallback'] else ''} "
                f"class={len(out.get('class', []))}/{out.get('n_moves', '-')} "
                f"modal_out={out.get('modal_outside')} p_in={out.get('p_pilot_in_class')} "
                f"half_exact={out.get('half', {}).get('exact')} "
                f"{'honest=' + str(hon['identical']) + '/' + str(hon['worlds']) + ' ' if hon else ''}"
                f"plan={out['plan_wall_s']}s pilot={out['pilot_wall_s']}s wall={out['wall_s']}s "
                f"elapsed={time.perf_counter() - t0:.0f}s",
            )
    finally:
        sched.close()
    log(plog, f"probe done: {args.out} in {time.perf_counter() - t0:.0f} s")
    return 0


# ── report ───────────────────────────────────────────────────────────────


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def _dist(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 3),
        "p50": _q(xs, 0.5),
        "p90": _q(xs, 0.9),
        "max": max(xs),
    }


def _rate(k: int, n: int) -> dict[str, Any]:
    if n == 0:
        return {"k": 0, "n": 0}
    p = k / n
    return {"k": k, "n": n, "rate": round(p, 4), "se": round(math.sqrt(p * (1 - p) / n), 4)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    planned = [r for r in ok if not r["fallback"]]
    out: dict[str, Any] = {
        "positions": len(rows),
        "errors": len(rows) - len(ok),
        "fallback": _rate(sum(r["fallback"] for r in ok), len(ok)),
        "modal_outside_class": _rate(sum(r["modal_outside"] for r in planned), len(planned)),
        "p_pilot_in_class_lt_half": _rate(
            sum(r["p_pilot_in_class"] < 0.5 for r in planned), len(planned)
        ),
        "split_half_exact": _rate(sum(r["half"]["exact"] for r in planned), len(planned)),
        "split_half_top_cross": _rate(sum(r["half"]["top_cross"] for r in planned), len(planned)),
        "split_half_jaccard_mean": round(statistics.fmean(r["half"]["jaccard"] for r in planned), 4)
        if planned
        else None,
        "class_size": _dist([len(r["class"]) for r in planned]),
        "n_moves": _dist([r["n_moves"] for r in planned]),
        "class_is_all_moves": _rate(
            sum(len(r["class"]) == r["n_moves"] for r in planned), len(planned)
        ),
        "pilot_unknown_moves": sum(r.get("pilot_unknown_moves", 0) for r in planned),
        "playout_leaves_per_planning_point": _dist([r["playout_leaves"] for r in ok]),
        "playouts_per_planning_point": _dist([r["playouts"] for r in planned]),
        "plan_wall_s": _dist([r["plan_wall_s"] for r in planned]),
        "plan_cpu_s": _dist([r["plan_cpu_s"] for r in planned]),
        "pilot_cpu_s_8_seeds": _dist([r["pilot_cpu_s"] for r in ok]),
        "position_wall_s": _dist([r["wall_s"] for r in ok]),
    }
    hon = [r["honesty"] for r in ok if r.get("honesty")]
    out["honesty"] = {
        "positions": len(hon),
        "worlds": sum(h["worlds"] for h in hon),
        "identical": sum(h["identical"] for h in hon),
        "worlds_differ": sum(h["worlds_differ"] for h in hon),
        "positions_all_identical": sum(h["identical"] == h["worlds"] for h in hon),
    }
    return out


def verdict(s: dict[str, Any]) -> list[str]:
    lines = []
    dis = s["modal_outside_class"].get("rate")
    lines.append(
        f"disagreement (modal pilot move outside class) = {dis} "
        f"({'KILL' if dis is not None and dis < BAR_DISAGREE else 'pass'}: bar >= {BAR_DISAGREE})"
    )
    sh = s["split_half_exact"].get("rate")
    lines.append(
        f"split-half class agreement (exact) = {sh} "
        f"({'BELOW BAR: rerun at n=256' if sh is not None and sh < BAR_SPLIT_HALF else 'pass'}: "
        f"bar >= {BAR_SPLIT_HALF})"
    )
    h = s["honesty"]
    lines.append(
        f"honesty = {h['identical']}/{h['worlds']} identical on {h['positions']} positions "
        f"({'FAIL: fix first' if h['identical'] < h['worlds'] else 'pass' if h['worlds'] else 'not run'})"
    )
    fb = s["fallback"].get("rate")
    lines.append(
        f"fallback rate = {fb} ({'KILL' if fb is not None and fb > BAR_FALLBACK else 'pass'}: "
        f"bar <= {BAR_FALLBACK})"
    )
    return lines


def cmd_report(args: argparse.Namespace) -> int:
    with open(args.results, encoding="utf-8") as fh:
        header = json.loads(fh.readline())["header"]
        rows = [json.loads(line) for line in fh if line.strip()]
    last: dict[str, dict[str, Any]] = {}
    for r in rows:
        last[r["pos_id"]] = r
    rows = list(last.values())
    rep: dict[str, Any] = {"header": header, "all": summarize(rows)}
    for room in ("boss", "elite"):
        rep[room] = summarize([r for r in rows if r.get("room") == room])
    rep["verdict"] = verdict(rep["all"])
    print(json.dumps({k: v for k, v in rep.items() if k != "header"}, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P6-0: the oracle-planner probe (GH #830)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--set", required=True, help="the dev panel fight set")
        p.add_argument("--procs", type=int, default=4)
        p.add_argument("--sims", type=int, default=1000)
        p.add_argument("--boss-sims", type=int, default=4000)
        p.add_argument("--rollout-net", default=None, help="STS2_ROLLOUT_NET for pilot and leaves")
        p.add_argument("--timeout-s", type=int, default=TASK_TIMEOUT_S)
        p.add_argument("--progress-log", default=None)

    p = sub.add_parser("positions", help="draw the position set")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--boss", type=int, default=150)
    p.add_argument("--elite", type=int, default=150)
    p.add_argument("--rep", type=int, default=POSITION_REP)
    p.add_argument("--order-seed", type=int, default=ORDER_SEED)
    p.set_defaults(fn=cmd_positions)

    p = sub.add_parser("probe", help="run P6-0 on a position set")
    common(p)
    p.add_argument("--positions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=N_DEFAULT)
    p.add_argument("--k-chance", type=int, default=K_CHANCE)
    p.add_argument("--leaf-cap", type=int, default=LEAF_CAP)
    p.add_argument("--pilot-seeds", type=int, default=PILOT_SEEDS)
    p.add_argument("--honesty", type=int, default=HONESTY_POSITIONS, help="first N positions")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("report", help="aggregate a probe file")
    p.add_argument("--results", required=True)
    p.add_argument("--json", default=None)
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    # Every engine call runs in a task process; each gets one rayon thread
    # (`values_many` is rayon-parallel), so --procs is the CPU budget.
    os.environ["RAYON_NUM_THREADS"] = "1"
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
