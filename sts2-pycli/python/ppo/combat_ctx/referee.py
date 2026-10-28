"""Fair re-scoring of the students' disagreements (PREREG §5.5, DESIGN §j).

Root Q is not `Q^teacher(s, a)`: UCB leaves a class with few visits close to
the value of rl2 playouts from that child, which favours rl2-shaped students.
So every TEST E decision on which the frozen picks (3 student arms x 3 seeds)
are not all one class is re-searched FAIRLY: each legal action gets its own
independent search (`mcts_recommend(fair=True)`).

* The TRUE state is rebuilt from the trajectory record
  (`value_data.walk_record`, hash-checked against the decision's
  `ordered_hash`); only `cs.belief_view(k)` is searched (in fair mode, applying
  a stochastic action to the raw state would realize the true draw).
* For k in {1, 2} and two referees: F1 the deployed playout
  (`rollout_policy=None`), F2 the heuristic playout (removes the alignment
  with how rl2 would continue); budget = the decision's room tier.
* A class's fair score is the mean of its member actions' fair Q; a referee's
  score averages k = 1, 2.

Resumable: `scores.jsonl` is appended per decision, `progress.log` every 20.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from typing import Any

import numpy as np

from ppo.deepbets import value_data as vd

from . import gen as G

REFEREES = {"F1": None, "F2": "heuristic"}
KS = (1, 2)


def _canon(a: Any) -> str:
    return json.dumps(a, sort_keys=True)


def score_decision(job: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Fair Q per action class under each referee (a Pool worker)."""
    import sts2_rs

    dec, traj = job
    states, _verified = vd.walk_record(traj, [int(dec["t"])])
    cs = states.get(int(dec["t"]))
    if cs is None or int(cs.ordered_hash()) != int(dec["ohash"]):
        return {"id": dec["id"], "error": "rebuild hash mismatch"}
    sims = G.BOSS_SIMS if dec["room"] == "boss" else G.SIMS
    acts = dec["actions"]
    index = {_canon(a): i for i, a in enumerate(acts)}
    cls = np.asarray(dec["cls"], np.int64)
    out: dict[str, Any] = {"id": dec["id"], "sims": sims}
    for name, policy in REFEREES.items():
        per_k = []
        for k in KS:
            res = sts2_rs.mcts_recommend(
                cs.belief_view(k),
                sims=sims,
                seed=k,
                threads=1,
                fair=True,
                rollout_policy=policy,
                hidden_info="beliefdraw",
            )
            q = np.full(len(acts), np.nan)
            for row in res["actions"]:
                i = index.get(_canon(row["action"]))
                if i is not None:
                    q[i] = float(row["q"])
            qc = np.asarray(
                [
                    np.nanmean(q[cls == c]) if np.isfinite(q[cls == c]).any() else np.nan
                    for c in range(cls.max() + 1)
                ]
            )
            per_k.append(qc)
        out[name] = np.nanmean(np.stack(per_k), 0).tolist()
    return out


def cmd_referee(args: argparse.Namespace) -> int:
    os.makedirs(args.out, exist_ok=True)
    log = vd.Progress(os.path.join(args.out, "progress.log"), "combat_ctx referee")
    with open(args.ids, encoding="utf-8") as f:
        want = {line.strip() for line in f if line.strip()}
    path = os.path.join(args.out, "scores.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    trajs = {}
    for g in args.gen:
        with open(os.path.join(g, "traj.jsonl"), encoding="utf-8") as f:
            next(f)
            for line in f:
                t = json.loads(line)
                trajs[t["key"], int(t["rep"])] = t
    jobs = []
    for g in args.gen:
        for r in G.iter_records(g):
            if r["id"] in want and r["id"] not in done:
                slim = {k: r[k] for k in ("id", "t", "ohash", "room", "actions", "cls")}
                jobs.append((slim, trajs[r["key"], int(r["rep"])]))
    log(f"{len(want)} decisions wanted, {len(done)} done, {len(jobs)} to score")
    t0 = time.time()
    with (
        get_context("spawn").Pool(args.procs, initializer=vd.init_worker, initargs=({},)) as pool,
        open(path, "a", encoding="utf-8") as f,
    ):
        for n, res in enumerate(pool.imap_unordered(score_decision, jobs), 1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(jobs):
                log(f"{n}/{len(jobs)} scored, {time.time() - t0:.0f}s")
    return 0


def add_parser(sub: Any) -> None:
    r = sub.add_parser("referee", help="fair re-search of the disagreement subset")
    r.add_argument("--gen", nargs="+", required=True)
    r.add_argument("--ids", required=True, help="file with one decision id per line")
    r.add_argument("--out", required=True)
    r.add_argument("--procs", type=int, default=40)
    r.set_defaults(func=cmd_referee)
