"""Record the deployed pilot's searched decisions on human-bank fights (GH #830 step 0).

DESIGN §a. Each job is one `(fight, rep)` fight-play, seeded
`_fight_seed(key)·100 + 60 + rep` (offsets 60-63 are unused by every earlier
pipeline, so no state here is byte-identical to one rl2 or net3 trained on).
A worker builds the fight like `rollout_learn gen` (`rollout_starts.build_fight`)
and drives today's deployed pilot, `MctsCombatAgent(seed, 1000, 4000)` with
every engine seam scrubbed, to the end of the fight. At every step it:

1. takes the honest snapshot of `belief_view(0)` and feeds the fight's
   `IntentLog` (every step, so no turn start is missed);
2. asserts the legal set is observable (`legal_actions(cs) ==
   legal_actions(bv)`), groups the actions into classes on `bv`, and reads
   rl2's v1 features and the built-in net's logits;
3. on a decision selected for a shadow (100% of TEST/DEV, the fixed 10% of
   TRAIN with `sha1(id) % 10 == 0`), runs a second, independent search of the
   same position on a clone with a fresh agent (`shadow_seed`);
4. runs the primary search, whose move is played, and asserts the knobs held
   (`retained_visits is None`, `expand_order` all zero; `tree_reuse` reused
   nothing by the end of the fight).

Three outputs per fight-play, written by the parent once the fight ended:

* `traj.jsonl`: `rollout_starts`-shaped trajectory records (seed + run
  actions + `ordered_hash` per step), so `value_data.walk_record` can rebuild
  any decision's TRUE state, hash-checked (the referee uses it);
* `dec/part-NNNN.jsonl[.gz]`: one record per searched decision (>= 2 legal
  actions, root visits > 0): snapshot, actions, classes, rl2 block, both
  searches' root statistics, and the fight outcome `z`;
* `rl2v1/{train,dev,test}/shard-*.npz`: rl2-format rows (`rollout_learn
  ShardWriter`), exported from the decision records at the end (`export_rl2`),
  so the unchanged `rollout_learn train --data` runs on them.

The split is `sha1(run file) % 10` mapped by a scheme (`SCHEMES`). The holdout
(every boss800 run, every run with a Waterfall Giant fight, every combat-bench
MANIFEST run) is never played. The run is resumable (`done.txt`, keyed
`key|rep`) and writes `progress.log` every 50 fight-plays.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import pathlib
import random
import shutil
import time
from multiprocessing import get_context
from typing import Any

from ppo import rollout_starts
from ppo.combat_bench import agents
from ppo.combat_bench.holdout import holdout_runs, manifest_paths
from ppo.deepbets import value_data as vd
from ppo.pilot_human_decks import _fight_seed

from .intent_log import IntentLog
from .snapshot import action_classes, honest_snapshot, obs_actions

SCHEMA = "combat_ctx.gen.v1"
REP0 = 60
SIMS, BOSS_SIMS = 1000, 4000
MAX_STEPS = 3000
PART_EVERY = 200
PROGRESS_EVERY = 50
SHADOW_K = 17  # `rollout_starts._branch_seed` k: never a traj branch seed
BUILTIN_SHA = "a451168d07b9668a8d5af106683d61089808df6da38b9c9ed446261917cfb683"
LOG_DIR = os.path.abspath(os.path.join(vd.REPO_SUBDIR, "logs", "2026-10-18-rollout-net-recipe"))
BOSS800 = os.path.join(LOG_DIR, "boss800.jsonl")
WG = "WATERFALL_GIANT_BOSS"
# fold -> split; a fold absent from a scheme is never played under it.
SCHEMES: dict[str, dict[int, str]] = {
    "S1": {0: "test", 1: "dev", **{f: "train" for f in range(2, 10)}},
    "S2": {0: "test", 1: "test", 2: "dev", **{f: "train" for f in range(3, 10)}},
    # Engineering smoke only: folds 0 and 1 (TEST under S1/S2) are never played.
    "SMOKE": {2: "test", 3: "dev", **{f: "train" for f in range(4, 10)}},
    # Power pilot: fold 2 only (never TEST under either scheme), shadow 100%.
    "PILOT": {2: "pilot"},
}
ENV_PREFIXES = ("STS2_", "RAYON_", "OMP_", "MKL_")


class RuntimeAssert(AssertionError):
    """A pinned knob did not hold (INVALID #3): the gen stops."""


class _BuildError(Exception):
    """The bank row does not build (`rollout_starts.build_fight` ValueError)."""


# ── split, holdout, jobs ─────────────────────────────────────────────────


def fold_of(run_file: str) -> int:
    """`sha1(run file) % 10` (`value_data.split_of`'s hash)."""
    return int(hashlib.sha1(run_file.encode()).hexdigest(), 16) % 10


def split_of(run_file: str, scheme: str) -> str | None:
    return SCHEMES[scheme].get(fold_of(run_file))


def read_bank(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def holdout(rows: list[dict[str, Any]], boss800: str = BOSS800) -> dict[str, set[str]]:
    """The three held-out run sets and their union (`all`)."""
    with open(boss800, encoding="utf-8") as f:
        b8 = {json.loads(line)["file"] for line in f if line.strip()}
    wg = {r["file"] for r in rows if r["encounter"] == WG}
    bench = holdout_runs()
    return {"boss800": b8, "wg": wg, "bench": bench, "all": b8 | wg | bench}


def fight_key(row: dict[str, Any]) -> str:
    return f"{row['file']}:{row['floor']}"


def seed_of(key: str, rep: int) -> int:
    if not 0 <= rep <= 3:
        raise ValueError(f"rep {rep} outside 0-3 (offsets 60-63)")
    return _fight_seed(key) * 100 + REP0 + rep


def _sha_order(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def build_jobs(
    rows: list[dict[str, Any]],
    scheme: str,
    held: set[str],
    eval_rep1: bool = False,
    extension_rep: int | None = None,
) -> list[dict[str, Any]]:
    """The fight-plays of `scheme` (PREREG §2 'Plays and seeds').

    rep 0: every pool boss/elite fight and every TRAIN hallway fight; rep 1:
    TRAIN boss fights (and TEST/DEV boss+elite with `eval_rep1`);
    `extension_rep`: that rep of every TEST fight only (the §7.5 extension).
    """
    jobs = []
    for i, r in enumerate(rows):
        if r["file"] in held:
            continue
        split = split_of(r["file"], scheme)
        room = r["room_type"]
        if split is None or room not in ("boss", "elite", "monster"):
            continue
        if room == "monster" and split not in ("train", "pilot"):
            continue
        if extension_rep is not None:
            if split == "test" and room in ("boss", "elite"):
                jobs.append(_job(r, i, extension_rep, split))
            continue
        jobs.append(_job(r, i, 0, split))
        if (split == "train" and room == "boss") or (
            eval_rep1 and split in ("test", "dev") and room in ("boss", "elite")
        ):
            jobs.append(_job(r, i, 1, split))
    return jobs


def _job(row: dict[str, Any], idx: int, rep: int, split: str) -> dict[str, Any]:
    return {
        "row": row,
        "idx": idx,
        "rep": rep,
        "split": split,
        "fold": fold_of(row["file"]),
    }


def pilot_jobs(rows: list[dict[str, Any]], held: set[str], n_runs: int = 20, n_hall: int = 20):
    """The power pilot: every boss/elite fight of the first `n_runs` fold-2 runs
    (sha1(run file) order) with such fights, plus `n_hall` fold-2 hallway
    fights (sha1(key) order); rep 0."""
    pool = [(i, r) for i, r in enumerate(rows) if r["file"] not in held and fold_of(r["file"]) == 2]
    runs = sorted(
        {r["file"] for _, r in pool if r["room_type"] in ("boss", "elite")}, key=_sha_order
    )
    keep = set(runs[:n_runs])
    jobs = [
        _job(r, i, 0, "pilot")
        for i, r in pool
        if r["file"] in keep and r["room_type"] in ("boss", "elite")
    ]
    hall = sorted(
        [(i, r) for i, r in pool if r["room_type"] == "monster"],
        key=lambda p: _sha_order(fight_key(p[1])),
    )
    jobs += [_job(r, i, 0, "pilot") for i, r in hall[:n_hall]]
    return jobs


def smoke_jobs(
    rows: list[dict[str, Any]], held: set[str], n: int, n_hall: int
) -> list[dict[str, Any]]:
    """Engineering smoke under `SMOKE`: `n` boss+elite fights (1/5 smoke-TEST
    fold 2, 1/5 smoke-DEV fold 3, the rest TRAIN folds 4-9, half boss) and
    `n_hall` TRAIN hallway fights; rep 0 only."""
    by: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r["file"] in held:
            continue
        split = split_of(r["file"], "SMOKE")
        if split is None:
            continue
        by[split, r["room_type"]].append((i, r))
    for v in by.values():
        v.sort(key=lambda p: _sha_order(fight_key(p[1])))
    want = {"test": n // 5, "dev": n // 5, "train": n - 2 * (n // 5)}
    jobs = []
    for split, k in want.items():
        nb = k // 2
        for room, m in (("boss", nb), ("elite", k - nb)):
            jobs += [_job(r, i, 0, split) for i, r in by[split, room][:m]]
    jobs += [_job(r, i, 0, "train") for i, r in by["train", "monster"][:n_hall]]
    return jobs


def decision_id(key: str, rep: int, t: int) -> str:
    return f"{key}|r{rep}|t{t}"


def shadow_seed(seed: int, t: int) -> int:
    return rollout_starts._branch_seed(seed, t, SHADOW_K)


def shadow_selected(dec_id: str, split: str, shadow_all: bool) -> bool:
    if shadow_all or split in ("test", "dev", "pilot"):
        return True
    return int(hashlib.sha1(dec_id.encode()).hexdigest(), 16) % 10 == 0


# ── worker ───────────────────────────────────────────────────────────────


def env_hash(env: dict[str, str] | None = None) -> str:
    env = dict(os.environ) if env is None else env
    items = sorted(
        (k, v) for k, v in env.items() if k.startswith(ENV_PREFIXES) or k == "PYTHONHASHSEED"
    )
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def expected_worker_env() -> dict[str, str]:
    """The env a worker runs under after `worker_init`: the parent's, every
    engine seam removed, one rayon thread."""
    env = {k: v for k, v in os.environ.items() if not agents.is_engine_env(k)}
    env["RAYON_NUM_THREADS"] = "1"
    return env


def check_worker_env(env: dict[str, str] | None = None) -> None:
    env = dict(os.environ) if env is None else env
    bad = sorted(k for k in env if k.startswith(("STS2_RUNAI_", "STS2_ROLLOUT_")))
    if bad:
        raise RuntimeAssert(f"engine seams set in a gen worker: {bad}")


def worker_init() -> None:
    vd.init_worker({})
    check_worker_env()


_LP: Any = None


def _builtin_playout() -> Any:
    global _LP
    if _LP is None:
        import sts2_rs

        with open(vd.BUILTIN_NET, encoding="utf-8") as f:
            _LP = sts2_rs.LearnedPlayout(f.read())
    return _LP


def _canon(a: Any) -> str:
    return json.dumps(a, sort_keys=True)


def check_step_record(rec: dict[str, Any] | None, where: str) -> None:
    """The per-step runtime asserts (INVALID #3)."""
    if rec is None:
        return
    if rec.get("retained_visits") is not None:
        raise RuntimeAssert(f"{where}: retained_visits = {rec['retained_visits']!r}")
    search = rec.get("combat_search") or {}
    eo = search.get("expand_order")
    if eo and any(int(v) != 0 for v in eo.values()):
        raise RuntimeAssert(f"{where}: expand_order {eo}")


def check_end_of_fight(stats: dict[str, Any], where: str) -> None:
    if int(stats.get("reused", 0)) != 0:
        raise RuntimeAssert(f"{where}: tree_reuse reused {stats['reused']}")


def search_block(search: dict[str, Any], acts: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """Per-legal-action root statistics of one search; `(block, unmatched)`."""
    n = len(acts)
    index = {_canon(a): i for i, a in enumerate(acts)}
    blk: dict[str, Any] = {
        "visits": [0] * n,
        "q": [None] * n,
        "wins": [0] * n,
        "value_var": [None] * n,
        "hp": [None] * n,
    }
    unmatched = 0
    for row in search.get("actions", []):
        i = index.get(_canon(row["action"]))
        if i is None:
            unmatched += 1
            continue
        v = int(row.get("visits", 0))
        blk["visits"][i] += v
        blk["wins"][i] += int(row.get("wins", 0))
        if v > 0:
            blk["q"][i] = float(row["q"])
            blk["value_var"][i] = row.get("value_var")
            blk["hp"][i] = row.get("mean_final_hp")
    blk["rec"] = index.get(_canon(search.get("recommended") or {}), -1)
    blk["unmatched"] = unmatched
    return blk, unmatched


def taken_index(run_action: dict[str, Any], acts: list[dict[str, Any]]) -> int | None:
    """The legal action the run action executed, when it names one uniquely."""
    k = run_action.get("action")
    if k == "play_card":
        want = {
            "action": "play",
            "card_index": run_action.get("hand_idx"),
            "target_index": run_action.get("target_index"),
        }
    elif k == "end_turn":
        want = {"action": "end_turn"}
    elif k == "drink_potion":
        want = {
            "action": "drink_potion",
            "slot": run_action.get("potion_slot"),
            "target_index": run_action.get("target_index"),
        }
    else:
        return None
    w = {_canon(want), _canon({**want, "action": "potion"})}
    hits = [i for i, a in enumerate(acts) if _canon(a) in w]
    return hits[0] if len(hits) == 1 else None


def play_job(job: dict[str, Any]) -> dict[str, Any]:
    """Play one fight-play with the teacher; every record of it (a Pool worker)."""
    import sts2_rs

    row, rep, split = job["row"], job["rep"], job["split"]
    key = fight_key(row)
    seed = seed_of(key, rep)
    room = row["room_type"]
    shadow_all = bool(job.get("shadow_all"))
    out: dict[str, Any] = {
        "key": key,
        "rep": rep,
        "split": split,
        "env_hash": env_hash(),
        "decisions": [],
        "counts": collections.Counter(),
        "cpu": collections.Counter(),
    }
    traj: dict[str, Any] = {
        "key": key,
        "file": row["file"],
        "rep": rep,
        "seed": seed,
        "room": room,
        "encounter": row["encounter"],
        "act": row.get("act"),
        "fight": row,
        "fight_idx": job["idx"],
        "split": split,
        "fold": job["fold"],
    }
    actions: list[Any] = []
    hashes: list[int] = []
    flags: list[bool] = []
    qs: list[float | None] = []
    t_start = time.process_time()
    cnt, cpu = out["counts"], out["cpu"]
    end = None
    try:
        try:
            rm = rollout_starts.build_fight(row, seed)
        except ValueError as e:
            # A bank row the engine cannot build (an unknown relic or card): no
            # data, like `rollout_learn rl`'s eval skips it.
            raise _BuildError(repr(e)) from e
        agent = sts2_rs.MctsCombatAgent(seed=seed, sims=SIMS, boss_sims=BOSS_SIMS)
        lp = _builtin_playout()
        ilog = IntentLog()
        t = 0
        while rollout_starts.in_combat(rm):
            if t >= MAX_STEPS:
                end = "step_cap"
                break
            c0 = time.process_time()
            cs = rm.combat_state()
            h = rollout_starts.ordered_hash(cs)
            acts = cs.legal_actions()
            snap = bv = None
            try:
                bv = cs.belief_view(0)
                snap = ilog.observe(honest_snapshot(bv, room))
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:  # a PyO3 panic is a BaseException
                cnt["obs_fail"] += 1
            if bv is not None and bv.legal_actions() != acts:
                raise RuntimeAssert(f"{decision_id(key, rep, t)}: legal set differs in bv")
            multi = len(acts) >= 2
            dec_id = decision_id(key, rep, t)
            feat: dict[str, Any] | None = None
            if multi:
                rf = sts2_rs.rollout_features(cs, version=1)
                if [_canon(a) for a in rf["actions"]] != [_canon(a) for a in acts]:
                    raise RuntimeAssert(f"{dec_id}: rollout_features actions differ")
                feat = {
                    "state": [float(v) for v in rf["state"]],
                    "x": [[float(v) for v in xx] for xx in rf["x"]],
                    "keys": list(rf["keys"]),
                    "logits": [float(v) for v in lp.explain(cs)["logits"]],
                }
                act_obs = cls = None
                if snap is not None:
                    act_obs = obs_actions(acts, cs.pending_choice(), bv.pending_choice())
                    cls = action_classes(bv, act_obs)
                feat["act_obs"] = None if act_obs == acts else act_obs
                feat["cls"] = cls
            cpu["feat"] += time.process_time() - c0
            shadow = None
            if multi and shadow_selected(dec_id, split, shadow_all):
                c1 = time.process_time()
                ss = shadow_seed(seed, t)
                srec = sts2_rs.MctsCombatAgent(seed=ss, sims=SIMS, boss_sims=BOSS_SIMS).step(
                    rm.clone()
                )
                check_step_record(srec, f"{dec_id} shadow")
                cnt["asserts"] += 1
                if srec is not None and srec.get("combat_search"):
                    shadow, um = search_block(srec["combat_search"], acts)
                    shadow["seed"] = ss
                    cnt["unmatched"] += um
                cpu["shadow"] += time.process_time() - c1
            t0chk = None
            if t == 0 and job.get("t0_check") and multi:
                c2 = time.process_time()
                chk = sts2_rs.MctsCombatAgent(seed=seed, sims=SIMS, boss_sims=BOSS_SIMS).step(
                    rm.clone()
                )
                t0chk = chk
                cpu["t0"] += time.process_time() - c2
            c3 = time.process_time()
            rec = agent.step(rm)
            cpu["search"] += time.process_time() - c3
            check_step_record(rec, dec_id)
            cnt["asserts"] += 1
            if rec is None:
                end = "no_action"
                break
            was_searched = rollout_starts.searched(rec)
            hashes.append(h)
            actions.append(rec["action"])
            flags.append(was_searched)
            qs.append(rollout_starts.q_best(rec))
            if t0chk is not None:
                a = [
                    (_canon(x["action"]), x["visits"], x["q"])
                    for x in (rec.get("combat_search") or {}).get("actions", [])
                ]
                b = [
                    (_canon(x["action"]), x["visits"], x["q"])
                    for x in (t0chk.get("combat_search") or {}).get("actions", [])
                ]
                traj["t0_repro"] = a == b
            if multi and was_searched and feat is not None:
                prim, um = search_block(rec["combat_search"], acts)
                cnt["unmatched"] += um
                ti = taken_index(rec["action"], acts)
                prim["taken"] = prim["rec"] if ti is None else ti
                cnt["shadows"] += shadow is not None
                out["decisions"].append(
                    {
                        "id": dec_id,
                        "key": key,
                        "file": row["file"],
                        "floor": row["floor"],
                        "rep": rep,
                        "seed": seed,
                        "t": t,
                        "fold": job["fold"],
                        "split": split,
                        "room": room,
                        "encounter": row["encounter"],
                        "fight_idx": job["idx"],
                        "ohash": h,
                        "obs": snap,
                        "actions": acts,
                        "act_obs": feat.pop("act_obs"),
                        "cls": feat.pop("cls"),
                        "rl2": feat,
                        "A": prim,
                        "B": shadow,
                    }
                )
            t += 1
        if end is None:
            end = "died" if rm.is_over() and not rm.player_won else "won"
        check_end_of_fight(agent.tree_reuse_stats(), key)
        cnt["asserts"] += 1
        traj["end"] = end
        traj["won"] = end == "won"
        traj["value"] = rollout_starts.final_value(rm) if end in ("won", "died") else None
        traj["hp_end"] = int(rm.current_hp)
        cnt["ilog_resets"] += ilog.resets
    except RuntimeAssert as e:
        out["assert"] = f"{key}|r{rep}: {e}"
        traj["end"] = "error"
    except (KeyboardInterrupt, SystemExit):
        raise
    except _BuildError as e:
        traj["end"] = "build_error"
        traj["error"] = str(e)
        cnt["build_errors"] += 1
    except BaseException as e:  # a PyO3 panic is a BaseException
        traj["end"] = "error"
        traj["error"] = repr(e)
        cnt["errors"] += 1
    traj["actions"] = actions
    traj["hashes"] = hashes
    traj["searched"] = flags
    traj["q_best"] = qs
    traj["cpu_s"] = round(time.process_time() - t_start, 3)
    cpu["total"] = time.process_time() - t_start
    z = {
        "end": traj.get("end"),
        "won": traj.get("won"),
        "value": traj.get("value"),
        "hp_end": traj.get("hp_end"),
        "n_dec": len(out["decisions"]),
    }
    for d in out["decisions"]:
        d["z"] = z
    out["traj"] = traj
    out["counts"] = dict(cnt)
    out["cpu"] = dict(cpu)
    return out


# ── parent ───────────────────────────────────────────────────────────────


class Parts:
    """`dec/part-NNNN.jsonl`, rotated every `every` fight-plays, gzipped on rotation."""

    def __init__(self, d: str, every: int) -> None:
        self.d = d
        self.every = every
        os.makedirs(d, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(d, "part-*.jsonl"))):
            _gzip(p)
        self.n = len(glob.glob(os.path.join(d, "part-*.jsonl.gz")))
        self.f = None
        self.plays = 0
        self.gz_bytes = 0
        self.records = 0

    def write(self, recs: list[dict[str, Any]]) -> None:
        if self.f is None:
            self.f = open(os.path.join(self.d, f"part-{self.n:04d}.jsonl"), "w", encoding="utf-8")
        for r in recs:
            self.f.write(json.dumps(r, separators=(",", ":")) + "\n")
        self.f.flush()
        self.plays += 1
        if self.plays % self.every == 0:
            self.rotate()

    def rotate(self) -> None:
        if self.f is None:
            return
        path = self.f.name
        self.f.close()
        self.f = None
        self.gz_bytes += _gzip(path)
        self.n += 1

    def close(self) -> None:
        self.rotate()


def _gzip(path: str) -> int:
    with open(path, "rb") as src, gzip.open(path + ".gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(path)
    return pathlib.Path(path + ".gz").stat().st_size


def dec_paths(gen_dir: str) -> list[str]:
    d = os.path.join(gen_dir, "dec")
    return sorted(
        glob.glob(os.path.join(d, "part-*.jsonl.gz")) + glob.glob(os.path.join(d, "part-*.jsonl"))
    )


def iter_records(gen_dirs: list[str] | str):
    """Every decision record of the gen dirs, a re-played fight's duplicates
    resolved to the LAST occurrence (a crash between the decision write and
    `done.txt` re-plays that fight on resume)."""
    if isinstance(gen_dirs, str):
        gen_dirs = [gen_dirs]
    paths = [p for g in gen_dirs for p in dec_paths(g)]

    def lines(p: str):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as f:
            for line in f:
                if line.strip():
                    yield line

    last: dict[str, tuple[int, int]] = {}
    for pi, p in enumerate(paths):
        for li, line in enumerate(lines(p)):
            i = line.find('"id":"')
            rid = line[i + 6 : line.index('"', i + 6)]
            last[rid] = (pi, li)
    for pi, p in enumerate(paths):
        for li, line in enumerate(lines(p)):
            r = json.loads(line)
            if last.get(r["id"]) == (pi, li):
                yield r


class Progress(vd.Progress):
    pass


def header(args: argparse.Namespace, scheme: str, power: dict[str, Any] | None) -> dict[str, Any]:
    import sts2_rs

    net_sha = vd.sha256_file(vd.BUILTIN_NET)
    if net_sha != BUILTIN_SHA:
        raise SystemExit(f"built-in net sha {net_sha} != pinned {BUILTIN_SHA} (INVALID #3)")
    info = sts2_rs.build_info()
    if not info.get("pgo") and not args.allow_plain_wheel:
        raise SystemExit("the wheel is not a PGO build (INVALID #3); --allow-plain-wheel for tests")
    return {
        "schema": rollout_starts.SCHEMA,
        "gen_schema": SCHEMA,
        "spec": agents.mcts_spec(SIMS, BOSS_SIMS, {}),
        "engine": sts2_rs.src_hash(),
        "build_info": info,
        "builtin_net_sha256": net_sha,
        "rep0": REP0,
        "scheme": scheme,
        "fights": os.path.abspath(args.fights),
        "fights_sha256": vd.sha256_file(args.fights),
        "max_steps": MAX_STEPS,
        "worker_env_hash": env_hash(expected_worker_env()),
        "bench_manifests": [os.path.basename(p) for p in manifest_paths()],
        "power": power,
        "mode": args.mode,
    }


def run_gen(args: argparse.Namespace) -> int:
    rows = read_bank(args.fights)
    held = holdout(rows, args.boss800)
    power = None
    if args.power and os.path.exists(args.power):
        with open(args.power, encoding="utf-8") as f:
            power = json.load(f)
    scheme = args.scheme or (power or {}).get("scheme")
    if args.mode == "pilot":
        scheme = "PILOT"
        jobs = pilot_jobs(rows, held["all"])
    elif args.mode == "smoke":
        scheme = "SMOKE"
        jobs = smoke_jobs(rows, held["all"], args.smoke_n, args.smoke_hall)
    else:
        if scheme not in ("S1", "S2"):
            raise SystemExit("--scheme S1|S2 (or a power.json naming one) is required")
        eval_rep1 = bool((power or {}).get("eval_rep1"))
        jobs = build_jobs(rows, scheme, held["all"], eval_rep1, args.extension_rep)
    for j in jobs:
        assert j["row"]["file"] not in held["all"]
        j["shadow_all"] = args.mode == "pilot"
        j["t0_check"] = args.mode in ("pilot", "smoke") or args.t0_check
    os.makedirs(args.out, exist_ok=True)
    hdr = header(args, scheme, power)
    hpath = os.path.join(args.out, "header.json")
    if os.path.exists(hpath):
        with open(hpath, encoding="utf-8") as f:
            old = json.load(f)
        diff = [
            k
            for k in ("engine", "scheme", "rep0", "spec", "builtin_net_sha256", "worker_env_hash")
            if old.get(k) != hdr.get(k)
        ]
        if diff:
            raise SystemExit(f"{hpath}: header differs in {diff}; use a new --out")
    else:
        vd.write_json(hpath, hdr)
    traj_path = os.path.join(args.out, "traj.jsonl")
    if not os.path.exists(traj_path):
        with open(traj_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(hdr) + "\n")
    done_path = os.path.join(args.out, "done.txt")
    done = (
        set(open(done_path, encoding="utf-8").read().split())
        if os.path.exists(done_path)
        else set()
    )
    todo = [j for j in jobs if f"{fight_key(j['row'])}|{j['rep']}" not in done]
    # Long fights first (bosses), so the tail is short; stable random within.
    rnd = random.Random(0)
    rnd.shuffle(todo)
    todo.sort(key=lambda j: {"boss": 0, "elite": 1, "monster": 2}[j["row"]["room_type"]])
    if args.max_jobs:
        todo = todo[: args.max_jobs]
    log = Progress(os.path.join(args.out, "progress.log"), "combat_ctx gen")
    log(
        f"{len(jobs)} fight-plays under {scheme} ({len(done)} done, {len(todo)} to play); "
        f"holdout runs {len(held['all'])} (boss800 {len(held['boss800'])}, WG {len(held['wg'])}, "
        f"bench {len(held['bench'])}); procs {args.procs}"
    )
    parts = Parts(os.path.join(args.out, "dec"), PART_EVERY)
    tot: collections.Counter = collections.Counter()
    cpu_room: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    plays_room: collections.Counter = collections.Counter()
    dec_room: collections.Counter = collections.Counter()
    t0 = time.time()
    want_hash = hdr["worker_env_hash"]
    n = 0
    failed = None
    with (
        get_context("spawn").Pool(args.procs, initializer=worker_init) as pool,
        open(traj_path, "a", encoding="utf-8") as tf,
        open(done_path, "a", encoding="utf-8") as df,
    ):
        for res in pool.imap_unordered(play_job, todo):
            if res["env_hash"] != want_hash:
                failed = f"worker env hash {res['env_hash']} != {want_hash} (INVALID #3)"
                break
            if "assert" in res:
                failed = f"runtime assert: {res['assert']} (INVALID #3)"
                break
            parts.write(res["decisions"])
            tf.write(json.dumps(res["traj"]) + "\n")
            tf.flush()
            df.write(f"{res['key']}|{res['rep']}\n")
            df.flush()
            n += 1
            room = res["traj"]["room"]
            plays_room[room] += 1
            dec_room[room] += len(res["decisions"])
            tot.update(res["counts"])
            tot["decisions"] += len(res["decisions"])
            tot["t0_checked"] += "t0_repro" in res["traj"]
            tot["t0_mismatch"] += res["traj"].get("t0_repro") is False
            for k, v in res["cpu"].items():
                cpu_room[room][k] += v
            if n % PROGRESS_EVERY == 0 or n == len(todo):
                per = {
                    r: f"{cpu_room[r]['total'] / plays_room[r]:.2f}s/{dec_room[r] / plays_room[r]:.1f}dec"
                    for r in plays_room
                }
                log(
                    f"{n}/{len(todo)} plays, {tot['decisions']} decisions, {tot['shadows']} "
                    f"shadowed, {tot['errors']} errors, {tot['build_errors']} unbuildable rows, "
                    f"{tot['unmatched']} unmatched, "
                    f"{tot['obs_fail']} obs failures, {tot['asserts']} asserts passed, "
                    f"t0 {tot['t0_checked'] - tot['t0_mismatch']}/{tot['t0_checked']} reproduced, "
                    f"ilog resets {tot['ilog_resets']}, per room {per}, {time.time() - t0:.0f}s"
                )
        if failed:
            pool.terminate()
    parts.close()
    stats = {
        "plays": dict(plays_room),
        "decisions": dict(dec_room),
        "cpu_s": {r: dict(v) for r, v in cpu_room.items()},
        "counts": dict(tot),
        "wall_s": round(time.time() - t0, 1),
        "procs": args.procs,
        "dec_gz_bytes": parts.gz_bytes,
    }
    vd.write_json(os.path.join(args.out, f"stats-{int(time.time())}.json"), stats)
    if failed:
        log(f"STOPPED: {failed}")
        return 2
    if not args.no_export:
        export_rl2(args.out)
    log("DONE")
    return 0


# ── rl2-format export ────────────────────────────────────────────────────

ROOMS_RL2 = {"monster": 0, "elite": 1, "boss": 2}


def rl2_row(r: dict[str, Any]) -> dict[str, Any]:
    vis = [float(v) for v in r["A"]["visits"]]
    tot = sum(vis)
    return {
        "state": r["rl2"]["state"],
        "keys": r["rl2"]["keys"],
        "x": r["rl2"]["x"],
        "visits": vis,
        "chosen": int(r["A"]["rec"]),
        "pwin": (sum(r["A"]["wins"]) / tot) if tot else 0.0,
        "room": ROOMS_RL2.get(r["room"], 0),
        "fight": int(r["fight_idx"]),
    }


def export_rl2(gen_dir: str, per_shard: int = 20000) -> dict[str, int]:
    """`rl2v1/{split}/shard-*.npz` from the decision records (rebuilt from scratch)."""
    from ppo.combat_bench.holdout import subtract_bench_runs
    from ppo.rollout_learn import BENCH_META, FEATURES_META, ShardWriter

    root = os.path.join(gen_dir, "rl2v1")
    if os.path.exists(root):
        shutil.rmtree(root)
    writers: dict[str, Any] = {}
    counts: collections.Counter = collections.Counter()
    for r in iter_records(gen_dir):
        s = r["split"]
        if s not in writers:
            d = os.path.join(root, s)
            writers[s] = ShardWriter(d, per_shard)
            with open(os.path.join(d, FEATURES_META), "w", encoding="utf-8") as f:
                json.dump({"feature_version": 1}, f)
            _, meta = subtract_bench_runs([], lambda x: x, tag="[combat_ctx export]")
            with open(os.path.join(d, BENCH_META), "w", encoding="utf-8") as f:
                json.dump(meta, f)
        writers[s].add([rl2_row(r)])
        counts[s] += 1
    for w in writers.values():
        w.flush()
    return dict(counts)


def add_parser(sub: Any) -> None:
    g = sub.add_parser("gen", help="record the deployed pilot's searched decisions")
    g.add_argument("--fights", required=True, help="human fight bank (hbank.jsonl)")
    g.add_argument("--out", required=True)
    g.add_argument("--mode", choices=("full", "pilot", "smoke"), default="full")
    g.add_argument("--scheme", choices=("S1", "S2"))
    g.add_argument("--power", help="power.json from `power` (names the scheme)")
    g.add_argument("--boss800", default=BOSS800)
    g.add_argument("--extension-rep", type=int, help="stage-2 extension: this rep of TEST only")
    g.add_argument("--smoke-n", type=int, default=40)
    g.add_argument("--smoke-hall", type=int, default=10)
    g.add_argument("--max-jobs", type=int, default=0)
    g.add_argument(
        "--t0-check", action="store_true", help="t = 0 reproduction check (on in pilot/smoke)"
    )
    g.add_argument("--procs", type=int, default=40)
    g.add_argument("--no-export", action="store_true")
    g.add_argument("--allow-plain-wheel", action="store_true")
    g.set_defaults(func=run_gen)
