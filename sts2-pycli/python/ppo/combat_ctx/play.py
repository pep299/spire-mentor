"""Play-alone driver: a policy plays whole fights with no search (GH #830 step 0, DESIGN §e).

One loop for every policy. A job is `(bank row, build seed, playout seed)`;
the WORKER builds the fight (`rollout_starts.build_fight`), because a
`CombatState` cannot cross a process boundary, and batches its policy's
forward over its live jobs:

    acts = cs.legal_actions()            # empty -> terminal
    logits = policy.logits(batch)        # aligned with legal_actions
    a = argmax (greedy) or a sample of softmax(logits / T) (numpy rng, playout seed)
    cs.apply(acts[a]); steps += 1        # step cap 600 -> done, not won

The driver never reads a state field itself (only `legal_actions`, `apply`,
`terminal_value`); each policy's reads are covered by the honesty tests.

Policies (`policy_from_spec`):

* `rl2` / `rl2:<json>`: the Rust `LearnedPlayout(json).explain(cs)` logits:
  exactly what a playout samples from (built-in `ironclad.json` by default);
* `rl2torch:<json>`: `rollout_features(cs, 1)` -> the torch twin
  (`rollout_learn._load_playout_net`), for the logit-parity check;
* `ctx:<best.pt>`: `CombatCtxNet` on the honest snapshot of
  `belief_view(0)` + the job's own `IntentLog` + rl2's v1 features;
* `heur`: the heuristic playout's own logit (v1 feature 21).

The anchor (`anchor`) reproduces `rollout_learn rl`'s eval, read from its
code: each boss800 row built from `_fight_seed(key)·100 + 77` (a `ValueError`
skips the row), jobs = each state x 4 in file order, playout seed
`1_000_003·i + 17`, temperature 1.0, raw view, 600 steps. It runs (1) logit
parity (torch twin vs Rust on every state the first jobs visit), (2) the Rust
reference `LearnedPlayout.play_many` -> `W_rust` against rl2's logged 39.6%,
and (3) the Python driver on the same jobs -> `W_py`, paired by fight
(|W_py - W_rust| <= 2pp with a CI covering 0; PREREG INVALID #9).

Sealing: a student checkpoint never plays boss800 without `--final` and a
matching `FROZEN.json`.

`play-contrast` is the play-alone routing quantity (PREREG §5.6): two
policies' `play --out` files per training seed, the same jobs in each, `won`
averaged over each side's seeds, the paired difference with a two-level
bootstrap clustered by run (runs, then each side's seeds) and, for reference,
by bank fight.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from typing import Any

import numpy as np

from ppo import rollout_starts
from ppo.deepbets import value_data as vd
from ppo.pilot_human_decks import _fight_seed

from . import gen as G

PLAY_SEED_OFFSET = 77
MAX_STEPS = 600
RL2_ANCHOR = 0.396


def build_seed(row: dict[str, Any]) -> int:
    return _fight_seed(f"{row['file']}:{row['floor']}") * 100 + PLAY_SEED_OFFSET


def make_jobs(rows: list[dict[str, Any]], reps: int) -> tuple[list[dict[str, Any]], int]:
    """`rl`'s eval jobs: rows that build, each x `reps`, seeds `1_000_003·i + 17`.

    Returns `(jobs, skipped rows)`. Building here is only the ValueError check
    (the worker rebuilds)."""
    jobs, skipped, i = [], 0, 0
    for fi, row in enumerate(rows):
        s = build_seed(row)
        try:
            rollout_starts.build_fight(row, s)
        except ValueError:
            skipped += 1
            continue
        for rep in range(reps):
            jobs.append({"i": i, "fight": fi, "rep": rep, "row": row, "build_seed": s,
                         "seed": 1_000_003 * i + 17})  # fmt: skip
            i += 1
    return jobs, skipped


# ── policies ─────────────────────────────────────────────────────────────


class Rl2Explain:
    def __init__(self, path: str | None) -> None:
        import sts2_rs

        with open(path or vd.BUILTIN_NET, encoding="utf-8") as f:
            text = f.read()
        self.no_early_end = bool(json.loads(text).get("no_early_end_turn"))
        self.lp = sts2_rs.LearnedPlayout(text)

    def observe(self, job: dict[str, Any], cs: Any) -> None:
        pass

    def logits(self, batch: list[tuple[dict[str, Any], Any, list]]) -> list[np.ndarray]:
        out = []
        for _job, cs, acts in batch:
            z = np.asarray(self.lp.explain(cs)["logits"], np.float64)
            if self.no_early_end and any(a.get("action") == "play" for a in acts):
                z[[a.get("action") == "end_turn" for a in acts]] = -np.inf
            out.append(z)
        return out


class Rl2Torch:
    def __init__(self, path: str | None) -> None:
        import torch

        from ppo.rollout_learn import _load_playout_net

        torch.set_num_threads(1)
        with open(path or vd.BUILTIN_NET, encoding="utf-8") as f:
            net = json.load(f)
        self.torch = torch
        self.net = _load_playout_net(torch, net).eval()
        self.rows = {k: i for i, k in enumerate(net["keys"])}

    def observe(self, job: dict[str, Any], cs: Any) -> None:
        pass

    def features(self, cs: Any) -> dict[str, Any]:
        import sts2_rs

        return sts2_rs.rollout_features(cs, version=1)

    def logits(self, batch: list[tuple[dict[str, Any], Any, list]]) -> list[np.ndarray]:
        torch = self.torch
        feats = [self.features(cs) for _j, cs, _a in batch]
        na = [len(f["keys"]) for f in feats]
        am = max(na)
        S = torch.tensor([f["state"] for f in feats], dtype=torch.float32)
        X = torch.zeros(len(feats), am, len(feats[0]["x"][0]))
        K = torch.zeros(len(feats), am, dtype=torch.long)
        M = torch.zeros(len(feats), am, dtype=torch.bool)
        for i, f in enumerate(feats):
            X[i, : na[i]] = torch.tensor(f["x"], dtype=torch.float32)
            K[i, : na[i]] = torch.tensor([self.rows.get(k, 0) for k in f["keys"]])
            M[i, : na[i]] = True
        with torch.no_grad():
            z = self.net(S, X, K, M).double().numpy()
        return [z[i, : na[i]] for i in range(len(feats))]


class Heur:
    def observe(self, job: dict[str, Any], cs: Any) -> None:
        pass

    def logits(self, batch: list[tuple[dict[str, Any], Any, list]]) -> list[np.ndarray]:
        import sts2_rs

        return [
            np.asarray([x[21] for x in sts2_rs.rollout_features(cs, version=1)["x"]], np.float64)
            for _j, cs, _a in batch
        ]


class CtxNet:
    """A `CombatCtxNet` checkpoint, on the honest snapshot (CPU, one thread)."""

    def __init__(self, path: str) -> None:
        import torch

        from . import tokens as T
        from .data import RareMap
        from .model import CombatCtxNet

        torch.set_num_threads(1)
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck["vocab"] != T.vocab().digest() or ck["n_cont"] != T.N_CONT:
            raise SystemExit(f"{path}: token schema differs from this checkout's")
        self.arm = ck["arm"]
        self.net = CombatCtxNet.from_config(ck["config"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.rare = RareMap.from_json(ck["rare"])
        self.torch = torch
        self.T = T

    def observe(self, job: dict[str, Any], cs: Any) -> None:
        from .intent_log import IntentLog
        from .snapshot import honest_snapshot

        bv = cs.belief_view(0)
        if "ilog" not in job:
            job["ilog"] = IntentLog()
        job["snap"] = job["ilog"].observe(honest_snapshot(bv, job["row"]["room_type"]))
        job["bv"] = bv

    def logits(self, batch: list[tuple[dict[str, Any], Any, list]]) -> list[np.ndarray]:
        import sts2_rs

        from .snapshot import obs_actions

        torch, T = self.torch, self.T
        toks = []
        for job, cs, acts in batch:
            rf = sts2_rs.rollout_features(cs, version=1)
            act_obs = obs_actions(acts, cs.pending_choice(), job["bv"].pending_choice())
            tk = T.tokens(job["snap"], act_obs, rf, self.arm)
            toks.append(tk)
        nt = [len(t["sec"]) for t in toks]
        na = [t["n_act"] for t in toks]
        tm, am = max(nt), max(na)
        b = len(toks)
        cat = np.tile(T.vocab().pads.astype(np.int64), (b, tm, 1))
        cont = np.zeros((b, tm, T.N_CONT), np.float32)
        sec = np.zeros((b, tm), np.int64)
        pad = np.ones((b, tm), bool)
        act_pos = np.zeros((b, am), np.int64)
        amask = np.zeros((b, am), bool)
        for i, t in enumerate(toks):
            cat[i, : nt[i]] = self.rare.apply(t["cat"])
            cont[i, : nt[i]] = T.f16(t["cont"]).astype(np.float32)
            sec[i, : nt[i]] = t["sec"]
            pad[i, : nt[i]] = False
            act_pos[i, : na[i]] = np.arange(nt[i] - na[i], nt[i])
            amask[i, : na[i]] = True
        tt = torch.from_numpy
        with torch.no_grad():
            logits, _ = self.net(tt(cat), tt(cont), tt(sec), tt(pad), tt(act_pos), tt(amask))
        z = logits.double().numpy()
        return [z[i, : na[i]] for i in range(b)]


def policy_from_spec(spec: str) -> Any:
    kind, _, arg = spec.partition(":")
    if kind == "rl2":
        return Rl2Explain(arg or None)
    if kind == "rl2torch":
        return Rl2Torch(arg or None)
    if kind == "ctx":
        return CtxNet(arg)
    if kind == "heur":
        return Heur()
    raise ValueError(f"unknown policy {spec!r}")


def is_student(spec: str) -> bool:
    kind, _, arg = spec.partition(":")
    if kind in ("heur",):
        return False
    if kind in ("rl2", "rl2torch"):
        return bool(arg) and os.path.abspath(arg) != os.path.abspath(vd.BUILTIN_NET)
    return True


# ── the loop ─────────────────────────────────────────────────────────────


def _won(cs: Any) -> bool:
    term, val = cs.terminal_value()
    return bool(term) and float(val) > 0


def play_jobs(
    policy: Any,
    jobs: list[dict[str, Any]],
    temperature: float,
    greedy: bool,
    view: str = "raw",
    max_steps: int = MAX_STEPS,
    batch: int = 64,
) -> list[dict[str, Any]]:
    """Play `jobs` to the end in this process; one result per job."""
    live = []
    for job in jobs:
        cs = rollout_starts.build_fight(job["row"], job["build_seed"]).combat_state()
        if view == "belief":
            cs = cs.belief_view(int(job["seed"]))
        job = dict(job)
        job.update(cs=cs, rng=np.random.default_rng(int(job["seed"])), steps=0, decisions=0)
        live.append(job)
    done = []
    while live:
        nxt = []
        want = []
        for job in live:
            cs = job["cs"]
            acts = cs.legal_actions()
            if not acts or job["steps"] >= max_steps:
                done.append(
                    {
                        "i": job["i"],
                        "fight": job["fight"],
                        "file": job["row"]["file"],
                        "key": f"{job['row']['file']}:{job['row']['floor']}",
                        "rep": job["rep"],
                        "won": _won(cs) if not acts else False,
                        "capped": bool(acts),
                        "steps": job["steps"],
                        "decisions": job["decisions"],
                    }
                )
                continue
            policy.observe(job, cs)
            if len(acts) == 1:
                cs.apply(acts[0])
                job["steps"] += 1
                nxt.append(job)
            else:
                want.append((job, cs, acts))
        for k in range(0, len(want), batch):
            chunk = want[k : k + batch]
            for (job, cs, acts), z in zip(chunk, policy.logits(chunk), strict=True):
                if len(z) != len(acts):
                    raise RuntimeError("policy logits not aligned with legal_actions")
                if greedy:
                    a = int(np.argmax(z))
                else:
                    zz = (z - np.max(z)) / max(temperature, 1e-3)
                    p = np.exp(zz)
                    p /= p.sum()
                    a = int(job["rng"].choice(len(p), p=p))
                cs.apply(acts[a])
                job["steps"] += 1
                job["decisions"] += 1
                nxt.append(job)
        live = nxt
    return done


def _worker(job: tuple[str, list[dict[str, Any]], float, bool, str, int]) -> list[dict[str, Any]]:
    spec, jobs, temperature, greedy, view, max_steps = job
    return play_jobs(policy_from_spec(spec), jobs, temperature, greedy, view, max_steps)


def play_many(
    spec: str,
    jobs: list[dict[str, Any]],
    temperature: float = 1.0,
    greedy: bool = False,
    view: str = "raw",
    max_steps: int = MAX_STEPS,
    procs: int = 40,
) -> list[dict[str, Any]]:
    """Shard `jobs` by fight across spawn workers; results in job order."""
    by_fight: dict[int, list[dict[str, Any]]] = {}
    for j in jobs:
        by_fight.setdefault(j["fight"], []).append(j)
    groups = list(by_fight.values())
    n_chunks = max(1, min(len(groups), procs * 4))
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(n_chunks)]
    for k, g in enumerate(groups):
        chunks[k % n_chunks].extend(g)
    work = [(spec, c, temperature, greedy, view, max_steps) for c in chunks if c]
    out: list[dict[str, Any]] = []
    with get_context("spawn").Pool(procs, initializer=vd.init_worker, initargs=({},)) as pool:
        for res in pool.imap_unordered(_worker, work):
            out.extend(res)
    return sorted(out, key=lambda r: r["i"])


def fight_ci(
    won: np.ndarray, fight: np.ndarray, n_boot: int = 10_000, level: float = 0.95
) -> dict[str, float]:
    """Win rate with a fight-clustered bootstrap CI."""
    u, inv = np.unique(fight, return_inverse=True)
    sums = np.bincount(inv, weights=won.astype(np.float64))
    cnt = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(0)
    stats = []
    for _ in range(n_boot):
        c = rng.integers(0, len(u), len(u))
        stats.append(sums[c].sum() / cnt[c].sum())
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {
        "win": float(won.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": len(won),
        "fights": len(u),
    }


def paired_fight_diff(
    a: np.ndarray, b: np.ndarray, fight: np.ndarray, n_boot: int = 10_000
) -> dict[str, float]:
    """`mean(a) − mean(b)` paired by fight, fight-clustered 95% CI."""
    u, inv = np.unique(fight, return_inverse=True)
    d = np.bincount(inv, weights=(a.astype(float) - b.astype(float)))
    cnt = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(1)
    stats = []
    for _ in range(n_boot):
        c = rng.integers(0, len(u), len(u))
        stats.append(d[c].sum() / cnt[c].sum())
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {"diff": float(d.sum() / cnt.sum()), "lo": float(lo), "hi": float(hi)}


# ── anchor ───────────────────────────────────────────────────────────────


def logit_parity(
    json_path: str | None, jobs: list[dict[str, Any]], temperature: float = 1.0
) -> dict[str, Any]:
    """Torch twin vs Rust logits on every state `jobs` visit (Rust policy plays)."""
    import sts2_rs

    rust = Rl2Explain(json_path)
    twin = Rl2Torch(json_path)
    worst = 0.0
    n_states = 0
    order_ok = True
    for job in jobs:
        cs = rollout_starts.build_fight(job["row"], job["build_seed"]).combat_state()
        rng = np.random.default_rng(int(job["seed"]))
        steps = 0
        while steps < MAX_STEPS:
            acts = cs.legal_actions()
            if not acts:
                break
            if len(acts) >= 2:
                zr = rust.logits([(job, cs, acts)])[0]
                zt = twin.logits([(job, cs, acts)])[0]
                rf = sts2_rs.rollout_features(cs, version=1)
                order_ok &= [json.dumps(a, sort_keys=True) for a in rf["actions"]] == [
                    json.dumps(a, sort_keys=True) for a in acts
                ]
                worst = max(worst, float(np.max(np.abs(zr - zt))))
                n_states += 1
                p = np.exp((zr - zr.max()) / temperature)
                a = int(rng.choice(len(p), p=p / p.sum()))
            else:
                a = 0
            cs.apply(acts[a])
            steps += 1
    return {
        "states": n_states,
        "max_abs_diff": worst,
        "order_ok": bool(order_ok),
        "pass": bool(order_ok and worst <= 1e-4),
    }


def cmd_anchor(args: argparse.Namespace) -> int:
    import sts2_rs

    os.makedirs(args.out, exist_ok=True)
    log = vd.Progress(os.path.join(args.out, "progress.log"), "combat_ctx anchor")
    rows = G.read_bank(args.fights)
    if args.max_fights:
        rows = rows[: args.max_fights]
    jobs, skipped = make_jobs(rows, args.reps)
    log(f"{len(rows)} rows, {skipped} skipped (ValueError), {len(jobs)} jobs")
    res: dict[str, Any] = {"fights_file": args.fights, "rows": len(rows), "skipped": skipped,
                           "jobs": len(jobs), "reps": args.reps, "temperature": 1.0, "view": "raw",
                           "max_steps": MAX_STEPS, "net_sha256": vd.sha256_file(vd.BUILTIN_NET),
                           "engine": sts2_rs.src_hash()}  # fmt: skip
    t0 = time.time()
    res["parity"] = logit_parity(None, jobs[: args.parity_jobs])
    log(f"logit parity: {res['parity']} ({time.time() - t0:.0f}s)")
    with open(vd.BUILTIN_NET, encoding="utf-8") as f:
        lp = sts2_rs.LearnedPlayout(f.read())
    t0 = time.time()
    states = [rollout_starts.build_fight(j["row"], j["build_seed"]).combat_state() for j in jobs]
    raw = lp.play_many(states, [j["seed"] for j in jobs], temperature=1.0, no_early_end=False,
                       max_steps=MAX_STEPS, belief=False)  # fmt: skip
    from ppo.rollout_learn import _unpack_playouts

    p = _unpack_playouts(raw)
    won_rust = p["won"].astype(bool)
    fight = np.asarray([j["fight"] for j in jobs])
    res["rust"] = fight_ci(won_rust, fight)
    res["rust"]["decisions_per_playout"] = float(p["steps"].mean())
    res["rust"]["anchor_logged"] = RL2_ANCHOR
    res["rust"]["anchor_inside_ci"] = res["rust"]["lo"] <= RL2_ANCHOR <= res["rust"]["hi"]
    log(f"W_rust {res['rust']} ({time.time() - t0:.0f}s)")
    t0 = time.time()
    py = play_many("rl2", jobs, 1.0, False, "raw", MAX_STEPS, args.procs)
    won_py = np.asarray([r["won"] for r in py], bool)
    res["py"] = fight_ci(won_py, fight)
    res["py"]["decisions_per_playout"] = float(np.mean([r["decisions"] for r in py]))
    res["py"]["capped_share"] = float(np.mean([r["capped"] for r in py]))
    res["py"]["steps_per_playout"] = float(np.mean([r["steps"] for r in py]))
    d = paired_fight_diff(won_py, won_rust, fight)
    d["pass"] = abs(d["diff"]) <= 0.02 and d["lo"] <= 0 <= d["hi"]
    res["py_minus_rust"] = d
    log(f"W_py {res['py']}; W_py - W_rust {d} ({time.time() - t0:.0f}s)")
    with open(os.path.join(args.out, "anchor.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0


def select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = G.read_bank(args.fights)
    if args.fold is not None:
        held = G.holdout(rows)["all"]
        rows = [r for r in rows if r["file"] not in held and G.fold_of(r["file"]) == args.fold
                and r["room_type"] in args.rooms.split(",")]  # fmt: skip
    if args.max_fights:
        rows = rows[: args.max_fights]
    return rows


def cmd_run(args: argparse.Namespace) -> int:
    rows = select_rows(args)
    with open(G.BOSS800, encoding="utf-8") as f:
        b8 = {json.loads(line)["file"] for line in f if line.strip()}
    if is_student(args.policy) and any(r["file"] in b8 for r in rows):
        if not args.final:
            raise SystemExit("a student checkpoint plays boss800 only with --final and FROZEN.json")
        from .metrics import check_frozen

        check_frozen(args.final_dir, [args.policy.partition(":")[2]])
    jobs, skipped = make_jobs(rows, args.reps)
    ident = policy_identity(args.policy)
    t0 = time.time()
    res = play_many(
        args.policy, jobs, args.temperature, args.greedy, args.view, MAX_STEPS, args.procs
    )
    won = np.asarray([r["won"] for r in res], bool)
    fight = np.asarray([r["fight"] for r in res])
    out = {
        "policy": args.policy,
        **ident,
        "fights_file": args.fights,
        "fold": args.fold,
        "temperature": None if args.greedy else args.temperature,
        "greedy": args.greedy,
        "view": args.view,
        "skipped": skipped,
        **fight_ci(won, fight),
        "decisions_per_playout": float(np.mean([r["decisions"] for r in res])),
        "capped_share": float(np.mean([r["capped"] for r in res])),
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": out, "results": res}, f)
    return 0


def policy_identity(spec: str) -> dict[str, Any]:
    """The checkpoint's sha256 and (for a `train` checkpoint) arm and seed."""
    kind, _, arg = spec.partition(":")
    if not arg:
        return {"ckpt_sha256": None, "arm": kind, "seed": None}
    from .metrics import ckpt_identity, sha256

    try:
        ident = ckpt_identity(arg)
    except SystemExit:
        ident = {"arm": kind, "seed": None}
    return {"ckpt_sha256": sha256(arg), "arm": ident["arm"], "seed": ident["seed"]}


def _job_sig(r: dict[str, Any]) -> tuple:
    return (r["i"], r["fight"], r.get("file"), r["rep"])


def load_play(paths: list[str]) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """`(won [n_seeds, n_jobs], the jobs, the summaries)` of `play --out` files
    that played the same jobs under the same protocol."""
    won, jobs, summ = [], None, []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        res = sorted(d["results"], key=lambda r: r["i"])
        if any("file" not in r for r in res):
            raise SystemExit(f"{p}: results carry no run file (rerun `play`)")
        if jobs is None:
            jobs = res
        elif [_job_sig(r) for r in res] != [_job_sig(r) for r in jobs]:
            raise SystemExit(f"{p}: other jobs than {paths[0]}")
        won.append([bool(r["won"]) for r in res])
        summ.append(d["summary"])
    proto = {
        (s["fights_file"], s.get("fold"), s["greedy"], s["temperature"], s["view"]) for s in summ
    }
    if len(proto) != 1:
        raise SystemExit(
            f"{paths}: different fights / temperature / view: {sorted(map(str, proto))}"
        )
    shas = [s["ckpt_sha256"] for s in summ if s.get("ckpt_sha256")]
    if len(set(shas)) != len(shas):
        raise SystemExit(f"{paths}: one checkpoint counted twice")
    return np.asarray(won, np.float64), jobs or [], summ


def cmd_contrast(args: argparse.Namespace) -> int:
    """Paired play-alone contrast `a − b` (PREREG §5.6): seed-averaged, run-clustered."""
    from .metrics import bootstrap_contrast

    wa, ja, sa = load_play(args.a)
    wb, jb, sb = load_play(args.b)
    if [_job_sig(r) for r in ja] != [_job_sig(r) for r in jb]:
        raise SystemExit("the two policies played different jobs")
    runs = np.asarray([r["file"] for r in ja])
    keys = np.asarray([r["key"] for r in ja])
    res = bootstrap_contrast(wa, wb, runs, n_boot=args.n_boot, level=args.level)
    res["fight_clustered"] = bootstrap_contrast(
        wa, wb, keys, n_boot=args.n_boot, level=args.level, seed=1
    )
    out = {
        "a": {"policies": [s["policy"] for s in sa], "seeds": [s.get("seed") for s in sa],
              "win": float(wa.mean())},
        "b": {"policies": [s["policy"] for s in sb], "seeds": [s.get("seed") for s in sb],
              "win": float(wb.mean())},
        "jobs": len(ja),
        "runs": len(set(runs.tolist())),
        "fights": len(set(keys.tolist())),
        "greedy": sa[0]["greedy"],
        "view": sa[0]["view"],
        "level": args.level,
        "a_minus_b": res,
        "ci_excludes_0": bool(res["lo"] > 0 or res["hi"] < 0),
    }  # fmt: skip
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


TEMPERATURES = (0.25, 0.5, 1.0)


def cmd_select_temperature(args: argparse.Namespace) -> int:
    """T* per student: mean win rate over 4 reps on DEV-fold boss fights only.

    The output records the checkpoint it played (sha256, arm, training seed),
    so a T* can be matched to the checkpoint that is frozen; a checkpoint
    that changes while the temperatures are played is refused."""
    if args.fold not in {f for f, s in G.SCHEMES[args.scheme].items() if s == "dev"}:
        raise SystemExit(f"fold {args.fold} is not the DEV fold of {args.scheme}")
    args.rooms = "boss"
    args.fights = args.fights or G.BOSS800
    rows = select_rows(args)
    jobs, _ = make_jobs(rows, 4)
    ident = policy_identity(args.policy)
    wins = {}
    for t in TEMPERATURES:
        res = play_many(args.policy, jobs, t, False, "raw", MAX_STEPS, args.procs)
        wins[t] = float(np.mean([r["won"] for r in res]))
    if policy_identity(args.policy) != ident:
        raise SystemExit(f"{args.policy}: the checkpoint changed while T* was played")
    best = max(TEMPERATURES, key=lambda t: (wins[t], -abs(t - 1.0)))
    out = {"policy": args.policy, **ident, "fold": args.fold, "scheme": args.scheme, "fights": len(rows),
           "wins": {str(k): v for k, v in wins.items()}, "T_star": best}  # fmt: skip
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


def add_parser(sub: Any) -> None:
    a = sub.add_parser("anchor", help="rl2 play-alone anchor: logit parity, W_rust, W_py")
    a.add_argument("--fights", default=G.BOSS800)
    a.add_argument("--out", required=True)
    a.add_argument("--reps", type=int, default=4)
    a.add_argument("--max-fights", type=int, default=0)
    a.add_argument("--parity-jobs", type=int, default=200)
    a.add_argument("--procs", type=int, default=40)
    a.set_defaults(func=cmd_anchor)
    r = sub.add_parser("play", help="play whole fights with one policy, no search")
    r.add_argument(
        "--policy", required=True, help="rl2[:json] | rl2torch:json | ctx:best.pt | heur"
    )
    r.add_argument("--fights", default=G.BOSS800)
    r.add_argument("--fold", type=int, help="instead: fold-F bank fights (holdout removed)")
    r.add_argument("--rooms", default="boss")
    r.add_argument("--max-fights", type=int, default=0)
    r.add_argument("--reps", type=int, default=1)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--greedy", action="store_true")
    r.add_argument("--view", choices=("raw", "belief"), default="raw")
    r.add_argument("--procs", type=int, default=40)
    r.add_argument("--final", action="store_true")
    r.add_argument("--final-dir")
    r.add_argument("--out")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("select-temperature", help="T* in {0.25, 0.5, 1} on DEV-fold boss fights")
    s.add_argument("--policy", required=True)
    s.add_argument("--fights", required=True, help="the human bank (hbank.jsonl)")
    s.add_argument("--scheme", choices=("S1", "S2", "SMOKE"), required=True)
    s.add_argument("--fold", type=int, required=True, help="the scheme's DEV fold")
    s.add_argument("--max-fights", type=int, default=0)
    s.add_argument("--procs", type=int, default=40)
    s.add_argument("--out")
    s.set_defaults(func=cmd_select_temperature)
    c = sub.add_parser(
        "play-contrast", help="paired play-alone a − b over training seeds, run-clustered (§5.6)"
    )
    c.add_argument("--a", nargs="+", required=True, help="`play --out` files, one per seed")
    c.add_argument("--b", nargs="+", required=True, help="`play --out` files, one per seed")
    c.add_argument("--n-boot", type=int, default=10_000)
    c.add_argument("--level", type=float, default=0.95)
    c.add_argument("--out")
    c.set_defaults(func=cmd_contrast)
