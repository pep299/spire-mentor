"""G-D labels: `python -m ppo.deepbets.value_label {states,label,refs,bench}` (GH #830).

Plan Step 2 G-D, value-net-first section 2's state recipe plus the turn
afterstate stratum. Every playout is `sts2_rs.values_many` in an honest mode.

    states  Sample state SPECS from pilot trajectories of non-bench human-bank
            fights (`value_data.load_source`). Per searched decision s:
              * s itself (`decision`);
              * its distinct children by `observation_hash` (`child`), each
                tagged `chance` when the action moves the belief view's
                `rng_fingerprint`, plus an `endturn` member (s scored in
                `belief_endturn` mode: the honest Q of ending the turn now)
                when EndTurn is legal; together a sibling group `g:<id>`;
              * `--n-desc` descendants (`desc`), k ~ U{3..15} steps deep,
                moved 60% by the playout net at T=1, 20% at T=2, 20% uniformly;
              * for `--deep-frac` of decisions, descendant 0's sibling group
                (`deep_child` / `deep_endturn`, group `dg:<id>`).
            For `--after-frac` of searched train turn starts (`--probe-after-frac`
            in probe-test / probe-dev: their 1,000 afterstate groups need every
            turn), up to `--max-plans`
            deterministic plans of the turn (`after`, group `a:<id>`, CRN): a
            DFS over the belief view with every chance edge cut and plans
            deduplicated on `observation_hash`; the immediate EndTurn and the
            pilot's own plan (when deterministic) are always kept.
            Split: `sha1(run file) % 10`, 0 test, 1 dev, 2-9 train.
    label   K playouts per spec (`--k`, default 128) in the spec's mode, plus
            the honest input features; sharded `.npz`, resumable.
    refs    Reference playouts for the probe groups of one split: contested
            (parent p in [lo, hi]) boss and elite sibling groups, decided-win
            groups (p > hi), and afterstate groups. `--n-ref` playouts per
            member (the first `--n-r` are the reference R, the rest the
            estimator pool) plus `--n-fresh` fresh ones for the sanity checks.
            Sibling groups use independent seeds per member; afterstate groups
            CRN seeds shared by every plan of the turn.
    bench   Playouts per CPU second of `values_many` on spec states, one thread.

Worker processes run with `RAYON_NUM_THREADS=1`; `--procs` is the CPU budget.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import random
import sys
import time
from multiprocessing import get_context
from typing import Any

import numpy as np

from ppo.deepbets import value_data as vd

LABEL_SALT = "label"
REF_SALT = "ref"
DESC_POLICIES = (("T1", 0.6), ("T2", 0.2), ("U", 0.2))

# ── sampler ──────────────────────────────────────────────────────────────

_LP: Any = None  # the playout net as a LearnedPlayout, per worker


def _playout_net() -> Any:
    global _LP
    if _LP is None:
        import sts2_rs

        ident = vd.playout_net_identity()
        with open(ident["path"], encoding="utf-8") as f:
            _LP = sts2_rs.LearnedPlayout(f.read())
    return _LP


def _terminal(s: Any) -> bool:
    return bool(s.terminal_value()[0])


def _pick_policy(rng: random.Random) -> str:
    x = rng.random()
    for name, p in DESC_POLICIES:
        if x < p:
            return name
        x -= p
    return DESC_POLICIES[-1][0]


def _net_action(s: Any, temperature: float, rng: random.Random) -> dict[str, Any]:
    e = _playout_net().explain(s)
    z = np.asarray(e["logits"], np.float64) / temperature
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return e["actions"][int(np.searchsorted(np.cumsum(p), rng.random() * p.sum()))]


def descend(cs: Any, rng: random.Random, k: int, policy: str) -> tuple[list[dict[str, Any]], Any]:
    """Walk up to `k` steps from `cs` under `policy`; stops before a terminal step.

    Returns:
        `(path, state)`: the actions taken and the live state they reach.
    """
    s = cs.clone()
    path: list[dict[str, Any]] = []
    for _ in range(k):
        acts = s.legal_actions()
        if not acts:
            break
        if policy == "U":
            a = rng.choice(acts)
        else:
            a = _net_action(s, 1.0 if policy == "T1" else 2.0, rng)
        c = s.clone()
        try:
            c.apply(a)
        except BaseException:
            break
        if _terminal(c):
            break
        s = c
        path.append(a)
    return path, s


def sibling_specs(
    cs: Any, base: dict[str, Any], prefix: list[dict[str, Any]], parent: str, gid: str, deep: bool
) -> list[dict[str, Any]]:
    """The sibling group of `cs`: distinct children plus the EndTurn member.

    `cs` is the TRUE state reached by `prefix` from the spec's decision state;
    chance tagging runs on its belief view (`rng_fingerprint` is valid only there).
    """
    out: list[dict[str, Any]] = []
    acts = cs.legal_actions()
    bv = cs.belief_view(vd.CHANCE_VIEW_SEED)
    same = bv.legal_actions() == acts
    fp0 = bv.rng_fingerprint() if same else None
    seen: set[int] = set()
    child_kind, end_kind = ("deep_child", "deep_endturn") if deep else ("child", "endturn")
    for i, a in enumerate(acts):
        if a.get("action") == "end_turn":
            out.append(
                {
                    **base,
                    "id": f"{parent}:E",
                    "kind": end_kind,
                    "mode": "belief_endturn",
                    "path": list(prefix),
                    "ohash": int(cs.observation_hash()),
                    "group": gid,
                    "parent": parent,
                    "chance": False,
                }
            )
            continue
        c = cs.clone()
        try:
            c.apply(a)
        except BaseException:
            continue
        h = int(c.observation_hash())
        if h in seen:
            continue
        seen.add(h)
        chance = None
        if same:
            cv = bv.clone()
            try:
                cv.apply(a)
                chance = cv.rng_fingerprint() != fp0
            except BaseException:
                chance = None
        out.append(
            {
                **base,
                "id": f"{parent}:c{i}",
                "kind": child_kind,
                "mode": "belief",
                "path": [*prefix, a],
                "ohash": h,
                "group": gid,
                "parent": parent,
                "chance": chance,
                "terminal": _terminal(c),
            }
        )
    return out


def enumerate_plans(
    bv: Any, cap: int, rng: random.Random
) -> tuple[list[tuple[Any, list[dict[str, Any]]]], bool, int]:
    """Every deterministic plan end of the turn from the belief view `bv`.

    DFS over non-EndTurn actions (potions and card choices included); an
    action that moves `rng_fingerprint` is a chance edge and is cut; positions
    are deduplicated on `observation_hash`. A plan end is a position where
    EndTurn is legal, or a terminal one (a lethal plan).

    Returns:
        `(nodes, capped, n_seen)`: `(state, path)` per plan end, whether the
        `cap` on distinct positions stopped the walk, and positions seen.
    """
    seen = {int(bv.observation_hash())}
    nodes: list[tuple[Any, list[dict[str, Any]]]] = []
    stack: list[tuple[Any, list[dict[str, Any]]]] = [(bv, [])]
    capped = False
    while stack:
        s, path = stack.pop()
        if _terminal(s):
            nodes.append((s, path))
            continue
        acts = s.legal_actions()
        if any(a.get("action") == "end_turn" for a in acts):
            nodes.append((s, path))
        if capped:
            continue
        order = [a for a in acts if a.get("action") != "end_turn"]
        rng.shuffle(order)
        fp = s.rng_fingerprint()
        for a in order:
            c = s.clone()
            try:
                c.apply(a)
            except BaseException:
                continue
            if c.rng_fingerprint() != fp:
                continue
            h = int(c.observation_hash())
            if h in seen:
                continue
            seen.add(h)
            stack.append((c, [*path, a]))
            if len(seen) >= cap:
                capped = True
                break
    return nodes, capped, len(seen)


def _base(rec: dict[str, Any], t: int) -> dict[str, Any]:
    return {
        "rec": rec["_idx"],
        "key": rec["key"],
        "file": rec["file"],
        "rep": rec["rep"],
        "split": vd.split_of(rec["file"]),
        "room": rec["room"],
        "t": t,
        "hash": int(rec["hashes"][t]),
        "view": None,
    }


def sample_job(job: tuple[Any, ...]) -> dict[str, Any]:
    """Specs for one trajectory's chosen decisions and turn starts (a Pool worker)."""
    rec, decisions, turns, cfg = job
    rid0 = f"{rec['key']}|{rec['rep']}"
    rng = random.Random(f"{cfg['seed']}|{rid0}")
    need = set(decisions) | set(turns)
    ends: dict[int, int] = {}
    for t0 in turns:
        for t1 in range(t0, len(rec["actions"])):
            if rec["actions"][t1].get("action") == "end_turn":
                ends[t0] = t1
                need.add(t1)
                break
    states, verified = vd.walk_record(rec, need)
    specs: list[dict[str, Any]] = []
    stats = collections.Counter()
    stats["verified_decisions"] += verified
    stats["records"] += 1
    for t in sorted(set(decisions) | set(turns)):
        cs = states.get(t)
        if cs is None:
            stats["unverified"] += 1
            continue
        base = _base(rec, t)
        rid = f"{rid0}|{t}"
        specs.append(
            {**base, "id": rid, "kind": "decision", "mode": "belief", "path": [],
             "ohash": int(cs.observation_hash()), "group": None, "parent": None}
        )  # fmt: skip
        stats["decisions"] += 1
        if t in decisions:
            specs += sibling_specs(cs, base, [], rid, f"g:{rid}", deep=False)
            for d in range(cfg["n_desc"]):
                k = rng.randint(cfg["desc_min"], cfg["desc_max"])
                pol = _pick_policy(rng)
                path, s = descend(cs, rng, k, pol)
                if not path:
                    stats["desc_empty"] += 1
                    continue
                did = f"{rid}:d{d}"
                specs.append(
                    {**base, "id": did, "kind": "desc", "mode": "belief", "path": path,
                     "ohash": int(s.observation_hash()), "group": None, "parent": rid,
                     "pol": pol, "depth": len(path)}
                )  # fmt: skip
                if d == 0 and rng.random() < cfg["deep_frac"]:
                    specs += sibling_specs(s, base, path, did, f"dg:{did}", deep=True)
                    stats["deep_groups"] += 1
        if t in turns:
            specs += afterstate_specs(
                rec, t, cs, states.get(ends.get(t, -1)), base, rid, rng, cfg, stats
            )
    stats["specs"] = len(specs)
    return {"specs": specs, "stats": dict(stats)}


def afterstate_specs(
    rec: dict[str, Any],
    t: int,
    cs: Any,
    pilot_end: Any,
    base: dict[str, Any],
    rid: str,
    rng: random.Random,
    cfg: dict[str, Any],
    stats: collections.Counter,
) -> list[dict[str, Any]]:
    """Up to `max_plans` afterstates of the turn starting at decision `t`."""
    vseed = vd.seed_base("view", rid) & vd.SEED_MASK
    bv = cs.belief_view(vseed)
    if not any(a.get("action") == "end_turn" for a in bv.legal_actions()):
        stats["after_no_endturn"] += 1
        return []
    t0 = time.process_time()
    nodes, capped, n_seen = enumerate_plans(bv, cfg["plan_cap"], rng)
    stats["after_turns"] += 1
    stats["after_capped"] += int(capped)
    stats["after_plans_total"] += len(nodes)
    stats["after_dfs_ms"] += int((time.process_time() - t0) * 1000)
    pilot_h = int(pilot_end.observation_hash()) if pilot_end is not None else None
    root = [i for i, (_, p) in enumerate(nodes) if not p]
    pilot = [
        i
        for i, (s, _) in enumerate(nodes)
        if pilot_h is not None and int(s.observation_hash()) == pilot_h
    ]
    keep = list(dict.fromkeys(root + pilot[:1]))
    rest = [i for i in range(len(nodes)) if i not in keep]
    rng.shuffle(rest)
    keep += rest[: max(0, cfg["max_plans"] - len(keep))]
    stats["after_pilot_found"] += int(bool(pilot))
    out = []
    for j, i in enumerate(keep):
        s, path = nodes[i]
        out.append(
            {**base, "id": f"{rid}:a{j}", "kind": "after", "mode": "belief_endturn",
             "view": vseed, "path": path, "ohash": int(s.observation_hash()),
             "group": f"a:{rid}", "parent": rid, "crn": True, "pilot": i in pilot[:1],
             "terminal": _terminal(s), "n_plans": len(nodes), "capped": capped}
        )  # fmt: skip
    return out


def choose_jobs(
    recs: list[dict[str, Any]], args: argparse.Namespace
) -> list[tuple[dict[str, Any], list[int], list[int]]]:
    """`(record, decisions, turn starts)` per trajectory, seeded and capped per split."""
    rng = random.Random(args.seed)
    order = list(range(len(recs)))
    rng.shuffle(order)
    if args.traj_frac < 1.0:
        order = order[: round(len(order) * args.traj_frac)]

    def caps(items: list[str] | None) -> dict[str, int]:
        return {k: int(v) for k, v in (x.split("=") for x in items)} if items else {}

    cap_d, cap_t = caps(args.cap_decisions), caps(args.cap_turns)
    used_d: collections.Counter[str] = collections.Counter()
    used_t: collections.Counter[str] = collections.Counter()
    jobs = []
    for i in order:
        rec = recs[i]
        split = vd.split_of(rec["file"])
        acts = rec["actions"]
        searched = [t for t, s in enumerate(rec["searched"]) if s]
        starts = {t for t in range(len(acts)) if t == 0 or acts[t - 1].get("action") == "end_turn"}
        dec = [t for t in searched if rng.random() < args.decision_frac]
        after_frac = args.after_frac if split == "train" else args.probe_after_frac
        turns = [t for t in searched if t in starts and rng.random() < after_frac]
        if args.per_traj:
            dec = sorted(rng.sample(dec, min(len(dec), args.per_traj)))
            turns = sorted(rng.sample(turns, min(len(turns), args.per_traj)))
        if cap_d or cap_t:
            dec = dec[: max(0, cap_d.get(split, 0) - used_d[split])]
            turns = turns[: max(0, cap_t.get(split, 0) - used_t[split])]
            used_d[split] += len(dec)
            used_t[split] += len(turns)
        if dec or turns:
            jobs.append((rec, dec, turns))
    return jobs


def cmd_states(args: argparse.Namespace) -> int:
    os.makedirs(args.out, exist_ok=True)
    log = vd.Progress(os.path.join(args.out, "progress.log"), "states")
    env = vd.worker_env(args.rollout_net)
    vd.init_worker(env)
    header, recs, meta = vd.load_source(args.traj, args.hbank, args.manifest)
    log(f"source: {json.dumps(meta)}")
    jobs = choose_jobs(recs, args)
    cfg = {
        "seed": args.seed,
        "n_desc": args.n_desc,
        "desc_min": args.desc_min,
        "desc_max": args.desc_max,
        "deep_frac": args.deep_frac,
        "max_plans": args.max_plans,
        "plan_cap": args.plan_cap,
    }
    log(f"{len(jobs)} trajectories, {sum(len(j[1]) for j in jobs)} decisions, "
        f"{sum(len(j[2]) for j in jobs)} turn starts; procs {args.procs}")  # fmt: skip
    work = [(rec, dec, turns, cfg) for rec, dec, turns in jobs]
    stats = collections.Counter()
    n_specs = 0
    shard: list[dict[str, Any]] = []
    shard_i = 0

    def flush() -> None:
        nonlocal shard, shard_i
        if not shard:
            return
        path = os.path.join(args.out, f"specs-{shard_i:05d}.jsonl")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for s in shard:
                f.write(json.dumps(s, separators=(",", ":")) + "\n")
        pathlib.Path(path + ".tmp").replace(path)
        shard, shard_i = [], shard_i + 1

    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(args.procs, initializer=vd.init_worker, initargs=(env,)) as pool:
        for n, res in enumerate(pool.imap_unordered(sample_job, work, chunksize=4), 1):
            stats.update(res["stats"])
            shard += res["specs"]
            n_specs += len(res["specs"])
            if len(shard) >= args.shard_size:
                flush()
            if n % 50 == 0 or n == len(work):
                el = time.time() - t0
                log(f"{n}/{len(work)} trajectories, {n_specs} specs, {el:.0f}s, "
                    f"eta {el / n * (len(work) - n):.0f}s")  # fmt: skip
    flush()
    by = collections.Counter((s_split, s_kind) for s_split, s_kind in _iter_kinds(args.out))
    summary = {
        "schema": "deepbets.value_specs.v1",
        "args": vars(args),
        "source": meta,
        "traj_header_spec": header.get("spec"),
        "playout_net": vd.playout_net_identity(),
        "engine": vd.engine_identity(),
        "stats": dict(stats),
        "specs": n_specs,
        "by_split_kind": {f"{a}/{b}": c for (a, b), c in sorted(by.items())},
        "secs": round(time.time() - t0, 1),
    }
    vd.write_json(os.path.join(args.out, "manifest.json"), summary)
    log(f"done: {n_specs} specs; stats {json.dumps(dict(stats))}")
    return 0


def _iter_kinds(d: str):
    for p in vd.shard_paths(d, "specs-", ".jsonl"):
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                yield s["split"], s["kind"]


# ── labels ───────────────────────────────────────────────────────────────

_RECS: list[dict[str, Any]] | None = None


def _records(traj: str) -> list[dict[str, Any]]:
    global _RECS
    if _RECS is None:
        _RECS = vd.read_traj_records(traj)
        for i, r in enumerate(_RECS):
            r["_idx"] = i
    return _RECS


def rebuild_specs(traj: str, specs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], Any]]:
    """`(spec, state)` for specs of one trajectory; unrebuildable ones are dropped."""
    rec = _records(traj)[specs[0]["rec"]]
    states, _ = vd.walk_record(rec, {s["t"] for s in specs})
    out = []
    for s in specs:
        cs = states.get(s["t"])
        if cs is None or int(cs.ordered_hash()) != int(s["hash"]):
            continue
        try:
            out.append((s, vd.spec_state(cs, s)))
        except (RuntimeError, ValueError):
            continue
    return out


def playout_stats(vals: np.ndarray, won: np.ndarray, term: np.ndarray) -> dict[str, float]:
    """The label of one state from its playouts' `search_value`s and outcomes."""
    n = len(vals)
    nw = int(won.sum())
    nl = n - nw
    return {
        "n": n,
        "mean": float(vals.mean()),
        "var": float(vals.var()),
        "n_won": nw,
        "mean_win": float(vals[won].mean()) if nw else math.nan,
        "mean_loss": float(vals[~won].mean()) if nl else math.nan,
        "n_term": int(term.sum()),
    }


def run_playouts(state: Any, mode: str, seed_list: list[int]) -> tuple[np.ndarray, ...]:
    import sts2_rs

    assert mode in vd.HONEST_MODES, mode
    rows = sts2_rs.values_many([state] * len(seed_list), seed_list, mode=mode)
    vals = np.fromiter((r["search_value"] for r in rows), np.float32, len(rows))
    won = np.fromiter((r["won"] for r in rows), bool, len(rows))
    term = np.fromiter((r["terminal"] for r in rows), bool, len(rows))
    return vals, won, term


def label_job(job: tuple[Any, ...]) -> dict[str, Any]:
    """Labels and features for one trajectory's specs (a Pool worker)."""
    traj, specs, k = job
    t0 = time.process_time()
    play = 0.0
    rows = []
    built = rebuild_specs(traj, specs)
    if len(built) < len(specs):
        have = {s["id"] for s, _ in built}
        rows += [{"id": s["id"], "error": "rebuild"} for s in specs if s["id"] not in have]
    for spec, st in built:
        try:
            feats = vd.features(st, spec["mode"] == "belief_endturn")
        except BaseException as e:
            rows.append({"id": spec["id"], "error": f"features: {e!r}"})
            continue
        key = spec["group"] if spec.get("crn") else spec["id"]
        p0 = time.process_time()
        try:
            vals, won, term = run_playouts(st, spec["mode"], vd.seeds(LABEL_SALT, key, k))
        except BaseException as e:
            rows.append({"id": spec["id"], "error": f"values_many: {e!r}"})
            continue
        play += time.process_time() - p0
        rows.append({"spec": spec, "feats": feats, "label": playout_stats(vals, won, term)})
    return {"rows": rows, "n_specs": len(specs), "cpu_s": time.process_time() - t0, "play_s": play}


class FeatureShard:
    """Columnar `.npz` writer for labelled states (vocabularies shard-local)."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def write(self, path: str) -> None:
        rows = self.rows
        n = len(rows)
        vocab: dict[str, int] = {"<pad>": 0}
        mvocab: dict[str, int] = {"<none>": 0}
        tok, tok_zone, tok_row = [], [], []
        mon_sp = np.zeros((n, vd.MONSTER_SLOTS), np.int32)
        mon_f = np.zeros((n, vd.MONSTER_SLOTS, vd.MON_FEATS), np.float32)
        for i, r in enumerate(rows):
            f = r["feats"]
            for z, cards in enumerate([*f["zones"], f["potions"]]):
                for c in cards:
                    tok.append(vocab.setdefault(c, len(vocab)))
                    tok_zone.append(z)
                    tok_row.append(i)
            for j, (sp, hpf, blk, alive) in enumerate(f["mons"]):
                mon_sp[i, j] = mvocab.setdefault(sp, len(mvocab))
                mon_f[i, j] = (hpf, blk, float(alive))
        spec = [r["spec"] for r in rows]
        lab = [r["label"] for r in rows]
        arrays = {
            "id": np.array([s["id"] for s in spec]),
            "rec": np.array([s["rec"] for s in spec], np.int64),
            "parent": np.array([s.get("parent") or "" for s in spec]),
            "group": np.array([s.get("group") or "" for s in spec]),
            "fight": np.array([s["key"] for s in spec]),
            "split": np.array([vd.SPLIT_NAMES.index(s["split"]) for s in spec], np.int8),
            "room": np.array([vd.ROOMS.index(s["room"]) for s in spec], np.int8),
            "kind": np.array([vd.KINDS.index(s["kind"]) for s in spec], np.int8),
            "after": np.array([r["feats"]["after"] for r in rows], np.int8),
            "chance": np.array(
                [-1 if s.get("chance") is None else int(s["chance"]) for s in spec], np.int8
            ),
            "step": np.array([r["feats"]["step"] for r in rows], np.float32).reshape(
                n, vd.STEP_DIM
            ),
            "wide": np.array([r["feats"]["wide"] for r in rows], np.float32).reshape(
                n, vd.WIDE_DIM
            ),
            "texel": np.array([r["feats"]["texel"] for r in rows], np.float32),
            "tok": np.array(tok, np.int32),
            "tok_zone": np.array(tok_zone, np.int8),
            "tok_row": np.array(tok_row, np.int32),
            "vocab": np.array(sorted(vocab, key=vocab.get)),
            "mon_sp": mon_sp,
            "mon_f": mon_f,
            "mvocab": np.array(sorted(mvocab, key=mvocab.get)),
        }
        for key in ("n", "n_won", "n_term"):
            arrays[key] = np.array([lb[key] for lb in lab], np.int32)
        for key in ("mean", "var", "mean_win", "mean_loss"):
            arrays[key] = np.array([lb[key] for lb in lab], np.float32)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        pathlib.Path(tmp).replace(path)


def _spec_groups(spec_dir: str, splits: set[str] | None) -> list[list[dict[str, Any]]]:
    by_rec: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for s in vd.read_jsonl(vd.shard_paths(spec_dir, "specs-", ".jsonl")):
        if splits is None or s["split"] in splits:
            by_rec[s["rec"]].append(s)
    return [by_rec[k] for k in sorted(by_rec)]


def _done_values(out_dir: str, prefix: str, field: str) -> tuple[set[Any], int]:
    """Values of `field` in the finished shards of `out_dir`, and the next shard index."""
    done: set[Any] = set()
    paths = vd.shard_paths(out_dir, prefix, ".npz")
    for p in paths:
        done.update(np.load(p)[field].tolist())
    nxt = 1 + max((int(os.path.basename(p)[len(prefix) : -4]) for p in paths), default=-1)
    return done, nxt


def cmd_label(args: argparse.Namespace) -> int:
    os.makedirs(args.out, exist_ok=True)
    log = vd.Progress(os.path.join(args.out, "progress.log"), "label")
    env = vd.worker_env(args.rollout_net)
    vd.init_worker(env)
    smeta = json.load(open(os.path.join(args.specs, "manifest.json"), encoding="utf-8"))
    net = vd.playout_net_identity()
    vd.check_same_net(smeta["playout_net"], net, "specs vs this run")
    traj = smeta["args"]["traj"]
    groups = _spec_groups(args.specs, set(args.splits) if args.splits else None)
    total = sum(len(g) for g in groups)
    meta = {
        "schema": "deepbets.value_labels.v1",
        "args": vars(args),
        "k": args.k,
        "specs": os.path.abspath(args.specs),
        "playout_net": net,
        "engine": vd.engine_identity(),
        "engine_env": vd.engine_env(),
        "modes": list(vd.HONEST_MODES),
    }
    mpath = os.path.join(args.out, "manifest.json")
    if os.path.exists(mpath):
        old = json.load(open(mpath, encoding="utf-8"))
        vd.check_same_net(old["playout_net"], net, "existing labels vs this run")
        if old["k"] != args.k:
            raise SystemExit(f"{args.out} holds K={old['k']} labels; this run asks K={args.k}")
    vd.write_json(mpath, meta)
    # Resume: a trajectory is done when a shard holds its record index.
    done, shard_i = _done_values(args.out, "label-", "rec")
    todo = [g for g in groups if g[0]["rec"] not in done]
    left = sum(len(g) for g in todo)
    log(f"{total} specs in {len(groups)} trajectories, K={args.k}, procs {args.procs}; "
        f"{len(groups) - len(todo)} trajectories already labelled, {left} specs to go")  # fmt: skip
    t0 = time.time()
    done_states = errors = n_traj = 0
    cpu = play = 0.0
    shard = FeatureShard()
    jobs = [(traj, g, args.k) for g in todo]
    ctx = get_context("spawn")
    with ctx.Pool(args.procs, initializer=vd.init_worker, initargs=(env,)) as pool:
        for res in pool.imap_unordered(label_job, jobs):
            cpu += res["cpu_s"]
            play += res["play_s"]
            n_traj += 1
            left -= res["n_specs"]
            for r in res["rows"]:
                if "error" in r:
                    errors += 1
                    if errors <= 20:
                        log(f"error {r['id']}: {r['error']}")
                else:
                    shard.add(r)
                    done_states += 1
            if n_traj % args.chunk == 0 or n_traj == len(jobs):
                if len(shard):
                    shard.write(os.path.join(args.out, f"label-{shard_i:05d}.npz"))
                    shard_i += 1
                    shard = FeatureShard()
                el = time.time() - t0
                rate = done_states * args.k / max(cpu, 1e-9)
                log(f"{n_traj}/{len(jobs)} trajectories, {done_states} states, errors {errors}, "
                    f"{el:.0f}s wall; per CPU {rate:.0f} playouts/s all-in, "
                    f"{done_states * args.k / max(play, 1e-9):.0f}/s inside values_many; "
                    f"load {os.getloadavg()[0]:.0f}; "
                    f"eta {left * args.k / max(rate, 1e-9) / args.procs:.0f}s")  # fmt: skip
    log(f"done: {done_states} states, {errors} errors, {time.time() - t0:.0f}s")
    return 0


# ── references ───────────────────────────────────────────────────────────


def load_labels(label_dir: str, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Concatenate the scalar `fields` of every label shard."""
    parts: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    for p in vd.shard_paths(label_dir, "label-", ".npz"):
        z = np.load(p)
        for f in fields:
            parts[f].append(z[f])
    return {f: np.concatenate(v) if v else np.zeros(0) for f, v in parts.items()}


def select_groups(
    specs: list[dict[str, Any]], labels: dict[str, np.ndarray], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """The probe groups of one split, stratified and seeded.

    A group's eligible members are its deterministic children (chance edges
    excluded: their reference would be one realized outcome, not the action's
    value) and its EndTurn member, or every plan of an afterstate group; a
    group needs two. Stratum by the parent's label p = n_won / n.
    """
    p_of = {
        str(i): float(w) / max(1, int(n))
        for i, w, n in zip(labels["id"], labels["n_won"], labels["n"], strict=True)
    }
    have = set(p_of)
    members: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for s in specs:
        if s.get("group") and s["id"] in have and s.get("chance") is not True:
            if s["kind"] in ("child", "deep_child") and s.get("chance") is None:
                continue
            members[s["group"]].append(s)
    rng = random.Random(f"refs|{args.seed}|{args.split}")
    pools: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for gid, ms in sorted(members.items()):
        if len(ms) < 2:
            continue
        parent = ms[0]["parent"]
        if parent not in p_of:
            continue
        p = p_of[parent]
        if gid.startswith("a:"):
            stratum = "afterstate"
        elif p > args.hi:
            stratum = "decided_win"
        elif p >= args.lo:
            stratum = f"contested_{ms[0]['room']}"
        else:
            continue
        if stratum == "afterstate" and p < args.lo and not args.after_any:
            continue
        pools[stratum].append({"group": gid, "stratum": stratum, "p_parent": p, "members": ms,
                               "room": ms[0]["room"], "fight": ms[0]["key"], "crn": gid.startswith("a:"),
                               "deep": gid.startswith("dg:")})  # fmt: skip
    want = {
        "contested_boss": args.contested_boss,
        "contested_elite": args.contested_elite,
        "decided_win": args.decided,
        "afterstate": args.after,
    }
    out = []
    for stratum, n in want.items():
        pool = pools.get(stratum, [])
        rng.shuffle(pool)
        out += pool[: n if n >= 0 else len(pool)]
    return out


def ref_job(job: tuple[Any, ...]) -> dict[str, Any]:
    """Reference playouts for the groups of one trajectory (a Pool worker)."""
    traj, groups, n_total = job
    t0 = time.process_time()
    specs = [m for g in groups for m in g["members"]]
    built = {s["id"]: st for s, st in rebuild_specs(traj, specs)}
    out = []
    for g in groups:
        if any(m["id"] not in built for m in g["members"]):
            out.append({"group": g["group"], "error": "rebuild"})
            continue
        vals = np.zeros((len(g["members"]), n_total), np.float32)
        for j, m in enumerate(g["members"]):
            key = g["group"] if g["crn"] else m["id"]
            v, _, _ = run_playouts(built[m["id"]], m["mode"], vd.seeds(REF_SALT, key, n_total))
            vals[j] = v
        out.append({"group": g, "vals": vals})
    return {"groups": out, "cpu_s": time.process_time() - t0, "playouts": len(specs) * n_total}


def cmd_refs(args: argparse.Namespace) -> int:
    os.makedirs(args.out, exist_ok=True)
    log = vd.Progress(os.path.join(args.out, "progress.log"), f"refs-{args.split}")
    env = vd.worker_env(args.rollout_net)
    vd.init_worker(env)
    smeta = json.load(open(os.path.join(args.specs, "manifest.json"), encoding="utf-8"))
    lmeta = json.load(open(os.path.join(args.labels, "manifest.json"), encoding="utf-8"))
    net = vd.playout_net_identity()
    vd.check_same_net(smeta["playout_net"], net, "specs vs this run")
    vd.check_same_net(lmeta["playout_net"], net, "labels vs this run")
    traj = smeta["args"]["traj"]
    specs = [
        s
        for s in vd.read_jsonl(vd.shard_paths(args.specs, "specs-", ".jsonl"))
        if s["split"] == args.split
    ]
    labels = load_labels(args.labels, ("id", "n", "n_won"))
    groups = select_groups(specs, labels, args)
    counts = collections.Counter(g["stratum"] for g in groups)
    n_total = args.n_ref + args.n_fresh
    assert 0 < args.n_r < args.n_ref
    log(f"{len(groups)} groups {dict(counts)}; {sum(len(g['members']) for g in groups)} members x "
        f"{n_total} playouts; procs {args.procs}")  # fmt: skip
    by_rec: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for g in groups:
        by_rec[g["members"][0]["rec"]].append(g)
    jobs = [(traj, by_rec[k], n_total) for k in sorted(by_rec)]
    meta = {
        "schema": "deepbets.value_refs.v1",
        "args": vars(args),
        "split": args.split,
        "n_ref": args.n_ref,
        "n_r": args.n_r,
        "n_fresh": args.n_fresh,
        "strata": dict(counts),
        "specs": os.path.abspath(args.specs),
        "labels": os.path.abspath(args.labels),
        "playout_net": net,
        "engine": vd.engine_identity(),
        "engine_env": vd.engine_env(),
    }
    mpath = os.path.join(args.out, "manifest.json")
    if os.path.exists(mpath):
        old = json.load(open(mpath, encoding="utf-8"))
        same = ("n_ref", "n_r", "n_fresh", "split")
        if any(old[k] != meta[k] for k in same) or old["strata"] != meta["strata"]:
            raise SystemExit(f"{args.out} holds references of another design; use a new --out")
    vd.write_json(mpath, meta)
    # Resume: a group is done when a shard holds it.
    done_groups, shard_i = _done_values(args.out, "refs-", "group")
    jobs = [
        (t, [g for g in gs if g["group"] not in done_groups], n)
        for t, gs, n in jobs
        if any(g["group"] not in done_groups for g in gs)
    ]
    left = sum(len(g["members"]) for _, gs, _ in jobs for g in gs) * n_total
    log(f"{len(done_groups)} groups already done; {len(jobs)} trajectories to go")
    t0 = time.time()
    cpu = 0.0
    playouts = errors = n_traj = 0
    buf: list[dict[str, Any]] = []
    ctx = get_context("spawn")
    with ctx.Pool(args.procs, initializer=vd.init_worker, initargs=(env,)) as pool:
        for res in pool.imap_unordered(ref_job, jobs):
            cpu += res["cpu_s"]
            playouts += res["playouts"]
            left -= res["playouts"]
            n_traj += 1
            for g in res["groups"]:
                if "error" in g:
                    errors += 1
                else:
                    buf.append(g)
            if n_traj % args.chunk == 0 or n_traj == len(jobs):
                if buf:
                    write_refs(os.path.join(args.out, f"refs-{shard_i:05d}.npz"), buf)
                    shard_i += 1
                    buf = []
                rate = playouts / max(cpu, 1e-9)
                log(f"{n_traj}/{len(jobs)} trajectories, errors {errors}, {playouts} playouts, "
                    f"{rate:.0f}/cpu-s, load {os.getloadavg()[0]:.0f}, {time.time() - t0:.0f}s, "
                    f"eta {left / max(rate, 1e-9) / args.procs:.0f}s")  # fmt: skip
    log(f"done: {time.time() - t0:.0f}s, {errors} groups failed to rebuild")
    return 0


def write_refs(path: str, groups: list[dict[str, Any]]) -> None:
    off = [0]
    mids, mkinds, vals = [], [], []
    for r in groups:
        g = r["group"]
        for m in g["members"]:
            mids.append(m["id"])
            mkinds.append(vd.KINDS.index(m["kind"]))
        vals.append(r["vals"])
        off.append(off[-1] + len(g["members"]))
    arrays = {
        "group": np.array([r["group"]["group"] for r in groups]),
        "stratum": np.array([r["group"]["stratum"] for r in groups]),
        "room": np.array([r["group"]["room"] for r in groups]),
        "fight": np.array([r["group"]["fight"] for r in groups]),
        "crn": np.array([r["group"]["crn"] for r in groups], bool),
        "deep": np.array([r["group"]["deep"] for r in groups], bool),
        "p_parent": np.array([r["group"]["p_parent"] for r in groups], np.float32),
        "off": np.array(off, np.int64),
        "member": np.array(mids),
        "member_kind": np.array(mkinds, np.int8),
        "vals": np.concatenate(vals) if vals else np.zeros((0, 0), np.float32),
    }
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    pathlib.Path(tmp).replace(path)


# ── throughput ───────────────────────────────────────────────────────────


def cmd_bench(args: argparse.Namespace) -> int:
    """values_many playouts per CPU second on spec states, single-threaded."""
    os.environ["RAYON_NUM_THREADS"] = "1"
    vd.init_worker(vd.worker_env(args.rollout_net))
    smeta = json.load(open(os.path.join(args.specs, "manifest.json"), encoding="utf-8"))
    traj = smeta["args"]["traj"]
    groups = _spec_groups(args.specs, None)
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    # A few specs of each mode from many trajectories: the cost varies far
    # more between fights than within one.
    built = []
    need = collections.Counter({m: args.n for m in vd.HONEST_MODES})
    for g in groups:
        pick = []
        for mode in vd.HONEST_MODES:
            ms = [s for s in g if s["mode"] == mode]
            pick += rng.sample(ms, min(len(ms), args.per_traj, need[mode]))
        if pick:
            built += rebuild_specs(traj, pick)
            need.subtract(s["mode"] for s in pick)
        if all(v <= 0 for v in need.values()):
            break
    rng.shuffle(built)
    trajs = len({s["rec"] for s, _ in built})
    out: dict[str, Any] = {"load1_start": os.getloadavg()[0], "k": args.k, "trajectories": trajs}
    for mode in vd.HONEST_MODES:
        sel = [(s, st) for s, st in built if s["mode"] == mode][: args.n]
        cpu = wall = 0.0
        n = 0
        steps_term = 0
        for s, st in sel:
            seeds = vd.seeds("bench", s["id"], args.k)
            w0, c0 = time.perf_counter(), time.process_time()
            _, _, term = run_playouts(st, mode, seeds)
            wall += time.perf_counter() - w0
            cpu += time.process_time() - c0
            n += args.k
            steps_term += int(term.sum())
        if n:
            out[mode] = {
                "states": len(sel),
                "playouts": n,
                "us_per_playout_cpu": round(cpu / n * 1e6, 1),
                "us_per_playout_wall": round(wall / n * 1e6, 1),
                "playouts_per_cpu_s": round(n / cpu, 1),
                "terminal_share": round(steps_term / n, 3),
            }
    t0 = time.process_time()
    nf = 0
    for s, st in built[: args.n]:
        vd.features(st, s["mode"] == "belief_endturn")
        nf += 1
    out["features_us"] = round((time.process_time() - t0) / max(1, nf) * 1e6, 1)
    out["load1_end"] = os.getloadavg()[0]
    print(json.dumps(out, indent=1))
    if args.out:
        vd.write_json(args.out, out)
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ppo.deepbets.value_label")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--procs", type=int, default=1, help="worker processes (one CPU each)")
        p.add_argument(
            "--rollout-net",
            default=None,
            help="playout net file (STS2_ROLLOUT_NET); default the wheel's built-in net",
        )

    p = sub.add_parser("states", help="sample state specs")
    common(p)
    p.add_argument("--traj", required=True)
    p.add_argument("--hbank", required=True)
    p.add_argument("--manifest", action="append", required=True, help="bench MANIFEST (repeat)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=830)
    p.add_argument("--decision-frac", type=float, default=1.0)
    p.add_argument("--after-frac", type=float, default=0.5, help="share of train turn starts")
    p.add_argument(
        "--probe-after-frac",
        type=float,
        default=1.0,
        help="share of probe-test / probe-dev turn starts (1000 afterstate groups need all)",
    )
    p.add_argument("--n-desc", type=int, default=2)
    p.add_argument("--desc-min", type=int, default=3)
    p.add_argument("--desc-max", type=int, default=15)
    p.add_argument("--deep-frac", type=float, default=0.25)
    p.add_argument("--max-plans", type=int, default=16)
    p.add_argument("--plan-cap", type=int, default=2000)
    p.add_argument(
        "--cap-decisions", nargs="*", default=None, help="smoke caps per split, e.g. test=3 dev=3"
    )
    p.add_argument("--cap-turns", nargs="*", default=None, help="smoke caps on turn starts")
    p.add_argument(
        "--traj-frac", type=float, default=1.0, help="a seeded share of the trajectories"
    )
    p.add_argument(
        "--per-traj",
        type=int,
        default=0,
        help="at most this many decisions (and turns) per trajectory",
    )
    p.add_argument("--shard-size", type=int, default=20000)
    p.set_defaults(fn=cmd_states)

    p = sub.add_parser("label", help="K playouts + features per spec")
    common(p)
    p.add_argument("--specs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--chunk", type=int, default=100, help="trajectories per shard")
    p.add_argument("--splits", nargs="*", default=None)
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("refs", help="reference playouts for one split's probe groups")
    common(p)
    p.add_argument("--specs", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--split", choices=("test", "dev"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=830)
    p.add_argument("--lo", type=float, default=0.1)
    p.add_argument("--hi", type=float, default=0.9)
    p.add_argument("--contested-boss", type=int, default=1000)
    p.add_argument("--contested-elite", type=int, default=1000)
    p.add_argument("--decided", type=int, default=1000)
    p.add_argument("--after", type=int, default=1000)
    p.add_argument("--after-any", action="store_true", help="afterstate groups of lost parents too")
    p.add_argument("--n-ref", type=int, default=1024)
    p.add_argument("--n-r", type=int, default=768)
    p.add_argument("--n-fresh", type=int, default=16)
    p.add_argument("--chunk", type=int, default=50, help="trajectories per shard")
    p.set_defaults(fn=cmd_refs)

    p = sub.add_parser("bench", help="values_many throughput, one thread")
    p.add_argument("--specs", required=True)
    p.add_argument("--rollout-net", default=None)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--per-traj", type=int, default=2, help="specs per mode per trajectory")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_bench)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
