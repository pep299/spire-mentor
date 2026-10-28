"""Decision records -> a token cache and per-decision targets (GH #830 step 0, DESIGN §d).

`cache` reads `gen`'s decision records and writes, per split, one `.npz` of
flat ragged arrays plus the decision ids:

* tokens of `S_ctx` (`cat` int16, `cont` float16, `sec` int8). The other
  transformer arms are DERIVED from these at load (`tokens.derive`, pinned
  equal to `tokens(..., arm)` by a test), so one cache serves every arm;
* per action: class id, both searches' visits and Q, rl2's 24 features, its
  action key (as a universe row), the built-in net's logit, the primary and
  secondary targets (DESIGN §d) and the class regret under each search;
* per decision: run / fight ids, room, the recommended classes of both
  searches, the E flag (PREREG §6), the fight's final search value and rl2's
  32 state features.

Only `record_inputs(rec)` -> `(obs, actions, rl2feat)` reaches the token
builder, so a record's search statistics, outcome and hashes cannot (test 10).

Targets, per action class `c` (two actions are one class iff they reach one
observation; the search's visits sit on one copy of a duplicate):

    primary   t(c) = 0.8·1[c = class(rec_A)] + 0.2·visits_A(c) / Σ visits_A
    secondary t(c) = 0.5·1[c = class(rec_A)] + 0.5·softmax(Q_A(c) / 0.05)

each spread evenly over the class's member actions. `Q_X(c)` is the
visit-weighted mean root Q of the members; regret under X is
`max_c Q_X(c) - Q_X(c)`.

`RareMap` is the one rare-id rule (PREREG §4): an id seen in fewer than 3
TRAIN runs maps to UNK in every split.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from multiprocessing import get_context
from typing import Any

import numpy as np

from . import tokens as T
from .gen import dec_paths

CACHE_VERSION = 1
SPLITS = ("train", "dev", "test", "pilot")
ROOM_ID = {"monster": 0, "elite": 1, "boss": 2}
TARGET_MIX = 0.2
SECONDARY_TAU = 0.05
E_ACTIONS = frozenset({"play", "potion", "drink_potion", "end_turn"})
RARE_MIN_RUNS = 3


def record_inputs(
    rec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """The only parts of a decision record the token builder may see."""
    acts = rec.get("act_obs") or rec["actions"]
    rl2 = {k: rec["rl2"][k] for k in ("state", "x", "keys")}
    return rec["obs"], acts, rl2


def class_stats(cls: np.ndarray, vis: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(visits per class, visit-weighted mean Q per class (nan = unvisited))`."""
    n_cls = int(cls.max()) + 1
    v = np.bincount(cls, weights=vis, minlength=n_cls)
    qv = np.bincount(cls, weights=np.where(vis > 0, np.nan_to_num(q) * vis, 0.0), minlength=n_cls)
    with np.errstate(invalid="ignore", divide="ignore"):
        qc = np.where(v > 0, qv / np.maximum(v, 1e-12), np.nan)
    return v, qc


def _search_arrays(blk: dict[str, Any] | None, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    if blk is None:
        return np.zeros(n), np.full(n, np.nan), -1
    vis = np.asarray(blk["visits"], np.float64)
    q = np.asarray([np.nan if x is None else x for x in blk["q"]], np.float64)
    return vis, q, int(blk["rec"])


def spread(cls: np.ndarray, tc: np.ndarray) -> np.ndarray:
    """Class targets spread evenly over each class's members."""
    size = np.bincount(cls)
    return tc[cls] / size[cls]


def decision_arrays(rec: dict[str, Any]) -> dict[str, Any]:
    """Targets, regrets and flags of one record (numpy, per action / per decision)."""
    acts = rec["actions"]
    n = len(acts)
    cls = np.asarray(rec["cls"], np.int64)
    visA, qA, recA = _search_arrays(rec["A"], n)
    visB, qB, recB = _search_arrays(rec.get("B"), n)
    vA, qcA = class_stats(cls, visA, qA)
    vB, qcB = class_stats(cls, visB, qB)
    n_cls = len(vA)
    rc = int(cls[recA]) if recA >= 0 else -1
    one = np.zeros(n_cls)
    if rc >= 0:
        one[rc] = 1.0
    share = vA / max(vA.sum(), 1e-12)
    t1 = spread(cls, (1 - TARGET_MIX) * one + TARGET_MIX * share)
    qz = np.where(np.isnan(qcA), -np.inf, qcA / SECONDARY_TAU)
    if np.isfinite(qz).any():
        sm = np.exp(qz - qz[np.isfinite(qz)].max())
        sm = sm / sm.sum()
    else:
        sm = np.zeros(n_cls)
    t2 = spread(cls, 0.5 * one + 0.5 * sm)
    regA = np.nanmax(qcA) - qcA if np.isfinite(qcA).any() else np.full(n_cls, np.nan)
    has_b = rec.get("B") is not None and vB.sum() > 0
    regB = (np.nanmax(qcB) - qcB) if has_b and np.isfinite(qcB).any() else np.full(n_cls, np.nan)
    choice_open = rec["obs"].get("choice") is not None
    card_phase = (not choice_open) and all(a.get("action") in E_ACTIONS for a in acts)
    covered = has_b and n_cls >= 2 and bool((vA > 0).all()) and bool((vB > 0).all())
    in_e = bool(vA.sum() > 0 and card_phase and covered)
    return {
        "cls": cls,
        "visA": visA,
        "qA": qA,
        "visB": visB,
        "qB": qB,
        "t1": t1,
        "t2": t2,
        "regA": regA[cls],
        "regB": regB[cls],
        "recA": recA,
        "recB": recB,
        "rec_clsA": rc,
        "rec_clsB": int(cls[recB]) if recB >= 0 else -1,
        "in_E": in_e,
        "card_phase": card_phase,
        "has_B": has_b,
        "n_cls": n_cls,
    }


def _process_part(job: tuple[str, list[int]]) -> dict[str, Any]:
    """Tokenize the kept lines of one dec part (a Pool worker)."""
    import gzip

    path, keep = job
    keep_s = set(keep)
    out: dict[str, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    skipped = collections.Counter()
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        li = -1
        for line in f:
            if not line.strip():
                continue
            li += 1
            if li not in keep_s:
                continue
            rec = json.loads(line)
            if rec.get("obs") is None or rec.get("cls") is None:
                skipped["no_obs"] += 1
                continue
            obs, acts, rl2 = record_inputs(rec)
            tk = T.tokens(obs, acts, rl2, "S_ctx")
            da = decision_arrays(rec)
            o = out[rec["split"]]
            o["cat"].append(tk["cat"])
            o["cont"].append(T.f16(tk["cont"]))
            o["sec"].append(tk["sec"])
            o["n_tok"].append(len(tk["sec"]))
            o["n_act"].append(len(acts))
            o["n_oov"].append(tk["n_oov"])
            o["n_ids"].append(tk["n_ids"])
            for k in ("cls", "visA", "qA", "visB", "qB", "t1", "t2", "regA", "regB"):
                o[k].append(np.asarray(da[k], np.float32 if k != "cls" else np.int16))
            o["rl2_x"].append(np.asarray(rl2["x"], np.float32))
            o["rl2_logit"].append(np.asarray(rec["rl2"]["logits"], np.float32))
            o["rl2_key"].append(tk["cat"][-len(acts) :, T.S["rl2key"]].astype(np.int32))
            o["rl2_state"].append(np.asarray(rl2["state"], np.float32))
            for k in ("recA", "recB", "rec_clsA", "rec_clsB", "n_cls"):
                o[k].append(da[k])
            for k in ("in_E", "card_phase", "has_B"):
                o[k].append(da[k])
            z = rec.get("z") or {}
            o["z_value"].append(np.nan if z.get("value") is None else float(z["value"]))
            o["won"].append(bool(z.get("won")))
            o["room"].append(ROOM_ID[rec["room"]])
            o["fight_idx"].append(int(rec["fight_idx"]))
            o["meta"].append((rec["id"], rec["file"], rec["key"], int(rec["rep"]), int(rec["t"])))
    return {"out": {s: dict(v) for s, v in out.items()}, "skipped": dict(skipped)}


def build_cache(gen_dirs: list[str], out_dir: str, procs: int, log: Any = print) -> dict[str, Any]:
    """Write `out_dir/{split}.npz` + `{split}.meta.json` and `cache.json`."""
    import gzip

    paths = [p for g in gen_dirs for p in dec_paths(g)]
    last: dict[str, tuple[int, int]] = {}
    for pi, p in enumerate(paths):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as f:
            li = -1
            for line in f:
                if not line.strip():
                    continue
                li += 1
                i = line.find('"id":"')
                last[line[i + 6 : line.index('"', i + 6)]] = (pi, li)
    keep: dict[int, list[int]] = collections.defaultdict(list)
    for pi, li in last.values():
        keep[pi].append(li)
    jobs = [(paths[pi], sorted(v)) for pi, v in sorted(keep.items())]
    t0 = time.time()
    merged: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    skipped: collections.Counter = collections.Counter()
    with get_context("spawn").Pool(max(1, min(procs, len(jobs)))) as pool:
        for k, res in enumerate(pool.imap(_process_part, jobs)):
            for s, d in res["out"].items():
                for name, v in d.items():
                    merged[s][name].extend(v)
            skipped.update(res["skipped"])
            log(f"cache: part {k + 1}/{len(jobs)} done, {time.time() - t0:.0f}s")
    os.makedirs(out_dir, exist_ok=True)
    summary: dict[str, Any] = {"version": CACHE_VERSION, "gen_dirs": gen_dirs, "splits": {}}
    for s, d in merged.items():
        arrays = pack(d)
        np.savez(os.path.join(out_dir, f"{s}.npz"), **arrays)
        meta = {"ids": [m[0] for m in d["meta"]], "files": [m[1] for m in d["meta"]],
                "keys": [m[2] for m in d["meta"]], "reps": [m[3] for m in d["meta"]],
                "ts": [m[4] for m in d["meta"]]}  # fmt: skip
        with open(os.path.join(out_dir, f"{s}.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        summary["splits"][s] = {
            "decisions": len(d["n_tok"]),
            "tokens": int(np.sum(d["n_tok"])),
            "E": int(np.sum(d["in_E"])),
            "oov_share": float(np.sum(d["n_oov"]) / max(1, np.sum(d["n_ids"]))),
        }
    summary["skipped"] = dict(skipped)
    summary["vocab"] = T.vocab().digest()
    summary["n_cont"] = T.N_CONT
    with open(os.path.join(out_dir, "cache.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


def pack(d: dict[str, list]) -> dict[str, np.ndarray]:
    n_tok = np.asarray(d["n_tok"], np.int64)
    n_act = np.asarray(d["n_act"], np.int64)
    out = {
        "cat": np.concatenate(d["cat"]).astype(np.int16),
        "cont": np.concatenate(d["cont"]).astype(np.float16),
        "sec": np.concatenate(d["sec"]).astype(np.int8),
        "tok_start": np.concatenate([[0], np.cumsum(n_tok)[:-1]]).astype(np.int64),
        "n_tok": n_tok,
        "act_start": np.concatenate([[0], np.cumsum(n_act)[:-1]]).astype(np.int64),
        "n_act": n_act,
        "rl2_state": np.stack(d["rl2_state"]).astype(np.float32),
    }
    for k in (
        "cls",
        "visA",
        "qA",
        "visB",
        "qB",
        "t1",
        "t2",
        "regA",
        "regB",
        "rl2_logit",
        "rl2_key",
    ):
        out[k] = np.concatenate(d[k])
    out["rl2_x"] = np.concatenate(d["rl2_x"]).astype(np.float32)
    for k in (
        "recA",
        "recB",
        "rec_clsA",
        "rec_clsB",
        "n_cls",
        "room",
        "fight_idx",
        "n_oov",
        "n_ids",
    ):
        out[k] = np.asarray(d[k], np.int64)
    for k in ("in_E", "card_phase", "has_B", "won"):
        out[k] = np.asarray(d[k], bool)
    out["z_value"] = np.asarray(d["z_value"], np.float32)
    return out


class Split:
    """One split of the cache, numpy, plus its ids / run / fight indices."""

    def __init__(self, cache_dir: str, split: str) -> None:
        z = np.load(os.path.join(cache_dir, f"{split}.npz"))
        self.a: dict[str, np.ndarray] = {k: z[k] for k in z.files}
        with open(os.path.join(cache_dir, f"{split}.meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.split = split
        self.n = len(self.a["n_tok"])
        files = self.meta["files"]
        self.runs = sorted(set(files))
        rid = {f: i for i, f in enumerate(self.runs)}
        self.run = np.asarray([rid[f] for f in files], np.int64)
        # `fight` is one fight-play (key and rep); `bank_fight` is the bank
        # fight, every rep of it together (same deck, relics, encounter and
        # HP; only the seed differs), the cluster of the fight-clustered CI.
        fk = [f"{k}|{r}" for k, r in zip(self.meta["keys"], self.meta["reps"], strict=True)]
        self.fights = sorted(set(fk))
        fid = {f: i for i, f in enumerate(self.fights)}
        self.fight = np.asarray([fid[f] for f in fk], np.int64)
        self.bank_fights = sorted(set(self.meta["keys"]))
        bid = {k: i for i, k in enumerate(self.bank_fights)}
        self.bank_fight = np.asarray([bid[k] for k in self.meta["keys"]], np.int64)
        self.act_dec = np.repeat(np.arange(self.n), self.a["n_act"])

    def subset_runs(self, keep_runs: set[str]) -> np.ndarray:
        return np.asarray([f in keep_runs for f in self.meta["files"]], bool)


class RareMap:
    """Per content slot: universe row -> row, ids seen in < `RARE_MIN_RUNS`
    TRAIN runs sent to UNK (PAD stays PAD)."""

    def __init__(self, maps: dict[str, np.ndarray], counts: dict[str, dict[int, int]]) -> None:
        self.maps = maps
        self.counts = counts

    @classmethod
    def fit(
        cls, train: Split, min_runs: int = RARE_MIN_RUNS, keep: np.ndarray | None = None
    ) -> RareMap:
        """Fit on the TRAIN decisions a student trains on.

        Args:
            keep: Per TRAIN decision, whether the student trains on it (a
                learning-curve fraction, the boss/elite-only retrain); None =
                all of TRAIN. An id seen only in decisions outside `keep` is
                rare for that student, so it maps to the trained UNK row
                rather than to a row that never gets a gradient.
        """
        v = T.vocab()
        tok_run = np.repeat(train.run, train.a["n_tok"])
        tok_keep = None if keep is None else np.repeat(np.asarray(keep, bool), train.a["n_tok"])
        maps, counts = {}, {}
        for slot in T.CONTENT_SLOTS:
            s = T.S[slot]
            tab = v.tables[s]
            ids = train.a["cat"][:, s].astype(np.int64)
            sel = ids < tab.unk
            if tok_keep is not None:
                sel &= tok_keep
            pairs = np.unique(ids[sel] * (len(train.runs) + 1) + tok_run[sel])
            ids_u = pairs // (len(train.runs) + 1)
            c = np.bincount(ids_u, minlength=tab.size)
            m = np.arange(tab.size, dtype=np.int64)
            rare = (c < min_runs) & (m < tab.unk)
            m[rare] = tab.unk
            maps[slot] = m
            counts[slot] = {int(i): int(c[i]) for i in np.flatnonzero(c)}
        return cls(maps, counts)

    def apply(self, cat: np.ndarray) -> np.ndarray:
        out = cat.astype(np.int64)
        for slot, m in self.maps.items():
            s = T.S[slot]
            out[:, s] = m[out[:, s]]
        return out

    def report(self, sp: Split) -> dict[str, float]:
        """UNK share (rare + OOV) of content ids in `sp`, per slot."""
        v = T.vocab()
        rep = {}
        for slot in T.CONTENT_SLOTS:
            s = T.S[slot]
            ids = sp.a["cat"][:, s].astype(np.int64)
            sel = ids != v.tables[s].pad
            mapped = self.maps[slot][ids[sel]]
            rep[slot] = float((mapped == v.tables[s].unk).mean()) if sel.any() else 0.0
        return rep

    def to_json(self) -> dict[str, Any]:
        return {k: m.tolist() for k, m in self.maps.items()}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RareMap:
        return cls({k: np.asarray(v, np.int64) for k, v in d.items()}, {})


def vocab_report(cache_dir: str) -> dict[str, Any]:
    """Universe size, ids seen in TRAIN, rare ids and the UNK/OOV shares on DEV
    and TEST, per content table (DESIGN §b)."""
    train = Split(cache_dir, "train")
    rm = RareMap.fit(train)
    v = T.vocab()
    out: dict[str, Any] = {}
    for slot in T.CONTENT_SLOTS:
        tab = v.tables[T.S[slot]]
        seen = rm.counts[slot]
        out[slot] = {
            "universe": tab.size - 2,
            "seen_train": len(seen),
            "rare": sum(1 for c in seen.values() if c < RARE_MIN_RUNS),
        }
    for s in ("dev", "test"):
        if os.path.exists(os.path.join(cache_dir, f"{s}.npz")):
            sp = Split(cache_dir, s)
            for slot, share in rm.report(sp).items():
                out[slot][f"unk_share_{s}"] = share
            out[f"oov_share_{s}"] = float(sp.a["n_oov"].sum() / max(1, sp.a["n_ids"].sum()))
    return out


def cmd_cache(args: argparse.Namespace) -> int:
    s = build_cache(args.gen, args.out, args.procs)
    print(json.dumps(s, indent=1))
    if os.path.exists(os.path.join(args.out, "train.npz")):
        rep = vocab_report(args.out)
        with open(os.path.join(args.out, "vocab_report.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps(rep, indent=1))
    return 0


def add_parser(sub: Any) -> None:
    c = sub.add_parser("cache", help="decision records -> token cache + targets")
    c.add_argument("--gen", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--procs", type=int, default=16)
    c.set_defaults(func=cmd_cache)
