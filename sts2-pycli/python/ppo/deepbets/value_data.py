"""Shared pieces of the G-D value-net pipeline (GH #830 deep bets, plan Step 2 G-D).

The pipeline is `value_label` (states, labels, references), `value_train`
(ReturnValueNet on the GPUs) and `value_neq` (the pre-registered n_eq
metrics). This module holds what more than one of them needs:

* the SOURCE: pilot trajectories (`rollout_starts traj` records) of human-bank
  fights, minus every run any combat-bench MANIFEST holds out, with an
  assertion that the overlap left is zero;
* the SPLIT: `sha1(run file) % 10`, 0 = probe-test, 1 = probe-dev, 2-9 = train;
* a STATE SPEC, a JSON-able recipe that rebuilds one position
  deterministically: replay the record's run actions to decision `t` (each
  state checked against its recorded `ordered_hash`), optionally take the
  honest `belief_view(view)`, then apply `path` (combat actions). The
  rebuilt position's `observation_hash` is checked against `ohash`;
* the playout-net IDENTITY every label, reference and net file pins;
* the honest FEATURES the value net reads, computed on
  `belief_view(FEATURE_VIEW_SEED)`, so they depend on what the player can see
  and nothing else (no draw order, no RNG cursors).

Every label and reference is a `values_many` playout in an HONEST mode
(`belief` or `belief_endturn`); `raw` on a true state is never used for either.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from collections.abc import Iterable
from typing import Any

# Modes a label or reference may use: honest only. `raw` on a true state reads
# the hidden draw order and RNG, so it never labels anything here.
HONEST_MODES = ("belief", "belief_endturn")
# Spec kinds. `child` / `endturn` form a decision's sibling group (the EndTurn
# member is the parent itself scored in `belief_endturn` mode, i.e. the honest
# Q of ending the turn now); `deep_*` the same one level down a descendant;
# `after` a turn's afterstates (one deterministic plan each, CRN group).
KINDS = ("decision", "child", "endturn", "desc", "deep_child", "deep_endturn", "after")
ROOMS = ("elite", "boss", "monster")
SPLIT_NAMES = ("test", "dev", "train")
# `belief_view` seed for the features: any fixed seed gives the same
# observable part; the seed only picks the canonical RNG the net never reads.
FEATURE_VIEW_SEED = 0
# `belief_view` seed of the view the chance detector (`rng_fingerprint`) runs on.
CHANCE_VIEW_SEED = 1
MONSTER_SLOTS = 5
ZONES = ("hand", "draw_pile", "discard_pile", "exhaust_pile")
STEP_DIM = 96  # `rollout_features(version=2)["state"]`, the v2 StepCtx vector
WIDE_DIM = 216  # `feature_vector_wide()`, TexelWide's inputs
MON_FEATS = 3  # hp fraction, block / 20, alive
SEED_MASK = (1 << 63) - 1
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_SUBDIR = os.path.abspath(os.path.join(SRC_DIR, "..", "..", "..", ".."))
BUILTIN_NET = os.path.join(REPO_SUBDIR, "sts2-rs", "data", "rollout_nets", "ironclad.json")


# ── split and seeds ──────────────────────────────────────────────────────


def split_of(run_file: str) -> str:
    """`probe-test` / `probe-dev` / `train` by `sha1(run file) % 10` (0 / 1 / 2-9)."""
    h = int(hashlib.sha1(run_file.encode()).hexdigest(), 16) % 10
    return SPLIT_NAMES[h] if h < 2 else "train"


def seed_base(salt: str, key: str) -> int:
    """A 60-bit seed base for (`salt`, `key`); consecutive offsets are the seeds."""
    return int(hashlib.sha1(f"{salt}|{key}".encode()).hexdigest()[:15], 16)


def seeds(salt: str, key: str, n: int, offset: int = 0) -> list[int]:
    """`n` playout seeds for `key` (a spec id, or a group id for CRN seeds)."""
    base = seed_base(salt, key)
    return [(base + offset + j) & SEED_MASK for j in range(n)]


def frac_key(key: str) -> float:
    """A fixed pseudo-uniform [0, 1) per fight key: nested learning-curve subsets."""
    return int(hashlib.sha1(f"lc|{key}".encode()).hexdigest()[:12], 16) / float(1 << 48)


# ── source ───────────────────────────────────────────────────────────────


def holdout_union(manifests: Iterable[str]) -> tuple[set[str], dict[str, int]]:
    """Every run file any of the bench `manifests` holds out, and the per-file counts."""
    out: set[str] = set()
    counts = {}
    for path in manifests:
        with open(path, encoding="utf-8") as f:
            runs = set(json.load(f)["held_out_runs"])
        counts[os.path.basename(path)] = len(runs)
        out |= runs
    return out, counts


def load_source(
    traj: str, hbank: str, manifests: list[str], rooms: tuple[str, ...] = ("boss", "elite")
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """The pilot trajectories the states come from, bench runs removed.

    Args:
        traj: A `rollout_starts traj` file (schema v2) of human-bank fights.
        hbank: The human fight bank the trajectories were drawn from; every
            kept record's run file must be in it.
        manifests: Combat-bench MANIFESTs; the union of their `held_out_runs`
            is removed.
        rooms: Rooms kept.

    Returns:
        `(header, records, meta)`. Records keep their line index as `_idx`.

    Raises:
        AssertionError: A kept record is a bench run, or not a human-bank run.
    """
    from ppo import rollout_starts

    header, recs = rollout_starts.read_traj(traj)
    held, per_manifest = holdout_union(manifests)
    with open(hbank, encoding="utf-8") as f:
        bank_files = {json.loads(line)["file"] for line in f if line.strip()}
    kept = []
    dropped = {"error": 0, "room": 0, "bench": 0, "empty": 0}
    for i, r in enumerate(recs):
        r["_idx"] = i
        if r.get("end") == "error":
            dropped["error"] += 1
        elif r["room"] not in rooms:
            dropped["room"] += 1
        elif r["file"] in held:
            dropped["bench"] += 1
        elif not r.get("actions"):
            dropped["empty"] += 1
        else:
            kept.append(r)
    overlap = {r["file"] for r in kept} & held
    assert not overlap, f"{len(overlap)} bench runs left in the source"
    not_bank = {r["file"] for r in kept} - bank_files
    assert not not_bank, f"{len(not_bank)} source runs are not human-bank runs"
    meta = {
        "traj": os.path.abspath(traj),
        "traj_sha256": sha256_file(traj),
        "traj_engine": header.get("engine"),
        "hbank": os.path.abspath(hbank),
        "manifests": per_manifest,
        "holdout_union": len(held),
        "records": len(recs),
        "kept": len(kept),
        "kept_runs": len({r["file"] for r in kept}),
        "dropped": dropped,
        "bench_overlap_after": 0,
    }
    return header, kept, meta


def read_traj_records(traj: str) -> list[dict[str, Any]]:
    """Every record of `traj`, header skipped, by line index."""
    from ppo import rollout_starts

    return rollout_starts.read_traj(traj)[1]


def walk_record(rec: dict[str, Any], ts: Iterable[int]) -> tuple[dict[int, Any], int]:
    """The TRUE combat states at decisions `ts` of `rec`, replayed and hash-checked.

    The replay stops at the first decision whose state does not hash as
    recorded (an engine change since the trajectory was recorded): every
    state returned is verified, later ones are missing.

    Returns:
        `({t: CombatState}, verified)`, `verified` = the decisions that hashed.
    """
    from ppo import rollout_starts

    want = set(ts)
    out: dict[int, Any] = {}
    if not want:
        return out, 0
    last = max(want)
    rm = rollout_starts.build_fight(rec["fight"], int(rec["seed"]))
    verified = 0
    for t, action in enumerate(rec["actions"]):
        cs = rm.combat_state()
        if cs is None or int(cs.ordered_hash()) != int(rec["hashes"][t]):
            break
        verified = t + 1
        if t in want:
            out[t] = cs.clone()
        if t >= last:
            break
        try:
            rm.take_action(action)
        except BaseException:  # a PyO3 panic is a BaseException
            break
    return out, verified


def apply_path(cs: Any, view: int | None, path: list[dict[str, Any]]) -> Any:
    """The position `path` reaches from `cs` (its `belief_view(view)` when given)."""
    s = cs.belief_view(view) if view is not None else cs.clone()
    for a in path:
        s.apply(a)
    return s


def spec_state(cs: Any, spec: dict[str, Any]) -> Any:
    """Rebuild `spec`'s position from its decision state `cs`, checked on `ohash`.

    Raises:
        RuntimeError: The rebuilt position is not the one the spec recorded.
    """
    s = apply_path(cs, spec.get("view"), spec.get("path") or [])
    want = spec.get("ohash")
    if want is not None and int(s.observation_hash()) != int(want):
        raise RuntimeError(f"spec {spec['id']}: rebuilt observation_hash differs")
    return s


# ── identities ───────────────────────────────────────────────────────────


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def playout_net_identity() -> dict[str, Any]:
    """The playout net `values_many` plays: `STS2_ROLLOUT_NET` if set, else built in.

    The built-in net is `sts2-rs/data/rollout_nets/ironclad.json` compiled into
    the wheel; its file in this checkout is hashed (the wheel's build commit is
    recorded beside it by `engine_identity`).
    """
    env = os.environ.get("STS2_ROLLOUT_NET")
    path = os.path.abspath(env) if env else BUILTIN_NET
    return {
        "source": "env" if env else "builtin",
        "path": path,
        "sha256": sha256_file(path) if os.path.isfile(path) else None,
        "temp_env": os.environ.get("STS2_ROLLOUT_TEMP"),
    }


def engine_identity() -> dict[str, Any]:
    import sts2_rs

    info = sts2_rs.build_info()
    return {
        "src_hash": sts2_rs.src_hash(),
        "build_git": info.get("build_git"),
        "pgo": info.get("pgo"),
    }


def engine_env() -> dict[str, str]:
    """The engine seams set in this process (values_many reads `STS2_RUNAI_*`)."""
    from ppo.combat_bench import agents

    return {k: v for k, v in sorted(os.environ.items()) if agents.is_engine_env(k)}


def check_same_net(a: dict[str, Any], b: dict[str, Any], what: str) -> None:
    """Refuse to mix artifacts labelled under different playout nets.

    Raises:
        SystemExit: The two identities name different net weights.
    """
    if a.get("sha256") != b.get("sha256"):
        raise SystemExit(
            f"{what}: playout net {a.get('sha256')} != {b.get('sha256')}; relabel under one net"
        )


# ── features ─────────────────────────────────────────────────────────────


def card_token(c: dict[str, Any]) -> str:
    return str(c["id"]) + ("+" if c.get("upgraded") else "")


def features(cs: Any, afterstate: bool) -> dict[str, Any]:
    """The value net's raw inputs at `cs`, from its honest view only.

    Returns:
        `step` (96, the v2 StepCtx vector), `wide` (216, TexelWide's inputs),
        `zones` (card tokens of hand / draw / discard / exhaust; the draw
        pile as a multiset, order-free), `potions`, `mons` (up to 5 of
        `(species, hp fraction, block / 20, alive)`), `after` (scored as a
        turn afterstate) and `texel` (`search_playout_value` of the view: the
        TexelWide depth-cut leaf the search would score it with).
    """
    import sts2_rs

    bv = cs.belief_view(FEATURE_VIEW_SEED)
    rf = sts2_rs.rollout_features(bv, version=2)
    d = sts2_rs.state_to_dict(bv)
    zones = [sorted(card_token(c) for c in d["deck"].get(z, [])) for z in ZONES]
    potions = sorted(str(p) for p in d["player"].get("potions", []) if p)
    mons = []
    for m in d.get("encounter", [])[:MONSTER_SLOTS]:
        alive = not m.get("is_dead") and not m.get("is_escaped")
        mhp = max(1, int(m.get("max_hp", 1)))
        mons.append(
            (str(m["kind"]), float(m.get("hp", 0)) / mhp, float(m.get("block", 0)) / 20.0, alive)
        )
    return {
        "step": list(rf["state"]),
        "wide": list(bv.feature_vector_wide()),
        "zones": zones,
        "potions": potions,
        "mons": mons,
        "after": bool(afterstate),
        "texel": float(sts2_rs.search_playout_value(bv)),
    }


# ── small IO helpers ─────────────────────────────────────────────────────


class Progress:
    """Timestamped lines to stdout and an append-only `progress.log`."""

    def __init__(self, path: str | None, tag: str) -> None:
        self.path = path
        self.tag = tag
        self.t0 = time.time()

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{self.tag}] {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    pathlib.Path(tmp).replace(path)


def read_jsonl(paths: Iterable[str]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def shard_paths(d: str, prefix: str, ext: str) -> list[str]:
    return sorted(
        os.path.join(d, f) for f in os.listdir(d) if f.startswith(prefix) and f.endswith(ext)
    )


def worker_env(rollout_net: str | None) -> dict[str, str]:
    """The engine env label / reference workers run under: nothing inherited,
    plus `STS2_ROLLOUT_NET` when an experiment net replaces the built-in one."""
    return {"STS2_ROLLOUT_NET": os.path.abspath(rollout_net)} if rollout_net else {}


def init_worker(env: dict[str, str]) -> None:
    """Pool initializer: one rayon thread, the spec's engine env exactly."""
    os.environ["RAYON_NUM_THREADS"] = "1"
    from ppo.combat_bench import agents

    agents.apply_env(env)
