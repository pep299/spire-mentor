"""C# save/load identity check: does a reloaded checkpoint continue exactly
like the run that wrote it?

This is the prerequisite for starting ``act23_state_parity`` from act-2/3
checkpoints instead of replaying act 1 every trial. Checkpointing is only
sound if ``write_continue_save`` + ``load_save`` is lossless: a reload must
continue bit-for-bit like the process that kept going, or every divergence
found from a checkpoint could be a save/load artifact rather than a parity bug.

Per (character, seed):

1. **Process A** starts a run and random-walks it with the same policy as
   ``act23_state_parity`` (``run_parity.enumerate_choices``, HP pinned to
   ``INVINCIBLE_HP`` outside combat and events) until it first reaches the
   ``map_select`` of act ``--act``. It writes a continue save there and keeps
   walking for ``--steps`` more steps, logging every pin, action and view.
2. **Process B** is a fresh oracle process. It loads the save and replays A's
   pins and actions.
3. The check diffs A's and B's view after every step. At every later
   ``map_select`` both processes also write a save, and the two saves must
   match apart from wall-clock fields. The save holds every run RNG stream's
   counter, so this catches a stream that moved differently before the
   difference shows up in a view.

Why not the per-read RNG trace (``STS2_TRACE_RNG``): on v0.111.0 an ``Rng``
restored from a save has no seed label (``seed=?``), so reads can't be
grouped by stream, and the trace includes cosmetic streams that differ
between any two processes.

Exit code 0 when every run is identical, 1 when any diverged, 2 on a harness
error.

Usage::

    STS2_GAME_VERSION=v0.111.0 scripts/wheel-venv.sh run -- \\
        python python/save_load_identity.py --all-characters -n 5 --act 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from act23_state_parity import ALL_CHARACTERS, INVINCIBLE_HP, _act_of, _is_err
from run_parity import enumerate_choices
from sts2_client import Game

# Save fields that differ between any two processes: wall clock, and the
# reload counter that a reload bumps by design.
_VOLATILE_SAVE_KEYS = frozenset({"run_time", "save_time", "start_time", "win_time", "num_reloads"})

# The game's own save round-trip flips this: ``SerializableMapPoint.CanBeModified``
# defaults to ``true`` but is written ``WhenWritingDefault``, so ``false`` is
# never written and loads back as ``true``. Inert after generation: the only
# reader is ``MapPathPruning``, which runs while a map is generated
# (``StandardActMap.cs:104``, ``SpoilsActMap.cs:100``).
_LOSSY_MAP_POINT_KEYS = frozenset({"can_modify"})


# Per-player discovery logs (``Player.DiscoveredPotions`` …). The only reader
# is the run summary screen (``NRunSummary.cs:157-192``). An item is logged
# when the in-memory profile hasn't seen it, and a fresh process reloading a
# save has a fresh profile, so it logs again what the writer had seen.
_PROFILE_LOG_KEYS = frozenset(
    {
        "discovered_cards",
        "discovered_relics",
        "discovered_potions",
        "discovered_enemies",
        "discovered_epochs",
    }
)

# View fields rolled from ``Rng.Chaotic``, the game's unseeded cosmetic RNG:
# they differ between any two processes. ``EntrantNumber``: ``Trial.cs:215``.
_CHAOTIC_VIEW_KEYS = frozenset({"EntrantNumber"})


def _strip_view(obj: Any) -> Any:
    """Drop view fields that come from the unseeded cosmetic RNG."""
    if isinstance(obj, dict):
        return {k: _strip_view(v) for k, v in obj.items() if k not in _CHAOTIC_VIEW_KEYS}
    if isinstance(obj, list):
        return [_strip_view(v) for v in obj]
    return obj


def _normalize_save(obj: Any) -> Any:
    """Drop the save keys a reload can't reproduce (see the constants above)."""
    if isinstance(obj, dict):
        drop = _PROFILE_LOG_KEYS
        if "coord" in obj and "type" in obj:
            drop = drop | _LOSSY_MAP_POINT_KEYS
        return {k: _normalize_save(v) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [_normalize_save(v) for v in obj]
    return obj


@dataclass
class Step:
    pin: bool
    verb: str
    kw: dict[str, Any]
    view: dict[str, Any]


@dataclass
class Result:
    character: str
    seed: str
    status: str  # identical | diverged | no_checkpoint | error
    detail: str = ""
    walk_steps: int = 0
    checked_steps: int = 0
    saves_compared: int = 0
    secs_a: float = 0.0
    secs_b: float = 0.0
    diffs: list[str] = field(default_factory=list)
    soft_paths: list[str] = field(default_factory=list)


def _phase(view: dict[str, Any]) -> str:
    return str(view.get("decision") or view.get("type") or "?")


def _pin_here(view: dict[str, Any]) -> bool:
    """Same pin rule as ``act23_state_parity``: never in combat or an event."""
    return _phase(view) not in ("combat_play", "event_choice")


def _diff(a: Any, b: Any, path: str = "", out: list[str] | None = None, cap: int = 12) -> list[str]:
    """Structural diff of two JSON values, as ``path: A=… B=…`` lines."""
    out = [] if out is None else out
    if len(out) >= cap:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: {'missing in A' if k not in a else 'missing in B'}")
            else:
                _diff(a[k], b[k], f"{path}.{k}", out, cap)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _diff(x, y, f"{path}[{i}]", out, cap)
    elif a != b:
        out.append(f"{path}: A={json.dumps(a)[:120]} B={json.dumps(b)[:120]}")
    return out


def _path(diff_line: str) -> str:
    """The JSON path a ``_diff`` line is about."""
    return diff_line.split(": ", 1)[0]


def _save_json(g: Game, path: Path) -> dict[str, Any]:
    saved = g.write_continue_save(str(path))
    if not saved.get("success"):
        raise RuntimeError(f"save failed: {saved}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return _normalize_save({k: v for k, v in data.items() if k not in _VOLATILE_SAVE_KEYS})


def _pick(
    g: Game, view: dict[str, Any], rng: random.Random
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Act on a random legal choice, re-picking on a rejection."""
    pool = enumerate_choices(g, view)
    while pool:
        verb, kw = pool.pop(rng.randrange(len(pool)))
        nxt = g.act(verb, **kw)
        if not _is_err(nxt):
            return verb, kw, nxt
    return None


def check_one(
    character: str, seed: str, act: int, steps: int, ascension: int, max_walk: int
) -> Result:
    """Run the A/B identity check for one (character, seed)."""
    res = Result(character=character, seed=seed, status="error")
    rng = random.Random(f"saveload:{seed}")
    tmp = Path(tempfile.mkdtemp(prefix="saveload-"))
    checkpoint = tmp / "checkpoint.save"
    a = b = None
    try:
        # ── Process A: walk to the checkpoint, save, keep walking. ──
        t0 = time.monotonic()
        a = Game()
        view = a.start(character=character, seed=seed, ascension=ascension)
        for _ in range(max_walk):
            if _phase(view) == "map_select" and (_act_of(view) or 0) >= act:
                break
            if _phase(view) == "game_over":
                res.status, res.detail = "no_checkpoint", f"game over before act {act}"
                return res
            if _pin_here(view):
                a.set_player(hp=INVINCIBLE_HP, max_hp=INVINCIBLE_HP)
            picked = _pick(a, view, rng)
            if picked is None:
                res.status, res.detail = "no_checkpoint", f"stuck at {_phase(view)}"
                return res
            view = picked[2]
            res.walk_steps += 1
        else:
            res.status, res.detail = "no_checkpoint", f"act {act} not reached in {max_walk} steps"
            return res

        _save_json(a, checkpoint)
        start_view = view

        trail: list[Step] = []
        saves_a: dict[int, dict[str, Any]] = {}
        for i in range(steps):
            if _phase(view) == "game_over":
                break
            if i > 0 and _phase(view) == "map_select":
                saves_a[i] = _save_json(a, tmp / f"a-{i}.save")
            pin = _pin_here(view)
            if pin:
                a.set_player(hp=INVINCIBLE_HP, max_hp=INVINCIBLE_HP)
            picked = _pick(a, view, rng)
            if picked is None:
                break
            verb, kw, view = picked
            trail.append(Step(pin, verb, kw, view))
        res.secs_a = time.monotonic() - t0

        # ── Process B: load the save, replay A's continuation. ──
        t0 = time.monotonic()
        b = Game()
        view = b.load_save(path=str(checkpoint))
        if _is_err(view):
            res.detail = f"load failed: {view}"
            return res
        # A view field that already differs right after the load, while the
        # saves agree, is transient state the save doesn't hold (Permafrost's
        # per-combat ``ActivatedThisCombat``, shown as the relic ``counter``,
        # resets on the next combat entry). Track those paths as soft and fail
        # only on a new path or a save mismatch.
        soft = {
            _path(line) for line in _diff(_strip_view(start_view), _strip_view(view), cap=10_000)
        }
        res.soft_paths = sorted(soft)
        for i, st in enumerate(trail):
            if i in saves_a:
                d = _diff(saves_a[i], _save_json(b, tmp / f"b-{i}.save"))
                res.saves_compared += 1
                if d:
                    res.status, res.detail, res.diffs = "diverged", f"save before step {i}", d
                    return res
            if st.pin:
                b.set_player(hp=INVINCIBLE_HP, max_hp=INVINCIBLE_HP)
            got = b.act(st.verb, **st.kw)
            res.checked_steps = i + 1
            if _is_err(got):
                res.status = "diverged"
                res.detail = f"step {i} {st.verb}({st.kw}) rejected after reload: {got}"
                return res
            full = _diff(_strip_view(st.view), _strip_view(got), cap=10_000)
            d = [line for line in full if _path(line) not in soft][:12]
            # A soft path stops being soft once it matches again, so a later
            # real divergence on the same field still fails.
            soft &= {_path(line) for line in full}
            if d:
                res.status = "diverged"
                res.detail = f"step {i} after {st.verb}({st.kw}), phase {_phase(st.view)}"
                res.diffs = d
                return res
        res.secs_b = time.monotonic() - t0
        res.status = "identical"
        return res
    except Exception as exc:  # harness failure, reported not raised
        res.detail = f"{type(exc).__name__}: {exc}"
        return res
    finally:
        for g in (a, b):
            if g is not None:
                g.close()


def main() -> int:
    """CLI entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description="C# save/load identity check")
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--all-characters", action="store_true")
    ap.add_argument("-n", type=int, default=3, help="seeds per character")
    ap.add_argument("--start", type=int, default=1000, help="first integer seed")
    ap.add_argument("--act", type=int, default=2, help="checkpoint at the first map of this act")
    ap.add_argument("--steps", type=int, default=400, help="steps to compare after the save")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--max-walk", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", type=Path, help="write per-run results here")
    args = ap.parse_args()

    chars = ALL_CHARACTERS if args.all_characters else [args.character]
    jobs = [(ch, str(s)) for ch in chars for s in range(args.start, args.start + args.n)]
    # Each run owns two oracle processes and a private temp dir, so runs share
    # no state and can go in parallel.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(check_one, ch, s, args.act, args.steps, args.ascension, args.max_walk)
            for ch, s in jobs
        ]
        results: list[Result] = []
        for fut in futs:  # in submission order, so the report is stable
            r = fut.result()
            results.append(r)
            print(
                f"{r.character}:{r.seed} {r.status:<13} walk={r.walk_steps} checked={r.checked_steps} "
                f"saves={r.saves_compared} A={r.secs_a:.0f}s B={r.secs_b:.0f}s {r.detail}",
                flush=True,
            )
            if r.soft_paths:
                print(f"    soft after load: {', '.join(r.soft_paths)}", flush=True)
            for line in r.diffs:
                print(f"    {line}", flush=True)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print("summary:", counts)
    if args.json:
        args.json.write_text(json.dumps([r.__dict__ for r in results], indent=1), encoding="utf-8")
    if counts.get("error"):
        return 2
    return 1 if counts.get("diverged") else 0


if __name__ == "__main__":
    raise SystemExit(main())
