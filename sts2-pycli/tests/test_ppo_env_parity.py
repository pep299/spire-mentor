"""G1 parity gate: PpoRunEnv expert mode must match run_arena --policy heuristic.

The env's expert mode wraps the same HeuristicRunAgent
(MctsCombatPolicy<HeuristicPolicy>) that `run_arena --policy heuristic`
drives, but routes non-combat multi-option decisions through the env's
yield/observe path (including event materialization in `observe()`). Any
divergence from run_arena per-seed outcomes means the env's observation
path perturbed the run — a real bug; do not weaken this test to make it
pass.

Plan: sts2-rs docs/plans/2026-07-03-ppo-fullrun-plan.md M-P1, gate G1.
"""

import functools
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARENA = os.path.join(ROOT, "sts2-rs", "target", "release", "run_arena")
PY = os.path.join(ROOT, "sts2-pycli", "python")
N = int(os.environ.get("PARITY_SEEDS", "20"))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("STS2_RUNAI_VALUE", raising=False)
    monkeypatch.syspath_prepend(PY)


def _env_expert_outcome(seed: int, act1: str = "overgrowth") -> dict:
    from ppo.env import PpoRunEnv, Terminal

    env = PpoRunEnv(seed=seed, expert=True, act1=act1)
    terminal = None
    for out in env.expert_run():
        if isinstance(out, Terminal):
            terminal = out
    assert terminal is not None
    # Ownership audit (G1b, per-seed slice): every logged resolution class
    # must be one of the three expected ones, and every yielded decision
    # must have a known surface.
    for _, phase, surface, klass in env.resolution_log:
        assert klass in ("combat-agent", "single-option-auto", "expert", "potion-swap-rule"), (
            f"unexpected resolution class {klass} at phase {phase}"
        )
        if klass == "expert":
            # "potion-swap" joined this list with GH #379, which turned
            # `interactive_potion_swap` on for both arms — a full-belt potion
            # reward is now a yielded decision rather than a silent drop.
            assert surface in (
                "path",
                "card-pick",
                "event",
                "shop",
                "rest",
                "treasure",
                "potion-swap",
            ), f"yielded decision with unclassified surface: phase={phase}"
    return {
        "seed": seed,
        "won": bool(env.rm.player_won),
        "final_hp": int(env.rm.current_hp),
        "act_index": int(env.rm.act_index),
    }


# Cross-worker arena cache: the first xdist worker to take the flock runs
# the (deterministic) sweep; the rest block on the lock then read the cache.
# The cache path is keyed on a content hash of everything that determines the
# sweep output — the run_arena binary, the seed count and the act1 token
# (GH #545: overgrowth/underdocks/roll walk different content, so each token
# needs its own cache row) — so a rebuilt engine / different N / different
# act1 invalidates it automatically instead of a stale /tmp file producing
# false parity failures.
def _arena_cache_path(act1: str) -> str:
    h = hashlib.sha256()
    with open(ARENA, "rb") as fh:
        h.update(fh.read())
    h.update(str(N).encode())
    h.update(act1.encode())
    return f"/tmp/ppo-parity-arena.{h.hexdigest()[:16]}.jsonl"


#: What to run when the binary is missing or out of date. Named in every
#: failure message below so the fix never has to be looked up.
REBUILD_CMD = "cargo build --release --bin run_arena   # from sts2-rs/"

RS_ROOT = os.path.join(ROOT, "sts2-rs")


@functools.cache
def _arena_stale_reason() -> str | None:
    """Is `run_arena` older than the Rust sources it was built from?

    `run_arena` is a BUILD ARTIFACT, and nothing in the suite rebuilds it — so
    it silently rots while the engine moves. Its CLI is part of that rot: on
    2026-08-26 a two-day-old binary predated `--act1`, so every invocation
    below exited 2 and all 60 parametrisations blew up as a bare
    `CalledProcessError` deep inside `subprocess.run`, which reads like an
    engine regression rather than "your binary is old". Diagnosing that cost
    more than the real bug sitting next to it.

    Compares mtimes rather than probing `--help` for specific flags: flag
    drift is only one way to rot, and "the sources moved and you did not
    rebuild" covers behaviour changes that keep the same CLI too. Returns the
    reason as a string, or `None` when the binary is current (or absent — the
    caller skips on that, which is the long-standing contract for an
    unbuilt optional artifact).
    """
    try:
        built = pathlib.Path(ARENA).stat().st_mtime
    except OSError:
        return None
    newest, newest_path = 0.0, ""
    for base, dirs, files in os.walk(RS_ROOT):
        # `target/` is the artifact itself; dotdirs are VCS/editor scratch.
        dirs[:] = [d for d in dirs if d != "target" and not d.startswith(".")]
        for name in files:
            if not (name.endswith(".rs") or name in ("Cargo.toml", "Cargo.lock")):
                continue
            path = os.path.join(base, name)
            try:
                mtime = pathlib.Path(path).stat().st_mtime
            except OSError:
                continue
            if mtime > newest:
                newest, newest_path = mtime, path
    if newest <= built:
        return None
    return (
        f"{os.path.relpath(newest_path, ROOT)} is newer than the run_arena "
        f"binary ({newest - built:.0f}s), so the sweep it answers with is not "
        f"this tree's. Rebuild it:\n    {REBUILD_CMD}"
    )


def _require_current_arena() -> None:
    """Skip when `run_arena` is absent; FAIL when it is merely out of date.

    The two cases are deliberately different. Absent means the optional
    artifact was never built and the gate opts out — the long-standing
    behaviour. Stale means a binary IS there and is answering for a tree that
    no longer exists, which is a wrong answer wearing a right answer's
    clothes. Skipping on that would drop 60 cross-engine parity assertions
    without saying so, and this repo has no CI behind the local gates to
    notice (see the root CLAUDE.md) — so it fails, loudly and with the fix.
    """
    if not os.path.exists(ARENA):
        pytest.skip(f"run_arena release binary not built:\n    {REBUILD_CMD}")
    reason = _arena_stale_reason()
    if reason is not None:
        pytest.fail(f"stale run_arena binary — {reason}", pytrace=False)


def _arena_rows(act1: str) -> dict:
    import fcntl

    cache = _arena_cache_path(act1)
    with open(f"{cache}.lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(cache):
            tmp = f"{cache}.{os.getpid()}"
            argv = [
                ARENA,
                "sweep",
                "--seeds",
                str(N),
                "--seed-start",
                "0",
                "--policy",
                "heuristic",
                "--combat",
                "eval-cheap",
                "--jobs",
                "0",
                "--act1",
                act1,
                "--out",
                tmp,
            ]
            # CAPTURED, not inherited: `check=True` on an inheriting call
            # raises a `CalledProcessError` whose repr is just the argv and an
            # exit code, while the one line that explains it ("unexpected
            # argument '--act1' found") has already gone to the terminal —
            # and under xdist, into a worker's discarded stream. Surface it in
            # the failure itself, with the rebuild that fixes the likeliest
            # cause; `_require_current_arena` catches the stale case first,
            # but not a binary whose mtime lies (a fresh `git checkout` of an
            # OLD tree restores old contents with a new mtime).
            proc = subprocess.run(argv, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise AssertionError(
                    f"run_arena sweep failed (exit {proc.returncode}). If this names "
                    f"an unknown argument the binary predates the flag — rebuild it:\n"
                    f"    {REBUILD_CMD}\n"
                    f"argv: {' '.join(argv)}\n"
                    f"stderr:\n{proc.stderr.strip()}\n"
                    f"stdout:\n{proc.stdout.strip()}"
                )
            pathlib.Path(tmp).replace(cache)
    with open(cache, encoding="utf-8") as fh:
        return {r["seed"]: r for r in (json.loads(line) for line in fh)}


@pytest.mark.slow
@pytest.mark.timeout(0)  # full runs — disable the global cap
@pytest.mark.parametrize("act1", ["overgrowth", "underdocks", "roll"])
@pytest.mark.parametrize("seed", range(N))
def test_ppo_env_expert_matches_run_arena(seed, act1):
    """Per-seed so xdist parallelizes the 20 env runs (was one 27-min test).

    Parameterised over the GH #545 act1 token so G1 (env vs `arena::run_cell`
    parity) covers Underdocks and the seed-driven roll, not just the historical
    Overgrowth-only default — an arena arm and a training env that disagree
    about the act is the silent-divergence class this gate exists to catch.
    """
    _require_current_arena()
    ar = _arena_rows(act1)[seed]
    a = _env_expert_outcome(seed, act1)
    got = (a["won"], a["final_hp"], a["act_index"])
    want = (bool(ar["won"]), int(ar["final_hp"]), int(ar["act_index"]))
    assert got == want, f"PpoRunEnv expert != run_arena on seed {seed} act1={act1}: {got} vs {want}"


def test_classify_surface_unit():
    """Pure-python classifier unit checks (no engine)."""
    sys.path.insert(0, PY)
    from ppo.env import classify_surface

    assert classify_surface("map_choice", [{"action": "move"}] * 2) == "path"
    assert classify_surface("card_reward", [{"action": "pick_card_reward"}] * 3) == "card-pick"
    assert classify_surface("boss_relic", [{"action": "pick_boss_relic"}] * 3) == "card-pick"
    assert classify_surface("shop", [{"action": "shop_buy_card"}] * 4) == "shop"
    assert classify_surface("event", [{"action": "event_choice"}] * 2) == "event"
    assert classify_surface("rest_site", [{"action": "rest_option"}] * 2) == "rest"
    assert classify_surface("treasure", [{"action": "collect_treasure"}] * 2) == "treasure"
    # Smith target preempts the rest surface (ablation.rs contract).
    assert classify_surface("rest_site", [{"action": "smith_card"}] * 3) == "card-pick"
    # Phase-agnostic deck mutations are card picks.
    assert classify_surface("event", [{"action": "remove_deck_card"}] * 5) == "card-pick"
    # Mixed lists fall through to the phase.
    assert (
        classify_surface("event", [{"action": "remove_deck_card"}, {"action": "event_choice"}])
        == "event"
    )
    assert classify_surface("combat", [{"action": "end_turn"}]) is None


# ── potion-stack seams (plan 2026-08-20-ppo-potion-bootstrap P2) ──────────

_OC_TABLE = {
    "version": 1,
    "character": "ironclad",
    "units": "hp_equiv",
    "source": "manual",
    "default_oc": 8.0,
    "entries": [
        {"potion": "POTION.FIRE_POTION", "act": 1, "oc": 4.0},
        {"potion": "POTION.POTION_SHAPED_ROCK", "act": 1, "oc": 0.0},
    ],
}


def _set_potion_seams(monkeypatch, tmp_path):
    table = tmp_path / "oc.json"
    table.write_text(json.dumps(_OC_TABLE))
    monkeypatch.setenv("STS2_RUNAI_POTION_OC", str(table))
    monkeypatch.setenv("STS2_RUNAI_POTION_WEIGHT", "0")
    monkeypatch.setenv("STS2_RUNAI_POTION_SWAP", "1")


def test_potion_swap_flag_has_one_source_of_truth(monkeypatch, tmp_path):
    """The env's RunManager takes `interactive_potion_swap` from the SAME
    resolver `arena.rs::new_arena_run_manager` reads — unset means off,
    seams-on means on, with no env.py-local reimplementation."""
    import sts2_rs
    from ppo.env import PpoRunEnv

    monkeypatch.delenv("STS2_RUNAI_POTION_SWAP", raising=False)
    monkeypatch.delenv("STS2_RUNAI_POTION_OC", raising=False)
    assert sts2_rs.resolve_potion_swap_flag() is False
    env = PpoRunEnv(seed=0)
    assert env.rm.interactive_potion_swap is False

    _set_potion_seams(monkeypatch, tmp_path)
    assert sts2_rs.resolve_potion_swap_flag() is True
    env = PpoRunEnv(seed=0)
    assert env.rm.interactive_potion_swap is True


def test_potion_swap_resolved_rule_side_never_yielded(monkeypatch, tmp_path):
    """A pending full-belt swap is resolved by the Rust OC rule inside the
    env loop — logged as `potion-swap-rule`, never yielded to the net and
    never counted as a decision (no encoder surface, GH #379)."""
    from ppo.env import PpoRunEnv, Terminal

    _set_potion_seams(monkeypatch, tmp_path)
    env = PpoRunEnv(seed=0)
    # Rock (OC 0) + Block (default 8) held; Fire (OC 4) offered — the
    # rule discards the worthless Rock (same fixture as the Rust test
    # `potion_swap_decision_is_resolved_by_the_oc_rule`).
    env.rm.debug_set_potion_state(["POTION_SHAPED_ROCK", "BLOCK_POTION"], pending=["FIRE_POTION"])
    out = env.reset()
    assert not isinstance(out, Terminal)
    assert env.rm.pending_potion_swap is None, "swap left pending"
    assert "FirePotion" in env.rm.potions(), "offer not claimed"
    assert "PotionShapedRock" not in env.rm.potions(), "argmin slot kept"
    classes = [k for *_, k in env.resolution_log]
    assert "potion-swap-rule" in classes
    # The swap never surfaced as a net decision: every yielded action set
    # so far is swap-free (the first Decision is the Neow event).
    assert all(a.get("action") != "potion_swap" for a in out.actions)
