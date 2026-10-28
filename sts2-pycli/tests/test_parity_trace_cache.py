"""Unit tests for the C# reference-trace cache (issue #3).

Pure-Python — no game engine. The C# recording (`_record_cs_trace`) is
monkeypatched to a deterministic stub so these cover the cache plumbing
(keying, namespacing, hit/miss, atomic write, format invalidation, the replay
oracle, and the `run_trial` dispatch) without spawning a .NET subprocess.

The live-vs-cache *equivalence* check that actually drives both engines is the
`@pytest.mark.slow` test at the bottom.
"""

from __future__ import annotations

import json
import os

import parity_audit as pa
import pytest
from parity_fuzz_scenario import Scenario


def _sc(deck=("STRIKE_IRONCLAD",), encounter="SHRINKER_BEETLE_WEAK", **kw) -> Scenario:
    base = dict(
        character="Ironclad",
        deck=tuple(deck),
        hp=70,
        max_hp=80,
        relics=(),
        potions=(),
        encounter=encounter,
    )
    base.update(kw)
    return Scenario(**base)


# ---------------------------------------------------------------------------
# _trial_key
# ---------------------------------------------------------------------------


def test_trial_key_is_stable_and_identity_sensitive() -> None:
    sc = _sc()
    k = pa._trial_key(3, 6, sc)
    assert k == pa._trial_key(3, 6, sc)  # deterministic
    assert k != pa._trial_key(4, 6, sc)  # seed
    assert k != pa._trial_key(3, 5, sc)  # max_turns
    assert k != pa._trial_key(3, 6, _sc(deck=("DEFEND_IRONCLAD",)))  # scenario content


def test_trial_key_tracks_scenario_not_just_seed() -> None:
    # The whole point of keying on content: two trials that reuse the same seed
    # index (assemble_v2_trials shuffles) but differ in scenario must NOT collide.
    a = pa._trial_key(0, 6, _sc(encounter="SHRINKER_BEETLE_WEAK"))
    b = pa._trial_key(0, 6, _sc(encounter="SLITHERING_STRANGLER_NORMAL"))
    assert a != b


# ---------------------------------------------------------------------------
# cs_version_namespace
# ---------------------------------------------------------------------------


def _make_oracle_tree(root: str, dll_bytes: bytes, harness_src: bytes = b"class RunSim {}") -> None:
    libdir = os.path.join(root, "lib", "v0.107.1")
    os.makedirs(libdir)
    with open(os.path.join(libdir, "sts2.dll"), "wb") as f:
        f.write(dll_bytes)
    with open(os.path.join(libdir, "0Harmony.dll"), "wb") as f:
        f.write(b"harmony")
    src = os.path.join(root, "src", "Sts2Headless")
    os.makedirs(src)
    with open(os.path.join(src, "RunSimulator.cs"), "wb") as f:
        f.write(harness_src)
    with open(os.path.join(src, "Program.cs"), "wb") as f:
        f.write(b"static void Main() {}")
    # build output must be IGNORED — a fresh-MVID rebuild here must not move
    # the namespace (that is what makes the cache reusable across checkouts).
    binp = os.path.join(src, "bin", "v0.107.1", "Debug", "net9.0")
    os.makedirs(binp)
    with open(os.path.join(binp, "Sts2Headless.dll"), "wb") as f:
        f.write(b"NON-REPRODUCIBLE-BUILD-MVID-XYZ")


def test_namespace_changes_with_dll_bytes(tmp_path) -> None:
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _make_oracle_tree(a, b"GAME-V1")
    _make_oracle_tree(b, b"GAME-V2")  # one game DLL byte differs
    na, nb = pa.cs_version_namespace(a), pa.cs_version_namespace(b)
    assert na and nb and na != nb
    assert na == pa.cs_version_namespace(a)  # stable
    assert na.startswith("v0.107.1-") and len(na.split("-", 1)[1]) == 16


def test_namespace_changes_with_harness_source(tmp_path) -> None:
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _make_oracle_tree(a, b"GAME", harness_src=b"class RunSim { int v = 1; }")
    _make_oracle_tree(b, b"GAME", harness_src=b"class RunSim { int v = 2; }")
    assert pa.cs_version_namespace(a) != pa.cs_version_namespace(b)


def test_namespace_is_path_independent_for_cross_checkout_reuse(tmp_path) -> None:
    # Two checkouts at DIFFERENT paths but identical lib bytes + harness source
    # (incl. a differing, non-reproducible build output) MUST share a namespace
    # — otherwise worktrees / cloud sessions can't reuse a recorded bundle.
    a, b = str(tmp_path / "worktree-a"), str(tmp_path / "checkout-b")
    _make_oracle_tree(a, b"GAME")
    _make_oracle_tree(b, b"GAME")
    # diverge ONLY the (ignored) built DLL, as a real rebuild would
    with open(
        os.path.join(
            b, "src", "Sts2Headless", "bin", "v0.107.1", "Debug", "net9.0", "Sts2Headless.dll"
        ),
        "wb",
    ) as f:
        f.write(b"a-totally-different-build-mvid")
    assert pa.cs_version_namespace(a) == pa.cs_version_namespace(b)


def test_namespace_empty_when_no_oracle_bytes(tmp_path) -> None:
    # No lib/ and no src/ → cannot namespace → empty (cache disabled).
    assert pa.cs_version_namespace(str(tmp_path)) == ""


# ---------------------------------------------------------------------------
# _ReplayCsOracle
# ---------------------------------------------------------------------------


def test_replay_oracle_serves_recorded_sequence() -> None:
    trace = {
        "v": pa._TRACE_FORMAT,
        "init": {"tag": "init"},
        "steps": [
            {
                "actions": [{"action": "end_turn"}],
                "action": {"action": "end_turn"},
                "snap": {"t": 0},
            },
            {
                "actions": [{"action": "end_turn"}],
                "action": {"action": "end_turn"},
                "snap": {"t": 1},
            },
            {"actions": []},  # terminal: legal() returns [] and the loop breaks
        ],
    }
    o = pa._ReplayCsOracle(trace)
    assert o.init() == {"tag": "init"}
    # step 0
    assert o.legal() == [{"action": "end_turn"}]
    o.apply({"action": "end_turn"})  # no-op
    assert o.snapshot() == {"t": 0}
    # step 1
    assert o.legal() == [{"action": "end_turn"}]
    o.apply({"action": "end_turn"})
    assert o.snapshot() == {"t": 1}
    # step 2 — terminal empty actions, no snapshot taken
    assert o.legal() == []
    o.close()


def test_replay_oracle_reraises_init_error() -> None:
    o = pa._ReplayCsOracle(
        {"v": pa._TRACE_FORMAT, "init_error": "RuntimeError('boom')", "steps": []}
    )
    with pytest.raises(RuntimeError):
        o.init()


@pytest.mark.parametrize("stage", ["legal", "apply", "post"])
def test_replay_oracle_reraises_step_errors(stage) -> None:
    step = {"actions": [{"action": "end_turn"}], "action": {"action": "end_turn"}}
    if stage != "legal":
        step["snap"] = {"t": 0}
    step["error"] = [stage, "RuntimeError('mid-trial')"]
    o = pa._ReplayCsOracle({"v": pa._TRACE_FORMAT, "init": {}, "steps": [step]})
    o.init()
    if stage == "legal":
        with pytest.raises(RuntimeError):
            o.legal()
    elif stage == "apply":
        assert o.legal() == [{"action": "end_turn"}]
        with pytest.raises(RuntimeError):
            o.apply({"action": "end_turn"})
    else:  # post
        assert o.legal() == [{"action": "end_turn"}]
        o.apply({"action": "end_turn"})
        with pytest.raises(RuntimeError):
            o.snapshot()


# ---------------------------------------------------------------------------
# _TraceCache.get_or_record
# ---------------------------------------------------------------------------


def _stub_record(monkeypatch) -> dict:
    """Replace _record_cs_trace with a counter-bumping stub; return the call log."""
    log = {"n": 0}

    def fake(seed, max_turns, sc):
        log["n"] += 1
        return {"v": pa._TRACE_FORMAT, "init": {"seed": seed}, "steps": []}

    monkeypatch.setattr(pa, "_record_cs_trace", fake)
    return log


def test_cache_miss_then_hit(tmp_path, monkeypatch) -> None:
    log = _stub_record(monkeypatch)
    cache = pa._TraceCache(str(tmp_path), "vNS")
    sc = _sc()

    trace1, status1 = cache.get_or_record(7, 6, sc)
    assert status1 == "miss"
    assert log["n"] == 1
    assert trace1["init"] == {"seed": 7}

    trace2, status2 = cache.get_or_record(7, 6, sc)
    assert status2 == "hit"
    assert log["n"] == 1  # not re-recorded
    assert trace2 == trace1


def test_cache_writes_under_namespace_dir_atomically(tmp_path, monkeypatch) -> None:
    _stub_record(monkeypatch)
    cache = pa._TraceCache(str(tmp_path), "vNS")
    cache.get_or_record(1, 6, _sc())
    ns_dir = tmp_path / "vNS"
    assert ns_dir.is_dir()
    files = list(ns_dir.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".json"
    assert not any(p.name.endswith(".tmp") for p in files)  # no leftover temp
    # the persisted file is valid JSON with the format stamp
    with open(files[0], encoding="utf-8") as f:
        assert json.load(f)["v"] == pa._TRACE_FORMAT


def test_cache_rerecords_on_format_mismatch(tmp_path, monkeypatch) -> None:
    log = _stub_record(monkeypatch)
    cache = pa._TraceCache(str(tmp_path), "vNS")
    sc = _sc()
    path = os.path.join(str(tmp_path), "vNS", pa._trial_key(2, 6, sc) + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"v": pa._TRACE_FORMAT + 999, "init": {}, "steps": []}, f)
    _, status = cache.get_or_record(2, 6, sc)
    assert status == "miss"  # stale-format trace ignored + re-recorded
    assert log["n"] == 1


def test_cache_rerecords_on_corrupt_file(tmp_path, monkeypatch) -> None:
    log = _stub_record(monkeypatch)
    cache = pa._TraceCache(str(tmp_path), "vNS")
    sc = _sc()
    path = os.path.join(str(tmp_path), "vNS", pa._trial_key(9, 6, sc) + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{ this is not json")
    _, status = cache.get_or_record(9, 6, sc)
    assert status == "miss"
    assert log["n"] == 1


# ---------------------------------------------------------------------------
# run_trial dispatch + _init_worker_cache
# ---------------------------------------------------------------------------


def test_init_worker_cache_sets_and_clears_global(tmp_path) -> None:
    try:
        pa._init_worker_cache(str(tmp_path), "vNS")
        assert isinstance(pa._CACHE, pa._TraceCache)
        assert pa._CACHE.namespace == "vNS"
        pa._init_worker_cache(None, None)
        assert pa._CACHE is None
        pa._init_worker_cache(str(tmp_path), None)  # namespace missing → disabled
        assert pa._CACHE is None
    finally:
        pa._init_worker_cache(None, None)


def test_run_trial_uses_live_oracle_without_cache(monkeypatch) -> None:
    seen = {}

    def fake_core(seed, max_turns, sc, oracle):
        seen["oracle"] = oracle
        return {"category": "clean"}

    monkeypatch.setattr(pa, "_run_trial_core", fake_core)
    monkeypatch.setattr(pa, "_LiveCsOracle", lambda sc: ("LIVE", sc))
    pa._init_worker_cache(None, None)
    try:
        out = pa.run_trial(seed=0, max_turns=6, scenario=_sc())
        assert seen["oracle"][0] == "LIVE"
        assert "_cache" not in out
    finally:
        pa._init_worker_cache(None, None)


def test_run_trial_replays_with_cache(tmp_path, monkeypatch) -> None:
    _stub_record(monkeypatch)
    captured = {}

    def fake_core(seed, max_turns, sc, oracle):
        captured["oracle_type"] = type(oracle).__name__
        return {"category": "clean"}

    monkeypatch.setattr(pa, "_run_trial_core", fake_core)
    pa._init_worker_cache(str(tmp_path), "vNS")
    try:
        sc = _sc()
        out1 = pa.run_trial(seed=5, max_turns=6, scenario=sc)
        assert captured["oracle_type"] == "_ReplayCsOracle"
        assert out1["_cache"] == "miss"  # first time records
        out2 = pa.run_trial(seed=5, max_turns=6, scenario=sc)
        assert out2["_cache"] == "hit"  # second time served from disk
    finally:
        pa._init_worker_cache(None, None)


# ---------------------------------------------------------------------------
# End-to-end: a cached replay must equal a live run (needs the real engines).
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cache_replay_matches_live(tmp_path) -> None:
    """For several random trials, the cached replay result must be byte-for-byte
    the live result (modulo the transient `_cache` status). This is the
    correctness guarantee: caching the oracle never changes the verdict."""
    pa._init_worker_cache(None, None)
    try:
        for seed in range(6):
            live = pa.run_trial(seed=seed, max_turns=4)
            assert "_cache" not in live

            pa._init_worker_cache(str(tmp_path), "vTEST")
            recorded = pa.run_trial(seed=seed, max_turns=4)
            assert recorded.pop("_cache") == "miss"
            replayed = pa.run_trial(seed=seed, max_turns=4)
            assert replayed.pop("_cache") == "hit"
            pa._init_worker_cache(None, None)

            assert recorded == live, f"recorded != live @ seed {seed}"
            assert replayed == live, f"replayed != live @ seed {seed}"
    finally:
        pa._init_worker_cache(None, None)
