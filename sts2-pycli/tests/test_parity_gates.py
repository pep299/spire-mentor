"""item#8: oracle-freshness gate + invariant-gate flag.

``assert_oracle_fresh`` closes the stale-binary hole: ``cs_version_namespace``
hashes ``.cs`` SOURCE, so a built dll older than its source silently serves an
old oracle under a fresh namespace (this bit a whole session — see the function
docstring). ``_CHECK_INVARIANTS`` gates the engine invariant suite OFF by
default so the synthetic coverage sweep isn't drowned by migration-debt /
instanced-power artifacts.
"""

from __future__ import annotations

import os
import time

import parity_audit as pa
import pytest
import sts2_client


def _mk_oracle_tree(tmp_path, src_newer: bool):
    """Build <root>/src with a built dll and one .cs, controlling which is newer."""
    src = tmp_path / "src"
    # The tree is built for the selected build: ``assert_oracle_fresh`` checks
    # ``bin/<GAME_VERSION>/``, which follows ``--game-version``.
    bindir = src / "Sts2Headless" / "bin" / sts2_client.GAME_VERSION / "Debug" / "net9.0"
    bindir.mkdir(parents=True)
    dll = bindir / "Sts2Headless.dll"
    cs = src / "Sts2Headless" / "RunSimulator.cs"
    dll.write_text("binary")
    cs.write_text("// source")
    old, new = time.time() - 100, time.time()
    if src_newer:
        os.utime(dll, (old, old))
        os.utime(cs, (new, new))
    else:
        os.utime(cs, (old, old))
        os.utime(dll, (new, new))
    return str(tmp_path)


def test_oracle_fresh_passes_when_dll_newer(tmp_path) -> None:
    root = _mk_oracle_tree(tmp_path, src_newer=False)
    pa.assert_oracle_fresh(root)  # must not raise


def test_oracle_fresh_aborts_when_source_newer(tmp_path) -> None:
    root = _mk_oracle_tree(tmp_path, src_newer=True)
    with pytest.raises(SystemExit, match="STALE C# ORACLE"):
        pa.assert_oracle_fresh(root)


def test_oracle_fresh_aborts_when_unbuilt(tmp_path) -> None:
    src = tmp_path / "src" / "Sts2Headless"
    src.mkdir(parents=True)
    (src / "RunSimulator.cs").write_text("// source")
    with pytest.raises(SystemExit, match="NOT BUILT"):
        pa.assert_oracle_fresh(str(tmp_path))


def test_oracle_fresh_bypass_env(tmp_path, monkeypatch) -> None:
    root = _mk_oracle_tree(tmp_path, src_newer=True)
    monkeypatch.setenv("STS2_SKIP_ORACLE_CHECK", "1")
    pa.assert_oracle_fresh(root)  # bypassed → no raise


def test_assemble_v2_deep_horizon_max_turns() -> None:
    # item#3: max_turns flows into every trial tuple.
    trials = pa.assemble_v2_trials("smoke", ["fuzz"], "Ironclad", max_turns=30)
    assert trials and all(mt == 30 for _, _, mt in trials)


def test_assemble_v2_multipath_replicates_scenarios() -> None:
    # item#4: paths=K → K trials per scenario, distinct consecutive action seeds.
    one = pa.assemble_v2_trials("smoke", ["fuzz"], "Ironclad", paths=1)
    three = pa.assemble_v2_trials("smoke", ["fuzz"], "Ironclad", paths=3)
    assert len(three) == 3 * len(one)
    seeds = [s for s, _, _ in three]
    assert seeds == list(range(len(three)))  # every path has a unique action seed
    # the same scenario appears in a consecutive run of `paths` (group of 3)
    scen0 = [three[i][1] for i in range(3)]
    assert scen0[0] == scen0[1] == scen0[2]


def test_assemble_v2_rejects_bad_paths_and_turns() -> None:
    import pytest as _pt

    with _pt.raises(SystemExit):
        pa.assemble_v2_trials("smoke", ["fuzz"], "Ironclad", paths=0)
    with _pt.raises(SystemExit):
        pa.assemble_v2_trials("smoke", ["fuzz"], "Ironclad", max_turns=0)


def test_powers_in_collects_player_and_monster_kinds() -> None:
    state = {
        "player": {"powers": [{"kind": "STRENGTH", "amount": 2}, {"kind": "DEXTERITY"}]},
        "monsters": [
            {"powers": [{"kind": "RITUAL", "amount": 1}]},
            {"powers": [{"kind": "STRENGTH"}]},  # dedup across player+monster
            {"powers": []},
        ],
    }
    assert pa._powers_in(state) == {"STRENGTH", "DEXTERITY", "RITUAL"}


def test_powers_in_empty() -> None:
    assert pa._powers_in({"player": {}, "monsters": []}) == set()


def test_invariant_gate_off_by_default(monkeypatch) -> None:
    # Default (flag off): the helper is a no-op even if inner would report.
    monkeypatch.setattr(pa, "_CHECK_INVARIANTS", False)

    class _Stub:
        inner = type("I", (), {"check_invariants": lambda self, ctx: "boom"})()

    assert pa._rs_invariant_violation(_Stub(), "ctx") is None
    monkeypatch.setattr(pa, "_CHECK_INVARIANTS", True)
    assert pa._rs_invariant_violation(_Stub(), "ctx") == "boom"
