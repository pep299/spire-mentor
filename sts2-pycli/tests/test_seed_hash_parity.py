"""GH #792: one seed string hashes identically in Python, Rust and the C# oracle.

v0.111.0 replaced the game's 32-bit seed hash with XxHash64 into a ``ulong``
(``StringHelper.cs:140``; ``RunRngSet.cs:107``), so every seed's streams
changed. Three independent implementations must agree for each build:

* the pure-Python reference in :mod:`workbench.seed`,
* the engine's ``sts2_rs.seed_hash`` / ``run_seed_hash`` (what every harness
  path actually uses — ``parity_fuzz.rust_master_seed``, ``RunManager``),
* the build's own C# oracle (``rng_probe``: ``RunRngSet.Seed``, the per-name
  hash, and the first draws of ``new Rng(Seed, name)``).

v0.107.1 must keep producing the seed it always did
(``deterministic_hash_code(s) & 0xFFFFFFFF``).
"""

from __future__ import annotations

import os

import parity_fuzz
import pytest
import sts2_client
from workbench import seed as wseed

sts2_rs = pytest.importorskip("sts2_rs")

SEEDS = ["3ACCFREXKR", "PAR1TY_FUZZ", "ABCDEF123456", "RP0", "Y3X2YZSBQJ0Z"]
NAMES = ["up_front", "act_1_map", "NEOW", "SLIMES_WEAK", "rewards", "a" * 40]
DRAWS = 6


@pytest.fixture
def restore_version():
    before = sts2_client.GAME_VERSION
    env = os.environ.get(sts2_client.GAME_VERSION_ENV)
    yield
    sts2_client.set_game_version(before)
    if env is None:
        os.environ.pop(sts2_client.GAME_VERSION_ENV, None)
    else:
        os.environ[sts2_client.GAME_VERSION_ENV] = env


@pytest.mark.parametrize("ver", sts2_client.SUPPORTED_GAME_VERSIONS)
def test_python_reference_matches_the_engine(ver) -> None:
    for s in [*SEEDS, *NAMES, "", "ünïcödé", "old3ACCFREXKR", "x" * 100]:
        assert wseed.seed_hash(s, ver) == sts2_rs.seed_hash(s, game_version=ver), (ver, s)
        assert wseed.run_seed_hash(s, ver) == sts2_rs.run_seed_hash(s, game_version=ver), (ver, s)


def test_v0107_master_seed_is_unchanged() -> None:
    # The harness used to spell the Rust seed this way; the versioned helper
    # must keep every v0.107.1 seed bit-identical.
    for s in SEEDS:
        legacy = sts2_rs.deterministic_hash_code(s) & 0xFFFFFFFF
        assert parity_fuzz.rust_master_seed(s, "v0.107.1") == legacy
        assert wseed.run_seed_hash(s, "v0.107.1") == legacy


def test_v0111_master_seed_is_xxhash64() -> None:
    assert parity_fuzz.rust_master_seed("3ACCFREXKR", "v0.111.0") == 0xCCD08772EAB003D2
    assert wseed.xxh64(b"3ACCFREXKR") == 0xCCD08772EAB003D2
    # Save-migrated seeds keep the old hash of the remainder (RunRngSet.cs:107).
    assert parity_fuzz.rust_master_seed("old3ACCFREXKR", "v0.111.0") == (
        sts2_rs.deterministic_hash_code("3ACCFREXKR") & 0xFFFFFFFF
    )


@pytest.mark.parametrize("ver", sts2_client.SUPPORTED_GAME_VERSIONS)
def test_seed_hash_matches_the_csharp_oracle(ver, restore_version) -> None:
    if not sts2_client.oracle_available(ver) or not sts2_client.oracle_built(ver):
        pytest.skip(f"C# oracle for {ver} not available/built (`just build-cs {ver}`)")
    sts2_client.set_game_version(ver)
    g = sts2_client.Game()
    try:
        for s in SEEDS:
            r = g.send({"cmd": "rng_probe", "seed": s, "names": NAMES, "draws": DRAWS})
            assert r.get("type") == "rng_probe", r
            canon = wseed.canonicalize_seed(s)
            assert r["seed"] == canon
            run_seed = int(r["run_seed"])
            assert run_seed == wseed.run_seed_hash(canon, ver), (ver, s)
            assert run_seed == sts2_rs.run_seed_hash(canon, game_version=ver), (ver, s)
            assert run_seed == parity_fuzz.rust_master_seed(canon, ver), (ver, s)
            for name in NAMES:
                assert int(r["name_hashes"][name]) == wseed.seed_hash(name, ver), (ver, name)
            # The engine's streams, seeded from that master, draw what C# draws.
            draws = sts2_rs.debug_named_stream_draws(run_seed, NAMES, DRAWS, game_version=ver)
            for name in NAMES:
                assert draws[name] == r["streams"][name], (ver, s, name)
            assert draws["up_front"] == r["up_front"], (ver, s)
    finally:
        g.close()


def test_json_exact_seed_strings_only_past_2_53():
    """GH #878: the browser rounds a JSON number past 2^53; a v0.111.0 ``ulong``
    run seed goes out as a decimal string, every smaller one stays a number."""
    from workbench.seed import json_exact_seed

    assert json_exact_seed(44815703) == 44815703
    assert json_exact_seed((1 << 53) - 1) == (1 << 53) - 1
    assert json_exact_seed(1 << 53) == "9007199254740992"
    assert json_exact_seed((1 << 64) - 1) == "18446744073709551615"
