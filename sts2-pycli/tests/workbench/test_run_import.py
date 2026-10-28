"""Tests for the run lane of the live-state import.

The messages these build are synthetic in one specific way: their ``map`` and
``rng`` blocks come from a REAL engine run at the same seed. That is
deliberate — it is the only way to assert the graft landed on the right node
and at the right RNG cursor without a live game in the loop, and it pins the
property the whole feature rests on: a run rebuilt at the paste's seed already
owns the map the player was standing on.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("sts2_rs", reason="the run import needs the sts2_rs extension")

from workbench.backends.rust_game import RustGame
from workbench.state_import import parse_pasted_state

SEED = "TESTSEED42"
DECK = [
    {"id": "STRIKE_IRONCLAD", "upgrade_level": 0},
    {"id": "STRIKE_IRONCLAD", "upgrade_level": 1},
    {"id": "DEFEND_IRONCLAD", "upgrade_level": 0},
    {"id": "BASH", "upgrade_level": 0},
]


@pytest.fixture
def live_run():
    """A real RunManager at SEED, to borrow a genuine map + RNG cursor from."""
    import sts2_rs

    return sts2_rs.RunManager(string_seed=SEED, character="Ironclad", act1="roll")


def _map_block(rm) -> dict:
    """The mod's map-graph shape, built from the engine's own act-1 map."""
    full = rm.full_map()
    nodes = [
        {
            "col": node["col"],
            "row": node["row"],
            "type": node["type"],
            "children": node.get("children") or [],
        }
        for row in full["rows"]
        for node in row
    ]
    first = next(n for n in nodes if n["row"] == 1)
    return {
        "nodes": nodes,
        "current": {"col": first["col"], "row": first["row"]},
        "visited": [{"col": first["col"], "row": first["row"]}],
    }


def _message(rm, **overrides) -> dict:
    msg = {
        "type": "advice_request",
        "advice_for": "map_select",
        "seed": SEED,
        "act": 1,
        "floor": 2,
        "map": _map_block(rm),
        "rng": {"run_seed": rm.run_seed, "counters": dict(rm.rng_counters())},
        "run_state": {
            "character": "IRONCLAD",
            "ascension": 0,
            "hp": 55,
            "max_hp": 80,
            "gold": 217,
            "potion_slot_count": 3,
            "master_deck": DECK,
            "relics": [{"id": "BURNING_BLOOD"}, {"id": "ANCHOR"}],
            "potions": [{"slot": 0, "id": "FIRE_POTION"}, {"slot": 1, "empty": True}],
        },
        "nodes": [],
    }
    msg.update(overrides)
    return msg


def _imported(rm, **overrides):
    return parse_pasted_state(json.dumps(_message(rm, **overrides)))


def _key(engine_id: str) -> str:
    """Compare inventory ids without pinning the engine's spelling.

    The engine emits `RELIC.ANCHOR` for relics and the Debug enum form
    (`FirePotion`) for potions; both mean the wire's `ANCHOR` / `FIRE_POTION`.
    """
    return engine_id.split(".")[-1].replace("_", "").upper()


def test_grafts_deck_relics_gold_and_hp(live_run):
    imported = _imported(live_run)
    game = RustGame()
    game.start_imported(imported.run)

    assert game.rm.current_hp == 55
    assert game.rm.max_hp == 80
    assert game.rm.gold == 217
    # The deck is the pasted one, not the starter — including WHICH copy is
    # upgraded. "two Strikes, one of them upgraded" must land as exactly that.
    deck = sorted(tuple(entry) for entry in game.rm.master_deck_detailed())
    assert deck == sorted(
        [
            ("StrikeIronclad", False),
            ("StrikeIronclad", True),
            ("DefendIronclad", False),
            ("Bash", False),
        ]
    )
    # The engine spells inventory ids in its own Debug form (`FirePotion`,
    # `RELIC.ANCHOR`); compare on a shape-insensitive key rather than pinning
    # a formatting detail this test does not care about.
    assert {"BURNINGBLOOD", "ANCHOR"} <= {_key(r) for r in game.rm.relics()}
    assert [_key(p) for p in game.rm.potions()] == ["FIREPOTION"]


def test_relic_grants_do_not_upgrade_the_grafted_deck(live_run):
    """Whetstone-style on-obtain upgrades must not chew on the imported deck.

    ``force_add_relic`` fires ``after_obtained`` eagerly, so the graft empties
    the deck BEFORE granting relics. Without that ordering an imported deck
    silently gains upgrades the live run never had.
    """
    msg = _message(live_run)
    msg["run_state"]["relics"] = [{"id": "BURNING_BLOOD"}, {"id": "WHETSTONE"}]
    imported = parse_pasted_state(json.dumps(msg))
    game = RustGame()
    game.start_imported(imported.run)

    upgraded = [entry for entry in game.rm.master_deck_detailed() if entry[1]]
    assert len(upgraded) == 1, "only the copy the paste said was upgraded"


def test_parks_on_the_pasted_node_with_its_own_children(live_run):
    imported = _imported(live_run)
    game = RustGame()
    state = game.start_imported(imported.run)

    assert state["decision"] == "map_select"
    assert tuple(game.rm.map_coord()) == imported.run.coord
    # Being PLACED on the map, not dropped beside it: the run has real moves.
    assert game.rm.available_actions(), "a placed run has somewhere to go"
    notes = game.import_notes()
    assert not any("does not have" in n for n in notes), notes


def test_a_coord_off_this_seeds_map_is_refused(live_run):
    """Paste and engine disagreeing about the run is louder than a quiet restart."""
    msg = _message(live_run)
    msg["map"]["current"] = {"col": 99, "row": 99}
    imported = parse_pasted_state(json.dumps(msg))
    with pytest.raises(ValueError, match="not a live node"):
        RustGame().start_imported(imported.run)


def test_rng_cursors_are_restored(live_run):
    """The point of the exercise: the next roll is the live run's roll."""
    # Advance the source run so its cursors sit somewhere non-trivial.
    live_run.dump_card_reward(3, "Monster")
    counters = dict(live_run.rng_counters())
    assert counters["rewards"] > 0

    imported = _imported(live_run)
    assert imported.run.rng is not None

    game = RustGame()
    game.start_imported(imported.run)
    restored = dict(game.rm.rng_counters())
    for stream in ("rewards", "shops", "transformations", "monster_ai", "niche", "up_front"):
        assert restored[stream] == counters[stream], stream
    notes = game.import_notes()
    assert not [n for n in notes if "not restored" in n], notes


def test_a_foreign_seeds_cursors_are_refused_not_applied(live_run):
    msg = _message(live_run)
    msg["rng"] = {"run_seed": 12345, "counters": {"rewards": 3}}
    imported = parse_pasted_state(json.dumps(msg))
    game = RustGame()
    game.start_imported(imported.run)
    # Refused, and said so: a cursor from another run is worse than none.
    assert any("not restored" in n for n in game.import_notes())


def test_seek_rebuilds_an_imported_run(live_run, tmp_path):
    """`seek` must reconstruct through the import, not as a fresh seed run."""
    from workbench.branching import BranchingLiveSession

    imported = _imported(live_run)
    sess = BranchingLiveSession(logs_dir=str(tmp_path / "logs"), backend="rust", loc={})
    root = sess.start_imported(imported.run)
    assert root["decision"] == "map_select"

    back = sess.seek(sess._root_id)
    assert back["decision"] == "map_select"
    assert sess.game.rm.current_hp == 55
    assert sess.game.rm.gold == 217


def test_a_v0111_seed_above_i64_max_is_restored():
    """v0.111.0 master seeds are a full u64 (GH #788); about half of them are
    above i64::MAX and once raised OverflowError out of the import (#855)."""
    import sts2_rs

    v111 = "v0.111.0"
    seed, rm = next(
        (s, r)
        for s, r in (
            (
                s,
                sts2_rs.RunManager(
                    string_seed=s, character="Ironclad", act1="roll", game_version=v111
                ),
            )
            for s in (f"BIGSEED{i}" for i in range(64))
        )
        if r.run_seed > 2**63 - 1
    )
    rm.dump_card_reward(3, "Monster")
    counters = dict(rm.rng_counters())
    msg = _message(rm, seed=seed)
    imported = parse_pasted_state(json.dumps(msg))
    assert imported.run.rng is not None
    assert imported.run.rng["run_seed"] == rm.run_seed

    game = RustGame(game_version=v111)
    game.start_imported(imported.run)
    assert dict(game.rm.rng_counters())["rewards"] == counters["rewards"]
    assert not [n for n in game.import_notes() if "not restored" in n]


def test_a_negative_32bit_seed_reads_as_its_uint_view():
    """The Rust ``state_import::rng_block`` reads a signed 32-bit seed as its
    ``uint`` view; the Python parser must agree or the two hosts diverge."""
    from workbench.state_import import _rng_block

    block = _rng_block({"rng": {"run_seed": -1, "counters": {"rewards": 1}}})
    assert block == {"run_seed": 0xFFFF_FFFF, "counters": {"rewards": 1}}
    assert _rng_block({"rng": {"run_seed": 2**64, "counters": {"rewards": 1}}}) is None
