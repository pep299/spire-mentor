"""GH #858 through the PyO3 surface the combat pro bench builds fights with.

``sts2_rs.build_scenario_state`` must open each fight with the C#
``AfterAddedToRoom`` amount at the requested ascension: Vantom's Slippery is
``GetValueIfAscension(ToughEnemies, 9, 8)`` (``Vantom.cs:64``) and the Frog
Knight's Plating ``(.., 19, 15)`` (``FrogKnight.cs:43``), which is also its
opening block. Before the fix every A8+ fight opened with the A0 amounts.
"""

from __future__ import annotations

import pytest

sts2_rs = pytest.importorskip("sts2_rs")

DECK = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]


def _enemy(encounter: str, ascension: int) -> dict:
    state = sts2_rs.build_scenario_state(1, "Ironclad", DECK, 80, 80, [], encounter, ascension)
    enemies = [e for e in sts2_rs.state_to_dict(state)["encounter"] if not e.get("is_dead")]
    assert len(enemies) == 1, f"{encounter}: expected one enemy, got {len(enemies)}"
    return enemies[0]


def _power(enemy: dict, name: str) -> int:
    """The amount of ``name`` on ``enemy``; asserts it is present at all."""
    hits = [p for p in enemy["powers"] if p["name"] in (name, f"{name}_POWER")]
    assert hits, f"{name} missing from {[p['name'] for p in enemy['powers']]}"
    return sum(p["amount"] for p in hits)


@pytest.mark.parametrize(("ascension", "want"), [(7, 8), (8, 9), (10, 9)])
def test_vantom_opens_with_scaled_slippery(ascension: int, want: int) -> None:
    assert _power(_enemy("VANTOM_BOSS", ascension), "SLIPPERY") == want


@pytest.mark.parametrize(("ascension", "want"), [(7, 15), (10, 19)])
def test_frog_knight_plating_and_opening_block(ascension: int, want: int) -> None:
    enemy = _enemy("FROG_KNIGHT_NORMAL", ascension)
    assert _power(enemy, "PLATING") == want
    assert enemy["block"] == want
