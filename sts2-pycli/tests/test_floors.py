"""GH #1051: the game floor <-> engine ``total_floor`` conversion and its readers.

The game's ``TotalFloor`` counts every act's start room (Neow, the Ancients);
the engine's ``total_floor`` counts none, so the offset is the 1-based act.
``floors`` is the one Python home of that rule; ``realgame_ppo`` (pinned in
``test_realgame_ppo.TestTotalFloorOffset``), ``ppo.pro_decisions`` and
``workbench.backends.rust_game`` all go through it.
"""

from __future__ import annotations

import floors
import pytest
from ppo import pro_decisions


@pytest.mark.parametrize(
    ("act", "want"),
    [(1, 1), (3, 3), (None, 1), (0, 1), (-2, 1)],
)
def test_act_number_reads_a_missing_or_nonpositive_act_as_act_1(act, want) -> None:
    assert floors.act_number(act) == want


@pytest.mark.parametrize(
    ("game_floor", "act", "engine_floor"),
    [
        (1, 1, 0),  # Neow
        (19, 2, 17),  # act 2's first map choice after a 17-floor act 1
        (48, 3, 45),  # a won A0 run
        (12, None, 11),
        (12, 0, 11),
        (None, 2, 0),
        (1, 3, 0),  # never negative
    ],
)
def test_engine_total_floor_subtracts_the_act(game_floor, act, engine_floor) -> None:
    assert floors.engine_total_floor(game_floor, act) == engine_floor


@pytest.mark.parametrize(("total_floor", "act_index"), [(0, 1), (17, 2), (45, 3)])
def test_game_floor_round_trips(total_floor, act_index) -> None:
    game = floors.game_floor(total_floor, act_index)
    assert game == total_floor + act_index
    assert floors.engine_total_floor(game, act_index) == total_floor


@pytest.mark.parametrize(("act", "floor", "engine_floor"), [(1, 12, 11), (2, 19, 17), (3, 48, 45)])
def test_pro_decisions_metrics_use_the_act_offset(act, floor, engine_floor) -> None:
    """The SpireCodex replay's ``floor`` is the game's TotalFloor, like the
    bridge's; a flat ``- 1`` scored acts 2-3 one and two floors too deep."""
    rs = pro_decisions.RunState({})
    rs.feed({"t": "noop", "floor": floor, "act": act})
    metrics = rs.metrics()
    assert metrics["total_floor"] == engine_floor
    assert metrics["act_index"] == act
