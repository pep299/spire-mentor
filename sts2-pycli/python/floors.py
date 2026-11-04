"""The game's floor number vs the engine's ``total_floor`` (GH #1051).

The game's ``RunState.TotalFloor`` sums ``MapPointHistory``, which counts
every act's start room (Neow, the Ancients) as a floor. The engine's
``rm.total_floor()`` counts none of them. So the two differ by the 1-based act
number: 1 at Neow, 2 through act 2, 3 through act 3 and the A10 second boss
(a won A0 run is 45 in the engine, 48 in the game).

Every reader of a game floor (the bridge's ``floor``, a pasted position, a
SpireCodex replay) converts through here, and so does the engine side that
writes one. Rust twins: ``sts2_core::runstate::engine_total_floor`` and
``RunState::game_floor``.
"""

from __future__ import annotations

from typing import Any


def act_number(act: Any) -> int:
    """The 1-based act; a missing, zero or negative act reads as act 1."""
    return max(1, int(act or 1))


def engine_total_floor(game_floor: Any, act: Any) -> int:
    """The engine's ``total_floor`` for a game floor in 1-based ``act``; never negative."""
    return max(0, int(game_floor or 0) - act_number(act))


def game_floor(total_floor: int, act_index: int) -> int:
    """The floor the game shows for the engine's ``total_floor`` in 1-based ``act_index``."""
    return total_floor + act_number(act_index)
