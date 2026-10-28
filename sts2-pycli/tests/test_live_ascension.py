"""GH #848: every live combat lane rebuilds the board at the RUN's ascension.

The mod reports ``run_state.ascension``. A lane that drops it plans every enemy
turn after the current telegraph against A0 tables, so at A9+ the search sees
softer fights than the real game deals.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
import realgame_advisor as ra
import realgame_mcts as rm
from realgame_adapter import run_ascension
from sts2_hint import adapter
from sts2_hint.recommender import RecommendConfig


def _board(ascension: int) -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
    state = json.loads(path.read_text(encoding="utf-8"))["state"]
    state.pop("game_version", None)
    state["type"] = "combat_action"
    state["run_state"]["ascension"] = ascension
    return state


class _Stop(Exception):
    pass


@pytest.fixture
def translated(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the ``ascension`` every ``adapter.translate`` call is given."""
    seen: list[int] = []
    real = adapter.translate

    def spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs.get("ascension", 0))
        return real(*args, **kwargs)

    monkeypatch.setattr(adapter, "translate", spy)
    return seen


def test_run_ascension_prefers_the_mod_and_falls_back_to_the_default() -> None:
    assert run_ascension({"ascension": 10}) == 10
    assert run_ascension({"ascension": 0}, default=7) == 0
    assert run_ascension({}, default=7) == 7
    assert run_ascension({}) == 0


def test_advisor_lane_rebuilds_at_the_run_ascension(
    monkeypatch: pytest.MonkeyPatch, translated: list[int]
) -> None:
    sts2_rs = pytest.importorskip("sts2_rs")

    def fake_search(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise _Stop

    monkeypatch.setattr(sts2_rs, "mcts_recommend", fake_search)
    with pytest.raises(_Stop):
        ra.compute_combat_advice(_board(10), RecommendConfig(threads=1), 8)
    assert translated == [10]


def test_mcts_driver_lane_rebuilds_at_the_run_ascension(translated: list[int]) -> None:
    pytest.importorskip("sts2_rs")
    rm.pick_combat_action_mcts(_board(10), RecommendConfig(sims=4, depth=4), random.Random(0))
    assert translated and set(translated) == {10}


def test_mirrored_card_select_rebuilds_at_the_run_ascension(translated: list[int]) -> None:
    class _Mirror:
        def reconcile(self, wire: dict[str, Any]) -> None:
            raise _Stop

    state = {"type": "card_select", "run_state": {"ascension": 9}, "combat": _board(9)}
    with pytest.raises(_Stop):
        rm.pick_card_select_mirrored(state, RecommendConfig(), _Mirror())  # type: ignore[arg-type]
    assert translated == [9]
