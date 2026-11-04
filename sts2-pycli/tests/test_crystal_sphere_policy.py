"""GH #872: the Crystal Sphere is played by a geometric rule, not at random.

The messages here are shaped exactly like ``SimCrystalSphereScreenHandler``'s
(PROTOCOL.md ``crystal_sphere``): one ``divine_cell`` option per hidden cell in
the container's x-major order, plus a trailing ``proceed`` once the screen
enables it.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import crystal_sphere_policy as csp
import pytest
import realgame_ppo as rp

_MOD_TWIN = Path(__file__).resolve().parents[2] / "spire-mentor-mod" / "CrystalSpherePick.cs"


def _fresh_hidden() -> list[tuple[int, int]]:
    """The 11x11 grid minus the four Manhattan-radius-2 corner diamonds the
    minigame clears before placing items (``CrystalSphereMinigame`` ctor)."""
    n = csp.GRID_SIZE
    return [(x, y) for x in range(n) for y in range(n) if min(x, n - 1 - x) + min(y, n - 1 - y) > 2]


def _message(hidden: list[tuple[int, int]], *, proceed: bool = False) -> dict[str, Any]:
    options: list[dict[str, Any]] = [
        {"index": i, "action": "divine_cell", "x": x, "y": y, "enabled": True}
        for i, (x, y) in enumerate(hidden)
    ]
    if proceed:
        options.append({"index": len(hidden), "action": "proceed", "enabled": True})
    return {"type": "crystal_sphere", "options": options}


def _divine(hidden: list[tuple[int, int]], x: int, y: int) -> list[tuple[int, int]]:
    """The Big tool: clear (x, y) and its eight neighbours."""
    return [(a, b) for a, b in hidden if abs(a - x) > 1 or abs(b - y) > 1]


def _cell(action: dict[str, Any], hidden: list[tuple[int, int]]) -> tuple[int, int]:
    return hidden[action["index"]]


class TestRule:
    def test_a_fresh_board_starts_the_first_column_not_the_first_cell(self) -> None:
        hidden = _fresh_hidden()
        action = csp.choose(_message(hidden))
        assert action is not None
        assert _cell(action, hidden) == (3, 2)
        assert hidden[0] == (0, 3), "the old F7 pick: an edge cell"

    def test_six_divinations_clear_a_six_by_nine_rectangle(self) -> None:
        hidden = _fresh_hidden()
        order = []
        for _ in range(6):
            action = csp.choose(_message(hidden))
            assert action is not None
            x, y = _cell(action, hidden)
            order.append((x, y))
            hidden = _divine(hidden, x, y)
        assert order == list(csp.CENTERS)
        assert len(_fresh_hidden()) - len(hidden) == 54, "the blocks never overlap"

    def test_proceed_wins_once_offered(self) -> None:
        # Proceed only enables when the divinations are spent; a cell click
        # then is a no-op that would loop the screen.
        hidden = _fresh_hidden()
        action = csp.choose(_message(hidden, proceed=True))
        assert action is not None
        assert action["index"] == len(hidden)
        assert action["annotations"]["policy"] == csp.POLICY_NAME

    def test_a_centre_cleared_by_hand_is_skipped(self) -> None:
        hidden = _divine(_fresh_hidden(), 3, 2)
        action = csp.choose(_message(hidden))
        assert action is not None
        assert _cell(action, hidden) == (3, 5)

    def test_past_the_list_the_densest_block_wins(self) -> None:
        hidden = [(0, 5), (9, 9), (9, 10), (10, 9), (10, 10), (5, 0)]
        assert hidden[csp.pick_cell(hidden)] == (9, 9)
        lone = [(0, 5), (5, 4), (10, 5)]
        assert lone[csp.pick_cell(lone)] == (5, 4), "tie: nearest the centre"

    def test_disabled_and_malformed_options_are_ignored(self) -> None:
        msg = {
            "type": "crystal_sphere",
            "options": [
                {"index": 0, "action": "divine_cell", "x": 3, "y": 2, "enabled": False},
                {"index": 1, "action": "divine_cell", "x": "?", "y": 5},
                {"index": 2, "action": "divine_cell", "x": 7, "y": 7, "enabled": True},
                {"index": 3, "action": "proceed", "enabled": False},
            ],
        }
        action = csp.choose(msg)
        assert action is not None
        assert action["index"] == 2

    def test_nothing_usable_returns_none(self) -> None:
        assert csp.choose({"type": "crystal_sphere", "options": []}) is None
        assert csp.pick_cell([]) is None

    def test_beats_the_old_picks_on_simulated_boards(self) -> None:
        """A cheap reproduction of the measurement behind the order: on random
        boards (placement per the decomp), the rule reveals more items than
        the first hidden cell or a random one at both divination counts."""
        items = [(4, 4)] + [(1, 3)] * 2 + [(2, 2)] * 5 + [(1, 1)] * 5 + [(2, 1)] * 2

        def revealed(pick: Any, divs: int, rng: random.Random) -> int:
            hidden = set(_fresh_hidden())
            placed: list[set[tuple[int, int]]] = []
            free = set(hidden)
            for w, h in items:
                cand = [
                    (x, y)
                    for x in range(csp.GRID_SIZE - w + 1)
                    for y in range(csp.GRID_SIZE - h + 1)
                    if all((x + i, y + j) in free for i in range(w) for j in range(h))
                ]
                x, y = rng.choice(cand)
                cells = {(x + i, y + j) for i in range(w) for j in range(h)}
                free -= cells
                placed.append(cells)
            for _ in range(divs):
                order = sorted(hidden)
                cx, cy = order[pick(order, rng)]
                hidden -= {(cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            return sum(not (cells & hidden) for cells in placed)

        def rule(order: list[tuple[int, int]], _rng: random.Random) -> int:
            i = csp.pick_cell(order)
            assert i is not None
            return i

        def first(_order: list[tuple[int, int]], _rng: random.Random) -> int:
            return 0

        def uniform(order: list[tuple[int, int]], rng: random.Random) -> int:
            return rng.randrange(len(order))

        for divs in (3, 6):
            totals = {}
            for name, pick in (("rule", rule), ("first", first), ("random", uniform)):
                rng = random.Random(872)
                totals[name] = sum(revealed(pick, divs, rng) for _ in range(300))
            assert totals["rule"] > totals["random"] > totals["first"], (divs, totals)


class TestDriverLane:
    def test_the_driver_answers_with_the_rule_and_names_it(self) -> None:
        hidden = _fresh_hidden()
        action, log_extra = rp.pick_noncombat_action_ppo(
            _message(hidden), random.Random(0), policies=None, session={}
        )
        assert log_extra is None, "a rule answer is not a PPO fallback"
        assert _cell(action, hidden) == (3, 2)
        assert action["annotations"]["policy"] == csp.POLICY_NAME

    def test_an_unusable_message_still_gets_the_random_fallback(self) -> None:
        msg = {"type": "crystal_sphere", "options": [{"index": 0, "label": "?"}]}
        action, log_extra = rp.pick_noncombat_action_ppo(
            msg, random.Random(0), policies=None, session={}
        )
        assert log_extra is None
        assert action == {"action": "choose", "index": 0}


def test_the_mod_twin_plays_the_same_order() -> None:
    """``CrystalSpherePick.cs`` (F7's lane) and this module must agree."""
    if not _MOD_TWIN.is_file():
        pytest.skip(f"mod source not in this checkout: {_MOD_TWIN}")
    text = _MOD_TWIN.read_text(encoding="utf-8")
    block = re.search(r"Centers\s*=\s*\{(.*?)\};", text, re.S)
    assert block, "Centers array not found"
    pairs = [(int(a), int(b)) for a, b in re.findall(r"\((\d+),\s*(\d+)\)", block.group(1))]
    assert tuple(pairs) == csp.CENTERS
