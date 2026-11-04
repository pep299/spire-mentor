"""Co-op: throwing an ``AnyPlayer`` potion at a teammate (GH #1017).

The engine offers ``drink_potion`` with an ``ally_index`` beside the plain
drink (``Action::DrinkPotionAtAlly``). The advisor recommends it as a mod
``potion`` verb with that ``ally_index``, and the mirror, the coach and the
co-op replay follow a throw the world reports.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from realgame_adapter import to_headless_combat_play
from sts2_hint.adapter import translate

sts2_rs = pytest.importorskip("sts2_rs")

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
)
BLOCK = {"empty": False, "id": "BLOCK_POTION", "name": "Block Potion"}


def _mate(character: str, net_id: str) -> dict[str, Any]:
    return {
        "character": character,
        "net_id": net_id,
        "hp": 50,
        "max_hp": 70,
        "block": 0,
        "is_dead": False,
        "energy": 3,
        "max_energy": 3,
        "powers": [],
        "hand": [],
        "draw_pile": {"type": "Draw", "count": 0, "cards": []},
        "discard_pile": {"type": "Discard", "count": 0, "cards": []},
        "exhaust_pile": {"type": "Exhaust", "count": 0, "cards": []},
        "play_pile": {"type": "Play", "count": 0, "cards": []},
        "relics": [],
        "stars": 0,
        "ended_turn": False,
    }


def _board(mates: int) -> dict[str, Any]:
    """The fixture's Ironclad holding a Block Potion in belt slot 1, with
    ``mates`` teammates (their whole seats sent)."""
    board = json.loads(FIXTURE.read_text(encoding="utf-8"))["state"]
    board["player"]["net_id"] = "1"
    board["potions"][1] = {"slot": 1, **BLOCK}
    board["allies"] = [_mate(c, str(2 + i)) for i, c in enumerate(["SILENT", "DEFECT"][:mates])]
    return board


def _state(mates: int):
    return sts2_rs.state_from_dict(translate(to_headless_combat_play(_board(mates))))


def _uses(state) -> list[int | None]:
    return [a.get("ally_index") for a in state.legal_actions() if a.get("action") == "drink_potion"]


def test_the_engine_offers_a_throw_per_teammate_beside_the_drink() -> None:
    assert _uses(_state(2)) == [None, 0, 1]
    assert _uses(_state(1)) == [None, 0]
    assert _uses(_state(0)) == [None]


@pytest.mark.parametrize("ally", [0, 1])
def test_a_thrown_block_potion_lands_on_that_teammate(ally: int) -> None:
    state = _state(2)
    throw = next(a for a in state.legal_actions() if a.get("ally_index") == ally)
    assert throw["target_index"] is None
    state.apply(throw)
    after = sts2_rs.state_to_dict(state)
    want = [0, 0]
    want[ally] = 12
    assert [a["block"] for a in after["allies"]] == want
    assert after["player"]["block"] == 0


def test_the_advisor_tells_the_mod_to_throw_it() -> None:
    from realgame_advisor import _best_to_action, _tag_label
    from realgame_mcts import hint_tag_to_action
    from sts2_hint.recommender import POTION_STABLE_IDX_BASE, HintTag

    board = _board(2)
    tag = HintTag(
        stable_idx=POTION_STABLE_IDX_BASE, win_prob=0.5, bucket_label="", is_best=True, ally_index=1
    )
    # Dense engine slot 0 is the board's belt slot 1.
    assert _best_to_action(tag, board) == {"action": "potion", "slot": 1, "ally_index": 1}
    assert hint_tag_to_action(tag) == {"action": "potion", "slot": 0, "ally_index": 1}
    assert _tag_label(tag, board).endswith("→ DEFECT")


def _human_throw(ally: int | None) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "type": "human_action",
        "decision_type": "combat_action",
        "action": "card_play",
        "detail": "IRONCLAD used BLOCK_POTION",
        "card": None,
        "potion": {"id": "BLOCK_POTION"},
        "target_index": None,
    }
    if ally is not None:
        msg["ally_index"] = ally
    return msg


@pytest.mark.parametrize("ally", [None, 0, 1])
def test_the_mirror_follows_a_drink_or_a_throw(ally: int | None) -> None:
    from realgame_advisor_mirror import AdvisorMirror
    from sts2_hint.recommender import RecommendConfig

    world = _state(2)
    advisor = AdvisorMirror(RecommendConfig(sims=24))
    floor = {"act": 1, "floor": 3}
    assert advisor.on_combat_board(sts2_rs.state_to_dict(world), floor) is not None
    world.apply(
        next(
            a
            for a in world.legal_actions()
            if a.get("action") == "drink_potion" and a.get("ally_index") == ally
        )
    )
    advisor.on_world_action(_human_throw(ally))
    assert advisor.stats["followed"] == 1
    assert advisor.live is not None
    got = sts2_rs.state_to_dict(advisor.live.state)
    want = sts2_rs.state_to_dict(world)
    assert [a["block"] for a in got["allies"]] == [a["block"] for a in want["allies"]]
    assert got["player"]["block"] == want["player"]["block"]


def test_the_coach_grades_a_throw_against_its_own_row() -> None:
    from realgame_coach import _match_row_index, native_action, native_from_row

    info = {"hand": [], "potions": ["BLOCK_POTION"]}
    native = native_action(_human_throw(1), info)
    assert native == {"action": "potion", "slot": 0, "ally_index": 1}
    rows = [
        {"stable_idx": -100, "target_index": None, "ally_index": None},
        {"stable_idx": -100, "target_index": None, "ally_index": 0},
        {"stable_idx": -100, "target_index": None, "ally_index": 1},
    ]
    assert _match_row_index(rows, native) == 2
    assert native_from_row(rows[2]) == native


def test_the_replay_follows_a_logged_throw() -> None:
    """Our Block Potion thrown at the Defect (net id 3) in a 3-player fight."""
    import coop_replay

    before = _board(2)
    after = copy.deepcopy(before)
    after["potions"][1] = {"slot": 1, "empty": True}
    after["allies"][1]["block"] = 12
    a = to_headless_combat_play(before)
    b = to_headless_combat_play(after)
    b["combat_log_raw"] = [
        {
            "kind": "potion",
            "round": 1,
            "side": "player",
            "potion": "BLOCK_POTION",
            "owner": "1",
            "target": {"kind": "player", "id": "DEFECT", "uid": None, "net_id": "3"},
            "target_index": None,
        }
    ]
    res = coop_replay.replay_pair(a, b)
    assert res.get("skipped") is None, res
    assert res["potions"] == 1
    assert res["diffs"] == {}
