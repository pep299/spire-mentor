"""Co-op: the teammate reaches the engine, so ``AnyAlly`` cards are played.

Without ``$.allies`` the engine sees no living teammate and, like C#'s
``NoLivingAllies``, refuses every ``AnyAlly`` card -- so the advisor never
recommended Lift, Mimic, Intercept and the rest in a co-op fight.
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
TEAMMATE = {"character": "SILENT", "hp": 50, "max_hp": 70, "block": 4, "is_dead": False}


def _board(allies: list[dict[str, Any]] | None) -> dict[str, Any]:
    board = json.loads(FIXTURE.read_text(encoding="utf-8"))["state"]
    if allies is not None:
        board["allies"] = allies
    return board


def _lift_state(wire: dict[str, Any]):
    wire = copy.deepcopy(wire)
    lift = wire["deck"]["hand"][0]
    lift.update(
        {"id": "LIFT", "upgraded": False, "cost_override": 1, "base_cost": 1, "type": "Skill"}
    )
    wire["player"]["energy"] = 3
    return sts2_rs.state_from_dict(wire)


def _plays_of(state, card_id: str) -> list[dict[str, Any]]:
    hand = sts2_rs.state_to_dict(state)["deck"]["hand"]
    return [
        a
        for a in state.legal_actions()
        if a.get("action") == "play" and hand[a["card_index"]]["id"] == card_id
    ]


def test_teammate_rides_the_wire_and_solo_stays_unchanged() -> None:
    solo = translate(to_headless_combat_play(_board(None)))
    assert "allies" not in solo
    wire = translate(to_headless_combat_play(_board([TEAMMATE])))
    assert wire["allies"] == [{"hp": 50, "max_hp": 70, "block": 4}]
    assert sts2_rs.state_to_dict(sts2_rs.state_from_dict(wire))["allies"] == wire["allies"]


def test_lift_is_playable_only_with_a_teammate_and_blocks_them() -> None:
    solo = translate(to_headless_combat_play(_board(None)))
    assert _plays_of(_lift_state(solo), "LIFT") == []

    coop = translate(to_headless_combat_play(_board([TEAMMATE])))
    state = _lift_state(coop)
    plays = _plays_of(state, "LIFT")
    assert len(plays) == 1
    block_before = sts2_rs.state_to_dict(state)["player"]["block"]
    state.apply(plays[0])
    after = sts2_rs.state_to_dict(state)
    assert after["allies"][0]["block"] == 4 + 11
    assert after["player"]["block"] == block_before


def test_a_dead_teammate_makes_any_ally_cards_unplayable() -> None:
    dead = dict(TEAMMATE, hp=0, is_dead=True)
    wire = translate(to_headless_combat_play(_board([dead])))
    assert _plays_of(_lift_state(wire), "LIFT") == []


def _mawler_with_lift(allies: list[dict[str, Any]]):
    """Mawler about to hit for 14; we hold Lift, Strike and Defend with 1 energy."""
    wire = sts2_rs.state_to_dict(sts2_rs.CombatState(seed=1, encounter="Mawler"))
    wire["encounter"][0]["intents"] = [{"amount": 14, "effective_amount": 14, "kind": "Attack"}]
    hand = wire["deck"]["hand"]
    hand[0].update(id="LIFT", base_damage=0, base_block=11)
    hand[2].update(id="DEFEND_IRONCLAD", base_damage=0, base_block=5)
    wire["player"]["energy"] = 1
    wire["allies"] = allies
    return sts2_rs.state_from_dict(wire)


def _lift_gain(state) -> float:
    """Q of playing Lift over ending the turn."""
    out = sts2_rs.mcts_recommend(state, sims=1500, threads=1, seed=3)
    hand = sts2_rs.state_to_dict(state)["deck"]["hand"]
    q = {}
    for row in out["actions"]:
        a = row["action"]
        q[hand[a["card_index"]]["id"] if a.get("action") == "play" else a["action"]] = row["q"]
    return q["LIFT"] - q["end_turn"]


def test_the_search_values_blocking_an_exposed_teammate() -> None:
    """The enemy turn hits the teammate too and their lost HP counts, so
    Lift's 11 block on them is worth 11 HP -- unless they are already safe."""
    exposed = _lift_gain(_mawler_with_lift([{"hp": 40, "max_hp": 70, "block": 0}]))
    assert exposed == pytest.approx(11 * 0.5 * 100 / 14000, abs=0.01)
    safe = _lift_gain(_mawler_with_lift([{"hp": 40, "max_hp": 70, "block": 20}]))
    assert abs(safe) < 0.01


def test_with_two_teammates_each_is_its_own_play_and_gets_the_card() -> None:
    """GH #1017: with three players Lift is offered once per living teammate,
    ``ally_index`` naming the ``allies[]`` entry (no enemy ``target_index``),
    and lands on that one."""
    second = dict(TEAMMATE, character="DEFECT", block=0)
    wire = translate(to_headless_combat_play(_board([TEAMMATE, second])))
    plays = _plays_of(_lift_state(wire), "LIFT")
    assert sorted(p.get("ally_index") for p in plays) == [0, 1]
    assert all(p.get("target_index") is None for p in plays)
    for ally in (0, 1):
        state = _lift_state(wire)
        play = next(p for p in _plays_of(state, "LIFT") if p.get("ally_index") == ally)
        state.apply(play)
        blocks = [a["block"] for a in sts2_rs.state_to_dict(state)["allies"]]
        want = [4, 0]
        want[ally] += 11
        assert blocks == want


def test_a_dead_second_teammate_leaves_one_untargeted_play() -> None:
    dead = dict(TEAMMATE, hp=0, is_dead=True)
    wire = translate(to_headless_combat_play(_board([dead, TEAMMATE])))
    plays = _plays_of(_lift_state(wire), "LIFT")
    assert [(p.get("target_index"), p.get("ally_index")) for p in plays] == [(None, None)]


def test_the_advisor_names_the_teammate_an_any_ally_card_goes_to() -> None:
    from enemy_labels import ally_label
    from realgame_advisor import _best_to_action, _tag_label
    from sts2_hint.recommender import HintTag

    allies = [{"character": "SILENT"}, {"character": "DEFECT"}, {"character": "SILENT"}]
    assert ally_label(allies, 1) == "DEFECT"
    assert ally_label(allies, 2) == "SILENT #2"
    assert ally_label(allies, 3) is None
    board = {
        "allies": allies,
        "enemies": [{"name": "Cultist"}],
        "hand": [{"name": "Lift", "target_type": "AnyAlly"}],
    }
    tag = HintTag(stable_idx=0, win_prob=0.5, bucket_label="", is_best=True, ally_index=2)
    assert _tag_label(tag, board) == "Lift → SILENT #2"
    assert _best_to_action(tag, board) == {"action": "play", "card_index": 0, "ally_index": 2}


def _human_lift(ally: int | None) -> dict[str, Any]:
    """The mod's report of the human's Lift, at ``allies[ally]`` (GH #1017)."""
    msg: dict[str, Any] = {
        "type": "human_action",
        "decision_type": "combat_action",
        "action": "card_play",
        "detail": "IRONCLAD played LIFT",
        "card": {"id": "LIFT", "upgraded": False},
        "target_index": None,
    }
    if ally is not None:
        msg["ally_index"] = ally
    return msg


@pytest.mark.parametrize("ally", [0, 1])
def test_the_mirror_follows_a_human_play_to_the_teammate_they_picked(ally: int) -> None:
    """The mirror aims the human's Lift where the human did, not at the first
    living teammate. Checked on the mirror's own prediction: reconciling with
    the next board would hide a miss, since a teammate's block is a field
    their own moves change (``teammate_adoptions``, not a desync)."""
    from realgame_advisor_mirror import AdvisorMirror
    from sts2_hint.recommender import RecommendConfig

    second = dict(TEAMMATE, character="DEFECT", block=0)
    wire = translate(to_headless_combat_play(_board([TEAMMATE, second])))
    world = _lift_state(wire)
    advisor = AdvisorMirror(RecommendConfig(sims=24))
    board = {"act": 1, "floor": 3}
    assert advisor.on_combat_board(sts2_rs.state_to_dict(world), board) is not None
    play = next(p for p in _plays_of(world, "LIFT") if p.get("ally_index") == ally)
    world.apply_step(play)
    advisor.on_world_action(_human_lift(ally))
    assert advisor.stats["followed"] == 1
    assert advisor.live is not None
    predicted = [a["block"] for a in sts2_rs.state_to_dict(advisor.live.state)["allies"]]
    assert predicted == [a["block"] for a in sts2_rs.state_to_dict(world)["allies"]]


def test_the_coach_grades_a_human_play_against_its_teammates_row() -> None:
    """With a row per teammate, the human's Lift at ``allies[1]`` is that row."""
    from realgame_coach import _match_row_index, native_action

    info = {"hand": [("LIFT", False)], "potions": []}
    native = native_action(_human_lift(1), info)
    assert native == {"action": "play", "card_index": 0, "ally_index": 1}
    rows = [
        {"stable_idx": 0, "target_index": None, "ally_index": 0},
        {"stable_idx": 0, "target_index": None, "ally_index": 1},
    ]
    assert _match_row_index(rows, native) == 1
    plain = native_action(_human_lift(None), info)
    assert plain == {"action": "play", "card_index": 0, "target_index": None}
    assert _match_row_index([{"stable_idx": 0, "target_index": None}], plain) == 0
