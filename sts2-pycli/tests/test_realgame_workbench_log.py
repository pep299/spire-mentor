"""WorkbenchRunLogger -> TimelineStore round trip (live-run mirroring).

The advisor mirrors live runs into GameLogger-schema JSONLs the workbench
replays natively; this pins the round trip that validated the lane before it
shipped 2026-08-21: a decision + human action + run end written by the
logger must come back out of ``TimelineStore.load_jsonl`` as steps the
workbench can render.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pytest
from realgame_workbench_log import WorkbenchRunLogger
from workbench.timeline import TimelineStore

EVENT_MSG: dict[str, Any] = {
    "type": "advice_request",
    "advice_for": "event",
    "seed": "S1",
    "floor": 3,
    "act": 1,
    "state_id": 7,
    "run_state": {
        "character": "IRONCLAD",
        "ascension": 8,
        "hp": 60,
        "max_hp": 75,
        "gold": 120,
        "master_deck": [{"id": "STRIKE_IRONCLAD", "upgrade_level": 0}],
        "relics": [{"id": "BURNING_BLOOD"}],
        "potions": [],
    },
    "options": [
        {
            "index": 0,
            "label": "Decipher",
            "description": "Lose 3 Max HP.",
            "enabled": True,
            "event_id": "TABLET_OF_TRUTH",
        },
        {
            "index": 1,
            "label": "Smash",
            "description": "Heal 20 HP.",
            "enabled": True,
            "event_id": "TABLET_OF_TRUTH",
        },
    ],
}


def _run_log(tmp_path: Path) -> Path:
    wlog = WorkbenchRunLogger(str(tmp_path))
    assert wlog.log_decision(EVENT_MSG, "event") is not None
    path = wlog.path
    assert path is not None
    wlog.log_human_action({"detail": "Smash", "action": "choose"})
    wlog.log_run_end(
        {
            "type": "game_over",
            "victory": False,
            "floor": 3,
            "act": 1,
            "run_state": EVENT_MSG["run_state"],
        }
    )
    return Path(path)


class TestRoundTrip:
    def test_decision_action_and_terminal_come_back_as_steps(self, tmp_path: Path) -> None:
        store = TimelineStore.load_jsonl(_run_log(tmp_path))
        assert len(store) == 2  # the event decision + the terminal state
        step = store.get(0)
        assert step["decision"] == "event_choice"
        assert step["state"]["floor"] == 3
        assert [o.get("title") for o in step["state"]["options"]] == ["Decipher", "Smash"]
        # The human action attached to the decision step it answered.
        assert step["chosen_action"] is not None
        assert step["chosen_action"].get("label") == "Smash"

    def test_live_context_key_never_collides_with_card_select_context(self, tmp_path: Path) -> None:
        # `context` is the card_select PURPOSE STRING in native workbench
        # logs; the live mirror's envelope must ride under `live_context`
        # (a dict under `context` crashed /api/step — shipped bug).
        store = TimelineStore.load_jsonl(_run_log(tmp_path))
        state = store.get(0)["state"]
        assert "context" not in state
        assert state["live_context"]["floor"] == 3

    def test_duplicate_board_is_not_logged_twice(self, tmp_path: Path) -> None:
        wlog = WorkbenchRunLogger(str(tmp_path))
        assert wlog.log_decision(EVENT_MSG, "event") is not None
        assert wlog.log_decision(EVENT_MSG, "event") is None
        wlog.close()

    def test_disabled_logger_writes_nothing(self, tmp_path: Path) -> None:
        wlog = WorkbenchRunLogger(None)
        assert wlog.log_decision(EVENT_MSG, "event") is None
        assert wlog.path is None


class TestCombatLog:
    """The game's own combat-log rows ride along on the state record.

    They are stored RAW on purpose: the advisor has no localization, so ids
    stay ids until the workbench server renders them. What is pinned here is
    that they survive the write/read round trip with the character id the
    renderer needs.
    """

    def test_rows_and_character_survive_the_round_trip(self, tmp_path: Path) -> None:
        rows = [{"kind": "damage", "text": "IRONCLAD dealt 6 damage to NIBBIT"}]
        wlog = WorkbenchRunLogger(str(tmp_path))
        assert wlog.log_decision({**EVENT_MSG, "combat_log": rows}, "event") is not None
        assert wlog.path is not None

        state = TimelineStore.load_jsonl(Path(wlog.path)).get(0)["state"]
        assert state["combat_log_raw"] == rows
        assert state["character_id"] == "IRONCLAD"

    def test_a_decision_with_no_rows_adds_no_key(self, tmp_path: Path) -> None:
        # An older mod sends no `combat_log`; don't write an empty key into
        # every state of every run for it.
        wlog = WorkbenchRunLogger(str(tmp_path))
        assert wlog.log_decision(EVENT_MSG, "event") is not None
        assert wlog.path is not None
        state = TimelineStore.load_jsonl(Path(wlog.path)).get(0)["state"]
        assert "combat_log_raw" not in state


class TestActName:
    """The act's NAME reaches the replayed step, not just its index.

    `pickScene` keys every painted region off `meta.act_name` and falls back to
    the act-1 Overgrowth room without it — which is why a replayed live run used
    to paint Overgrowth for all three acts (act 1 ships two models, GH #544, so
    the index alone cannot name the room). The mod supplies it through
    `ApplyEnvelope`; this pins that the logger carries it into the step.
    """

    def test_act_name_survives_the_round_trip(self, tmp_path: Path) -> None:
        wlog = WorkbenchRunLogger(str(tmp_path))
        assert wlog.log_decision({**EVENT_MSG, "act_name": "Underdocks"}, "event") is not None
        assert wlog.path is not None
        step = TimelineStore.load_jsonl(Path(wlog.path)).get(0)
        assert step["state"]["act_name"] == "Underdocks"
        # `meta` is what the frontend reads.
        assert step["meta"]["act_name"] == "Underdocks"

    def test_a_mod_that_sends_no_act_name_yields_none(self, tmp_path: Path) -> None:
        # Pre-fix mod builds send no `act_name`; it must read as absent rather
        # than as the empty string, which is not a room key either way but would
        # make the fallback look like a deliberate choice.
        wlog = WorkbenchRunLogger(str(tmp_path))
        assert wlog.log_decision(EVENT_MSG, "event") is not None
        assert wlog.path is not None
        assert TimelineStore.load_jsonl(Path(wlog.path)).get(0)["state"]["act_name"] is None


class TestEnemyKind:
    """The monster's canonical kind rides along on a combat step.

    Art resolution needs it and the display name cannot give it: the three
    Decimillipede segments all localize to "Decimillipede". It is grafted on the
    RENDER side (`_decorate_combat_for_render`), never into the dict the search
    reconstructs the fight from.
    """

    @staticmethod
    def _combat_msg() -> dict[str, Any]:
        return {
            "type": "advice_request",
            "advice_for": "combat_action",
            "seed": "S1",
            "floor": 24,
            "act": 2,
            "act_name": "Hive",
            "state_id": 11,
            "round": 1,
            "energy": 3,
            "max_energy": 3,
            "run_state": EVENT_MSG["run_state"],
            "player": {"hp": 60, "max_hp": 75, "block": 0, "powers": []},
            "hand": [],
            "draw_pile": {"cards": []},
            "discard_pile": {"cards": []},
            "exhaust_pile": {"cards": []},
            "play_pile": {"cards": []},
            "enemies": [
                {
                    "id": "DECIMILLIPEDE_SEGMENT_FRONT",
                    "name": "Decimillipede",
                    "hp": 42,
                    "max_hp": 42,
                    "block": 0,
                    "is_alive": True,
                    "powers": [],
                    "intents": [],
                },
                {
                    "id": "DECIMILLIPEDE_SEGMENT_MIDDLE",
                    "name": "Decimillipede",
                    "hp": 46,
                    "max_hp": 46,
                    "block": 0,
                    "is_alive": True,
                    "powers": [],
                    "intents": [],
                },
            ],
        }

    def test_kind_is_grafted_per_enemy(self, tmp_path: Path) -> None:
        wlog = WorkbenchRunLogger(str(tmp_path))
        assert wlog.log_decision(self._combat_msg(), "combat_action") is not None
        assert wlog.path is not None
        state = TimelineStore.load_jsonl(Path(wlog.path)).get(0)["state"]
        assert [e["kind"] for e in state["enemies"]] == [
            "DECIMILLIPEDE_SEGMENT_FRONT",
            "DECIMILLIPEDE_SEGMENT_MIDDLE",
        ]

    def test_an_enemy_the_mod_sends_no_id_for_adds_no_key(self, tmp_path: Path) -> None:
        msg = self._combat_msg()
        for enemy in msg["enemies"]:
            enemy.pop("id")
        wlog = WorkbenchRunLogger(str(tmp_path))
        assert wlog.log_decision(msg, "combat_action") is not None
        assert wlog.path is not None
        state = TimelineStore.load_jsonl(Path(wlog.path)).get(0)["state"]
        assert all("kind" not in e for e in state["enemies"])


def _combat_board() -> dict[str, Any]:
    import json

    path = Path(__file__).parent / "fixtures" / "realgame_adapter" / "single_enemy_ironclad.json"
    msg = json.loads(path.read_text(encoding="utf-8"))["state"]
    msg.pop("game_version", None)
    msg.update({"type": "advice_request", "advice_for": "combat_action"})
    msg["potions"] = [
        {"slot": 0, "empty": True},
        {"slot": 1, "id": "FIRE_POTION", "name": "Fire Potion", "empty": False},
    ]
    return msg


class TestHumanActionArgs:
    """The run log carries the action's machine form, not just its label --
    the coach (and the workbench's follow-mode grading) needs the move."""

    def _logged(self, tmp_path: Path, msg: dict[str, Any], human: dict[str, Any]) -> Any:
        wlog = WorkbenchRunLogger(str(tmp_path))
        assert wlog.log_decision(msg, msg["advice_for"]) is not None
        assert wlog.path is not None
        wlog.log_human_action(human)
        return TimelineStore.load_jsonl(Path(wlog.path)).get(0)["chosen_action"]

    def test_log_human_action_records_play_args(self, tmp_path: Path) -> None:
        human = {
            "type": "human_action",
            "decision_type": "combat_action",
            "action": "play",
            "detail": "Bash → Shrinker Beetle",
            "card": {"id": "BASH", "upgraded": False},
            "target_index": 0,
            "advice_request_id": "a1",
            "position_id": 7,
            "peeked": True,
        }
        chosen = self._logged(tmp_path, _combat_board(), human)
        # Bash is the fifth card in the logged hand -- found by identity.
        assert chosen["args"] == {"action": "play", "card_index": 4, "target_index": 0}
        assert chosen["peeked"] is True
        assert chosen["position_id"] == 7

    def test_log_human_action_records_end_turn_and_potion_args(self, tmp_path: Path) -> None:
        end = {"decision_type": "combat_action", "action": "end_turn", "detail": "End turn"}
        assert self._logged(tmp_path / "a", _combat_board(), end)["args"] == {"action": "end_turn"}
        drink = {
            "decision_type": "combat_action",
            "action": "potion",
            "detail": "Fire Potion",
            "potion": {"id": "FIRE_POTION"},
            "target_index": 0,
        }
        # The engine keeps the belt dense: the only potion is engine slot 0.
        assert self._logged(tmp_path / "b", _combat_board(), drink)["args"] == {
            "action": "potion",
            "slot": 0,
            "target_index": 0,
        }

    def test_an_unresolvable_card_logs_no_args(self, tmp_path: Path) -> None:
        human = {
            "decision_type": "combat_action",
            "action": "play",
            "card": {"id": "WHIRLWIND", "upgraded": False},
        }
        assert self._logged(tmp_path, _combat_board(), human)["args"] == {}

    def test_log_human_action_records_choice_index(self, tmp_path: Path) -> None:
        human = {
            "decision_type": "event",
            "action": "choose",
            "detail": "Smash",
            "choice": {"index": 1, "label": "Smash"},
            "advice_request_id": "e1",
            "position_id": 7,
            "peeked": False,
        }
        chosen = self._logged(tmp_path, EVENT_MSG, human)
        assert chosen["args"] == {"index": 1}
        assert chosen["position_id"] == 7 and chosen["peeked"] is False

    def test_a_skip_logs_a_null_index(self, tmp_path: Path) -> None:
        human = {
            "decision_type": "event",
            "action": "skip",
            "choice": {"index": None, "label": "Skip"},
        }
        assert self._logged(tmp_path, EVENT_MSG, human)["args"] == {"index": None}


class TestRootOutcomes:
    def test_root_outcomes_carry_final_hp(self, tmp_path: Path) -> None:
        import json

        from realgame_workbench_log import _root_outcomes

        advice = {
            "options": [
                {
                    "stable_idx": 4,
                    "label": "Bash",
                    "target_index": 0,
                    "p_win": 0.9,
                    "q": 0.8,
                    "visits": 500,
                    "wins": 450,
                    "losses": 50,
                    "ties": 0,
                    "is_best": True,
                    "mean_final_hp": 61.5,
                    "hp_n": 420,
                }
            ]
        }
        (row,) = _root_outcomes(advice)
        assert row["win_prob"] == pytest.approx(0.9)
        assert row["mean_final_hp"] == pytest.approx(61.5)
        assert row["hp_n"] == 420
        json.dumps(row)  # still a plain record


class TestLogPosition:
    """Coach records link to their decision by its replay step index."""

    def test_log_decision_returns_replay_index(self, tmp_path: Path) -> None:
        wlog = WorkbenchRunLogger(str(tmp_path))
        assert wlog.log_decision(EVENT_MSG, "event") == 0
        wlog.log_human_action({"detail": "Smash", "action": "choose"})
        assert wlog.log_decision(EVENT_MSG, "event") is None  # duplicate board
        wlog.log_coach({"kind": "choice", "id": "choice:1"})
        assert wlog.log_decision({**EVENT_MSG, "floor": 4}, "event") == 1
        wlog.log_advice({**EVENT_MSG, "floor": 4}, {"options": []}, 0.1)
        assert wlog.log_decision({**EVENT_MSG, "floor": 5}, "event") == 2
        assert wlog.path is not None
        store = TimelineStore.load_jsonl(Path(wlog.path))
        assert [store.get(i)["state"]["floor"] for i in range(3)] == [3, 4, 5]

    def test_log_name_is_the_served_name_and_survives_close(self, tmp_path: Path) -> None:
        from realgame_workbench_log import run_log_name

        assert run_log_name(str(tmp_path / "a b.jsonl")) == "a_b.jsonl"
        assert run_log_name(None) is None
        wlog = WorkbenchRunLogger(str(tmp_path))
        assert wlog.log_name is None
        wlog.log_decision(EVENT_MSG, "event")
        open_name = wlog.log_name
        assert open_name == Path(wlog.path or "").name
        wlog.close()
        assert wlog.path is None and wlog.log_name == open_name

    def test_autopsy_written_after_log_closed(self, tmp_path: Path) -> None:
        import json

        path = _run_log(tmp_path)  # ends with log_run_end: the file is closed
        wlog = WorkbenchRunLogger(str(tmp_path))
        wlog.log_decision(EVENT_MSG, "event")
        closed = wlog.path
        wlog.log_run_end({"type": "game_over", "victory": False, **EVENT_MSG})
        assert wlog.path is None
        wlog.log_coach({"kind": "autopsy", "id": "autopsy:S1", "items": []})
        wlog.log_coach({"kind": "turn", "id": "turn:9"})  # late grades are not
        lines = [json.loads(x) for x in Path(closed or "").read_text("utf-8").splitlines()]
        coach = [x["data"] for x in lines if x["type"] == "coach"]
        assert coach == [{"kind": "autopsy", "id": "autopsy:S1", "items": []}]
        assert len(TimelineStore.load_jsonl(Path(closed or ""))) == 2  # decision + terminal
        assert path.exists()

    def test_index_restarts_with_a_new_run_file(self, tmp_path: Path) -> None:
        wlog = WorkbenchRunLogger(str(tmp_path))
        assert wlog.log_decision(EVENT_MSG, "event") == 0
        assert wlog.log_decision({**EVENT_MSG, "seed": "S2"}, "event") == 0
        wlog.close()


def test_log_round_trip_keeps_hp_stats_and_coach(tmp_path: Path) -> None:
    """#850: what the live advisor writes, the workbench loader reads back."""
    wlog = WorkbenchRunLogger(str(tmp_path))
    assert wlog.log_decision(EVENT_MSG, "event") == 0
    advice = {
        "options": [
            {
                "stable_idx": 0,
                "label": "Decipher",
                "p_win": 0.7,
                "visits": 300,
                "is_best": True,
                "mean_final_hp": 55.0,
                "hp_n": 280,
            },
            {
                "stable_idx": 1,
                "label": "Smash",
                "p_win": 0.6,
                "visits": 100,
                "mean_final_hp": 48.5,
                "hp_n": 90,
            },
        ]
    }
    wlog.log_advice(EVENT_MSG, advice, 0.2)
    wlog.log_human_action({"detail": "Smash", "action": "choose"})
    assert wlog.log_decision({**EVENT_MSG, "floor": 4}, "event") == 1
    wlog.log_coach({"kind": "choice", "id": "choice:1", "verdict": "disagree", "log_idx": 0})
    path = wlog.path
    wlog.close()
    store = TimelineStore.load_jsonl(Path(path or ""))
    scores = store.get(0)["scores"]
    assert [s["mean_final_hp"] for s in scores] == pytest.approx([55.0, 48.5])
    assert [s["hp_n"] for s in scores] == [280, 90]
    assert store.coach()["by_step"] == {"0": ["choice:1"]}
