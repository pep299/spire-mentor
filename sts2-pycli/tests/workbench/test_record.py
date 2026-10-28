"""Tests for the MCTS run recorder (``workbench.record``).

Three layers of coverage:

* ``test_record_fixture_emits_mcts_records`` — drives the recorder over a
  real ``.run-source`` Vantom-boss fixture from the scraped corpus, when that
  corpus is present (skipped otherwise).
* ``test_record_fixture_tiny_inrepo`` — writes a tiny schema=2 fixture to
  ``tmp_path`` so the native loop always has coverage, even without the corpus.
* the unit-level classes below — the pure mapping/record helpers plus the
  best-effort full-run recorder driven against a scripted fake ``RustGame``,
  so no corpus (and no real run-sim) is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sts2_rs", reason="recorder requires the sts2_rs extension")

from sts2_hint.recommender import RecommendConfig
from workbench import record as record_mod
from workbench.record import record_fixture

VANTOM = Path("/tmp/run-fixtures-full/the-ironclad")

# Guard on the FIXTURE, not on the directory. This is a shared /tmp path that
# other work also harvests into, so the directory can exist while holding a
# different corpus shape (`onmanifold_*.toml` and no `*vantom_boss.toml`).
# `VANTOM.exists()` then passes, the glob comes back empty, and `next()` raised
# StopIteration -- a hard failure where a skip was meant.
_VANTOM_FIXTURES = sorted(VANTOM.glob("*vantom_boss.toml")) if VANTOM.is_dir() else []


# Minimal but real schema=2 fixture: Ironclad starter-ish deck vs Vantom.
_TINY_FIXTURE = """\
schema = 2

[provenance]
encounter = "ENCOUNTER.VANTOM_BOSS"
room_type = "boss"

[player]
class = "Ironclad"
hp = 60
max_hp = 70
deck = ["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "DEFEND_IRONCLAD", "BASH"]
relics = ["BURNING_BLOOD"]
potions = []

[[monsters]]
kind = "VANTOM"
"""


def _assert_log_shape(log_path: str) -> None:
    with open(log_path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    types = [e["type"] for e in lines]
    assert types[0] == "meta"
    assert "mcts" in types
    assert any(e["type"] == "meta" and e["data"].get("summary") for e in lines)

    mrec = next(e for e in lines if e["type"] == "mcts")["data"]
    assert mrec["root_outcomes"], "root_outcomes must not be empty"
    r0 = mrec["root_outcomes"][0]
    for k in ("visits", "q", "wins", "losses", "ties", "action_label"):
        assert k in r0, f"root_outcome missing {k!r}"

    # Every logged combat state must carry the report/aggregate fields.
    states = [e["data"] for e in lines if e["type"] == "state"]
    for st in states:
        assert "player" in st and "hp" in st["player"]
        assert "enemies" in st
        assert "hand" in st

    # Guard the combat-start fix (start_combat + start_player_turn in the
    # `combat_state_from_scenario_toml` binding): without it the loaded state
    # sits at round 0 with an empty hand and only ``end_turn`` legal, so every
    # logged state would have an empty hand and every MCTS root_outcome would
    # be "end turn". Assert the combat actually started and drew a hand, and
    # that at least one decision had a real card play to choose from.
    assert any(st["hand"] for st in states), (
        "no logged state drew a hand — combat never started "
        "(start_combat/start_player_turn missing from the scenario loader?)"
    )
    mcts_records = [e["data"] for e in lines if e["type"] == "mcts"]
    assert any(
        any(o["action_label"] != "end turn" for o in m["root_outcomes"]) for m in mcts_records
    ), "every MCTS root_outcome was 'end turn' — combat never offered a card play"


def test_record_fixture_tiny_inrepo(tmp_path):
    """Always-on coverage: drive the recorder against a tiny in-repo fixture."""
    fx = tmp_path / "tiny_vantom_boss.toml"
    fx.write_text(_TINY_FIXTURE)
    log_path = record_fixture(
        str(fx), out_dir=str(tmp_path / "logs"), sims=12, seed=1, max_steps=40
    )
    _assert_log_shape(log_path)


@pytest.mark.skipif(not _VANTOM_FIXTURES, reason="run-fixtures corpus not present")
def test_record_fixture_emits_mcts_records(tmp_path):
    fx = _VANTOM_FIXTURES[0]
    log_path = record_fixture(str(fx), out_dir=str(tmp_path), sims=20, seed=1, max_steps=200)
    with open(log_path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    types = [e["type"] for e in lines]
    assert types[0] == "meta"
    assert "mcts" in types
    assert any(e["type"] == "meta" and e["data"].get("summary") for e in lines)
    mrec = next(e for e in lines if e["type"] == "mcts")["data"]
    r0 = mrec["root_outcomes"][0]
    for k in ("visits", "q", "wins", "losses", "ties", "action_label"):
        assert k in r0


# ── unit: action-dict → (stable_idx, label, target) ───────────────────────
_HAND = [
    {"index": 0, "id": "CARD.STRIKE_IRONCLAD", "name": "Strike"},
    {"index": 1, "id": "CARD.BASH"},  # no display name -> falls back to id
]


class TestStableIdxAndLabel:
    def test_play_uses_hand_name_and_index(self):
        idx, label, target = record_mod._stable_idx_and_label(
            {"action": "play", "card_index": 0, "target_index": 2}, _HAND
        )
        assert (idx, label, target) == (0, "Strike", 2)

    def test_play_falls_back_to_card_id_then_question_mark(self):
        _, label, target = record_mod._stable_idx_and_label(
            {"action": "play", "card_index": 1}, _HAND
        )
        assert (label, target) == ("CARD.BASH", None)
        idx, label, _ = record_mod._stable_idx_and_label({"action": "play", "card_index": 7}, _HAND)
        assert (idx, label) == (7, "?")

    def test_end_turn_is_the_minus_one_sentinel(self):
        assert record_mod._stable_idx_and_label({"action": "end_turn", "target_index": 3}, []) == (
            -1,
            "end turn",
            None,
        )

    @pytest.mark.parametrize("kind", ["potion", "drink_potion"])
    def test_potion_slot_maps_below_the_potion_base(self, kind):
        idx, label, target = record_mod._stable_idx_and_label(
            {"action": kind, "slot": 2, "target_index": 0}, _HAND
        )
        assert (idx, label, target) == (-102, "drink potion slot 2", 0)

    def test_unknown_kind_gets_a_deterministic_sentinel(self):
        assert record_mod._stable_idx_and_label({"action": "channel"}, _HAND) == (
            -999,
            "channel",
            None,
        )


# ── unit: the mcts record shared with report/aggregate ────────────────────
def _cfg(sims: int = 64) -> RecommendConfig:
    """The real config object, not a stand-in: `_mcts_record` logs the RESOLVED
    exploration constant (`resolved_c_puct`), which only a `RecommendConfig`
    knows how to derive from the mode + budget."""
    return RecommendConfig(sims=sims, c_puct=1.0, depth=200, seed=7)


class TestMctsRecord:
    def test_best_entry_drives_the_recommendation(self):
        result = {
            "actions": [
                {
                    "action": {"action": "end_turn"},
                    "visits": 90,
                    "q": 0.1,
                    "wins": 3,
                    "losses": 4,
                    "ties": 1,
                },
                {
                    "action": {"action": "play", "card_index": 0},
                    "visits": 10,
                    "value": 0.9,
                    "is_best": True,
                },
            ],
            "pv": [{"action": "play", "card_index": 0}, {"action": "end_turn"}],
        }
        rec, action = record_mod._mcts_record(result, {"hand": _HAND}, _cfg(), 4)

        assert action == {"action": "play", "card_index": 0}
        assert rec["decision_idx"] == 4
        assert rec["config"] == {"sims": 64, "c_puct": 1.0, "depth": 200, "seed": 7}
        assert rec["total_visits"] == 100
        assert rec["recommended_stable_idx"] == 0
        assert rec["recommended_label"] == "Strike"
        assert rec["pv"] == ["Strike", "end turn"]
        # ``q`` falls back to ``value`` when the engine omits it.
        assert [o["q"] for o in rec["root_outcomes"]] == [0.1, 0.9]
        assert [o["is_best"] for o in rec["root_outcomes"]] == [False, True]
        assert rec["root_outcomes"][0]["wins"] == 3

    def test_falls_back_to_the_most_visited_child_without_is_best(self):
        result = {
            "actions": [
                {"action": {"action": "play", "card_index": 0}, "visits": 5},
                {"action": {"action": "potion", "slot": 1}, "visits": 42},
            ]
        }
        rec, action = record_mod._mcts_record(result, {"hand": _HAND}, _cfg(), 0)
        assert action == {"action": "potion", "slot": 1}
        assert rec["recommended_stable_idx"] == -101
        assert rec["recommended_label"] == "drink potion slot 1"

    def test_empty_search_result_recommends_end_turn(self):
        rec, action = record_mod._mcts_record({}, {}, _cfg(), 0)
        assert action == {"action": "end_turn"}
        assert rec["recommended_stable_idx"] == -1
        assert rec["recommended_label"] == "end turn"
        assert rec["total_visits"] == 0
        assert rec["root_outcomes"] == []


# ── unit: fixture provenance ──────────────────────────────────────────────
class TestFixtureProvenance:
    def test_reads_provenance_and_cased_character(self, tmp_path):
        fx = tmp_path / "vantom_boss.toml"
        fx.write_text(
            'schema = 2\n\n[provenance]\nbuild_id = "v0.107.1"\n'
            'encounter = "ENCOUNTER.VANTOM_BOSS"\n\n[player]\nclass = "Silent"\n'
        )
        assert record_mod._fixture_provenance(str(fx)) == {
            "fixture_stem": "vantom_boss",
            "build_id": "v0.107.1",
            "encounter": "ENCOUNTER.VANTOM_BOSS",
            "character": "Silent",
        }

    def test_defaults_when_sections_are_absent(self, tmp_path):
        fx = tmp_path / "bare.toml"
        fx.write_text("schema = 2\n")
        assert record_mod._fixture_provenance(str(fx)) == {
            "fixture_stem": "bare",
            "build_id": None,
            "encounter": None,
            "character": "Ironclad",
        }

    def test_unreadable_fixture_still_yields_a_stem(self, tmp_path):
        missing = tmp_path / "gone.toml"
        assert record_mod._fixture_provenance(str(missing)) == {
            "fixture_stem": "gone",
            "build_id": None,
            "encounter": None,
            "character": "Ironclad",
        }

    def test_malformed_toml_is_swallowed(self, tmp_path):
        fx = tmp_path / "broken.toml"
        fx.write_text("this is = not [valid toml\n")
        assert record_mod._fixture_provenance(str(fx))["character"] == "Ironclad"


# ── unit: loc loading ─────────────────────────────────────────────────────
class TestLoadCardLoc:
    def test_merges_relics_and_potions_and_tolerates_bad_json(self, tmp_path, monkeypatch):
        loc_dir = tmp_path / "localization_eng"
        loc_dir.mkdir()
        (loc_dir / "cards.json").write_text(json.dumps({"STRIKE_IRONCLAD.title": "Strike"}))
        (loc_dir / "relics.json").write_text("{ not json at all")  # must not raise
        (loc_dir / "potions.json").write_text(json.dumps({"FIRE_POTION.title": "Fire Potion"}))
        monkeypatch.setattr(record_mod, "_REPO_ROOT", tmp_path)

        loc = record_mod._load_card_loc()
        assert loc["STRIKE_IRONCLAD.title"] == "Strike"
        assert loc["FIRE_POTION.title"] == "Fire Potion"

    def test_missing_optional_files_are_skipped(self, tmp_path, monkeypatch):
        loc_dir = tmp_path / "localization_eng"
        loc_dir.mkdir()
        (loc_dir / "cards.json").write_text(json.dumps({"BASH.title": "Bash"}))
        monkeypatch.setattr(record_mod, "_REPO_ROOT", tmp_path)
        assert record_mod._load_card_loc() == {"BASH.title": "Bash"}


# ── unit: native action → RustGame verb ───────────────────────────────────
class TestNativeToRunVerb:
    def test_play_card(self):
        assert record_mod._native_to_run_verb(
            {"action": "play", "card_index": 3, "target_index": 1}
        ) == ("play_card", {"card_index": 3, "target_index": 1})

    def test_play_card_without_target(self):
        verb, args = record_mod._native_to_run_verb({"action": "play", "card_index": 0})
        assert (verb, args) == ("play_card", {"card_index": 0, "target_index": None})

    @pytest.mark.parametrize("kind", ["potion", "drink_potion"])
    def test_potion_slot_becomes_potion_index(self, kind):
        assert record_mod._native_to_run_verb({"action": kind, "slot": 2, "target_index": 0}) == (
            "use_potion",
            {"potion_index": 2, "target_index": 0},
        )

    @pytest.mark.parametrize("action", [{"action": "end_turn"}, {"action": "channel"}, {}])
    def test_everything_else_ends_the_turn(self, action):
        assert record_mod._native_to_run_verb(action) == ("end_turn", {})


# ── unit: default non-combat run action ───────────────────────────────────
class TestDefaultRunAction:
    def _act(self, state):
        return record_mod._default_run_action(state, game=object())

    def test_map_select_takes_the_first_choice_column(self):
        assert self._act({"decision": "map_select", "choices": [{"col": 4}, {"col": 5}]}) == (
            "select_map_node",
            {"col": 4},
        )

    def test_map_select_without_choices_falls_through_to_proceed(self):
        assert self._act({"decision": "map_select", "choices": []}) == ("proceed", {})

    def test_card_reward_is_skipped(self):
        assert self._act({"decision": "card_reward"}) == ("skip_card_reward", {})

    def test_bundle_select_takes_the_first_bundle(self):
        assert self._act({"decision": "bundle_select"}) == ("select_bundle", {"bundle_index": 0})

    def test_card_select_prefers_skipping(self):
        state = {"decision": "card_select", "can_skip": True, "cards": [{"index": 2}]}
        assert self._act(state) == ("skip_select", {})

    def test_card_select_picks_the_first_card_as_a_string_index(self):
        state = {"decision": "card_select", "can_skip": False, "cards": [{"index": 2}]}
        assert self._act(state) == ("select_cards", {"indices": "2"})

    def test_card_select_without_cards_falls_through_to_proceed(self):
        assert self._act({"decision": "card_select", "cards": []}) == ("proceed", {})

    def test_event_choice_takes_the_first_option(self):
        state = {"decision": "event_choice", "options": [{"index": 1}, {"index": 2}]}
        assert self._act(state) == ("choose_option", {"option_index": 1})

    def test_event_choice_without_options_falls_through_to_proceed(self):
        assert self._act({"decision": "event_choice"}) == ("proceed", {})

    def test_shop_always_leaves_the_room(self):
        assert self._act({"decision": "shop", "options": [{"index": 0}]}) == ("leave_room", {})

    def test_rest_site_takes_the_first_option(self):
        assert self._act({"decision": "rest_site", "options": [{"index": 3}]}) == (
            "choose_option",
            {"option_index": 3},
        )

    def test_rest_site_without_options_falls_through_to_proceed(self):
        assert self._act({"decision": "rest_site", "options": []}) == ("proceed", {})

    def test_unknown_decision_proceeds(self):
        assert self._act({"decision": "combat_reward"}) == ("proceed", {})


# ── full-run recorder against a scripted fake RustGame ────────────────────
class FakeRustGame:
    """Scripted ``RustGame`` stand-in: replays ``states`` one ``act`` at a time."""

    def __init__(self, states=None):
        self.states = list(states or [])
        self.calls: list[tuple[str, dict]] = []
        self.start_kwargs: dict = {}
        self._i = 0

    def start(self, **kwargs):
        self.start_kwargs = kwargs
        return self.states[0]

    def act(self, verb, **args):
        self.calls.append((verb, args))
        self._i += 1
        return self.states[min(self._i, len(self.states) - 1)]

    def combat_state(self):
        state = self.states[min(self._i, len(self.states) - 1)]
        return "NATIVE_CS" if state.get("decision") == "combat_play" else None


_FAKE_COMBAT_VIEW = {
    "decision": "combat_play",
    "round": 2,
    "player": {"hp": 44, "max_hp": 70},
    "enemies": [{"index": 0, "hp": 12}],
    "hand": _HAND,
}


@pytest.fixture
def stub_recorder(monkeypatch):
    """Patch out the native seams of :func:`record_run` (loc / view / MCTS)."""
    seen_chars: list[str] = []

    def fake_view(cs, character, loc):
        seen_chars.append(character)
        return dict(_FAKE_COMBAT_VIEW)

    monkeypatch.setattr(record_mod, "_load_card_loc", lambda: {})
    monkeypatch.setattr(record_mod, "_combat_state_dict", fake_view)
    monkeypatch.setattr(
        record_mod,
        "_mcts_recommend",
        lambda cs, cfg: {
            "actions": [
                {
                    "action": {"action": "play", "card_index": 0, "target_index": 0},
                    "visits": 9,
                    "q": 0.7,
                    "is_best": True,
                }
            ],
            "pv": [],
        },
    )
    return SimpleNamespace(seen_chars=seen_chars)


def _read_log(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _install_game(monkeypatch, states):
    import workbench.backends.rust_game as rg

    game = FakeRustGame(states)
    monkeypatch.setattr(rg, "RustGame", lambda: game)
    return game


class TestRecordRun:
    def test_combat_decisions_are_searched_and_the_run_summarized(
        self, tmp_path, monkeypatch, stub_recorder
    ):
        game = _install_game(
            monkeypatch,
            [
                {"decision": "map_select", "choices": [{"col": 1}]},
                {"decision": "combat_play"},
                {"decision": "game_over", "victory": True, "round": 9, "player": {"hp": 30}},
            ],
        )

        path = record_mod.record_run(
            "ironclad", "SEED1", out_dir=str(tmp_path), sims=5, max_steps=10
        )
        lines = _read_log(path)

        assert lines[0]["data"]["seed"] == "SEED1"
        assert lines[1]["data"]["fixture_stem"] == "ironclad-SEED1"
        assert lines[1]["data"]["sims"] == 5
        assert game.start_kwargs == {"character": "ironclad", "seed": "SEED1", "ascension": 0}
        # the lower-cased engine name is display-cased before it reaches the view
        assert stub_recorder.seen_chars == ["Ironclad"]
        # map node first, then the MCTS-recommended card play
        assert game.calls == [
            ("select_map_node", {"col": 1}),
            ("play_card", {"card_index": 0, "target_index": 0}),
        ]

        mcts = [e["data"] for e in lines if e["type"] == "mcts"]
        assert len(mcts) == 1
        assert mcts[0]["decision_idx"] == 0
        assert mcts[0]["recommended_label"] == "Strike"

        summary = next(e["data"] for e in lines if e["type"] == "meta" and e["data"].get("summary"))
        assert summary["final_won"] is True
        assert summary["final_turn"] == 9
        assert summary["final_player_hp"] == 30

    def test_cased_character_is_passed_through_unchanged(
        self, tmp_path, monkeypatch, stub_recorder
    ):
        _install_game(monkeypatch, [{"decision": "combat_play"}, {"decision": "game_over"}])
        record_mod.record_run("Necrobinder", "S", out_dir=str(tmp_path), max_steps=4)
        assert stub_recorder.seen_chars == ["Necrobinder"]

    def test_error_state_stops_the_best_effort_run(self, tmp_path, monkeypatch, stub_recorder):
        game = _install_game(
            monkeypatch,
            [
                {"decision": "shop", "round": 3, "player": {"hp": 51, "max_hp": 70}},
                {"type": "error", "message": "run-sim cannot advance"},
            ],
        )
        path = record_mod.record_run("Ironclad", "S", out_dir=str(tmp_path), max_steps=50)
        lines = _read_log(path)

        assert game.calls == [("leave_room", {})]
        summary = next(e["data"] for e in lines if e["type"] == "meta" and e["data"].get("summary"))
        # last_state stays at the pre-error state, so its HP is what gets logged
        assert summary == {
            "summary": True,
            "final_won": False,
            "final_turn": 3,
            "final_player_hp": 51,
            "final_player_max_hp": 70,
        }

    def test_combat_decision_without_a_native_state_falls_back_to_default(
        self, tmp_path, monkeypatch, stub_recorder
    ):
        import workbench.backends.rust_game as rg

        game = FakeRustGame([{"decision": "combat_play"}, {"decision": "game_over"}])
        game.combat_state = lambda: None
        monkeypatch.setattr(rg, "RustGame", lambda: game)

        path = record_mod.record_run("Ironclad", "S", out_dir=str(tmp_path), max_steps=6)
        lines = _read_log(path)

        assert game.calls == [("proceed", {})]
        assert not [e for e in lines if e["type"] == "mcts"]
        assert stub_recorder.seen_chars == []

    def test_max_steps_caps_the_loop(self, tmp_path, monkeypatch, stub_recorder):
        game = _install_game(monkeypatch, [{"decision": "rest_site", "options": [{"index": 0}]}])
        path = record_mod.record_run("Ironclad", "S", out_dir=str(tmp_path), max_steps=3)
        assert len(game.calls) == 3
        summary = next(
            e["data"] for e in _read_log(path) if e["type"] == "meta" and e["data"].get("summary")
        )
        assert summary["final_won"] is False


# ── fixture recorder: the max_steps escape hatch ──────────────────────────
def test_record_fixture_stops_at_max_steps(tmp_path):
    """A max_steps cut-off still writes a summary (the loop's ``else`` arm)."""
    fx = tmp_path / "tiny_vantom_boss.toml"
    fx.write_text(_TINY_FIXTURE)
    log_path = record_fixture(str(fx), out_dir=str(tmp_path / "logs"), sims=4, seed=1, max_steps=1)
    lines = _read_log(log_path)

    assert len([e for e in lines if e["type"] == "mcts"]) == 1
    summary = next(e["data"] for e in lines if e["type"] == "meta" and e["data"].get("summary"))
    assert summary["final_won"] is False  # nowhere near terminal after one ply
    assert summary["final_player_max_hp"] == 70


# ── CLI ───────────────────────────────────────────────────────────────────
class TestMain:
    def test_fixture_mode_prints_the_log_path(self, monkeypatch, capsys, tmp_path):
        seen: dict = {}

        def fake_record_fixture(fixture, **kwargs):
            seen.update({"fixture": fixture, **kwargs})
            return "/logs/out.jsonl"

        monkeypatch.setattr(record_mod, "record_fixture", fake_record_fixture)
        rc = record_mod.main(
            ["--fixture", "f.toml", "--out", str(tmp_path), "--sims", "11", "--mcts-seed", "5"]
        )

        assert rc == 0
        assert capsys.readouterr().out == "/logs/out.jsonl\n"
        assert seen == {"fixture": "f.toml", "out_dir": str(tmp_path), "sims": 11, "seed": 5}

    def test_run_mode_forwards_character_seed_and_ascension(self, monkeypatch, capsys, tmp_path):
        seen: dict = {}
        monkeypatch.setattr(
            record_mod,
            "record_run",
            lambda **kwargs: (seen.update(kwargs), "/logs/run.jsonl")[1],
        )
        rc = record_mod.main(
            [
                "--run",
                "--character",
                "silent",
                "--seed",
                "ABC",
                "--ascension",
                "4",
                "--out",
                str(tmp_path),
            ]
        )

        assert rc == 0
        assert capsys.readouterr().out == "/logs/run.jsonl\n"
        assert seen == {
            "character": "silent",
            "seed": "ABC",
            "ascension": 4,
            "out_dir": str(tmp_path),
            "sims": 3000,
            "mcts_seed": 1,
        }

    def test_a_source_is_required(self):
        with pytest.raises(SystemExit):
            record_mod.main([])
