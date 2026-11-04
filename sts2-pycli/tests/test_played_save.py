"""Tests for the opt-in ``played_save`` bootstrap on ``start_run``.

Context: real-game bridge parity gate (2026-07-11). A fresh headless save
(``played_save`` unset/False, the default) always opens act 1 on the Neow
blessing altar (``event_choice`` for "Neow") -- this matches a real
brand-new player's very first run, where NEOW_EPOCH has never been obtained.
A REAL save with many runs played can still be sitting on NEOW_EPOCH at
``EpochState.Obtained`` (earned) without ever being ``Revealed`` (the player
never clicked "Reveal" on the Timeline screen) -- and in that state, the game
skips the altar entirely and opens directly on ``map_select`` with a plain
10-card deck (no Neow relic/curse).

``played_save=True`` bootstraps exactly that minimal "obtained-not-revealed"
progress state (mirrors ``SaveManager.ResetTimelineProgress``'s own
``ObtainEpochOverride(NEOW_EPOCH, Obtained)`` call) before the run is created,
so headless reproduces it deterministically. See
``RunSimulator.BuildUnlockState``'s doc comment (C#) for the full mechanism
and the RNG-alignment pitfall it documents.

Verified against a real recorded trail: seed "12345" / Ironclad /
``played_save=True`` reproduces the live game's actual first map node
(col=3, row=0, Monster) and first combat (Twig Slime (S), Leaf Slime (M),
Leaf Slime (S), opening hand Strike/Defend/Defend/Strike/Strike) exactly --
see ``logs/realgame_parity/20260711T033316Z_seed12345/real.jsonl``.
"""

import pytest
import sts2_client


class TestPlayedSaveDefault:
    def test_default_still_opens_on_neow(self, game):
        """Byte-identical default: played_save unset must behave exactly like
        before this flag existed -- fresh save, Neow blessing altar first."""
        state = game.start(character="Ironclad", seed="12345")
        assert state["decision"] == "event_choice"
        assert state.get("event_name") == "Neow"


class TestPlayedSaveOptIn:
    def test_opens_directly_on_map_select(self, game):
        state = game.start(character="Ironclad", seed="12345", played_save=True)
        assert state["decision"] == "map_select"

    def test_deck_has_no_neow_relic_or_curse(self, game):
        """played_save must never have visited the Neow event -- no relic,
        no curse, deck stays at the character's plain starting size."""
        state = game.start(character="Ironclad", seed="12345", played_save=True)
        player = state["player"]
        assert player["deck_size"] == 10
        assert [r["id"] for r in player["relics"]] == ["BURNING_BLOOD"]

    @pytest.mark.skipif(
        sts2_client.GAME_VERSION != "v0.107.1",
        reason=(
            "pinned to a v0.107.1 real-game recording; v0.111.0 rehashes seeds "
            "(XxHash64), so seed 12345 walks another map, and no v0.111.0 "
            "real-game trail has been recorded"
        ),
    )
    def test_matches_recorded_real_trail_for_seed_12345(self, game):
        """Regression pin against the exact seed-12345 real-game reference
        trail (logs/realgame_parity/20260711T033316Z_seed12345/real.jsonl):
        first map node, first encounter, and opening hand must match exactly."""
        state = game.start(character="Ironclad", seed="12345", played_save=True)
        assert state["choices"] == [{"col": 3, "row": 0, "type": "Monster"}]

        combat = game.act("select_map_node", col=3, row=0)
        assert combat["decision"] == "combat_play"
        assert [e["name"] for e in combat["enemies"]] == [
            "Twig Slime (S)",
            "Leaf Slime (M)",
            "Leaf Slime (S)",
        ]
        assert [c["name"] for c in combat["hand"]] == [
            "Strike",
            "Defend",
            "Defend",
            "Strike",
            "Strike",
        ]
