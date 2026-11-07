"""Every card select in an auto-play batch reaches the player (GH #1131).

``Catastrophe.OnPlay`` awaits ``CardCmd.AutoPlay`` for each card, and
the real game shows one upgrade screen per auto-played Armaments. The headless
used to answer the second screen itself with option 0.
"""

from __future__ import annotations


def _hand_catastrophe(view: dict) -> int | None:
    hand = view.get("hand") or []
    return next((i for i, c in enumerate(hand) if c["id"] == "CARD.CATASTROPHE"), None)


def test_catastrophe_surfaces_each_armaments_select(game):
    state = game.start(character="Ironclad", seed="CHAINED1131")
    game.skip_neow(state)
    # Everything outside Catastrophe is Armaments, so both auto-plays open a select.
    deck = ["CATASTROPHE", *["ARMAMENTS"] * 6]
    r = game.set_player(hp=70, max_hp=80, deck=deck, relics=[], potions=[])
    assert r.get("type") != "error", r
    view = game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
    if _hand_catastrophe(view) is None:
        # Which seed deals it differs per game build, so draw it next turn.
        assert game.set_draw_order(["CATASTROPHE"]).get("type") != "error"
        view = game.end_turn()
    index = _hand_catastrophe(view)
    assert index is not None, view.get("hand")

    first = game.play_card(index, None)
    assert first.get("decision") == "card_select", first
    second = game.select_cards("0")
    assert second.get("decision") == "card_select", second
    assert game.select_cards("0").get("decision") == "combat_play"
