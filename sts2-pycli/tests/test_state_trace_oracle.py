"""The C# oracle's state-mutation trace (``trace`` / ``get_trace``, GH #1088)."""

from __future__ import annotations


def test_strike_records_attributed_damage_energy_and_pile_moves(game):
    state = game.start(seed="trace1")
    game.skip_neow(state)
    game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
    on = game.send({"cmd": "trace", "on": True})
    assert on["enabled"] is True

    before = game.get_state()
    seq = before["trace_seq"]
    strike = before["hand_ids"].index("CARD.STRIKE_IRONCLAD")
    game.act("play_card", card_index=strike, target_index=0)

    segs = game.send({"cmd": "get_trace", "since": seq})["segments"]
    assert [s["action"] for s in segs] == ["play_card"]
    events = segs[0]["events"]

    hp = [e for e in events if e["k"] == "hp" and e["who"] == "monsters[0]"]
    assert len(hp) == 1
    assert hp[0]["old"] - hp[0]["new"] == 6
    assert hp[0]["src"].endswith("StrikeIronclad.OnPlay")

    energy = [e for e in events if e["k"] == "energy"]
    assert [(e["old"], e["new"]) for e in energy] == [(3, 2)]

    moves = [(e["k"], e["pile"]) for e in events if e["k"].startswith("pile_")]
    assert moves == [
        ("pile_remove", "Hand"),
        ("pile_add", "Play"),
        ("pile_remove", "Play"),
        ("pile_add", "Discard"),
    ]
    assert game.get_state()["trace_seq"] == segs[0]["seq"]


def test_trace_is_off_by_default(game):
    state = game.start(seed="trace2")
    game.skip_neow(state)
    game.enter_room("combat", encounter="SHRINKER_BEETLE_WEAK")
    assert game.get_state()["trace_seq"] is None
