import pytest
from workbench import reasoning


class _FakeTag:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_score_maps_tags_to_rows(monkeypatch, combat_state):
    fake = {
        0: _FakeTag(
            stable_idx=0,
            action_label=None,
            win_prob=0.71,
            q=0.42,
            visits=2600,
            is_best=True,
            q_saturated=False,
            low_confidence=False,
            target_index=0,
        ),
        -1: _FakeTag(
            stable_idx=-1,
            action_label=None,
            win_prob=0.31,
            q=-0.1,
            visits=400,
            is_best=False,
            q_saturated=False,
            low_confidence=True,
            target_index=None,
        ),
    }
    monkeypatch.setattr(reasoning, "_recommend", lambda state, cfg: fake)

    out = reasoning.score(combat_state, {"sims": 4000})
    assert out["pv"] is None
    rows = out["scores"]
    assert rows[0]["is_best"] is True  # best sorted first
    assert abs(rows[0]["win_prob"] - 0.71) < 1e-9
    assert rows[0]["action"] == "Strike"  # resolved from hand[0]
    assert rows[1]["action"] == "end turn"  # sentinel stable_idx -1
    assert out["config"]["sims"] == 4000


def test_action_label_resolves_potion_sentinel():
    # Potion sentinel -100 - slot resolves to the player's potion name.
    state = {"player": {"potions": [{"name": "Stable Serum"}, {"name": "Fire Potion"}]}, "hand": []}
    assert reasoning._action_label(-100, state) == "drink Stable Serum"
    assert reasoning._action_label(-101, state) == "drink Fire Potion"
    # Out-of-range slot degrades gracefully rather than raising.
    assert reasoning._action_label(-105, state) == "drink potion slot 5"


def test_score_degrades_on_error(monkeypatch, combat_state):
    def boom(state, cfg):
        raise RuntimeError("sts2_rs missing")

    monkeypatch.setattr(reasoning, "_recommend", boom)
    out = reasoning.score(combat_state, {})
    assert "error" in out
    assert out["scores"] == []


def test_search_tree_passes_through_and_echoes_config(monkeypatch, combat_state):
    fake_tree = {
        "nodes": [
            {"id": 0, "parent": None, "visits": 300, "q": 0.8, "children": [1]},
            {"id": 1, "parent": 0, "visits": 250, "q": 0.9, "children": [], "label": "Strike"},
        ],
        "root": 0,
        "pv": [{"action": "play", "card_index": 0}],
        "sims": 300,
        "c_puct": 1.4,
    }
    monkeypatch.setattr(reasoning, "_search_tree", lambda state, cfg: dict(fake_tree))
    out = reasoning.search_tree(combat_state, {"sims": 300})
    assert out["root"] == 0
    assert len(out["nodes"]) == 2
    assert out["config"]["sims"] == 300  # echoed back


def test_search_tree_degrades_on_error(monkeypatch, combat_state):
    def boom(state, cfg):
        raise RuntimeError("sts2_rs missing")

    monkeypatch.setattr(reasoning, "_search_tree", boom)
    out = reasoning.search_tree(combat_state, {})
    assert "error" in out
    assert out["nodes"] == []


def test_rows_from_mcts_carries_wlt():
    from workbench.reasoning import rows_from_mcts

    raw = {
        "actions": [
            {
                "action": {"action": "end_turn"},
                "visits": 10,
                "q": -1.0,
                "p_win": 0.0,
                "wins": 0,
                "losses": 10,
                "ties": 0,
                "is_best": True,
            },
        ]
    }
    rows = rows_from_mcts(raw, {"hand": []})
    assert rows[0]["wins"] == 0 and rows[0]["losses"] == 10 and rows[0]["ties"] == 0


def test_rows_from_mcts_crowns_the_engines_next_choice():
    """GH #1022: the engine's pick is a row this boundary does not surface (a
    channel); the star goes to its next choice by ``pick_rank``, and every row
    keeps its band flag."""
    from workbench.reasoning import rows_from_mcts

    raw = {
        "actions": [
            {
                "action": {"action": "channel"},
                "visits": 50,
                "p_win": 0.9,
                "is_best": True,
                "pick_rank": 0,
                "in_band": True,
            },
            {"action": {"action": "end_turn"}, "visits": 10, "p_win": 0.8, "pick_rank": 2},
            {
                "action": {"action": "play", "card_index": 0},
                "visits": 40,
                "p_win": 0.7,
                "pick_rank": 1,
                "in_band": True,
            },
        ]
    }
    rows = rows_from_mcts(raw, {"hand": [{"index": 0, "name": "Zap"}]})
    assert [(r["stable_idx"], r["is_best"], r["in_band"]) for r in rows] == [
        (0, True, True),
        (-1, False, False),
    ]
    assert [r["pick_rank"] for r in rows] == [1, 2]
    assert all(r["pick_reason"] is None for r in rows), "an heir has no reason"


def test_rows_from_mcts_passes_final_hp():
    from workbench.reasoning import rows_from_mcts

    raw = {
        "actions": [
            {
                "action": {"action": "end_turn"},
                "visits": 10,
                "p_win": 1.0,
                "mean_final_hp": 61.5,
                "hp_n": 30,
                "is_best": True,
            },
            {"action": {"action": "play", "card_index": 0}, "visits": 5, "p_win": 1.0},
        ]
    }
    rows = rows_from_mcts(raw, {"hand": [{"index": 0, "name": "Strike"}]})
    by = {r["stable_idx"]: r for r in rows}
    assert by[-1]["mean_final_hp"] == pytest.approx(61.5) and by[-1]["hp_n"] == 30
    assert by[0]["mean_final_hp"] is None and by[0]["hp_n"] == 0


def test_search_tree_native_forwards_honest_search_knobs(monkeypatch, combat_state):
    """The tree inspector must run the SAME search as the hint panel.

    ``mcts_search_tree`` had no ``hidden_info`` / ``draw_widening_c``
    parameter, so it silently fell back to ``MctsConfig::default()``'s leaky
    ``Clairvoyant`` while ``score_native`` beside it ran the deployed honest
    ``beliefdraw`` — two different searches over one position, with no UI knob
    able to reconcile them. Pin the forwarding, not just the call.
    """
    sts2_rs = pytest.importorskip("sts2_rs")
    seen: dict = {}

    def fake_tree(cs, **kw):
        seen.update(kw)
        return {"nodes": [], "root": 0, "pv": []}

    monkeypatch.setattr(sts2_rs, "mcts_search_tree", fake_tree)
    reasoning.search_tree_native(
        object(),
        {"sims": 300, "hidden_info": "beliefdraw", "draw_widening_c": 2.0},
    )
    assert seen["hidden_info"] == "beliefdraw"
    assert seen["draw_widening_c"] == pytest.approx(2.0)
