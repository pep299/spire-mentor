"""Tests for the post-combat regret review (``workbench.regret_review``).

The pure scoring/aggregation core needs no game engine — it consumes the same
``scores`` rows the rest of the workbench already produces. The replay endpoint
test drives a recorded JSONL whose ``mcts`` records carry the per-action stats,
so it too runs without the Rust wheel. The live ``last_move_regret`` test uses
the shared ``FakeGame`` stub with ``reasoning.score`` monkeypatched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from workbench import regret_review
from workbench.server import create_app


# ── pure core ────────────────────────────────────────────────────────────────
def _score(stable_idx, action, win_prob, is_best, target=None, visits=100):
    return {
        "stable_idx": stable_idx,
        "action": action,
        "win_prob": win_prob,
        "is_best": is_best,
        "target_index": target,
        "visits": visits,
    }


def _combat_step(idx, hand, scores, played_args, turn=3):
    return {
        "idx": idx,
        "state": {"decision": "combat_play", "round": turn, "hand": hand},
        "scores": scores,
        "chosen_action": {"args": played_args},
        "meta": {"turn": turn},
    }


def _hp_step(best_kw, played_kw, **extra):
    best = {**_score(1, "Defend", 1.0, True, visits=900), "mean_final_hp": 70.0, "hp_n": 400}
    played = {**_score(-1, "end turn", 1.0, False, visits=120), "mean_final_hp": 58.0, "hp_n": 120}
    best.update(best_kw)
    played.update(played_kw)
    step = _combat_step(
        0,
        hand=[{"index": 1, "name": "Defend"}],
        scores=[played, best],
        played_args={"action": "end_turn"},
    )
    step.update(extra)
    return step


def test_saturated_fight_hp_gap_is_a_blunder():
    d = regret_review.assess_decision(_hp_step({}, {}))
    assert d.verdict == "blunder" and d.is_blunder
    assert d.hp_drop == 12
    assert d.message.endswith(", ≈12 HP")


def test_small_hp_gap_is_a_slip():
    d = regret_review.assess_decision(_hp_step({}, {"mean_final_hp": 65.0}))
    assert d.verdict == "slip" and not d.is_blunder
    assert d.hp_drop == 5


def test_within_noise_on_all_signals_is_a_tie():
    d = regret_review.assess_decision(
        _hp_step({"win_prob": 1.0}, {"win_prob": 0.98, "mean_final_hp": 68.0})
    )
    assert d.verdict == "tie" and not d.is_blunder


def test_low_visit_played_move_is_uncertain():
    d = regret_review.assess_decision(_hp_step({}, {"visits": 12}))
    assert d.verdict == "uncertain" and not d.is_blunder


def test_missing_hp_falls_back_to_win_prob():
    step = _hp_step(
        {"win_prob": 0.9, "mean_final_hp": None, "hp_n": 0},
        {"win_prob": 0.8, "mean_final_hp": None, "hp_n": 0},
    )
    d = regret_review.assess_decision(step)
    assert d.hp_drop is None
    assert d.verdict == "blunder"
    assert "HP" not in d.message


def test_hp_below_min_samples_ignored():
    d = regret_review.assess_decision(_hp_step({}, {"hp_n": 5}))
    assert d.hp_drop is None


def test_player_beats_ai_line_is_a_tie():
    d = regret_review.assess_decision(_hp_step({}, {"mean_final_hp": 75.0}))
    assert d.hp_drop == 0
    assert d.verdict == "tie"


def test_hp_gap_within_noise_reads_zero():
    # GH #888: two near-equal rows, 2 HP apart with overlapping spread
    # (SE of the difference 4 HP) -- rollout noise, not advice.
    noisy = {"mean_final_hp": 60.0, "hp_sd": 20.0, "hp_n": 50}
    d = regret_review.assess_decision(_hp_step(noisy, {**noisy, "mean_final_hp": 58.0}))
    assert d.hp_drop == 0
    assert d.verdict == "tie"
    assert "HP" not in d.message


def test_hp_gap_clearing_noise_counts():
    d = regret_review.assess_decision(_hp_step({"hp_sd": 5.0}, {"hp_sd": 5.0}))
    assert d.hp_drop == 12
    assert d.verdict == "blunder"


def test_noise_gaps_do_not_sum_into_a_turn_grade():
    # Clamp-then-sum turned three +1.5 HP jitters into a 4.5 HP slip.
    noisy = {"mean_final_hp": 60.0, "hp_sd": 20.0, "hp_n": 50}
    played = {**noisy, "mean_final_hp": 58.5}
    regrets = [regret_review.assess_decision(_hp_step(noisy, played)) for _ in range(3)]
    grade = regret_review.grade_turn(regrets)
    assert grade.hp_drop is None
    assert grade.verdict == "best"


def test_best_line_is_echoed():
    line = ["Bash", "Defend", "end turn"]
    d = regret_review.assess_decision(_hp_step({}, {}, best_line=line))
    assert d.best_line == line


def test_action_stable_key_maps_native_verbs():
    assert regret_review.action_stable_key({"action": "play", "card_index": 2}, []) == (2, None)
    assert regret_review.action_stable_key(
        {"action": "play", "card_index": 1, "target_index": 0}, []
    ) == (1, 0)
    assert regret_review.action_stable_key({"action": "end_turn"}, []) == (-1, None)
    assert regret_review.action_stable_key({"action": "drink_potion", "slot": 1}, []) == (
        -101,
        None,
    )
    assert regret_review.action_stable_key(None, []) == (None, None)


def test_assess_decision_flags_blunder_with_narrative():
    scores = [
        _score(0, "Strike", 0.58, is_best=False, target=0),
        _score(1, "Defend", 0.71, is_best=True),
    ]
    step = _combat_step(
        0,
        hand=[{"index": 0, "name": "Strike"}, {"index": 1, "name": "Defend"}],
        scores=scores,
        played_args={"action": "play", "card_index": 0, "target_index": 0},
    )
    d = regret_review.assess_decision(step)
    assert d is not None
    assert d.is_blunder is True
    assert d.matched_best is False
    assert d.drop_pct == 13
    assert abs(d.regret - 0.13) < 1e-9
    assert d.player_rank == 1  # one move (Defend) ranked above Strike
    assert d.message == ("Turn 3: Strike instead of Defend dropped P(win) 71%→58% (−13 pts)")


def test_assess_decision_wrong_target_same_card_names_both_enemies():
    # Same card (Strike), different targets: hitting Cultist when Jaw Worm was
    # the line. The label must surface the enemy so it doesn't read as the
    # target-blind "Strike instead of Strike".
    scores = [
        _score(0, "Strike", 0.58, is_best=False, target=0),
        _score(0, "Strike", 0.71, is_best=True, target=1),
    ]
    step = _combat_step(
        0,
        hand=[{"index": 0, "name": "Strike"}],
        scores=scores,
        played_args={"action": "play", "card_index": 0, "target_index": 0},
    )
    step["state"]["enemies"] = [
        {"index": 0, "name": "Cultist", "hp": 30},
        {"index": 1, "name": "Jaw Worm", "hp": 40},
    ]
    d = regret_review.assess_decision(step)
    assert d is not None
    assert d.is_blunder is True
    assert d.matched_best is False
    assert d.played_label == "Strike → Cultist"
    assert d.best_label == "Strike → Jaw Worm"
    assert d.played_label != d.best_label
    assert d.message == (
        "Turn 3: Strike → Cultist instead of Strike → Jaw Worm dropped P(win) 71%→58% (−13 pts)"
    )


def test_assess_decision_single_enemy_keeps_card_only_label():
    # One enemy -> the target can't be wrong, so the label stays clean.
    scores = [
        _score(0, "Strike", 0.58, is_best=False, target=0),
        _score(1, "Defend", 0.71, is_best=True),
    ]
    step = _combat_step(
        0,
        hand=[{"index": 0, "name": "Strike"}, {"index": 1, "name": "Defend"}],
        scores=scores,
        played_args={"action": "play", "card_index": 0, "target_index": 0},
    )
    step["state"]["enemies"] = [{"index": 0, "name": "Cultist", "hp": 30}]
    d = regret_review.assess_decision(step)
    assert d is not None
    assert d.played_label == "Strike"
    assert d.best_label == "Defend"


def test_assess_decision_matched_best_is_not_a_blunder():
    scores = [
        _score(0, "Strike", 0.58, is_best=False, target=0),
        _score(1, "Defend", 0.71, is_best=True),
    ]
    step = _combat_step(
        0,
        hand=[{"index": 0, "name": "Strike"}, {"index": 1, "name": "Defend"}],
        scores=scores,
        played_args={"action": "play", "card_index": 1},
    )
    d = regret_review.assess_decision(step)
    assert d is not None
    assert d.matched_best is True
    assert d.is_blunder is False
    assert abs(d.regret) < 1e-9
    assert d.player_rank == 0


def test_assess_decision_sub_threshold_drop_not_flagged():
    scores = [
        _score(0, "Strike", 0.69, is_best=False, target=0),
        _score(1, "Defend", 0.71, is_best=True),
    ]
    step = _combat_step(
        0,
        hand=[{"index": 0, "name": "Strike"}, {"index": 1, "name": "Defend"}],
        scores=scores,
        played_args={"action": "play", "card_index": 0, "target_index": 0},
    )
    d = regret_review.assess_decision(step)  # 2pt drop, default threshold 8
    assert d is not None and d.is_blunder is False


def test_assess_decision_none_without_alternatives_or_scores():
    # single legal move -> no counterfactual
    assert (
        regret_review.assess_decision(
            _combat_step(0, [], [_score(-1, "end turn", 0.5, True)], {"action": "end_turn"})
        )
        is None
    )
    # no scores at all
    assert regret_review.assess_decision(_combat_step(0, [], [], {"action": "end_turn"})) is None


def test_review_steps_orders_top_blunders_and_summarizes():
    big = _combat_step(
        0,
        hand=[{"index": 0, "name": "Strike"}, {"index": 1, "name": "Defend"}],
        scores=[_score(0, "Strike", 0.50, False, target=0), _score(1, "Defend", 0.71, True)],
        played_args={"action": "play", "card_index": 0, "target_index": 0},
        turn=2,
    )
    small = _combat_step(
        1,
        hand=[{"index": 0, "name": "Strike"}, {"index": 1, "name": "Defend"}],
        scores=[_score(0, "Strike", 0.60, False, target=0), _score(1, "Defend", 0.73, True)],
        played_args={"action": "play", "card_index": 0, "target_index": 0},
        turn=3,
    )
    out = regret_review.review_steps([big, small], top_n=1)
    assert out["summary"]["total_decisions"] == 2
    assert out["summary"]["blunder_count"] == 2
    assert out["summary"]["worst_drop_pct"] == 21
    assert len(out["blunders"]) == 1  # top_n
    assert out["blunders"][0]["turn"] == 2  # biggest drop first


def test_review_steps_uses_rescore_when_no_recorded_scores():
    step = _combat_step(
        0,
        hand=[{"index": 0, "name": "Strike"}, {"index": 1, "name": "Defend"}],
        scores=[],  # nothing recorded
        played_args={"action": "play", "card_index": 0, "target_index": 0},
    )

    def fake_rescore(_s):
        return [
            _score(0, "Strike", 0.58, is_best=False, target=0),
            _score(1, "Defend", 0.71, is_best=True),
        ]

    out = regret_review.review_steps([step], rescore=fake_rescore)
    assert out["summary"]["blunder_count"] == 1
    assert out["blunders"][0]["drop_pct"] == 13


# ── replay endpoint (recorded scores, no engine) ─────────────────────────────
def _line(step, kind, data):
    return json.dumps({"step": step, "ts": "2026-05-24T00:00:00", "type": kind, "data": data})


def _blunder_log(path):
    """A 1-combat-decision run: best move is Defend (71%) but Strike (58%) was
    played — a 13pt blunder, all from recorded MCTS stats."""
    state = {
        "decision": "combat_play",
        "round": 3,
        "hand": [
            {"index": 0, "name": "Strike"},
            {"index": 1, "name": "Defend"},
        ],
        "player": {"name": "Ironclad"},
        "enemies": [],
    }
    mcts = {
        "root_outcomes": [
            {
                "stable_idx": 0,
                "action_label": "Strike",
                "target_index": 0,
                "visits": 100,
                "q": 0.1,
                "wins": 58,
                "losses": 42,
                "ties": 0,
                "is_best": False,
            },
            {
                "stable_idx": 1,
                "action_label": "Defend",
                "target_index": None,
                "visits": 100,
                "q": 0.3,
                "wins": 71,
                "losses": 29,
                "ties": 0,
                "is_best": True,
            },
        ],
        "pv": [],
    }
    action = {"args": {"action": "play", "card_index": 0, "target_index": 0}}
    path.write_text(
        "\n".join(
            [
                _line(1, "state", state),
                _line(1, "mcts", mcts),
                _line(1, "action", action),
                _line(2, "state", {"decision": "map_select", "round": 0}),
            ]
        )
        + "\n"
    )


def _client(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    return TestClient(create_app(images_root=images, logs_dir=tmp_path))


def test_regret_review_endpoint_400_without_run(tmp_path):
    assert _client(tmp_path).post("/api/regret-review", json={}).status_code == 400


def test_regret_review_endpoint_surfaces_recorded_blunder(tmp_path):
    log = tmp_path / "run.jsonl"
    _blunder_log(log)
    c = _client(tmp_path)
    assert c.post("/api/load", json={"path": str(log)}).status_code == 200

    r = c.post("/api/regret-review", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["total_decisions"] == 1
    assert body["summary"]["blunder_count"] == 1
    assert len(body["blunders"]) == 1
    b = body["blunders"][0]
    assert b["played_label"] == "Strike"
    assert b["best_label"] == "Defend"
    assert b["drop_pct"] == 13
    assert b["is_blunder"] is True
    assert "71%→58%" in b["message"]


def test_regret_review_surfaces_blunder_in_ended_live_run(tmp_path, monkeypatch):
    """End-to-end guard for the "0 decisions · clean run" bug: a live run that is
    ended (``quit``) and reopened in the replay viewer must still have its combat
    decisions reviewed. ``quit`` writes the action args in the native engine shape
    so ``action_stable_key`` can resolve the played move; otherwise every combat
    step is skipped and the review reports a (false) clean run."""
    from fake_game import FakeGame
    from workbench import branching, reasoning
    from workbench.branching import BranchingLiveSession

    monkeypatch.setattr(branching, "Game", FakeGame)
    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="csharp")
    sess.start("Ironclad", "s", 0)
    sess.apply("select_map_node", {"col": 0, "row": 0})  # -> combat_play
    sess.apply("play_card", {"card_index": 0, "target_index": 0})  # the blunder
    log_path = sess.quit()["log_path"]
    assert log_path is not None

    # The ended-run log carries no MCTS records, so the endpoint re-scores each
    # combat step: Defend (71%) beats the Strike (58%) that was played.
    monkeypatch.setattr(
        reasoning,
        "score",
        lambda state, cfg: {
            "scores": [
                _score(0, "Strike", 0.58, is_best=False, target=0),
                _score(1, "Defend", 0.71, is_best=True),
            ],
            "pv": None,
            "config": cfg,
        },
    )
    c = _client(tmp_path)
    assert c.post("/api/load", json={"path": log_path}).status_code == 200
    body = c.post("/api/regret-review", json={}).json()
    assert body["summary"]["total_decisions"] == 1
    assert body["summary"]["blunder_count"] == 1
    assert body["blunders"][0]["played_label"] == "Strike"
    assert body["blunders"][0]["best_label"] == "Defend"
    assert body["blunders"][0]["drop_pct"] == 13


# ── live "blunder right after you did one" ───────────────────────────────────
def test_last_move_regret_flags_just_played_blunder(tmp_path, monkeypatch):
    from fake_game import FakeGame
    from workbench import branching, reasoning
    from workbench.branching import BranchingLiveSession

    monkeypatch.setattr(branching, "Game", FakeGame)
    # Score the parent combat position: ending the turn (71%) beats the Strike
    # the player just threw (58%).
    monkeypatch.setattr(
        reasoning,
        "score",
        lambda state, cfg: {
            "scores": [
                _score(0, "Strike", 0.58, is_best=False, target=0),
                _score(-1, "end turn", 0.71, is_best=True),
            ],
            "pv": None,
            "config": cfg,
        },
    )

    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="csharp")
    sess.start("Ironclad", "s", 0)
    sess.apply("select_map_node", {"col": 0, "row": 0})  # -> combat_play
    sess.apply("play_card", {"card_index": 0, "target_index": 0})  # the blunder

    result = sess.last_move_regret({})
    assert result is not None
    assert result.get("error") is None
    assert result["is_blunder"] is True
    assert result["played_label"] == "Strike"
    assert result["best_label"] == "end turn"
    assert result["drop_pct"] == 13


# ── display names: the review must never print raw localization keys ─────────
def _loc_key_log(path):
    """A 1-combat-decision run saved the way the LIVE session saves one: raw
    engine state (card names are localization keys) and NO ``mcts`` records."""
    state = {
        "decision": "combat_play",
        "round": 2,
        "hand": [
            {"index": 0, "name": "STRIKE_IRONCLAD.title"},
            {"index": 1, "name": "DEFEND_IRONCLAD.title"},
        ],
        "player": {"name": "Ironclad", "potions": []},
        "enemies": [],
    }
    action = {"args": {"action": "play", "card_index": 0, "target_index": 0}}
    path.write_text(
        "\n".join(
            [
                _line(1, "state", state),
                _line(1, "action", action),
                _line(2, "state", {"decision": "map_select", "round": 0}),
            ]
        )
        + "\n"
    )


def test_regret_review_endpoint_resolves_loc_keys(tmp_path, monkeypatch):
    """A run saved by the live workbench persists raw loc keys in its states.
    The review labels rows off ``state["hand"]``, so without resolution it
    printed ``played STRIKE_IRONCLAD.title instead of DEFEND_IRONCLAD.title``.
    """
    from workbench import reasoning

    class _Tag:
        def __init__(self, win_prob, is_best):
            self.win_prob = win_prob
            self.q = win_prob
            self.visits = 100
            self.is_best = is_best
            self.q_saturated = False
            self.low_confidence = False
            self.target_index = None

    # Patch the ENGINE call, not `reasoning.score`: the labels under test are
    # built by `score` itself out of the state's hand, so a patched `score`
    # would hand back labels the fix never touched.
    monkeypatch.setattr(
        reasoning, "_recommend", lambda state, cfg: {0: _Tag(0.40, False), 1: _Tag(0.90, True)}
    )

    log = tmp_path / "raw.jsonl"
    _loc_key_log(log)
    c = _client(tmp_path)
    assert c.post("/api/load", json={"path": str(log)}).status_code == 200
    body = c.post("/api/regret-review", json={}).json()

    assert body["summary"]["total_decisions"] == 1
    blunder = body["blunders"][0]
    assert blunder["played_label"] == "Strike"
    assert blunder["best_label"] == "Defend"
    assert ".title" not in blunder["message"]


def test_regret_review_endpoint_resolves_recorded_row_labels(tmp_path):
    """A RECORDED score row carries its own label, so resolving the state is
    not enough — a log written before labels were resolved at record time still
    holds raw keys there."""
    state = {
        "decision": "combat_play",
        "round": 2,
        "hand": [
            {"index": 0, "name": "STRIKE_IRONCLAD.title"},
            {"index": 1, "name": "DEFEND_IRONCLAD.title"},
        ],
        "player": {"name": "Ironclad", "potions": []},
        "enemies": [],
    }
    mcts = {
        "root_outcomes": [
            {
                "stable_idx": 0,
                "action_label": "STRIKE_IRONCLAD.title",
                "target_index": None,
                "visits": 100,
                "q": 0.1,
                "wins": 40,
                "losses": 60,
                "ties": 0,
                "is_best": False,
            },
            {
                "stable_idx": 1,
                "action_label": "DEFEND_IRONCLAD.title",
                "target_index": None,
                "visits": 60,
                "q": 0.4,
                "wins": 54,
                "losses": 6,
                "ties": 0,
                "is_best": True,
            },
        ],
        "pv": [],
    }
    log = tmp_path / "rawrows.jsonl"
    log.write_text(
        "\n".join(
            [
                _line(1, "state", state),
                _line(1, "mcts", mcts),
                _line(1, "action", {"args": {"action": "play", "card_index": 0}}),
                _line(2, "state", {"decision": "map_select", "round": 0}),
            ]
        )
        + "\n"
    )
    c = _client(tmp_path)
    assert c.post("/api/load", json={"path": str(log)}).status_code == 200
    blunder = c.post("/api/regret-review", json={}).json()["blunders"][0]
    assert blunder["played_label"] == "Strike"
    assert blunder["best_label"] == "Defend"
    assert ".title" not in blunder["message"]


# ── faithfulness: the review grades the agent with the agent's own search ────
def test_review_run_uses_the_agents_own_search(tmp_path, monkeypatch):
    """An autoplayed move must be graded against the distribution that CHOSE
    it, not against a fresh search — otherwise the AI blunders against itself.
    """
    from fake_game import FakeGame
    from workbench import branching, reasoning
    from workbench.branching import BranchingLiveSession

    monkeypatch.setattr(branching, "Game", FakeGame)

    def _boom(*_a, **_k):
        raise AssertionError("re-searched a decision that has a recorded search")

    monkeypatch.setattr(reasoning, "score", _boom)

    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="csharp")
    sess.start("Ironclad", "s", 0)
    sess.apply("select_map_node", {"col": 0, "row": 0})  # -> combat_play
    sess.apply("play_card", {"card_index": 0, "target_index": 0})

    # Stand in for `_agent_step`, which stashes the native agent's root
    # distribution on the node its move produced.
    child = sess._nodes[sess._active_id]
    child.state["mcts"] = {
        "actions": [
            {
                "action": {"action": "play", "card_index": 0, "target_index": 0},
                "p_win": 0.62,
                "q": 0.2,
                "visits": 700,
                "is_best": True,
            },
            {
                "action": {"action": "end_turn"},
                "p_win": 0.55,
                "q": 0.1,
                "visits": 300,
                "is_best": False,
            },
        ]
    }

    out = sess.review_run({})
    assert out["summary"]["total_decisions"] == 1
    # The agent took its own top line, so there is nothing to regret.
    assert out["summary"]["blunder_count"] == 0
    assert out["decisions"][0]["matched_best"] is True
    assert out["decisions"][0]["visits"] == 700


def test_quit_persists_the_agent_search_for_replay(tmp_path, monkeypatch):
    """The same numbers must survive into the run log, or a replay of the run
    falls back to re-searching and disagrees with the run it is reviewing."""
    from fake_game import FakeGame
    from workbench import branching, reasoning
    from workbench.branching import BranchingLiveSession

    monkeypatch.setattr(branching, "Game", FakeGame)

    sess = BranchingLiveSession(logs_dir=str(tmp_path), backend="csharp")
    sess.start("Ironclad", "s", 0)
    sess.apply("select_map_node", {"col": 0, "row": 0})
    sess.apply("play_card", {"card_index": 0, "target_index": 0})
    child = sess._nodes[sess._active_id]
    child.state["mcts"] = {
        "actions": [
            {
                "action": {"action": "play", "card_index": 0, "target_index": 0},
                "p_win": 0.41,
                "q": 0.1,
                "visits": 400,
                "is_best": False,
            },
            {
                "action": {"action": "end_turn"},
                "p_win": 0.79,
                "q": 0.4,
                "visits": 600,
                "is_best": True,
            },
        ]
    }
    log_path = sess.quit()["log_path"]
    assert log_path is not None

    def _boom(*_a, **_k):
        raise AssertionError("re-searched a decision whose search was recorded")

    monkeypatch.setattr(reasoning, "score", _boom)
    c = _client(tmp_path)
    assert c.post("/api/load", json={"path": log_path}).status_code == 200
    body = c.post("/api/regret-review", json={}).json()

    assert body["summary"]["total_decisions"] == 1
    blunder = body["blunders"][0]
    # 79 - 41, straight off the agent's own p_win — not wins/visits.
    assert blunder["drop_pct"] == 38
    assert blunder["best_label"] == "end turn"


# ── per-turn grading rule (shared fixture with advisors::grade_turn) ─────────
_GRADE_TURN_CASES = (
    Path(__file__).resolve().parents[1] / "fixtures" / "coach" / "grade_turn_cases.json"
)


def _fixture_regret(d):
    if d is None:
        return None
    return regret_review.DecisionRegret(
        idx=0,
        node_id=None,
        turn=1,
        floor=1,
        act=1,
        played_label="played",
        played_win_prob=d["played_win_prob"],
        best_label="best",
        best_win_prob=d["best_win_prob"],
        regret=max(0.0, d["best_win_prob"] - d["played_win_prob"]),
        drop_pct=d["drop_pct"],
        player_rank=0,
        n_actions=2,
        visits=100,
        matched_best=d["verdict"] == "best",
        is_blunder=d["verdict"] == "blunder",
        message="",
        hp_drop=d["hp_drop"],
        severity=d["severity"],
        verdict=d["verdict"],
    )


@pytest.mark.parametrize(
    "case",
    json.loads(_GRADE_TURN_CASES.read_text(encoding="utf-8"))["cases"],
    ids=lambda c: c["name"],
)
def test_grade_turn_shared_cases(case):
    grade = regret_review.grade_turn([_fixture_regret(r) for r in case["regrets"]])
    if case["expect"] is None:
        assert grade is None
    else:
        assert grade is not None
        assert grade.to_dict() == case["expect"]
