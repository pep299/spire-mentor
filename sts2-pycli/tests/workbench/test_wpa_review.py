"""WPA run review (issue #201): pure core, live value capture, endpoints.

The pure scoring core (`workbench.wpa_review`) needs neither engine nor net;
the live-session tests fake the rust backend's ``rm`` and monkeypatch
``ppo_policy.evaluate`` so value capture runs without torch or a checkpoint.
"""

import json
import sys
from types import SimpleNamespace

import pytest
from fake_game import FakeGame  # shared stub from tests/workbench/fake_game.py
from fastapi.testclient import TestClient
from workbench import wpa_review
from workbench.server import create_app
from workbench.timeline import TimelineStore

# ── pure core ────────────────────────────────────────────────────────────


def _rows(values, floors=None):
    floors = floors or list(range(1, len(values) + 1))
    return [
        {
            "node_id": 10 + i,
            "idx": i,
            "decision": "card_reward",
            "floor": floors[i],
            "act": 1,
            "value": v,
            "taken_label": f"Take card {i}",
        }
        for i, v in enumerate(values)
    ]


def test_build_points_skips_missing_values_and_tracks_deltas():
    rows = _rows([0.5, None, 0.4, 0.7])
    points = wpa_review.build_points(rows)
    assert [p.value for p in points] == [0.5, 0.4, 0.7]
    assert points[0].delta_prev == pytest.approx(0.0)
    assert points[1].delta_prev == pytest.approx(-0.1)
    assert points[2].delta_prev == pytest.approx(0.3)
    assert points[0].node_id == 10 and points[1].node_id == 12


def test_turning_points_attribution_ranking_and_span():
    # Swings: p0 +0.02 (filtered by min_swing), p1 -0.42, p2 +0.30.
    rows = _rows([0.50, 0.52, 0.10, 0.40])
    rows[2]["idx"] = 5  # a fight happened between p1 (idx 1) and p2 (idx 5)
    points = wpa_review.build_points(rows)
    tps = wpa_review.turning_points(points, top_k=5, min_swing=0.05)
    assert [t["node_id"] for t in tps] == [11, 12]  # ranked by |swing|
    top = tps[0]
    assert top["swing"] == pytest.approx(-0.42)
    assert top["value_before"] == pytest.approx(0.52)
    assert top["value_after"] == pytest.approx(0.1)
    assert "dropped" in top["message"] and "Take card 1" in top["message"]
    assert "spans the fight" in top["message"]  # idx gap 1 → 5
    assert "spans the fight" not in tps[1]["message"]
    assert len(wpa_review.turning_points(points, top_k=1, min_swing=0.05)) == 1


def test_review_payload_and_unavailability():
    out = wpa_review.review(_rows([0.5, 0.3, 0.6]), top_k=2, min_swing=0.05, won=True)
    assert out["available"] is True
    assert out["summary"]["n_points"] == 3
    assert out["summary"]["biggest_drop"] == pytest.approx(-0.2)
    assert out["summary"]["biggest_gain"] == pytest.approx(0.3)
    assert out["summary"]["won"] is True
    assert len(out["caveats"]) == 3
    assert len(out["turning_points"]) == 2

    empty = wpa_review.review(_rows([0.5, None, None]), reason_if_empty="nope")
    assert empty["available"] is False and empty["reason"] == "nope"
    assert empty["turning_points"] == []


# ── live capture + endpoints ─────────────────────────────────────────────


class FakeRunGame(FakeGame):
    """FakeGame variant with non-combat decisions and a stub native run, so the
    branching session's value capture fires (`_node_value` needs a native run
    and a non-combat decision)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._rm = SimpleNamespace(player_won=False)
        self._floor = 0

    def native_run(self):
        return self._rm

    def start(self, character="Ironclad", seed="test", ascension=0, lang="en", act1="overgrowth"):
        super().start(character=character, seed=seed, ascension=ascension)
        self._state = {"decision": "map_select", "choices": [], "floor": 0, "act": 1}
        return self._state

    def act(self, action, **args):
        self.calls.append((action, args))
        self._floor += 1
        self._state = {"decision": "card_reward", "floor": self._floor, "act": 1, "cards": []}
        return self._state


def _client(tmp_path, monkeypatch, game_cls=FakeRunGame, values=None):
    from workbench import branching, ppo_policy

    monkeypatch.setattr(branching, "Game", game_cls)
    if values is not None:
        seq = iter(values)
        monkeypatch.setattr(
            ppo_policy,
            "evaluate",
            lambda rm, *, character="Ironclad", ascension=0, impl="python": SimpleNamespace(
                value=next(seq)
            ),
        )
        # `recommend` (the `ppo` brain's map-route seed, `_driving_first_col`)
        # goes through `evaluate` too; keep it from consuming the sequence.
        monkeypatch.setattr(ppo_policy, "recommend", lambda rm, **kw: None)
    images = tmp_path / "images"
    images.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    return TestClient(create_app(images_root=images, logs_dir=logs))


def _start(c):
    r = c.post("/api/live/start", json={"character": "Ironclad", "seed": "s", "backend": "csharp"})
    assert r.status_code == 200


def test_live_wpa_review_unavailable_without_values(tmp_path, monkeypatch):
    # Plain FakeGame: no rm → no values ever recorded.
    c = _client(tmp_path, monkeypatch, game_cls=FakeGame)
    assert c.post("/api/live/wpa-review", json={}).status_code == 400  # no run
    _start(c)
    body = c.post("/api/live/wpa-review", json={}).json()
    assert body["available"] is False and "value data" in body["reason"]


def test_live_wpa_review_curve_and_turning_points(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, values=[0.50, 0.52, 0.10, 0.40])
    _start(c)
    for _ in range(3):
        assert (
            c.post("/api/live/action", json={"action": "choose_option", "args": {}}).status_code
            == 200
        )
    body = c.post("/api/live/wpa-review", json={"config": {"min_swing": 0.05}}).json()
    assert body["available"] is True
    assert [p["value"] for p in body["points"]] == [0.50, 0.52, 0.10, 0.40]
    # Turning points carry live node ids (seek targets) + the taken action label.
    top = body["turning_points"][0]
    assert top["swing"] == pytest.approx(-0.42)
    assert isinstance(top["node_id"], int)
    assert top["taken_label"]


def test_quit_persists_wpa_records_and_replay_review_reads_them(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, values=[0.50, 0.52, 0.10, 0.40])
    _start(c)
    for _ in range(3):
        c.post("/api/live/action", json={"action": "choose_option", "args": {}})
    quit_body = c.post("/api/live/quit").json()
    log_path = quit_body["log_path"]
    assert log_path

    # The log carries one wpa record per valued decision, attached to its state.
    with open(log_path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    wpa_records = [e for e in lines if e.get("type") == "wpa"]
    assert [r["data"]["value"] for r in wpa_records] == [0.50, 0.52, 0.10, 0.40]
    assert all("taken_label" in r["data"] for r in wpa_records)

    store = TimelineStore.load_jsonl(log_path)
    assert [s["wpa"]["value"] for s in store.all() if s.get("wpa")] == [0.50, 0.52, 0.10, 0.40]

    # Replay endpoint reads the recorded values — no net involved.
    assert c.post("/api/load", json={"path": log_path}).status_code == 200
    body = c.post("/api/wpa-review", json={"config": {"min_swing": 0.05}}).json()
    assert body["available"] is True
    assert [p["value"] for p in body["points"]] == [0.50, 0.52, 0.10, 0.40]
    assert body["turning_points"][0]["swing"] == pytest.approx(-0.42)
    assert body["turning_points"][0]["node_id"] is None  # replay: seek by idx


def test_replay_wpa_review_unavailable_on_legacy_logs(tmp_path, monkeypatch):
    # A log recorded before value capture (no wpa records) → available: False.
    c = _client(tmp_path, monkeypatch, game_cls=FakeGame)
    _start(c)
    c.post("/api/live/action", json={"action": "end_turn", "args": {}})
    log_path = c.post("/api/live/quit").json()["log_path"]
    assert c.post("/api/load", json={"path": log_path}).status_code == 200
    body = c.post("/api/wpa-review", json={}).json()
    assert body["available"] is False and "no recorded value data" in body["reason"]


def test_wpa_verify_error_contracts(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, game_cls=FakeGame)
    assert c.post("/api/live/wpa-verify", json={"node_id": 0}).status_code == 400  # no run
    _start(c)
    # FakeGame has no rm → the backend can't autoplay a counterfactual.
    assert c.post("/api/live/wpa-verify", json={"node_id": 0}).status_code == 400
    assert c.post("/api/live/wpa-verify", json={"node_id": 999}).status_code == 404


def test_replay_wpa_review_reports_won_from_terminal_victory(tmp_path):
    # Fix #4: the replay review surfaces the run outcome from the logged
    # ``game_over`` state's ``victory`` (the live path reads it from ``rm``).
    log = tmp_path / "won_run.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"type": "state", "step": 0, "data": {"decision": "card_reward", "floor": 1}},
                {"type": "wpa", "step": 0, "data": {"value": 0.50, "taken_label": "Take A"}},
                {"type": "state", "step": 1, "data": {"decision": "card_reward", "floor": 2}},
                {"type": "wpa", "step": 1, "data": {"value": 0.30, "taken_label": "Take B"}},
                {"type": "state", "step": 2, "data": {"decision": "game_over", "victory": True}},
            ]
        )
        + "\n"
    )
    images = tmp_path / "images"
    images.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    c = TestClient(create_app(images_root=images, logs_dir=logs))
    assert c.post("/api/load", json={"path": str(log)}).status_code == 200
    body = c.post("/api/wpa-review", json={"config": {"min_swing": 0.05}}).json()
    assert body["available"] is True
    assert body["summary"]["won"] is True  # derived from the terminal victory


# ── wpa_verify orchestration (success + fix verification) ─────────────────


def _verify_session(tmp_path, monkeypatch, *, options):
    """A BranchingLiveSession with a 3-node line (root → decision → played
    child) and the heavy engine calls stubbed, so ``wpa_verify``'s
    orchestration (arm running, alternative selection, verdict) runs without a
    rust backend. The decision under test is node id 1; the played move is the
    edge decision→played (id 2). Returns ``(sess, apply_calls)``; the caller
    installs a ``run_mode_step`` stub driving each arm's outcome."""
    from workbench import branching

    # ``wpa_verify`` builds a fresh combat agent per arm (``import sts2_rs``);
    # stub the module so it works with no rust extension present.
    monkeypatch.setitem(
        sys.modules, "sts2_rs", SimpleNamespace(HeuristicRunAgent=lambda **kw: object())
    )

    sess = branching.BranchingLiveSession(logs_dir=str(tmp_path), backend="csharp")
    n = branching._Node
    root = n(id=0, parent_id=None, idx=0, decision="map_select", state={}, label="root", value=0.5)
    decision = n(
        id=1,
        parent_id=0,
        idx=1,
        decision="card_reward",
        state={},
        action="choose",
        args={"idx": 9},
        label="decision",
        value=0.52,
    )
    played = n(
        id=2,
        parent_id=1,
        idx=2,
        decision="card_reward",
        state={},
        action="choose",
        args={"idx": 0},
        label="Take Bash",
        value=0.1,
    )
    root.child_ids = [1]
    decision.child_ids = [2]
    sess._nodes = {0: root, 1: decision, 2: played}
    sess._root_id = 0
    sess._active_id = 2
    sess._next_id = 3
    sess.seed = "123"
    sess.game = FakeRunGame()  # a backend WITH a native run

    seeks: list[int] = []
    apply_calls: list[dict] = []

    def fake_seek(node_id):
        sess._active_id = node_id
        seeks.append(node_id)
        return {}

    def fake_apply(action, args):
        apply_calls.append({"action": action, "args": args})
        return {"error": None, "state": {}, "label": "alt"}

    monkeypatch.setattr(sess, "seek", fake_seek)
    monkeypatch.setattr(sess, "apply", fake_apply)
    monkeypatch.setattr(sess, "ppo_policy_panel", lambda: {"options": options})
    return sess, apply_calls


def _arm_driver(sess, monkeypatch, *, played_won, alt_won, over=True):
    """Install a ``run_mode_step`` stub returning a terminal outcome keyed to
    which arm is live: arm 1 (played) is active right after seek→played child,
    arm 2 (alternative) right after ``apply``."""
    state = {"phase": "played"}
    orig_apply = sess.apply

    def apply_marks_alt(action, args):
        state["phase"] = "alt"
        return orig_apply(action, args)

    monkeypatch.setattr(sess, "apply", apply_marks_alt)

    def fake_step():
        won = played_won if state["phase"] == "played" else alt_won
        return {"done": bool(over), "over": bool(over), "won": won, "error": None, "state": {}}

    monkeypatch.setattr(sess, "run_mode_step", fake_step)


_STD_OPTIONS = [
    {"action": "choose", "args": {"idx": 1}, "label": "Skip", "prob": 0.6},
    {"action": "choose", "args": {"idx": 0}, "label": "Take Bash", "prob": 0.4},  # == played move
]


@pytest.mark.parametrize(
    "played_won,alt_won,verdict",
    [(False, True, "alternative_better"), (True, False, "played_better"), (True, True, "same")],
)
def test_wpa_verify_verdicts(tmp_path, monkeypatch, played_won, alt_won, verdict):
    sess, apply_calls = _verify_session(tmp_path, monkeypatch, options=_STD_OPTIONS)
    _arm_driver(sess, monkeypatch, played_won=played_won, alt_won=alt_won)

    out = sess.wpa_verify(1)

    assert out["verdict"] == verdict
    assert out["node_id"] == 1
    assert out["played"]["label"] == "Take Bash" and out["played"]["won"] is played_won
    # The alternative is the highest-prob option that isn't the played move.
    assert out["alternative"]["label"] == "Skip" and out["alternative"]["won"] is alt_won
    assert apply_calls == [{"action": "choose", "args": {"idx": 1}}]
    assert "not a probability" in out["note"]
    # The active leaf is restored (last seek returns to the original active id).
    assert sess._active_id == 2


def test_wpa_verify_alternative_dedup_is_or_not_and(tmp_path, monkeypatch):
    # Fix #2: the only non-played option shares the played move's LABEL but has
    # a different native action. The old AND-dedup (differ in action AND label)
    # would reject it → "no alternative"; the OR-dedup accepts it.
    options = [
        {"action": "choose", "args": {"idx": 0}, "label": "Take Bash", "prob": 0.7},  # exact played
        {
            "action": "choose",
            "args": {"idx": 2},
            "label": "Take Bash",
            "prob": 0.3,
        },  # same label only
    ]
    sess, apply_calls = _verify_session(tmp_path, monkeypatch, options=options)
    _arm_driver(sess, monkeypatch, played_won=False, alt_won=True)

    out = sess.wpa_verify(1)

    assert out["verdict"] == "alternative_better"
    # It picked the genuine alternative (different action), not the played move.
    assert apply_calls == [{"action": "choose", "args": {"idx": 2}}]


def test_wpa_verify_no_alternative_raises(tmp_path, monkeypatch):
    # Only the exact played move is on offer → nothing to compare against.
    only_played = [{"action": "choose", "args": {"idx": 0}, "label": "Take Bash", "prob": 1.0}]
    sess, _ = _verify_session(tmp_path, monkeypatch, options=only_played)
    _arm_driver(sess, monkeypatch, played_won=False, alt_won=False)
    with pytest.raises(ValueError, match="no alternative"):
        sess.wpa_verify(1)
    assert sess._active_id == 2  # active leaf still restored on the error path


def test_wpa_verify_nonterminal_arm_is_inconclusive_not_a_false_verdict(tmp_path, monkeypatch):
    # Fix #5: an arm that never ends (here: capped) must NOT be reported as a
    # loss — the verdict falls back to "same" with an inconclusive note, and
    # the arm carries an explanatory error instead of a bogus win flag.
    sess, _ = _verify_session(tmp_path, monkeypatch, options=_STD_OPTIONS)
    _arm_driver(sess, monkeypatch, played_won=False, alt_won=False, over=False)

    out = sess.wpa_verify(1, max_steps=3)

    assert out["verdict"] == "same"
    assert "inconclusive" in out["note"]
    assert out["played"]["over"] is False
    assert "cap" in (out["played"]["error"] or "")
    assert out["played"]["steps"] == 3  # bounded by max_steps


def _verify_line_session(tmp_path, monkeypatch, *, options):
    """A BranchingLiveSession with a 5-node recorded line

        root(0) → A(1, decision) → mid(2) → B(3, decision) → leaf(4)

    so a turning point (node 3) sits ABOVE the final leaf, matching a real
    run's review. Heavy engine calls stubbed exactly like ``_verify_session``;
    ``seek`` is the real read-only rewind (just moves ``_active_id``, no new
    node). Returns ``(sess, apply_calls)``."""
    from workbench import branching

    monkeypatch.setitem(
        sys.modules, "sts2_rs", SimpleNamespace(HeuristicRunAgent=lambda **kw: object())
    )
    sess = branching.BranchingLiveSession(logs_dir=str(tmp_path), backend="csharp")
    n = branching._Node
    nodes = {
        0: n(id=0, parent_id=None, idx=0, decision="map_select", state={}, label="root", value=0.5),
        1: n(
            id=1,
            parent_id=0,
            idx=1,
            decision="card_reward",
            state={},
            action="choose",
            args={"idx": 9},
            label="A",
            value=0.52,
        ),
        2: n(
            id=2,
            parent_id=1,
            idx=2,
            decision="map_select",
            state={},
            action="choose",
            args={"idx": 0},
            label="Take A-card",
            value=0.4,
        ),
        3: n(
            id=3,
            parent_id=2,
            idx=3,
            decision="card_reward",
            state={},
            action="choose",
            args={"idx": 9},
            label="B",
            value=0.6,
        ),
        4: n(
            id=4,
            parent_id=3,
            idx=4,
            decision="card_reward",
            state={},
            action="choose",
            args={"idx": 0},
            label="Take Bash",
            value=0.1,
        ),
    }
    nodes[0].child_ids = [1]
    nodes[1].child_ids = [2]
    nodes[2].child_ids = [3]
    nodes[3].child_ids = [4]
    sess._nodes = nodes
    sess._root_id = 0
    sess._active_id = 4  # review is generated with the final leaf active
    sess._next_id = 5
    sess.seed = "123"
    sess.game = FakeRunGame()  # a backend WITH a native run

    apply_calls: list[dict] = []

    def fake_seek(node_id):
        sess._active_id = node_id  # real seek semantics: read-only rewind
        return {}

    def fake_apply(action, args):
        apply_calls.append({"action": action, "args": args})
        return {"error": None, "state": {}, "label": "alt"}

    monkeypatch.setattr(sess, "seek", fake_seek)
    monkeypatch.setattr(sess, "apply", fake_apply)
    monkeypatch.setattr(sess, "ppo_policy_panel", lambda: {"options": options})
    return sess, apply_calls


def test_wpa_verify_after_seeking_to_inspect_a_turning_point(tmp_path, monkeypatch):
    # Regression: the Run Review UI seeks the active leaf to a turning point
    # when you click it (to jump the viewer there). ``wpa_verify`` must still
    # verify a turning point after that rewind — it grades the RECORDED line,
    # not wherever the active leaf currently sits. Before the fix, verifying a
    # turning point that is no longer a strict ancestor of the rewound active
    # leaf 409'd with "node N is not on the active line".
    sess, apply_calls = _verify_line_session(tmp_path, monkeypatch, options=_STD_OPTIONS)
    _arm_driver(sess, monkeypatch, played_won=False, alt_won=True)

    # User clicked the floor-8 turning point (node 1) → active leaf rewinds to 1.
    sess.seek(1)
    assert sess._active_id == 1

    # Now verify the LATER turning point (node 3): it is a descendant of the
    # rewound active leaf, not an ancestor, so the old active-line check failed.
    out = sess.wpa_verify(3)

    assert out["node_id"] == 3
    assert out["verdict"] == "alternative_better"
    assert out["played"]["label"] == "Take Bash"  # node 3's recorded child (id 4)
    assert apply_calls == [{"action": "choose", "args": {"idx": 1}}]
    # The active leaf the user parked on (node 1) is restored afterwards.
    assert sess._active_id == 1


def test_wpa_verify_of_the_active_leaf_itself_uses_the_recorded_child(tmp_path, monkeypatch):
    # And verifying the SAME turning point you just seeked to (active == node)
    # must work too, using the node's recorded on-line child rather than
    # 409'ing "no follow-up move".
    sess, _ = _verify_line_session(tmp_path, monkeypatch, options=_STD_OPTIONS)
    _arm_driver(sess, monkeypatch, played_won=True, alt_won=False)
    sess.seek(3)  # click the floor-N turning point, then verify it
    out = sess.wpa_verify(3)
    assert out["verdict"] == "played_better"
    assert out["played"]["label"] == "Take Bash"


def test_wpa_verify_wall_clock_deadline(tmp_path, monkeypatch):
    # Fix #5: the wall-clock deadline bounds an arm even when steps would return
    # (a fake monotonic clock jumps past the deadline before the first step).
    from workbench import branching

    sess, _ = _verify_session(tmp_path, monkeypatch, options=_STD_OPTIONS)
    _arm_driver(sess, monkeypatch, played_won=True, alt_won=True, over=True)
    clock = {"t": 0.0}

    def fake_monotonic():
        t = clock["t"]
        clock["t"] += 1000.0  # every read jumps 1000s → deadline is always in the past
        return t

    monkeypatch.setattr(branching.time, "monotonic", fake_monotonic)

    out = sess.wpa_verify(1, max_seconds=5.0)

    assert out["played"]["steps"] == 0 and out["played"]["over"] is False
    assert "timed out" in (out["played"]["error"] or "")
    assert out["verdict"] == "same" and "inconclusive" in out["note"]
