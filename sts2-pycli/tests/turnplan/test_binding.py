"""`MctsCombatAgent.turn_plan_labels`: one teacher-labelled position (GH #830).

The binding exposes the turn-plan enumerator, plan features, anchors and the
teacher prices (the deployed search forced onto each plan) as one dict per
position. These tests pin its shape and that labelling is a pure read: it
moves neither the run nor the agent's own decision stream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ppo.combat_bench.run import build_run, cell_seed, load_set

sts2_rs = pytest.importorskip("sts2_rs")

BENCH = Path(__file__).resolve().parents[2] / "data" / "combat_bench"
PLAN_DIM = 18


def _first_dev_fight() -> dict[str, Any]:
    """The first `devpool_v1` fight outside `dev_v1` (the G1/G2 screen set)."""
    screened = {f["key"] for f in load_set(str(BENCH / "ironclad_dev_v1.jsonl"))}
    pool = load_set(str(BENCH / "ironclad_devpool_v1.jsonl"))
    return next(f for f in pool if f["key"] not in screened)


def _view(rm: Any) -> str:
    return json.dumps(sts2_rs.state_to_dict(rm.combat_state()), sort_keys=True)


def test_labels_on_first_decision_of_a_dev_fight() -> None:
    fight = _first_dev_fight()
    rm = build_run(fight, cell_seed(fight["key"], 0))
    agent = sts2_rs.MctsCombatAgent(seed=cell_seed(fight["key"], 0))
    before = _view(rm)

    labels = agent.turn_plan_labels(rm, sims_per_plan=50)

    assert _view(rm) == before, "labelling must not advance the run"
    assert labels is not None
    plans = labels["plans"]
    assert labels["n_plans"] == len(plans) >= 1
    assert isinstance(labels["cap_hit"], bool)
    deck = labels["deck"]
    assert deck["cards"] and len(deck["ctx"]) == 6
    assert all(isinstance(name, str) and isinstance(n, int) for name, n in deck["powers"])
    for p in plans:
        assert len(p["features"]) == PLAN_DIM
        assert p["end"] in {"end_turn", "chance", "terminal"}
        assert p["actions"] and all(isinstance(a, dict) for a in p["actions"])
        assert isinstance(p["logp_heur"], float) and p["logp_heur"] <= 0.0
        assert p["logp_rl2"] is None or p["logp_rl2"] <= 0.0
        assert set(p["anchors"]) == {"lethal", "max_block", "max_damage", "heuristic", "rl2"}
        assert isinstance(p["q"], float) and isinstance(p["q_var"], float)
        if p["end"] == "terminal":
            assert p["visits"] == 0
        else:
            assert 0 < p["visits"] <= 50
    assert any(p["end"] == "end_turn" for p in plans)
    assert sum(p["anchors"]["max_block"] for p in plans) == 1
    # Ironclad ships a playout net, so the RL2 log-probability is present.
    assert all(p["logp_rl2"] is not None for p in plans)


def test_labels_do_not_perturb_the_agents_play() -> None:
    """A labels call leaves every later `step()` exactly as it would have been."""
    fight = _first_dev_fight()
    seed = cell_seed(fight["key"], 0)

    def play(label_first: bool) -> list[Any]:
        rm = build_run(fight, seed)
        agent = sts2_rs.MctsCombatAgent(seed=seed, sims=40, boss_sims=40)
        if label_first:
            agent.turn_plan_labels(rm, sims_per_plan=10)
        return [agent.step(rm) for _ in range(4)]

    assert play(label_first=True) == play(label_first=False)


def test_labels_are_none_outside_combat() -> None:
    rm = sts2_rs.RunManager(seed=1, character="Ironclad", ascension_level=10)
    agent = sts2_rs.MctsCombatAgent(seed=1)
    assert rm.phase() != "combat"
    assert agent.turn_plan_labels(rm, sims_per_plan=10) is None
