"""GH #28: acquisition-time card decisions remain explicit PPO actions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sts2_rs
from ppo import encoder, vocab
from ppo.env import metrics_snapshot, observe

SEED = 28028
MODEL = Path(__file__).resolve().parents[1] / "data/ppo/ppo_fullrun_ironclad_cont300_it1150.onnx"


def _run(deck: list[str], relic: str):
    rm = sts2_rs.RunManager(
        seed=SEED,
        character="Ironclad",
        ascension_level=10,
        game_version="v0.111.0",
        interactive_card_transform=True,
    )
    rm.debug_set_loadout(deck=deck, relics=[])
    assert rm.force_add_relic(relic)
    return rm


def _obs(rm):
    return observe(rm, rm.phase(), "card-pick", seed=SEED, ascension=10)


def test_claws_confirm_zero_and_same_onnx_token_in_both_encoders():
    rm = _run(["STRIKE_IRONCLAD"] * 8, "CLAWS")
    actions = rm.available_actions()
    assert len(actions) == 9
    assert actions[-1] == {"action": "transform_deck_card", "idx": 8}
    assert len(rm.master_deck_detailed()) == 8
    before = rm.master_deck_detailed()

    obs = _obs(rm)
    rust_obs = json.loads(sts2_rs.ppo_observe_json(rm, rm.phase(), "card-pick", SEED, 10))
    assert obs["transform_select_deck_indices"] == rust_obs["transform_select_deck_indices"]
    features = encoder.featurize_row(obs, actions, metrics_snapshot(rm))
    assert features.act_cat[-1][6] == vocab.fixed_index("skip_card")

    assert rm.take_action(actions[-1]) == "advanced"
    assert rm.master_deck_detailed() == before


def test_claws_one_target_survives_reindexing_before_confirm():
    rm = _run(["STRIKE_IRONCLAD", "STRIKE_IRONCLAD", "BASH"], "CLAWS")
    assert rm.take_action({"action": "transform_deck_card", "idx": 2}) == "advanced"
    assert rm.master_deck_detailed()[-1][0].endswith("BASH")
    actions = rm.available_actions()
    assert len(actions) == 3
    assert actions[-1] == {"action": "transform_deck_card", "idx": 2}
    assert rm.take_action(actions[-1]) == "advanced"
    assert rm.master_deck_detailed()[-1][0].endswith("MAUL")


@pytest.mark.parametrize(
    "relic,deck,expected",
    [
        ("TRI_BOOMERANG", ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"], 3),
        ("PAELS_GROWTH", ["STRIKE_IRONCLAD", "BASH", "DEFEND_IRONCLAD"], 1),
    ],
)
def test_relic_enchantment_is_not_auto_applied(relic, deck, expected):
    rm = _run(deck, relic)
    actions = rm.available_actions()
    assert len(actions) >= 2
    assert all(a["action"] == "enchant_deck_card" for a in actions)
    assert rm.pending_deck_enchants == expected
    obs = _obs(rm)
    assert obs["enchant_select_deck_indices"]
    assert len(encoder.featurize_row(obs, actions, metrics_snapshot(rm)).act_cat) == len(actions)


def test_claws_confirm_action_can_be_scored_by_existing_onnx():
    pytest.importorskip("onnxruntime")
    if not MODEL.exists():
        pytest.skip(f"champion ONNX absent: {MODEL}")
    from ppo.onnx_policy import OnnxPolicy

    rm = _run(["STRIKE_IRONCLAD"] * 8, "CLAWS")
    actions = rm.available_actions()
    probs, _ = OnnxPolicy(str(MODEL)).distribution(_obs(rm), actions, metrics_snapshot(rm))
    assert len(probs) == len(actions)
    assert abs(sum(probs) - 1.0) < 1e-6
