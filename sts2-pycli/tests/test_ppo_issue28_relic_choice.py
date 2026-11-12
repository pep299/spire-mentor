"""GH #28: optional Claws confirmation is an encoded PPO decision."""

from __future__ import annotations

from pathlib import Path

import pytest
from ppo import encoder, vocab
from ppo.env import assert_encodable, classify_surface

MODEL = Path(__file__).resolve().parents[1] / "data/ppo/ppo_fullrun_ironclad_cont300_it1150.onnx"


def _obs():
    return {
        "phase": "card_reward",
        "surface": "card-pick",
        "hp": 70,
        "max_hp": 80,
        "gold": 99,
        "deck": [("STRIKE_IRONCLAD", False), ("DEFEND_IRONCLAD", False)],
        "relics": ["CLAWS"],
        "potions": [],
        "map": None,
        "map_coord": None,
        "transform_select_deck_indices": [0, 1],
    }


def test_claws_confirm_is_recognized_and_uses_existing_skip_token():
    actions = [
        {"action": "transform_deck_card", "idx": 0},
        {"action": "transform_deck_card", "idx": 1},
        {"action": "confirm_deck_selection"},
    ]
    assert_encodable(actions, phase="card_reward")
    assert classify_surface("map_choice", actions) == "card-pick"
    assert encoder.fine_surface(actions) == "card-transform"
    assert encoder.fine_surface(actions[-1:]) == "card-transform"
    features = encoder.featurize_row(_obs(), actions, _obs())
    assert features.act_cat[-1, 6] == vocab.fixed_index("skip_card")


def test_existing_onnx_scores_claws_choices_without_shape_change():
    pytest.importorskip("onnxruntime")
    if not MODEL.exists():
        pytest.skip(f"published Ironclad ONNX absent: {MODEL}")
    from ppo.onnx_policy import OnnxPolicy

    actions = [
        {"action": "transform_deck_card", "idx": 0},
        {"action": "confirm_deck_selection"},
    ]
    probs, _ = OnnxPolicy(str(MODEL)).distribution(_obs(), actions, _obs())
    assert len(probs) == 2
    assert abs(sum(probs) - 1.0) < 1e-6
