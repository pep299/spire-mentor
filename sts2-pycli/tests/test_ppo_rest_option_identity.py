"""GH #24: rest-site PPO options are encoded by action ID, never list position."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import sts2_rs
from ppo import encoder, vocab
from ppo.env import metrics_snapshot, observe

SEED = 984001164
MODEL = Path(__file__).resolve().parents[1] / "data/ppo/ppo_fullrun_ironclad_cont300_it1150.onnx"


def _token(option_ids: list[str], idx: int) -> list[int]:
    obs = {
        "phase": "rest_site",
        "surface": "rest",
        "rest_option_labels": option_ids,
        "hp": 60,
        "max_hp": 80,
        "gold": 0,
        "deck": [],
        "relics": [],
        "potions": [],
        "map": None,
        "map_coord": None,
    }
    actions = [{"action": "rest_option", "idx": idx}]
    feat = encoder.featurize_row(obs, actions, obs)
    return feat.act_cat[0].tolist()


@pytest.mark.parametrize(
    "option_id,kind,stable_idx",
    [
        ("HEAL", "rest_heal", 0),
        ("SMITH", "rest_smith", 1),
        ("DIG", "rest_other", 0),
        ("LIFT", "rest_other", 1),
        ("COOK", "rest_other", 2),
        ("CLONE", "rest_other", 3),
        ("HATCH", "rest_other", 4),
        ("KINDLE", "rest_other", 5),
        ("MEND", "rest_other", 6),
        ("FUTURE_OPTION", "rest_other", 7),
    ],
)
def test_all_rest_option_ids_are_distinguishable(option_id, kind, stable_idx):
    cat = _token([option_id], 0)
    assert cat[6] == vocab.fixed_index(kind)
    assert cat[7] == stable_idx


def test_reindexed_heal_and_smith_keep_their_original_onnx_tokens():
    assert _token(["HEAL", "SMITH"], 1) == _token(["SMITH"], 0)
    assert _token(["HEAL", "SMITH"], 0) == _token(["HEAL"], 0)
    # Old episode rows remain readable and unchanged on the ordinary site.
    assert _token(["rest", "smith"], 0) == _token(["HEAL"], 0)
    assert _token(["rest", "smith"], 1) == _token(["SMITH"], 0)
    assert _token([], 0) == _token(["HEAL"], 0)
    assert _token([], 1) == _token(["SMITH"], 0)


def _enter_rest_site(*relic_ids: str):
    rm = sts2_rs.RunManager(
        seed=SEED,
        character="Ironclad",
        ascension_level=10,
        game_version="v0.111.0",
    )
    rm.debug_set_loadout(relics=[*rm.relics(), *relic_ids])
    rows = rm.full_map()["rows"]
    nodes = {(node["col"], node["row"]): node for row in rows for node in row}
    for row in rows:
        for parent in row:
            for child in parent["children"]:
                dest = nodes.get((child["col"], child["row"]))
                if dest is None or dest["type"] != "RestSite":
                    continue
                rm.debug_place_on_map(
                    1, parent["col"], parent["row"], [], parent["row"] + 1, parent["row"] + 1
                )
                move = next(
                    (
                        action
                        for action in rm.available_actions()
                        if action["action"] == "move" and action["col"] == child["col"]
                    ),
                    None,
                )
                if move is not None:
                    rm.take_action(move)
                    assert rm.phase() == "rest_site"
                    return rm
    pytest.fail(f"seed {SEED}: no reachable rest-site edge on the act-1 map")


def _assert_live_obs_ids(rm):
    options = rm.rest_site_options()
    actions = rm.available_actions()
    observed = observe(rm, rm.phase(), "rest", seed=SEED, ascension=10)
    rust_obs = json.loads(sts2_rs.ppo_observe_json(rm, rm.phase(), "rest", SEED, 10))
    ids = [option_id for option_id, _enabled in options]
    assert observed["rest_option_labels"] == ids
    assert rust_obs["rest_option_labels"] == ids
    for action in actions:
        if action["action"] == "rest_option":
            idx = action["idx"]
            assert options[idx][1], (action, options)
            expected = _token(ids, idx)
            encoded = encoder.featurize_row(observed, [action], metrics_snapshot(rm))
            assert encoded.act_cat[0].tolist() == expected
    return options


def test_miniature_tent_second_pick_is_smith_at_index_zero():
    rm = _enter_rest_site("MINIATURE_TENT", "SHOVEL", "GIRYA")
    initial = _assert_live_obs_ids(rm)
    assert initial[0][0] == "HEAL"
    assert initial[1][0] == "SMITH"
    assert {"DIG", "LIFT"}.issubset({option_id for option_id, _ in initial})

    rm.set_player(hp=40)
    rm.take_action({"action": "rest_option", "idx": 0})
    assert rm.phase() == "rest_site"
    after = _assert_live_obs_ids(rm)
    assert after[0][0] == "SMITH"
    assert "HEAL" not in [option_id for option_id, _ in after]
    assert _token([option_id for option_id, _ in after], 0) == _token(
        [option_id for option_id, _ in initial], 1
    )

    # The moved idx=0 really opens Smith, not Heal.
    rm.take_action({"action": "rest_option", "idx": 0})
    assert any(a["action"] == "smith_card" for a in rm.available_actions())


def test_miniature_tent_after_smith_exposes_heal_at_index_zero():
    rm = _enter_rest_site("MINIATURE_TENT")
    _assert_live_obs_ids(rm)
    rm.take_action({"action": "rest_option", "idx": 1})
    smith = next(a for a in rm.available_actions() if a["action"] == "smith_card")
    rm.take_action(smith)
    assert rm.phase() == "rest_site"
    options = _assert_live_obs_ids(rm)
    assert [option_id for option_id, _ in options] == ["HEAL"]
    assert _token(["HEAL"], 0) == _token(["HEAL", "SMITH"], 0)


def test_unchanged_onnx_can_score_reindexed_rest_site():
    pytest.importorskip("onnxruntime")
    if not MODEL.exists():
        pytest.skip(f"published Ironclad ONNX absent: {MODEL}")
    from ppo.onnx_policy import OnnxPolicy

    rm = _enter_rest_site("MINIATURE_TENT")
    rm.take_action({"action": "rest_option", "idx": 0})
    assert rm.phase() == "rest_site"
    obs = observe(rm, rm.phase(), "rest", seed=SEED, ascension=10)
    actions = rm.available_actions()
    assert any(a["action"] == "rest_option" and a["idx"] == 0 for a in actions)
    probs, _ = OnnxPolicy(str(MODEL)).distribution(obs, actions, metrics_snapshot(rm))
    assert len(probs) == len(actions)
    assert abs(sum(probs) - 1.0) < 1e-6
