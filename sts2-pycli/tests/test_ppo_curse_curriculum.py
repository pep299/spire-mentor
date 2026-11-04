"""GH #977: the training-only curse curriculum and its removal diagnostic."""

from __future__ import annotations

import pytest
from ppo import trainer, workers


def test_curse_removal_is_none_when_no_curse_is_offered() -> None:
    obs = {"card_removal_options": [["STRIKE_IRONCLAD", False], ["DEFEND_IRONCLAD", False]]}
    actions = [{"action": "remove_deck_card", "idx": 0}, {"action": "remove_deck_card", "idx": 1}]
    assert workers.curse_removal(obs, actions, 0) is None


def test_curse_removal_flags_offered_and_taken_on_both_surfaces() -> None:
    curse = sorted(workers._removable_curses())[0]
    removal = {"card_removal_options": [["DEFEND_IRONCLAD", False], [f"CARD.{curse}", False]]}
    acts = [{"action": "remove_deck_card", "idx": 0}, {"action": "remove_deck_card", "idx": 1}]
    assert workers.curse_removal(removal, acts, 0) == 0
    assert workers.curse_removal(removal, acts, 1) == 1
    shop = {"deck": [["STRIKE_IRONCLAD", False], [curse, False]]}
    shop_acts = [
        {"action": "leave_shop"},
        {"action": "shop_remove_card", "deck_idx": 0},
        {"action": "shop_remove_card", "deck_idx": 1},
    ]
    assert workers.curse_removal(shop, shop_acts, 0) == 0  # leaving counts as not taken
    assert workers.curse_removal(shop, shop_acts, 2) == 1


def test_curse_removal_matches_the_decks_pascal_case_ids() -> None:
    """The registry says `POOR_SLEEP`, the deck says `PoorSleep`: a plain
    upper-case compare misses every multi-word curse."""
    assert "POORSLEEP" in workers._removable_curses()
    shop = {"deck": [["StrikeIronclad", False], ["PoorSleep", False]]}
    acts = [{"action": "leave_shop"}, {"action": "shop_remove_card", "deck_idx": 1}]
    assert workers.curse_removal(shop, acts, 1) == 1


def test_the_pool_is_removable_curses_only() -> None:
    import sts2_rs

    pool = set(sts2_rs.removable_curse_cards())
    assert pool, "no removable curse in the engine"
    # Eternal curses never reach a removal screen.
    assert not pool & set(sts2_rs.non_removable_cards())
    assert "ASCENDERS_BANE" not in pool
    assert {"INJURY", "CLUMSY", "REGRET"} <= pool


@pytest.mark.parametrize("seed", [3, 11])
def test_injection_is_deterministic_and_off_by_default(seed: int) -> None:
    from ppo.env import PpoRunEnv

    base = PpoRunEnv(seed=seed, ascension=10)
    assert base.injected_curse is None
    a = PpoRunEnv(seed=seed, ascension=10, curse_inject_p=1.0)
    b = PpoRunEnv(seed=seed, ascension=10, curse_inject_p=1.0)
    assert a.injected_curse is not None and a.injected_curse == b.injected_curse
    want = workers._bare_id(a.injected_curse)
    deck = [workers._bare_id(c) for c in a.rm.master_deck()]
    assert deck.count(want) == 1 + [workers._bare_id(c) for c in base.rm.master_deck()].count(want)


def test_behavior_stats_always_emits_the_curse_keys() -> None:
    rows = [
        {
            "injected_curse": "INJURY",
            "transitions": [
                {"surface": "card-pick", "fine_surface": "card-remove", "curse_rm": 1},
                {"surface": "card-pick", "fine_surface": "card-reward", "bundle": True},
                {"surface": "shop", "fine_surface": "shop", "curse_rm": 0},
                {"surface": "shop", "fine_surface": "shop", "curse_rm": None},
            ],
        },
        {"injected_curse": None, "transitions": []},
    ]
    out = trainer.behavior_stats(rows)
    assert out["curse_rm/card-remove_offered_per_run"] == pytest.approx(0.5)
    assert out["curse_rm/card-remove_taken_frac"] == pytest.approx(1.0)
    assert out["curse_rm/shop_offered_per_run"] == pytest.approx(0.5)
    assert out["curse_rm/shop_taken_frac"] == pytest.approx(0.0)
    assert out["curse_rm/injected_frac"] == pytest.approx(0.5)
    assert out["dec_per_run/card-bundle"] == pytest.approx(0.5)
    empty = trainer.behavior_stats([{"transitions": []}])
    assert empty["curse_rm/shop_taken_frac"] == pytest.approx(0.0)  # present, not absent
