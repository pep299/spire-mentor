"""``generate_loadout``: the random start of ``--random-loadout`` runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from parity_fuzz_scenario import (
    ALL_POTIONS,
    ALL_RELICS,
    EVENT_RELICS,
    _v2_card_pool,
    generate_loadout,
)

_WIKI_DATA = Path(__file__).resolve().parents[2] / "sts2-wiki" / "data"


def test_same_seed_same_loadout() -> None:
    assert generate_loadout("1000", "Silent") == generate_loadout("1000", "Silent")


def test_seed_and_character_both_vary_it() -> None:
    base = generate_loadout("1000", "Silent")
    assert generate_loadout("1001", "Silent") != base
    assert generate_loadout("1000", "Defect") != base


@pytest.mark.parametrize("character", ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"])
def test_loadout_draws_from_the_v2_pools(character: str) -> None:
    pool = set(_v2_card_pool(character))
    for seed in range(50):
        lo = generate_loadout(str(seed), character)
        assert {c.removesuffix("+1") for c in lo.deck} <= pool
        assert set(lo.relics) <= set(ALL_RELICS)
        assert not set(lo.relics) & EVENT_RELICS
        assert len(set(lo.relics)) == len(lo.relics)
        assert set(lo.potions) <= set(ALL_POTIONS)
        assert 0 <= lo.gold <= 999


@pytest.mark.parametrize("version", ["v0.107.1", "v0.111.0"])
def test_event_relics_are_the_event_rarity_ones(version: str) -> None:
    # An event grants these by name, so a grafted copy would become a second.
    path = _WIKI_DATA / version / "relics.json"
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    rarity = {r["loc_key"]: r["rarity"] for r in json.loads(path.read_text(encoding="utf-8"))}
    assert {r for r in ALL_RELICS if rarity[r] in ("Event", "Ancient")} == EVENT_RELICS
