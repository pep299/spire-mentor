"""Pure-python tests for sweep-v2 scenario assembly (no engines)."""

import pytest
from parity_audit import _FUZZ_SEEDS, assemble_v2_trials


def test_smoke_tier_is_tiny() -> None:
    trials = assemble_v2_trials(tier="smoke", axes=["upgraded", "fuzz"])
    assert len(trials) <= 5 + _FUZZ_SEEDS["smoke"]  # upgraded smoke-slice cap + fuzz seeds


def test_axes_filter() -> None:
    only_up = assemble_v2_trials(tier="full", axes=["upgraded"])
    assert all(sc.coverage_subject.startswith("card+:") for _, sc, _ in only_up if sc)


def test_full_upgraded_count_matches_pools() -> None:
    import parity_fuzz_scenario as m

    trials = assemble_v2_trials(tier="full", axes=["upgraded"])
    # the pools of the selected build (`--game-version`): v0.111.0 drops/adds cards
    pool = m.class_cards_for_build("Ironclad") + m.colorless_cards_for_build()
    assert len(trials) == len(pool)


def test_fuzz_tier_sizes() -> None:
    assert len(assemble_v2_trials(tier="quick", axes=["fuzz"])) == 500
    assert len(assemble_v2_trials(tier="full", axes=["fuzz"])) == 5000


def test_trials_are_seed_indexed_and_deterministic() -> None:
    a = assemble_v2_trials(tier="quick", axes=["fuzz"])
    b = assemble_v2_trials(tier="quick", axes=["fuzz"])
    assert [(s, sc) for s, sc, _ in a] == [(s, sc) for s, sc, _ in b]


def test_unknown_axis_rejected() -> None:
    with pytest.raises(SystemExit):
        assemble_v2_trials(tier="quick", axes=["bogus"])
