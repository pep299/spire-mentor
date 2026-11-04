"""The advisor's shown win estimate is the calibrated search p_win."""

from __future__ import annotations

import pytest
import realgame_advisor as ra
from sts2_hint.calibration import calibrated_p_win
from sts2_hint.recommender import HintTag


def test_the_map_is_increasing_and_never_certain() -> None:
    xs = [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 0.999, 1.0]
    ys = [calibrated_p_win(x) for x in xs]
    assert ys == sorted(ys)
    assert all(0.0 < y < 1.0 for y in ys)


def test_a_low_raw_estimate_reads_higher() -> None:
    # The fitted shift: the playout net loses fights the search wins.
    assert calibrated_p_win(0.05) > 0.25
    assert calibrated_p_win(0.5) > 0.9
    assert calibrated_p_win(0.0) < 0.02


def _tag(stable_idx: int, win_prob: float, *, is_best: bool) -> HintTag:
    return HintTag(
        stable_idx=stable_idx,
        win_prob=win_prob,
        bucket_label="even",
        is_best=is_best,
        visits=100,
        q_saturated=False,
        low_confidence=False,
        target_index=None,
        q=2 * win_prob - 1,
        wins=1,
        losses=0,
        ties=0,
    )


def test_build_advice_shows_the_calibrated_estimate_and_keeps_the_raw_one() -> None:
    state = {"type": "combat_action", "hand": [{"id": "STRIKE_IRONCLAD"}] * 2, "potions": []}
    tags = [_tag(0, 0.08, is_best=True), _tag(1, 0.03, is_best=False)]

    advice = ra.build_advice(tags, state, sims=4000)

    assert advice["p_win_raw"] == pytest.approx(0.08)
    assert advice["p_win"] == pytest.approx(round(calibrated_p_win(0.08), 4))
    assert [o["p_win"] for o in advice["options"]] == pytest.approx([0.08, 0.03]), (
        "the rows stay raw"
    )
