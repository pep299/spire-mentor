"""Pins for the pending-choice tag encoding + the mirrored-select picks.

Root cause pinned here (mirror_d1 corpus, seeds 1004/1047, the entire
"post-Gambling-Chip-mulligan hand short by 1" family):

* The engine's multi-pick root enumerates one incremental pick per option
  PLUS one empty-selection FINALIZE/SKIP (`sts2-core legal_actions`), and
  its wire dict deliberately collapses the skip's back-compat ``index`` to
  0 (``crates/sts2-py/src/lib.rs``: "the full selection list disambiguates
  the empty-selection SKIP from picking option 0").
* ``_stable_idx_for_action`` read only ``index``, so the ranked skip became
  a phantom "choose option 0" — a DUPLICATE index in the world's
  ``select_cards`` (e.g. ``[1,3,0,0,2]``).
* The DLL's ``ResolvePendingByIndices`` (RunSimulator.cs:7119-7127) keeps
  duplicates and resolves by ORIGINAL index, so Gambling Chip's
  ``CardCmd.DiscardAndDraw(list, list.Count)`` (GamblingChip.cs:20-23,
  v0.107.1 decomp) received a 5-entry list naming only 4 distinct cards:
  4 discarded, 5 drawn. The engine's
  ``resolve_discard_from_hand_and_draw`` skips the already-discarded
  duplicate and draws per ACTUAL discard: 4 discarded, 4 drawn. Hence the
  world's post-mulligan hand was one card long vs the mirror, every time.

These tests need no ``sts2_rs`` wheel — they pin the pure-python mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from sts2_hint.recommender import (
    CHOOSE_SKIP_STABLE_IDX,
    CHOOSE_STABLE_IDX_BASE,
    END_TURN_STABLE_IDX,
    FINALIZE,
    HintTag,
    _stable_idx_for_action,
    choice_picks_from_tags,
    next_choice_pick_from_tags,
)


def _tag(stable_idx: int, win_prob: float, is_best: bool = False) -> HintTag:
    return HintTag(
        stable_idx=stable_idx,
        win_prob=win_prob,
        bucket_label="x",
        is_best=is_best,
    )


def _choose(i: int) -> int:
    return CHOOSE_STABLE_IDX_BASE - i


class TestStableIdxForChoose:
    def test_empty_selection_is_the_skip_sentinel_not_option_zero(self) -> None:
        # The engine emits the finalize as {"index": 0, "selection": []} —
        # index alone is ambiguous by design; `selection` disambiguates.
        action = {"action": "choose", "index": 0, "selection": []}
        assert _stable_idx_for_action(action, hand=[]) == CHOOSE_SKIP_STABLE_IDX

    def test_selection_leading_index_wins_over_index_key(self) -> None:
        action = {"action": "choose", "index": 2, "selection": [2]}
        assert _stable_idx_for_action(action, hand=[]) == _choose(2)

    def test_index_only_back_compat(self) -> None:
        assert _stable_idx_for_action({"action": "choose", "index": 1}, hand=[]) == _choose(1)

    def test_skip_sentinel_is_outside_the_choose_range(self) -> None:
        # Both mirrored drivers filter option tags with
        # `stable_idx <= CHOOSE_STABLE_IDX_BASE`; the skip must never pass
        # that filter and never collide with a real option.
        assert CHOOSE_SKIP_STABLE_IDX > CHOOSE_STABLE_IDX_BASE


class TestChoicePicksFromTags:
    def test_gambling_chip_regression_no_duplicate_no_phantom(self) -> None:
        # Seed-1004 record 92 shape: 5 options, choose-any (min 0 / max 5).
        # Search ranks: opt1, opt3, SKIP, opt0, opt2, opt4. The old encoding
        # turned SKIP into a phantom 0 and emitted [1,3,0,0,2] — the
        # duplicate that made the DLL draw 5 for 4 discards.
        tags = [
            _tag(_choose(1), 0.9, is_best=True),
            _tag(_choose(3), 0.8),
            _tag(CHOOSE_SKIP_STABLE_IDX, 0.7),
            _tag(_choose(0), 0.6),
            _tag(_choose(2), 0.5),
            _tag(_choose(4), 0.4),
        ]
        picks = choice_picks_from_tags(tags, n_options=5, min_select=0, max_select=5)
        assert picks == [1, 3], "the ranked skip is a stop signal, not a pick"
        assert len(picks) == len(set(picks))

    def test_skip_ranked_last_takes_all_options(self) -> None:
        tags = [_tag(_choose(i), 0.9 - i * 0.1) for i in range(3)]
        tags.append(_tag(CHOOSE_SKIP_STABLE_IDX, 0.1))
        assert choice_picks_from_tags(tags, n_options=3, min_select=0, max_select=3) == [0, 1, 2]

    def test_skip_best_on_skippable_screen_means_skip(self) -> None:
        tags = [
            _tag(CHOOSE_SKIP_STABLE_IDX, 0.9, is_best=True),
            _tag(_choose(0), 0.5),
        ]
        assert choice_picks_from_tags(tags, n_options=1, min_select=0, max_select=1) == []

    def test_skip_before_minimum_stops_as_soon_as_legal(self) -> None:
        # min 2: an early skip is not yet legal — collect to the minimum,
        # then stop there instead of running on to max.
        tags = [
            _tag(CHOOSE_SKIP_STABLE_IDX, 0.9, is_best=True),
            _tag(_choose(2), 0.8),
            _tag(_choose(0), 0.7),
            _tag(_choose(1), 0.6),
        ]
        assert choice_picks_from_tags(tags, n_options=3, min_select=2, max_select=3) == [2, 0]

    def test_max_select_caps_the_picks(self) -> None:
        tags = [_tag(_choose(i), 0.9 - i * 0.1) for i in range(5)]
        assert choice_picks_from_tags(tags, n_options=5, min_select=0, max_select=2) == [0, 1]

    def test_out_of_range_and_duplicate_tags_are_dropped(self) -> None:
        tags = [
            _tag(_choose(7), 0.9),  # beyond the screen's options
            _tag(_choose(1), 0.8),
            _tag(_choose(1), 0.7),  # duplicate row (per-target fan-out)
            _tag(-1, 0.6),  # end_turn tag must never leak in
        ]
        assert choice_picks_from_tags(tags, n_options=3, min_select=0, max_select=3) == [1]

    def test_minimum_is_padded_when_tags_run_dry(self) -> None:
        assert choice_picks_from_tags([], n_options=3, min_select=2, max_select=3) == [0, 1]


class TestNextChoicePickFromTags:
    """The one-step reader the sequential multi-pick walk uses.

    `choice_picks_from_tags` reads a whole answer off ONE root ranking; this
    reads only the next pick so `CombatMirror.search_choice_picks` can apply
    it and re-search — the engine's own decomposition (bead `sts2-rs-sopgf`).
    """

    def test_returns_the_best_option(self) -> None:
        tags = [_tag(_choose(2), 0.9, is_best=True), _tag(_choose(0), 0.5)]
        assert next_choice_pick_from_tags(tags, n_options=3, taken=[]) == 2

    def test_skips_options_already_taken(self) -> None:
        tags = [_tag(_choose(2), 0.9, is_best=True), _tag(_choose(0), 0.5)]
        assert next_choice_pick_from_tags(tags, n_options=3, taken=[2]) == 0

    def test_finalize_ranked_first_is_the_stop_signal(self) -> None:
        tags = [_tag(CHOOSE_SKIP_STABLE_IDX, 0.9, is_best=True), _tag(_choose(0), 0.5)]
        assert next_choice_pick_from_tags(tags, n_options=3, taken=[]) is FINALIZE

    def test_finalize_is_ignored_while_stopping_is_illegal(self) -> None:
        # Same rule as `choice_picks_from_tags`: below `min_select` the caller
        # passes allow_finalize=False and gets the best remaining option.
        tags = [_tag(CHOOSE_SKIP_STABLE_IDX, 0.9, is_best=True), _tag(_choose(1), 0.5)]
        assert next_choice_pick_from_tags(tags, n_options=3, taken=[], allow_finalize=False) == 1

    def test_no_usable_row_returns_none(self) -> None:
        assert next_choice_pick_from_tags([], n_options=3, taken=[]) is None
        # Every option already taken, and no finalize row offered.
        tags = [_tag(_choose(0), 0.9)]
        assert next_choice_pick_from_tags(tags, n_options=3, taken=[0]) is None

    def test_out_of_range_rows_are_dropped(self) -> None:
        tags = [_tag(_choose(7), 0.9), _tag(-1, 0.8), _tag(_choose(1), 0.7)]
        assert next_choice_pick_from_tags(tags, n_options=3, taken=[]) == 1


# ---------------------------------------------------------------------------
# GH #360 — an unmappable ranked row must not cost the whole ranking.
# ---------------------------------------------------------------------------


def _entry(action: dict, *, p_win: float, is_best: bool = False) -> dict:
    return {
        "action": action,
        "p_win": p_win,
        "is_best": is_best,
        "visits": 10,
    }


def test_unmappable_play_row_is_dropped_not_fatal() -> None:
    """A play indexing past the world's hand drops THAT row only.

    2026-08-03 live corpus: 151 of 151 such rows landed on the decision right
    after a ``FromChooseACardScreen`` choose, where ``CombatMirror.reconcile``
    defers grading for one decision — so the search legitimately ran on a hand
    the world had not confirmed, and 150 of the 151 were exactly one card long.
    Aborting the ranking collapsed each of those decisions to the uniform
    random-legal policy; the other ranked actions were always fine.
    """
    from sts2_hint.recommender import tags_from_result

    hand = [{"stable_idx": 7}, {"stable_idx": 9}]
    result = {
        "actions": [
            # The phantom card: index 2 into a 2-card hand.
            _entry({"action": "play", "card_index": 2}, p_win=0.9, is_best=True),
            _entry({"action": "play", "card_index": 1}, p_win=0.6),
            _entry({"action": "end_turn"}, p_win=0.3),
        ]
    }
    tags = tags_from_result(result, hand)
    assert [t.stable_idx for t in tags] == [9, END_TURN_STABLE_IDX], (
        "the two addressable rows survive, best-first"
    )
    assert all(t.win_prob < 0.9 for t in tags), "the phantom row (the best one) is gone"


def test_every_row_unmappable_yields_an_empty_ranking() -> None:
    """All rows unmappable is still 'no legal actions' — the caller's own
    ``if not tags: raise`` keeps handling it, so nothing is silently invented.
    """
    from sts2_hint.recommender import tags_from_result

    result = {"actions": [_entry({"action": "play", "card_index": 3}, p_win=0.9)]}
    assert tags_from_result(result, [{"stable_idx": 1}]) == []


def test_unknown_action_kind_still_raises() -> None:
    """Drift in the action vocabulary is NOT recoverable and must still raise —
    only the out-of-range case is downgraded to a dropped row.
    """
    import pytest
    from sts2_hint.recommender import tags_from_result

    with pytest.raises(KeyError):
        tags_from_result({"actions": [_entry({"action": "teleport"}, p_win=0.5)]}, [])


def test_unmappable_engine_pick_crowns_the_engines_next_choice() -> None:
    """GH #1022: when the dropped row was the engine's pick, the crown goes to
    its own next choice (lowest ``pick_rank``), not to the best ``p_win``."""
    from sts2_hint.recommender import tags_from_result

    hand = [{"stable_idx": 7}, {"stable_idx": 9}]
    result = {
        "actions": [
            {
                **_entry({"action": "play", "card_index": 2}, p_win=0.9, is_best=True),
                "pick_rank": 0,
            },
            {**_entry({"action": "play", "card_index": 1}, p_win=0.6), "pick_rank": 2},
            {**_entry({"action": "end_turn"}, p_win=0.3), "pick_rank": 1, "in_band": True},
        ]
    }
    tags = tags_from_result(result, hand)
    assert [(t.stable_idx, t.is_best) for t in tags] == [(END_TURN_STABLE_IDX, True), (9, False)]
    assert tags[0].pick_rank == 1 and tags[0].in_band
