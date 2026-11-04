"""`ppo.bc_human.tie_agree`: the per-arm picks and the paired agreement read (GH #1022)."""

from __future__ import annotations

import pytest
from ppo.bc_human.tie_agree import paired, tier_of

RULES = {
    "lost_position",
    "proven_win",
    "argmax",
    "band_visits",
    "block_guard",
    "free_play",
    "junk_dispose",
    "free_lasting",
    "order",
    "keep_potion",
    "spend",
    "order_block",
    "upgraded",
    "illegal_fallback",
}


def _row(file: str, pick: str, arm: str, human: str) -> dict:
    return {"file": file, "pick": pick, "arms": [arm], "human": human}


def test_paired_reads_only_the_decisions_where_the_arm_moves_the_pick():
    rows = [
        _row("a", "Strike", "Strike", "Strike"),  # same pick: no information
        _row("a", "EndTurn", "Defend", "Defend"),  # arm agrees, pilot does not
        _row("a", "EndTurn", "Defend", "Bash"),  # neither agrees
        _row("b", "Defend", "Strike", "Defend"),  # pilot agrees, arm does not
        _row("b", "EndTurn", "Defend", "Defend"),  # arm agrees
    ]
    s = paired(rows, 0)
    assert (s["flips"], s["arm"], s["pilot"]) == (4, 2, 1)
    assert s["diff"] == pytest.approx(0.25)
    # Clusters a = [+1, 0] and b = [-1, +1]: residuals 1 - 0.5 and 0 - 0.5.
    assert s["se"] == pytest.approx((0.25 + 0.25) ** 0.5 * 2**0.5 / 4)


def test_paired_without_flips_has_no_estimate():
    s = paired([_row("a", "Strike", "Strike", "Bash")], 0)
    assert s == {"flips": 0, "arm": 0, "pilot": 0, "diff": None, "se": None}


def test_tier_follows_the_encounter_suffix():
    assert tier_of("KNOWLEDGE_DEMON_BOSS") == "boss"
    assert tier_of("BYRDONIS_ELITE") == "elite"
    assert tier_of("NIBBITS_WEAK") == "common"


def test_tie_rule_picks_keys_every_arm_over_one_tree():
    sts2_rs = pytest.importorskip("sts2_rs")
    state = sts2_rs.build_scenario_state(
        5,
        "Ironclad",
        ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 5,
        80,
        80,
        [],
        "SHRINKER_BEETLE_WEAK",
    )
    arms = ["none", "block+free+junk"]
    res = sts2_rs.tie_rule_picks(state, "common", 64, 1, arms, human={"action": "end_turn"})
    assert res["legal"] >= 2
    assert res["human"] == "EndTurn"
    assert len(res["arms"]) == 2
    assert all(a["rule"] in RULES for a in res["arms"])
    # The deployed rule set over the same tree is the pilot's own pick.
    assert res["arms"][1]["key"] == res["pick"]["key"]
    with pytest.raises(ValueError, match="tier"):
        sts2_rs.tie_rule_picks(state, "hallway", 8, 1, arms)
    with pytest.raises(ValueError, match="unknown tie rule"):
        sts2_rs.tie_rule_picks(state, "common", 8, 1, ["blok"])
