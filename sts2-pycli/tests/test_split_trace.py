"""Tests for split_trace — explaining a full-state split with the C# state trace (GH #1088)."""

from __future__ import annotations

from split_trace import explain, suspects


def _segments() -> list[dict]:
    """One ``play_card`` segment as the oracle's ``get_trace`` returns it."""
    return [
        {
            "seq": 812,
            "action": "play_card",
            "args": "card_index=2,target_index=0",
            "events": [
                {"k": "energy", "who": "player", "old": 3, "new": 2, "src": ""},
                {
                    "k": "pile_remove",
                    "pile": "Hand",
                    "id": "CARD.STRIKE_IRONCLAD",
                    "idx": 2,
                    "src": "",
                },
                {
                    "k": "pile_add",
                    "pile": "Play",
                    "id": "CARD.STRIKE_IRONCLAD",
                    "idx": 0,
                    "src": "",
                },
                {
                    "k": "hp",
                    "who": "monsters[0]",
                    "old": 100,
                    "new": 94,
                    "src": "StrikeIronclad.OnPlay",
                },
                {
                    "k": "hp",
                    "who": "monsters[0]",
                    "old": 94,
                    "new": 88,
                    "src": "Hellraiser.AfterCardDrawnEarly>StrikeIronclad.OnPlay",
                },
                {"k": "rng", "who": "shuffle", "new": 13, "src": ""},
                {"k": "rng", "who": "shuffle", "new": 14, "src": ""},
                {"k": "rng", "who": "shuffle", "new": 15, "src": ""},
                {
                    "k": "power",
                    "who": "monsters[1]",
                    "id": "VULNERABLE_POWER",
                    "old": 0,
                    "new": 2,
                    "src": "Bash.OnPlay",
                },
                {
                    "k": "pile_remove",
                    "pile": "Play",
                    "id": "CARD.STRIKE_IRONCLAD",
                    "idx": 0,
                    "src": "",
                },
                {
                    "k": "pile_add",
                    "pile": "Discard",
                    "id": "CARD.STRIKE_IRONCLAD",
                    "idx": 3,
                    "src": "",
                },
            ],
        }
    ]


def test_creature_field_history_names_each_source():
    lines = explain(_segments(), [{"path": "monsters[0].hp", "csharp": 88, "rust": 94}])
    assert lines[0] == "monsters[0].hp: cs=88 rs=94"
    assert lines[1].endswith("100 -> 94  StrikeIronclad.OnPlay")
    assert lines[2].endswith("94 -> 88  Hellraiser.AfterCardDrawnEarly>StrikeIronclad.OnPlay")
    assert lines[1].startswith("  [812 play_card(card_index=2,target_index=0)]")
    assert len(lines) == 4


def test_paths_sharing_a_pile_print_its_history_once():
    divs = [
        {"path": "discard_count", "csharp": 4, "rust": 3},
        {"path": "piles.discard.len", "csharp": 4, "rust": 3},
        {"path": "piles.discard[3]", "csharp": "STRIKE_IRONCLAD", "rust": None},
    ]
    lines = explain(_segments(), divs)[:-1]
    assert lines == [
        "discard_count: cs=4 rs=3",
        "  [812 play_card(card_index=2,target_index=0)] +STRIKE_IRONCLAD @3  (engine)",
    ]


def test_rng_draws_collapse_into_one_row():
    lines = explain(_segments(), [{"path": "rng.shuffle", "csharp": 15, "rust": 12}])
    assert lines[1].endswith("3 draws -> cursor 13..15  (engine)")
    assert len(lines) == 3


def test_powers_and_untraced_fields():
    divs = [
        {"path": "monsters[1].powers.len", "csharp": 1, "rust": 0},
        {"path": "orbs[0].kind", "csharp": "Frost", "rust": "Lightning"},
    ]
    lines = explain(_segments(), divs)
    assert lines[1].endswith("VULNERABLE_POWER 0 -> 2  Bash.OnPlay")
    assert lines[2] == "orbs[0].kind: cs='Frost' rs='Lightning'  (not traced)"


def test_field_rust_changed_but_csharp_did_not():
    lines = explain(_segments(), [{"path": "player.block", "csharp": 0, "rust": 3}])
    assert lines[1].startswith("  (no C# change this step")
    # The step's C# sources, for hunting the hook Rust fired where C# did not.
    assert lines[2] == (
        "C# sources this step: StrikeIronclad.OnPlay, "
        "Hellraiser.AfterCardDrawnEarly>StrikeIronclad.OnPlay, Bash.OnPlay"
    )


def test_suspects_are_innermost_sources_of_diverging_fields():
    divs = [
        {"path": "monsters[0].hp", "csharp": 88, "rust": 94},
        {"path": "monsters[1].powers[0].amount", "csharp": 2, "rust": 1},
    ]
    assert suspects(_segments(), divs) == ["StrikeIronclad.OnPlay", "Bash.OnPlay"]
