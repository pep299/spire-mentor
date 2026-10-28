"""Duplicated enemy names say WHICH enemy, on every workbench label.

Three Two-Tailed Rats all render as "Two-Tailed Rat"; every label naming an
enemy numbers a shared name by its rank among the board's live enemies in slot
order ("Two-Tailed Rat #2"), a unique name stays bare. These pin the Python
mirrors of the Rust ``sts2-workbench`` labels (``enemy_labels.rs``) to the same
strings — the Rust twins assert the identical cases.
"""

from __future__ import annotations

from enemy_labels import enemy_label, enemy_labels
from simlab.position import action_label
from workbench.regret_review import _targeted_label
from workbench.render import describe_action
from workbench.statediff import diff_states

RAT = "Two-Tailed Rat"


def test_enemy_labels_numbers_shared_names_only() -> None:
    es = [{"name": RAT}, {"name": "Louse"}, {"name": RAT}, {"name": RAT}]
    assert enemy_labels(es) == [f"{RAT} #1", "Louse", f"{RAT} #2", f"{RAT} #3"]
    assert enemy_label(es, 2) == f"{RAT} #2"
    assert enemy_label(es, 4) is None


# Slot 2 died, so the view lists slots 0, 1, 3, 4: rats #1..#3 + a Louse.
_BOARD = {
    "hand": [{"index": 0, "name": "Strike"}],
    "player": {"potions": [{"index": 0, "name": "Fire Potion"}]},
    "enemies": [
        {"index": 0, "name": RAT},
        {"index": 1, "name": "Louse"},
        {"index": 3, "name": RAT},
        {"index": 4, "name": RAT},
    ],
}


def test_describe_action_numbers_the_target() -> None:
    def play(t: int) -> str:
        args = {"card_index": 0, "target_index": t}
        return describe_action(_BOARD, {"cmd": "action", "action": "play_card", "args": args})

    assert play(0) == f"Play Strike → {RAT} #1"
    assert play(1) == "Play Strike → Louse"
    assert play(3) == f"Play Strike → {RAT} #2"
    assert play(4) == f"Play Strike → {RAT} #3"
    drink = {
        "cmd": "action",
        "action": "use_potion",
        "args": {"potion_index": 0, "target_index": 4},
    }
    assert describe_action(_BOARD, drink) == f"Drink Fire Potion → {RAT} #3"
    # Slot 2 is not on the board: no label, not the slot-3 rat at list
    # position 2.
    assert play(2) == "Play Strike → enemy 2"


def test_describe_action_falls_back_to_position_without_indices() -> None:
    board = {**_BOARD, "enemies": [{"name": RAT}, {"name": RAT}]}
    args = {"card_index": 0, "target_index": 1}
    act = {"cmd": "action", "action": "play_card", "args": args}
    assert describe_action(board, act) == f"Play Strike → {RAT} #2"


def test_regret_targeted_label_numbers_the_target() -> None:
    enemies = _BOARD["enemies"][:3]

    def row(t: int) -> dict:
        return {"action": "Strike", "target_index": t}

    assert _targeted_label(row(0), enemies) == f"Strike → {RAT} #1"
    assert _targeted_label(row(1), enemies) == "Strike → Louse"
    assert _targeted_label(row(3), enemies) == f"Strike → {RAT} #2"
    assert _targeted_label(row(2), enemies) == "Strike"


def _enemy(index: int, hp: int, powers: list | None = None) -> dict:
    e = {"index": index, "name": RAT, "hp": hp, "max_hp": 50, "block": 0}
    if powers is not None:
        e["powers"] = powers
    return e


def test_statediff_labels_from_the_board_each_enemy_is_on() -> None:
    a = {
        "player": {"hp": 70, "block": 0},
        "enemies": [_enemy(0, 20), _enemy(1, 20), _enemy(2, 20, [{"name": "Weak", "amount": 1}])],
    }
    # Rat #1 dies and leaves the list, rat #2 loses HP, rat #3 its Weak, and a
    # fourth rat spawns into slot 3.
    b = {
        "player": {"hp": 70, "block": 0},
        "enemies": [_enemy(1, 14), _enemy(2, 20, []), _enemy(3, 20)],
    }
    assert diff_states(a, b) == [
        f"{RAT} #1 HP 20→0  ☠ defeated",
        f"{RAT} #2 HP 20→14 (−6)",
        f"{RAT} #3 −Weak",
        # On the AFTER board it is the third rat.
        f"Spawned: {RAT} #3",
    ]


def test_simlab_action_label_targets_by_slot() -> None:
    # Slot 1 is dead: the engine's target is the encounter SLOT, so target 2
    # is the second live rat — the alive-filtered list's [2] was the Louse.
    wire = {
        "deck": {"hand": [{"id": "STRIKE_R"}]},
        "encounter": [
            {"kind": "TWO_TAILED_RAT"},
            {"kind": "TWO_TAILED_RAT", "is_dead": True},
            {"kind": "TWO_TAILED_RAT"},
            {"kind": "LOUSE"},
        ],
    }

    def label(t: int) -> str:
        return action_label({"action": "play", "card_index": 0, "target_index": t}, wire)

    assert label(0) == "Play Strike R → Two Tailed Rat #1"
    assert label(2) == "Play Strike R → Two Tailed Rat #2"
    assert label(3) == "Play Strike R → Louse"
    assert label(1) == "Play Strike R → enemy 1"
    # A numeric-string target resolves like the int.
    assert label("2") == "Play Strike R → Two Tailed Rat #2"  # type: ignore[arg-type]


def test_simlab_action_label_skips_escaped_enemies() -> None:
    # The sandbox board drops an escaped Thief, so the one left is bare.
    wire = {
        "deck": {"hand": [{"id": "STRIKE_R"}]},
        "encounter": [{"kind": "THIEF", "is_escaped": True}, {"kind": "THIEF"}],
    }
    act = {"action": "play", "card_index": 0, "target_index": 1}
    assert action_label(act, wire) == "Play Strike R → Thief"
