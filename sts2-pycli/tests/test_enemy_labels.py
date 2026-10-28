"""``enemy_labels``: a shared enemy name says which one it is."""

from __future__ import annotations

from enemy_labels import enemy_label, enemy_labels


def test_shared_names_are_numbered_in_list_order() -> None:
    enemies = [{"name": "Louse"}, {"name": "Cultist"}, {"name": "Louse"}]
    assert enemy_labels(enemies) == ["Louse #1", "Cultist", "Louse #2"]


def test_unique_names_stay_bare() -> None:
    assert enemy_labels([{"name": "Jaw Worm"}]) == ["Jaw Worm"]


def test_id_then_index_fallbacks() -> None:
    assert enemy_labels([{"id": "RAT"}, {}]) == ["RAT", "enemy 1"]


def test_out_of_range_or_non_int_index_is_none() -> None:
    enemies = [{"name": "Rat"}, {"name": "Rat"}]
    assert enemy_label(enemies, 1) == "Rat #2"
    assert enemy_label(enemies, 2) is None
    assert enemy_label(enemies, -1) is None
    assert enemy_label(enemies, None) is None
    assert enemy_label(enemies, True) is None


def test_off_board_entries_take_no_number() -> None:
    enemies = [
        {"name": "Rat"},
        {"name": "Rat", "is_alive": False},
        {"name": "Rat", "is_escaped": True},
        {"name": "Rat", "is_dead": True},
        {"name": "Rat"},
    ]
    assert enemy_labels(enemies) == ["Rat #1", "Rat", "Rat", "Rat", "Rat #2"]
    assert enemy_labels([{"name": "Rat"}, {"name": "Rat", "is_alive": False}]) == ["Rat", "Rat"]
