"""Name enemies so a duplicated name still says WHICH one.

Three Two-Tailed Rats all render as "Two-Tailed Rat", so "Strike → Two-Tailed
Rat" does not say which rat to hit. A name shared by several enemies gets its
1-based position among the enemies sharing it, in list order (left to right
on the board): "Two-Tailed Rat #2". A unique name stays bare.

This is the same rule as the workbench's ``enemyLabeller``
(``web/src/enemyLabels.ts``), the Rust ``sts2-workbench`` ``enemy_labels.rs``
and the mod's under-enemy tag (``spire-mentor-mod/EnemyTagText.cs``, drawn by
``SimHud.cs``), so every surface numbers the same enemy the same way. Only
enemies still on the board are counted: an entry marked dead or escaped
(``is_dead`` / ``is_escaped`` / ``is_alive: false`` -- the game keeps a dying
monster in its list until its move resolves) keeps its bare name and takes no
number. So the numbers close up when one dies -- consistent at any moment, not
across the fight.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def off_board(enemy: dict[str, Any]) -> bool:
    """True for an entry the board no longer shows: dead or escaped."""
    return bool(enemy.get("is_dead") or enemy.get("is_escaped")) or enemy.get("is_alive") is False


def enemy_labels(enemies: list[dict[str, Any]]) -> list[str]:
    """One display label per entry of ``enemies``, in the same order."""
    names = [str(e.get("name") or e.get("id") or f"enemy {i}") for i, e in enumerate(enemies)]
    counted = [not off_board(e) for e in enemies]
    total = Counter(n for n, c in zip(names, counted, strict=True) if c)
    seen: Counter[str] = Counter()
    labels: list[str] = []
    for name, c in zip(names, counted, strict=True):
        if c and total[name] > 1:
            seen[name] += 1
            labels.append(f"{name} #{seen[name]}")
        else:
            labels.append(name)
    return labels


def enemy_label(enemies: list[dict[str, Any]], index: Any) -> str | None:
    """The label of ``enemies[index]``, or ``None`` when out of range."""
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(enemies):
        return None
    return enemy_labels(enemies)[index]
