"""Crystal Sphere: which cell to divine (GH #872).

The minigame (``CrystalSphereMinigame.cs``) hides fifteen items on an 11x11 grid
and reveals an item only once EVERY cell it covers is clear. The mod never
changes the tool, so each ``divine_cell`` is the default Big tool: the clicked
cell and its eight neighbours. The game enables proceed only once the
divinations are spent (``NCrystalSphereScreen``: ``if (_entity.IsFinished)``),
so stopping early is never a choice -- only which cells to clear. A cell click
after that is a no-op, which is why proceed always wins when it is offered:
picking a cell there just loops the screen.

The bridge sends only the hidden cells' coordinates, not the board's contents,
so the rule is geometric. The relic is 4x4: only adjacent, non-overlapping 3x3
blocks can uncover it. :data:`CENTERS` clears a 3-wide column first (the
3-divination event option) and then the column beside it (the 6-divination
option, a 6x9 rectangle). A Monte Carlo over the decomp's placement rules
(n=4000; item values relic 150, rare card 45, rare potion 40, ..., curse -60)
put this order at EV ~46 / ~167 for 3 / 6 divinations, against ~34 / ~69 for a
uniformly random hidden cell (what ``realgame_random`` did) and ~21 / ~50 for
the first hidden cell (what F7 clicked).

A centre already cleared (the player divined by hand first) is skipped; past
the list, the cell clearing the most still-hidden cells wins, nearest the grid
centre on a tie.

Twin of ``spire-mentor-mod/CrystalSpherePick.cs`` (F7's lane, where the mod
answers the screen itself). Keep the two orders identical.
"""

from __future__ import annotations

from typing import Any

#: ``_defaultWidth`` / ``_defaultHeight`` (``CrystalSphereMinigame.cs``).
GRID_SIZE = 11

#: Big-tool centres, in order: column x 2..4, then x 5..7.
CENTERS: tuple[tuple[int, int], ...] = ((3, 2), (3, 5), (3, 8), (6, 2), (6, 5), (6, 8))

#: Names the deciding policy in the action's ``annotations`` (and so in the
#: driver's JSONL step record).
POLICY_NAME = "crystal_sphere_rule"


def pick_cell(hidden: list[tuple[int, int]]) -> int | None:
    """Index into ``hidden`` of the cell to divine, or ``None`` if none is hidden."""
    if not hidden:
        return None
    for center in CENTERS:
        if center in hidden:
            return hidden.index(center)
    hidden_set = set(hidden)
    mid = GRID_SIZE // 2

    def key(i: int) -> tuple[int, int, int]:
        x, y = hidden[i]
        cover = sum((x + dx, y + dy) in hidden_set for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        return (-cover, (x - mid) ** 2 + (y - mid) ** 2, i)

    return min(range(len(hidden)), key=key)


def choose(state: dict[str, Any]) -> dict[str, Any] | None:
    """The bridge answer to a ``crystal_sphere`` message, or ``None`` when no
    enabled option is usable (the caller falls back).

    The message shape is ``SimCrystalSphereScreenHandler``'s (PROTOCOL.md
    ``crystal_sphere``): ``{index, action: "divine_cell", x, y, enabled}`` per
    hidden cell plus a trailing ``{index, action: "proceed", enabled}``.
    """
    options = state.get("options") or []
    enabled = [
        (pos, o) for pos, o in enumerate(options) if isinstance(o, dict) and o.get("enabled", True)
    ]

    def _index(pos: int, o: dict[str, Any]) -> int:
        idx = o.get("index")
        return idx if isinstance(idx, int) else pos

    for pos, o in enabled:
        if o.get("action") == "proceed":
            return {
                "action": "choose",
                "index": _index(pos, o),
                "annotations": {
                    "note": f"{POLICY_NAME}: divinations spent -> proceed",
                    "policy": POLICY_NAME,
                },
            }

    cells: list[tuple[int, dict[str, Any], tuple[int, int]]] = []
    for pos, o in enabled:
        if o.get("action") != "divine_cell":
            continue
        try:
            cells.append((pos, o, (int(o["x"]), int(o["y"]))))
        except (KeyError, TypeError, ValueError):
            continue
    picked = pick_cell([xy for _, _, xy in cells])
    if picked is None:
        return None
    pos, o, (x, y) = cells[picked]
    return {
        "action": "choose",
        "index": _index(pos, o),
        "annotations": {
            "note": f"{POLICY_NAME}: divine cell ({x},{y}) (fixed 3x3 block order)",
            "policy": POLICY_NAME,
        },
    }
