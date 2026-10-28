"""Human-readable diff between two consecutive game states.

The workbench event log (Play and Replay) shows what an action *did* — damage
dealt/taken, block, buffs, deaths, and resource changes — by comparing the state
before an action with the state after it. This module is the single place those
effect lines are derived. Functions are pure and defensive: a missing key skips
a line rather than raising.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from enemy_labels import enemy_labels

_MINUS = "−"  # real minus glyph, for aligned signs
_DEAD = "☠ defeated"


def _signed(delta: int) -> str:
    """Signed magnitude like ``(-6)`` / ``(+16)`` using a real minus glyph."""
    return f"({_MINUS if delta < 0 else '+'}{abs(delta)})"


def _player_hp_block(prev: dict[str, Any], nxt: dict[str, Any], out: list[str]) -> None:
    p0 = prev.get("player") or {}
    p1 = nxt.get("player") or {}
    h0, h1 = p0.get("hp"), p1.get("hp")
    if isinstance(h0, int) and isinstance(h1, int) and h0 != h1:
        out.append(f"You HP {h0}→{h1} {_signed(h1 - h0)}")
    b0, b1 = p0.get("block", 0), p1.get("block", 0)
    if b0 != b1:
        out.append(f"You Block {b0}→{b1}")


def _enemy_name_at(
    enemies: list[dict[str, Any]], labels: list[str], pos: int, fallback: str
) -> str:
    """The display name of ``enemies[pos]``: its board label, else ``fallback``.

    ``labels`` is ``enemy_labels(enemies)``, computed once per state by the
    caller rather than once per enemy.

    The label is the name plus ``" #k"`` when several enemies in ``enemies``
    share it (:func:`enemy_labels.enemy_labels`). A diff spans two states, so
    each enemy is labelled from ONE of them: an enemy present before the action
    by its label in the BEFORE state (the board the player acted on — a kill
    still names the rat that was hit, though the after-board's numbers close
    up), a newly spawned one by its label in the AFTER state (the only board
    it is on). Rust twin: ``statediff.rs::enemy_name_at``.
    """
    if "name" not in enemies[pos]:
        return fallback
    return labels[pos]


def _enemy_lines(prev: dict[str, Any], nxt: dict[str, Any], out: list[str]) -> None:
    nxt_by_idx = {e.get("index"): e for e in (nxt.get("enemies") or [])}
    prev_es = prev.get("enemies") or []
    prev_labels = enemy_labels(prev_es)
    for pos, e0 in enumerate(prev_es):
        idx = e0.get("index")
        name = _enemy_name_at(prev_es, prev_labels, pos, f"enemy {idx}")
        h0 = e0.get("hp", 0)
        e1 = nxt_by_idx.get(idx)
        if e1 is None:
            # A live enemy vanishing = defeated; an already-0 corpse being cleaned up is silent.
            if h0 > 0:
                out.append(f"{name} HP {h0}→0  {_DEAD}")
            continue
        h1 = e1.get("hp", 0)
        if h0 != h1:
            line = f"{name} HP {h0}→{h1} {_signed(h1 - h0)}"
            if h1 <= 0:
                line += f"  {_DEAD}"
            out.append(line)
        b0, b1 = e0.get("block", 0), e1.get("block", 0)
        if b0 != b1:
            out.append(f"{name} Block {b0}→{b1}")


def _diff_powers(
    who: str,
    prev_powers: list[dict[str, Any]] | None,
    nxt_powers: list[dict[str, Any]] | None,
    out: list[str],
) -> None:
    p0 = {p.get("name"): p.get("amount", 0) for p in (prev_powers or [])}
    p1 = {p.get("name"): p.get("amount", 0) for p in (nxt_powers or [])}
    for name, amt in p1.items():
        if name not in p0:
            if amt:
                # Render the magnitude's sign properly so a debuff gained at a
                # negative amount reads "You −1 Shrink", not "You +-1 Shrink".
                sign = "+" if amt > 0 else _MINUS
                out.append(f"{who} {sign}{abs(amt)} {name}")
            else:
                out.append(f"{who} +{name}")
        elif amt != p0[name]:
            out.append(f"{who} {name} {p0[name]}→{amt}")
    for name in p0:
        if name not in p1:
            out.append(f"{who} {_MINUS}{name}")


def _power_lines(prev: dict[str, Any], nxt: dict[str, Any], out: list[str]) -> None:
    _diff_powers("You", prev.get("player_powers"), nxt.get("player_powers"), out)
    prev_list = prev.get("enemies") or []
    prev_es = {e.get("index"): (pos, e) for pos, e in enumerate(prev_list)}
    prev_labels = enemy_labels(prev_list)
    for e1 in nxt.get("enemies") or []:
        hit = prev_es.get(e1.get("index"))
        if hit is None:
            continue
        pos, e0 = hit
        # Labelled from the BEFORE state, like the HP lines (``_enemy_name_at``).
        name = _enemy_name_at(prev_list, prev_labels, pos, f"enemy {e0.get('index')}")
        _diff_powers(name, e0.get("powers"), e1.get("powers"), out)


def _spawn_lines(prev: dict[str, Any], nxt: dict[str, Any], out: list[str]) -> None:
    prev_idx = {e.get("index") for e in (prev.get("enemies") or [])}
    nxt_es = nxt.get("enemies") or []
    nxt_labels = enemy_labels(nxt_es)
    for pos, e in enumerate(nxt_es):
        if e.get("index") not in prev_idx:
            out.append(f"Spawned: {_enemy_name_at(nxt_es, nxt_labels, pos, 'enemy')}")


def _resource_lines(prev: dict[str, Any], nxt: dict[str, Any], out: list[str]) -> None:
    p0 = prev.get("player") or {}
    p1 = nxt.get("player") or {}
    g0, g1 = p0.get("gold"), p1.get("gold")
    if isinstance(g0, int) and isinstance(g1, int) and g0 != g1:
        out.append(f"Gold {g0}→{g1}")
    r0 = {r.get("name") for r in (p0.get("relics") or [])}
    for r in p1.get("relics") or []:
        if r.get("name") not in r0:
            out.append(f"+Relic {r.get('name')}")
    pot0 = Counter(pt.get("name") for pt in (p0.get("potions") or []))
    pot1 = Counter(pt.get("name") for pt in (p1.get("potions") or []))
    for name, n in (pot1 - pot0).items():
        out.extend([f"+Potion {name}"] * n)
    for name, n in (pot0 - pot1).items():
        out.extend([f"Used {name}"] * n)


def _orb_passive_lines(prev: dict[str, Any], nxt: dict[str, Any], out: list[str]) -> None:
    """Attribute end-of-turn effects to the Defect's orb passives.

    Orb passives fire when the player ends their turn; a ``round`` increment
    marks that boundary. The net HP/block diff above bundles enemy-turn effects
    with the orb passives, so this annotates *what each orb did* (using the
    pre-turn-end queue) to make the numbers traceable. Mechanics follow the C#
    decomp (``…/Models.Orbs``): Lightning hits one random enemy, Glass hits all,
    Frost grants block, Plasma grants energy, Dark stores its evoke value.
    """
    r0, r1 = prev.get("round"), nxt.get("round")
    if not (isinstance(r0, int) and isinstance(r1, int) and r1 > r0):
        return
    orbs = prev.get("orbs") or []
    if not orbs:
        return
    descs: list[str] = []
    for o in orbs:
        kind = (o.get("type") or "").lower()
        passive = o.get("passive", 0)
        if kind == "lightning":
            descs.append(f"Lightning orb passive: {passive} damage to a random enemy")
        elif kind == "frost":
            descs.append(f"Frost orb passive: +{passive} block")
        elif kind == "dark":
            descs.append(f"Dark orb passive: +{passive} stored damage")
        elif kind == "plasma":
            descs.append(f"Plasma orb passive: +{passive} energy")
        elif kind == "glass":
            descs.append(f"Glass orb passive: {passive} damage to all enemies")
        else:
            descs.append(f"{o.get('type', 'Orb')} orb passive")
    # Collapse identical lines (e.g. two Lightning orbs) into a "(×N)" suffix.
    for desc, n in Counter(descs).items():
        out.append(desc if n == 1 else f"{desc} (×{n})")


def diff_states(prev: dict[str, Any], nxt: dict[str, Any]) -> list[str]:
    """Return ordered, human-readable effect lines for the ``prev``→``nxt`` transition.

    Args:
        prev: The state an action was taken from.
        nxt: The state resulting from that action.

    Returns:
        Lines like ``"Nibbit HP 50→44 (−6)"``; empty when nothing notable changed.
    """
    out: list[str] = []
    _player_hp_block(prev, nxt, out)
    _enemy_lines(prev, nxt, out)
    _orb_passive_lines(prev, nxt, out)
    _power_lines(prev, nxt, out)
    _spawn_lines(prev, nxt, out)
    _resource_lines(prev, nxt, out)
    return out
