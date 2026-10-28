"""Human-readable rendering of card text and chosen actions.

The game ships card descriptions as templates (``"Deal {Damage:diff()}
damage."``) plus a per-card ``stats`` dict with the resolved numbers, and logs
actions as ``{cmd, action, args}`` dicts referencing cards by index. This module
turns both into plain English for display.
"""

from __future__ import annotations

import re
from typing import Any

from enemy_labels import enemy_labels

# Sentinel verb for a step taken by the NATIVE run agent (run-mode autoplay).
# Its ``args`` are a raw Rust ``RunAction`` dict rather than workbench-verb
# args. Defined here (not in ``workbench.branching``, which imports this
# module) so both the recorder and the renderer share one spelling.
NATIVE_ACTION = "__agent_native__"

# Native RunAction / logged-native kinds → the C# verb vocabulary the branches
# below speak. Only the combat kinds ``_native_action`` rewrites need an entry;
# everything else already round-trips as its own kind.
_NATIVE_KIND_VERB = {
    "play": "play_card",
    "drink_potion": "use_potion",
}

_TAG = re.compile(r"\[/?[a-zA-Z0-9_]+\]")  # BBCode color tags: [gold] ... [/gold]
_MULTISPACE = re.compile(r"\s{2,}")


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split ``text`` on ``sep``, ignoring separators inside ``{...}``."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _render_template(
    text: str, stats: dict[str, Any], flags: dict[str, bool], markup: bool = False
) -> str:
    """Recursively resolve ``{...}`` directives in ``text`` (balanced braces).

    Handles three directive shapes (branches may nest further ``{...}``):
    ``{Cond:then|else}`` boolean wrappers (``InCombat`` / ``IfUpgraded``, the
    latter with an optional ``show:`` prefix), ``{Var:plural:singular|plural}``
    selection, and ``{Var:...}`` value substitution from ``stats``.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "{":
            out.append(ch)
            i += 1
            continue
        # Find the matching close brace for this open brace.
        depth = 0
        j = i
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= n:  # unbalanced — emit the rest verbatim
            out.append(text[i:])
            break
        out.append(_eval_directive(text[i + 1 : j], stats, flags, markup))
        i = j + 1
    return "".join(out)


def _choose_branch(rest: str, value: Any) -> str | None:
    """Resolve a ``choose(opt1|opt2|...):branch1|branch2|...`` directive body.

    ``value`` (a card's morph var, e.g. CardType "Skill") selects the option
    index in the ``choose(...)`` list, then the parallel branch. Returns the
    raw chosen branch (caller renders it), or None if ``rest`` isn't a choose
    directive. Missing/unknown ``value`` falls back to the first branch.
    """
    if not rest.startswith("choose("):
        return None
    sep = rest.find("):")
    if sep < 0:
        return None
    options = [o.strip().lower() for o in rest[len("choose(") : sep].split("|")]
    branches = _split_top_level(rest[sep + 2 :], "|")
    idx = (
        options.index(str(value).lower())
        if value is not None and str(value).lower() in options
        else 0
    )
    return branches[idx] if idx < len(branches) else ""


def _eval_directive(
    body: str, stats: dict[str, Any], flags: dict[str, bool], markup: bool = False
) -> str:
    head, _, rest = body.partition(":")
    key = head.strip().lower()

    # ``{Var:choose(a|b|c):branchA|branchB|branchC}`` — pick a branch by the
    # card's morph value (Mad Science's CardType). The value may travel in the
    # morph ``flags`` or, for callers that stash it there, in ``stats``.
    chosen = _choose_branch(rest, flags.get(key, stats.get(key)))
    if chosen is not None:
        return _render_template(chosen, stats, flags, markup)

    # Boolean ``{Cond:then|else}`` wrappers: ``InCombat`` / ``IfUpgraded`` plus
    # any per-card morph flag the caller supplied (Mad Science's HasRider and
    # rider names). A flag is a condition iff it's present in ``flags``.
    if key in flags:
        branches = rest[len("show:") :] if rest.startswith("show:") else rest
        # Split on the top-level ``|`` (branches may contain nested braces).
        halves = _split_top_level(branches, "|")
        then_text = halves[0]
        else_text = halves[1] if len(halves) > 1 else ""
        chosen = then_text if flags.get(key) else else_text
        return _render_template(chosen, stats, flags, markup)

    # Cost glyphs. The game draws these as little energy orbs / stars inline —
    # "Gain ●●●", "Add a 0● copy", "Whenever you spend ★" — in four shapes:
    #   {Energy:energyIcons()}        N from stats
    #   {energyPrefix:energyIcons(1)} N given literally, no stat
    #   {Stars:starIcons()}           the Regent's star equivalent of the first
    #   {singleStarIcon}              exactly one star, no formatter at all
    # In markup mode they become ``[energy]N[/energy]`` / ``[star]N[/star]`` for
    # the card skin to draw. Plain mode keeps its existing output — a bare
    # number for the stat forms, nothing for the literal ones — because action
    # labels and logs read that text.
    icon = _ICON_FMT.match(rest.strip())
    if icon is not None or key in _PSEUDO_VARS:
        kind = "star" if key in _STAR_PSEUDO_VARS else "energy"
        count: Any = 1
        if icon is not None:
            kind = "star" if icon.group(1).lower() == "staricons" else "energy"
            # ``energyIcons(1)`` states its count; ``energyIcons()`` reads it.
            count = int(icon.group(2)) if icon.group(2) else stats.get(key)
        if markup and isinstance(count, (int, float)):
            return f"[{kind}]{int(count)}[/{kind}]"
        # Plain text has no glyphs, so it spells the resource out — matching the
        # wiki's pre-resolved ``description_plain`` (extract_cards.py), which is
        # the OTHER text the same card can display. The two must agree or a card
        # reads differently depending on which fallback the backend took.
        if key in _PSEUDO_VARS:
            # ``energyPrefix`` is a bare unit: the number is already in the
            # surrounding text ("a 0{energyPrefix…} copy" -> "a 0 Energy copy"),
            # so its literal count is an icon count, never a value to print.
            return "★" if kind == "star" else " Energy"
        noun = "Stars" if kind == "star" else "Energy"
        return f"{int(count)} {noun}" if isinstance(count, (int, float)) else noun

    if rest.startswith("plural:"):
        halves = _split_top_level(rest[len("plural:") :], "|")
        singular = halves[0]
        plural = halves[1] if len(halves) > 1 else ""
        val = stats.get(key)
        chosen = singular if isinstance(val, (int, float)) and val == 1 else plural
        return _render_template(chosen, stats, flags, markup)

    # Plain value substitution (``{Var}`` / ``{Var:diff()}``). Unknown vars
    # fall back to their bare name.
    val = stats.get(key)
    if val is None:
        return head
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return str(val)


# Icon / prefix tokens that look like ``{Var}`` substitutions but never come
# from a backend's ``stats`` dict — ``resolve_card_text`` handles them inline.
_PSEUDO_VARS = frozenset({"energyprefix", "singlestaricon"})
_STAR_PSEUDO_VARS = frozenset({"singlestaricon"})

# ``energyIcons()`` / ``starIcons(2)`` — an inline run of cost glyphs, with the
# count either read from ``stats`` (empty parens) or stated literally.
_ICON_FMT = re.compile(r"^(energyIcons|starIcons)\((\d*)\)$", re.IGNORECASE)


def _collect_vars(text: str, flags: dict[str, bool], out: set[str]) -> None:
    """Collect the lowercased ``stats`` keys ``text`` would look up (see
    :func:`required_stat_vars`). Mirrors :func:`_render_template`'s brace walk
    and :func:`_eval_directive`'s dispatch, following the conditional branch the
    flags actually select so only rendered vars are reported."""
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= n:
            break
        _collect_directive(text[i + 1 : j], flags, out)
        i = j + 1


def _collect_directive(body: str, flags: dict[str, bool], out: set[str]) -> None:
    head, _, rest = body.partition(":")
    key = head.strip().lower()
    # ``choose(...)`` picks one branch by a morph value carried in ``flags``
    # (not ``stats`` here — required_stat_vars has no values, only the flags),
    # so follow only the selected branch's vars.
    chosen = _choose_branch(rest, flags.get(key))
    if chosen is not None:
        # A chooser the caller can't supply is a REQUIRED var, so the card takes
        # the pre-resolved-plain fallback instead of a branch picked by default.
        # ``_choose_branch`` resolves an unknown value to branch 0, which for
        # Shiv's v0.107.1 `{TargetType:choose(AllEnemies): to ALL enemies|}` is
        # the Fan-of-Knives morph: every plain Shiv would have read "Deal 4
        # damage to ALL enemies." Only Mad Science's ``cardtype`` is supplied.
        if key not in flags:
            out.add(key)
        _collect_vars(chosen, flags, out)
        return
    if key in flags:
        branches = rest[len("show:") :] if rest.startswith("show:") else rest
        halves = _split_top_level(branches, "|")
        chosen = halves[0] if flags.get(key) else (halves[1] if len(halves) > 1 else "")
        _collect_vars(chosen, flags, out)
        return
    if rest.startswith("plural:"):
        if key not in _PSEUDO_VARS:
            out.add(key)
        for half in _split_top_level(rest[len("plural:") :], "|"):
            _collect_vars(half, flags, out)
        return
    restl = rest.lower()
    if key in _PSEUDO_VARS or "energyicons" in restl or "staricons" in restl:
        return
    out.add(key)


def required_stat_vars(
    description: str | None,
    *,
    in_combat: bool = False,
    upgraded: bool = False,
    extra_flags: dict[str, bool] | None = None,
) -> set[str]:
    """Return the lowercased ``stats`` keys ``description`` needs to fully resolve.

    Walks the same ``{...}`` DSL as :func:`resolve_card_text`, following the
    ``InCombat`` / ``IfUpgraded`` branch the flags select, and excluding icon /
    prefix pseudo-tokens (``energyPrefix``, ``energyIcons``, ``starIcons``) that
    don't come from ``stats``. Callers compare the result against the ``stats``
    a backend supplied: if some var is missing, the backend couldn't fully
    resolve the card and the pre-resolved plain description should be used
    instead (the Rust backend can't attach a card's full ``stats`` in combat).
    """
    if not description:
        return set()
    out: set[str] = set()
    flags = {"incombat": in_combat, "ifupgraded": upgraded, **(extra_flags or {})}
    _collect_vars(description, flags, out)
    return out


def resolve_card_text(
    description: str | None,
    stats: dict[str, Any] | None,
    *,
    in_combat: bool = False,
    upgraded: bool = False,
    extra_flags: dict[str, bool] | None = None,
    keep_markup: bool = False,
) -> str:
    """Resolve a card description template to plain display text.

    Resolves the STS2 template DSL: ``{Var:diff()}`` value substitution from
    ``stats`` (keyed by lowercased var name), ``{Var:plural:singular|plural}``
    selection, the ``{InCombat:then|else}`` / ``{IfUpgraded:show:then|else}``
    boolean wrappers, and the morph directives ``{Var:choose(a|b|c):...}``
    (branch by a value in ``stats``/``extra_flags``) + ``{Cond:then|else}`` for
    any boolean in ``extra_flags`` — used by Mad Science's CardType + riders.
    Branches may nest. BBCode ``[tag]`` markup is stripped; the energy-icon
    prefix pseudo-token renders empty. Unknown variables fall back to their name.

    Args:
        description: Raw template, e.g. ``"Deal {Damage:diff()} damage."``.
        stats: Resolved values, e.g. ``{"damage": 6}``. May be None.
        in_combat: Resolve ``{InCombat:..}`` to its then-branch (else otherwise).
            False for shop / deck / reward displays.
        upgraded: Resolve ``{IfUpgraded:..}`` to its then-branch.
        extra_flags: Per-card morph variables — booleans treated as
            ``{Cond:..}`` conditions plus the ``choose`` selector value (e.g.
            ``{"cardtype": "Skill", "hasrider": True, "sapping": True}``).
        keep_markup: Leave the BBCode ``[tag]`` markup in place instead of
            stripping it. The game's own localization marks its keywords up as
            ``[gold]Vulnerable[/gold]``, and the card skin renders them gold —
            so the card frame asks for the marked-up form while every other
            caller (action labels, logs, tooltips) keeps the plain one.

    Returns:
        Plain text, e.g. ``"Deal 6 damage."``. Empty string for falsy input.
    """
    if not description:
        return description or ""
    flags = {"incombat": in_combat, "ifupgraded": upgraded, **(extra_flags or {})}
    out = _render_template(description, stats or {}, flags, markup=keep_markup)
    if not keep_markup:
        out = _TAG.sub("", out)
    return _MULTISPACE.sub(" ", out).strip()


def _find(items: list[dict[str, Any]] | None, idx: int | None) -> dict[str, Any] | None:
    """Return the item at stable ``index`` (or positional) ``idx``, or None."""
    if not items or idx is None:
        return None
    for it in items:
        if it.get("index") == idx:
            return it
    if 0 <= idx < len(items):
        return items[idx]
    return None


def _named(items: list[dict[str, Any]] | None, idx: int | None, kind: str) -> str:
    """Return the name of the item at stable ``index`` (or positional) ``idx``."""
    it = _find(items, idx)
    return it.get("name", f"{kind} {idx}") if it is not None else f"{kind} {idx}"


def _enemy_named(items: list[dict[str, Any]] | None, idx: int | None) -> str:
    """:func:`_named` for an enemy target: the enemy's board label.

    A name shared by several enemies carries its ``" #k"``
    (:func:`enemy_labels.enemy_labels`), so "Play Strike → Two-Tailed Rat #2"
    says WHICH rat. The Rust twin is ``render.rs::enemy_named``.
    """
    if not items or idx is None:
        return f"enemy {idx}"
    pos = next((i for i, e in enumerate(items) if e.get("index") == idx), None)
    # The positional fallback is only for a list with no stable ``index`` at
    # all: when the view carries indices, a missing one is a slot that is not
    # on the board, and its list position names a DIFFERENT enemy.
    if pos is None and not any("index" in e for e in items) and 0 <= idx < len(items):
        pos = idx
    if pos is None or "name" not in items[pos]:
        return f"enemy {idx}"
    return enemy_labels(items)[pos]


def _card_name(cards: list[dict[str, Any]] | None, idx: int | None) -> str:
    """Like :func:`_named` but marks upgraded cards with a trailing ``+``."""
    it = _find(cards, idx)
    if it is None:
        return f"card {idx}"
    name = it.get("name", f"card {idx}")
    return f"{name}+" if it.get("upgraded") else name


def describe_action(state: dict[str, Any] | None, action: dict[str, Any] | None) -> str:
    """Render the action taken from ``state`` as a short English phrase.

    Args:
        state: The decision-point state the action was chosen from (used to
            resolve card indices to names).
        action: The logged action ``data`` dict (``{cmd, action, args}``).

    Returns:
        e.g. ``"Play Strike → enemy 0"``, ``"End turn"``, ``"Start run · Ironclad"``.
        Empty string when there is no action.
    """
    if not action:
        return ""
    state = state or {}
    # A pre-rendered label wins (live-game human actions carry only a display
    # string -- realgame_workbench_log.log_human_action).
    label = action.get("label")
    if label:
        return str(label)
    cmd = action.get("cmd")
    if cmd == "start_run":
        return f"Start run · {action.get('character', '?')}"

    act = action.get("action", cmd)
    args = action.get("args") or {}
    if act == NATIVE_ACTION:
        # A native-agent step recorded before the writer persisted ``label``
        # (workbench runs up to 2026-08-26). Its real kind sits inside ``args``;
        # left wrapped, every autoplayed combat step in the replay's history
        # read "__agent_native__" where the live log had said "Play Strike →
        # Twig Slime". Newer logs never reach here — their ``label`` wins above.
        kind = str(args.get("action") or "")
        act = _NATIVE_KIND_VERB.get(kind, kind or "?")
        args = {k: v for k, v in args.items() if k != "action"}
    if act == "play_card":
        # Both index spellings: the C# verb's ``card_index`` and the raw
        # RunAction's ``hand_idx`` (same semantics — a position in the hand).
        card_idx = next(
            (args[k] for k in ("card_index", "hand_idx") if args.get(k) is not None), None
        )
        name = _card_name(state.get("hand"), card_idx)
        target = args.get("target_index")
        if target is not None:
            return f"Play {name} → {_enemy_named(state.get('enemies'), target)}"
        return f"Play {name}"
    if act == "end_turn":
        return "End turn"
    if act in ("use_potion", "drink_potion"):
        # Both index spellings: ``to_workbench_verb`` emits ``potion_index``,
        # the native RunAction shape ``slot``.
        slot = next((args[k] for k in ("potion_index", "slot") if args.get(k) is not None), None)
        # The BELT, not ``state["potions"]`` — that key is the shop's stock (see
        # ``buy_potion`` below), so reading it here named a drink after whatever
        # happened to be on sale. Top level is the fallback for a backend that
        # puts the belt there.
        belt = (state.get("player") or {}).get("potions") or state.get("potions")
        name = _named(belt, slot, "potion")
        target = args.get("target_index")
        if target is not None:
            return f"Drink {name} → {_enemy_named(state.get('enemies'), target)}"
        return f"Drink {name}"
    if act == "select_map_node":
        return f"Map → ({args.get('col')},{args.get('row')})"
    if act == "select_card_reward":
        return f"Take {_card_name(state.get('cards'), args.get('card_index'))}"
    if act == "skip_card_reward":
        return "Skip card reward"
    if act == "buy_card":
        return f"Buy {_card_name(state.get('cards'), args.get('card_index'))}"
    if act in ("buy_relic", "buy_potion"):
        key, idx_arg = {
            "buy_relic": ("relics", "relic_index"),
            "buy_potion": ("potions", "potion_index"),
        }[act]
        return f"Buy {_named(state.get(key), args.get(idx_arg), key[:-1])}"
    if act == "choose_option":
        oi = args.get("option_index")
        options = state.get("options") or []
        if isinstance(oi, int) and 0 <= oi < len(options):
            opt = options[oi]
            # C# options carry ``text``/``name``; the Rust backend emits
            # ``title`` (generic screens) or ``option_id`` (rest sites).
            label = (
                opt.get("text") or opt.get("name") or opt.get("title") or opt.get("option_id")
                if isinstance(opt, dict)
                else str(opt)
            )
            if label:
                return f"Choose: {label}"
        return f"Choose option {oi}"
    if act == "select_cards":
        idxs = [int(t) for t in str(args.get("indices", "")).split(",") if t.strip().isdigit()]
        # A SHOP's only ``select_cards`` is the removal service, and its
        # indices are MASTER-DECK positions (``run_policy.to_workbench_verb``
        # maps ``shop_remove_card{deck_idx}`` here) — while a shop state's
        # ``cards`` is the STOCK. Naming them against the stock labelled a
        # removal with whatever happened to be on sale at that index: report
        # wb-c2adad9c8eeb21f7 logged "Select Thunderclap" for a 75g removal of
        # deck card #0. Say which deck card, the way the PPO panel already
        # does (``ppo_policy.shop_remove_label``).
        deck = (state.get("player") or {}).get("deck")
        if state.get("decision") == "shop" and deck:
            named = [f"{_card_name(deck, i)} (deck #{i})" for i in idxs]
            return f"Remove {', '.join(named)}" if named else "Remove card"
        names = [_card_name(state.get("cards"), i) for i in idxs]
        # The Rust backend stamps the card_select's purpose on ``context``
        # ("upgrade" = rest-site Smith, "remove" = deck removal, otherwise a
        # combat pending-choice kind).
        verb = {"upgrade": "Upgrade", "remove": "Remove"}.get(
            str(state.get("context") or ""), "Select"
        )
        return f"{verb} {', '.join(names)}" if names else "Select cards"
    if act == "skip_select":
        return "Skip selection"
    return str(act or "?")
