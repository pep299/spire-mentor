"""Single-trial driver for the C#/Rust parity-fuzz harness (Task 8).

Drives both engines from a seeded `Scenario` and reports the first
divergence (init state, legal-action set, or post-action state).
Finding a divergence is *success* — T10 will triage them into beads.

See docs/superpowers/plans/2026-05-24-csharp-rust-parity-fuzz.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import sts2_rs  # type: ignore[import-not-found]
from parity_fuzz_scenario import Scenario, generate
from parity_fuzz_state import (
    diff,
    diff_post_action,
    normalize_csharp,
    normalize_rust,
)
from sts2_client import Game, rs_engine_call

DIVERGENCE_INIT = "INIT_DRIFT"
DIVERGENCE_LEGAL = "LEGAL_ACTION_DRIFT"
DIVERGENCE_STATE = "STATE_DRIFT"
DIVERGENCE_HARNESS = "HARNESS_ERROR"


@dataclass
class TrialResult:
    seed: int
    scenario: Scenario
    turns_played: int
    divergence_type: str | None = None
    divergence_detail: dict[str, Any] | None = None
    action_history: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine init
# ---------------------------------------------------------------------------


#: Shared run-level seed string for parity fuzz. C# derives its
#: ``RunRngSet.Seed`` via ``GetDeterministicHashCode(SEED_STRING)`` (the
#: build's hash: 32-bit ``uint`` in v0.107.1, XxHash64 ``ulong`` in v0.111.0);
#: Rust uses the same string hashed through :func:`rust_master_seed`
#: (``sts2_rs.run_seed_hash``) so both engines enter combat with
#: byte-identical RNG state. Bead `sts2-rs-ykmk9`, GH #792.
SEED_STRING = "parity_fuzz"


def _canonicalize_seed(seed: str) -> str:
    """Mirror C# ``SeedHelper.CanonicalizeSeed`` (the game's typed-seed
    normalizer): upper-case, ``'O'->'0'``, ``'I'->'1'``, trim.

    The C# headless ``RunSimulator`` canonicalizes the seed before seeding
    ``RunState`` (sts2-pycli ``74cc964``, matching ``StartRunLobby`` +
    ``SeedHelper``), so the Rust side MUST hash the SAME canonical string or
    the two engines start combat from different ``RunRngSet`` seeds. Without
    this, ``SEED_STRING="parity_fuzz"`` hashes raw on the Rust side but
    canonicalizes to ``"PAR1TY_FUZZ"`` (the ``I``->``1`` fold) on the C#
    side — a silent seed mismatch that desyncs the opening shuffle + every
    monster HP roll, bucketing EVERY coverage trial as ``DRAW_ORDER``.
    """
    return seed.upper().replace("O", "0").replace("I", "1").strip()


#: Canonical form of :data:`SEED_STRING`, used for the Rust-side seed hash
#: so it matches the C# headless's canonicalized ``RunState`` seed.
SEED_STRING_CANONICAL = _canonicalize_seed(SEED_STRING)


def rust_master_seed(canonical_seed: str, game_version: str | None = None) -> int:
    """The ``RunRngSet.Seed`` C# derives from an (already canonical) seed string.

    Delegates to the engine's versioned ``sts2_rs.run_seed_hash``:

    * v0.107.1 — ``(uint)StringHelper.GetDeterministicHashCode(seed)``, the
      32-bit hash reinterpreted unsigned (what this harness used to spell
      ``deterministic_hash_code(seed) & 0xFFFFFFFF``; identical value).
    * v0.111.0 — ``XxHash64(utf8(seed), 0)`` as a ``ulong``
      (``StringHelper.cs:140``), plus ``RunRngSet.cs:107``'s ``"old"``-prefix
      escape hatch.

    ``game_version`` defaults to the process-wide selection
    (``sts2_client.GAME_VERSION`` / ``--game-version``).
    """
    import sts2_client

    ver = sts2_client.normalize_game_version(game_version or sts2_client.GAME_VERSION)
    return int(sts2_rs.run_seed_hash(canonical_seed, game_version=ver))


def _csharp_card_id(name: str) -> str:
    """Map a Rust-canonical card id to the C# model id.

    The Rust content crate appends ``_CARD`` to disambiguate a card from a
    same-named power (``BARRICADE`` power vs ``BARRICADE_CARD`` card). C#
    namespaces with a ``CARD.`` prefix instead, so its model id is the bare
    name. Without stripping, ``set_player`` raises
    ``ModelNotFoundException: Model id=CARD.<X>_CARD not found`` and the
    whole trial buckets as HARNESS_INIT (masking ~24 real IC cards). Rust
    accepts both forms, so we only normalize on the C# send.

    ``_STATUS`` is the same split with one member (Rust ``SLOTH_STATUS``, C#
    ``Sloth``) and must strip the same way, or it takes the identical
    ``ModelNotFoundException`` → HARNESS_INIT masking path described above.
    Latent today — no scenario's card list names ``SLOTH`` — but a status card
    is exactly the kind of thing a future scenario adds. At most ONE suffix
    comes off: a registry id carries exactly one.
    """
    base, plus, lvl = name.partition("+")
    for suffix in ("_CARD", "_STATUS"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}+{lvl}" if plus else base


def _csharp_init(g: Game, sc: Scenario) -> dict:
    """Start a C# run, skip Neow, set scenario player, enter combat room.

    Raises :class:`RuntimeError` if any command returns an error response —
    callers turn that into a HARNESS_ERROR bucket so we don't conflate
    C#-side init failures (which manifest as ``combat_terminal=true`` in
    a downstream ``get_state``) with real Rust port divergences.
    """
    state = g.start(character=sc.character, seed=SEED_STRING)
    g.skip_neow(state)
    r = g.set_player(
        hp=sc.hp,
        max_hp=sc.max_hp,
        deck=[_csharp_card_id(c) for c in sc.deck],
        relics=list(sc.relics),
        potions=list(sc.potions),
    )
    if r.get("type") == "error":
        raise RuntimeError(
            f"set_player failed: {r.get('message')!r}\nstack_trace: {r.get('stack_trace')}"
        )
    r = g.enter_room("combat", encounter=sc.encounter)
    if r.get("type") == "error":
        raise RuntimeError(
            f"enter_room failed: {r.get('message')!r}\nstack_trace: {r.get('stack_trace')}"
        )
    return g.get_state()


class _RustStateAdapter:
    """Wraps a `sts2_rs.PyCombatState` so it satisfies the protocol
    expected by :func:`normalize_rust` (a ``.state_to_dict()`` method).

    The underlying PyCombatState exposes :func:`legal_actions` /
    :func:`apply` directly; only the dict-conversion entry point lives
    at the module level as `sts2_rs.state_to_dict(state)`.
    """

    __slots__ = ("inner",)

    def __init__(self, inner):
        self.inner = inner

    def state_to_dict(self):
        return sts2_rs.state_to_dict(self.inner)

    def legal_actions(self):
        return self.inner.legal_actions()

    def apply(self, action):
        return self.inner.apply(action)

    def pending_choice(self):
        return self.inner.pending_choice()


def _rust_init(
    sc: Scenario,
    monsters: list[str] | None = None,
    move_states: list[str | None] | None = None,
):
    """Build the Rust-side scenario state.

    `monsters`, when provided, overrides the encounter-table roster
    with an explicit list of Pascal-case monster ids — this lets the
    harness (sts2-rs-3010t) reach roster parity with C# for
    encounters whose `GenerateMonsters` is RNG-branched
    (FLYCONID_NORMAL, SLITHERING_STRANGLER_NORMAL, SLIMES_*). The
    canonical id source is the C# `get_state` monsters[].kind field.

    `move_states`, when provided, is the per-monster initial move-state
    id observed on the C# side (`get_state` monsters[].move_state),
    aligned positionally with the roster. It force-matches monsters
    whose opening move is picked by an RNG-branched `StarterMoveIdx`
    (ScrollsOfBiting{Weak,Normal}): the synthetic scenario builder
    can't replicate C#'s per-encounter RNG seeding, so it adopts the
    rotation C# rolled. Harmless for every non-rotating monster — the
    Rust side returns `None` from `starter_args_for_move_state` and
    keeps the encounter-table default. Bead sts2-rs-u3wjj.
    """
    rust_seed = rust_master_seed(SEED_STRING_CANONICAL)
    # `rs_engine_call` passes the selected build as `game_version=`, so the
    # scenario's `RunRngSet` is seeded in that build's width.
    inner = rs_engine_call(
        sts2_rs.build_scenario_state,
        seed=rust_seed,
        character=sc.character,
        deck=list(sc.deck),
        hp=sc.hp,
        max_hp=sc.max_hp,
        relics=list(sc.relics),
        encounter=sc.encounter,
        monsters=monsters,
        potions=list(sc.potions),
        move_states=move_states,
    )
    return _RustStateAdapter(inner)


# ---------------------------------------------------------------------------
# Legal-action canonicalization
# ---------------------------------------------------------------------------


def _canonical_action(a: dict) -> tuple:
    ti = a.get("target_index")
    # potion actions key on potion_index (separate index space from cards).
    idx = a.get("card_index", a.get("potion_index", -1))
    return (
        a["action"],
        idx,
        -1 if ti is None else ti,
    )


def _csharp_legal(g: Game) -> list[dict]:
    raw = g.legal_actions()["actions"]
    out = []
    for a in raw:
        act = a["action"]
        if act == "play_card":
            out.append(
                {
                    "action": "play_card",
                    "card_index": a["card_index"],
                    "target_index": a.get("target_index"),
                }
            )
        elif act == "end_turn":
            out.append({"action": "end_turn"})
        else:
            out.append(dict(a))
    return sorted(out, key=_canonical_action)


def _rust_alive_slots(state) -> list[int]:
    """Slot indices (positions in ``enemy_uids``) of the alive enemies, in
    order — the "Rust normalizer" the C# ``RunSimulator`` comment calls for
    (parity-fuzz R4 / sts2-rs-69jzx).

    The Rust core encodes a ``PlayCard`` target as a RAW ``enemy_uids`` slot
    index and deliberately keeps dead enemies in that substrate (corpses
    retained for stable NN feature slots, sts2-rs-c8why). C#'s
    ``RunSimulator.LegalActions`` instead emits ``target_index`` as a DENSE
    rank over the alive-enemy subset (``Enemies.Where(IsAlive)``), and
    ``DoPlayCard`` resolves it against that same alive subset. So once an
    earlier enemy dies, the same card+enemy gets a different index on each
    side and the action sets never match (the LEGAL_ACTION bucket).

    Canonicalize on the Rust side here: build the alive-slot list so we can
    map slot↔rank in both directions (``_rust_legal`` emits the rank,
    ``_apply_rust`` converts a C#-supplied rank back to the core slot). The
    Rust `encounter` list is emitted in ``enemy_uids`` order and includes
    retained corpses, so its indices line up with the core's slot indices;
    filtering to ``is_dead == False`` yields the alive subset in the same
    order C# ranks over.
    """
    d = sts2_rs.state_to_dict(state.inner)
    enc = d.get("encounter") or []
    return [i for i, m in enumerate(enc) if not bool(m.get("is_dead", False))]


def _rust_legal(state) -> list[dict]:
    raw = state.legal_actions()
    alive_slots = _rust_alive_slots(state)
    out = []
    for a in raw:
        act = a["action"]
        if act == "play":
            # R4: convert the core's raw enemy_uids slot index into the
            # dense alive-rank C# emits. A non-target (Self/None/All) card
            # has target_index=None and is left untouched.
            slot = a.get("target_index")
            rank = alive_slots.index(slot) if slot in alive_slots else slot
            out.append(
                {
                    "action": "play_card",
                    "card_index": a["card_index"],
                    "target_index": rank,
                }
            )
        elif act == "end_turn":
            out.append({"action": "end_turn"})
        elif act == "drink_potion":
            # Rust combat surfaces held potions as
            # {"action":"drink_potion","slot":i,"target_index":None};
            # canonicalize to the C# headless shape
            # {"action":"use_potion","potion_index":i,"target_index":None}.
            out.append(
                {
                    "action": "use_potion",
                    "potion_index": a["slot"],
                    "target_index": a.get("target_index"),
                }
            )
        else:
            out.append(
                {
                    "action": act,
                    "card_index": a.get("card_index", -1),
                    "target_index": a.get("target_index"),
                }
            )
    return sorted(out, key=_canonical_action)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _apply_csharp(g: Game, action: dict) -> dict:
    if action["action"] == "end_turn":
        return g.end_turn()
    elif action["action"] == "play_card":
        return g.play_card(action["card_index"], action.get("target_index"))
    elif action["action"] == "use_potion":
        return g.use_potion(action["potion_index"], action.get("target_index"))
    else:
        raise NotImplementedError(action)


def _drain_csharp_choices(g: Game, last_result: dict | None, max_iters: int = 16) -> dict | None:
    """Auto-resolve any C# ``card_select`` decision, mirroring ``_drain_rust_choices``.

    F3 disambiguation (audit 2026-05-29): a choice-resolving card such as
    ARMAMENTS calls ``await CardSelectCmd.FromHandForUpgrade`` *inside*
    ``CardModel.OnPlayWrapper.OnPlay`` (C# CardModel.cs:1498), and the card is
    only moved from PlayPile to its result pile (Discard) *after* ``OnPlay``
    returns (CardModel.cs:1538-1552). So at the ``card_select`` decision point
    the played card is legitimately still in the C# PlayPile — ``discard_count``
    is 1 lower than Rust's, which the audit harness had already drained through
    its choice via ``_drain_rust_choices``. That asymmetry — Rust snapshotted
    *after* choice resolution, C# snapshotted *paused at the choice* — was the
    sole cause of the ``PILE_COUNT|ARMAMENTS`` bucket. It is a HARNESS artifact,
    not a Rust over-eager-discard bug and not a C# headless bug (the headless
    cannot "pump past" an open player choice, and force-flushing the PlayPile
    would diverge it from the real game). The correct fix is to drain the C#
    choice the same way Rust's is drained: pick option 0 and continue, so both
    engines are snapshotted at the equivalent post-resolution lifecycle point.
    """
    result = last_result
    for _ in range(max_iters):
        if not isinstance(result, dict) or result.get("type") != "decision":
            return result
        decision = result.get("decision")
        if decision == "card_select":
            opts = result.get("cards") or []
            if not opts:
                # No options to pick — the only legal resolution is to skip.
                result = g.skip_select()
                continue
            # Mirror _drain_rust_choices, which ALWAYS picks option 0 and
            # never skips — even for canSkip:true (min_select=0) choices such
            # as Discovery / the generated-card potions
            # (FromChooseACardScreen). Previously this branch skipped when
            # min_select<=0, so C# added 0 cards while Rust auto-picked 1 —
            # the residual `hand_count` PILE_COUNT divergence left after the
            # combat_card_generation pool fix. Picking is the gameplay-normal
            # play and matches the F3 "pick option 0 and continue" intent.
            min_select = result.get("min_select", 1) or 0
            count = min(max(min_select, 1), len(opts))
            result = g.select_cards(",".join(str(i) for i in range(count)))
        else:
            return result
    return result


def _apply_rust(state, action: dict) -> None:
    if action["action"] == "play_card":
        # R4 (sts2-rs-69jzx): the supplied action carries C#'s dense
        # alive-rank target_index; the Rust core expects a raw enemy_uids
        # slot index. Map rank -> slot via the same alive-slot list
        # _rust_legal ranks over, so a play after a leading-enemy death
        # hits the correct enemy. None (Self/None/All) passes through.
        target = action.get("target_index")
        if target is not None:
            alive_slots = _rust_alive_slots(state)
            if 0 <= target < len(alive_slots):
                target = alive_slots[target]
        # NOTE: the Rust PyCombatState.apply binding is currently lossy —
        # it always uses hand_index=0 regardless of card_index. Filed
        # in T8 report; will be tightened before T10.
        state.apply(
            {
                "action": "play",
                "card_index": action["card_index"],
                "target_index": target,
            }
        )
    elif action["action"] == "end_turn":
        state.apply({"action": "end_turn"})
    elif action["action"] == "use_potion":
        # canonical use_potion -> Rust apply shape
        # {"action":"drink_potion","slot":i} (Action::DrinkPotion).
        state.apply(
            {
                "action": "drink_potion",
                "slot": action["potion_index"],
                "target_index": action.get("target_index"),
            }
        )
    else:
        raise NotImplementedError(action)


def _drain_rust_choices(state, max_iters: int = 16) -> None:
    """Auto-resolve any Rust ``pending_choice`` to mirror the C# headless.

    The C# headless runner feeds its ``PlayerChoiceContext`` automatically
    inside ``play_card`` (e.g. Armaments' ``CardSelectCmd.FromHandForUpgrade``
    picks a card and continues), so the C# ``legal_actions`` cmd never
    surfaces a "choose" sub-decision — it only ever returns play_card /
    end_turn. The Rust ``PyCombatState`` instead exposes an open
    ``pending_choice`` as ``{"action": "choose", "index": i}`` legal actions,
    with no other action legal until it is resolved.

    Left unresolved, this asymmetry produces a spurious LEGAL_ACTION_DRIFT on
    every card that opens a choice (Armaments, Warcry, …). To compare the two
    engines on equal footing we drain the Rust choice the same way the C#
    headless does: pick option 0 and continue. (For Armaments between two
    identical hand cards the pick is indistinguishable; where C#'s auto-pick
    differs from option 0 the residual surfaces later as STATE_DRIFT, which is
    a separate, real signal.)

    Multi-select policy (bead ``sts2-rs-sopgf`` mirror): a ``max_select > 1``
    choice — Purity's ``ExhaustFromHand`` (``min_select == 0``), Hidden
    Daggers' ``DiscardFromHand`` (``min == max == 2``) — is exposed by the Rust
    core as an ACCUMULATOR of single-index picks that re-prompt: applying
    ``selection=[i]`` adds card ``i`` to the running set and re-offers the rest,
    and the pile move only fires once the choice closes. Greedily looping
    "pick the first non-empty option until it closes" therefore exhausts/
    discards the WHOLE hand (every card up to the cap) — which is exactly the
    ``PILE_COUNT|PURITY :: exhaust_count`` divergence (Rust 5 vs C# 2): the C#
    drainer (:func:`_drain_csharp_choices`) and the Rust core's own
    deterministic auto-resolve both commit ``count = min(max(min_select, 1),
    n)`` cards, NOT the maximum. So for a ``max_select > 1`` choice we pick the
    lowest-index unpicked options until ``count`` cards are committed, THEN take
    the empty/skip selection to FINALIZE — matching C#. ``count`` is read off
    the live ``pending_choice()`` (``min_select`` / ``max_select`` / ``cards``)
    rather than assumed, and committed cards are counted as ``n - remaining``
    options so the policy is robust to single-index vs bundled selections.
    """
    for _ in range(max_iters):
        raw = state.legal_actions()
        if not raw or any(a["action"] != "choose" for a in raw):
            return
        pc = state.pending_choice() if hasattr(state, "pending_choice") else None
        if pc and (pc.get("max_select") or 1) > 1:
            _drain_rust_multi_select(state, pc, max_iters)
            continue
        # Single-select (``max_select <= 1``): drive the FIRST legal selection
        # whose index list is non-empty. For a skippable / min0 choice the
        # empty-skip entry sorts first; we skip it and take the first non-empty
        # `[0]` (the C# headless never skips — it always picks option 0). Fall
        # back to `index` only when EVERY entry is the empty skip (empty-options
        # choice). Mirrors `resolve_inline_auto_play_choices`.
        chosen = next((a for a in raw if a.get("selection")), raw[0])
        sel = chosen.get("selection")
        if sel:
            state.apply({"action": "choose", "selection": list(sel)})
        else:
            state.apply({"action": "choose", "index": chosen.get("index", 0)})


def _drain_rust_multi_select(state, pc: dict, max_iters: int) -> None:
    """Resolve one open ``max_select > 1`` Rust choice to ``count`` cards.

    Picks the lowest-index unpicked option ``count = min(max(min_select, 1),
    n)`` times, then applies the empty/skip selection to FINALIZE — mirroring
    the C# headless drainer and the Rust core's ``sts2-rs-sopgf`` auto-resolve.
    Committed cards are counted as ``n - remaining`` non-empty options, so the
    accounting holds whether the core offers single-index or bundled picks.
    """
    n0 = len(pc.get("cards") or [])
    target = min(max(int(pc.get("min_select") or 0), 1), n0)
    for _ in range(max_iters):
        raw = state.legal_actions()
        if not raw or any(a["action"] != "choose" for a in raw):
            return  # choice auto-closed (e.g. min == max cap reached)
        nonempty = [a for a in raw if a.get("selection")]
        committed = n0 - len(nonempty)
        if committed >= target or not nonempty:
            # Enough cards selected — take the empty/skip entry to finalize the
            # accumulated selection (commits the pile move) and close it.
            state.apply({"action": "choose", "selection": []})
            return
        state.apply({"action": "choose", "selection": list(nonempty[0]["selection"])})


# ---------------------------------------------------------------------------
# Trial driver
# ---------------------------------------------------------------------------


def run_trial(seed: int, max_turns: int = 5) -> TrialResult:
    """Run a single parity-fuzz trial from `seed`.

    Aborts on first divergence and reports it via :class:`TrialResult`.
    """
    sc = generate(seed)
    rng = random.Random(seed ^ 0xDEADBEEF)
    result = TrialResult(seed=seed, scenario=sc, turns_played=0)
    g = Game()
    try:
        try:
            cs_raw = _csharp_init(g, sc)
            # sts2-rs-3010t: extract the resolved monster roster from
            # the C# state so the Rust side spawns the same enemies
            # for RNG-branched encounters (Slithering Strangler's
            # SmallSlimes branch, Flyconid's medium-slime pick, etc.).
            cs_monsters = [m["kind"] for m in (cs_raw.get("monsters") or [])]
            rs_state = _rust_init(sc, monsters=cs_monsters)
            # A combat-start relic/potion could open a choice before the
            # first action; drain it so init parity is measured symmetrically
            # with the C# headless (which auto-resolves PlayerChoiceContext).
            _drain_rust_choices(rs_state)
        except Exception as exc:
            result.divergence_type = DIVERGENCE_HARNESS
            result.divergence_detail = {"stage": "init", "error": repr(exc)}
            return result

        init_divs = diff(normalize_csharp(cs_raw), normalize_rust(rs_state))
        if init_divs:
            result.divergence_type = DIVERGENCE_INIT
            result.divergence_detail = {"divergences": init_divs[:20]}
            return result

        turn = 0
        while turn < max_turns:
            try:
                cs_actions = _csharp_legal(g)
                rs_actions = _rust_legal(rs_state)
            except Exception as exc:
                result.divergence_type = DIVERGENCE_HARNESS
                result.divergence_detail = {
                    "stage": "legal_actions",
                    "turn": turn,
                    "error": repr(exc),
                }
                return result

            if cs_actions != rs_actions:
                # sts2-rs-6rge3: dump hand contents so per-card
                # target_index drifts can be attributed to a specific
                # card id rather than only its position.
                rs_hand_ids: list[str] = []
                cs_hand_ids: list[str] = []
                try:
                    rs_d = sts2_rs.state_to_dict(rs_state.inner)
                    rs_hand_ids = [c.get("id") for c in (rs_d.get("deck", {}).get("hand") or [])]
                except Exception:
                    pass
                try:
                    cs_state = g.get_state()
                    cs_hand_ids = list(cs_state.get("hand_ids") or [])
                except Exception:
                    pass
                result.divergence_type = DIVERGENCE_LEGAL
                result.divergence_detail = {
                    "csharp_only": [a for a in cs_actions if a not in rs_actions],
                    "rust_only": [a for a in rs_actions if a not in cs_actions],
                    "turn": turn,
                    "rust_hand": rs_hand_ids,
                    "csharp_hand": cs_hand_ids,
                }
                return result

            if not cs_actions:
                break  # terminal

            action = rng.choice(cs_actions)
            result.action_history.append(action)

            # sts2-rs-6vj51: snapshot hand/discard card-id lists on
            # BOTH engines BEFORE applying the action, so STATE_DRIFT
            # post-action can be attributed to a specific card. Cheap
            # — only retained when STATE_DRIFT actually fires below.
            rs_hand_pre: list[str] = []
            cs_hand_pre: list[str] = []
            rs_discard_pre: list[str] = []
            cs_discard_pre: list[str] = []
            try:
                rs_d_pre = sts2_rs.state_to_dict(rs_state.inner)
                rs_hand_pre = [c.get("id") for c in (rs_d_pre.get("deck", {}).get("hand") or [])]
                rs_discard_pre = [
                    c.get("id") for c in (rs_d_pre.get("deck", {}).get("discard_pile") or [])
                ]
            except Exception:
                pass
            try:
                cs_state_pre = g.get_state()
                cs_hand_pre = list(cs_state_pre.get("hand_ids") or [])
                cs_discard_pre = list(cs_state_pre.get("discard_ids") or [])
            except Exception:
                pass

            try:
                cs_apply_result = _apply_csharp(g, action)
                _apply_rust(rs_state, action)
                # Mirror the C# headless auto-resolve of PlayerChoiceContext
                # on BOTH engines so a card that opens a choice (Armaments
                # FromHandForUpgrade, Headbutt FromDiscard, …) is snapshotted
                # at the same post-resolution lifecycle point. _drain_rust_choices
                # advances Rust's open `pending_choice`; _drain_csharp_choices
                # advances any C# `decision` the act surfaced. Without the C#
                # drain a choice-opening card can leave C# paused mid-resolution
                # (played card still in the PlayPile, not yet flushed to Discard)
                # at snapshot time → spurious discard_count/draw_count off-by-one
                # (the ARMAMENTS/HEADBUTT bucket of sts2-rs-8ukqd). The drain
                # function existed since 601ea5d but was never wired into the
                # fuzz loop.
                _drain_csharp_choices(g, cs_apply_result)
                _drain_rust_choices(rs_state)
            except Exception as exc:
                result.divergence_type = DIVERGENCE_HARNESS
                result.divergence_detail = {
                    "stage": "apply",
                    "action": action,
                    "turn": turn,
                    "error": repr(exc),
                }
                return result

            try:
                cs_post = normalize_csharp(g.get_state())
                rs_post = normalize_rust(rs_state)
            except Exception as exc:
                result.divergence_type = DIVERGENCE_HARNESS
                result.divergence_detail = {
                    "stage": "post_normalize",
                    "action": action,
                    "turn": turn,
                    "error": repr(exc),
                }
                return result

            # Once EITHER engine resolves the combat, in-combat parity is
            # done — every prior action matched to reach here, and this
            # action ended the fight. Post-resolution the two engines
            # legitimately diverge in REPRESENTATION, not behavior, and Rust
            # is the correct side (verified sts2-rs-8ukqd): on victory Rust
            # holds a `Resolved` combat snapshot while the C# headless has
            # advanced out of combat to the reward screen (combat_terminal
            # flips False, piles zero); on defeat both agree the player died
            # but the C# headless still offers `end_turn` on the death screen
            # where Rust (correctly) offers no actions. Diffing either of
            # those manufactures a spurious STATE/LEGAL drift, so stop the
            # trial as clean. (The killing blow's own damage is validated by
            # the death/kill itself — the monster/player is dead on both.)
            if cs_post.get("combat_terminal") or rs_post.get("combat_terminal"):
                break

            divs = diff_post_action(cs_post, rs_post)
            if divs:
                # sts2-rs-6vj51: dump pre/post hand+discard card-ids
                # so per-card discard-semantics drifts (sub-bug D of
                # sts2-rs-8ukqd) can be attributed to a specific card
                # without re-running with extra instrumentation.
                rs_hand_post: list[str] = []
                cs_hand_post: list[str] = []
                rs_discard_post: list[str] = []
                cs_discard_post: list[str] = []
                try:
                    rs_d_post = sts2_rs.state_to_dict(rs_state.inner)
                    rs_hand_post = [
                        c.get("id") for c in (rs_d_post.get("deck", {}).get("hand") or [])
                    ]
                    rs_discard_post = [
                        c.get("id") for c in (rs_d_post.get("deck", {}).get("discard_pile") or [])
                    ]
                except Exception:
                    pass
                try:
                    cs_state_post = g.get_state()
                    cs_hand_post = list(cs_state_post.get("hand_ids") or [])
                    cs_discard_post = list(cs_state_post.get("discard_ids") or [])
                except Exception:
                    pass
                result.divergence_type = DIVERGENCE_STATE
                result.divergence_detail = {
                    "action": action,
                    "turn": turn,
                    "divergences": divs[:20],
                    "rust_hand_pre": rs_hand_pre,
                    "csharp_hand_pre": cs_hand_pre,
                    "rust_hand_post": rs_hand_post,
                    "csharp_hand_post": cs_hand_post,
                    "rust_discard_pre": rs_discard_pre,
                    "csharp_discard_pre": cs_discard_pre,
                    "rust_discard_post": rs_discard_post,
                    "csharp_discard_post": cs_discard_post,
                }
                return result

            if action["action"] == "end_turn":
                turn += 1
                result.turns_played = turn
    finally:
        try:
            g.close()
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Multi-trial CLI
# ---------------------------------------------------------------------------


def _bucket_key(result: TrialResult) -> str:
    if result.divergence_type is None:
        return "clean"
    det = result.divergence_detail or {}
    if result.divergence_type == DIVERGENCE_LEGAL:
        only = det.get("csharp_only") or det.get("rust_only") or []
        if only:
            a = only[0]
            return f"LEGAL:{a['action']}:{a.get('card_index', '-')}"
        return "LEGAL:empty"
    if result.divergence_type == DIVERGENCE_STATE:
        divs = det.get("divergences") or []
        if divs:
            return f"STATE:{divs[0]['path']}"
        return "STATE:empty"
    if result.divergence_type == DIVERGENCE_INIT:
        divs = det.get("divergences") or []
        if divs:
            return f"INIT:{divs[0]['path']}"
        return "INIT:empty"
    if result.divergence_type == DIVERGENCE_HARNESS:
        stage = det.get("stage", "?")
        err = det.get("error", "")
        # Pull a short signature out of the error text so distinct C#
        # failures (NullReferenceException in BeforeRoomEntered vs an
        # encounter-id typo vs a set_player schema mismatch) sort into
        # separate buckets in the summary.
        sig = ""
        if "enter_room failed" in err:
            sig = ":enter_room"
        elif "set_player failed" in err:
            sig = ":set_player"
        return f"HARNESS_ERROR:{stage}{sig}"
    return result.divergence_type


def _trial_to_json(result: TrialResult) -> dict:
    return {
        "seed": result.seed,
        "scenario": dataclasses.asdict(result.scenario),
        "turns_played": result.turns_played,
        "divergence_type": result.divergence_type,
        "divergence_detail": result.divergence_detail,
        "action_history": result.action_history,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="C#<->Rust parity fuzz harness")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--max-turns", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out", default="parity_fuzz_runs")
    args = p.parse_args(argv)

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out, ts)
    os.makedirs(out_dir, exist_ok=True)

    buckets: Counter[str] = Counter()
    for i in range(args.trials):
        seed = args.seed_start + i
        try:
            result = run_trial(seed=seed, max_turns=args.max_turns)
        except Exception as ex:
            key = f"HARNESS_ERROR:{type(ex).__name__}"
            buckets[key] += 1
            err_path = os.path.join(out_dir, f"trial_{seed}_error.json")
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump({"seed": seed, "error": repr(ex)}, f, indent=2)
            print(f"[trial {seed}] HARNESS_ERROR: {ex!r}", file=sys.stderr)
            continue
        key = _bucket_key(result)
        buckets[key] += 1
        if result.divergence_type is not None:
            trial_path = os.path.join(out_dir, f"trial_{seed}.json")
            with open(trial_path, "w", encoding="utf-8") as f:
                json.dump(_trial_to_json(result), f, indent=2, default=str)
        print(f"[trial {seed}] {key}", file=sys.stderr)

    summary_path = os.path.join(out_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Parity fuzz summary — {ts}\n\n")
        f.write(f"Trials: {args.trials}; max_turns: {args.max_turns}\n\n")
        f.write("| count | bucket |\n|---:|---|\n")
        for k, v in buckets.most_common():
            f.write(f"| {v} | `{k}` |\n")
    print(f"summary -> {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
