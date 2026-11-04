#!/usr/bin/env python3
"""C#↔Rust EVENT parity — the per-act event POOL and per-event OPTION lists.

Two complementary checks, both combat-free (no map walk, no winning combat —
so they cover ALL THREE ACTS, not just act 1, which the lockstep whole-run
sweep can only reach when random play happens to clear the bosses):

  POOL parity  — does each act's shuffled event pool (the `RoomSet.events`
    list, `(AllEvents ++ AllSharedEvents)` UnstableShuffled off the `up_front`
    stream in C# `ActModel.GenerateRooms`) match Rust's `events_shuffled`
    entry-for-entry, in order, across acts 1/2/3? This catches a wrong event
    set OR a wrong shuffle (RNG misalignment). Uses the headless `dump_rooms`
    command vs the PyO3 `generated_rooms()` accessor — both pre-generated at
    run start, no traversal needed.

  OPTION parity — for every event in the union of those pools, force the
    event on each engine and diff its presented options. C# side:
    `enter_room {type: event, event: <Id.Entry>}` (`RunSimulator.EnterRoomImpl`)
    → the `event_choice` decision view. Rust side: the new
    `RunManager.force_event(<python_name>)` + `event_option_infos()`. We
    compare, by option position:
      * OPTION COUNT                       (hard — a missing/extra option)
      * the ENABLED / locked vector        (hard — C# `is_locked` ⇔ Rust
                                            `!enabled`; a wrongly (un)locked
                                            option is a real divergence)
    and report, advisory-only:
      * option_id STRING drift             (Rust emits a short canonical
                                            `option_id` e.g. `REACH`; C# emits
                                            the loc-key tail `text_key` e.g.
                                            `REACH_INTO_THE_FLESH` — different
                                            id namespaces BY DESIGN, same as the
                                            accepted SelfHelpBook READ_PASSAGE
                                            case, so this never fails the run).

  EFFECT parity (``--effect``) — option count + enabled is necessary but not
    sufficient: a handler can present an identical menu while granting the wrong
    relic, no-opping a deck mutation, finishing a multi-page chain early, or
    skipping a nested combat. With ``--effect`` we additionally RESOLVE every
    option (one fresh forced entry per option) on both engines and diff the
    resulting state — relics, deck multiset, potions, gold, hp, the landed
    phase, and the multi-page follow-up option count. This is the check that
    catches the `sts2-rs-m3n0g` event-side stubs that `OPTION parity` is blind
    to. Combat-free, so it covers all three acts.

Usage:
    scripts/wheel-venv.sh run -- python python/event_parity.py
    scripts/wheel-venv.sh run -- python python/event_parity.py --all-characters
    scripts/wheel-venv.sh run -- python python/event_parity.py --pool-seeds 200
    scripts/wheel-venv.sh run -- python python/event_parity.py --effect
    # effect check only (skip the pool/option sweeps):
    scripts/wheel-venv.sh run -- python python/event_parity.py --no-pool --no-options --effect
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any

sys.path.insert(0, "python")

import sts2_client
from run_parity import Snap, diff_snaps
from sts2_client import Game
from workbench.backends.rust_game import RustGame


def _rust_run_manager(character: str, seed: str, ascension: int) -> Any:
    """A fresh Rust `RunManager` for the build the C# oracle runs.

    ``game_version`` is the selected build (``STS2_GAME_VERSION``), like
    `act23_map_parity._rust_act_maps` (GH #893). Without it the pool and option
    checks compared a v0.111.0 C# run against a v0.107.1 Rust run, so every
    v0.111.0 act pool "diverged" in order.
    """
    import sts2_rs  # type: ignore[import-not-found]  # lazy: keep module wheel-free to import

    return sts2_rs.RunManager(
        string_seed=seed,
        character=character,
        ascension_level=ascension,
        neow_start=True,
        game_version=sts2_client.GAME_VERSION,
    )


def _rust_backend() -> RustGame:
    """Rust backend in C# convention so its serialized views share the exact
    decision/player shape the C# `Game` emits (what `Snap.of` consumes)."""
    return RustGame(target_convention="csharp")


# Characters as the C# headless / PyO3 wrapper name them (matches
# `run_parity.CHARACTERS`, the ported roster).
ALL_CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]

#: Events whose OPTION STRUCTURE (count and/or locked state) depends on
#: accumulated run state — potions / relics / enchantable cards / gold / shop
#: stock — that a bare fresh-run force-entry CANNOT reproduce. Each is gated by
#: an `IsAllowed` precondition (verified against the v0.103.2 decomp) that
#: force-entry deliberately bypasses, so the event gets presented in a state it
#: would NEVER naturally occur in; its options then differ (locked options, or
#: an empty/fallback trade stock) for reasons that vanish in real play, where
#: the event only spawns once its precondition holds. The option COUNT +
#: enabled-vector check is therefore meaningless for these under force-entry —
#: we run it (so a regression in the *other* fields is still visible in the
#: dump) but classify any diff as ADVISORY, never a hard failure. Their pool
#: membership + shuffle order is still hard-checked by `pool_parity`. Mirrors
#: `spawn_roster_parity.RNG_BRANCHED_ALLOWLIST`.
#:
#: C# precondition (`…Models.Events/<Event>.cs`):
#:   StoneOfAllTime      — IsAllowed: act 2 AND every player has a potion
#:                         (LIFT locks when `DrinkAndLiftPotion == null`).
#:   RanwidTheElder      — IsAllowed: act != 1 AND `GetValidRelics(p).Any()`
#:                         (POTION/RELIC options lock without a potion/relic).
#:   GraveOfTheForgotten — IsAllowed: every player `HasEnchantableCards`
#:                         (CONFRONT enchants — locks with no enchantable card).
#:   ZenWeaver           — IsAllowed: `Gold >= EmotionalAwarenessCost`; the two
#:                         pricier options lock when `gold < cost`.
#:   RelicTrader         — stock = `GetValidRelics(Owner).Take(3)`; PROCEED-only
#:                         fallback with no tradeable relics.
#:   TheFutureOfPotions  — stock built from `Owner.Potions`; empty with none.
#:   FakeMerchant        — builds a card/relic/potion buy stock (needs gold).
CONTEXT_DEPENDENT_EVENTS = frozenset(
    {
        "StoneOfAllTime",
        "RanwidTheElder",
        "GraveOfTheForgotten",
        "ZenWeaver",
        "RelicTrader",
        "TheFutureOfPotions",
        "FakeMerchant",
    }
)

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def _canon(s: str | None) -> str:
    """Canonicalize a loc key / display string / option id to a bare id.

    Strips a trailing ``.title``/``.name``/``.description`` suffix, takes the
    rightmost ``.``-segment, uppercases, drops non-alphanumerics. Mirrors
    ``parity_walk._loc_id`` so ``REACH_INTO_THE_FLESH`` → ``REACHINTOTHEFLESH``
    and ``EVT.pages.INITIAL.options.REACH`` → ``REACH``.
    """
    if not s:
        return ""
    for suffix in (".title", ".name", ".description"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return _NON_ALNUM.sub("", s.rsplit(".", 1)[-1].upper())


# ─── pool parity ────────────────────────────────────────────────────────────


def _cs_pools(character: str, seed: str, ascension: int) -> list[list[str]]:
    """Per-act C# `RoomSet.events` lists (Id.Entry strings)."""
    g = Game()
    try:
        g.start(character=character, seed=seed, ascension=ascension)
        acts = g.send({"cmd": "dump_rooms"}).get("acts", [])
    finally:
        g.close()
    return [list(a.get("events", [])) for a in acts]


def _rust_pools(character: str, seed: str, ascension: int) -> list[list[str]]:
    """Per-act Rust `events_shuffled` lists (EventId python names)."""
    rm = _rust_run_manager(character, seed, ascension)
    return [list(a.get("events_shuffled", [])) for a in rm.generated_rooms()]


def pool_parity(characters: list[str], base: str, ascension: int, n: int) -> int:
    """Diff each act's shuffled event pool, order-exact. Returns failure count."""
    acts_ok = acts_total = fails = 0
    for character in characters:
        for i in range(n):
            seed = base if n == 1 else f"{base}{i}"
            cs = _cs_pools(character, seed, ascension)
            rs = _rust_pools(character, seed, ascension)
            for ai in range(min(len(cs), len(rs))):
                ce = [_canon(x) for x in cs[ai]]
                re_ = [_canon(x) for x in rs[ai]]
                acts_total += 1
                if ce == re_:
                    acts_ok += 1
                    continue
                fails += 1
                only_cs = sorted(set(ce) - set(re_))
                only_rs = sorted(set(re_) - set(ce))
                print(f"[{character} {seed} act{ai + 1}] POOL DIFF (cs={len(ce)} rs={len(re_)})")
                if only_cs or only_rs:
                    print(f"    only_cs={only_cs}")
                    print(f"    only_rs={only_rs}")
                else:  # same set, different ORDER → shuffle/RNG misalignment
                    print(f"    order cs={ce}")
                    print(f"          rs={re_}")
    print(
        f"\npool parity: {acts_ok}/{acts_total} act-pools order-exact across {len(characters)} char(s)"
    )
    return fails


# ─── option parity ───────────────────────────────────────────────────────────


def _event_universe(character: str, seed: str, ascension: int) -> list[tuple[str, str]]:
    """Union of all acts' events → sorted ``(rust_python_name, cs_entry)`` pairs.

    Paired by `_canon` (the pool check already proved the canon ids align), so
    each event is forced on both engines with its engine-native id.
    """
    rs = _rust_pools(character, seed, ascension)
    cs = _cs_pools(character, seed, ascension)
    cs_by_canon = {_canon(e): e for act in cs for e in act}
    pairs: dict[str, tuple[str, str]] = {}
    for act in rs:
        for name in act:
            c = _canon(name)
            if c in cs_by_canon:
                pairs[c] = (name, cs_by_canon[c])
    return [pairs[k] for k in sorted(pairs)]


def _rust_options(character: str, seed: str, ascension: int, name: str) -> list[tuple[str, bool]]:
    """``(canon_option_id, enabled)`` for a forced Rust event, in order."""
    rm = _rust_run_manager(character, seed, ascension)
    if rm.force_event(name) is None:
        raise LookupError(f"rust: unknown event {name!r}")
    opts = rm.event_option_infos() or []
    return [(_canon(o.get("option_id")), bool(o.get("enabled", True))) for o in opts]


def _cs_options(character: str, seed: str, ascension: int, entry: str) -> list[tuple[str, bool]]:
    """``(canon_text_key, enabled)`` for a C# enter_room event, in order."""
    g = Game()
    try:
        g.start(character=character, seed=seed, ascension=ascension)
        st = g.enter_room("event", event=entry)
        if st.get("type") == "error":
            raise LookupError(f"csharp: enter_room {entry!r}: {st.get('message')}")
        opts = st.get("options") or []
        # C# `is_locked == True` ⇒ option unavailable ⇒ Rust `enabled == False`.
        return [
            (_canon(o.get("text_key") or o.get("title")), not o.get("is_locked", False))
            for o in opts
        ]
    finally:
        g.close()


def option_parity(characters: list[str], seed: str, ascension: int) -> int:
    """Force every event on both engines; diff option count + enabled vector.

    Returns the count of HARD failures (count or enabled-vector mismatch).
    option_id string drift is reported advisory-only and never fails.
    """
    hard_fails = id_drift = checked = ctx_advisory = 0
    for character in characters:
        universe = _event_universe(character, seed, ascension)
        for rust_name, cs_entry in universe:
            try:
                rs = _rust_options(character, seed, ascension, rust_name)
                cs = _cs_options(character, seed, ascension, cs_entry)
            except LookupError as e:
                print(f"[{character}] SKIP {rust_name}: {e}")
                continue
            count_ok = len(rs) == len(cs)
            enabled_ok = count_ok and [e for _, e in rs] == [e for _, e in cs]
            ids_ok = count_ok and [i for i, _ in rs] == [i for i, _ in cs]
            if rust_name in CONTEXT_DEPENDENT_EVENTS:
                # Precondition-gated: a bare force-entry shows it in an
                # impossible state, so count/enabled diffs are expected and
                # advisory only (pool_parity still hard-checks its presence).
                if not (count_ok and enabled_ok):
                    ctx_advisory += 1
                    print(f"[{character}] ctx-advisory {rust_name}: rs={rs} cs={cs}")
                continue
            checked += 1
            if not (count_ok and enabled_ok):
                hard_fails += 1
                print(f"[{character}] OPTION DIFF {rust_name} (cs={cs_entry})")
                if not count_ok:
                    print(f"    count cs={len(cs)} rs={len(rs)}")
                print(f"    rs={rs}")
                print(f"    cs={cs}")
            elif not ids_ok:
                id_drift += 1  # cosmetic: short option_id vs loc-key tail
    print(
        f"\noption parity: {checked - hard_fails}/{checked} non-context events match on "
        f"count+enabled ({id_drift} with advisory option_id drift, "
        f"{ctx_advisory} context-dependent advisory) across {len(characters)} char(s)"
    )
    return hard_fails


# ─── effect parity ───────────────────────────────────────────────────────────
#
# Option STRUCTURE (count + enabled) matching is necessary but not sufficient:
# a handler that grants the wrong relic, no-ops a deck mutation, finishes a
# multi-page chain early, or skips a nested combat presents an IDENTICAL menu
# while doing the wrong thing. `option_parity` (above) is blind to all of it —
# it never resolves an option. This sweep DOES: for every event, on a fresh
# forced entry, it resolves EACH option (one clean run per option) on both
# engines and diffs the resulting state (relics, deck, potions, gold, hp, the
# landed phase, and multi-page follow-up options). This is the check that
# would have caught the `sts2-rs-m3n0g` event-side stubs.


def _follow_options(view: dict) -> int:
    """Count of follow-up options still presented after a choice (multi-page)."""
    return len(view.get("options") or [])


def _effect_fields(cs_snap: Snap, cs_view: dict, rs_snap: Snap, rs_view: dict) -> list[str]:
    """All effect divergences for one resolved option (empty == faithful).

    `diff_snaps` covers the carried state (relics / deck / potions / gold / hp /
    landed phase). On top of that, when BOTH engines stay in the event phase we
    compare the follow-up option count — a multi-page chain that diverges
    mid-flow (the finish-vs-continue case is already caught by the `phase`
    diff). Pure (no engine) so it's unit-testable.
    """
    fields = diff_snaps(cs_snap, rs_snap)
    if cs_snap.phase == "event" and rs_snap.phase == "event":
        cf, rf = _follow_options(cs_view), _follow_options(rs_view)
        if cf != rf:
            fields.append(f"followup_options: cs={cf} rs={rf}")
    return fields


def _rust_effect(
    character: str, seed: str, ascension: int, name: str, idx: int
) -> tuple[Snap, dict]:
    """Force `name`, resolve option `idx`, return (post-choice Snap, view)."""
    r = _rust_backend()
    try:
        r.start(character=character, seed=seed, ascension=ascension)
        entered = r.force_event(name)
        if entered.get("error") or entered.get("type") == "error":
            raise LookupError(f"rust: force_event {name!r}: {entered.get('error')}")
        post = r.act("choose_option", option_index=idx)
        if post.get("error") or post.get("type") == "error":
            raise RuntimeError(f"rust: choose_option {idx}: {post.get('error')}")
        return Snap.of(post, "rs"), post
    finally:
        r.close()


def _cs_effect(
    character: str, seed: str, ascension: int, entry: str, idx: int
) -> tuple[Snap, dict]:
    """Enter `entry`, resolve option `idx`, return (post-choice Snap, view)."""
    g = Game()
    try:
        g.start(character=character, seed=seed, ascension=ascension)
        pre = g.enter_room("event", event=entry)
        if pre.get("type") == "error":
            raise LookupError(f"csharp: enter_room {entry!r}: {pre.get('message')}")
        post = g.choose_option(idx)
        if post.get("type") == "error":
            raise RuntimeError(f"csharp: choose_option {idx}: {post.get('message')}")
        return Snap.of(post, "cs"), post
    finally:
        g.close()


def effect_parity(characters: list[str], seed: str, ascension: int) -> int:
    """Resolve every option of every event on both engines; diff the effect.

    Returns the count of HARD failures (any state-delta divergence on a
    resolved option). Context-dependent events are skipped (their bare
    force-entry state is impossible — same rationale as `option_parity`).
    """
    hard_fails = checked = 0
    for character in characters:
        universe = _event_universe(character, seed, ascension)
        for rust_name, cs_entry in universe:
            if rust_name in CONTEXT_DEPENDENT_EVENTS:
                continue
            try:
                rs_opts = _rust_options(character, seed, ascension, rust_name)
                cs_opts = _cs_options(character, seed, ascension, cs_entry)
            except LookupError as e:
                print(f"[{character}] SKIP {rust_name}: {e}")
                continue
            # Resolve by index, so the options must align positionally. The
            # contract for that is exactly `option_parity`'s HARD contract:
            # same count + same enabled vector (option_id STRINGS legitimately
            # differ by namespace — `GRAB` vs `GRABPOTIONS` — which is advisory,
            # NOT a reorder). A count/enabled mismatch is already a hard failure
            # there, so skip the effect resolve rather than double-report.
            if len(rs_opts) != len(cs_opts) or [e for _, e in rs_opts] != [e for _, e in cs_opts]:
                print(
                    f"[{character}] SKIP-EFFECT {rust_name}: option count/enabled differ "
                    "(see option parity)"
                )
                continue
            for idx in range(len(rs_opts)):
                # Only resolve an option enabled (unlocked) on BOTH engines.
                if not (rs_opts[idx][1] and cs_opts[idx][1]):
                    continue
                oid = rs_opts[idx][0] or idx
                try:
                    rs_snap, rs_view = _rust_effect(character, seed, ascension, rust_name, idx)
                    cs_snap, cs_view = _cs_effect(character, seed, ascension, cs_entry, idx)
                except (LookupError, RuntimeError) as e:
                    hard_fails += 1
                    print(f"[{character}] EFFECT RESOLVE-ERROR {rust_name}[{idx}={oid}]: {e}")
                    continue
                checked += 1
                fields = _effect_fields(cs_snap, cs_view, rs_snap, rs_view)
                if fields:
                    hard_fails += 1
                    print(f"[{character}] EFFECT DIFF {rust_name}[{idx}={oid}] (cs={cs_entry})")
                    for f in fields:
                        print(f"    {f}")
    print(
        f"\neffect parity: {checked - hard_fails}/{checked} resolved options match "
        f"across {len(characters)} char(s)"
    )
    return hard_fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--all-characters", action="store_true", help="run all 5 characters")
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--pool-seeds", type=int, default=50, help="seeds for POOL parity sweep")
    ap.add_argument("--base", default="EVTPAR", help="seed prefix for the pool sweep")
    ap.add_argument("--no-options", action="store_true", help="skip the per-event option check")
    ap.add_argument("--no-pool", action="store_true", help="skip the event-pool sweep")
    ap.add_argument(
        "--effect",
        action="store_true",
        help="ALSO resolve every option and diff the state EFFECT (relics/deck/"
        "potions/gold/hp/phase/multi-page) — catches handler bugs the menu check misses",
    )
    args = ap.parse_args()

    from parity_audit import assert_wheel_fresh  # lazy: keeps the module wheel-free to import

    assert_wheel_fresh()
    characters = ALL_CHARACTERS if args.all_characters else [args.character]

    fails = 0
    if not args.no_pool:
        fails += pool_parity(characters, args.base, args.ascension, args.pool_seeds)
    if not args.no_options:
        # Option lists on a fresh run are deterministic; one seed per character
        # exercises every event's option structure.
        fails += option_parity(characters, f"{args.base}0", args.ascension)
    if args.effect:
        # Effects on a fresh forced entry are deterministic too; one seed per
        # character resolves every option of every event.
        fails += effect_parity(characters, f"{args.base}0", args.ascension)
    print("\n✓ EVENT PARITY CLEAN" if fails == 0 else f"\n✗ {fails} EVENT PARITY FAILURE(S)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
