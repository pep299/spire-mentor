"""Repro for workbench bug report wb-27d55ac0e5e1df7f.

"Perfect Fit" (FieldOfManSizedHoles -> Enter Your Hole) opens an ENCHANT
card_select over the whole deck. The PPO panel ranked `SLEIGHT_OF_FLESH`
at 26.0% and `SLEIGHT_OF_FLESH+` at 0.1% — the upgraded copy of the very
same card, on a surface where "upgraded" is strictly better.

Rebuilds the reported deck against the live rust engine, opens a whole-deck
enchant, and reads the deployed Necrobinder champion's distribution:

  A — the reported decision.
  B — control: de-upgrade one copy so the deck holds TWO identical base
      copies. If they then score equally, slot position contributes
      nothing and the upgrade bit is the whole story.
  C — control: the SAME deck offered as card REMOVAL. Removal is where
      "upgraded => don't pick" is CORRECT; matching polarity means the
      prior is shared across two opposite questions.
  D — causal test: hold state and actions fixed, rewrite ONLY the surface
      integer, and sweep it through all 15 surfaces.

Findings: docs/investigations/wb-27d55ac0-enchant-prefers-base-copy.md
Run from sts2-pycli:  uv run python python/repro_wb27d55ac0.py
"""

from __future__ import annotations

import sys

import sts2_rs

SEED = 12311
CHARACTER = "Necrobinder"

# The 24-card master deck from the report, in report order. (card_id, upgraded)
REPORT_DECK: list[tuple[str, bool]] = [
    ("STRIKE_NECROBINDER", False),
    ("STRIKE_NECROBINDER", False),
    ("STRIKE_NECROBINDER", False),
    ("DEFEND_NECROBINDER", True),
    ("DEFEND_NECROBINDER", True),
    ("DEFEND_NECROBINDER", True),
    ("DEFEND_NECROBINDER", False),
    ("BODYGUARD", True),
    ("UNLEASH", False),
    ("HIGH_FIVE", False),
    ("COUNTDOWN", False),
    ("SCULPTING_STRIKE", False),
    ("GRAVE_WARDEN", False),
    ("AFTERLIFE", False),
    ("DRAMATIC_ENTRANCE", False),
    ("BONE_SHARDS", False),
    ("SLEIGHT_OF_FLESH", True),
    ("HANG", False),
    ("SLEIGHT_OF_FLESH", False),
    ("SCOURGE", False),
    ("HIGH_FIVE", True),
    ("PULL_AGGRO", True),
    ("LANTERN_KEY", False),
    ("RATTLE", False),
]

REPORT_RELICS = [
    "BOUND_PHYLACTERY",
    "POMANDER",
    "GAME_PIECE",
    "BOOK_REPAIR_KNIFE",
    "SOZU",
    "BRONZE_SCALES",
    "MANGO",
    "LETTER_OPENER",
]


def norm(card_id: str) -> str:
    """`master_deck_detailed` emits the Debug enum name (`StrikeNecrobinder`);
    the report uses UPPER_SNAKE. Compare case/underscore-insensitively."""
    s = card_id.replace("_", "").lower()
    # A few engine enum names carry a `Card` suffix the report id does not
    # (`COUNTDOWN` -> `CountdownCard`).
    return s[: -len("card")] if s.endswith("card") and s != "card" else s


def deck_of(rm) -> list[tuple[str, bool]]:
    return list(rm.master_deck_detailed())


def label(entry: tuple[str, bool]) -> str:
    cid, up = entry
    return f"{cid}{'+' if up else ''}"


def build_rm(deck: list[tuple[str, bool]]):
    """A Necrobinder run whose master deck is exactly `deck`, in order."""
    rm = sts2_rs.RunManager(
        seed=SEED,
        character=CHARACTER,
        mode="agent",
        interactive_potion_swap=True,
    )
    for cid, _ in deck_of(rm):
        rm.force_remove_card(cid)
    left = deck_of(rm)
    if left:
        raise SystemExit(f"starter deck not fully stripped: {left}")

    # Grant the relics while the deck is EMPTY. `force_add_relic` fires
    # `after_obtained` eagerly, and Pomander queues a deck upgrade — on a
    # populated deck that upgrade decision sits in front of the enchant we
    # want to measure (and would mutate the deck if drained).
    for rid in REPORT_RELICS:
        rm.force_add_relic(rid)

    # Lay the deck down left-to-right, then upgrade. `force_upgrade_card`
    # hits the FIRST un-upgraded match, so the upgraded copies of a repeated
    # card land front-most — which is the report's own layout for DEFEND
    # (3 upgraded, then 1 not) and for SLEIGHT_OF_FLESH (16 up, 18 not).
    for cid, _ in deck:
        rm.force_add_card(cid)
    if len(deck_of(rm)) != len(deck):
        raise SystemExit(f"deck size {len(deck_of(rm))} != {len(deck)}")
    want_counts: dict[str, int] = {}
    for cid, upgraded in deck:
        if upgraded:
            want_counts[norm(cid)] = want_counts.get(norm(cid), 0) + 1
    for cid_n, n in want_counts.items():
        for _ in range(n):
            if not rm.force_upgrade_card(cid_n):
                raise SystemExit(f"could not upgrade {cid_n}")

    rm.set_player(2, 80, 442)
    pend = (rm.pending_deck_upgrades, rm.pending_deck_removals, rm.pending_deck_transforms)
    if any(pend):
        raise SystemExit(f"deck-mutation queue not empty after build: {pend}")
    return rm


def open_enchant(rm) -> None:
    """Queue an unrestricted whole-deck enchant card_select.

    The report's own trigger is FieldOfManSizedHoles -> "Enter Your Hole"
    (`events/act2_hive.rs:533`), but `force_event` only binds `current_event`;
    it does not put the run in `Phase::Event`, so the `event_choice` action is
    not offered without a real map walk.

    Kifuda (`relics/shop_event.rs:3074`) queues the same thing through
    `try_queue_interactive_enchant` with `Adroit`, which — like Perfect Fit —
    carries NO card-type restriction, so the offer is the whole 24-card deck.

    This substitution is exact from the net's point of view, and that is the
    second finding: `PendingDeckEnchant.enchantment` is featurized NOWHERE.
    `obs` carries only `enchant_select_deck_indices` (`ppo/env.py:175`,
    `ppo/obs.rs:264`), and the action token is `card_tok(card_id, upgraded, 0)`
    either way (`ppo/encoder.py:319`). Perfect Fit and Adroit produce byte-
    identical features over the same deck — as would Goopy, whose "adds
    exhaust" is a DOWNSIDE and inverts which card you want to pick.
    """
    before = rm.pending_deck_enchants
    rm.force_add_relic("KIFUDA")
    if rm.pending_deck_enchants <= before:
        raise SystemExit("Kifuda did not queue an interactive enchant")


def option_labels(rm, actions: list[dict]) -> list[str]:
    """Label each masked action with the deck card it actually targets.

    Keyed off the action KIND, not the coarse `classify_surface` class — that
    collapses every deck mutation to `"card-pick"`.
    """
    kinds = {a.get("action") for a in actions}
    deck = deck_of(rm)
    if kinds == {"enchant_deck_card"}:
        idxs = rm.enchant_select_deck_indices()
        return [label(deck[idxs[int(a["idx"])]]) for a in actions]
    if kinds == {"remove_deck_card"}:
        opts = rm.card_removal_options()
        return [label(opts[int(a["idx"])]) for a in actions]
    return []


def show(rm, title: str, impl: str = "rust") -> tuple[str, list[str], list[float]] | None:
    import workbench.ppo_policy as pp

    ev = pp.evaluate(rm, character=CHARACTER, ascension=0, impl=impl)
    print(f"\n=== {title} [{impl}] ===")
    if ev is None:
        print("  PPO declined (no distribution)")
        return None
    if ev.probs is None:
        print(f"  ERROR: {ev.error}")
        return None
    labels = option_labels(rm, ev.actions)
    kind = sorted({a.get("action") for a in ev.actions})
    print(
        f"  classify_surface={ev.surface}  kinds={kind}  phase={ev.phase}  V={ev.value}  n={len(ev.probs)}"
    )
    for i, p in enumerate(ev.probs):
        lab = labels[i] if i < len(labels) else str(ev.actions[i])
        star = "  <-- BEST" if i == ev.chosen else ""
        print(f"  {i:2d} {lab:28s} {p * 100:6.2f}%{star}")
    return ev.surface, labels, list(ev.probs)


def paired(labels: list[str], probs: list[float]) -> None:
    """Every card id that appears both upgraded and not, side by side."""
    by: dict[str, dict[bool, float]] = {}
    for lab, p in zip(labels, probs):
        up = lab.endswith("+")
        by.setdefault(lab.rstrip("+"), {})[up] = p
    rows = [(c, d[False], d[True]) for c, d in by.items() if True in d and False in d]
    if not rows:
        print("  (no paired card id in this deck)")
        return
    print("  paired — same card id, upgrade bit flipped:")
    for cid, base, upg in rows:
        ratio = (base / upg) if upg else float("inf")
        print(f"    {cid:24s} base {base * 100:6.2f}%   upgraded {upg * 100:6.2f}%   {ratio:8.1f}x")


def surface_sweep(rm) -> None:
    """Experiment D — how much does the SURFACE token actually move the answer?

    An `enchant_deck_card` option and a `remove_deck_card` option featurize to
    byte-identical action tokens: both are `card_tok(card_id, upgraded, 0)`
    (`ppo/encoder.py`, and its rust mirror `encoder.rs:495-515`). The ONLY
    thing in the whole feature tensor that says "this is enchant, not
    removal" is one integer — `state_cat[8] = surface_index(fine_surface())`.

    So: hold the state and the action list fixed, rewrite that one integer to
    each surface in turn, and watch the distribution. If the numbers barely
    move, the net is not conditioning on the surface at all — it is answering
    whatever question the card tokens alone imply.
    """
    import ppo.encoder as enc
    import workbench.ppo_policy as pp
    from ppo import vocab
    from ppo.env import mask_rejectable, metrics_snapshot, observe

    policy = pp._get_policy(CHARACTER)
    if policy is None:
        print("\n=== D. surface sweep === torch backend unavailable:", pp._policy_errors)
        return

    phase = rm.phase()
    full = rm.available_actions()
    masked = mask_rejectable(rm, phase, full)
    obs = observe(rm, phase, "card-pick", seed=int(rm.seed), ascension=0)
    metrics = metrics_snapshot(rm)
    labels = option_labels(rm, masked)

    print("\n=== D. surface sweep — same state, same actions, one integer changed ===")
    print(f"  (baseline fine_surface = {enc.fine_surface(masked)!r})")
    header = f"  {'surface':16s}"
    tracked = [i for i, lab in enumerate(labels) if lab.startswith("SleightOfFlesh")]
    for i in tracked:
        header += f" {labels[i]:>22s}"
    print(header)

    original = dict(enc._KIND_SURFACE)
    try:
        for surf in vocab.SURFACES:
            enc._KIND_SURFACE["enchant_deck_card"] = surf
            probs, _ = policy.distribution(obs, masked, metrics)
            row = f"  {surf:16s}"
            for i in tracked:
                row += f" {probs[i] * 100:21.2f}%"
            print(row)
    finally:
        enc._KIND_SURFACE.clear()
        enc._KIND_SURFACE.update(original)


def main() -> int:
    rm = build_rm(REPORT_DECK)
    got = deck_of(rm)
    print("deck as built:")
    for i, e in enumerate(got):
        print(f"  {i:2d} {label(e)}")
    want = sorted((norm(c), u) for c, u in REPORT_DECK)
    have = sorted((norm(c), u) for c, u in got)
    if want != have:
        print("\n!! deck multiset differs from the report", file=sys.stderr)
        print(f"   want {want}", file=sys.stderr)
        print(f"   have {have}", file=sys.stderr)
        return 1
    # `force_upgrade_card` upgrades the FIRST un-upgraded match, so a repeated
    # card's upgraded copy always lands front-most. That reproduces the report
    # for DEFEND and SLEIGHT_OF_FLESH but inverts HIGH_FIVE (report: base at 9,
    # upgraded at 20). Useful rather than harmful: HIGH_FIVE now varies slot
    # order independently of the other two pairs.
    if [label(e) for e in got] != [f"{c}{'+' if u else ''}" for c, u in REPORT_DECK]:
        print("  (note: slot order differs from the report for repeated cards — see above)")

    # --- C: removal control on the SAME deck, before the event is forced.
    rmC = build_rm(REPORT_DECK)
    rmC.force_add_relic("PRECISE_SCISSORS")
    resC = show(rmC, "C. CONTROL — card removal on the same deck")
    if resC:
        paired(resC[1], resC[2])

    # --- A: the reported enchant decision.
    open_enchant(rm)
    print(
        f"\npending_deck_enchants={rm.pending_deck_enchants}  options={len(rm.enchant_select_deck_indices())}"
    )
    resA = show(rm, "A. ENCHANT over the whole deck (Perfect Fit-shaped)")
    if resA:
        paired(resA[1], resA[2])

    # --- B: de-upgrade the SLEIGHT_OF_FLESH copy at slot 16, so the deck
    # holds TWO identical base copies. Everything else is byte-identical to
    # A. If they now score ~equally and near A's 26%, the upgrade bit is the
    # entire explanation for A's 260x split; anything else means a slot
    # artifact is in play.
    both_base = list(REPORT_DECK)
    both_base[16] = ("SLEIGHT_OF_FLESH", False)
    rmB = build_rm(both_base)
    open_enchant(rmB)
    resB = show(rmB, "B. CONTROL — same deck, both Sleight of Flesh copies BASE")
    if resB:
        for lab, p in zip(resB[1], resB[2]):
            if lab.startswith("SleightOfFlesh") or lab.startswith("SLEIGHT"):
                print(f"    {lab:28s} {p * 100:6.2f}%")

    # --- A again through the torch backend, then D.
    show(rm, "A'. enchant — torch backend (parity check on A)", impl="python")
    surface_sweep(rm)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
