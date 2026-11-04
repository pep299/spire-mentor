"""Scope check for wb-27d55ac0e5e1df7f: is the anti-upgraded enchant prior
Necrobinder-only, or does every deployed champion carry it?

Same instrument as `repro_wb27d55ac0.py`, reduced to the one paired
measurement that matters: a deck holding the SAME card twice, once base and
once upgraded, offered as a whole-deck ENCHANT. Under Perfect Fit / Adroit /
Sharp the upgraded copy is the strictly better target, so a champion that
ranks base above upgraded is wrong on this surface.

Run from sts2-pycli:  uv run python python/repro_wb27d55ac0_chars.py
"""

from __future__ import annotations

import champions
import sts2_rs

# Upgrade exactly ONE starter Strike, leaving its siblings base. Starter
# Strikes carry real probability mass on every champion, so the paired
# base-vs-upgraded comparison is not a ratio between two rounding errors —
# which is what an unloved rare card gives you.
PAIR_CARD = {
    "Ironclad": "STRIKE_IRONCLAD",
    "Silent": "STRIKE_SILENT",
    "Defect": "STRIKE_DEFECT",
    "Regent": "STRIKE_REGENT",
    "Necrobinder": "STRIKE_NECROBINDER",
}


def deck_of(rm):
    return list(rm.master_deck_detailed())


def build(character: str, pair_card: str):
    rm = sts2_rs.RunManager(
        seed=12311,
        character=character,
        mode="agent",
    )
    # Starter deck as-is; upgrade one copy of the pair card.
    if not rm.force_upgrade_card(pair_card):
        raise SystemExit(f"{character}: could not upgrade {pair_card}")
    before = rm.pending_deck_enchants
    rm.force_add_relic("KIFUDA")  # Adroit — unrestricted, whole-deck offer
    if rm.pending_deck_enchants <= before:
        raise SystemExit(f"{character}: Kifuda queued no enchant")
    return rm


def main() -> int:
    import workbench.ppo_policy as pp

    print(
        f"{'character':14s} {'card':20s} {'base':>8s} {'upgraded':>9s} {'ratio':>9s}  best option"
    )
    for character, pair_card in PAIR_CARD.items():
        if champions.onnx_path(character) is None:
            print(f"{character:14s} (no committed champion)")
            continue
        rm = build(character, pair_card)
        ev = pp.evaluate(rm, character=character, ascension=0, impl="rust")
        if ev is None or ev.probs is None:
            print(f"{character:14s} ERROR {ev.error if ev else 'declined'}")
            continue
        deck = deck_of(rm)
        idxs = rm.enchant_select_deck_indices()
        labels = [deck[idxs[int(a["idx"])]] for a in ev.actions]
        base = upg = None
        for (cid, up), p in zip(labels, ev.probs):
            if cid.replace("_", "").lower() == pair_card.replace("_", "").lower():
                if up:
                    upg = p
                else:
                    base = p
        if base is None or upg is None:
            print(
                f"{character:14s} {pair_card:20s} pair not offered (labels={set(c for c, _ in labels)})"
            )
            continue
        bcid, bup = labels[ev.chosen]
        ratio = base / upg if upg else float("inf")
        print(
            f"{character:14s} {pair_card:20s} {base * 100:7.2f}% {upg * 100:8.2f}% "
            f"{ratio:8.1f}x  {bcid}{'+' if bup else ''}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
