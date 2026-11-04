"""Dump the C# oracle's enchantment-eligibility table (GH #1021).

For every card the Rust registry knows (`sts2_rs.card_kinds()`) and every
v0.107.1 enchantment, ask the headless C# game to put the card in a deck
and enchant it (`set_player` with ``CARD@ENCH:1``, which runs
`CardCmd.Enchant` and so `EnchantmentModel.CanEnchant`). The result is
the fixture `sts2-rs/crates/sts2-content/tests/fixtures/enchant_eligibility_v0107.json`
that `enchant_eligibility_1021.rs` checks `can_enchant_card` against.

    python python/dump_enchant_eligibility.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sts2_rs
from parity_fuzz import _csharp_card_id
from sts2_client import Game

#: Every v0.107.1 `Models.Enchantments` class with a Rust `EnchantmentId`
#: (Rust's `Favored` has no v0.107.1 class; `DeprecatedEnchantment` is a
#: save-load stand-in).
ENCHANTMENTS = (
    "ADROIT",
    "CLONE",
    "CORRUPTED",
    "GLAM",
    "GOOPY",
    "IMBUED",
    "INKY",
    "INSTINCT",
    "MOMENTUM",
    "NIMBLE",
    "PERFECT_FIT",
    "ROYALLY_APPROVED",
    "SHARP",
    "SLITHER",
    "SLUMBERING_ESSENCE",
    "SOULS_POWER",
    "SOWN",
    "SPIRAL",
    "STEADY",
    "SWIFT",
    "TEZCATARAS_EMBER",
    "VIGOROUS",
)

DEFAULT_OUT = (
    Path(__file__).resolve().parents[2]
    / "sts2-rs/crates/sts2-content/tests/fixtures/enchant_eligibility_v0107.json"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    g = Game()
    state = g.start(character="Ironclad", seed="ENCHANTELIG")
    g.skip_neow(state)
    table: dict[str, list[str]] = {}
    skipped: list[str] = []
    for name, _kind, _rarity in sts2_rs.card_kinds():
        cs_id = _csharp_card_id(name)
        probe = g.set_player(deck=[cs_id])
        if probe.get("type") == "error":
            skipped.append(name)
            continue
        ok = []
        for ench in ENCHANTMENTS:
            r = g.set_player(deck=[f"{cs_id}@{ench}:1"])
            if r.get("type") != "error":
                ok.append(ench)
            elif "cannot enchant" not in str(r.get("message")):
                raise RuntimeError(f"{name}@{ench}: {r.get('message')}")
        table[name] = ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "game_version": "v0.107.1",
        "enchantments": list(ENCHANTMENTS),
        "eligible": table,
        "not_in_oracle": skipped,
    }
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(table)} cards, {len(skipped)} not in the oracle -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
