#!/usr/bin/env python3
"""v0.111.0 smoke for the deployed champions (GH #799).

One full run per (character, seed) at game build v0.111.0, driven end to end by
the deployed run agent (``sts2_rs.PpoRunAgent``: the champion ONNX for every
meta decision, the combat MCTS for every fight) with the champion
``champions.toml`` names. A run must finish (won, or lost on HP) without an
error, and must put a v0.111.0-only id in front of the net: the default seeds
are ones whose Neow offer carries a v0.111.0 Neow relic (22: DowsingRod,
20: NeowsSacrifice), whose label the encoder resolves to that relic's new row.
Every v0.111.0-only id seen in the run state is recorded per run.

USAGE (from sts2-pycli, worktree wheel venv)::

    python scripts/smoke_v0111_champions.py --out smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]


def v0111_only_ids() -> set[str]:
    """Canonical ids the v0.107.1 vocab did not have (the appended pin tail)."""
    from ppo import vocab

    out = set()
    for reg, old_n in ((vocab.cards(), 580), (vocab.relics(), 297), (vocab.potions(), 63)):
        out |= {vocab.canonical(n) for n in reg.names[old_n:]}
    return out


def seen_ids(rm) -> set[str]:
    from ppo import vocab

    # Each accessor answers None outside the phase it describes.
    names: list[str] = []
    if rm.phase() == "event":
        names += [label for _opt, label in rm.event_options() or []]
    names += list(rm.master_deck() or []) + list(rm.relics() or [])
    names += [str(p) for p in rm.potions() or []]
    if rm.phase() == "card_reward":
        names += list(rm.card_reward_option_ids() or [])
    return {vocab.canonical(n) for n in names if isinstance(n, str)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[22, 20])
    ap.add_argument("--characters", nargs="+", default=CHARACTERS)
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--boss-sims", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import champions
    import sts2_rs

    new_ids = v0111_only_ids()
    assert len(new_ids) == 22, sorted(new_ids)
    ok = True
    with a.out.open("w") as fh:
        for character in a.characters:
            champ = champions.get(character)
            assert champ is not None, character
            for seed in a.seeds:
                t0 = time.time()
                rm = sts2_rs.RunManager(  # ty: ignore[unresolved-attribute]
                    seed=seed,
                    character=character,
                    neow_start=True,
                    mode="agent",
                    act1="roll",
                    game_version="v0.111.0",
                )
                assert rm.game_version == "v0.111.0"
                agent = sts2_rs.PpoRunAgent(  # ty: ignore[unresolved-attribute]
                    seed=seed,
                    sims=a.sims,
                    boss_sims=a.boss_sims,
                    model_path=str(champ.onnx_path),
                )
                seen: set[str] = set()
                steps, error = 0, None
                try:
                    while not rm.is_over() and steps < a.max_steps:
                        seen |= seen_ids(rm) & new_ids
                        if agent.step(rm) is None:
                            break
                        steps += 1
                    seen |= seen_ids(rm) & new_ids
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    traceback.print_exc()
                row = {
                    "character": character,
                    "seed": seed,
                    "champion": champ.onnx,
                    "game_version": rm.game_version,
                    "over": bool(rm.is_over()),
                    "won": bool(rm.player_won),
                    "act": rm.act_index,
                    "floor": rm.total_floor(),
                    "steps": steps,
                    "v0111_ids_seen": sorted(seen),
                    "error": error,
                    "secs": round(time.time() - t0, 1),
                }
                good = error is None and row["over"] and bool(seen)
                ok &= good
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(json.dumps(row), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
