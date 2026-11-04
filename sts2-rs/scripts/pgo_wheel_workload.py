#!/usr/bin/env python3
"""PGO profile workload + identity trace for the sts2_rs PyO3 wheel.

Drives full runs through the wheel the way the PPO trainer and its gates do
(``sts2_rs.RunManager`` constructed with the trainer's parity-pinned flags,
combat decided by the wheel's native MCTS), so an instrumented build collects
a profile of the real training hot path. Stdlib-only on purpose: worktree
venvs carry just the wheel. Two agents:

* ``--agent heuristic`` (the default): ``sts2_rs.HeuristicRunAgent`` = MCTS
  combat + heuristic meta, Ascension 0, one character. It dies early (act 1,
  rarely act 2) and never sees a relic-heavy late boss.
* ``--agent ppo``: ``sts2_rs.PpoRunAgent`` = MCTS combat + the deployed PPO
  champion (the ONNX that ``sts2-pycli/data/ppo/champions.toml`` names, at the
  encoder schema it declares) for the meta decisions. All five characters at
  mixed ascensions, Act 1 rolled, full runs: the champions win many runs, so
  this reaches the relic-heavy act-2/3 bosses that dominate combat wall. This
  is the profile ``scripts/pgo_wheel.sh`` collects by default: a relic-less
  profile measured 1.13x on bosses against 1.35x for a matched one.

Two uses (see scripts/pgo_wheel.sh):

* phase-2 profile collection: run with the instrumented wheel; the LLVM
  runtime flushes .profraw files on clean interpreter exit.
* identity check: ``--trace out.json`` records the per-step decision
  sequence + terminal summary (ppo mode adds each combat root's visit /
  win / loss / tie / value sums); the same seeds/sims must produce an
  identical trace on the plain and PGO wheels.

The run loop mirrors sts2-pycli/python/ppo/env.py `PpoRunEnv.__post_init__`
construction; keep the RunManager kwargs in sync with it. The heuristic
mode's flags, defaults and trace format are pinned by the engine-speed
exactness harness: change them and old traces no longer compare.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

CHARACTERS = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")
DEFAULT_CHAMPIONS = (
    Path(__file__).resolve().parent.parent.parent
    / "sts2-pycli"
    / "data"
    / "ppo"
    / "champions.toml"
)


def _val(obj: Any, name: str) -> Any:
    """Read a PyO3 attr that may be a getter-property or a method."""
    v = getattr(obj, name)
    return v() if callable(v) else v


def play_run(
    sts2_rs, seed: int, sims: int, boss_sims: int, max_steps: int, character: str
) -> dict:
    rm = sts2_rs.RunManager(
        seed=seed,
        character=character,
        ascension_level=0,
        neow_start=True,
        mode="agent",
    )
    agent = sts2_rs.HeuristicRunAgent(seed=seed, sims=sims, boss_sims=boss_sims)
    steps: list[dict] = []
    while not _val(rm, "is_over") and len(steps) < max_steps:
        rec = agent.step(rm)
        if rec is None:
            break
        steps.append(
            {
                "phase": rec["phase"],
                # The action dict is small and JSON-safe (strings/ints);
                # sort keys so the trace is byte-comparable.
                "action": json.dumps(rec["action"], sort_keys=True, default=str),
                "result": rec["result"],
            }
        )
    return {
        "seed": seed,
        "character": character,
        "sims": sims,
        "boss_sims": boss_sims,
        "steps": steps,
        "n_steps": len(steps),
        "over": bool(_val(rm, "is_over")),
        "won": bool(_val(rm, "player_won")),
        "hp": int(_val(rm, "current_hp")),
        "gold": int(_val(rm, "gold")),
        "act_index": int(_val(rm, "act_index")),
    }


def load_champions(path: Path) -> dict[str, tuple[Path, str]]:
    """``{normalized character: (onnx path, features spec)}`` from the
    manifest, read the way ``sts2-pycli/python/champions.py`` reads it."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    out = {}
    for character, entry in raw.items():
        onnx = path.parent / entry["onnx"]
        if not onnx.is_file():
            raise FileNotFoundError(
                f"{path}: [{character}] names {onnx}, which is absent"
            )
        out[character.strip().lower()] = (onnx, entry.get("features", ""))
    return out


def ascension_for(character: str, seed: int, ascensions: list[int]) -> int:
    """Round-robin over (character, seed), as a pure function of the pair so
    pgo_wheel.sh can shard one process per run and get the same workload."""
    rank = CHARACTERS.index(character) if character in CHARACTERS else 0
    return ascensions[(rank + seed) % len(ascensions)]


def play_ppo_run(
    sts2_rs,
    seed: int,
    sims: int,
    boss_sims: int,
    max_steps: int,
    character: str,
    ascension: int,
    act1: str,
    champion: tuple[Path, str],
) -> dict:
    onnx, features = champion
    # == ppo/env.py's RunManager (deck-mutation flags on, the env-resolved
    # potion-swap flag, the Act-1 token) …
    rm = sts2_rs.RunManager(
        seed=seed,
        character=character,
        ascension_level=ascension,
        neow_start=True,
        mode="agent",
        interactive_potion_swap=sts2_rs.resolve_potion_swap_flag(),
        act1=act1,
    )
    # … driven by the Rust twin of the trainer's acting policy: the same
    # MCTS combat cell, the champion's ONNX for every meta decision, served
    # at its declared schema (workbench/ppo_policy.py constructs it so).
    agent = sts2_rs.PpoRunAgent(
        seed=seed,
        sims=sims,
        boss_sims=boss_sims,
        model_path=str(onnx),
        features=features,
    )
    steps: list[dict] = []
    while not _val(rm, "is_over") and len(steps) < max_steps:
        rec = agent.step(rm)
        if rec is None:
            break
        step = {
            "phase": rec["phase"],
            "action": json.dumps(rec["action"], sort_keys=True, default=str),
            "result": rec["result"],
        }
        # The root statistics pin the search itself, not just its argmax:
        # a PGO build that perturbed one rollout would show up here long
        # before it flipped a decision.
        search = rec.get("combat_search")
        if search is not None:
            step["search"] = [
                [o["visits"], o["wins"], o["losses"], o["ties"], o["value_sum"]]
                for o in search["actions"]
            ]
        steps.append(step)
    return {
        "agent": "ppo",
        "seed": seed,
        "character": character,
        "ascension": ascension,
        "act1": str(_val(rm, "act1")),
        "sims": sims,
        "boss_sims": boss_sims,
        "steps": steps,
        "n_steps": len(steps),
        "over": bool(_val(rm, "is_over")),
        "won": bool(_val(rm, "player_won")),
        "hp": int(_val(rm, "current_hp")),
        "gold": int(_val(rm, "gold")),
        "act_index": int(_val(rm, "act_index")),
        "floor": int(_val(rm, "total_floor")),
        "n_relics": len(_val(rm, "relics")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--agent", choices=("heuristic", "ppo"), default="heuristic")
    ap.add_argument("--seeds", default="1 4 11 14", help="space-separated run seeds")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--boss-sims", type=int, default=800)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="decision cap per run (default: 150 heuristic, 5000 = a full run for ppo)",
    )
    ap.add_argument(
        "--character",
        default=None,
        help="space-separated characters (default: Ironclad heuristic, all 5 ppo)",
    )
    ap.add_argument(
        "--ascensions",
        default=None,
        help="ppo only: space-separated ascensions, assigned round-robin over "
        "(character, seed) by ascension_for (default: '0 5 10')",
    )
    ap.add_argument(
        "--act1", default=None, help="ppo only: Act-1 token (default: roll)"
    )
    ap.add_argument(
        "--champions",
        default=None,
        help=f"ppo only: champion manifest (default: {DEFAULT_CHAMPIONS})",
    )
    ap.add_argument("--trace", default=None, help="write the decision trace JSON here")
    args = ap.parse_args()
    ppo = args.agent == "ppo"
    if not ppo:
        for flag in ("ascensions", "act1", "champions"):
            if getattr(args, flag) is not None:
                ap.error(f"--{flag} applies to --agent ppo only")
    max_steps = args.max_steps if args.max_steps is not None else (5000 if ppo else 150)
    characters = (
        args.character.split()
        if args.character
        else (list(CHARACTERS) if ppo else ["Ironclad"])
    )
    seeds = [int(s) for s in args.seeds.split()]
    if ppo:
        ascensions = [int(a) for a in (args.ascensions or "0 5 10").split()]
        act1 = args.act1 or "roll"
        manifest = load_champions(
            Path(args.champions) if args.champions else DEFAULT_CHAMPIONS
        )
        missing = [c for c in characters if c.lower() not in manifest]
        if missing:
            ap.error(f"no champion declared for {missing}")

    import sts2_rs

    src_hash = getattr(sts2_rs, "src_hash", lambda: "?")()
    runs = []
    t0 = time.monotonic()
    for character in characters:
        for seed in seeds:
            t = time.monotonic()
            if ppo:
                asc = ascension_for(character, seed, ascensions)
                run = play_ppo_run(
                    sts2_rs,
                    seed,
                    args.sims,
                    args.boss_sims,
                    max_steps,
                    character,
                    asc,
                    act1,
                    manifest[character.lower()],
                )
                tag = (
                    f"{character} A{asc} {run['act1']} act={run['act_index']} "
                    f"floor={run['floor']} relics={run['n_relics']} "
                )
            else:
                run = play_run(
                    sts2_rs, seed, args.sims, args.boss_sims, max_steps, character
                )
                tag = ""
            print(
                f"pgo-workload: {tag}seed={seed} steps={run['n_steps']} "
                f"over={run['over']} won={run['won']} hp={run['hp']} "
                f"({time.monotonic() - t:.1f}s)",
                file=sys.stderr,
            )
            runs.append(run)
    total = time.monotonic() - t0
    print(
        f"pgo-workload: {len(runs)} runs, "
        f"{sum(r['n_steps'] for r in runs)} decisions, {total:.1f}s "
        f"(wheel src_hash={src_hash})",
        file=sys.stderr,
    )

    if args.trace:
        with open(args.trace, "w") as f:
            json.dump(
                {"src_hash": str(src_hash), "runs": runs}, f, indent=1, sort_keys=True
            )
        print(f"pgo-workload: trace -> {args.trace}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
