"""Thin Python wrapper over ``run_arena label-campaign`` (full-run-AI M2).

Bead ``sts2-rs-40ikw.6``. Wraps the Rust BINARY via subprocess with the
build-freshness guard from ``run_arena.py`` (the stale-artifact trap has
bitten twice), then summarizes the produced ``labels.jsonl``:

  * rows / decisions / options-per-decision histogram,
  * per-surface counts,
  * tier-1-vs-rollout agreement (how often the heuristic's pick matches
    the rollout-preferred option — the number ``retune_v.py`` tries to
    raise).

Run:

  python python/label_campaign.py run --seeds 2 --combat none \
      --probe-sims 4 --k 4 --t-turns 3 --out-labels labels.jsonl
  python python/label_campaign.py run --panel dev-200 --combat eval-cheap
  python python/label_campaign.py summary labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from run_arena import _resolve_panel, ensure_binary_fresh


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_labels(rows: list[dict]) -> None:
    """Print the label-corpus summary + heuristic/rollout agreement."""
    by_decision: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        ref = row["decision_ref"]
        by_decision[ref["seed"], ref["step"]].append(row)

    surfaces = Counter(row["surface"] for row in rows)
    n_options = Counter(len(opts) for opts in by_decision.values())

    agree = 0
    scored = 0
    for opts in by_decision.values():
        best = max(opts, key=lambda r: r["rollout_score"])
        chosen = [r for r in opts if r["chosen"]]
        if chosen:
            scored += 1
            if chosen[0]["option"] == best["option"]:
                agree += 1

    print("── label campaign summary (python) ──")
    print(f"  label rows        : {len(rows)}")
    print(f"  labeled decisions : {len(by_decision)}")
    print(f"  options/decision  : {dict(sorted(n_options.items()))}")
    print(f"  surfaces          : {dict(surfaces.most_common())}")
    if scored:
        print(
            f"  tier1↔rollout agreement: {agree}/{scored} "
            f"({100.0 * agree / scored:.1f}%) — retune_v.py's target metric"
        )
    shas = {row["engine_git_sha"] for row in rows}
    print(f"  engine identity   : {sorted(shas)}")
    if len(shas) > 1:
        print("  WARNING: mixed engine identities — relabel (plan §9 engine-drift risk)")


def cmd_run(args: argparse.Namespace) -> None:
    binary = ensure_binary_fresh(no_build=args.no_build)
    cmd = [str(binary), "label-campaign", "--combat", args.combat]
    if args.panel:
        cmd += ["--panel", str(_resolve_panel(args.panel))]
    elif args.seeds is not None:
        cmd += ["--seeds", str(args.seeds), "--seed-start", str(args.seed_start)]
    else:
        sys.exit("pass --panel <name|file> or --seeds <N>")
    if args.sims is not None:
        cmd += ["--sims", str(args.sims)]
    if args.boss_sims is not None:
        cmd += ["--boss-sims", str(args.boss_sims)]
    if args.jobs:
        cmd += ["--jobs", str(args.jobs)]
    cmd += [
        "--probe-sims",
        str(args.probe_sims),
        "--k",
        str(args.k),
        "--t-turns",
        str(args.t_turns),
        "--out-labels",
        str(args.out_labels),
        "--out-decisions",
        str(args.out_decisions),
    ]

    print(f"[label_campaign] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)
    summarize_labels(load_jsonl(Path(args.out_labels)))


def cmd_summary(args: argparse.Namespace) -> None:
    summarize_labels(load_jsonl(Path(args.labels)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="run a labeling campaign via the Rust binary")
    p_run.add_argument("--panel", help="panel name (dev-200|…) or JSON path")
    p_run.add_argument("--seeds", type=int, help="ad-hoc seed range size")
    p_run.add_argument("--seed-start", type=int, default=0)
    p_run.add_argument(
        "--combat", default="eval-cheap", choices=["none", "eval-cheap", "eval-strong"]
    )
    p_run.add_argument("--sims", type=int, default=None, help="combat sims debug override")
    p_run.add_argument("--boss-sims", type=int, default=None)
    p_run.add_argument("--probe-sims", type=int, default=100)
    p_run.add_argument("--k", type=int, default=16)
    p_run.add_argument("--t-turns", type=int, default=5)
    p_run.add_argument("--jobs", type=int, default=0)
    p_run.add_argument("--out-labels", default="labels.jsonl")
    p_run.add_argument("--out-decisions", default="decisions.jsonl")
    p_run.add_argument("--no-build", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_sum = sub.add_parser("summary", help="summarize an existing labels.jsonl")
    p_sum.add_argument("labels")
    p_sum.set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)
    if args.command is None:
        # Default to `run` semantics for bare invocation convenience.
        parser.print_help()
        sys.exit(2)
    args.func(args)


if __name__ == "__main__":
    main()
