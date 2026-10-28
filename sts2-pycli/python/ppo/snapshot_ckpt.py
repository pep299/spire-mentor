"""Snapshot the newest finalized PPO campaign checkpoint into the repo.

Picks the highest-N ``<stem>.iter_N.pt`` under ``--src`` (iter checkpoints are
write-once; ``<stem>.pt`` is overwritten every iteration and must NOT be
copied), strips the optimizer state, and writes three committed artifacts
under ``--out``: weights-only ``ppo_fullrun_it<N>.pt``, a provenance note, and
a copy of the campaign ``<stem>.stats.jsonl``.

Usage (from sts2-pycli/):
    uv run python python/ppo/snapshot_ckpt.py \
        --src /tmp/ppo-m5 --stem main --out data/ppo \
        --note "ppo-silent 400-iter campaign (PR #197 recipe), engine 92135e3b3"
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import re
import shutil
import subprocess

import torch

_ITER_RE = re.compile(r"\.iter_(\d+)\.pt$")
_LOG_RE = re.compile(r"^it\s*(\d+) win=([0-9.]+)")


def newest_iter_ckpt(src: pathlib.Path, stem: str) -> tuple[int, pathlib.Path]:
    best: tuple[int, pathlib.Path] | None = None
    for p in src.glob(f"{stem}.iter_*.pt"):
        m = _ITER_RE.search(p.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    if best is None:
        raise SystemExit(f"no {stem}.iter_N.pt under {src}")
    return best


def win_band(log: pathlib.Path, it: int, window: int = 10) -> str:
    """Mean self-play win over the ``window`` iterations up to ``it``."""
    wins = []
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            m = _LOG_RE.match(line)
            if m and it - window < int(m.group(1)) <= it:
                wins.append(float(m.group(2)))
    if not wins:
        return "unknown"
    return f"{sum(wins) / len(wins):.3f} (mean of {len(wins)} iters ending at it{it})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=pathlib.Path)
    ap.add_argument("--stem", default="main")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    it, ckpt = newest_iter_ckpt(args.src, args.stem)
    args.out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = ck.get("state_dict", ck)
    out_pt = args.out / f"ppo_fullrun_it{it}.pt"
    torch.save({"state_dict": state}, out_pt)

    stats_src = args.src / f"{args.stem}.stats.jsonl"
    if stats_src.exists():
        shutil.copyfile(stats_src, args.out / f"ppo_fullrun_it{it}.stats.jsonl")

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    prov = args.out / f"ppo_fullrun_it{it}.provenance.md"
    prov.write_text(
        f"""# PPO campaign snapshot — iteration {it}

- **Source**: `{ckpt}` (write-once iter checkpoint; optimizer state stripped)
- **Extracted**: {datetime.date.today().isoformat()} at repo commit `{sha}`
- **Self-play win band**: {win_band(args.src / f"{args.stem}.log", it)}
- **Net**: `ppo.model.RunPolicyNet` defaults (dim=192, n_layers=3, n_heads=4)
- **Note**: {args.note or "(none)"}
- **Command**: `uv run python python/ppo/snapshot_ckpt.py --src {args.src} --stem {args.stem} --out {args.out}`

The campaign was still RUNNING when this snapshot was taken — this is not a
blessed champion; deployment gating (Rule-0 G7 / release-2000 G8) is owned by
the campaign plan `sts2-rs/docs/plans/2026-07-03-ppo-fullrun-plan.md`.
"""
    )
    print(f"wrote {out_pt} + provenance + stats (iter {it})")


if __name__ == "__main__":
    main()
