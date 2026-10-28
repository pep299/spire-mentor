"""Per-player index of the spire-codex corpus (GH #777/#778): find strong players.

The average uploaded v0.107.1 Ironclad A10 run wins ~14%, about the bot's level,
so "humans" as a pool is no ceiling. This indexes every single-player standard
run by `username` so strong players can be selected, emitting one row per run:
username, build, ascension, win, start_time, file.

Selecting players by their observed win rate and then reporting those same
runs is a winner's-curse estimate (a 3/5 player looks like a 60% player).
`--select` therefore picks players on runs from builds OTHER than `--eval-build`
and reports their `--eval-build` runs, which played no part in the selection.

Abandoned runs are dropped by default. That inflates players who abandon their
losing runs (one top-25 player had abandoned 14 of 21), so `index
--include-abandoned` keeps them, tagged `"abandoned": true`, and `select
--abandoned-as-losses` counts them as losses. Without that flag `select` still
drops tagged rows, so an index that keeps them ranks exactly like one that
does not (GH #830).

    python -m ppo.human_players index --out players.jsonl
    python -m ppo.human_players select --index players.jsonl --min-runs 20
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import sys
from multiprocessing import get_context
from typing import Any

DEFAULT_ROOT = "../../sts2_run_scraper/data/files-spire-codex"


def classify_run(
    d: dict[str, Any], path: str, character: str, keep_abandoned: bool = False
) -> tuple[dict[str, Any] | None, str]:
    """Index row for one loaded run file, or None and the filter that dropped it.

    The reason strings name the scan's filter stages, so a caller can count how
    much of the corpus each stage removes (the combat bench MANIFEST does).

    Args:
        d: The parsed run file.
        path: Its path, stored in the row.
        character: Keep runs of this character only (e.g. "Ironclad").
        keep_abandoned: Keep abandoned runs, tagged `"abandoned": true`. Off,
            they are dropped and rows carry no `abandoned` key (the original
            index format).

    Returns:
        `(row, "ok")`, or `(None, reason)` with reason one of `not_standard`,
        `not_single_player`, `other_character`, `abandoned`.
    """
    players = d.get("players") or []
    if d.get("game_mode") != "standard":
        return None, "not_standard"
    if len(players) != 1:
        return None, "not_single_player"
    if players[0].get("character") != f"CHARACTER.{character.upper()}":
        return None, "other_character"
    if d.get("was_abandoned") and not keep_abandoned:
        return None, "abandoned"
    row = {
        "username": d.get("username"),
        "build": d.get("build_id"),
        "ascension": d.get("ascension"),
        "win": bool(d.get("win")),
        "start_time": d.get("start_time"),
        "file": path,
    }
    if keep_abandoned:
        row["abandoned"] = bool(d.get("was_abandoned"))
    return row, "ok"


def scan_file(path: str, character: str, keep_abandoned: bool = False) -> tuple[dict | None, str]:
    """`classify_run` on a run file on disk; reason `unreadable` if it won't load."""
    try:
        with gzip.open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None, "unreadable"
    return classify_run(d, path, character, keep_abandoned)


def _row(path: str, character: str, keep_abandoned: bool = False) -> dict[str, Any] | None:
    return scan_file(path, character, keep_abandoned)[0]


def _job(a: tuple[str, str, bool]) -> tuple[dict | None, str]:
    return scan_file(*a)


def corpus_paths(root: str, character: str) -> list[str]:
    """Every run file of `character` under `root/<character>/<YYYY-MM>/`, sorted."""
    base = os.path.join(root, character.lower())
    return [
        os.path.join(base, m, f)
        for m in sorted(os.listdir(base))
        for f in sorted(os.listdir(os.path.join(base, m)))
    ]


def scan_corpus(
    root: str, character: str, procs: int, keep_abandoned: bool = False
) -> tuple[list[dict[str, Any]], collections.Counter]:
    """Index rows of every kept run plus a count of files per drop reason.

    Rows come back in path order regardless of worker scheduling, so two scans
    of the same corpus write identical indexes.
    """
    paths = corpus_paths(root, character)
    reasons: collections.Counter = collections.Counter({"files": len(paths)})
    rows = []
    with get_context("spawn").Pool(procs) as pool:
        jobs = [(p, character, keep_abandoned) for p in paths]
        for r, why in pool.imap(_job, jobs, chunksize=64):
            reasons[why] += 1
            if r is not None:
                rows.append(r)
    return rows, reasons


def index(args: argparse.Namespace) -> int:
    rows, reasons = scan_corpus(args.root, args.character, args.procs, args.include_abandoned)
    with open(args.out, "w", encoding="utf-8") as sink:
        for r in rows:
            sink.write(json.dumps(r) + "\n")
    print(
        f"[human_players] {reasons['files']} files, {len(rows)} single-player standard runs "
        f"-> {args.out} (dropped: {dict((k, v) for k, v in reasons.items() if k not in ('files', 'ok'))})"
    )
    return 0


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rank_players(
    rows: list[dict[str, Any]],
    ascensions: list[int],
    eval_build: str,
    min_runs: int,
    abandoned_as_losses: bool = False,
) -> tuple[list[tuple[str, int, int, float]], dict[str, list[int]]]:
    """Rank players on their selection-build runs; tally their eval-build runs.

    Args:
        rows: Index rows (`index` output).
        ascensions: Ascension levels that count.
        eval_build: Build whose runs are held out of the ranking.
        min_runs: Minimum selection-build runs to be ranked.
        abandoned_as_losses: Count rows tagged `"abandoned": true` as losses
            (their `win` is false) instead of dropping them.

    Returns:
        `(ranked, ev)`: `ranked` is `(user, wins, runs, wilson_lb)` sorted by
        the Wilson lower bound, best first; `ev` maps user -> `[wins, runs]` on
        `eval_build`, under the same abandoned-run rule.
    """
    rows = [
        r
        for r in rows
        if r["username"]
        and r["ascension"] in ascensions
        and (abandoned_as_losses or not r.get("abandoned"))
    ]
    sel: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    ev: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        bucket = ev if r["build"] == eval_build else sel
        bucket[r["username"]][0] += bool(r["win"]) and not r.get("abandoned")
        bucket[r["username"]][1] += 1
    # Rank on the Wilson LOWER bound of the selection-build win rate: it
    # penalises small samples, so a 3/4 player does not outrank a 40/100 one.
    # Ties keep first-seen order (Python's sort is stable).
    ranked = sorted(
        ((u, k, n, _wilson(k, n)[0]) for u, (k, n) in sel.items() if n >= min_runs),
        key=lambda t: -t[3],
    )
    return ranked, dict(ev)


def select(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in open(args.index, encoding="utf-8")]
    ranked, ev = rank_players(
        rows, args.ascension, args.eval_build, args.min_runs, args.abandoned_as_losses
    )
    print(
        f"players with >= {args.min_runs} selection runs at A{args.ascension}: {len(ranked)}; "
        f"eval build {args.eval_build}"
    )
    print(f"{'user':24s} {'sel W/N':>10s} {'sel lb':>7s} {'eval W/N':>10s}")
    for u, k, n, lb in ranked[: args.top]:
        ek, en = ev.get(u, (0, 0))
        print(f"{u[:24]:24s} {k:>4d}/{n:<5d} {lb:7.1%} {ek:>4d}/{en:<5d}")
    for top in (10, 25, 50, 100):
        users = [u for u, *_ in ranked[:top]]
        ek = sum(ev[u][0] for u in users if u in ev)
        en = sum(ev[u][1] for u in users if u in ev)
        lo, hi = _wilson(ek, en)
        rate = ek / en if en else float("nan")
        print(
            f"top-{top:<3d} held-out {args.eval_build} A{args.ascension}: {ek}/{en} = {rate:.1%} "
            f"[{lo:.1%}, {hi:.1%}] from {sum(1 for u in users if u in ev)} players"
        )
    if args.out_users:
        with open(args.out_users, "w", encoding="utf-8") as f:
            json.dump([u for u, *_ in ranked[: args.top]], f)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--root", default=DEFAULT_ROOT)
    a.add_argument("--character", default="Ironclad")
    a.add_argument("--procs", type=int, default=16)
    a.add_argument("--out", required=True)
    a.add_argument(
        "--include-abandoned",
        action="store_true",
        help='keep abandoned runs, tagged "abandoned": true (default: drop them)',
    )
    b = sub.add_parser("select")
    b.add_argument("--index", required=True)
    b.add_argument("--ascension", type=int, nargs="+", default=[10])
    b.add_argument("--eval-build", default="v0.107.1")
    b.add_argument("--min-runs", type=int, default=20)
    b.add_argument("--top", type=int, default=30)
    b.add_argument("--out-users", default=None)
    b.add_argument(
        "--abandoned-as-losses",
        action="store_true",
        help="count abandoned runs (an --include-abandoned index) as losses instead of dropping them",
    )
    args = ap.parse_args(argv)
    return index(args) if args.cmd == "index" else select(args)


if __name__ == "__main__":
    sys.exit(main())
