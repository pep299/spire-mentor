"""`python -m ppo.turnplan.pro_report`: the human-turn feasibility report (GH #830).

Reads a `ppo.turnplan.pro collect` dir (and its `teacher/` sample, if any)
and writes `summary.json` + `summary.md`:

1. sync: fights, log turns, turn starts reached in sync, first desync kinds;
2. legality of the human's turn from an in-sync start (`status`);
3. coverage: the human's plan located in the enumerated set, by cap and room,
   and the misses by reason;
4. recall: on located, uncapped turns, the rank of the human's plan under the
   heuristic, rl2 and the two G0 scorers; recall@K (plain and ∪ anchors)
   through `recall.report` (run-file cluster bootstrap), the human's plan
   standing in for the teacher's best;
5. teacher vs human on the `teacher/` sample.

Torch is needed for the G0 scorers: run it with the `.venv` interpreter.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ppo.turnplan import data, recall
from ppo.turnplan.pro import as_label, iter_turns

ROOMS = ("boss", "elite", "monster")


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} = {100 * a / b:.1f}%" if b else f"{a}/0"


def sync_stats(out: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    files = [json.loads(line) for line in (out / "files.jsonl").open()]
    ok = [f for f in files if "crash" not in f]
    desync: Counter[str] = Counter()
    log_turns: Counter[str] = Counter()
    for f in ok:
        desync.update(f["desync"])
        log_turns.update(f.get("log_turns", {}))
    in_sync = Counter(r["room_type"] for r in rows)
    return {
        "files": len(files),
        "files_crashed": len(files) - len(ok),
        "crashes": [f["crash"] for f in files if "crash" in f][:20],
        "fights": sum(f["fights"] for f in ok),
        "fights_by_first_desync": dict(desync.most_common()),
        "turn_errors": sum(len(f["errors"]) for f in ok),
        "log_turns": dict(log_turns),
        "in_sync_turns": dict(in_sync),
        "in_sync_rate": {r: in_sync[r] / log_turns[r] for r in ROOMS if log_turns[r]}
        | {"all": sum(in_sync.values()) / max(sum(log_turns.values()), 1)},
    }


def _has(row: dict[str, Any], kind: str) -> bool:
    used = row["locate"]["used"] or len(row["human_actions"])
    return any(a.get("action") == kind for a in row["human_actions"][:used])


def coverage_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def cell(sel: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sel)
        loc = sum(r["locate"]["index"] is not None for r in sel)
        return {"n": n, "located": loc, "rate": loc / n if n else None}

    out: dict[str, Any] = {"all": cell(rows)}
    for cap in (False, True):
        out[f"cap_hit={cap}"] = cell([r for r in rows if r["cap_hit"] == cap])
    for room in ROOMS:
        out[f"room={room}"] = cell([r for r in rows if r["room_type"] == room])
        out[f"room={room},uncapped"] = cell(
            [r for r in rows if r["room_type"] == room and not r["cap_hit"]]
        )
    for st in sorted({r["status"].split(":")[0] for r in rows}):
        out[f"status={st}"] = cell([r for r in rows if r["status"].split(":")[0] == st])
    miss: Counter[str] = Counter()
    for r in rows:
        loc = r["locate"]
        if loc["index"] is not None:
            continue
        if loc["failure"]:
            miss[f"cut failed: {loc['failure']} (turn {r['status']})"] += 1
        elif r["cap_hit"]:
            miss["cap hit (plan not in the partial list)"] += 1
        else:
            tags = [k for k in ("drink_potion", "choose") if _has(r, k)]
            miss[f"uncapped, end={loc['end']}, has={'+'.join(tags) or '-'}"] += 1
    out["misses"] = dict(miss.most_common())
    ends = Counter(r["locate"]["end"] for r in rows if r["locate"]["index"] is not None)
    out["located_plan_end"] = dict(ends)
    return out


def rank_stats(positions: list[data.Position], scores: dict[str, list[np.ndarray]]) -> dict:
    """The human plan's rank (1 = top; ties: plan index first) under each filter."""
    out = {}
    for f, ss in scores.items():
        ranks = []
        for p, s in zip(positions, ss, strict=True):
            order = np.argsort(-s, kind="stable")
            ranks.append(int(np.flatnonzero(order == int(np.argmax(p.q)))[0]) + 1)
        r = np.asarray(ranks)
        out[f] = {
            "median": float(np.median(r)),
            "mean": float(r.mean()),
            "p90": float(np.percentile(r, 90)),
        }
    out["n_plans"] = {
        "median": float(np.median([len(p.q) for p in positions])),
        "mean": float(np.mean([len(p.q) for p in positions])),
    }
    return out


def teacher_stats(tdir: Path) -> dict[str, Any] | None:
    rows = []
    errors = 0
    for f in sorted(tdir.glob("*.jsonl")):
        for line in f.open():
            d = json.loads(line)
            if "error" in d:
                errors += 1
            else:
                rows.append(d)
    if not rows:
        return None
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in rows:
        i = d["locate"]["index"]
        if i is None or d["cap_hit"]:
            continue
        q = np.asarray(d["q"], np.float64)
        v = np.asarray(d["visits"], np.float64)
        se = np.where(v > 0, np.sqrt(np.maximum(np.asarray(d["q_var"]), 0) / np.maximum(v, 1)), 0)
        best = int(np.argmax(q))
        band = q[best] - q[i] <= 2 * np.hypot(se[best], se[i])
        rank = 1 + int((q > q[i]).sum())
        below = q < q.max()
        resolved = bool(
            below.any()
            and q.max() - q[below].max()
            > 2 * np.hypot(se[best], se[int(np.flatnonzero(below)[np.argmax(q[below])])])
        )
        rec = {
            "human_is_best": bool(q[i] == q.max()),
            "in_band": bool(band),
            "rank": rank,
            "rank_frac": (rank - 1) / max(len(q) - 1, 1),
            "regret": float(q.max() - q[i]),
            "resolved": resolved,
            "n_plans": len(q),
            "agree_first": (
                None if d["agent_first"] is None else d["agent_first"] == d["human_first"]
            ),
        }
        by["all"].append(rec)
        by[d["room_type"]].append(rec)
        if resolved:
            by["resolved"].append(rec)

    def agg(recs: list[dict[str, Any]]) -> dict[str, Any]:
        a = [r["agree_first"] for r in recs if r["agree_first"] is not None]
        return {
            "n": len(recs),
            "human_is_teacher_best": float(np.mean([r["human_is_best"] for r in recs])),
            "human_within_2se_of_best": float(np.mean([r["in_band"] for r in recs])),
            "teacher_best_differs_beyond_noise": float(np.mean([not r["in_band"] for r in recs])),
            "human_rank_median": float(np.median([r["rank"] for r in recs])),
            "human_rank_mean": float(np.mean([r["rank"] for r in recs])),
            "human_rank_frac_mean": float(np.mean([r["rank_frac"] for r in recs])),
            "regret_mean": float(np.mean([r["regret"] for r in recs])),
            "regret_p90": float(np.percentile([r["regret"] for r in recs], 90)),
            "resolved_share": float(np.mean([r["resolved"] for r in recs])),
            "agent_first_action_agrees": float(np.mean(a)) if a else None,
            "agent_first_n": len(a),
            "n_plans_median": float(np.median([r["n_plans"] for r in recs])),
        }

    return {
        "rows": len(rows),
        "errors": errors,
        "relabel_mismatch": sum(d["locate"]["index"] != d["located_before"] for d in rows),
        "subsets": {k: agg(v) for k, v in by.items() if v},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.turnplan.pro_report", description=__doc__)
    ap.add_argument("--out", required=True, help="a `pro collect` dir")
    ap.add_argument("--model", required=True, help="G0 scorer dir (deck)")
    ap.add_argument("--model-nodeck", required=True)
    ap.add_argument("--report", required=True, help="where summary.json / .md go")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    out, rep_dir = Path(args.out), Path(args.report)
    rep_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_turns(out))
    summary: dict[str, Any] = {
        "qualifiers": {
            "character": "Ironclad",
            "game_version": "v0.111.0 replays (replay_version >= 6), engine v0.111.0 seams",
            "selection": json.loads((out / "selection.json").read_text())["census"],
            "plans": "enumerate_plans cap 512 plans / 20000 nodes, BeliefDraw root view",
        },
        "sync": sync_stats(out, rows),
        "legality": dict(Counter(r["status"] for r in rows).most_common()),
        "coverage": coverage_stats(rows),
    }
    from ppo.turnplan import train  # torch

    labelled = [as_label(r) for r in rows if r["locate"]["index"] is not None]
    scores: dict[str, list[np.ndarray]] = {}
    positions: list[data.Position] = []
    for name, d in (("scorer_deck", args.model), ("scorer_nodeck", args.model_nodeck)):
        model, vocab = train.load_model(d)
        positions = [data.to_position(r, vocab) for r in labelled]
        scores[name] = train.score(model, positions)
    scores["heur"] = [p.logp_heur for p in positions]
    scores["rl2"] = [p.logp_rl2 for p in positions]
    rep = recall.report(
        positions,
        scores,
        n_boot=args.n_boot,
        qualifiers={
            "teacher": "the human's own plan (q = 1 for it, 0 otherwise)",
            "fights": "SpireCodex v0.111.0 replays, in-sync turn starts",
            "split": "all replay files (none was in G0's training data)",
        },
    )
    rep.pop("verdict")
    meas = [p for p in positions if not p.cap_hit]
    summary["recall"] = rep
    summary["rank"] = rank_stats(
        meas,
        {
            f: [s for p, s in zip(positions, scores[f], strict=True) if not p.cap_hit]
            for f in scores
        },
    )
    t = teacher_stats(out / "teacher")
    if t is not None:
        summary["teacher"] = t
    (rep_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (rep_dir / "recall.md").write_text(recall.render_md({**rep, "verdict": {"line": "(no gate)"}}))
    print(json.dumps({k: summary[k] for k in ("sync", "legality")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
