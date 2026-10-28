"""`python -m ppo.turnplan.pro_filter`: the root-filter scorer (GH #830 screen).

Trains the no-deck `PlanScorer` on located pro turns for the root-filter
screen (`docs/superpowers/specs/2026-10-22-turnplan-pro-root-filter-screen-design.md`)
and exports it for the Rust forward pass.

    python -m ppo.turnplan.pro_filter --data <collect dir>... --out <dir>

1. **Leakage.** Every replay of `--data` is checked against the measurement
   panels (`pro_leak`: run hash, (player, seed)); overlapping files are
   dropped and the counts go to `leak.json`.
2. **Held-out estimate.** Fit on the train-split files (`label.split_of`),
   early-stop on the salted val fold, measure recall@K (∪ anchors) on the
   A10 holdout files of the first `--data` dir: `estimate/recall.{json,md}`.
3. **Final scorer.** Fit on every located turn (val fold `train.is_val_file`
   for early stopping; nothing held out): `final/` (torch checkpoint) and
   `final/root_filter.json` (the Rust weights, `export_rust`), plus
   `final/parity.json`: recorded positions with the torch scores, the Rust
   parity test's fixture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ppo.turnplan import data, pro_leak, recall
from ppo.turnplan.label import split_of
from ppo.turnplan.pro import _header, as_label, iter_turns

FORMAT = "turnplan.root_filter.v1"


def replay_meta(dirs: list[str]) -> list[dict[str, Any]]:
    """`{"file", "run_hash", "username", "seed"}` of every selected replay of `dirs`."""
    out = []
    for d in dirs:
        for r in json.loads((Path(d) / "selection.json").read_text())["files"]:
            h = _header(r["file"]) or {}
            out.append(
                {
                    "file": Path(r["file"]).name,
                    "run_hash": r["run_hash"],
                    "username": r.get("username"),
                    "seed": h.get("seed"),
                }
            )
    return out


def export_rust(model: Any, vocab: data.Vocab) -> dict[str, Any]:
    """The no-deck scorer as plain arrays for `sts2_mcts::turn_plan_scorer`.

    The deck branch is zero in a no-deck model, so only the score MLP's
    ctx and plan input columns are kept: `score = w2 · relu(w1 · [ctx_n,
    plan] + b1) + b2`, `plan = plan_mlp(plan_n) + mean(card_emb[cards])`.
    """
    from ppo.turnplan.train import DECK_DIM

    if not model.no_deck:
        raise ValueError("only a no-deck scorer exports to the root filter")
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}

    def lin(prefix: str, cols: slice | None = None) -> dict[str, Any]:
        w = sd[f"{prefix}.weight"]
        if cols is not None:
            w = w[:, cols]
        return {"w": w.tolist(), "b": sd[f"{prefix}.bias"].tolist()}

    return {
        "format": FORMAT,
        "ctx_mean": sd["ctx_mean"].tolist(),
        "ctx_std": sd["ctx_std"].tolist(),
        "plan_mean": sd["plan_mean"].tolist(),
        "plan_std": sd["plan_std"].tolist(),
        "plan_l1": lin("plan_mlp.0"),
        "plan_l2": lin("plan_mlp.2"),
        "card_vocab": vocab.card,
        "card_emb": sd["plan_card_bag.weight"].tolist(),
        "score_l1": lin("score_mlp.0", slice(DECK_DIM, None)),
        "score_l2": lin("score_mlp.2"),
    }


def parity_fixture(
    model: Any, recs: list[dict[str, Any]], vocab: data.Vocab, n: int = 40
) -> dict[str, Any]:
    """`n` recorded positions (ctx, plan features and card names) with the
    torch scores, for the Rust forward pass's parity test."""
    from ppo.turnplan import train

    picked = recs[:: max(len(recs) // n, 1)][:n]
    scores = train.score(model, [data.to_position(r, vocab) for r in picked])
    return {
        "format": FORMAT,
        "positions": [
            {
                "ctx": r["deck"]["ctx"],
                "plans": [{"features": p["features"], "cards": p["cards"]} for p in r["plans"]],
                "scores": s.astype(np.float64).tolist(),
            }
            for r, s in zip(picked, scores, strict=True)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.turnplan.pro_filter", description=__doc__)
    ap.add_argument("--data", nargs="+", required=True, help="collect dirs; the first is A10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    from ppo.turnplan import train  # torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[pro_filter] {msg}", file=sys.stderr, flush=True)
        with (out / "train.log").open("a") as f:
            f.write(msg + "\n")

    rows = pro_leak.panel_rows()
    seeds = {f: pro_leak.run_seed(f) for f in sorted({r["file"] for r in rows})}
    keys = pro_leak.panel_keys(rows, seeds)
    metas = replay_meta(args.data)
    hit = pro_leak.leaks(metas, keys)
    bad = {h for v in hit.values() for h in v}
    leak = {
        "panel_fights": len(rows),
        "panel_runs": len(keys["run_hash"]),
        "panel_runs_with_seed": sum(1 for s in seeds.values() if s),
        "panel_player_seed_pairs": len(keys["player_seed"]),
        "replays_checked": len(metas),
        "overlap_run_hash": len(hit["run_hash"]),
        "overlap_player_seed": len(hit["player_seed"]),
        "excluded_files": sorted(bad),
    }
    (out / "leak.json").write_text(json.dumps(leak, indent=1) + "\n")
    log(f"leakage: {leak}")

    def located(d: str) -> list[dict[str, Any]]:
        return [
            r
            for r in iter_turns(d)
            if r["locate"]["index"] is not None and r["file"].split(".")[0] not in bad
        ]

    per_dir = [located(d) for d in args.data]
    # After exclusion the overlap is zero by construction; assert it anyway.
    assert not any(r["file"].split(".")[0] in bad for rs in per_dir for r in rs)

    # 2. Held-out estimate.
    fit_recs = [as_label(r) for rs in per_dir for r in rs if split_of(r["file"]) == "train"]
    hold_recs = [as_label(r) for r in per_dir[0] if split_of(r["file"]) == "holdout"]
    vocab = data.Vocab.build(fit_recs)
    tr, val = train.split_val([data.to_position(r, vocab) for r in fit_recs])
    holdout = [data.to_position(r, vocab) for r in hold_recs]
    log(f"estimate: {len(tr)} fit / {len(val)} val / {len(holdout)} holdout positions")
    model, meta = train.fit(tr, val, vocab, train.TrainConfig(no_deck=True), log)
    train.save(model, vocab, meta, out / "estimate")
    scores = {
        "scorer_nodeck": train.score(model, holdout),
        "heur": [p.logp_heur for p in holdout],
        "rl2": [p.logp_rl2 for p in holdout],
    }
    # `recall.report` reads every filter it knows; the deck arm is not trained here.
    scores["scorer_deck"] = scores["scorer_nodeck"]
    rep = recall.report(
        holdout,
        scores,
        n_boot=args.n_boot,
        qualifiers={
            "teacher": "the human's own plan; no-deck scorer fit to human plans",
            "fights": "SpireCodex v0.111.0 Ironclad replays, in-sync turn starts",
            "split": "holdout replay files of the first --data dir: crc32(file) % 5 == 0",
            "n_fit": len(tr),
            "n_val": len(val),
            "note": "scorer_deck is a copy of scorer_nodeck (one arm trained)",
            "leak": {k: v for k, v in leak.items() if k != "excluded_files"},
        },
    )
    rep.pop("verdict")
    (out / "estimate" / "recall.json").write_text(json.dumps(rep, indent=2) + "\n")
    (out / "estimate" / "recall.md").write_text(
        recall.render_md({**rep, "verdict": {"line": "(no gate)"}})
    )

    # 3. Final scorer on every located turn.
    all_recs = [as_label(r) for rs in per_dir for r in rs]
    vocab = data.Vocab.build(all_recs)
    positions = [data.to_position(r, vocab) for r in all_recs]
    val = [p for p in positions if train.is_val_file(p.file)]
    tr = [p for p in positions if not train.is_val_file(p.file)]
    log(f"final: {len(tr)} fit / {len(val)} val positions (nothing held out)")
    model, meta = train.fit(tr, val, vocab, train.TrainConfig(no_deck=True), log)
    train.save(
        model, vocab, {**meta, "leak": leak, "n_fit": len(tr), "n_val": len(val)}, out / "final"
    )
    (out / "final" / "root_filter.json").write_text(json.dumps(export_rust(model, vocab)) + "\n")
    val_recs = [r for r in all_recs if train.is_val_file(r["file"])]
    (out / "final" / "parity.json").write_text(
        json.dumps(parity_fixture(model, val_recs, vocab)) + "\n"
    )
    log(f"final best epoch {meta['best_epoch']} -> {out / 'final'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
