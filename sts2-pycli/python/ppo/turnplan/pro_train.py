"""`python -m ppo.turnplan.pro_train`: a plan scorer fit to HUMAN plans (GH #830 feasibility).

The G0 `PlanScorer` (same architecture, same listwise loss) trained with the
human's located plan as the one best plan (`pro.as_label`), then measured on
held-out replay files by `recall.report`. It asks how far a filter trained on
pro turns gets toward recall@(6 ∪ anchors) >= 95 % on pro turns.

Split by replay FILE (`label.split_of`: crc32 % 5 == 0 -> holdout). The
holdout is read only from the `--data` dirs' A10 files (`--eval-data`); every
other located turn (the train files of `--eval-data` and all of `--extra-data`)
(train-split files only) is fit data, with the trainer's own validation fold
carved from it by file.
Capped turns are fit (a partial list is still a listwise target) and never
measured.

    python -m ppo.turnplan.pro_train --eval-data <a10 collect dir> \\
        --extra-data <a0-9 collect dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ppo.turnplan import data, recall
from ppo.turnplan.label import split_of
from ppo.turnplan.pro import as_label, iter_turns


def split_rows(
    eval_rows: list[dict[str, Any]], extra_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`(fit, holdout)` label records: located turns only; holdout = the
    `eval_rows` whose file `split_of` puts in the holdout; `extra_rows` are
    fit data from their train-split files only."""
    fit, hold = [], []
    for r in eval_rows:
        if r["locate"]["index"] is None:
            continue
        (hold if split_of(r["file"]) == "holdout" else fit).append(as_label(r))
    # `train.split_val` refuses a holdout-split file among the fit data, so
    # the extra rows' holdout-split files are left out (not measured either).
    fit += [
        as_label(r)
        for r in extra_rows
        if r["locate"]["index"] is not None and split_of(r["file"]) == "train"
    ]
    return fit, hold


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.turnplan.pro_train", description=__doc__)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    from ppo.turnplan import train  # torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = [r for d in args.extra_data for r in iter_turns(d)]
    fit_recs, hold_recs = split_rows(list(iter_turns(args.eval_data)), extra)
    vocab = data.Vocab.build(fit_recs)
    fit_all = [data.to_position(r, vocab) for r in fit_recs]
    holdout = [data.to_position(r, vocab) for r in hold_recs]
    tr, val = train.split_val(fit_all)
    scores: dict[str, list[Any]] = {}
    metas = {}
    for name, no_deck in (("scorer_deck", False), ("scorer_nodeck", True)):
        cfg = train.TrainConfig(no_deck=no_deck)
        log_path = out / f"{name}.log"

        def log(msg: str, path: Path = log_path) -> None:
            print(f"[pro_train] {msg}", file=sys.stderr, flush=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

        log(f"{len(tr)} fit / {len(val)} val / {len(holdout)} holdout positions")
        model, meta = train.fit(tr, val, vocab, cfg, log)
        train.save(model, vocab, meta, out / name)
        scores[name] = train.score(model, holdout)
        metas[name] = {k: meta[k] for k in ("best_epoch", "best_val_loss")} | {
            "val_top1_at_best": meta["val_top1"][meta["best_epoch"]]
        }
    scores["heur"] = [p.logp_heur for p in holdout]
    scores["rl2"] = [p.logp_rl2 for p in holdout]
    rep = recall.report(
        holdout,
        scores,
        n_boot=args.n_boot,
        qualifiers={
            "teacher": "the human's own plan; scorers fit to human plans",
            "fights": "SpireCodex v0.111.0 replays, in-sync turn starts",
            "split": "holdout replay files of --eval-data: crc32(file) % 5 == 0",
            "n_fit": len(tr),
            "n_val": len(val),
            "models": metas,
        },
    )
    rep.pop("verdict")
    (out / "recall.json").write_text(json.dumps(rep, indent=2) + "\n")
    (out / "recall.md").write_text(recall.render_md({**rep, "verdict": {"line": "(no gate)"}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
