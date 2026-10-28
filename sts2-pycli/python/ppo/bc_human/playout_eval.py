"""No-search playout win rate of playout policies on fight starts.

`python -m ppo.bc_human.playout_eval --fights boss800.jsonl --net rl2=a.json bc=b.json`

Each start is played `--reps` times per policy with the same seeds across
policies (paired). `heuristic` is the heuristic playout expressed as a net
(only feature 21, its own logit, at weight 1).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ppo.rollout_learn import ACTION_DIM, HEUR_FEATURE, STATE_DIM, _combat_states


def heuristic_net(like: dict) -> dict:
    h = dict(like)
    n, hid, e = len(like["keys"]), like["hidden"], like["emb_dim"]

    def z(*s: int) -> list:
        return np.zeros(s).tolist()

    h.update(
        w1=z(hid, STATE_DIM),
        b1=z(hid),
        w2=z(e, hid),
        b2=z(e),
        emb=z(n, e),
        bias=z(n),
        proj=z(e, ACTION_DIM),
        lin=[1.0 if i == HEUR_FEATURE else 0.0 for i in range(ACTION_DIM)],
        temperature=1.0,
        no_early_end_turn=False,
    )
    return h


def main() -> None:
    import sts2_rs

    ap = argparse.ArgumentParser()
    ap.add_argument("--fights", nargs="+", required=True)
    ap.add_argument("--net", nargs="+", required=True, help="name=path.json[@T]")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--no-early-end", action="store_true")
    args = ap.parse_args()

    nets = {}
    for spec in args.net:
        name, rest = spec.split("=", 1)
        path, _, temp = rest.partition("@")
        nets[name] = (json.load(open(path, encoding="utf-8")), float(temp) if temp else 1.0)
    first = next(iter(nets.values()))[0]
    nets = {"heuristic": (heuristic_net(first), 1.0)} | nets

    for fpath in args.fights:
        fights = [json.loads(line) for line in open(fpath, encoding="utf-8")]
        ev = _combat_states(sts2_rs, fights)
        jobs = [st for _, st in ev for _ in range(args.reps)]
        seeds = [1_000_003 * i + 17 for i in range(len(jobs))]
        res = {}
        for name, (js, temp) in nets.items():
            lp = sts2_rs.LearnedPlayout(json.dumps(js))
            out = lp.play_many(jobs, seeds, temperature=temp, no_early_end=args.no_early_end)
            res[name] = np.frombuffer(out["won"], np.uint8).astype(float)
        base = res["heuristic"]
        print(f"== {fpath}: {len(ev)} starts x {args.reps}")
        for name, won in res.items():
            d = won - base
            per = d.reshape(len(ev), args.reps).mean(1)  # cluster by start
            se = per.std(ddof=1) / np.sqrt(len(per))
            print(
                f"  {name:<14} {100 * won.mean():5.1f}%   vs heuristic {100 * d.mean():+5.1f} ± {100 * se:.1f}pp"
            )


if __name__ == "__main__":
    main()
