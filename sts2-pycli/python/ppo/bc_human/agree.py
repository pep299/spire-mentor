"""How often does a playout policy pick the human's move? `python -m ppo.bc_human.agree`.

Reads `gen` shards (default: the held-out runs) and scores top-1 agreement
with the human for the heuristic playout (feature 21 is its logit) and for
any v1 playout-net JSON, overall and by room and decision kind. Copies of a
card with identical key and features count as the same move.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any

import numpy as np

from ppo.rollout_learn import HEUR_FEATURE

ROOM_NAMES = {0: "monster", 1: "elite", 2: "boss"}


def load(data_dir: str) -> dict[str, Any]:
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(data_dir, "shard-*.npz")))]
    vocab = json.load(open(os.path.join(data_dir, "vocab.json"), encoding="utf-8"))
    inv = {i: k for k, i in vocab.items()}
    return {"parts": parts, "inv": inv}


class Net:
    """Numpy twin of `sts2_mcts::rollout_net` for v1 files (no species table)."""

    def __init__(self, path: str) -> None:
        d = json.load(open(path, encoding="utf-8"))
        if d.get("feature_version", 1) not in (1, 3):
            raise ValueError(f"{path}: only v1/v3 nets are supported here")
        self.w1, self.b1 = np.asarray(d["w1"]), np.asarray(d["b1"])
        self.w2, self.b2 = np.asarray(d["w2"]), np.asarray(d["b2"])
        self.emb, self.bias = np.asarray(d["emb"]), np.asarray(d["bias"])
        self.proj, self.lin = np.asarray(d["proj"]), np.asarray(d["lin"])
        self.row = {k: i for i, k in enumerate(d["keys"])}

    def logits(self, s: np.ndarray, keys: list[str], x: np.ndarray) -> np.ndarray:
        h = np.maximum(self.w1 @ s + self.b1, 0.0)
        q = self.w2 @ h + self.b2
        rows = [self.row.get(k, 0) for k in keys]
        return self.bias[rows] + self.emb[rows] @ q + x @ (self.lin + self.proj.T @ q)


def same_move(keys: list[str], x: np.ndarray, i: int, j: int) -> bool:
    return keys[i] == keys[j] and np.allclose(x[i], x[j])


def kind(key: str) -> str:
    return key.split(":")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a gen shard dir (e.g. <out>/holdout)")
    ap.add_argument("--net", nargs="*", default=[], help="name=path.json")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    data = load(args.data)
    policies: dict[str, Any] = {"heuristic": None}
    for spec in args.net:
        name, path = spec.split("=", 1)
        policies[name] = Net(path)
    hits: dict[str, dict[str, list[int]]] = {p: defaultdict(lambda: [0, 0]) for p in policies}
    n_rows = 0
    for part in data["parts"]:
        offs = np.concatenate([[0], np.cumsum(part["n_act"])])
        for r in range(len(part["n_act"])):
            a, b = offs[r], offs[r + 1]
            keys = [data["inv"][int(k)] for k in part["key"][a:b]]
            x = part["x"][a:b]
            human = int(part["chosen"][r])
            for name, pol in policies.items():
                lg = x[:, HEUR_FEATURE] if pol is None else pol.logits(part["state"][r], keys, x)
                ok = int(same_move(keys, x, int(np.argmax(lg)), human))
                for bucket in (
                    "all",
                    f"room:{ROOM_NAMES[int(part['room'][r])]}",
                    f"human:{kind(keys[human])}",
                ):
                    hits[name][bucket][0] += ok
                    hits[name][bucket][1] += 1
            n_rows += 1
            if args.max_rows and n_rows >= args.max_rows:
                break
        if args.max_rows and n_rows >= args.max_rows:
            break
    buckets = sorted({b for h in hits.values() for b in h}, key=lambda b: (b != "all", b))
    print(f"{'bucket':<22}{'n':>8}" + "".join(f"{p:>12}" for p in policies))
    for bkt in buckets:
        n = hits["heuristic"][bkt][1]
        row = "".join(f"{hits[p][bkt][0] / max(hits[p][bkt][1], 1):>12.1%}" for p in policies)
        print(f"{bkt:<22}{n:>8}" + row)


if __name__ == "__main__":
    main()
