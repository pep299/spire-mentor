"""Power projection before the full gen: choose the split scheme (PREREG §7.4).

From the power pilot (all boss/elite fights of 20 fold-2 runs, 100% shadow)
this estimates, on the pilot's E decisions, the per-decision variance and the
run intra-class correlation of two paired regret differences that need no
training: `rl2_zs − ref_teacher` (used) and `rl2_zs − heur` (reported). Two
decent students disagree with each other less often than these pairs, so the
projection is conservative. With `m̄` = E decisions per run,

    deff = 1 + (m̄ − 1)·ICC,   SE_proj = sqrt(var_d · deff / (m̄ · TEST runs))

for S1 and S2 (TEST runs with boss/elite fights counted from the bank). The
rule: S1 if `SE_proj ≤ 0.0007` (the 0.002 floor / 2.5, less 15% for the
training-seed variance the projection leaves out), else S2, else S2 with rep 1
of every TEST/DEV fight in the base gen, flagged as projected underpowered.
`power.json` is read by `gen --power` (it names the scheme).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any

import numpy as np

from . import gen as G
from .data import decision_arrays

SE_RULE = 0.0007


def icc_oneway(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC (ANOVA estimator, unbalanced groups), >= 0."""
    u, inv = np.unique(groups, return_inverse=True)
    k = len(u)
    n = len(values)
    if k < 2 or n <= k:
        return 0.0
    sizes = np.bincount(inv).astype(np.float64)
    means = np.bincount(inv, weights=values) / sizes
    grand = values.mean()
    ssb = float((sizes * (means - grand) ** 2).sum())
    ssw = float(((values - means[inv]) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    m0 = (n - (sizes**2).sum() / n) / (k - 1)
    den = msb + (m0 - 1) * msw
    return float(max(0.0, (msb - msw) / den)) if den > 0 else 0.0


def pilot_diffs(gen_dir: str) -> dict[str, Any]:
    """Per E decision: paired regret differences and the run / fight ids."""
    d1, d2, runs, fights, rooms = [], [], [], [], []
    for r in G.iter_records(gen_dir):
        if r.get("obs") is None or r.get("cls") is None or r["room"] not in ("boss", "elite"):
            continue
        da = decision_arrays(r)
        if not da["in_E"]:
            continue
        ra, rb = da["regA"], da["regB"]

        def reg(i: int, ra: np.ndarray = ra, rb: np.ndarray = rb) -> float:
            return 0.5 * (ra[i] + rb[i])

        zs = int(np.argmax(r["rl2"]["logits"]))
        heur = int(np.argmax([x[21] for x in r["rl2"]["x"]]))
        ref = 0.5 * (ra[da["recB"]] + rb[da["recA"]])
        d1.append(reg(zs) - ref)
        d2.append(reg(zs) - reg(heur))
        runs.append(r["file"])
        fights.append(f"{r['key']}|{r['rep']}")
        rooms.append(r["room"])
    return {
        "d_ref": np.asarray(d1),
        "d_heur": np.asarray(d2),
        "runs": np.asarray(runs),
        "fights": np.asarray(fights),
        "rooms": np.asarray(rooms),
    }


def test_runs(fights_path: str, scheme: str) -> int:
    rows = G.read_bank(fights_path)
    held = G.holdout(rows)["all"]
    return len(
        {
            r["file"]
            for r in rows
            if r["file"] not in held
            and r["room_type"] in ("boss", "elite")
            and G.split_of(r["file"], scheme) == "test"
        }
    )


def project(d: dict[str, Any], n_test_runs: dict[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_E": len(d["d_ref"]), "runs": len(set(d["runs"].tolist()))}
    if out["n_E"] < 2:
        raise SystemExit("too few E decisions in the pilot to project")
    m_bar = out["n_E"] / out["runs"]
    per_fight = collections.Counter(d["fights"].tolist())
    out["E_per_run"] = m_bar
    out["E_per_fight"] = {
        room: float(
            np.mean([per_fight[f] for f in set(d["fights"][d["rooms"] == room].tolist())] or [0])
        )
        for room in ("boss", "elite")
    }
    for name in ("d_ref", "d_heur"):
        v = d[name]
        icc = icc_oneway(v, d["runs"])
        deff = 1 + (m_bar - 1) * icc
        row = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "icc": icc, "deff": deff}
        for scheme, k in n_test_runs.items():
            row[f"SE_proj_{scheme}"] = float(np.sqrt(v.var(ddof=1) * deff / (m_bar * k)))
        out[name] = row
    se1, se2 = out["d_ref"]["SE_proj_S1"], out["d_ref"]["SE_proj_S2"]
    if se1 <= SE_RULE:
        out.update(scheme="S1", eval_rep1=False, projected_underpowered=False)
    elif se2 <= SE_RULE:
        out.update(scheme="S2", eval_rep1=False, projected_underpowered=False)
    else:
        out.update(scheme="S2", eval_rep1=True, projected_underpowered=True)
    out["rule"] = f"SE_proj <= {SE_RULE}"
    out["test_runs"] = n_test_runs
    return out


def cmd_power(args: argparse.Namespace) -> int:
    d = pilot_diffs(args.gen)
    res = project(d, {s: test_runs(args.fights, s) for s in ("S1", "S2")})
    res["pilot"] = os.path.abspath(args.gen)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0


def add_parser(sub: Any) -> None:
    p = sub.add_parser("power", help="power projection from the pilot -> power.json")
    p.add_argument("--gen", required=True, help="the pilot's gen dir")
    p.add_argument("--fights", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_power)
