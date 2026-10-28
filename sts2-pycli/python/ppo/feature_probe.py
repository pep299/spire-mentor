"""Offline feature-informativeness probe — minutes per idea, no RL.

The question this answers
------------------------
"Does this candidate observation feature carry outcome-relevant information
that the current encoding does not already have?"

It answers it on a **frozen corpus** of recorded runs, by fitting the run
policy's value head with and without the feature and comparing held-out
**explained variance** of the realized shaped return. No environment stepping,
no combat, no rollouts — so ten ideas fit in an afternoon instead of ten
campaigns fitting in a year.

What it is NOT
--------------
A verdict. Value-informativeness is **necessary** for a feature to help, not
sufficient:

- A feature can predict the return better without ever changing which action
  the policy would pick, and only the latter is worth a campaign.
- Policy cross-entropy on this corpus **cannot** improve, no matter how good
  the feature is: the net that generated the corpus could not condition on it,
  so the demonstrated actions carry no signal about it. A flat CE here is the
  expected result, not a null. (It is still printed — a CE that *degrades*
  means the extra table is costing capacity, which is worth knowing.)

So: use it to kill ideas cheaply and to rank the survivors. Promote a survivor
to `ppo.screen`, and only a screen survivor to a campaign.

Because the encoder schema is fixed at import time (`ppo.schema`), each arm
runs in its own interpreter — the same reason `ppo.screen` shells out.

Usage
-----
    PYTHONPATH=python python -m ppo.feature_probe \\
        --corpus /tmp/ppo-corpus/champion.jsonl \\
        --feature boss --repeats 3 --out /tmp/probe-boss

Record the corpus once and reuse it forever:

    PYTHONPATH=python python -m ppo.probe --policy net \\
        --ckpt data/ppo/<champion>.pt --record --seeds 2000 \\
        --out /tmp/ppo-corpus/champion.jsonl

Note the corpus does **not** need re-recording for the act-boss feature: the
boss identity is already in every observation (`full_map()`'s boss meta-node),
the encoder has simply never read it. A feature that needs a genuinely new obs
field does require a fresh corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .bc import Sample, load_corpus
from .model import RunPolicyNet, collate
from .rewards import ShapingConfig


@torch.no_grad()
def value_metrics(net: RunPolicyNet, samples: list[Sample], device, minibatch: int = 256) -> dict:
    """Held-out value-head quality on the realized shaped return.

    `explained_variance` is the headline: 1 - Var(ret - v) / Var(ret), the
    same quantity the trainer logs per iteration, so a probe number and a
    training number are directly comparable. Unlike raw MSE it is invariant to
    the return's scale, which matters because the shaping offset moves it.
    """
    net.eval()
    preds: list[float] = []
    rets: list[float] = []
    ce_total = 0.0
    n_ce = 0
    for s0 in range(0, len(samples), minibatch):
        mb = samples[s0 : s0 + minibatch]
        b = collate([s.feat for s in mb], device)
        logits, value, _ = net(b)
        act = torch.tensor([s.chosen for s in mb], dtype=torch.long, device=device)
        ce_total += float(nn.functional.cross_entropy(logits, act).item()) * len(mb)
        n_ce += len(mb)
        preds.extend(value.tolist())
        rets.extend(s.ret for s in mb)
    net.train()
    p = np.asarray(preds, dtype=np.float64)
    r = np.asarray(rets, dtype=np.float64)
    var = float(r.var())
    ev = 1.0 - float(((r - p) ** 2).mean()) / var if var > 0 else float("nan")
    return {
        "explained_variance": ev,
        "value_mse": float(((r - p) ** 2).mean()),
        "policy_ce": ce_total / max(1, n_ce),
        "n_val": len(samples),
    }


def fit_once(
    train_s: list[Sample],
    val_s: list[Sample],
    seed: int,
    *,
    epochs: int,
    minibatch: int,
    lr: float,
    value_coef: float,
    device,
    dim: int,
    n_layers: int,
    n_heads: int,
) -> dict:
    """One fit → held-out value metrics. `seed` varies the init and the shuffle.

    Repeats matter here: two fits of the same architecture on the same corpus
    differ by more than a small feature's effect, so a single-fit comparison
    would mostly measure initialization luck.
    """
    torch.manual_seed(seed)
    net = RunPolicyNet(dim=dim, n_layers=n_layers, n_heads=n_heads).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    order = np.arange(len(train_s))
    best = None
    for _ in range(epochs):
        rng.shuffle(order)
        for s0 in range(0, len(order), minibatch):
            mb = [train_s[i] for i in order[s0 : s0 + minibatch]]
            b = collate([s.feat for s in mb], device)
            logits, value, _ = net(b)
            act = torch.tensor([s.chosen for s in mb], dtype=torch.long, device=device)
            ret = torch.tensor([s.ret for s in mb], dtype=torch.float32, device=device)
            loss = nn.functional.cross_entropy(logits, act) + value_coef * nn.functional.mse_loss(
                value, ret
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        m = value_metrics(net, val_s, device)
        # Best-EV epoch, not last: the value head overfits this corpus well
        # before the policy head does, and comparing two arms at a fixed epoch
        # count would compare one converged fit against one that is not.
        if best is None or m["explained_variance"] > best["explained_variance"]:
            best = m
    assert best is not None
    return best


def welch_p(a: list[float], b: list[float]) -> float:
    """Two-sided Welch t-test p, normal-approximated.

    Deliberately crude: with 3-5 repeats it exists to stop a 0.002 EV gap from
    being written up as a finding, not to support a precise claim.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return 0.0 if ma != mb else 1.0
    z = abs(ma - mb) / se
    return math.erfc(z / math.sqrt(2.0))


def corpus_variation(corpus: str, feature: str, limit: int = 0) -> dict:
    """How much the corpus actually varies in `feature`.

    A null from a corpus where the feature is constant says nothing about the
    feature — it says the corpus could not have shown an effect. Measured
    rather than left as advice in the report, because a caveat nobody checks
    is not a guard.
    """
    from .encoder import act_boss_kind

    extract = {"boss": act_boss_kind}.get(feature)
    if extract is None:
        return {"supported": False}
    counts: dict[str, int] = {}
    n_runs = 0
    with open(corpus, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            decisions = row.get("decisions") or []
            if not decisions or row.get("outcome") in ("error", "timeout"):
                continue
            for d in decisions:
                key = str(extract(d["obs"]))
                counts[key] = counts.get(key, 0) + 1
            n_runs += 1
            if limit and n_runs >= limit:
                break
    total = sum(counts.values()) or 1
    top = max(counts.values(), default=0) / total
    return {
        "supported": True,
        "distinct_values": len(counts),
        "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:10]),
        "max_share": top,
    }


def run_fits(args: argparse.Namespace) -> dict:
    """Fit `--repeats` nets under WHATEVER schema this interpreter imported."""
    from . import schema

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    t0 = time.time()
    train_s, val_s, _ = load_corpus(args.corpus, ShapingConfig(), limit=args.limit)
    print(
        f"features={sorted(schema.ACTIVE_FEATURES)} N_CAT={schema.N_CAT} "
        f"device={device} featurized in {time.time() - t0:.0f}s",
        flush=True,
    )
    runs = []
    for i in range(args.repeats):
        m = fit_once(
            train_s,
            val_s,
            seed=i,
            epochs=args.epochs,
            minibatch=args.minibatch,
            lr=args.lr,
            value_coef=args.value_coef,
            device=device,
            dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        )
        print(
            f"  repeat {i}: EV={m['explained_variance']:.4f} "
            f"mse={m['value_mse']:.4f} ce={m['policy_ce']:.4f}",
            flush=True,
        )
        runs.append(m)
    return {
        "features": sorted(schema.ACTIVE_FEATURES),
        "n_cat": schema.N_CAT,
        "n_train": len(train_s),
        "n_val": len(val_s),
        "runs": runs,
    }


def _summarize(result: dict) -> dict[str, list[float]]:
    return {
        key: [r[key] for r in result["runs"]]
        for key in ("explained_variance", "value_mse", "policy_ce")
    }


MIN_VAL_DECISIONS = 2000


def report(base: dict, feat: dict, feature: str, variation: dict) -> dict:
    a, b = _summarize(base), _summarize(feat)
    out: dict[str, Any] = {
        "feature": feature,
        "baseline": base,
        "with_feature": feat,
        "corpus_variation": variation,
    }

    print("\n" + "=" * 72)
    print(f"FEATURE PROBE  '{feature}'   ({base['n_train']} train / {base['n_val']} val decisions)")
    print("=" * 72)

    # Two ways this probe can produce a meaningless null. Both are stated
    # BEFORE the numbers, so a run that cannot answer the question does not
    # get read as an answer.
    inconclusive = []
    if variation.get("supported"):
        print(
            f"corpus variation in '{feature}': {variation['distinct_values']} distinct values, "
            f"most common {variation['max_share']:.0%} of decisions  {variation['counts']}"
        )
        if variation["distinct_values"] < 2:
            inconclusive.append(
                f"the corpus has only {variation['distinct_values']} distinct value(s) of "
                f"'{feature}' — it could not have shown an effect"
            )
        elif variation["max_share"] > 0.95:
            inconclusive.append(
                f"'{feature}' is {variation['max_share']:.0%} one value in this corpus — "
                "near-constant, so a null is about the corpus, not the feature"
            )
    else:
        print(f"corpus variation in '{feature}': not measurable (no extractor registered)")
    if base["n_val"] < MIN_VAL_DECISIONS:
        inconclusive.append(
            f"only {base['n_val']} held-out decisions (< {MIN_VAL_DECISIONS}) — "
            "record a larger corpus before believing either direction"
        )
    if inconclusive:
        print("\n*** INCONCLUSIVE ***")
        for reason in inconclusive:
            print(f"  - {reason}")
    out["inconclusive"] = inconclusive
    print()
    print(f"{'metric':22s} {'baseline':>18s} {'+feature':>18s} {'delta':>10s} {'p':>8s}")
    for key, better in (
        ("explained_variance", "higher"),
        ("value_mse", "lower"),
        ("policy_ce", "lower"),
    ):
        ma, mb = statistics.mean(a[key]), statistics.mean(b[key])
        sa = statistics.stdev(a[key]) if len(a[key]) > 1 else 0.0
        sb = statistics.stdev(b[key]) if len(b[key]) > 1 else 0.0
        p = welch_p(a[key], b[key])
        print(
            f"{key:22s} {ma:10.4f}±{sa:<7.4f} {mb:10.4f}±{sb:<7.4f} "
            f"{mb - ma:+10.4f} {p:8.3f}   ({better} is better)"
        )
        out[key] = {"baseline": ma, "with_feature": mb, "delta": mb - ma, "p": p}

    ev = out["explained_variance"]
    print()
    if inconclusive:
        print(f"→ INCONCLUSIVE for '{feature}' — see the reasons above. Do not file this")
        print("  as a null; fix the corpus and re-run.")
    elif ev["delta"] > 0 and ev["p"] < 0.05:
        print(f"→ '{feature}' carries value-relevant information the current encoding lacks.")
        print("  NECESSARY, not sufficient: promote it to `ppo.screen` to find out whether")
        print("  the policy can act on it. Do not read this as a predicted win rate.")
    else:
        print(f"→ no detectable value-head gain from '{feature}' on this corpus.")
        print("  That is evidence against, not proof: a feature can matter for the ACTION")
        print("  without predicting the return of the demonstrator's own trajectory.")
        print("  Check the corpus actually varies in the feature before filing it as a null.")
    print("\npolicy_ce is expected to be FLAT — the demonstrator could not condition on")
    print("the feature, so its actions carry no signal about it. A ce that DEGRADES means")
    print("the extra table is costing capacity.")
    return out


_ARM_ENV = "STS2_PPO_FEATURE_PROBE_ARM"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", required=True, help="JSONL from `ppo.probe --record`")
    ap.add_argument("--feature", required=True, help="feature to test (see ppo.schema)")
    ap.add_argument("--out", default="", help="directory for per-arm JSON + report")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--value-coef", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="cap on runs loaded (smoke)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    args = ap.parse_args(argv)

    arm = os.environ.get(_ARM_ENV)
    if arm is not None:
        # Child: fit under this interpreter's schema and hand the result back.
        result = run_fits(args)
        with open(arm, "w", encoding="utf-8") as f:
            json.dump(result, f)
        return 0

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.corpus)), "probe")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for label, features in (("baseline", ""), ("with_feature", args.feature)):
        path = os.path.join(out_dir, f"{label}.json")
        if not os.path.exists(path):
            env = dict(os.environ, STS2_PPO_FEATURES=features, **{_ARM_ENV: path})
            print(f"\n--- {label} (STS2_PPO_FEATURES={features!r}) ---", flush=True)
            proc = subprocess.run(
                [sys.executable, "-m", "ppo.feature_probe", *(argv or sys.argv[1:])],
                env=env,
                check=False,
            )
            if proc.returncode != 0:
                raise SystemExit(f"{label} arm failed (rc={proc.returncode})")
        else:
            print(f"reusing {path}", flush=True)
        results[label] = json.load(open(path, encoding="utf-8"))

    variation = corpus_variation(args.corpus, args.feature, limit=args.limit)
    summary = report(results["baseline"], results["with_feature"], args.feature, variation)
    path = os.path.join(out_dir, "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nreport -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
