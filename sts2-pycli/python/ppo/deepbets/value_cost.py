"""G-D inference cost of a ReturnValueNet: `python -m ppo.deepbets.value_cost` (GH #830).

Reports, for a net file (or a hypothetical size):

* parameters (embedding tables included) and dense MACs / FLOPs per forward;
* the embedding-sum work, from the mean number of distinct (zone, card)
  entries per labelled state;
* a single-thread microbenchmark of the same forward pass in C, compiled for
  the baseline x86-64 target the wheel is built for (`build_info()["target_cpu"]`
  is None: no AVX/FMA), as the stand-in for a Rust port's MLP time (both are
  LLVM/GCC-vectorised SSE2 loops over the same arrays; plan G2's bar is
  <= 15 us median including features, kill > 30 us);
* the feature extraction measured through the Python bindings, an UPPER
  bound on the Rust cost (each call pays PyO3 conversion, `rollout_features`
  also builds every legal action's features).

No Rust is built: the C file is one translation unit, compiled with gcc.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any

from ppo.deepbets import value_data as vd

C_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define IN %(in_dim)d
#define H %(h)d
#define H2 %(h2)d
#define NT %(n_tok)d
#define D %(d)d
#define DM %(dm)d
#define NS %(n_sp)d
#define TOK %(tok)d
#define DENSE %(dense)d
static float W1[H][IN], B1[H], W2[H2][H], B2[H2], W3[3][H2], B3[3];
static float EC[NT][D], EM[NS][DM], MU[DENSE], SD[DENSE];
static float rnd(void) { return (float)rand() / RAND_MAX - 0.5f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static float W1T[IN][H], W2T[H][H2];
__attribute__((noinline)) static float forward_axpy(const float *raw, const int *tok, const int *zone,
                                                   const float *cnt, const int *sp, const float *mf) {
  float x[IN], h[H], h2[H2];
  int k = 0;
  for (int i = 0; i < DENSE; i++) x[k++] = (raw[i] - MU[i]) * SD[i];
  float *cards = x + k;
  for (int i = 0; i < 5 * D; i++) cards[i] = 0.f;
  for (int t = 0; t < TOK; t++) { float *e = EC[tok[t]]; float c = cnt[t]; float *o = cards + zone[t] * D;
    for (int j = 0; j < D; j++) o[j] += c * e[j]; }
  k += 5 * D;
  for (int s = 0; s < 5; s++) { for (int j = 0; j < DM; j++) x[k++] = EM[sp[s]][j];
    for (int j = 0; j < 3; j++) x[k++] = mf[s * 3 + j]; }
  x[k++] = 0.f; x[k++] = 1.f;
  for (int i = 0; i < H; i++) h[i] = B1[i];
  for (int j = 0; j < IN; j++) { float xj = x[j]; const float *w = W1T[j]; for (int i = 0; i < H; i++) h[i] += w[i] * xj; }
  for (int i = 0; i < H; i++) h[i] = h[i] > 0 ? h[i] : 0;
  for (int i = 0; i < H2; i++) h2[i] = B2[i];
  for (int j = 0; j < H; j++) { float hj = h[j]; const float *w = W2T[j]; for (int i = 0; i < H2; i++) h2[i] += w[i] * hj; }
  for (int i = 0; i < H2; i++) h2[i] = h2[i] > 0 ? h2[i] : 0;
  float z[3];
  for (int i = 0; i < 3; i++) { float a = B3[i]; for (int j = 0; j < H2; j++) a += W3[i][j] * h2[j]; z[i] = a; }
  float p = 1.f / (1.f + expf(-z[0])), mw = 1.f + 0.5f / (1.f + expf(-z[1])), ml = -1.45f + 0.4f / (1.f + expf(-z[2]));
  return p * mw + (1.f - p) * ml;
}
__attribute__((noinline)) static float forward(const float *raw, const int *tok, const int *zone,
                                              const float *cnt, const int *sp, const float *mf) {
  float x[IN], h[H], h2[H2];
  int k = 0;
  for (int i = 0; i < DENSE; i++) x[k++] = (raw[i] - MU[i]) * SD[i];
  float *cards = x + k;
  for (int i = 0; i < 5 * D; i++) cards[i] = 0.f;
  for (int t = 0; t < TOK; t++) { float *e = EC[tok[t]]; float c = cnt[t]; float *o = cards + zone[t] * D;
    for (int j = 0; j < D; j++) o[j] += c * e[j]; }
  k += 5 * D;
  for (int s = 0; s < 5; s++) { for (int j = 0; j < DM; j++) x[k++] = EM[sp[s]][j];
    for (int j = 0; j < 3; j++) x[k++] = mf[s * 3 + j]; }
  x[k++] = 0.f; x[k++] = 1.f;
  for (int i = 0; i < H; i++) { float a = B1[i]; for (int j = 0; j < IN; j++) a += W1[i][j] * x[j]; h[i] = a > 0 ? a : 0; }
  for (int i = 0; i < H2; i++) { float a = B2[i]; for (int j = 0; j < H; j++) a += W2[i][j] * h[j]; h2[i] = a > 0 ? a : 0; }
  float z[3];
  for (int i = 0; i < 3; i++) { float a = B3[i]; for (int j = 0; j < H2; j++) a += W3[i][j] * h2[j]; z[i] = a; }
  float p = 1.f / (1.f + expf(-z[0])), mw = 1.f + 0.5f / (1.f + expf(-z[1])), ml = -1.45f + 0.4f / (1.f + expf(-z[2]));
  return p * mw + (1.f - p) * ml;
}
int main(void) {
  srand(1);
  for (int i = 0; i < H; i++) { B1[i] = rnd(); for (int j = 0; j < IN; j++) W1[i][j] = rnd() * 0.1f; }
  for (int i = 0; i < H2; i++) { B2[i] = rnd(); for (int j = 0; j < H; j++) W2[i][j] = rnd() * 0.1f; }
  for (int i = 0; i < 3; i++) { B3[i] = rnd(); for (int j = 0; j < H2; j++) W3[i][j] = rnd() * 0.1f; }
  for (int i = 0; i < H; i++) for (int j = 0; j < IN; j++) W1T[j][i] = W1[i][j];
  for (int i = 0; i < H2; i++) for (int j = 0; j < H; j++) W2T[j][i] = W2[i][j];
  for (int i = 0; i < NT; i++) for (int j = 0; j < D; j++) EC[i][j] = rnd();
  for (int i = 0; i < NS; i++) for (int j = 0; j < DM; j++) EM[i][j] = rnd();
  for (int i = 0; i < DENSE; i++) { MU[i] = rnd(); SD[i] = 1.f + rnd(); }
  enum { NIN = 256 };
  static float raw[NIN][DENSE], cnt[NIN][TOK + 1], mf[NIN][15];
  static int tok[NIN][TOK + 1], zone[NIN][TOK + 1], sp[NIN][5];
  for (int n = 0; n < NIN; n++) {
    for (int i = 0; i < DENSE; i++) raw[n][i] = rnd();
    for (int t = 0; t < TOK; t++) { tok[n][t] = rand() %% NT; zone[n][t] = rand() %% 5; cnt[n][t] = 1 + rand() %% 3; }
    for (int s = 0; s < 5; s++) sp[n][s] = rand() %% NS;
    for (int i = 0; i < 15; i++) mf[n][i] = rnd();
  }
  volatile float sink = 0;
  int reps = %(reps)d;
  double best[2] = {1e9, 1e9}, total[2] = {0, 0};
  for (int r = 0; r < 7; r++) {
    for (int f = 0; f < 2; f++) {
      double t0 = now();
      if (f == 0) for (int i = 0; i < reps; i++) { int n = i & (NIN - 1); sink += forward(raw[n], tok[n], zone[n], cnt[n], sp[n], mf[n]); }
      else for (int i = 0; i < reps; i++) { int n = i & (NIN - 1); sink += forward_axpy(raw[n], tok[n], zone[n], cnt[n], sp[n], mf[n]); }
      double dt = (now() - t0) / reps * 1e6;
      if (dt < best[f]) best[f] = dt;
      total[f] += dt;
    }
  }
  float a = forward(raw[0], tok[0], zone[0], cnt[0], sp[0], mf[0]), b = forward_axpy(raw[0], tok[0], zone[0], cnt[0], sp[0], mf[0]);
  printf("{\"dot_us_best\": %%.4f, \"dot_us_mean\": %%.4f, \"axpy_us_best\": %%.4f, \"axpy_us_mean\": %%.4f, \"max_abs_diff\": %%g, \"sink\": %%f}\n",
         best[0], total[0] / 7, best[1], total[1] / 7, fabs(a - b), (double)sink);
  return 0;
}
"""


def dims(net: dict[str, Any]) -> dict[str, int]:
    wide = net["variant"] == "B"
    dense = vd.STEP_DIM + (vd.WIDE_DIM if wide else 0)
    in_dim = dense + 5 * net["d_card"] + vd.MONSTER_SLOTS * (net["d_mon"] + vd.MON_FEATS) + 2
    return {
        "in_dim": in_dim,
        "dense": dense,
        "h": int(net["hidden"]),
        "h2": int(net["hidden"]) // 2,
        "n_tok": len(net["vocab"]),
        "n_sp": len(net["species"]),
        "d": int(net["d_card"]),
        "dm": int(net["d_mon"]),
    }


def counts(net: dict[str, Any], tok_per_state: float) -> dict[str, Any]:
    d = dims(net)
    dense_macs = d["in_dim"] * d["h"] + d["h"] * d["h2"] + d["h2"] * 3
    emb_macs = tok_per_state * d["d"]
    params = sum(_numel(v) for v in net["weights"].values())
    elementwise = 2 * d["dense"] + d["h"] + d["h2"] + 3 * 4
    return {
        **d,
        "params_total": params,
        "params_embeddings": d["n_tok"] * d["d"] + d["n_sp"] * d["dm"],
        "dense_macs": dense_macs,
        "embedding_sum_macs": round(emb_macs, 1),
        "tokens_per_state": round(tok_per_state, 2),
        "flops": round(2 * (dense_macs + emb_macs) + elementwise, 1),
    }


def _numel(v: Any) -> int:
    if isinstance(v, list):
        return sum(_numel(x) for x in v) if v and isinstance(v[0], list) else len(v)
    return 1


def c_bench(
    d: dict[str, int], tok: int, march: str, workdir: str, reps: int = 20_000
) -> dict[str, Any]:
    src = os.path.join(workdir, f"rvnet_bench_{march}.c")
    exe = src[:-2]
    with open(src, "w", encoding="utf-8") as f:
        f.write(C_SRC % {**d, "tok": max(1, tok), "reps": reps})
    subprocess.run(["gcc", "-O3", f"-march={march}", "-o", exe, src, "-lm"], check=True)
    out = subprocess.run(
        ["taskset", "-c", str(os.sched_getaffinity(0).pop()), exe],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def python_feature_costs(specs_dir: str | None, n: int) -> dict[str, float]:
    """Per-call cost of the feature bindings on real states (upper bounds)."""
    import sts2_rs

    from ppo.deepbets import value_label as vl

    if not specs_dir:
        states = [sts2_rs.CombatState(seed=s, encounter="Nibbit") for s in range(n)]
    else:
        smeta = json.load(open(os.path.join(specs_dir, "manifest.json"), encoding="utf-8"))
        groups = vl._spec_groups(specs_dir, None)
        random.Random(1).shuffle(groups)
        states = []
        for g in groups:
            states += [st for _, st in vl.rebuild_specs(smeta["args"]["traj"], g[:5])]
            if len(states) >= n:
                break
    bvs = [s.belief_view(0) for s in states[:n]]

    def tm(fn: Any, reps: int = 20) -> float:
        t0 = time.process_time()
        for _ in range(reps):
            for b in bvs:
                fn(b)
        return (time.process_time() - t0) / (reps * len(bvs)) * 1e6

    lp = sts2_rs.LearnedPlayout(open(vd.playout_net_identity()["path"], encoding="utf-8").read())
    t0 = time.process_time()
    res = lp.play_many(
        [b for b in bvs for _ in range(8)], list(range(8 * len(bvs))), 1.0, False, 200, True
    )
    dt = time.process_time() - t0
    steps = int(sum(struct.unpack(f"<{8 * len(bvs)}I", res["steps"])))
    return {
        "playout_step_us": round(dt / max(1, steps) * 1e6, 2),
        "playout_steps_per_playout": round(steps / (8 * len(bvs)), 2),
        "states": len(bvs),
        "call_floor_terminal_value_us": round(tm(lambda b: b.terminal_value()), 2),
        "rollout_features_v2_us": round(tm(lambda b: sts2_rs.rollout_features(b, version=2)), 2),
        "feature_vector_wide_us": round(tm(lambda b: b.feature_vector_wide()), 2),
        "state_to_dict_us": round(tm(lambda b: sts2_rs.state_to_dict(b), 5), 2),
        "belief_view_us": round(tm(lambda b: b.belief_view(0)), 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppo.deepbets.value_cost")
    ap.add_argument("--net", default=None, help="net JSON; omit to use --variant/--hidden/--vocab")
    ap.add_argument("--variant", choices=("A", "B"), default="A")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=600)
    ap.add_argument("--species", type=int, default=120)
    ap.add_argument("--labels", default=None, help="label dir: measured tokens per state")
    ap.add_argument("--tokens", type=float, default=24.0, help="tokens per state without --labels")
    ap.add_argument("--specs", default=None, help="spec dir for the feature timings")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.net:
        net = json.load(open(args.net, encoding="utf-8"))
    else:
        from ppo.deepbets.value_train import D_CARD, D_MON

        net = {"variant": args.variant, "hidden": args.hidden, "d_card": D_CARD, "d_mon": D_MON,
               "vocab": [""] * args.vocab, "species": [""] * args.species, "weights": {}}  # fmt: skip
    tok = args.tokens
    if args.labels:
        from ppo.deepbets.value_train import load_label_data

        d = load_label_data(args.labels)
        tok = len(d.f_tok) / max(1, len(d.ids))
    c = counts(net, tok)
    if not args.net:
        c["params_total"] = c["dense_macs"] + c["h"] + c["h2"] + 3 + c["params_embeddings"]
    with tempfile.TemporaryDirectory() as wd:
        bench = {m: c_bench(dims(net), round(tok), m, wd) for m in ("x86-64", "native")}
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    feats = python_feature_costs(args.specs, args.n)
    mlp = bench["x86-64"]["axpy_us_best"]
    floor = feats["call_floor_terminal_value_us"]
    # The playout computes StepCtx (plus every action's features, the playout
    # net and an engine step) at each recorded step: a tighter upper bound
    # than the Python-side rollout_features call.
    step_bound = min(feats["playout_step_us"], max(0.0, feats["rollout_features_v2_us"] - floor))
    est_feat = step_bound + (
        max(0.0, feats["feature_vector_wide_us"] - floor) if net["variant"] == "B" else 0.0
    )
    out = {
        "counts": c,
        "c_bench_us": bench,
        "python_feature_costs_us": feats,
        "estimate_rust_us": {
            "mlp_and_embeddings": round(mlp, 3),
            "features_upper_bound": round(est_feat + 1.0, 2),
            "total_upper_bound": round(mlp + est_feat + 1.0, 2),
            "mlp_dot_form": round(bench["x86-64"]["dot_us_best"], 3),
            "note": "mlp: the C axpy-form kernel (vectorises on baseline SSE2 without reassociation; "
            "the dot form cannot); features: StepCtx bounded by one whole recorded playout step "
            "(StepCtx + every action's features + the playout net + an engine step), plus "
            "feature_vector_wide minus the PyO3 call floor for variant B, plus ~1 us for the "
            "card/monster walk; an upper bound",
        },
        "load1": os.getloadavg()[0],
    }
    print(json.dumps(out, indent=1))
    if args.out:
        vd.write_json(args.out, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
