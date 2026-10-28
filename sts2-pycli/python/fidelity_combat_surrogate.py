"""Stage 1 kill-gate: does a surrogate-driven run stay on the real run's HP curve?

Per-fight MAE can look fine while a small SYSTEMATIC bias compounds over a run
into a world with no risk. A simulated agent that never dies would silently
ruin PPO training — the run policy would train against a combat model that
hands out free HP, and every card/path decision would be miscalibrated.

Design
------
Two closed-loop simulations over the SAME fight sequence, differing ONLY in the
outcome model:

* reference — HP driven by the K-probe MCTS labels. By construction this
  reproduces the run's actual `entry_hp` trace exactly, because the non-combat
  HP delta (campfire / event / potion) is *defined* as the residual
  `entry_hp[i+1] - entry_hp[i] + hp_frac_probe[i]*max_hp` and is replayed
  identically in both arms. So it cancels, and any divergence is the surrogate.
* surrogate — HP driven by the net's `hp_frac_lost` prediction.

Closed-loop matters: the surrogate is re-queried at its OWN simulated HP
(scalars `hp` and `hp_frac`), so under-prediction feeds back — a net that
lowballs damage keeps the agent healthy, which makes it lowball again. Runaway
feedback is exactly the failure this gate exists to catch, and an open-loop
cumulative-error sum cannot see it.

CAVEAT baked into the verdict: the realistic corpus is driven by
`HeuristicPolicy` (combat_surrogate_data.rs:536), not PPO, and 98.3% of its
rows are Act 1 (mean 6.2 fights/run). A FAIL here is decisive. A PASS only
clears Act 1 against an off-policy driver.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoder_combat_surrogate import (
    EncoderBatch,
    EncoderSurrogate,
    build_id_maps,
    load_champion,
)
from train_combat_surrogate_transformer import (
    VAL_SEED_MOD,
    Batch,
    SurrogateTransformer,
    load_rows,
    train_model,
)

HP_IDX, HP_FRAC_IDX = 0, 2
ACT_IDX, ACT_FLOOR_IDX = 5, 6


def feed_fn(batch: Batch, dev: str):
    t = {
        "tok_id": torch.from_numpy(batch.tok_id).to(dev),
        "tok_type": torch.from_numpy(batch.tok_type).to(dev),
        "tok_num": torch.from_numpy(batch.tok_num).to(dev),
        "mask": torch.from_numpy(batch.mask).to(dev),
        "scalars": torch.from_numpy(batch.scalars).to(dev),
        "enc": torch.from_numpy(batch.enc).to(dev),
        "win": torch.from_numpy(batch.win).to(dev),
        "hp": torch.from_numpy(batch.hp).to(dev),
    }

    def feed(model, idx, train: bool):
        i = torch.from_numpy(idx).to(dev)
        wl, hp = model(
            t["tok_id"][i],
            t["tok_type"][i],
            t["tok_num"][i],
            t["mask"][i],
            t["scalars"][i],
            t["enc"][i],
        )
        return wl, hp, t["win"][i], t["hp"][i]

    return feed, t


class DistributionalSurrogate(SurrogateTransformer):
    """Adds a SPREAD head, so the net predicts a distribution over fight
    outcomes rather than a conditional mean.

    A mean-only head models a world with the right average damage and no
    catastrophes — measured 2026-08-13, 0.68 run survival against a 0.29
    reference — because death is a tail event and the tail was averaged away.
    The extra head is trained directly against the harvest's per-row spread
    across the K probes (`hp_frac_lost_std`), which is the observed spread of
    the very distribution we want back.

    Regressing the spread beats a Gaussian NLL here: the label is already a
    MEAN over K=12 probes, so an NLL fit against it would learn the (much
    smaller) variance of that mean, not the variance of a single fight.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        dim = self.win_head.in_features
        self.sd_head = nn.Linear(dim, 1)

    def forward(self, tok_id, tok_type, tok_num, mask, scalars, enc):
        h = self.id_emb(tok_id) + self.type_emb(tok_type) + self.num_proj(tok_num)
        h[:, 1, :] = self.enc_emb(enc) + self.type_emb(torch.ones_like(enc))
        h[:, 0, :] = h[:, 0, :] + self.scalar_proj(scalars)
        h = self.tr(h, src_key_padding_mask=mask)
        cls = self.norm(h[:, 0, :])
        return (
            self.win_head(cls).squeeze(-1),
            torch.sigmoid(self.hp_head(cls)).squeeze(-1),
            # softplus: a standard deviation is non-negative, and unlike a
            # sigmoid it does not saturate away the large spreads that are
            # exactly the catastrophic fights we are trying to recover.
            nn.functional.softplus(self.sd_head(cls)).squeeze(-1),
        )


@torch.no_grad()
def predict(model, batch: Batch, dev: str, scalars_override: np.ndarray | None = None):
    """Predict (win_prob, hp_frac_lost[, spread]) for every row in `batch`."""
    model.eval()
    sc = batch.scalars if scalars_override is None else scalars_override
    got = model(
        torch.from_numpy(batch.tok_id).to(dev),
        torch.from_numpy(batch.tok_type).to(dev),
        torch.from_numpy(batch.tok_num).to(dev),
        torch.from_numpy(batch.mask).to(dev),
        torch.from_numpy(sc).to(dev),
        torch.from_numpy(batch.enc).to(dev),
    )
    out = [torch.sigmoid(got[0]).cpu().numpy(), got[1].cpu().numpy()]
    if len(got) > 2:
        out.append(got[2].cpu().numpy())
    return tuple(out)


def train_distributional(model, batch, sd_true, dev, epochs, minibatch, lr):
    """Same optimizer/schedule as `train_model`, plus the spread term."""
    n = len(batch)
    t = {
        k: torch.from_numpy(v).to(dev)
        for k, v in (
            ("tok_id", batch.tok_id),
            ("tok_type", batch.tok_type),
            ("tok_num", batch.tok_num),
            ("mask", batch.mask),
            ("scalars", batch.scalars),
            ("enc", batch.enc),
            ("win", batch.win),
            ("hp", batch.hp),
            ("sd", sd_true),
        )
    }
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * math.ceil(n / minibatch))
    )
    bce, sl1 = nn.BCEWithLogitsLoss(), nn.SmoothL1Loss()
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n)
        tot, nb = 0.0, 0
        for s0 in range(0, n, minibatch):
            i = torch.from_numpy(order[s0 : s0 + minibatch]).to(dev)
            wl, hp, sd = model(
                t["tok_id"][i],
                t["tok_type"][i],
                t["tok_num"][i],
                t["mask"][i],
                t["scalars"][i],
                t["enc"][i],
            )
            loss = bce(wl, t["win"][i]) + sl1(hp, t["hp"][i]) + sl1(sd, t["sd"][i])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        print(f"  [distributional] epoch {ep}: loss {tot / max(nb, 1):.4f}", flush=True)
    return model


def simulate(runs, model, dev, make_batch, sample=False, seed=0):
    """Closed-loop replay. Returns per-run traces for both arms.

    With `sample=True` both arms draw an OUTCOME rather than subtracting a
    mean, and they draw it with the SAME standard normal deviate `z` per fight.
    That pairing is what makes the comparison clean: the two arms differ only
    in (mu, sigma), never in the luck of the draw, so any divergence is still
    attributable to the surrogate and not to sampling noise.

    The reference samples from the label's own (mean, std) — the K-probe
    estimate of what MCTS combat actually does — which is the distribution the
    surrogate is supposed to be reproducing.
    """
    rng = np.random.default_rng(seed)
    out = []
    for run_seed, rows in runs:
        max_hp = float(rows[0]["max_hp"])
        # Non-combat HP delta between fight i and i+1, DEFINED so that the
        # reference arm reproduces the observed entry_hp trace exactly.
        rest = []
        for i in range(len(rows) - 1):
            after = rows[i]["entry_hp"] - rows[i]["hp_frac_lost"] * max_hp
            rest.append(rows[i + 1]["entry_hp"] - after)

        ref_hp = sur_hp = float(rows[0]["entry_hp"])
        ref_trace, sur_trace = [ref_hp], [sur_hp]
        ref_surv = sur_surv = 1.0
        sur_dead_at = ref_dead_at = None
        for i, r in enumerate(rows):
            # Surrogate is re-queried at its OWN simulated HP (closed loop).
            row_sim = dict(r)
            sc = list(r["scalars"])
            sc[HP_IDX] = sur_hp
            sc[HP_FRAC_IDX] = max(sur_hp, 0.0) / max_hp
            row_sim["scalars"] = sc
            pred_out = predict(model, make_batch([row_sim]), dev)
            win_hat, hp_hat = pred_out[0], pred_out[1]

            if sample:
                z = rng.standard_normal()  # ONE deviate, shared by both arms
                sd_hat = float(pred_out[2][0]) if len(pred_out) > 2 else 0.0
                ref_loss = min(
                    max(r["hp_frac_lost"] + r.get("hp_frac_lost_std", 0.0) * z, 0.0), 1.0
                )
                sur_loss = min(max(float(hp_hat[0]) + sd_hat * z, 0.0), 1.0)
            else:
                ref_loss = r["hp_frac_lost"]
                sur_loss = float(hp_hat[0])

            ref_hp -= ref_loss * max_hp
            sur_hp -= sur_loss * max_hp
            ref_surv *= float(r["win_rate"])
            sur_surv *= float(win_hat[0])
            if ref_dead_at is None and ref_hp <= 0:
                ref_dead_at = i
            if sur_dead_at is None and sur_hp <= 0:
                sur_dead_at = i
            if i < len(rest):
                ref_hp += rest[i]
                sur_hp += rest[i]
            ref_trace.append(ref_hp)
            sur_trace.append(sur_hp)
        out.append(
            {
                "seed": run_seed,
                "n": len(rows),
                "max_hp": max_hp,
                "ref": ref_trace,
                "sur": sur_trace,
                "ref_surv": ref_surv,
                "sur_surv": sur_surv,
                "ref_dead_at": ref_dead_at,
                "sur_dead_at": sur_dead_at,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="realistic corpus (primary)")
    ap.add_argument("--labels-extra", default="", help="synthetic corpus, TRAIN only")
    ap.add_argument("--schema", default="")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--save-ckpt", default="", help="persist weights so re-analysis need not retrain"
    )
    ap.add_argument(
        "--encoder",
        default="",
        metavar="CHAMPION_PT",
        help=(
            "wire in the PPO champion's representation: potion IDENTITY tokens plus "
            "card/relic/potion embeddings copied from this checkpoint. Implies "
            "--distributional. Needs a schema with card_ids/relic_ids/potion_ids and "
            "rows with a `potions` list (harvested 2026-08-14 onward)."
        ),
    )
    ap.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="keep the transferred embeddings fixed instead of fine-tuning them",
    )
    ap.add_argument(
        "--distributional",
        action="store_true",
        help=(
            "add a spread head and SAMPLE fight outcomes in the loop instead of "
            "subtracting a mean. Requires `hp_frac_lost_std` in the labels (harvested "
            "2026-08-14 onward); the run aborts rather than silently degrading if the "
            "column is missing or all-zero."
        ),
    )
    args = ap.parse_args()

    schema = json.load(open(args.schema or f"{args.labels}.schema.json", encoding="utf-8"))
    n_cards, n_relics, n_enc = schema["n_cards"], schema["n_relics"], schema["n_encounters"]

    rows = load_rows(args.labels, source_tag="primary")
    extra = load_rows(args.labels_extra, source_tag="extra") if args.labels_extra else []
    tr_rows = [r for r in rows + extra if r["seed"] % VAL_SEED_MOD != 0]
    va_rows = [r for r in rows if r["seed"] % VAL_SEED_MOD == 0]
    print(
        f"train={len(tr_rows)} (primary {len(rows)} + extra {len(extra)})  "
        f"val={len(va_rows)} [realistic only]",
        flush=True,
    )

    dev = args.device
    if args.encoder:
        args.distributional = True  # the spread head is kept; this adds to it
        missing = [k for k in ("card_ids", "relic_ids", "potion_ids") if k not in schema]
        if missing:
            print(
                f"*** ABORT *** --encoder needs {missing} in the schema. Re-harvest with a "
                "build from 2026-08-14 or later."
            )
            return 2
        if not any(r.get("potions") for r in tr_rows):
            print(
                "*** ABORT *** --encoder needs a `potions` list on the rows and every "
                "training row is empty. Re-harvest."
            )
            return 2
        id_maps = build_id_maps(schema)
        champ = load_champion(args.encoder)
        tb = EncoderBatch(tr_rows, id_maps, args.max_len)
        vb = EncoderBatch(va_rows, id_maps, args.max_len)
        held = sum(len(r.get("potions", [])) for r in va_rows)
        print(
            f"encoder: {tb.n_potion_tokens} potion tokens in train, {held} in holdout; "
            f"truncated {tb.truncated} rows",
            flush=True,
        )
        model = EncoderSurrogate(champ, n_enc, tb.n_scalars, freeze=args.freeze_embeddings).to(dev)
        sd_tr = tb.sd
        if not np.any(sd_tr > 0):
            print(
                "*** ABORT *** --encoder implies --distributional, which needs "
                "`hp_frac_lost_std`; every training row is 0.0."
            )
            return 2
        print(
            f"spread label present: {(sd_tr > 0).mean():.1%} of training rows non-zero, "
            f"mean {sd_tr.mean():.4f}",
            flush=True,
        )
        train_distributional(model, tb, sd_tr, dev, args.epochs, args.minibatch, args.lr)
    elif args.distributional:
        tb = Batch(tr_rows, n_cards, n_relics, args.max_len)
        vb = Batch(va_rows, n_cards, n_relics, args.max_len)
        sd_tr = np.array([r.get("hp_frac_lost_std", 0.0) for r in tr_rows], dtype=np.float32)
        # A silently-absent column would train a spread head on all zeros and
        # then sample nothing — reproducing the mean-only failure while looking
        # like the fix had been tested. Fail loudly instead.
        if not np.any(sd_tr > 0):
            print(
                "*** ABORT *** --distributional needs `hp_frac_lost_std` in the labels, "
                "and every training row has 0.0. Re-harvest with a build from 2026-08-14 "
                "or later."
            )
            return 2
        nz = float((sd_tr > 0).mean())
        print(
            f"spread label present: {nz:.1%} of training rows non-zero, mean {sd_tr.mean():.4f}",
            flush=True,
        )
        model = DistributionalSurrogate(n_cards, n_relics, n_enc, tb.n_scalars, 192, 3, 4).to(dev)
        train_distributional(model, tb, sd_tr, dev, args.epochs, args.minibatch, args.lr)
    else:
        tb = Batch(tr_rows, n_cards, n_relics, args.max_len)
        vb = Batch(va_rows, n_cards, n_relics, args.max_len)
        model = SurrogateTransformer(n_cards, n_relics, n_enc, tb.n_scalars, 192, 3, 4).to(dev)
        feed, _ = feed_fn(tb, dev)
        train_model(model, feed, len(tb), args.epochs, args.minibatch, args.lr, "transformer")
    if args.save_ckpt:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "n_cards": n_cards,
                "n_relics": n_relics,
                "n_enc": n_enc,
                "n_scalars": tb.n_scalars,
                "max_len": args.max_len,
            },
            args.save_ckpt,
        )
        print(f"checkpoint -> {args.save_ckpt}", flush=True)

    # ---- per-fight calibration on the holdout -------------------------------
    pred = predict(model, vb, dev)
    win_hat, hp_hat = pred[0], pred[1]
    sd_hat = pred[2] if len(pred) > 2 else None
    hp_true, win_true = vb.hp, vb.win
    cell_mean = defaultdict(list)
    for r in tr_rows:
        cell_mean[int(r["enc_idx"]), float(r["scalars"][9])].append(r["hp_frac_lost"])
    lut = {k: float(np.mean(v)) for k, v in cell_mean.items()}
    glob = float(np.mean([r["hp_frac_lost"] for r in tr_rows]))
    lut_pred = np.array([lut.get(b, glob) for b in vb.bucket], dtype=np.float32)

    print("\n=== 1. per-fight calibration (realistic holdout) ===")
    print(f"{'predictor':<16}{'hp MAE':>10}{'hp BIAS':>10}")
    for name, p in (
        ("surrogate", hp_hat),
        ("lookup table", lut_pred),
        ("global mean", np.full_like(hp_true, glob)),
    ):
        print(f"{name:<16}{np.abs(p - hp_true).mean():>10.4f}{(p - hp_true).mean():>+10.4f}")
    print(
        f"\nwin prob: predicted mean {win_hat.mean():.4f} vs actual {win_true.mean():.4f} "
        f"(bias {win_hat.mean() - win_true.mean():+.4f})"
    )

    # Does the surrogate keep the TAIL? A mean-seeking regression head compresses
    # the spread, and death is a tail event — so a net with a fine mean and a
    # squashed upper tail produces a world that is systematically less lethal.
    print("\n=== 1b. spread / tail of hp_frac_lost (death is a TAIL event) ===")
    print(f"{'':<12}{'std':>8}{'p50':>8}{'p90':>8}{'p99':>8}{'frac>0.3':>10}{'frac>0.5':>10}")
    for name, v in (("predicted", hp_hat), ("actual", hp_true)):
        q = np.percentile(v, [50, 90, 99])
        print(
            f"{name:<12}{v.std():>8.4f}{q[0]:>8.4f}{q[1]:>8.4f}{q[2]:>8.4f}"
            f"{(v > 0.3).mean():>10.4f}{(v > 0.5).mean():>10.4f}"
        )
    print(f"  spread ratio (pred std / actual std): {hp_hat.std() / hp_true.std():.3f}")
    if sd_hat is not None:
        # THE metric for the distributional fix. The mean head's spread is
        # expected to stay compressed — a conditional mean SHOULD vary less
        # than the outcomes it averages. What has to match is the spread of
        # what the loop actually consumes: SAMPLED outcomes.
        z = np.random.default_rng(0).standard_normal(len(hp_hat)).astype(np.float32)
        sampled = np.clip(hp_hat + sd_hat * z, 0.0, 1.0)
        q = np.percentile(sampled, [50, 90, 99])
        print(
            f"{'SAMPLED':<12}{sampled.std():>8.4f}{q[0]:>8.4f}{q[1]:>8.4f}{q[2]:>8.4f}"
            f"{(sampled > 0.3).mean():>10.4f}{(sampled > 0.5).mean():>10.4f}"
        )
        print(
            f"  spread ratio (SAMPLED std / actual std): "
            f"{sampled.std() / hp_true.std():.3f}   <-- the one that matters"
        )
        print(
            f"  predicted spread head: mean {sd_hat.mean():.4f} vs label spread mean "
            f"{np.mean([r.get('hp_frac_lost_std', 0.0) for r in va_rows]):.4f}"
        )

    # ---- compounding --------------------------------------------------------
    by_seed = defaultdict(list)
    for r in va_rows:
        by_seed[r["seed"]].append(r)
    runs = [
        (s, sorted(v, key=lambda r: (r["scalars"][ACT_IDX], r["scalars"][ACT_FLOOR_IDX])))
        for s, v in sorted(by_seed.items())
    ]
    if args.encoder:

        def make_batch(rs):
            return EncoderBatch(rs, id_maps, args.max_len)
    else:

        def make_batch(rs):
            return Batch(rs, n_cards, n_relics, args.max_len)

    traces = simulate(runs, model, dev, make_batch, sample=args.distributional)

    print("\n=== 2. COMPOUNDING: surrogate HP vs reference HP, by fight index ===")
    print(f"{'after fight':<13}{'runs':>7}{'mean err HP':>13}{'mean |err|':>12}{'as % maxHP':>12}")
    rows_out = []
    for i in (1, 2, 3, 5, 8, 12, 20):
        errs = [(t["sur"][i] - t["ref"][i], t["max_hp"]) for t in traces if len(t["sur"]) > i]
        if len(errs) < 10:
            continue
        e = np.array([x[0] for x in errs])
        mh = np.array([x[1] for x in errs])
        print(
            f"{i:<13}{len(e):>7}{e.mean():>+13.2f}{np.abs(e).mean():>12.2f}"
            f"{100 * np.abs(e).mean() / mh.mean():>11.1f}%"
        )
        rows_out.append(
            {"fight": i, "n": len(e), "bias_hp": float(e.mean()), "mae_hp": float(np.abs(e).mean())}
        )

    final = np.array([t["sur"][-1] - t["ref"][-1] for t in traces])
    mh = np.array([t["max_hp"] for t in traces])
    print(
        f"\nend of run: mean err {final.mean():+.2f} HP  "
        f"({100 * final.mean() / mh.mean():+.1f}% of max HP), |err| {np.abs(final).mean():.2f}"
    )

    ref_deaths = sum(1 for t in traces if t["ref_dead_at"] is not None)
    sur_deaths = sum(1 for t in traces if t["sur_dead_at"] is not None)
    print("\n=== 3. death rate (HP<=0 in closed loop) ===")
    print(
        f"  reference (probe labels): {ref_deaths}/{len(traces)} = {ref_deaths / len(traces):.3f}"
    )
    print(
        f"  surrogate               : {sur_deaths}/{len(traces)} = {sur_deaths / len(traces):.3f}"
    )
    print(
        f"  survival-prob product   : ref {np.mean([t['ref_surv'] for t in traces]):.4f}  "
        f"sur {np.mean([t['sur_surv'] for t in traces]):.4f}"
    )

    if args.out:
        json.dump(
            {
                "per_fight": rows_out,
                "final_bias_hp": float(final.mean()),
                "ref_death_rate": ref_deaths / len(traces),
                "sur_death_rate": sur_deaths / len(traces),
            },
            open(args.out, "w", encoding="utf-8"),
            indent=2,
        )
        print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
