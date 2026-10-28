"""Deck-sensitivity test for a **transformer** combat surrogate.

The question
------------
`eval_combat_surrogate_net.py` gates a surrogate on whether it can rank states
*within* one `(encounter, ascension)` cell. The shipped surrogate is a single
hidden layer over a flattened context vector plus a 16-dim encounter
embedding, and the M1 eval found it barely beat a per-cell lookup table. That
is a plausible *architecture* limit rather than a limit on the idea: a flat
MLP over a 1482-dim bag has no natural way to represent "this deck".

This script asks whether a transformer that tokenizes the deck can. It is a
**feasibility probe, not a trainer for a shipped artifact** — it exports no
weights, because the point is to decide whether the surrogate line is worth
building at all.

Read it as a gate on two things at once:

* the full-fight surrogate (`ppo/env.py --combat surrogate`), which needs a
  deck gradient or it screens nothing; and
* an AlphaZero-style *leaf evaluator*, which needs the same representational
  question answered and carries far less risk (the real engine still runs the
  fight, so errors cannot compound and the policy cannot farm them).

Both die if a net cannot see deck quality at all. That is why this runs first.

Baselines, so the number means something
----------------------------------------
* ``bucket_mean`` — predict the training mean of the row's
  `(enc_idx, ascension)` cell. This is the lookup table, and its within-bucket
  rho is 0 by construction (a constant cannot rank). It is reported to confirm
  the metric does what it claims, not as a contender.
* ``mlp`` — the shipped architecture, retrained here on the same rows and the
  same split. Without it, a transformer result would confound architecture
  with "this harvest is simply better than the M1 one" — and it is: these
  labels are not saturated (see the harvest's mean win_rate), which was the
  M1 eval's stated confound.

Usage
-----
    PYTHONPATH=python python -m train_combat_surrogate_transformer \\
        --labels /path/labels_ic.jsonl --schema /path/labels_ic.jsonl.schema.json \\
        --device cuda:0 --epochs 12 --out /path/report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_combat_surrogate_net import (
    DECK_SENSITIVITY_MIN_RHO,
    _within_bucket_rank_agreement,
)

ASCENSION_SCALAR_IDX = 9  # schema scalar_names[9] == "ascension"
VAL_SEED_MOD = 5  # seed % 5 == 0 -> holdout; split by SEED, never by row


def load_rows(path: str, limit: int = 0, source_tag: str = "") -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if source_tag:
                r["_corpus"] = source_tag
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


# NOTE on mixing corpora: the holdout rule is `seed % VAL_SEED_MOD == 0`, a
# pure function of the seed, so a seed appearing in BOTH a synthetic and a
# realistic harvest lands on the same side of the split in both. Overlapping
# seed ranges therefore cannot leak, and the corpora need no de-duplication.
# (They also describe different states — synthetic synthesizes from the seed,
# realistic drives a run with it — so a shared seed is not shared data.)


class Batch:
    """Pre-tokenized rows. Token layout: [CLS, ENC, SCALARS, cards..., relics...]."""

    def __init__(self, rows: list[dict], n_cards: int, n_relics: int, max_len: int):
        n = len(rows)
        self.n_scalars = len(rows[0]["scalars"])
        # Token *type* ids let one embedding table serve cards and relics without
        # their index spaces colliding: card i and relic i are different things.
        self.tok_id = np.zeros((n, max_len), dtype=np.int64)
        self.tok_type = np.zeros((n, max_len), dtype=np.int64)
        self.tok_num = np.zeros((n, max_len, 2), dtype=np.float32)  # (count, upgraded)
        self.mask = np.ones((n, max_len), dtype=bool)  # True == PAD (torch convention)
        self.scalars = np.zeros((n, self.n_scalars), dtype=np.float32)
        self.enc = np.zeros(n, dtype=np.int64)
        self.win = np.zeros(n, dtype=np.float32)
        self.hp = np.zeros(n, dtype=np.float32)
        self.bucket: list[tuple] = []

        truncated = 0
        for i, r in enumerate(rows):
            toks: list[tuple[int, int, float, float]] = [(0, 0, 0.0, 0.0), (0, 1, 0.0, 0.0)]
            for card_idx, count, upg in r["deck"]:
                toks.append((int(card_idx), 2, float(count), float(upg)))
            for relic_idx, count in r["relics"]:
                toks.append((int(relic_idx), 3, float(count), 0.0))
            if len(toks) > max_len:
                truncated += 1
                toks = toks[:max_len]
            for j, (tid, ttype, a, b) in enumerate(toks):
                self.tok_id[i, j] = tid
                self.tok_type[i, j] = ttype
                self.tok_num[i, j, 0] = a
                self.tok_num[i, j, 1] = b
                self.mask[i, j] = False
            self.scalars[i] = np.asarray(r["scalars"], dtype=np.float32)
            self.enc[i] = int(r["enc_idx"])
            self.win[i] = float(r["win_rate"])
            self.hp[i] = float(r["hp_frac_lost"])
            self.bucket.append((int(r["enc_idx"]), float(r["scalars"][ASCENSION_SCALAR_IDX])))
        self.truncated = truncated
        self.n_cards = n_cards
        self.n_relics = n_relics

    def __len__(self) -> int:
        return len(self.enc)


class SurrogateTransformer(nn.Module):
    """Deck-as-tokens surrogate. Same depth/width as `RunPolicyNet` (192/3/4)."""

    def __init__(
        self,
        n_cards: int,
        n_relics: int,
        n_enc: int,
        n_scalars: int,
        dim: int,
        layers: int,
        heads: int,
    ):
        super().__init__()
        # One shared id table sized for the larger space, disambiguated by type.
        self.id_emb = nn.Embedding(max(n_cards, n_relics) + 1, dim)
        self.type_emb = nn.Embedding(4, dim)  # CLS / ENC / card / relic
        self.num_proj = nn.Linear(2, dim)
        self.enc_emb = nn.Embedding(n_enc + 1, dim)  # +1 = reserved OOD row
        self.scalar_proj = nn.Sequential(nn.Linear(n_scalars, dim), nn.ReLU(), nn.Linear(dim, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.win_head = nn.Linear(dim, 1)
        self.hp_head = nn.Linear(dim, 1)

    def forward(self, tok_id, tok_type, tok_num, mask, scalars, enc):
        h = self.id_emb(tok_id) + self.type_emb(tok_type) + self.num_proj(tok_num)
        # Slot 0 is CLS (pure type embedding); slot 1 carries the encounter and
        # slot 1's numeric channel is unused, so overwrite rather than add.
        h[:, 1, :] = self.enc_emb(enc) + self.type_emb(torch.ones_like(enc))
        h[:, 0, :] = h[:, 0, :] + self.scalar_proj(scalars)
        h = self.tr(h, src_key_padding_mask=mask)
        cls = self.norm(h[:, 0, :])
        return self.win_head(cls).squeeze(-1), torch.sigmoid(self.hp_head(cls)).squeeze(-1)


class SurrogateMLP(nn.Module):
    """The shipped architecture: densified context ++ encounter embedding, one
    hidden layer. Mirrors `train_combat_surrogate_net.py::SurrogateMLP`."""

    def __init__(self, context_dim: int, n_enc: int, enc_dim: int, hidden: int):
        super().__init__()
        self.enc_emb = nn.Embedding(n_enc + 1, enc_dim)
        self.trunk = nn.Linear(context_dim + enc_dim, hidden)
        self.win_head = nn.Linear(hidden, 1)
        self.hp_head = nn.Linear(hidden, 1)

    def forward(self, ctx, enc):
        h = torch.relu(self.trunk(torch.cat([ctx, self.enc_emb(enc)], dim=-1)))
        return self.win_head(h).squeeze(-1), torch.sigmoid(self.hp_head(h)).squeeze(-1)


def densify(rows: list[dict], n_cards: int, n_relics: int) -> np.ndarray:
    """Schema layout: deck(count,upg) per card ++ relic counts ++ scalars."""
    n_scalars = len(rows[0]["scalars"])
    out = np.zeros((len(rows), 2 * n_cards + n_relics + n_scalars), dtype=np.float32)
    for i, r in enumerate(rows):
        for card_idx, count, upg in r["deck"]:
            out[i, 2 * int(card_idx)] = count
            out[i, 2 * int(card_idx) + 1] = upg
        for relic_idx, count in r["relics"]:
            out[i, 2 * n_cards + int(relic_idx)] = count
        out[i, 2 * n_cards + n_relics :] = np.asarray(r["scalars"], dtype=np.float32)
    return out


def standardize(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return ((train - mu) / sd, *[(o - mu) / sd for o in others])


def train_model(model, feed, n_train, epochs, minibatch, lr, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * math.ceil(n_train / minibatch))
    )
    bce, sl1 = nn.BCEWithLogitsLoss(), nn.SmoothL1Loss()
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n_train)
        tot, nb = 0.0, 0
        for s0 in range(0, n_train, minibatch):
            idx = order[s0 : s0 + minibatch]
            win_logit, hp_pred, win_t, hp_t = feed(model, idx, train=True)
            loss = bce(win_logit, win_t) + sl1(hp_pred, hp_t)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        print(f"  [{label}] epoch {ep}: loss {tot / max(nb, 1):.4f}", flush=True)
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--schema", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--min-bucket-rows", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--labels-extra",
        default="",
        help=(
            "second corpus mixed into TRAINING only (scoring stays on --labels). "
            "The intended use is realistic --labels + synthetic --labels-extra: "
            "realistic states are pre-filtered by survival and saturate near "
            "win_rate 1.0, so synthetic rows supply the deck diversity and "
            "dynamic range that realistic harvests structurally lack."
        ),
    )
    args = ap.parse_args(argv)

    schema_path = args.schema or f"{args.labels}.schema.json"
    schema = json.load(open(schema_path, encoding="utf-8"))
    n_cards, n_relics, n_enc = schema["n_cards"], schema["n_relics"], schema["n_encounters"]

    rows = load_rows(args.labels, args.limit, source_tag="primary")
    extra = load_rows(args.labels_extra, 0, source_tag="extra") if args.labels_extra else []
    all_rows = rows + extra

    tr_rows = [r for r in all_rows if r["seed"] % VAL_SEED_MOD != 0]
    # Scoring happens on the PRIMARY corpus only. Mixing an easier corpus into
    # the holdout would move the gate's number for a reason that has nothing to
    # do with the net — the whole question is whether it ranks decks on the
    # distribution a screen would actually run in.
    va_rows = [r for r in rows if r["seed"] % VAL_SEED_MOD == 0]
    print(
        f"rows={len(all_rows)} (primary {len(rows)} + extra {len(extra)})  "
        f"train={len(tr_rows)} val={len(va_rows)} [val is PRIMARY only]  "
        f"(split by seed % {VAL_SEED_MOD})  n_cards={n_cards} n_enc={n_enc}",
        flush=True,
    )
    for tag, rs in (("primary", rows), ("extra", extra)):
        if rs:
            print(
                f"  {tag:<8} win_rate mean={np.mean([r['win_rate'] for r in rs]):.3f}  "
                f"hp_frac_lost mean={np.mean([r['hp_frac_lost'] for r in rs]):.3f}  "
                # win_rate is wins/probes, so a saturated row is exactly 1.0;
                # >= keeps ruff happy without changing what is counted.
                f"frac win==1 {np.mean([r['win_rate'] >= 1.0 for r in rs]):.3f}",
                flush=True,
            )

    device = torch.device(args.device)
    tb = Batch(tr_rows, n_cards, n_relics, args.max_len)
    vb = Batch(va_rows, n_cards, n_relics, args.max_len)
    if tb.truncated or vb.truncated:
        print(f"WARNING: truncated {tb.truncated + vb.truncated} rows at max_len", flush=True)

    def to_t(x, dt=None):
        return torch.as_tensor(x, dtype=dt, device=device)

    T = {
        k: (to_t(getattr(tb, k)), to_t(getattr(vb, k)))
        for k in ("tok_id", "tok_type", "tok_num", "mask", "scalars", "enc", "win", "hp")
    }

    results: dict[str, Any] = {}
    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ── transformer ──────────────────────────────────────────────────────
    tf = SurrogateTransformer(
        n_cards, n_relics, n_enc, tb.n_scalars, args.dim, args.layers, args.heads
    ).to(device)
    print(f"transformer params: {sum(p.numel() for p in tf.parameters()):,}", flush=True)

    def feed_tf(model, idx, train):
        s = 0 if train else 1
        i = to_t(idx, torch.long)
        wl, hp = model(
            T["tok_id"][s][i],
            T["tok_type"][s][i],
            T["tok_num"][s][i],
            T["mask"][s][i],
            T["scalars"][s][i],
            T["enc"][s][i],
        )
        return wl, hp, T["win"][s][i], T["hp"][s][i]

    t0 = time.time()
    train_model(tf, feed_tf, len(tb), args.epochs, args.minibatch, args.lr, "transformer")
    print(f"transformer trained in {time.time() - t0:.0f}s", flush=True)

    tf.eval()
    with torch.no_grad():
        wl, hp = [], []
        for s0 in range(0, len(vb), 1024):
            idx = np.arange(s0, min(s0 + 1024, len(vb)))
            a, b, _, _ = feed_tf(tf, idx, train=False)
            wl.append(torch.sigmoid(a).cpu().numpy())
            hp.append(b.cpu().numpy())
    preds["transformer"] = (np.concatenate(wl), np.concatenate(hp))

    # ── MLP baseline (shipped architecture, same rows/split) ─────────────
    ctx_tr, ctx_va = standardize(
        densify(tr_rows, n_cards, n_relics), densify(va_rows, n_cards, n_relics)
    )
    C = (to_t(ctx_tr), to_t(ctx_va))
    mlp = SurrogateMLP(ctx_tr.shape[1], n_enc, 16, 128).to(device)
    print(f"mlp params: {sum(p.numel() for p in mlp.parameters()):,}", flush=True)

    def feed_mlp(model, idx, train):
        s = 0 if train else 1
        i = to_t(idx, torch.long)
        wl, hp = model(C[s][i], T["enc"][s][i])
        return wl, hp, T["win"][s][i], T["hp"][s][i]

    train_model(mlp, feed_mlp, len(tb), args.epochs, args.minibatch, args.lr, "mlp")
    mlp.eval()
    with torch.no_grad():
        wl, hp = [], []
        for s0 in range(0, len(vb), 1024):
            idx = np.arange(s0, min(s0 + 1024, len(vb)))
            a, b, _, _ = feed_mlp(mlp, idx, train=False)
            wl.append(torch.sigmoid(a).cpu().numpy())
            hp.append(b.cpu().numpy())
    preds["mlp"] = (np.concatenate(wl), np.concatenate(hp))

    # ── lookup-table baseline: the training mean of the row's cell ───────
    cell: dict[tuple, list[int]] = defaultdict(list)
    for i, k in enumerate(tb.bucket):
        cell[k].append(i)
    win_mu = {k: float(tb.win[v].mean()) for k, v in cell.items()}
    hp_mu = {k: float(tb.hp[v].mean()) for k, v in cell.items()}
    gw, gh = float(tb.win.mean()), float(tb.hp.mean())
    preds["bucket_mean"] = (
        np.asarray([win_mu.get(k, gw) for k in vb.bucket], dtype=np.float64),
        np.asarray([hp_mu.get(k, gh) for k in vb.bucket], dtype=np.float64),
    )

    # ── the gate ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("DECK SENSITIVITY — within-(encounter, ascension) rank agreement")
    print(f"held-out rows {len(vb)}   min rows/bucket {args.min_bucket_rows}")
    print("=" * 72)
    print(f"{'model':<14}{'win rho':>10}{'hp rho':>10}{'buckets':>10}{'win MAE':>10}{'hp MAE':>10}")
    for name in ("bucket_mean", "mlp", "transformer"):
        pw, ph = preds[name]
        rw = _within_bucket_rank_agreement(vb.bucket, pw, vb.win, args.min_bucket_rows)
        rh = _within_bucket_rank_agreement(vb.bucket, ph, vb.hp, args.min_bucket_rows)
        results[name] = {
            "win_rho": rw["mean_rho"],
            "hp_rho": rh["mean_rho"],
            "win_buckets": rw["n_buckets"],
            "hp_buckets": rh["n_buckets"],
            "win_mae": float(np.mean(np.abs(pw - vb.win))),
            "hp_mae": float(np.mean(np.abs(ph - vb.hp))),
        }
        r = results[name]
        print(
            f"{name:<14}{r['win_rho']:>10.4f}{r['hp_rho']:>10.4f}"
            f"{rh['n_buckets']:>10}{r['win_mae']:>10.4f}{r['hp_mae']:>10.4f}"
        )

    tf_hp = results["transformer"]["hp_rho"]
    tf_win = results["transformer"]["win_rho"]
    n_buckets = max(results["transformer"]["hp_buckets"], results["transformer"]["win_buckets"])
    finite = [v for v in (tf_hp, tf_win) if not math.isnan(v)]
    best = max(finite) if finite else float("nan")
    print(f"\ngate: DECK_SENSITIVITY_MIN_RHO = {DECK_SENSITIVITY_MIN_RHO}")
    # A gate that cannot compute its own statistic must say so. Without this,
    # "no bucket had enough held-out rows" prints as FAIL and gets filed as
    # evidence against the surrogate, when it is evidence about the harvest.
    if not finite or n_buckets == 0:
        print("*** INCONCLUSIVE *** — no (encounter, ascension) cell had both")
        print(f"  >= {args.min_bucket_rows} held-out rows and non-degenerate truth.")
        print(f"  {len(vb)} held-out rows spread over {len(set(vb.bucket))} cells is too thin.")
        print("  Harvest more rows (or lower --min-bucket-rows); do NOT file this as a")
        print("  null — it says nothing about whether a transformer can rank decks.")
        results["verdict"] = "inconclusive"
    elif best >= DECK_SENSITIVITY_MIN_RHO:
        print(f"→ PASS ({best:.4f}). A transformer CAN rank decks within a fixed encounter.")
        results["verdict"] = "pass"
        print("  Necessary, not sufficient: this says the representation exists, not")
        print("  that a policy trained against it transfers. Prefer the leaf-evaluator")
        print("  route, where the real engine still runs the fight.")
    else:
        print(f"→ FAIL ({best:.4f} < {DECK_SENSITIVITY_MIN_RHO}).")
        results["verdict"] = "fail"
        print("  Even with card tokens the net is close to a per-cell lookup table, so")
        print("  surrogate-world combat presents no deck-quality gradient. Screen")
        print("  deck-shaped changes at low MCTS sims instead.")
    print("\nNote: win_rate here is bimodal (mostly 0 or 1), so within-bucket ties")
    print("depress win rho structurally. hp_frac_lost is the continuous target and")
    print("is the more informative of the two.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "labels": args.labels,
                    "n_train": len(tb),
                    "n_val": len(vb),
                    "min_bucket_rows": args.min_bucket_rows,
                    "gate_min_rho": DECK_SENSITIVITY_MIN_RHO,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nreport -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
