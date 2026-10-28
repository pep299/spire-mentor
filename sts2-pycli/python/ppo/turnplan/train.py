"""`python -m ppo.turnplan.train`: the deck-aware turn-plan scorer (GH #830).

Learns to rank a turn's plans the way the teacher prices them. Per position,
the loss is the cross-entropy between `softmax(score)` over its plans and
`softmax(q / tau)` over their teacher prices (variable plan count: padded and
masked). Features are normalised with fit-fold moments, which the
checkpoint carries. Training stops early on the validation loss (patience 3)
and keeps the best epoch.

The validation fold is carved from the TRAIN split by run file
(`is_val_file`: `crc32("turnplan-val:" + file) % 5 == 0`, ~20 % of the train
files; salted so it is independent of the holdout rule). The holdout split is
never read here: it is the recall measurement's (`ppo.turnplan.recall`), and
early-stopping on it would bias that measurement.

    python -m ppo.turnplan.train --data <label dir>... --out <model dir> \\
        [--no-deck] [--tau 0.05] [--epochs 30] [--seed 0]

`--no-deck` zeroes the deck branch's output; the branch still exists, so
both arms' checkpoints have one shape. Capped positions train like any other
(their partial plan list is still a valid listwise target); the recall
measurement excludes them.

Output in `<out>`: `model.pt` (state dict, model config, normaliser),
`vocab.json` (train split only), `meta.json` (args, vocab sizes, normaliser,
fit / val fold sizes, best epoch, per-epoch train / val loss and val top-1) and
`train.log`
(one line per epoch).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ppo.turnplan.data import CTX_DIM, PLAN_DIM, Normaliser, Position, Vocab, build_vocab, load
from ppo.turnplan.label import split_of

DECK_DIM = 64
PLAN_EMB = 64
VAL_SALT = b"turnplan-val:"
VAL_MOD = 5


@dataclass
class TrainConfig:
    """Optimisation knobs; `meta.json` records them."""

    no_deck: bool = False
    tau: float = 0.05
    epochs: int = 30
    seed: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    patience: int = 3
    emb: int = 32
    hidden: int = 128


@dataclass
class Batch:
    """Positions collated: deck bags per position, plans flattened over the batch.

    Plan `i` belongs to position `plan_pos[i]` at slot `plan_slot[i]`; `q` and
    `mask` are `[B, N]`, padded to the batch's largest plan list.
    """

    deck_cards: torch.Tensor
    deck_off: torch.Tensor
    relics: torch.Tensor
    relic_off: torch.Tensor
    powers: torch.Tensor
    power_amt: torch.Tensor
    power_off: torch.Tensor
    ctx: torch.Tensor
    plan_feats: torch.Tensor
    plan_cards: torch.Tensor
    plan_card_off: torch.Tensor
    plan_pos: torch.Tensor
    plan_slot: torch.Tensor
    q: torch.Tensor
    mask: torch.Tensor


def _bags(lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat ids + start offsets for an `EmbeddingBag` (empty bags allowed)."""
    lens = np.fromiter((len(x) for x in lists), np.int64, len(lists))
    off = np.zeros(len(lists), np.int64)
    np.cumsum(lens[:-1], out=off[1:])
    flat = np.fromiter((i for x in lists for i in x), np.int64, int(lens.sum()))
    return torch.from_numpy(flat), torch.from_numpy(off)


def collate(positions: list[Position]) -> Batch:
    """`positions` as one padded batch."""
    n = np.array([len(p.q) for p in positions])
    b, n_max = len(positions), int(n.max())
    q = np.zeros((b, n_max), np.float32)
    mask = np.zeros((b, n_max), bool)
    for i, p in enumerate(positions):
        q[i, : n[i]] = p.q
        mask[i, : n[i]] = True
    deck_cards, deck_off = _bags([p.deck_cards for p in positions])
    relics, relic_off = _bags([p.relics for p in positions])
    powers, power_off = _bags([[pid for pid, _ in p.powers] for p in positions])
    amt = [a for p in positions for _, a in p.powers]
    plan_cards, plan_card_off = _bags([c for p in positions for c in p.plan_cards])
    return Batch(
        deck_cards=deck_cards,
        deck_off=deck_off,
        relics=relics,
        relic_off=relic_off,
        powers=powers,
        power_amt=torch.tensor(amt, dtype=torch.float32),
        power_off=power_off,
        ctx=torch.from_numpy(np.stack([p.ctx for p in positions])),
        plan_feats=torch.from_numpy(np.concatenate([p.plan_feats for p in positions])),
        plan_cards=plan_cards,
        plan_card_off=plan_card_off,
        plan_pos=torch.from_numpy(np.repeat(np.arange(b), n)),
        plan_slot=torch.from_numpy(np.concatenate([np.arange(k) for k in n])),
        q=torch.from_numpy(q),
        mask=torch.from_numpy(mask),
    )


def is_val_file(file: str) -> bool:
    """Whether a TRAIN run file is in the early-stopping fold; by file, deterministic."""
    return zlib.crc32(VAL_SALT + file.encode()) % VAL_MOD == 0


def split_val(positions: list[Position]) -> tuple[list[Position], list[Position]]:
    """TRAIN positions as `(fit, val)` by run file (`is_val_file`).

    Raises:
        ValueError: A position comes from a holdout run file.
    """
    bad = sorted({p.file for p in positions if split_of(p.file) != "train"})
    if bad:
        raise ValueError(f"holdout run files in the train split: {bad[:3]}")
    val = [p for p in positions if is_val_file(p.file)]
    return [p for p in positions if not is_val_file(p.file)], val


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))


class PlanScorer(nn.Module):
    """Scores each plan of a position given the deck context.

    Deck branch: mean-pooled card and relic bags plus a power bag weighted by
    amount, through an MLP to 64. Plan branch: MLP(18 -> 64) plus a mean bag
    of the cards the plan plays. Score: MLP([deck, ctx, plan]) -> scalar.
    """

    # Train-split normaliser, registered as buffers so the checkpoint carries it.
    ctx_mean: torch.Tensor
    ctx_std: torch.Tensor
    plan_mean: torch.Tensor
    plan_std: torch.Tensor

    def __init__(
        self,
        sizes: dict[str, int],
        norm: Normaliser,
        *,
        no_deck: bool = False,
        emb: int = 32,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.sizes, self.no_deck, self.emb, self.hidden = dict(sizes), no_deck, emb, hidden
        self.card_bag = nn.EmbeddingBag(sizes["card"], emb, mode="mean")
        self.relic_bag = nn.EmbeddingBag(sizes["relic"], emb, mode="mean")
        self.power_bag = nn.EmbeddingBag(sizes["power"], emb, mode="sum")
        self.deck_mlp = _mlp(3 * emb, hidden, DECK_DIM)
        self.plan_mlp = _mlp(PLAN_DIM, hidden, PLAN_EMB)
        self.plan_card_bag = nn.EmbeddingBag(sizes["card"], PLAN_EMB, mode="mean")
        self.score_mlp = _mlp(DECK_DIM + CTX_DIM + PLAN_EMB, hidden, 1)
        for name in ("ctx_mean", "ctx_std", "plan_mean", "plan_std"):
            self.register_buffer(name, torch.from_numpy(getattr(norm, name).copy()))

    def normaliser(self) -> Normaliser:
        return Normaliser(
            *(
                getattr(self, k).cpu().numpy()
                for k in ("ctx_mean", "ctx_std", "plan_mean", "plan_std")
            )
        )

    def config(self) -> dict[str, Any]:
        return {
            "sizes": self.sizes,
            "no_deck": self.no_deck,
            "emb": self.emb,
            "hidden": self.hidden,
        }

    def forward(self, b: Batch) -> torch.Tensor:
        """Scores `[B, N]`; padded slots hold 0 (mask them with `b.mask`)."""
        bsz = b.ctx.shape[0]
        if self.no_deck:
            deck = b.ctx.new_zeros(bsz, DECK_DIM)
        else:
            deck = self.deck_mlp(
                torch.cat(
                    [
                        self.card_bag(b.deck_cards, b.deck_off),
                        self.relic_bag(b.relics, b.relic_off),
                        self.power_bag(b.powers, b.power_off, per_sample_weights=b.power_amt),
                    ],
                    1,
                )
            )
        ctx = (b.ctx - self.ctx_mean) / self.ctx_std
        feats = (b.plan_feats - self.plan_mean) / self.plan_std
        plan = self.plan_mlp(feats) + self.plan_card_bag(b.plan_cards, b.plan_card_off)
        s = self.score_mlp(torch.cat([deck[b.plan_pos], ctx[b.plan_pos], plan], 1)).squeeze(1)
        out = s.new_zeros(b.mask.shape)
        out[b.plan_pos, b.plan_slot] = s
        return out


def listwise_loss(
    scores: torch.Tensor, q: torch.Tensor, mask: torch.Tensor, tau: float
) -> torch.Tensor:
    """Per-position cross-entropy of `softmax(scores)` against `softmax(q / tau)`.

    Args:
        scores: `[B, N]` model scores.
        q: `[B, N]` teacher prices.
        mask: `[B, N]` true on real plans; padded slots never touch the loss.
        tau: Target temperature.

    Returns:
        `[B]` losses.
    """
    neg = torch.finfo(scores.dtype).min
    target = torch.softmax((q / tau).masked_fill(~mask, neg), 1)
    logp = torch.log_softmax(scores.masked_fill(~mask, neg), 1)
    return -torch.where(mask, target * logp, torch.zeros_like(logp)).sum(1)


def _chunks(positions: list[Position], size: int) -> list[list[Position]]:
    return [positions[i : i + size] for i in range(0, len(positions), size)]


@torch.no_grad()
def score(model: PlanScorer, positions: list[Position], batch_size: int = 256) -> list[np.ndarray]:
    """Each position's plan scores, in its plan order."""
    model.eval()
    out: list[np.ndarray] = []
    for chunk in _chunks(positions, batch_size):
        s = model(collate(chunk)).numpy()
        out += [s[i, : len(p.q)].copy() for i, p in enumerate(chunk)]
    return out


@torch.no_grad()
def _evaluate(model: PlanScorer, positions: list[Position], tau: float) -> tuple[float, float]:
    """Mean per-position loss and top-1 (the top-scored plan carries the max q)."""
    model.eval()
    losses: list[float] = []
    hits = 0
    for chunk in _chunks(positions, 256):
        b = collate(chunk)
        s = model(b)
        losses += listwise_loss(s, b.q, b.mask, tau).tolist()
        for i, p in enumerate(chunk):
            hits += bool(p.q[int(np.argmax(s[i, : len(p.q)].numpy()))] == p.q.max())
    return float(np.mean(losses)), hits / len(positions)


def fit(
    train: list[Position],
    val: list[Position],
    vocab: Vocab,
    cfg: TrainConfig,
    log: Any = None,
) -> tuple[PlanScorer, dict[str, Any]]:
    """Train a `PlanScorer`, early-stopped on the validation loss.

    Args:
        train: Fit positions.
        val: Validation positions (early stopping and the reported curve);
            never the recall holdout.
        vocab: The train-split vocab the positions were mapped with.
        cfg: Knobs.
        log: Called with one line per epoch; default prints to stderr.

    Returns:
        The best-epoch model and its meta (best epoch, per-epoch curves).
    """
    if not train or not val:
        raise ValueError(f"need fit and val positions, got {len(train)} / {len(val)}")
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    norm = Normaliser.fit(train)
    model = PlanScorer(vocab.sizes(), norm, no_deck=cfg.no_deck, emb=cfg.emb, hidden=cfg.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    curves: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_top1": []}
    best_loss, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        total, count = 0.0, 0
        for idx in _chunks([train[i] for i in rng.permutation(len(train))], cfg.batch_size):
            b = collate(idx)
            loss = listwise_loss(model(b), b.q, b.mask, cfg.tau)
            opt.zero_grad()
            loss.mean().backward()
            opt.step()
            total += float(loss.detach().sum())
            count += len(idx)
        v_loss, v_top1 = _evaluate(model, val, cfg.tau)
        curves["train_loss"].append(total / count)
        curves["val_loss"].append(v_loss)
        curves["val_top1"].append(v_top1)
        if v_loss < best_loss:
            best_loss, best_epoch = v_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(
            f"epoch {epoch}: train loss {total / count:.4f}, val loss {v_loss:.4f}, "
            f"val top-1 {v_top1:.3f}, best epoch {best_epoch} ({time.time() - t0:.1f} s)"
        )
        if epoch - best_epoch >= cfg.patience:
            log(f"early stop: no val improvement for {cfg.patience} epochs")
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    meta = {"best_epoch": best_epoch, "best_val_loss": best_loss, **curves}
    return model, meta


def save(model: PlanScorer, vocab: Vocab, meta: dict[str, Any], out: str | Path) -> None:
    """`model.pt`, `vocab.json` and `meta.json` into `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config(),
            "normaliser": model.normaliser().to_dict(),
        },
        out / "model.pt",
    )
    vocab.save(out / "vocab.json")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_model(model_dir: str | Path) -> tuple[PlanScorer, Vocab]:
    """The scorer and vocab `save` wrote to `model_dir`, in eval mode."""
    model_dir = Path(model_dir)
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = PlanScorer(
        cfg["sizes"],
        Normaliser.from_dict(ckpt["normaliser"]),
        no_deck=cfg["no_deck"],
        emb=cfg["emb"],
        hidden=cfg["hidden"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, Vocab.load(model_dir / "vocab.json")


def _parse(argv: list[str]) -> argparse.Namespace:
    d = TrainConfig()
    ap = argparse.ArgumentParser(
        prog="python -m ppo.turnplan.train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", nargs="+", required=True, help="label generator output dirs")
    ap.add_argument("--out", required=True, help="model dir")
    ap.add_argument("--no-deck", action="store_true", help="zero the deck branch")
    ap.add_argument("--tau", type=float, default=d.tau)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--patience", type=int, default=d.patience)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """`python -m ppo.turnplan.train ...`; returns the exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _parse(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    def log(msg: str) -> None:
        line = f"[turnplan.train] {msg}"
        print(line, file=sys.stderr, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

    cfg = TrainConfig(
        no_deck=args.no_deck,
        tau=args.tau,
        epochs=args.epochs,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    vocab = build_vocab(args.data)
    train, val = split_val(load(args.data, "train", vocab))
    capped = sum(p.cap_hit for p in train + val)
    n_files = (len({p.file for p in train}), len({p.file for p in val}))
    log(
        f"{len(train)} fit / {len(val)} val positions ({n_files[0]} / {n_files[1]} run files, "
        f"{capped} capped, kept; holdout not read), vocab {vocab.sizes()}, config {asdict(cfg)}"
    )
    model, meta = fit(train, val, vocab, cfg, log)
    meta = {
        "args": vars(args),
        "config": asdict(cfg),
        "vocab_sizes": vocab.sizes(),
        "normaliser": model.normaliser().to_dict(),
        "n_train": len(train),
        "n_val": len(val),
        "n_train_files": n_files[0],
        "n_val_files": n_files[1],
        "val_fold": f"crc32({VAL_SALT.decode()!r} + file) % {VAL_MOD} == 0 among train files",
        "n_capped": capped,
        **meta,
    }
    save(model, vocab, meta, out)
    log(f"best epoch {meta['best_epoch']}, val loss {meta['best_val_loss']:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
