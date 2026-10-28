"""BC warm-start from the expert corpus (M-P4) + SL-sanity gate (M-P2).

Trains ``RunPolicyNet`` supervised on recorded expert runs
(``ppo.probe --policy expert --record``): policy cross-entropy on the
expert's chosen index, value MSE on the realized shaped return-to-go
(``ppo.rewards`` — same shaping the PPO trainer uses, so the warm-started
value head is on the PPO target scale), and the 0.1·dest-room aux CE.

Held-out split is BY RUN (seed % 20 == 0), never by decision — decisions
within a run are correlated. The printed per-surface top-1 accuracy vs the
uniform-random baseline is the M-P2 SL-sanity gate; event-payload coverage
(fraction of event decisions with ≥1 parsed nonzero gold/HP payload,
plan §3.2 target ≥70%) is measured on the same pass over the corpus.

The checkpoint (``{"state_dict": ...}``) is directly consumable by
``ppo.trainer --init-ckpt`` and ``ppo.net_policy.NetPolicy`` (G4: evaluate
with ``ppo.probe --policy net --ckpt <save>``).

Search-teacher distillation (GH #452): a corpus recorded with
``ppo.probe --policy net_search --record`` carries per-decision ``pi``
(the Gumbel completed-Q improved policy) and ``search_value``. Rows with
``pi`` train under soft CE instead of hard CE, and ``--value-target
search`` regresses the value head to the teacher's root value (same
shaped-return scale) where recorded. Only the pre-registered teacher gate
(docs/experiments/2026-08-18-d1-teacher-gate/) authorizes a distillation
run — a search teacher that does not beat the raw champion must not be
distilled.

Usage (from sts2-pycli):
    PYTHONPATH=python uv run --frozen python -m ppo.bc \
        --corpus /tmp/ppo-bc/expert.jsonl --save /tmp/ppo-bc/bc.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import vocab
from .encoder import _RE_ENUM_LABEL, event_payload, featurize_row
from .model import AUX_IGNORE, Featurized, RunPolicyNet, collate
from .rewards import ShapingConfig, shaped_rewards


@dataclass
class Sample:
    feat: Featurized
    chosen: int
    ret: float
    surface: str
    n_actions: int
    # Search-teacher distillation targets (GH #452): the improved policy
    # π′ over the candidate list and the search's root-value estimate,
    # recorded by `ppo.probe --policy net_search --record`. None on plain
    # expert/net corpora — the hard-CE / realized-return path is unchanged.
    pi: list[float] | None = None
    search_value: float | None = None


@dataclass
class PayloadCoverage:
    """Event-option semantic coverage over the corpus (plan §3.2, ≥0.70).

    An option counts as covered when the encoder gives it a semantic
    feature beyond ``option_idx``: a matched relic/card identity (Neow's
    labels are relic ids) or a nonzero parsed gold/HP payload.
    """

    n_event_decisions: int = 0
    n_covered_decisions: int = 0  # ≥1 option with a semantic feature
    n_options: int = 0
    n_covered_options: int = 0

    @property
    def decision_coverage(self) -> float:
        return self.n_covered_decisions / max(1, self.n_event_decisions)


def _measure_payload(d: dict, cov: PayloadCoverage) -> None:
    opts = d["obs"].get("event_options") or []
    max_hp = int(d["obs"].get("max_hp") or 1)
    any_covered = False
    for a in d["actions"]:
        if a.get("action") != "event_choice":
            continue
        opt = opts[a["idx"]] if 0 <= int(a.get("idx", -1)) < len(opts) else {}
        label = str(opt.get("label") or "")
        identity = bool(_RE_ENUM_LABEL.match(label)) and (
            vocab.relics().index(label) != vocab.relics().oov
            or vocab.cards().index(label) != vocab.cards().oov
        )
        gold, hp = event_payload(f"{label}. {opt.get('description') or ''}", max_hp)
        cov.n_options += 1
        if identity or abs(gold) > 1e-9 or abs(hp) > 1e-9:
            cov.n_covered_options += 1
            any_covered = True
    cov.n_event_decisions += 1
    cov.n_covered_decisions += 1 if any_covered else 0


def load_corpus(
    path: str, shaping: ShapingConfig, limit: int = 0, val_mod: int = 20
) -> tuple[list[Sample], list[Sample], PayloadCoverage]:
    """→ (train, val, payload coverage); split by run seed, not decision."""
    train: list[Sample] = []
    val: list[Sample] = []
    cov = PayloadCoverage()
    n_runs = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            decisions = row.get("decisions") or []
            if not decisions or row.get("outcome") in ("error", "timeout"):
                continue
            # Ascension backfilled for corpora predating the `metrics_snapshot`
            # key `rewards.total_floors` reads (GH #464).
            asc = int(row.get("ascension") or 0)
            rewards, _ = shaped_rewards(
                [{"ascension": asc, **d["metrics"]} for d in decisions],
                {"ascension": asc, **row["final"]},
                bool(row["won"]),
                shaping,
            )
            # Return-to-go: G_t = Σ_{k≥t} r_k (γ=1, terminal bootstrap 0).
            rets = np.cumsum(rewards[::-1])[::-1]
            sink = val if row["seed"] % val_mod == 0 else train
            for d, ret in zip(decisions, rets):
                if d["surface"] == "event":
                    _measure_payload(d, cov)
                pi = d.get("pi")
                sv = d.get("search_value")
                sink.append(
                    Sample(
                        feat=featurize_row(d["obs"], d["actions"], d["metrics"]),
                        chosen=int(d["chosen"]),
                        ret=float(ret),
                        surface=d["surface"],
                        n_actions=len(d["actions"]),
                        pi=[float(p) for p in pi] if pi is not None else None,
                        search_value=float(sv) if sv is not None else None,
                    )
                )
            n_runs += 1
            if limit and n_runs >= limit:
                break
    print(f"corpus: {n_runs} runs → {len(train)} train / {len(val)} val decisions", flush=True)
    return train, val, cov


def _soft_targets(mb: list[Sample], n_cols: int, device: torch.device) -> torch.Tensor:
    """[B, n_cols] policy targets: π′ rows where a sample carries one
    (renormalized over its candidates), one-hot on `chosen` otherwise.
    Reduces exactly to hard CE on a π-free corpus."""
    t = torch.zeros(len(mb), n_cols, device=device)
    for i, s in enumerate(mb):
        if s.pi is not None:
            row = torch.tensor(s.pi[: s.n_actions], dtype=torch.float32, device=device)
            t[i, : s.n_actions] = row / row.sum().clamp_min(1e-8)
        else:
            t[i, s.chosen] = 1.0
    return t


def _losses(
    net: RunPolicyNet,
    mb: list[Sample],
    value_coef: float,
    aux_coef: float,
    device: torch.device,
    value_target: str = "mc",
):
    b = collate([s.feat for s in mb], device)
    logits, value, aux = net(b)
    act = torch.tensor([s.chosen for s in mb], dtype=torch.long, device=device)
    ret = torch.tensor(
        [
            s.search_value if value_target == "search" and s.search_value is not None else s.ret
            for s in mb
        ],
        dtype=torch.float32,
        device=device,
    )
    if any(s.pi is not None for s in mb):
        # Soft CE to the search-improved policy (GH #452). Padded logit
        # slots are -inf-masked; their targets are exactly 0, so mask the
        # log-probs there to avoid 0 · (-inf) = NaN.
        target = _soft_targets(mb, logits.size(1), device)
        logp = F.log_softmax(logits, dim=-1)
        pi_loss = -(target * torch.where(target > 0, logp, logp.new_zeros(()))).sum(-1).mean()
    else:
        pi_loss = F.cross_entropy(logits, act)
    v_loss = F.mse_loss(value, ret)
    aux_loss = F.cross_entropy(
        aux.reshape(-1, aux.size(-1)), b.aux_room.reshape(-1), ignore_index=AUX_IGNORE
    )
    if torch.isnan(aux_loss):
        aux_loss = logits.new_zeros(())
    loss = pi_loss + value_coef * v_loss + aux_coef * aux_loss
    correct = (logits.argmax(dim=-1) == act).float()
    return loss, pi_loss, v_loss, correct


@torch.no_grad()
def evaluate(
    net: RunPolicyNet,
    samples: list[Sample],
    device: torch.device,
    minibatch: int = 256,
    value_target: str = "mc",
) -> dict:
    """Val loss + per-surface top-1 accuracy vs uniform baseline (SL gate)."""
    net.eval()
    hits: collections.Counter[str] = collections.Counter()
    counts: collections.Counter[str] = collections.Counter()
    inv_n: collections.defaultdict[str, float] = collections.defaultdict(float)
    tot_pi = tot_v = 0.0
    for s0 in range(0, len(samples), minibatch):
        mb = samples[s0 : s0 + minibatch]
        _, pi_loss, v_loss, correct = _losses(net, mb, 0.0, 0.0, device, value_target)
        tot_pi += float(pi_loss.item()) * len(mb)
        tot_v += float(v_loss.item()) * len(mb)
        for s, c in zip(mb, correct.tolist()):
            hits[s.surface] += int(c)
            counts[s.surface] += 1
            inv_n[s.surface] += 1.0 / max(1, s.n_actions)
    net.train()
    n = max(1, len(samples))
    per_surface = {
        k: {
            "acc": hits[k] / counts[k],
            "uniform": inv_n[k] / counts[k],
            "n": counts[k],
        }
        for k in sorted(counts)
    }
    return {
        "pi_loss": tot_pi / n,
        "v_loss": tot_v / n,
        "acc": sum(hits.values()) / n,
        "per_surface": per_surface,
    }


def train_bc(
    corpus: str,
    save: str,
    epochs: int = 4,
    minibatch: int = 192,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    value_coef: float = 0.5,
    aux_coef: float = 0.1,
    max_grad_norm: float = 5.0,
    limit: int = 0,
    dim: int = 192,
    n_layers: int = 3,
    n_heads: int = 4,
    device: str = "auto",
    value_target: str = "mc",
) -> RunPolicyNet:
    torch.manual_seed(0)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    dev = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    print(f"device: {dev}", flush=True)
    shaping = ShapingConfig()
    t0 = time.time()
    train_s, val_s, cov = load_corpus(corpus, shaping, limit=limit)
    print(
        f"featurized in {time.time() - t0:.0f}s; event-option semantic coverage: "
        f"{cov.n_covered_decisions}/{cov.n_event_decisions} decisions = "
        f"{cov.decision_coverage:.2f} "
        f"({cov.n_covered_options}/{cov.n_options} options) "
        f"[plan §3.2 target ≥ 0.70]",
        flush=True,
    )

    net = RunPolicyNet(dim=dim, n_layers=n_layers, n_heads=n_heads).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    rng = np.random.default_rng(0)
    order = np.arange(len(train_s))
    best_val = float("inf")
    for ep in range(epochs):
        rng.shuffle(order)
        te0 = time.time()
        ep_loss = ep_acc = 0.0
        nb = 0
        for s0 in range(0, len(order), minibatch):
            mb = [train_s[i] for i in order[s0 : s0 + minibatch]]
            loss, _, _, correct = _losses(net, mb, value_coef, aux_coef, dev, value_target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()
            ep_loss += float(loss.item())
            ep_acc += float(correct.mean().item())
            nb += 1
        sched.step()
        ev = evaluate(net, val_s, dev, value_target=value_target)
        print(
            f"epoch {ep}: train loss={ep_loss / max(1, nb):.4f} "
            f"acc={ep_acc / max(1, nb):.3f} | val pi={ev['pi_loss']:.4f} "
            f"v={ev['v_loss']:.4f} acc={ev['acc']:.3f} "
            f"({time.time() - te0:.0f}s)",
            flush=True,
        )
        # Early-stop checkpointing: `save` is always the BEST-val-CE epoch
        # (the 2000-run corpus overfits from ~epoch 6; last-epoch weights
        # sample measurably worse). Last epoch goes to `<save>.last.pt`.
        if ev["pi_loss"] < best_val:
            best_val = ev["pi_loss"]
            torch.save({"state_dict": net.state_dict(), "bc_epoch": ep}, save)
        torch.save({"state_dict": net.state_dict(), "bc_epoch": ep}, f"{save}.last.pt")

    ck = torch.load(save, map_location=dev, weights_only=True)
    net.load_state_dict(ck["state_dict"])
    print(f"restored best-val epoch {ck.get('bc_epoch')} (val pi={best_val:.4f})")
    ev = evaluate(net, val_s, dev, value_target=value_target)
    print("\nSL-sanity gate (val top-1 acc vs uniform baseline, per surface):")
    for k, m in ev["per_surface"].items():
        verdict = "PASS" if m["acc"] > m["uniform"] else "FAIL"
        print(f"  {k:14s} acc={m['acc']:.3f}  uniform={m['uniform']:.3f}  n={m['n']:6d}  {verdict}")
    print(f"saved {save}", flush=True)
    return net


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0, help="cap on runs loaded (smoke)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--value-target",
        choices=("mc", "search"),
        default="mc",
        help="value regression target: realized shaped return-to-go (mc) or the "
        "search teacher's root value where recorded (search; GH #452 — rows "
        "without one fall back to mc)",
    )
    args = ap.parse_args()
    train_bc(
        corpus=args.corpus,
        save=args.save,
        epochs=args.epochs,
        minibatch=args.minibatch,
        lr=args.lr,
        limit=args.limit,
        device=args.device,
        value_target=args.value_target,
    )


if __name__ == "__main__":
    _cli()
