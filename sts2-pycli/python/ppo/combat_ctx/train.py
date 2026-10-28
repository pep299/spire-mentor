"""Train every student of the combat context probe on one cache (GH #830 step 0, PREREG §4).

One loader, one target, one rare-id rule and one checkpoint rule for all arms:

* `S_ctx`, `S_ctx_tok`, `S_ctxarch_rl2in`: `CombatCtxNet` over the arm's
  tokens (`tokens.derive` of the cached `S_ctx` tokens). AdamW (wd 0.01),
  dropout 0.1, batch 512, cosine schedule with one warmup epoch, bf16
  autocast, an auxiliary value MSE on the fight's final search value at
  λ_v = 0.1 (decisions of a step-capped fight get none).
* `S_rl2arch`: rl2's own architecture (`rollout_learn._playout_net`, v1)
  over rl2's 32 + 24 features and its action key, initialised AT the
  heuristic (lin[21] = 1) like `rollout_learn train`. Normal mode: TRAIN key
  vocabulary under the rare rule (row 0 = UNK), 5% key dropout, Adam (wd
  1e-4), batch 1024. `--rl2-mode compat` reproduces the unchanged trainer
  (TRAIN keys, unseen -> row 0, no dropout, its own 95% fight subset, AdamW
  lr 3e-3 wd 1e-4, batch 4096) for the §3 cross-check. Its best checkpoint is
  also written as an rl2 weights file (`rl2.json`; the final epoch as
  `rl2_last.json`, which the §3 cross-check compares), which the Rust
  `LearnedPlayout` loads.

Every arm: the loss is cross-entropy to the class target (primary
`0.8·onehot(rec class) + 0.2·visit share`, or `--target secondary`), each
content id is replaced by UNK with p = 0.05 in training, and the checkpoint
is the epoch with the best DEV teacher regret over
E (evaluated every epoch). TEST is never read. The DEV picks that select the
checkpoint (and are saved as `dev_picks.npz`) come from `ctx_predict` in fp32,
the same function `predict` uses for TEST: bf16 autocast is for the training
steps only, so the frozen DEV reads are the frozen model's. `summary.json`
records the best epoch, whether it fell in the last 10% of the cap (the
convergence rule's trigger), steps to best and the final DEV slope.

The rare-id rule is fit on the decisions the student trains on: all of TRAIN,
or a learning-curve fraction / the boss-elite rooms (`--train-frac`,
`--rooms`), so an id outside them maps to the trained UNK row. `--overfit N`
(INVALID #6) trains on the first N TRAIN decisions in sha1(id) order with no
dropout, no id dropout and no weight decay, for the cap or until TRAIN top-1
(on those N, every epoch) stalls for 10 epochs.

Long runs write `metrics.jsonl` per epoch and `progress.log`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import time
from typing import Any

import numpy as np

from . import tokens as T
from .data import RareMap, Split
from .metrics import dev_summary, ids_sha

CTX_ARMS = ("S_ctx", "S_ctx_tok", "S_ctxarch_rl2in")
ARMS = (*CTX_ARMS, "S_rl2arch")
HEUR_FEATURE = 21


def derive_split(a: dict[str, np.ndarray], arm: str) -> dict[str, np.ndarray]:
    """`arm`'s token arrays for a whole cached split (vectorized `tokens.derive`)."""
    cat, cont, sec = a["cat"], a["cont"], a["sec"]
    v = T.vocab()
    if arm == "S_ctx":
        return {
            "cat": cat,
            "cont": cont,
            "sec": sec,
            "tok_start": a["tok_start"],
            "n_tok": a["n_tok"],
        }
    if arm == "S_ctx_tok":
        cat = cat.copy()
        cont = cont.copy()
        cat[:, T.S["rl2key"]] = v.tables[T.S["rl2key"]].pad
        cont[:, T.C_RL2S :] = 0
        return {
            "cat": cat,
            "cont": cont,
            "sec": sec,
            "tok_start": a["tok_start"],
            "n_tok": a["n_tok"],
        }
    if arm != "S_ctxarch_rl2in":
        raise ValueError(arm)
    n_act = a["n_act"]
    new_n = 1 + n_act
    new_start = np.concatenate([[0], np.cumsum(new_n)[:-1]])
    n_new = int(new_n.sum())
    act_src = np.repeat(a["tok_start"] + a["n_tok"] - n_act, n_act) + (
        np.arange(int(n_act.sum())) - np.repeat(a["act_start"], n_act)
    )
    act_dst = np.repeat(new_start + 1, n_act) + (
        np.arange(int(n_act.sum())) - np.repeat(a["act_start"], n_act)
    )
    cat2 = np.tile(v.pads, (n_new, 1))
    cat2[act_dst, T.S["rl2key"]] = cat[act_src, T.S["rl2key"]]
    cont2 = np.zeros((n_new, cont.shape[1]), cont.dtype)
    cont2[new_start, T.C_RL2S : T.C_RL2S + T.RL2_STATE_DIM] = cont[
        a["tok_start"], T.C_RL2S : T.C_RL2S + T.RL2_STATE_DIM
    ]
    cont2[act_dst, T.C_RL2X :] = cont[act_src, T.C_RL2X :]
    sec2 = np.full(n_new, T.SEC["ACT_RL2"], np.int8)
    sec2[new_start] = T.SEC["STATE"]
    return {"cat": cat2, "cont": cont2, "sec": sec2, "tok_start": new_start, "n_tok": new_n}


class Log:
    def __init__(self, out: str) -> None:
        self.path = os.path.join(out, "progress.log")
        self.t0 = time.time()

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} [{time.time() - self.t0:7.0f}s] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class Device:
    """A split's arrays on the training device, with the batch assembly."""

    def __init__(
        self, torch: Any, sp: Split, arm: str, rare: RareMap, dev: Any, target: str
    ) -> None:
        self.torch, self.sp, self.arm, self.dev = torch, sp, arm, dev
        a = sp.a
        tt = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)  # noqa: E731
        self.n_act = tt(a["n_act"])
        self.act_start = tt(a["act_start"])
        self.target = tt(a["t1" if target == "primary" else "t2"].astype(np.float32))
        z = a["z_value"].astype(np.float32)
        self.z = tt(np.nan_to_num(z))
        self.z_ok = tt(np.isfinite(z))
        if arm in CTX_ARMS:
            d = derive_split(a, arm)
            self.cat = tt(rare.apply(d["cat"]).astype(np.int16))
            self.cont = tt(d["cont"])
            self.sec = tt(d["sec"])
            self.tok_start = tt(d["tok_start"])
            self.n_tok = tt(d["n_tok"])
            v = T.vocab()
            self.content = [T.S[s] for s in T.CONTENT_SLOTS]
            self.unk = tt(v.unks.astype(np.int64))
            self.pad = tt(v.pads.astype(np.int64))
        else:
            self.S = tt(a["rl2_state"])
            self.X = tt(a["rl2_x"])

    def ctx_batch(self, idx: Any, id_dropout: float = 0.0) -> tuple:
        torch = self.torch
        nt = self.n_tok[idx]
        tm = int(nt.max())
        ar = torch.arange(tm, device=self.dev)
        pad = ar[None, :] >= nt[:, None]
        pos = torch.where(pad, 0, self.tok_start[idx][:, None] + ar[None, :])
        cat = self.cat[pos].long()
        if id_dropout > 0:
            for s in self.content:
                c = cat[..., s]
                drop = (torch.rand(c.shape, device=self.dev) < id_dropout) & (c != self.pad[s])
                cat[..., s] = torch.where(drop, self.unk[s], c)
        cont = self.cont[pos]
        sec = self.sec[pos].long()
        na = self.n_act[idx]
        am = int(na.max())
        ara = torch.arange(am, device=self.dev)
        amask = ara[None, :] < na[:, None]
        act_pos = torch.where(amask, (nt - na)[:, None] + ara[None, :], 0)
        return cat, cont, sec, pad, act_pos, amask

    def act_index(self, idx: Any) -> tuple:
        torch = self.torch
        na = self.n_act[idx]
        am = int(na.max())
        ara = torch.arange(am, device=self.dev)
        amask = ara[None, :] < na[:, None]
        flat = torch.where(amask, self.act_start[idx][:, None] + ara[None, :], 0)
        return flat, amask


def ctx_predict(
    torch: Any, net: Any, D: Device, n: int, rows: np.ndarray | None = None
) -> np.ndarray:
    """Global-action argmax of `net` on the first `n` decisions of `D` (or on
    the decisions `rows`, the result aligned with them), in fp32 (no
    autocast): the one prediction path for DEV selection, the frozen DEV reads
    and TEST."""
    was_training = net.training
    net.eval()
    idx = torch.arange(n, device=D.dev) if rows is None else torch.as_tensor(rows, device=D.dev)
    out = np.empty(len(idx), np.int64)
    with torch.no_grad(), torch.autocast(D.dev.type, enabled=False):
        for k0, b in zip(range(0, len(idx), 2048), torch.split(idx, 2048), strict=True):
            cat, cont, sec, pad, act_pos, amask = D.ctx_batch(b)
            logits, _ = net(cat, cont, sec, pad, act_pos, amask)
            flat, _ = D.act_index(b)
            pick = flat.gather(1, logits.argmax(-1)[:, None])[:, 0].cpu().numpy()
            out[k0 : k0 + len(pick)] = pick
    net.train(was_training)
    return out


def rl2_key_rows(
    train: Split, mode: str, min_runs: int = 3, keep: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    """Universe rl2key row -> `_playout_net` row (0 = UNK), and the key names.

    normal: keys in >= `min_runs` TRAIN runs (the rare rule); compat: every
    TRAIN key (`rollout_learn train`'s vocabulary). `keep` (per TRAIN
    decision) restricts the count to the decisions the student trains on, as
    `RareMap.fit` does."""
    tab = T.vocab().tables[T.S["rl2key"]]
    keys = train.a["rl2_key"].astype(np.int64)
    run = train.run[train.act_dec]
    if keep is not None:
        act_keep = np.asarray(keep, bool)[train.act_dec]
        keys, run = keys[act_keep], run[act_keep]
    pairs = np.unique(keys * (len(train.runs) + 1) + run)
    c = np.bincount(pairs // (len(train.runs) + 1), minlength=tab.size)
    keep = np.flatnonzero(
        (c >= (min_runs if mode == "normal" else 1)) & (np.arange(tab.size) < tab.unk)
    )
    m = np.zeros(tab.size, np.int64)
    m[keep] = np.arange(1, len(keep) + 1)
    return m, ["<unk>"] + [tab.names[i] for i in keep]


def compat_val_fights(train: Split, seed: int, val_frac: float = 0.05) -> set[int]:
    """The unchanged trainer's validation fights (`rollout_learn.train`'s code)."""
    fights = np.unique(train.a["fight_idx"])
    rng = np.random.default_rng(seed)
    return set(rng.choice(fights, max(1, int(len(fights) * val_frac)), replace=False).tolist())


def json_safe(x: Any) -> Any:
    """`x` with every non-finite float as None (strict JSON for the Rust loader)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {k: v for k, v in vars(args).items() if k != "func"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    os.makedirs(args.out, exist_ok=True)
    log = Log(args.out)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    train = Split(args.cache, "train")
    dsp = Split(args.cache, "dev")
    keep = np.ones(train.n, bool)
    if args.train_frac < 1.0:
        from ppo.deepbets.value_data import frac_key

        keep &= np.asarray([frac_key(f) < args.train_frac for f in train.meta["files"]])
    if args.rooms == "boss_elite":
        keep &= train.a["room"] >= 1
    # The student's TRAIN (a learning-curve fraction, the boss/elite-only
    # retrain): the rare-id rule counts runs over it (PREREG §4), so an id
    # it never trains on maps to the trained UNK row. The compat twin's key
    # vocabulary is every TRAIN key (the unchanged trainer's), and the
    # overfit probe keeps the frozen model's encoding.
    fit_keep = None if keep.all() else keep.copy()
    if args.arm == "S_rl2arch" and args.rl2_mode == "compat":
        val = compat_val_fights(train, args.seed)
        keep &= ~np.isin(train.a["fight_idx"], list(val))
    if args.overfit:
        order = np.argsort([hashlib.sha1(i.encode()).hexdigest() for i in train.meta["ids"]])
        sel = np.zeros(train.n, bool)
        sel[order[: args.overfit]] = True
        keep &= sel
    rare = RareMap.fit(train, keep=fit_keep)
    tr_idx_np = np.flatnonzero(keep)
    log(
        f"arm {args.arm}: {len(tr_idx_np)}/{train.n} TRAIN decisions, {dsp.n} DEV "
        f"({int(dsp.a['in_E'].sum())} in E), device {dev}"
    )
    D = Device(torch, train, args.arm, rare, dev, args.target)
    DV = Device(torch, dsp, args.arm, rare, dev, args.target)
    # The overfit probe (PREREG §4, INVALID #6) drops every regulariser: no
    # dropout, no id dropout, no weight decay (and no early stopping: the
    # loop runs to the cap or until TRAIN top-1 stalls for 10 epochs).
    no_drop = bool(args.overfit)
    wd = 0.0 if no_drop else args.wd
    if args.arm in CTX_ARMS:
        from .model import CombatCtxNet, n_params

        net = CombatCtxNet(
            T.vocab().sizes,
            T.N_CONT,
            T.N_SECTION,
            args.dim,
            args.layers,
            args.heads,
            0.0 if no_drop else args.dropout,
        ).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=wd)
        batch = args.batch or 512
        id_drop = 0.0 if no_drop else args.id_dropout
    else:
        from ppo.rollout_learn import _playout_net

        key_map, key_names = rl2_key_rows(
            train, args.rl2_mode, keep=None if args.rl2_mode == "compat" else fit_keep
        )
        key_map_t = torch.as_tensor(key_map, device=dev)
        D.K = key_map_t[torch.as_tensor(train.a["rl2_key"].astype(np.int64), device=dev)]
        DV.K = key_map_t[torch.as_tensor(dsp.a["rl2_key"].astype(np.int64), device=dev)]
        net = _playout_net(torch, args.hidden, args.emb, len(key_names)).to(dev)
        for p in (net.emb.weight, net.bias.weight, net.proj.weight):
            torch.nn.init.normal_(p, std=0.01)
        with torch.no_grad():
            net.lin.weight.zero_()
            net.lin.weight[0, HEUR_FEATURE] = 1.0
        if args.rl2_mode == "compat":
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=wd)
            batch = args.batch or 4096
            id_drop = 0.0
        else:
            opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=wd)
            batch = args.batch or 1024
            id_drop = 0.0 if no_drop else args.id_dropout
        from .model import n_params
    regularisation = {
        "dropout": 0.0 if no_drop or args.arm not in CTX_ARMS else args.dropout,
        "id_dropout": id_drop,
        "weight_decay": wd,
    }
    log(
        f"model {n_params(net):,} parameters, batch {batch}, lr {args.lr}, wd {wd}, "
        f"regularisation {regularisation}"
    )
    tr_idx = torch.as_tensor(tr_idx_np, device=dev)
    steps_per_epoch = max(1, math.ceil(len(tr_idx) / batch))
    total = steps_per_epoch * args.epochs
    warm = (
        steps_per_epoch * args.warmup_epochs
        if args.arm in CTX_ARMS and args.schedule == "cosine"
        else 0
    )

    def lr_at(step: int) -> float:
        if args.schedule != "cosine":
            return args.lr
        if step < warm:
            return args.lr * (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    amp = dev.type == "cuda" and args.arm in CTX_ARMS

    def forward(Dx: Device, idx: Any, train_mode: bool) -> tuple:
        if args.arm in CTX_ARMS:
            cat, cont, sec, pad, act_pos, amask = Dx.ctx_batch(idx, id_drop if train_mode else 0.0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, value = net(cat, cont, sec, pad, act_pos, amask)
            flat, _ = Dx.act_index(idx)
            return logits, value, flat, amask
        flat, amask = Dx.act_index(idx)
        X = Dx.X[flat] * amask[..., None]
        K = Dx.K[flat] * amask
        if train_mode and id_drop > 0:
            K = torch.where(torch.rand(K.shape, device=dev) < id_drop, 0, K)
        logits = net(Dx.S[idx], X, K, amask)
        return logits, None, flat, amask

    def predict(Dx: Device, n: int, rows: np.ndarray | None = None) -> np.ndarray:
        """Picks on the first `n` decisions, or on `rows` (aligned with them)."""
        if args.arm in CTX_ARMS:
            return ctx_predict(torch, net, Dx, n, rows)
        net.eval()
        idx = torch.arange(n, device=dev) if rows is None else torch.as_tensor(rows, device=dev)
        out = np.empty(len(idx), np.int64)
        with torch.no_grad():
            for k0, b in zip(range(0, len(idx), 2048), torch.split(idx, 2048), strict=True):
                logits, _, flat, amask = forward(Dx, b, False)
                am = logits.argmax(-1)
                pick = flat.gather(1, am[:, None])[:, 0].cpu().numpy()
                out[k0 : k0 + len(pick)] = pick
        net.train()
        return out

    def train_top1() -> float:
        """Class top-1 on the probe's own TRAIN decisions (predicted on them only)."""
        has = train.a["rec_clsA"][tr_idx_np] >= 0
        p = predict(D, train.n, tr_idx_np)
        return float((train.a["cls"][p][has] == train.a["rec_clsA"][tr_idx_np][has]).mean())

    best: dict[str, Any] = {"R": float("inf"), "epoch": -1}
    history = []
    step = 0
    t_start = time.time()
    stale = 0
    best_tr = -1.0
    for epoch in range(1, args.epochs + 1):
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        tot = vtot = 0.0
        nb = 0
        for b in torch.split(perm, batch):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            logits, value, flat, amask = forward(D, b, True)
            tgt = D.target[flat] * amask
            lp = torch.log_softmax(logits.masked_fill(~amask, -1e9), -1)
            loss = -(tgt * lp).sum(-1).mean()
            if value is not None and args.lambda_v > 0:
                ok = D.z_ok[b]
                if ok.any():
                    vl = ((value[ok] - D.z[b][ok]) ** 2).mean()
                    vtot += float(vl.detach())
                    loss = loss + args.lambda_v * vl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()
            tot += float(loss.detach())
            nb += 1
            step += 1
        pick = predict(DV, dsp.n)
        ds = dev_summary(dsp, pick)
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": tot / nb,
            "value_mse": vtot / nb,
            "lr": lr_at(step),
            "wall_s": round(time.time() - t_start, 1),
            "dev": ds,
        }
        if args.overfit:
            row["train_top1"] = train_top1()
        history.append(row)
        with open(os.path.join(args.out, "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(row)) + "\n")
        log(
            f"epoch {epoch}/{args.epochs}: loss {tot / nb:.4f} vmse {vtot / nb:.4f} DEV R {ds['R']:.5f} "
            f"Agr {ds['Agr']:.4f} top1 {ds['top1']:.4f}"
            + (f" TRAIN top1 {row['train_top1']:.4f}" if args.overfit else "")
        )
        if (
            ds["R"] < best["R"]
            or (math.isnan(best["R"]) and not math.isnan(ds["R"]))
            or best["epoch"] < 0
        ):
            best = {"R": ds["R"], "epoch": epoch, "step": step, "dev": ds}
            save(args, net, rare, best, "best", torch)
            np.savez(os.path.join(args.out, "dev_picks.npz"), pick=pick, ids_sha=ids_sha(dsp))
            if args.arm == "S_rl2arch":
                with open(os.path.join(args.out, "rl2.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        net.to_json(
                            key_names, 1.0, json_safe({"combat_ctx": _settings(args), "best": best})
                        ),
                        f,
                    )
        if args.overfit:
            if row["train_top1"] > best_tr + 1e-4:
                best_tr, stale = row["train_top1"], 0
            else:
                stale += 1
            if stale >= 10:
                log("overfit probe: TRAIN top-1 stopped rising for 10 epochs")
                break
        if args.max_minutes and time.time() - t_start > 60 * args.max_minutes:
            log(f"time budget {args.max_minutes} min reached")
            break
    save(args, net, rare, best, "last", torch)
    if args.arm == "S_rl2arch":
        # The final epoch, unselected: what the unchanged trainer writes (§3).
        with open(os.path.join(args.out, "rl2_last.json"), "w", encoding="utf-8") as f:
            json.dump(net.to_json(key_names, 1.0, json_safe({"combat_ctx": _settings(args)})), f)
    last_epochs = len(history)
    rs = [h["dev"]["R"] for h in history]
    tail = rs[-max(2, len(rs) // 10) :]
    summary = {
        "arm": args.arm,
        "best_epoch": best["epoch"],
        "epochs_run": last_epochs,
        "cap": args.epochs,
        "best_in_last_10pct": best["epoch"] > 0.9 * args.epochs,
        "steps_to_best": best.get("step"),
        "final_slope": float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
        if len(tail) >= 2
        else None,
        "dev_best": best.get("dev"),
        "train_top1": history[-1].get("train_top1"),
        "train_top1_max": max(
            (h["train_top1"] for h in history if h.get("train_top1") is not None), default=None
        ),
        "train_decisions": len(tr_idx_np),
        "rare_fit_decisions": int(train.n if fit_keep is None else fit_keep.sum()),
        "regularisation": regularisation,
        "settings": _settings(args),
        "unk_share_dev": rare.report(dsp),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=1)
    log(f"done: best DEV R {best['R']:.5f} at epoch {best['epoch']}")
    return summary


def save(
    args: argparse.Namespace, net: Any, rare: RareMap, best: dict, tag: str, torch: Any
) -> None:
    ck = {
        "arm": args.arm,
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "config": getattr(net, "config", {"hidden": args.hidden, "emb": args.emb}),
        "rare": rare.to_json(),
        "best": best,
        "settings": _settings(args),
        "vocab": T.vocab().digest(),
        "n_cont": T.N_CONT,
    }
    tmp = os.path.join(args.out, f"{tag}.pt.tmp")
    torch.save(ck, tmp)
    pathlib.Path(tmp).replace(os.path.join(args.out, f"{tag}.pt"))


def cmd_train(args: argparse.Namespace) -> int:
    if args.clip is None:
        args.clip = 0.0 if args.arm == "S_rl2arch" else 1.0
    if args.arm == "S_rl2arch":
        if args.rl2_mode == "compat":
            args.lr = 3e-3 if args.lr is None else args.lr
            args.wd = 1e-4 if args.wd is None else args.wd
            args.epochs = args.epochs or 20
            args.schedule = "const"
        else:
            args.lr = 3e-3 if args.lr is None else args.lr
            args.wd = 1e-4 if args.wd is None else args.wd
            args.epochs = args.epochs or 150
            args.schedule = args.schedule or "const"
    else:
        args.lr = 3e-4 if args.lr is None else args.lr
        args.wd = 0.01 if args.wd is None else args.wd
        args.epochs = args.epochs or 40
        args.schedule = args.schedule or "cosine"
    run(args)
    return 0


def add_parser(sub: Any) -> None:
    t = sub.add_parser("train", help="train one student arm (DEV-selected checkpoint)")
    t.add_argument("--cache", required=True)
    t.add_argument("--arm", choices=ARMS, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--target", choices=("primary", "secondary"), default="primary")
    t.add_argument("--dim", type=int, default=192)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--hidden", type=int, default=64, help="S_rl2arch hidden")
    t.add_argument("--emb", type=int, default=32, help="S_rl2arch embedding")
    t.add_argument("--rl2-mode", choices=("normal", "compat"), default="normal")
    t.add_argument("--lr", type=float)
    t.add_argument("--wd", type=float)
    t.add_argument("--batch", type=int, default=0)
    t.add_argument("--epochs", type=int, default=0)
    t.add_argument("--warmup-epochs", type=int, default=1)
    t.add_argument("--schedule", choices=("cosine", "const"))
    t.add_argument("--lambda-v", type=float, default=0.1)
    t.add_argument("--id-dropout", type=float, default=0.05)
    t.add_argument(
        "--clip",
        type=float,
        help="grad-norm clip (default 1.0 for the transformers, none for S_rl2arch)",
    )
    t.add_argument(
        "--train-frac", type=float, default=1.0, help="learning curve: runs with frac_key < f"
    )
    t.add_argument("--rooms", choices=("all", "boss_elite"), default="all")
    t.add_argument("--overfit", type=int, default=0, help="overfit probe on N TRAIN decisions")
    t.add_argument("--max-minutes", type=float, default=0)
    t.set_defaults(func=cmd_train)


# ── the control cross-check (PREREG §3) ──────────────────────────────────


def rl2_json_picks(json_path: str, sp: Split) -> np.ndarray:
    """Global-action argmax of an rl2-format weights file on a cached split,
    through the torch twin (`rollout_learn._load_playout_net`)."""
    import torch

    from ppo.rollout_learn import _load_playout_net

    with open(json_path, encoding="utf-8") as f:
        net_json = json.load(f)
    net = _load_playout_net(torch, net_json).eval()
    rows = {k: i for i, k in enumerate(net_json["keys"])}
    names = T.vocab().tables[T.S["rl2key"]].names
    key_row = np.asarray(
        [rows.get(names[k], 0) if k < len(names) else 0 for k in sp.a["rl2_key"].astype(np.int64)]
    )
    a = sp.a
    out = np.empty(sp.n, np.int64)
    with torch.no_grad():
        for d0 in range(0, sp.n, 4096):
            idx = np.arange(d0, min(sp.n, d0 + 4096))
            na = a["n_act"][idx]
            am = int(na.max())
            flat = a["act_start"][idx][:, None] + np.arange(am)[None, :]
            mask = np.arange(am)[None, :] < na[:, None]
            flat = np.where(mask, flat, 0)
            X = torch.as_tensor(a["rl2_x"][flat] * mask[..., None])
            K = torch.as_tensor(key_row[flat] * mask)
            S = torch.as_tensor(a["rl2_state"][idx])
            z = net(S, X, K, torch.as_tensor(mask)).numpy()
            out[idx] = flat[np.arange(len(idx)), z.argmax(-1)]
    return out


def class_top1(sp: Split, pick: np.ndarray) -> float:
    has = sp.a["rec_clsA"] >= 0
    return float((sp.a["cls"][pick][has] == sp.a["rec_clsA"][has]).mean())


def cmd_crosscheck(args: argparse.Namespace) -> int:
    """DEV class-level top-1 of the unchanged trainer's JSON vs the compat twin's."""
    dsp = Split(args.cache, "dev")
    a = class_top1(dsp, rl2_json_picks(args.json, dsp))
    b = class_top1(dsp, rl2_json_picks(args.twin, dsp))
    res = {"unchanged": a, "compat_twin": b, "diff_pp": 100 * (a - b), "pass": abs(a - b) <= 0.015}
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0


def add_crosscheck_parser(sub: Any) -> None:
    c = sub.add_parser(
        "crosscheck", help="§3: unchanged rollout_learn train vs the compat twin on DEV"
    )
    c.add_argument("--cache", required=True)
    c.add_argument("--json", required=True, help="the unchanged trainer's weights file")
    c.add_argument(
        "--twin",
        required=True,
        help="the compat twin's rl2_last.json (its final epoch, like the unchanged)",
    )
    c.add_argument("--out")
    c.set_defaults(func=cmd_crosscheck)


# ── predictions ──────────────────────────────────────────────────────────


def ctx_picks(ckpt_path: str, sp: Split, device: str = "cpu") -> np.ndarray:
    """Global-action argmax of a `CombatCtxNet` checkpoint on a cached split."""
    import torch

    from .model import CombatCtxNet

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ck["vocab"] != T.vocab().digest() or ck["n_cont"] != T.N_CONT:
        raise SystemExit(f"{ckpt_path}: token schema differs from this checkout's")
    dev = torch.device(device)
    net = CombatCtxNet.from_config(ck["config"]).to(dev)
    net.load_state_dict(ck["state_dict"])
    D = Device(torch, sp, ck["arm"], RareMap.from_json(ck["rare"]), dev, "primary")
    return ctx_predict(torch, net, D, sp.n)


def ctx_unk(ckpt_path: str, sp: Split) -> np.ndarray:
    """Per decision: does the checkpoint's arm see an UNK (rare or OOV) id in
    it, under the checkpoint's own rare map (PREREG §5.7 zero-UNK subset)."""
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    d = derive_split(sp.a, ck["arm"])
    cat = RareMap.from_json(ck["rare"]).apply(d["cat"])
    v = T.vocab()
    tok = np.zeros(len(cat), bool)
    for slot in T.CONTENT_SLOTS:
        s = T.S[slot]
        tok |= cat[:, s] == v.tables[s].unk
    dec = np.repeat(np.arange(sp.n), d["n_tok"])
    return np.bincount(dec, weights=tok, minlength=sp.n) > 0


def rl2_json_unk(json_path: str, sp: Split) -> np.ndarray:
    """Per decision: does an rl2-format net map one of its action keys to
    row 0 (UNK)."""
    with open(json_path, encoding="utf-8") as f:
        keys = json.load(f)["keys"]
    rows = {k: i for i, k in enumerate(keys)}
    names = T.vocab().tables[T.S["rl2key"]].names
    row = np.asarray(
        [rows.get(names[k], 0) if k < len(names) else 0 for k in sp.a["rl2_key"].astype(np.int64)]
    )
    return np.bincount(sp.act_dec, weights=row == 0, minlength=sp.n) > 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Picks of one frozen-or-not checkpoint on a split (TEST: `--final` + FROZEN).

    The pick file carries the checkpoint's sha256, the arm and training seed
    it was trained as (so the sealed read can check its label) and a
    per-decision UNK flag."""
    from .metrics import check_frozen, ckpt_identity, sha256

    sp = Split(args.cache, args.split)
    if args.split == "test":
        if not args.final:
            raise SystemExit("TEST is sealed: --final with a FROZEN manifest")
        check_frozen(args.final_dir, [args.ckpt])
    try:
        ident = ckpt_identity(args.ckpt)
    except SystemExit:
        if args.split == "test":
            raise
        ident = {"arm": "", "seed": -1}
    if args.ckpt.endswith(".json"):
        pick = rl2_json_picks(args.ckpt, sp)
        unk = rl2_json_unk(args.ckpt, sp)
    else:
        pick = ctx_picks(args.ckpt, sp, args.device)
        unk = ctx_unk(args.ckpt, sp)
    np.savez(args.out, pick=pick, ids_sha=ids_sha(sp), ckpt_sha256=sha256(args.ckpt),
             arm=ident["arm"], seed=ident["seed"], has_unk=unk)  # fmt: skip
    who = f"{args.ckpt} ({ident['arm']} seed {ident['seed']})"
    print(f"{sp.n} picks of {who} on {args.split} -> {args.out}")
    return 0


def add_predict_parser(sub: Any) -> None:
    p = sub.add_parser(
        "predict", help="argmax picks of a checkpoint (best.pt or rl2.json) on a split"
    )
    p.add_argument("--cache", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="dev")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--final", action="store_true")
    p.add_argument("--final-dir")
    p.set_defaults(func=cmd_predict)
