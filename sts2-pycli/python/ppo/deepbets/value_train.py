"""G-D value net: `python -m ppo.deepbets.value_train {train,sweep}` (GH #830).

`ReturnValueNet` (rvnet.v1, value-net-first section 3) regresses the deployed
leaf playout's expected `search_value` (RichBoth units) from a position's
honest view:

* inputs: the v2 StepCtx vector (96), optionally the 216 TexelWide inputs
  (variant B), DeepSets sums of d=16 card embeddings over hand / draw /
  discard / exhaust and of potion embeddings (order-free, so the draw order
  cannot leak), five monster slots (species embedding + hp fraction, block,
  alive), the afterstate flag (scored as `belief_endturn`) and a boss flag;
* trunk `in -> H -> H/2` (ReLU); heads `p = sigmoid(z0)`,
  `mu_w = 1.0 + 0.5 sigmoid(z1)`, `mu_l = -1.45 + 0.4 sigmoid(z2)`;
  `V = p mu_w + (1 - p) mu_l` stays inside the RichBoth bands by construction.

Loss (boss : elite weighted 1 : 1): BCE(p, n_won / n) + MSE(mu_w, mean_win)
weighted by the win share + MSE(mu_l, mean_loss) weighted by the loss share +
MSE(V, mean) + `--sib-beta` x the sibling-centred MSE, `(V_c - mean_g V)`
against `(y_c - mean_g y)` over each group's members (batches are packed by
group). AdamW, cosine LR, the final epoch is the model.

The net file (JSON) carries the weights, the vocabularies, the input
normalisation and `meta.playout_net`, the playout net its labels were played
by; `value_neq` refuses a net whose playout net differs from its references'.

    sweep  {A, B} x hidden {64, 128, 256} x 2 seeds on the GPUs, each scored on
           probe-dev (`value_neq` cross-fitted dev n_eq), then the 25 / 50 /
           100 % learning curve of the best config.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ppo.deepbets import value_data as vd

D_CARD = 16
D_MON = 16
N_ZONES = 5  # hand, draw, discard, exhaust, potions
SCHEMA = "deepbets.rvnet.v1"


# ── data ─────────────────────────────────────────────────────────────────


@dataclass
class LabelData:
    """Every labelled state, columnar; tokens as CSR rows of (zone, token, count)."""

    ids: np.ndarray
    split: np.ndarray
    room: np.ndarray
    kind: np.ndarray
    after: np.ndarray
    group: np.ndarray
    fight: np.ndarray
    step: np.ndarray
    wide: np.ndarray
    texel: np.ndarray
    mon_sp: np.ndarray
    mon_f: np.ndarray
    row_off: np.ndarray
    f_zone: np.ndarray
    f_tok: np.ndarray
    f_cnt: np.ndarray
    n: np.ndarray
    n_won: np.ndarray
    mean: np.ndarray
    var: np.ndarray
    mean_win: np.ndarray
    mean_loss: np.ndarray
    vocab: list[str]
    species: list[str]


def _read_shards(label_dir: str) -> list[dict[str, np.ndarray]]:
    out = []
    for p in vd.shard_paths(label_dir, "label-", ".npz"):
        z = np.load(p)
        out.append({k: z[k] for k in z.files})
    if not out:
        raise SystemExit(f"no label shards in {label_dir}")
    return out


_CACHE: dict[tuple[str, int], LabelData] = {}


def load_label_data(
    label_dir: str, vocab: list[str] | None = None, species: list[str] | None = None
) -> LabelData:
    """Load every shard into one table, tokens on a global vocabulary.

    With `vocab` / `species` None they are built from the TRAIN split (row 0
    padding, row 1 unknown), so dev and test tokens never shape them. Tables
    are cached per process by (directory, shard files, vocabularies): a sweep
    scores many nets trained on the same labels.
    """
    key = (
        os.path.abspath(label_dir),
        hash(
            (
                tuple(vd.shard_paths(label_dir, "label-", ".npz")),
                tuple(vocab or ()),
                tuple(species or ()),
            )
        ),
    )
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = _load_label_data(label_dir, vocab, species)
    return _CACHE[key]


def _load_label_data(
    label_dir: str, vocab: list[str] | None, species: list[str] | None
) -> LabelData:
    shards = _read_shards(label_dir)
    train = vd.SPLIT_NAMES.index("train")
    if vocab is None:
        seen: collections.Counter[str] = collections.Counter()
        sp_seen: collections.Counter[str] = collections.Counter()
        for z in shards:
            tr = z["split"][z["tok_row"]] == train
            seen.update(z["vocab"][z["tok"][tr]].tolist())
            sp = z["mon_sp"][z["split"] == train].ravel()
            sp_seen.update(z["mvocab"][sp[sp > 0]].tolist())
        vocab = ["<pad>", "<unk>", *sorted(seen)]
        species = ["<none>", "<unk>", *sorted(sp_seen)]
    assert species is not None
    tok_id = {t: i for i, t in enumerate(vocab)}
    sp_id = {t: i for i, t in enumerate(species)}
    parts: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    base = 0
    for z in shards:
        n = len(z["id"])
        remap = np.array([tok_id.get(str(t), 1) for t in z["vocab"]], np.int32)
        sremap = np.array([0] + [sp_id.get(str(t), 1) for t in z["mvocab"][1:]], np.int32)
        key = (z["tok_row"].astype(np.int64) * N_ZONES + z["tok_zone"]) * (len(vocab) + 1) + remap[
            z["tok"]
        ]
        uk, cnt = np.unique(key, return_counts=True)
        tok = (uk % (len(vocab) + 1)).astype(np.int32)
        rz = uk // (len(vocab) + 1)
        parts["f_tok"].append(tok)
        parts["f_zone"].append((rz % N_ZONES).astype(np.int8))
        parts["f_row"].append((rz // N_ZONES + base).astype(np.int64))
        parts["f_cnt"].append(cnt.astype(np.float32))
        parts["mon_sp"].append(sremap[z["mon_sp"]])
        for k in ("id", "split", "room", "kind", "after", "group", "fight", "step", "wide", "texel",
                  "mon_f", "n", "n_won", "mean", "var", "mean_win", "mean_loss"):  # fmt: skip
            parts[k].append(z[k])
        base += n
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.argsort(cat["f_row"], kind="stable")
    rows = cat["f_row"][order]
    row_off = np.searchsorted(rows, np.arange(base + 1)).astype(np.int64)
    return LabelData(
        ids=cat["id"],
        split=cat["split"],
        room=cat["room"],
        kind=cat["kind"],
        after=cat["after"],
        group=cat["group"],
        fight=cat["fight"],
        step=cat["step"].astype(np.float32),
        wide=cat["wide"].astype(np.float32),
        texel=cat["texel"],
        mon_sp=cat["mon_sp"],
        mon_f=cat["mon_f"].astype(np.float32),
        row_off=row_off,
        f_zone=cat["f_zone"][order],
        f_tok=cat["f_tok"][order],
        f_cnt=cat["f_cnt"][order],
        n=cat["n"],
        n_won=cat["n_won"],
        mean=cat["mean"],
        var=cat["var"],
        mean_win=cat["mean_win"],
        mean_loss=cat["mean_loss"],
        vocab=list(vocab),
        species=list(species),
    )


def label_column(label_dir: str, ids: list[str], col: str) -> dict[str, float]:
    """`{id: value}` of one scalar label-shard column."""
    want = set(ids)
    out = {}
    for p in vd.shard_paths(label_dir, "label-", ".npz"):
        z = np.load(p)
        for i, v in zip(z["id"], z[col], strict=True):
            if str(i) in want:
                out[str(i)] = float(v)
    missing = want - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} reference members have no label row")
    return out


# ── model ────────────────────────────────────────────────────────────────


def build_model(torch: Any, n_tok: int, n_sp: int, hidden: int, wide: bool) -> Any:
    nn = torch.nn

    class ReturnValueNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wide = wide
            self.hidden = hidden
            self.card = nn.Embedding(n_tok, D_CARD, padding_idx=0)
            self.mon = nn.Embedding(n_sp, D_MON, padding_idx=0)
            self.in_dim = (
                vd.STEP_DIM + (vd.WIDE_DIM if wide else 0) + N_ZONES * D_CARD
                + vd.MONSTER_SLOTS * (D_MON + vd.MON_FEATS) + 2
            )  # fmt: skip
            self.l1 = nn.Linear(self.in_dim, hidden)
            self.l2 = nn.Linear(hidden, hidden // 2)
            self.head = nn.Linear(hidden // 2, 3)

        def forward(
            self, dense: Any, seg: Any, tok: Any, cnt: Any, n_rows: int, mon_sp: Any, mon_f: Any
        ) -> tuple[Any, Any, Any, Any]:
            cards = torch.zeros(n_rows * N_ZONES, D_CARD, device=dense.device, dtype=dense.dtype)
            cards.index_add_(0, seg, self.card(tok) * cnt[:, None])
            mons = torch.cat([self.mon(mon_sp), mon_f], -1).reshape(n_rows, -1)
            x = torch.cat([dense, cards.reshape(n_rows, -1), mons], -1)
            h = torch.relu(self.l2(torch.relu(self.l1(x))))
            z = self.head(h)
            p = torch.sigmoid(z[:, 0])
            mu_w = 1.0 + 0.5 * torch.sigmoid(z[:, 1])
            mu_l = -1.45 + 0.4 * torch.sigmoid(z[:, 2])
            return p, mu_w, mu_l, p * mu_w + (1.0 - p) * mu_l

    return ReturnValueNet()


class Batcher:
    """The label table on a device, sliced into model inputs by row index."""

    def __init__(
        self, torch: Any, d: LabelData, norm: dict[str, np.ndarray], wide: bool, device: str
    ) -> None:
        self.torch = torch
        t = lambda a, dt=None: torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=dt)  # noqa: E731
        step = (d.step - norm["step_mean"]) / norm["step_std"]
        cols = [step]
        if wide:
            cols.append((d.wide - norm["wide_mean"]) / norm["wide_std"])
        cols.append(d.after.astype(np.float32)[:, None])
        cols.append((d.room == vd.ROOMS.index("boss")).astype(np.float32)[:, None])
        self.dense = t(np.concatenate(cols, 1).astype(np.float32))
        self.row_off = t(d.row_off, torch.int64)
        self.f_zone = t(d.f_zone.astype(np.int64))
        self.f_tok = t(d.f_tok.astype(np.int64))
        self.f_cnt = t(d.f_cnt)
        self.mon_sp = t(d.mon_sp.astype(np.int64))
        self.mon_f = t(d.mon_f)
        self.device = device

    def inputs(self, rows: Any) -> tuple[Any, ...]:
        torch = self.torch
        starts = self.row_off[rows]
        lens = self.row_off[rows + 1] - starts
        b = torch.arange(len(rows), device=self.device)
        seg_row = torch.repeat_interleave(b, lens)
        first = torch.repeat_interleave(torch.cumsum(lens, 0) - lens, lens)
        pos = (
            torch.arange(int(lens.sum()), device=self.device)
            - first
            + torch.repeat_interleave(starts, lens)
        )
        seg = seg_row * N_ZONES + self.f_zone[pos]
        return (
            self.dense[rows],
            seg,
            self.f_tok[pos],
            self.f_cnt[pos],
            len(rows),
            self.mon_sp[rows],
            self.mon_f[rows],
        )


def norm_stats(d: LabelData, rows: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "step_mean": d.step[rows].mean(0),
        "step_std": np.maximum(d.step[rows].std(0), 1e-3),
        "wide_mean": d.wide[rows].mean(0),
        "wide_std": np.maximum(d.wide[rows].std(0), 1e-3),
    }


# ── training ─────────────────────────────────────────────────────────────


def group_index(groups: np.ndarray) -> np.ndarray:
    """`[N]` int group ids, -1 for a state in no group."""
    out = np.full(len(groups), -1, np.int64)
    ids: dict[str, int] = {}
    for i, g in enumerate(groups):
        if g:
            out[i] = ids.setdefault(str(g), len(ids))
    return out


def packed_batches(
    rows: np.ndarray, gidx: np.ndarray, batch: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Shuffled batches of `rows` that keep each group's members together."""
    units: dict[int, list[int]] = collections.defaultdict(list)
    solo = []
    for r in rows:
        g = gidx[r]
        if g < 0:
            solo.append([int(r)])
        else:
            units[int(g)].append(int(r))
    all_units = solo + list(units.values())
    order = rng.permutation(len(all_units))
    out, cur = [], []
    for u in order:
        cur.extend(all_units[u])
        if len(cur) >= batch:
            out.append(np.array(cur, np.int64))
            cur = []
    if cur:
        out.append(np.array(cur, np.int64))
    return out


def loss_terms(
    torch: Any, out: tuple[Any, ...], y: dict[str, Any], gid: Any, w: Any
) -> dict[str, Any]:
    """Per-term weighted means for one batch (see the module docstring)."""
    p, mu_w, mu_l, v = out
    eps = 1e-6
    pw = y["n_won"] / y["n"]
    ws = w.sum()
    bce = -(w * (pw * torch.log(p + eps) + (1 - pw) * torch.log(1 - p + eps))).sum() / ws
    has_w = torch.nan_to_num(y["mean_win"], nan=0.0)
    has_l = torch.nan_to_num(y["mean_loss"], nan=0.0)
    ww = w * pw
    wl = w * (1 - pw)
    mse_w = (ww * (mu_w - has_w) ** 2).sum() / ww.sum().clamp_min(eps)
    mse_l = (wl * (mu_l - has_l) ** 2).sum() / wl.sum().clamp_min(eps)
    mse_v = (w * (v - y["mean"]) ** 2).sum() / ws
    sib = torch.zeros((), device=v.device)
    m = gid >= 0
    if bool(m.any()):
        g = gid[m]
        uniq, inv = torch.unique(g, return_inverse=True)
        cnt = torch.zeros(len(uniq), device=v.device).index_add_(0, inv, torch.ones_like(v[m]))
        vm = torch.zeros(len(uniq), device=v.device).index_add_(0, inv, v[m]) / cnt
        ym = torch.zeros(len(uniq), device=v.device).index_add_(0, inv, y["mean"][m]) / cnt
        keep = cnt[inv] >= 2
        if bool(keep.any()):
            dv = (v[m] - vm[inv])[keep]
            dy = (y["mean"][m] - ym[inv])[keep]
            wg = w[m][keep] / cnt[inv][keep]
            sib = (wg * (dv - dy) ** 2).sum() / wg.sum()
    return {"bce": bce, "mse_w": mse_w, "mse_l": mse_l, "mse_v": mse_v, "sib": sib}


def room_weights(d: LabelData, rows: np.ndarray) -> np.ndarray:
    """Boss : elite 1 : 1, mean weight 1 over `rows`."""
    w = np.ones(len(d.ids), np.float32)
    rooms = d.room[rows]
    present = [r for r in np.unique(rooms)]
    for r in present:
        n_r = (rooms == r).sum()
        w[rows[rooms == r]] = len(rows) / (len(present) * n_r)
    return w


def train_rows(d: LabelData, frac: float) -> np.ndarray:
    train = vd.SPLIT_NAMES.index("train")
    keep = d.split == train
    if frac < 1.0:
        fk = np.array([vd.frac_key(str(f)) < frac for f in d.fight])
        keep &= fk
    return np.where(keep)[0]


def cmd_train(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    log = vd.Progress(args.out + ".log", f"train {os.path.basename(args.out)}")
    lmeta = json.load(open(os.path.join(args.labels, "manifest.json"), encoding="utf-8"))
    t0 = time.time()
    d = load_label_data(args.labels)
    rows = train_rows(d, args.frac)
    dev = np.where(d.split == vd.SPLIT_NAMES.index("dev"))[0]
    if not len(rows):
        raise SystemExit("no training rows")
    norm = norm_stats(d, rows)
    dev_t = args.device
    bt = Batcher(torch, d, norm, args.variant == "B", dev_t)
    model = build_model(torch, len(d.vocab), len(d.species), args.hidden, args.variant == "B").to(
        dev_t
    )
    w_all = torch.as_tensor(room_weights(d, rows), device=dev_t)
    gidx = group_index(d.group)
    gid_t = torch.as_tensor(gidx, device=dev_t)
    ys = {
        k: torch.as_tensor(getattr(d, k).astype(np.float32), device=dev_t)
        for k in ("n", "n_won", "mean", "mean_win", "mean_loss", "var")
    }
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    batches0 = packed_batches(rows, gidx, args.batch, rng)
    total_steps = max(1, len(batches0) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
    log(f"{len(rows)} train rows ({args.frac:.0%} of fights), {len(dev)} dev rows, vocab {len(d.vocab)}, "
        f"species {len(d.species)}, variant {args.variant}, hidden {args.hidden}, device {dev_t}, "
        f"load {time.time() - t0:.0f}s")  # fmt: skip
    metrics_path = args.out + ".metrics.jsonl"
    with open(metrics_path, "w", encoding="utf-8"):
        pass
    for ep in range(args.epochs):
        model.train()
        sums: collections.Counter[str] = collections.Counter()
        batches = batches0 if ep == 0 else packed_batches(rows, gidx, args.batch, rng)
        for b in batches:
            bi = torch.as_tensor(b, device=dev_t)
            out = model(*bt.inputs(bi))
            y = {k: v[bi] for k, v in ys.items()}
            terms = loss_terms(torch, out, y, gid_t[bi], w_all[bi])
            loss = (
                terms["bce"]
                + terms["mse_w"]
                + terms["mse_l"]
                + terms["mse_v"]
                + args.sib_beta * terms["sib"]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            for k, v in terms.items():
                sums[k] += float(v)
            sums["n"] += 1
        row = {"epoch": ep + 1, **{f"train_{k}": sums[k] / sums["n"] for k in sums if k != "n"}}
        if len(dev):
            row |= {
                f"dev_{k}": v for k, v in eval_rows(torch, model, bt, ys, gid_t, dev, dev_t).items()
            }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if (ep + 1) % max(1, args.log_every) == 0 or ep + 1 == args.epochs:
            log(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}))
    meta = {
        "playout_net": lmeta["playout_net"],
        "engine": lmeta.get("engine"),
        "labels": os.path.abspath(args.labels),
        "label_k": lmeta.get("k"),
        "train": vars(args) | {"rows": len(rows), "secs": round(time.time() - t0, 1)},
    }
    save_net(args.out, model, d.vocab, d.species, norm, args.variant, args.hidden, meta)
    log(f"saved {args.out} ({time.time() - t0:.0f}s)")
    return 0


def eval_rows(
    torch: Any, model: Any, bt: Batcher, ys: dict[str, Any], gid: Any, rows: np.ndarray, dev_t: str
) -> dict[str, float]:
    """Dev label metrics: noise-corrected MSE of V, sibling-centred MSE, BCE."""
    model.eval()
    with torch.no_grad():
        outs = []
        for i in range(0, len(rows), 16384):
            bi = torch.as_tensor(rows[i : i + 16384], device=dev_t)
            outs.append(model(*bt.inputs(bi)))
        cat = [torch.cat([o[j] for o in outs]) for j in range(4)]
        bi = torch.as_tensor(rows, device=dev_t)
        y = {k: v[bi] for k, v in ys.items()}
        terms = loss_terms(torch, tuple(cat), y, gid[bi], torch.ones(len(rows), device=dev_t))
        noise = float((y["var"] / y["n"]).mean())
    return {k: float(v) for k, v in terms.items()} | {
        "label_noise": noise,
        "mse_v_corrected": float(terms["mse_v"]) - noise,
    }


# ── net file ─────────────────────────────────────────────────────────────


def save_net(
    path: str, model: Any, vocab: list[str], species: list[str], norm: dict[str, np.ndarray],
    variant: str, hidden: int, meta: dict[str, Any],
) -> None:  # fmt: skip
    import torch

    w = {
        k: v.detach().cpu().numpy().astype(float).round(7).tolist()
        for k, v in model.state_dict().items()
    }
    obj = {
        "schema": SCHEMA,
        "variant": variant,
        "hidden": hidden,
        "d_card": D_CARD,
        "d_mon": D_MON,
        "vocab": vocab,
        "species": species,
        "norm": {k: v.astype(float).round(7).tolist() for k, v in norm.items()},
        "weights": w,
        "params": int(sum(p.numel() for p in model.parameters())),
        "meta": meta,
    }
    vd.write_json(path, obj)
    torch.save(model.state_dict(), path + ".pt")


def load_net(path: str) -> dict[str, Any]:
    """`{"model", "norm", "vocab", "species", "variant", "meta", ...}` on the CPU."""
    import torch

    obj = json.load(open(path, encoding="utf-8"))
    if obj.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema {obj.get('schema')!r}, expected {SCHEMA!r}")
    model = build_model(
        torch, len(obj["vocab"]), len(obj["species"]), obj["hidden"], obj["variant"] == "B"
    )
    model.load_state_dict(
        {k: torch.as_tensor(np.asarray(v, np.float32)) for k, v in obj["weights"].items()}
    )
    model.eval()
    obj["model"] = model
    obj["norm"] = {k: np.asarray(v, np.float32) for k, v in obj["norm"].items()}
    return obj


def predict(
    net: dict[str, Any], label_dir: str, ids: list[str], device: str = "cpu"
) -> dict[str, float]:
    """`{id: V}` for labelled states `ids` (the net's vocabularies; unseen tokens -> unk)."""
    import torch

    d = load_label_data(label_dir, net["vocab"], net["species"])
    pos = {str(i): j for j, i in enumerate(d.ids)}
    missing = [i for i in ids if i not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} ids have no label row")
    rows = np.array([pos[i] for i in ids], np.int64)
    model = net["model"].to(device)
    bt = Batcher(torch, d, net["norm"], net["variant"] == "B", device)
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 16384):
            bi = torch.as_tensor(rows[i : i + 16384], device=device)
            out.append(model(*bt.inputs(bi))[3].cpu().numpy())
    v = np.concatenate(out) if out else np.zeros(0)
    return {i: float(x) for i, x in zip(ids, v, strict=True)}


# ── sweep ────────────────────────────────────────────────────────────────


def _train_cmd(
    args: argparse.Namespace, out: str, variant: str, hidden: int, seed: int, frac: float
) -> list[str]:
    return [
        sys.executable, "-m", "ppo.deepbets.value_train", "train", "--labels", args.labels,
        "--out", out, "--variant", variant, "--hidden", str(hidden), "--seed", str(seed),
        "--frac", str(frac), "--epochs", str(args.epochs), "--batch", str(args.batch),
        "--lr", str(args.lr), "--sib-beta", str(args.sib_beta), "--device", "cuda:0",
    ]  # fmt: skip


def _run_queues(
    jobs: list[tuple[str, list[str]]], gpus: list[str], log: vd.Progress
) -> dict[str, int]:
    """Run `(name, argv)` jobs, one at a time per GPU; returns exit codes."""
    queue = list(jobs)
    running: dict[str, tuple[str, subprocess.Popen]] = {}
    codes: dict[str, int] = {}
    env0 = {k: v for k, v in os.environ.items()}
    while queue or running:
        for g in gpus:
            if g not in running and queue:
                name, argv = queue.pop(0)
                env = env0 | {
                    "CUDA_VISIBLE_DEVICES": g,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
                running[g] = (
                    name,
                    subprocess.Popen(
                        argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                    ),
                )
                log(f"start {name} on GPU {g}")
        time.sleep(2)
        for g in list(running):
            name, proc = running[g]
            if proc.poll() is not None:
                codes[name] = proc.returncode
                log(f"done {name} rc={proc.returncode}")
                del running[g]
    return codes


def dev_score(report: dict[str, Any], est: str = "V") -> float:
    """Selection score: the weakest contested stratum's weaker n_eq (dev, cross-fitted)."""
    vals = []
    for st in ("contested_boss", "contested_elite"):
        s = report["strata"].get(st)
        if s is None:
            continue
        e = s["estimators"][est]
        vals.append(e["neq_cal"] if s["uninformative"] else min(e["neq_cal"], e["neq_regret_M"]))
    return min(vals) if vals else float("nan")


def cmd_sweep(args: argparse.Namespace) -> int:
    os.makedirs(args.out, exist_ok=True)
    log = vd.Progress(os.path.join(args.out, "progress.log"), "sweep")
    gpus = args.gpus.split(",")
    configs = [(v, h, s) for v in args.variants for h in args.hiddens for s in args.seeds]
    jobs = []
    for v, h, s in configs:
        name = f"{v}_h{h}_s{s}"
        out = os.path.join(args.out, name + ".json")
        if not os.path.exists(out):
            jobs.append((name, _train_cmd(args, out, v, h, s, 1.0)))
    log(f"{len(configs)} configs, {len(jobs)} to train on GPUs {gpus}")
    codes = _run_queues(jobs, gpus, log)
    if any(codes.values()):
        log(f"FAILED trainings: {codes}")
    results = {}
    for v, h, s in configs:
        name = f"{v}_h{h}_s{s}"
        rep = _dev_eval(args, os.path.join(args.out, name + ".json"), log)
        results[name] = {
            "variant": v,
            "hidden": h,
            "seed": s,
            "dev_score": dev_score(rep),
            "dev": _brief(rep),
        }
        log(f"{name}: dev score {results[name]['dev_score']:.2f}")
    best = max(results, key=lambda k: (np.nan_to_num(results[k]["dev_score"], nan=-1.0), k))
    bv, bh, bs = results[best]["variant"], results[best]["hidden"], results[best]["seed"]
    log(f"best {best}; learning curve {args.fracs}")
    lc_jobs = []
    for frac in args.fracs:
        if frac >= 1.0:
            continue
        name = f"lc_{best}_f{int(frac * 100)}"
        out = os.path.join(args.out, name + ".json")
        if not os.path.exists(out):
            lc_jobs.append((name, _train_cmd(args, out, bv, bh, bs, frac)))
    _run_queues(lc_jobs, gpus, log)
    curve = {}
    for frac in args.fracs:
        name = best if frac >= 1.0 else f"lc_{best}_f{int(frac * 100)}"
        rep = _dev_eval(args, os.path.join(args.out, name + ".json"), log)
        curve[str(frac)] = {"net": name, "dev_score": dev_score(rep), "dev": _brief(rep)}
    s50 = curve.get("0.5", {}).get("dev_score")
    s100 = curve.get("1.0", {}).get("dev_score")
    gain = (s100 - s50) / s50 if s50 and s100 is not None and s50 > 0 else None
    summary = {
        "best": best,
        "results": results,
        "learning_curve": curve,
        "gain_50_to_100": gain,
        "retry_rule_data_limited": gain is not None and gain >= 0.15,
    }
    vd.write_json(os.path.join(args.out, "sweep.json"), summary)
    log(f"sweep done: best {best} dev score {results[best]['dev_score']:.2f}; 50->100% gain {gain}")
    return 0


def _dev_eval(args: argparse.Namespace, net: str, log: vd.Progress) -> dict[str, Any]:
    from ppo.deepbets import value_neq

    out = net[: -len(".json")] + ".dev.json"
    if not os.path.exists(out):
        value_neq.main(["eval", "--net", net, "--labels", args.labels, "--refs-dev", args.refs_dev,
                        "--out", out, "--n-boot", str(args.n_boot)])  # fmt: skip
    return json.load(open(out, encoding="utf-8"))


def _brief(rep: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for st, s in rep["strata"].items():
        e = s["estimators"].get("V", {})
        out[st] = {k: e.get(k) for k in ("neq_cal", "neq_regret_M", "neq_regret_V", "lambda")} | {
            "uninformative": s["uninformative"]
        }
    return out


# ── CLI ──────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ppo.deepbets.value_train")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def hp(p: argparse.ArgumentParser) -> None:
        p.add_argument("--labels", required=True)
        p.add_argument("--epochs", type=int, default=30)
        p.add_argument("--batch", type=int, default=4096)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--sib-beta", type=float, default=1.0)

    p = sub.add_parser("train", help="train one ReturnValueNet")
    hp(p)
    p.add_argument("--out", required=True, help="net JSON path")
    p.add_argument(
        "--variant", choices=("A", "B"), default="A", help="B adds the 216 TexelWide inputs"
    )
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frac", type=float, default=1.0, help="share of train fights (learning curve)")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-every", type=int, default=5)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("sweep", help="12-run sweep + learning curve on the GPUs")
    hp(p)
    p.add_argument("--refs-dev", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--variants", nargs="+", default=["A", "B"])
    p.add_argument("--hiddens", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--fracs", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    p.add_argument("--n-boot", type=int, default=200)
    p.set_defaults(fn=cmd_sweep)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
