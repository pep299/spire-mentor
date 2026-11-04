"""Run-level policy/value transformer (fork design, all-surface schema).

Every token (obs and action) shares one additive schema:
  token = section_embed + Σ present categorical embeds + Linear(sinusoid(cont))
with PAD rows (``padding_idx``) contributing zero. Obs tokens (state, deck,
relics, potions, map nodes) are concatenated with action tokens (one per
candidate), run through a small pre-norm/SwiGLU transformer, and the
action-token slice is scored by a single shared linear head (variable-size
action sets). A mean-pooled value head and a per-path-token dest-room
auxiliary head share the trunk.

Salvaged from branch ``runmodel-lightspeed`` (``run_ppo/model.py``) and
extended for the M-P2 all-surface encoder (plan
2026-07-03-ppo-fullrun-plan.md §3.2): new action sections (relic / potion /
event / fixed), a wider categorical schema (event id, fixed-action kind,
option index, decision surface), and payload/price continuous features.
Featurization lives in ``ppo.encoder``; the torch-free schema constants and
``Featurized`` live in ``ppo.schema`` (re-exported here); this module owns
``Batch``/``collate`` and the net.

Categorical/continuous schema: see ``ppo.schema``'s module docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import vocab
from .schema import (
    ACT_FLOOR_SCALE,
    ACT_SCALE,
    ASC_SCALE,
    AUX_IGNORE,
    C_ACT,
    C_ACTFLOOR,
    C_ASC,
    C_DECKSZ,
    C_DREST,
    C_DX,
    C_DY,
    C_ENABLED,
    C_FLOOR,
    C_GOLD,
    C_GOLD_D,
    C_HP,
    C_HP_D,
    C_MAXE,
    C_MINE,
    C_NREL,
    C_NUPG,
    C_PRICE,
    C_REACH,
    DECK_SCALE,
    FLOOR_SCALE,
    GOLD_SCALE,
    N_CAT,
    N_CONT,
    N_SECTION,
    SEC_CARD,
    SEC_DECK,
    SEC_EVENT_OPT,
    SEC_FIXED,
    SEC_MAPNODE,
    SEC_PATH,
    SEC_POTION,
    SEC_POTION_OPT,
    SEC_RELIC,
    SEC_RELIC_OPT,
    SEC_STATE,
    Featurized,
    empty_cat,
    pad_indices,
)

_CPU = torch.device("cpu")


@dataclass
class Batch:
    cat: torch.Tensor  # [B, T, N_CAT]
    cont: torch.Tensor  # [B, T, N_CONT]
    sec: torch.Tensor  # [B, T]
    pad_mask: torch.Tensor  # [B, T] True = pad
    obs_len: torch.Tensor  # [B] number of obs tokens
    act_mask: torch.Tensor  # [B, A] True = valid action
    aux_room: torch.Tensor  # [B, A]
    device: torch.device = _CPU


def collate(items: list[Featurized], device: torch.device | None = None) -> Batch:
    device = device or torch.device("cpu")
    B = len(items)
    obs_lens = [f.obs_cat.shape[0] for f in items]
    act_lens = [f.n_actions for f in items]
    Tmax = max(o + a for o, a in zip(obs_lens, act_lens))
    Amax = max(act_lens)
    pad = empty_cat()

    cat = np.tile(np.array(pad, dtype=np.int64), (B, Tmax, 1))
    cont = np.zeros((B, Tmax, N_CONT), dtype=np.float32)
    sec = np.zeros((B, Tmax), dtype=np.int64)
    pad_mask = np.ones((B, Tmax), dtype=bool)
    act_mask = np.zeros((B, Amax), dtype=bool)
    aux = np.full((B, Amax), AUX_IGNORE, dtype=np.int64)

    for i, f in enumerate(items):
        o, a = obs_lens[i], act_lens[i]
        cat[i, :o] = f.obs_cat
        cont[i, :o] = f.obs_cont
        sec[i, :o] = f.obs_sec
        cat[i, o : o + a] = f.act_cat
        cont[i, o : o + a] = f.act_cont
        sec[i, o : o + a] = f.act_sec
        pad_mask[i, : o + a] = False
        act_mask[i, :a] = True
        aux[i, :a] = f.aux_room

    return Batch(
        cat=torch.from_numpy(cat).to(device),
        cont=torch.from_numpy(cont).to(device),
        sec=torch.from_numpy(sec).to(device),
        pad_mask=torch.from_numpy(pad_mask).to(device),
        obs_len=torch.tensor(obs_lens, dtype=torch.long, device=device),
        act_mask=torch.from_numpy(act_mask).to(device),
        aux_room=torch.from_numpy(aux).to(device),
        device=device,
    )


class SinusoidalProj(nn.Module):
    """Per-scalar sinusoidal code (repr-lab: smooth, shared, load-bearing)
    then a linear mix."""

    def __init__(self, n_cont: int, dim: int, n_freq: int = 6) -> None:
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freq).float()  # 1,2,4,...
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(n_cont * n_freq * 2, dim)

    def forward(self, cont: torch.Tensor) -> torch.Tensor:
        # cont [B,T,C] → [B,T,C,F] → sin/cos → [B,T, C*F*2]
        x = cont.unsqueeze(-1) * self.freqs  # [B,T,C,F]
        codes = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        codes = codes.flatten(start_dim=2)
        return self.proj(codes)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mult: int = 4) -> None:
        super().__init__()
        self.n1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.RMSNorm(dim)
        hidden = dim * mult
        self.w1 = nn.Linear(dim, hidden)
        self.v = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + a
        h = self.n2(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.v(h))
        return x


class RunPolicyNet(nn.Module):
    def __init__(self, dim: int = 192, n_layers: int = 3, n_heads: int = 4) -> None:
        super().__init__()
        c = vocab.cards().size + 1
        r = vocab.relics().size + 1
        p = vocab.potions().size + 1
        rm = vocab.N_ROOM + 1
        ug = vocab.MAX_UPGRADE + 2
        ev = vocab.events().size + 1
        fx = vocab.N_FIXED + 1
        oi = vocab.MAX_OPT_IDX + 1
        sf = vocab.N_SURFACE + 1
        bo = vocab.bosses().size + 1
        do = vocab.N_DECK_OP + 1
        en = vocab.enchantments().size + 1
        ct = vocab.N_CARD_TYPE + 1
        cr = vocab.N_CARD_RARITY + 1
        self.emb_card = nn.Embedding(c, dim, padding_idx=c - 1)
        self.emb_upg = nn.Embedding(ug, dim, padding_idx=ug - 1)
        self.emb_relic = nn.Embedding(r, dim, padding_idx=r - 1)
        self.emb_potion = nn.Embedding(p, dim, padding_idx=p - 1)
        self.emb_room = nn.Embedding(rm, dim, padding_idx=rm - 1)
        self.emb_event = nn.Embedding(ev, dim, padding_idx=ev - 1)
        self.emb_fixed = nn.Embedding(fx, dim, padding_idx=fx - 1)
        self.emb_optidx = nn.Embedding(oi, dim, padding_idx=oi - 1)
        self.emb_surface = nn.Embedding(sf, dim, padding_idx=sf - 1)
        self.emb_boss = nn.Embedding(bo, dim, padding_idx=bo - 1)
        self.emb_deckop = nn.Embedding(do, dim, padding_idx=do - 1)
        self.emb_ench = nn.Embedding(en, dim, padding_idx=en - 1)
        self.emb_ctype = nn.Embedding(ct, dim, padding_idx=ct - 1)
        self.emb_rarity = nn.Embedding(cr, dim, padding_idx=cr - 1)
        self.emb_sec = nn.Embedding(N_SECTION, dim)
        self.cont_proj = SinusoidalProj(N_CONT, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(dim)
        self.policy_head = nn.Linear(dim, 1)
        self.value_norm = nn.RMSNorm(dim)
        self.value_head = nn.Linear(dim, 1)
        self.aux_head = nn.Linear(dim, vocab.N_ROOM)
        nn.init.uniform_(self.policy_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.policy_head.bias)
        # Zero-init the boss table so that adding this input is EXACTLY a
        # no-op on a warm start. `load_compat` keeps the net's own init for
        # keys the checkpoint lacks, and every other embedding is
        # N(0,1)-initialized -- so a randomly-initialized boss row would add
        # a large random vector to the state token of every single decision
        # and throw away the parent's policy on iteration 0. Zeros make the
        # warm-started net bit-identical to its parent until the first
        # gradient step, which is the property the champion playbook's
        # "no free lunch, no free damage" warm-start rule needs. Gradients
        # still flow (the rows are additive inputs, not a product), so the
        # table trains from zero exactly like an appended continuous slot.
        nn.init.zeros_(self.emb_boss.weight)
        # Same warm-start argument as `emb_boss` directly above: a randomly
        # initialized new table would add a large random vector to every card
        # token on iteration 0 and throw away the parent's policy. Zeros make
        # the warm-started net bit-identical to its parent until the first
        # gradient step, and gradients still flow because these are additive
        # inputs (GH #545).
        nn.init.zeros_(self.emb_deckop.weight)
        nn.init.zeros_(self.emb_ench.weight)
        # `card_kind` (GH #977): same warm-start argument.
        nn.init.zeros_(self.emb_ctype.weight)
        nn.init.zeros_(self.emb_rarity.weight)

    def _embed(self, b: Batch) -> torch.Tensor:
        cat = b.cat
        x = (
            self.emb_card(cat[..., 0])
            + self.emb_upg(cat[..., 1])
            + self.emb_relic(cat[..., 2])
            + self.emb_potion(cat[..., 3])
            + self.emb_room(cat[..., 4])
            + self.emb_event(cat[..., 5])
            + self.emb_fixed(cat[..., 6])
            + self.emb_optidx(cat[..., 7])
            + self.emb_surface(cat[..., 8])
            + self.emb_boss(cat[..., 9])
            + self.emb_deckop(cat[..., 10])
            + self.emb_ench(cat[..., 11])
            + self.emb_ctype(cat[..., 12])
            + self.emb_rarity(cat[..., 13])
            + self.emb_sec(b.sec)
            + self.cont_proj(b.cont)
        )
        return x

    def forward(self, b: Batch):
        x = self._embed(b)
        for blk in self.blocks:
            x = blk(x, b.pad_mask)
        x = self.norm(x)
        _, T, _ = x.shape
        A = b.act_mask.shape[1]
        # Action tokens occupy positions [obs_len, obs_len+A) per row.
        ar = torch.arange(A, device=b.device).unsqueeze(0)  # [1,A]
        act_pos = b.obs_len.unsqueeze(1) + ar  # [B,A]
        act_x = torch.gather(
            x, 1, act_pos.unsqueeze(-1).expand(-1, -1, x.size(-1)).clamp(max=T - 1)
        )  # [B,A,dim]
        logits = self.policy_head(act_x).squeeze(-1)  # [B,A]
        logits = logits.masked_fill(~b.act_mask, float("-inf"))
        # Value: mean-pool non-pad tokens.
        valid = (~b.pad_mask).unsqueeze(-1).float()
        pooled = (self.value_norm(x) * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        value = self.value_head(pooled).squeeze(-1)  # [B]
        aux_logits = self.aux_head(act_x)  # [B,A,N_ROOM]
        return logits, value, aux_logits


#: Embedding tables whose height is a CONTENT-DERIVED vocabulary, laid out as
#: ``[id_0 .. id_{n-1}, OOV, PAD]``. Growing one of these is not a plain
#: resize: the OOV and PAD rows live at the END, so they have to move while
#: the id rows stay put. See :func:`_widen_padded_embedding`.
_PADDED_VOCAB_EMBEDDINGS = (
    "emb_card.weight",
    "emb_relic.weight",
    "emb_potion.weight",
    "emb_event.weight",
    "emb_boss.weight",
    "emb_ench.weight",
)


def _widen_padded_embedding(have: torch.Tensor, want_rows: int) -> torch.Tensor:
    """Re-lay a ``[n_ids + 2, dim]`` embedding table out at a taller height.

    The vocabularies these tables index are **append-only** (Rust
    ``sts2_runai::ppo::vocab``'s pin files), so id row *i* means the same id
    before and after the growth and copies straight across. What moves is the
    tail: the OOV row and the ``padding_idx`` row sit at ``n_ids`` and
    ``n_ids + 1``, and both indices shift when ``n_ids`` grows.

    Rows for the newly-added ids are seeded from the OLD **OOV** row, not
    zeros. That is what the checkpoint actually learned to answer for "an id I
    have never seen", which is precisely what a new id is to it — and it keeps
    the widened net behaviourally identical on every id it was trained on.

    Ported for GH #544 (the Underdocks port grew the boss vocab 9 -> 12).
    Before this, the generic prefix-copy in :func:`load_compat` put the old
    OOV row at a real id's index and left the new PAD row zeroed, so a
    "widened" champion answered from the wrong row.
    """
    have_rows, dim = have.shape
    old_n_ids = have_rows - 2
    new_n_ids = want_rows - 2
    if new_n_ids < old_n_ids:
        raise ValueError(
            f"embedding shrank {have_rows} -> {want_rows} rows; the vocabs are "
            "append-only, so this means the pin was reordered or truncated, "
            "which silently repoints learned rows (GH #544)"
        )
    out = torch.empty(want_rows, dim, dtype=have.dtype, device=have.device)
    out[:old_n_ids] = have[:old_n_ids]
    out[old_n_ids:new_n_ids] = have[old_n_ids]  # new ids <- old OOV row
    out[new_n_ids] = have[old_n_ids]  # OOV
    out[new_n_ids + 1] = have[old_n_ids + 1]  # PAD
    return out


def load_compat(net: RunPolicyNet, state_dict: dict) -> RunPolicyNet:
    """Load `state_dict` into `net`, tolerating continuous-feature dim growth
    and content-vocabulary growth.

    When a checkpoint predates an appended continuous slot (e.g. an
    N_CONT=18 checkpoint loaded into an N_CONT=19 net after adding the
    ascension input), `cont_proj.proj.weight` is [dim, 216] vs [dim, 228].
    The sinusoidal features for the new slot occupy the *trailing* columns, so
    copying the overlapping block and zeroing the remainder reproduces the old
    model exactly (the new input contributes nothing until finetuned). Same-
    shape tensors load verbatim; this is a no-op for a matching checkpoint.

    Embedding tables over a content-derived vocabulary get the tail-aware
    treatment in :func:`_widen_padded_embedding` instead of the prefix copy,
    because their OOV and PAD rows move when the vocabulary grows.

    Whole tensors the checkpoint lacks — e.g. ``emb_boss.weight`` on any net
    predating the boss-identity input — keep the net's own initialization
    (``strict=False`` semantics). That is only warm-start-safe because
    :class:`RunPolicyNet` zero-initializes that table; see the comment there.
    """
    tgt = net.state_dict()
    merged = {}
    for k, want in tgt.items():
        have = state_dict.get(k)
        if have is None:
            continue  # keep net's init (strict=False semantics)
        # Checkpoints are always written on CPU; `net` may already live on a
        # GPU (trainer `--device`), so align before any cross-tensor write.
        have = have.to(want.device)
        if have.shape == want.shape:
            merged[k] = have
        elif k in _PADDED_VOCAB_EMBEDDINGS and have.ndim == 2:
            merged[k] = _widen_padded_embedding(have, want.shape[0])
        else:
            slot = want.clone().zero_()
            region = tuple(slice(0, min(a, b)) for a, b in zip(want.shape, have.shape))
            slot[region] = have[region]
            merged[k] = slot
    net.load_state_dict(merged, strict=False)
    return net
