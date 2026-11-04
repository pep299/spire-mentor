# PPO champion — `ppo_fullrun_ironclad_cont300_it1150`

**Missed its pre-registered A10 bar (p 0.021 against p < 0.01) on
2026-10-31 with every cell positive; promoted by user decision.**

- **Character**: Ironclad
- **Source**: `~/ppo-ironclad-cont300/main.iter_1150.pt`, copied byte for
  byte (the exact file every gate arm loaded). The net after the final logged
  iteration `it1149` is `iter_1150`.
- **Replaces**: `ppo_fullrun_ironclad_cardmods_it850` (promoted 2026-10-27).
- **Features**: `card_mods,reward_upgrade`, from the checkpoint's own `cfg`,
  unchanged from the parent.
- **Net**: `ppo.model.RunPolicyNet` defaults (dim=192, n_layers=3, n_heads=4),
  post-#1038 layout: it carries the `card_kind` tables (`emb_ctype`,
  `emb_rarity`). With `card_kind` off they only see their PAD row and are
  exactly zero.
- **Pre-registration and record**:
  `docs/experiments/2026-10-31-ironclad-a10-cont300.md`; harness in
  `docs/experiments/ironclad-a10-cont300/`.

## The recipe

A **true resume** (`--resume-from-iter 850`, optimizer state kept) of the
parent, it850 → it1150, with the parent's own recipe and no declared changes:
A10 only, `--act1 roll`, 960 games/iter at 250/1000 sims, entropy pinned at
0.0125, 2 epochs, PLR off, no curse injection, trained with
`STS2_RUNAI_POTION_SWAP=1` and **served with it OFF**. `--iterations 1150`
re-opened the absolute-iteration LR anneal (`lr_frac` 0.482 at it850 → 0.30
at it1149).

**The engine changed under the parent.** Trained on `main` `0e83e7f6d`, which
includes #1029 (no rewards after the first A10 final-act boss) and everything
merged since the parent's wheel. PGO wheel `src_hash da1b16130b0fd30b`.

The parent's `.pt` predates the `card_kind` tables, so
`add_card_kind_tables.py` inserted them at zero and renumbered the AdamW state
by parameter name for the strict resume.

**Attribution: none.** The engine change, 300 more A10 iterations and the LR
re-open are confounded.

## Training trajectory

2026-10-30 01:05 → 2026-10-31 15:52, ~410 s/iter (up to ~775 s while another
session's sweep shared the host), 0 errored games. A host-wide OOM from other
jobs killed the trainer after it1024; it resumed from it1020 on the same
pinned wheel and Python. Training win rate (acting policy, A10, 50-iteration
blocks): 14.8, 15.6, 15.9, 15.5, 16.6, 17.1%; `ent` rose 0.284 → 0.318 and
turned over to 0.28.

## The gate

Both arms: one binary (wheel `da1b16130b0fd30b`), `net_greedy`, shipped
Ironclad search, `--act1 roll`, potion-swap OFF, torch `.pt` inference, each
arm featurized at its own checkpoint cfg. Both arms measured fresh on this
wheel. **0 errored rows in 3,600.**

| cell | champion it850 | it1150 | Δ | discordant (champ/cand) | McNemar p |
|---|---|---|---|---|---|
| **A10 × 1000, 1000/4000**, panel 135,000,000 (primary) | 21.7% | 25.4% | **+3.70pp** | 104/141 | **0.021** |
| A9 estimate from those runs | 49.7% | 52.3% | +2.60pp | 166/192 | 0.19 |
| A9 × 400, 1000/4000, panel 135,100,000 (readout) | 50.7% | 56.2% | +5.50pp | 60/82 | 0.078 |
| standard eval 4000/16000, A10 × 400, seeds 91,000,000 (readout) | 27.0% | 30.0% | +3.00pp | 60/72 | 0.34 |
| A9 estimate from those runs | 57.2% | 59.2% | +2.00pp | 67/75 | 0.56 |

Rule: PROMOTE iff Δ > 0 and p < 0.01. **Missed (p 0.021);** Δ > 0, so by the
pre-registration promotion was the user's call, and the user approved it.

The parent's own promotion numbers (28.1% A10, 30.00% standard eval) were
measured on older wheels, before #1029; they are not comparable to this table.

## Serving checks

- **ONNX** exported from this checkpoint by `ppo.export_onnx`
  (`external_data=False`, no `.onnx.data` sidecar).
- **Overflow headroom**: max final deck 65 across 1,800 candidate runs (gate +
  standard eval), against the `obs + act ≤ T_MAX` binder at deck ≈ 126.
