"""GH #545: card rewards carry the upgrade flag the engine rolled for them.

`roll_for_upgrade` pre-upgrades non-rare reward options — odds `act_index *
0.25` (halved at A7+), so 25% in act 2 and 50% in act 3 — but the PPO obs had
no channel for it and the encoder passed `False` into `cat[1]` on the
most-exercised surface in training. A free upgraded card featurized as the
base card.

Reading the flag is gated behind the `reward_upgrade` feature
(`test_ppo_screen_features.py`) because every committed champion was trained
without it. These tests pin the layer BELOW that gate: the flags reach the obs
at all, index-aligned with the ids, on both the training path and the recorded
fixture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "sts2-rs/crates/sts2-runai/tests/fixtures/ppo_encoder_parity_reward_upgrade.jsonl"
)

# Two runs of random play is plenty: the contract under test holds on EVERY
# card-pick decision, and act 1 alone yields several per run. (Reaching a
# decision where a flag is actually True needs act 2+, which random play does
# not survive to — that half is pinned by the committed fixture below.)
SEEDS = range(1000, 1002)


@pytest.fixture(autouse=True)
def _syspath(monkeypatch):
    monkeypatch.syspath_prepend(PY)


# One test per seed so xdist runs the walks in parallel; each seed reaches
# card picks on its own (seed 1000: 18, seed 1001: 9).
@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed{s}")
def test_the_training_obs_ships_flags_aligned_with_the_ids(seed: int) -> None:
    """`ppo.env.observe` must ship one flag per offered card. A short list
    would silently read as "not upgraded" for the tail options rather than
    fail, so alignment is the thing to pin."""
    import random

    from ppo.env import Decision, PpoRunEnv

    seen = 0
    env = PpoRunEnv(seed=seed, ascension=0, character="Ironclad")
    rng = random.Random(seed)
    out = env.reset()
    while isinstance(out, Decision):
        if out.surface == "card-pick" and out.obs.get("card_reward_option_ids"):
            ids = out.obs["card_reward_option_ids"]
            flags = out.obs["card_reward_option_upgraded"]
            # Bundle offerings carry no upgrade state at all, so an EMPTY
            # list is legal; a partial one is not.
            assert flags == [] or len(flags) == len(ids), (seed, ids, flags)
            assert all(isinstance(f, bool) for f in flags)
            seen += 1
        out = env.step(rng.randrange(len(out.actions)))
    assert seen, "no card-pick decision reached — the walk pinned nothing"


def test_the_recorded_fixture_holds_real_pre_upgraded_rewards() -> None:
    """The engine really does hand these out, and the flag really does reach
    the tokens — pinned on the committed rows the Rust parity gate reads
    (`ppo_encoder_parity_reward_upgrade.jsonl`, dumped at act 2+ by
    `ppo.dump_parity_fixtures` with the feature on).

    Without this the whole feature could be wired end-to-end against an
    all-`False` reality and nothing would notice.
    """
    from ppo.encoder import featurize_row

    rows = [json.loads(line) for line in _FIXTURE.read_text().splitlines() if line.strip()]
    assert rows, "feature fixture is empty"
    upgraded_rows = 0
    for row in rows:
        assert row["features"] == ["reward_upgrade"]
        # Since the champion itself declares `reward_upgrade` (2026-08-28) this
        # fixture carries the FULL surface walk, not a 3-row supplement, so most
        # rows are not card rewards at all. The invariant is a count over the
        # file, asserted below -- never a property of every row.
        ids = row["obs"].get("card_reward_option_ids") or []
        flags = row["obs"].get("card_reward_option_upgraded") or []
        if not ids:
            continue
        assert len(flags) == len(row["obs"]["card_reward_option_ids"])
        if not any(flags):
            continue
        upgraded_rows += 1
        feat = featurize_row(row["obs"], row["actions"], row["metrics"], frozenset(row["features"]))
        assert feat.act_cat.tolist() == row["feat"]["act_cat"]
        # The recorded tokens say "upgraded" exactly where the engine did.
        for action, flag in zip(row["actions"], flags, strict=False):
            assert action["action"] == "pick_card_reward"
            assert feat.act_cat[action["idx"]][1] == int(flag)
    # A COUNT, not an equality with len(rows). This fixture became the full
    # surface walk when the champion started declaring the feature, so most of
    # its rows are events, paths and rests that cannot carry an upgraded
    # reward. What still has to hold is that the ON branch is genuinely
    # exercised somewhere -- act 1 can never produce an upgraded reward
    # (`roll_for_upgrade` = act_index * 0.25 off a 0.0 base), so a walk that
    # never left act 1 would pin nothing at all.
    assert upgraded_rows >= 3, (
        f"only {upgraded_rows} fixture rows carry a pre-upgraded card reward "
        f"(need >= 3) -- the walk never reached an act-2+ reward roll, so the "
        f"gated branch is unpinned"
    )
