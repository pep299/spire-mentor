"""Gates for the plasticity diagnostic (`ppo.plasticity`).

WHY THIS EXISTS. A PPO campaign trains against a target that moves every
iteration — the policy changes, so the returns it predicts change, and so does
the state distribution. Networks degrade under that in a way that costs
TRAINABILITY rather than knowledge: units go dormant (near-zero activation for
every input, so they pass no gradient) and the representation's effective rank
collapses. That is what BBF's periodic resets exist to undo.

Two things make it worth instrumenting here rather than assuming: `--epochs`
is going 2 -> 4, and more gradient steps per unit of fresh data is exactly the
condition that erodes plasticity fastest; and champions are warm-started from
champions (asc_it180 -> deckmut_it150 -> simscale_it200), so the deployed
lineage has accumulated far more training on a moving target than one
campaign's iteration count suggests.

This measures the SYMPTOM only. It cannot be screened -- `ppo.screen` resolves
~8pp (commit a760f3dd6) -- so the diagnostic is the cheap way to learn whether
resets have anything to fix.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from ppo.plasticity import dormant_fraction, effective_rank, plasticity_stats


def test_all_zero_units_are_all_dormant():
    assert dormant_fraction(torch.zeros(32, 8)) == pytest.approx(1.0)


def test_uniformly_active_units_are_none_dormant():
    assert dormant_fraction(torch.full((32, 8), 0.7)) == pytest.approx(0.0)


def test_half_the_units_silenced_reads_one_half():
    acts = torch.full((32, 8), 0.7)
    acts[:, :4] = 0.0
    assert dormant_fraction(acts) == pytest.approx(0.5)


def test_dormancy_is_scale_invariant():
    """The score normalises by the layer mean, so scaling every unit is a no-op.

    Without that normalisation a shrinking weight norm would masquerade as
    rising dormancy, which is the exact confound this metric exists to avoid.
    """
    acts = torch.rand(64, 16)
    assert dormant_fraction(acts * 1000.0) == dormant_fraction(acts)


def test_the_threshold_is_applied_to_the_normalised_score():
    """One quiet unit among H, straddling tau — arithmetic spelled out.

    With one unit at `a` and H-1 at 1.0, the layer mean is (a + H - 1)/H, so
    the quiet unit's score is a*H/(a + H - 1) — NOT `a`. Here 0.04*4/3.04 =
    0.05263. Getting this wrong is easy, and a threshold silently compared
    against the raw activation would still pass every other test in this file.
    """
    acts = torch.full((16, 4), 1.0)
    acts[:, 0] = 0.04
    assert dormant_fraction(acts, tau=0.05) == pytest.approx(0.0)  # 0.05263 > 0.05
    assert dormant_fraction(acts, tau=0.06) == pytest.approx(0.25)  # 0.05263 < 0.06


def test_effective_rank_of_a_rank_one_matrix_is_one():
    u = torch.randn(64, 1)
    v = torch.randn(1, 8)
    assert effective_rank(u @ v) == 1


def test_effective_rank_of_an_isotropic_matrix_is_full():
    assert effective_rank(torch.eye(8).repeat(8, 1)) == 8


def test_effective_rank_falls_when_the_representation_collapses():
    full = torch.randn(128, 16)
    collapsed = full.clone()
    collapsed[:, 4:] = 0.0  # only 4 directions carry signal
    assert effective_rank(collapsed) < effective_rank(full)


def _tiny_batch(net_dim_batch: int = 8):
    import numpy as np
    from ppo.model import Batch
    from ppo.schema import N_CONT, empty_cat

    B, T, A = net_dim_batch, 24, 4
    cat = torch.from_numpy(np.tile(np.array(empty_cat(), dtype=np.int64), (B, T, 1)))
    return Batch(
        cat=cat,
        cont=torch.zeros(B, T, N_CONT),
        sec=torch.zeros(B, T, dtype=torch.long),
        pad_mask=torch.zeros(B, T, dtype=torch.bool),
        obs_len=torch.full((B,), T - A, dtype=torch.long),
        act_mask=torch.ones(B, A, dtype=torch.bool),
        aux_room=torch.zeros(B, A, dtype=torch.long),
    )


def test_plasticity_stats_reports_every_block_and_the_trunk_rank():
    from ppo.model import RunPolicyNet

    net = RunPolicyNet(dim=32, n_layers=3, n_heads=4).eval()
    stats = plasticity_stats(net, _tiny_batch())
    assert set(stats) >= {
        "dormant_frac",
        "dormant_frac/block0",
        "dormant_frac/block1",
        "dormant_frac/block2",
        "effective_rank",
        "effective_rank_frac",
    }
    assert 0.0 <= stats["dormant_frac"] <= 1.0
    assert 1 <= stats["effective_rank"] <= 32
    assert stats["effective_rank_frac"] == pytest.approx(stats["effective_rank"] / 32)


def test_plasticity_stats_leaves_no_hooks_behind():
    """The probe installs forward hooks; leaking them would tax every later
    training step and silently change what the net computes."""
    from ppo.model import RunPolicyNet

    net = RunPolicyNet(dim=32, n_layers=3, n_heads=4).eval()
    before = [len(b.w2._forward_pre_hooks) for b in net.blocks]
    plasticity_stats(net, _tiny_batch())
    assert [len(b.w2._forward_pre_hooks) for b in net.blocks] == before


def test_plasticity_stats_does_not_build_a_graph_or_touch_gradients():
    from ppo.model import RunPolicyNet

    net = RunPolicyNet(dim=32, n_layers=3, n_heads=4)
    net.blocks[0].w1.weight.grad = None
    plasticity_stats(net, _tiny_batch())
    assert net.blocks[0].w1.weight.grad is None


def test_plasticity_stats_restores_training_mode():
    from ppo.model import RunPolicyNet

    net = RunPolicyNet(dim=32, n_layers=3, n_heads=4)
    net.train()
    plasticity_stats(net, _tiny_batch())
    assert net.training


# --- wiring into the trainer's per-iteration stats -------------------------


def _fake_transitions(n: int):
    """`n` Transitions carrying real Featurized rows (the probe featurizes)."""
    from ppo.encoder import featurize_row
    from ppo.trainer import Transition

    obs = {"deck": [], "relics": [], "potions": [], "map": {}}
    actions = [{"kind": "skip"}, {"kind": "take", "card": "Strike"}]
    metrics = {"total_floor": 3, "hp": 50, "max_hp": 80, "gold": 100, "act_index": 1}
    feat = featurize_row(obs, actions, metrics)
    return [Transition(feat=feat, action_idx=0, old_logprob=-0.7, value=0.3) for _ in range(n)]


def test_plasticity_sample_is_off_when_the_budget_is_zero():
    from ppo.model import RunPolicyNet
    from ppo.trainer import plasticity_sample

    net = RunPolicyNet(dim=32, n_layers=3, n_heads=4)
    assert plasticity_sample(net, _fake_transitions(8), 0, torch.device("cpu")) == {}


def test_plasticity_sample_is_off_when_there_are_no_transitions():
    from ppo.model import RunPolicyNet
    from ppo.trainer import plasticity_sample

    net = RunPolicyNet(dim=32, n_layers=3, n_heads=4)
    assert plasticity_sample(net, [], 256, torch.device("cpu")) == {}


def test_plasticity_sample_reports_the_diagnostic_keys():
    from ppo.model import RunPolicyNet
    from ppo.trainer import plasticity_sample

    net = RunPolicyNet(dim=32, n_layers=3, n_heads=4)
    out = plasticity_sample(net, _fake_transitions(8), 4, torch.device("cpu"))
    assert set(out) >= {"dormant_frac", "effective_rank", "effective_rank_frac"}


def test_plasticity_sample_is_deterministic_for_the_same_transitions():
    """A trend across iterations is the whole signal, so the probe must not add
    sampling noise of its own on top of the drifting state distribution."""
    from ppo.model import RunPolicyNet
    from ppo.trainer import plasticity_sample

    net = RunPolicyNet(dim=32, n_layers=3, n_heads=4)
    trans = _fake_transitions(8)
    a = plasticity_sample(net, trans, 4, torch.device("cpu"))
    b = plasticity_sample(net, trans, 4, torch.device("cpu"))
    assert a == b


def test_plasticity_budget_is_capped_by_the_transitions_available():
    from ppo.model import RunPolicyNet
    from ppo.trainer import plasticity_sample

    net = RunPolicyNet(dim=32, n_layers=3, n_heads=4)
    assert plasticity_sample(net, _fake_transitions(3), 256, torch.device("cpu"))


def test_plasticity_logging_is_on_by_default():
    from ppo.trainer import PPOConfig

    assert PPOConfig().plasticity_samples > 0


def test_a_real_iteration_writes_the_diagnostic_to_stats_jsonl(tmp_path):
    """End-to-end: the numbers have to reach the file a campaign is read from.

    Every unit above can pass while nothing calls the probe. This is the test
    that fails if the wiring is missing.
    """
    import json

    from ppo.trainer import PPOConfig, train

    save = tmp_path / "run"
    train(
        PPOConfig(
            character="Ironclad",
            iterations=1,
            games_per_iter=2,
            sims=4,
            boss_sims=4,
            procs=2,
            device="cpu",
            save_path=str(save),
        )
    )
    with open(f"{save}.stats.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    assert rows, "no stats were written"
    assert "dormant_frac" in rows[0]
    assert "effective_rank" in rows[0]
    assert "dormant_frac/block0" in rows[0]
