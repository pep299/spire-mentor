"""Unit tests for `ppo.shaping_gate` (GH #464).

The load-bearing one is `test_td_residuals_are_phi_invariant`: it is the
result the whole gate is built around, and the reason the gate reports a
magnitude rather than a winner.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
from ppo.rewards import ShapingConfig, base_value, shape_value, shaped_rewards
from ppo.shaping_gate import (
    Run,
    build_candidates,
    calibrate_alpha,
    design_matrix,
    evaluate,
    gae,
    kfold_predict,
    load_runs,
    main,
    parse_candidate,
    phi,
    terminal_targets,
)

CANDIDATES = [
    ShapingConfig(),
    ShapingConfig(hp_coef=0.0, upg_coef=0.0, relic_coef=0.0, offset=0.0),
    ShapingConfig(hp_coef=0.0351, upg_coef=0.0015, relic_coef=0.0, offset=0.0314),
    ShapingConfig(hp_coef=0.4, upg_coef=-0.2, relic_coef=0.11, offset=1.7),
]


def _metrics(floor: int, rng: random.Random) -> dict[str, int]:
    return {
        "total_floor": floor,
        "act_index": floor // 15,
        "act_floor": floor % 15,
        "hp": rng.randint(10, 80),
        "max_hp": 80,
        "gold": rng.randint(0, 400),
        "deck_size": 10 + floor // 2,
        "num_upgraded": rng.randint(0, floor),
        "num_relics": rng.randint(0, 8),
        "num_potions": rng.randint(0, 3),
    }


def _run(rng: random.Random, n: int | None = None, won: bool | None = None) -> Run:
    n = n if n is not None else rng.randint(8, 40)
    metrics = [_metrics(f, rng) for f in range(n)]
    final = _metrics(n, rng)
    return Run(metrics=metrics, final=final, won=rng.random() < 0.4 if won is None else won)


def _corpus(n_runs: int = 40, seed: int = 0) -> list[Run]:
    rng = random.Random(seed)
    return [_run(rng) for _ in range(n_runs)]


def _fixture(runs: list[Run], alpha: float = 1.0, folds: int = 5):
    """→ (X, run_id, m_hat, a_ref) — the shared, config-independent scaffolding."""
    X, rid = design_matrix(runs)
    y = terminal_targets(runs)
    m_hat = kfold_predict(X, y, rid, len(runs), folds, alpha)
    return X, rid, m_hat, y - m_hat


def _score(runs, cfg, fx, *, oracle: bool, lam=0.97, alpha=1.0, folds=5):
    X, rid, m_hat, a_ref = fx
    return evaluate(
        runs, cfg, a_ref, X, rid, folds=folds, alpha=alpha, lam=lam, m_hat=m_hat if oracle else None
    )


# ── the theorem the gate is built around ─────────────────────────────────────


@pytest.mark.parametrize("lam", [1.0, 0.97, 0.5, 0.0])
def test_td_residuals_are_phi_invariant(lam):
    """`V = m - Phi` ⇒ `delta_t = m_{t+1} - m_t`, so A_lam carries no coefficient.

    This is stronger than policy invariance: the gradient ESTIMATOR is
    untouched, at every lambda — which is why shaping can only act through
    critic misfit.
    """
    runs = _corpus(6, seed=1)
    rng = np.random.default_rng(0)
    # ANY m(s) works — the identity does not care what the critic predicts,
    # only that it is offset by exactly -Phi. Drawn once, shared by every
    # candidate, so the only thing varying across `advs` is the coefficients.
    m_per_run = [rng.normal(size=len(run.metrics)) for run in runs]

    advs = []
    for cfg in CANDIDATES:
        per_run = [
            gae(
                np.array(shaped_rewards(run.metrics, run.final, run.won, cfg)[0]),
                m - phi(run, cfg),
                lam=lam,
            )
            for run, m in zip(runs, m_per_run)
        ]
        advs.append(np.concatenate(per_run))
    for other in advs[1:]:
        assert np.allclose(advs[0], other, atol=1e-9)


def test_oracle_critic_collapses_every_candidate():
    """The gate's own null: with zero misfit the report MUST be flat."""
    runs = _corpus()
    fx = _fixture(runs)
    corrs = [_score(runs, cfg, fx, oracle=True)["adv_corr"] for cfg in CANDIDATES]
    assert max(corrs) - min(corrs) < 1e-9, corrs


def test_oracle_at_lambda_one_reproduces_the_reference_advantage():
    runs = _corpus(20, seed=7)
    fx = _fixture(runs)
    for cfg in CANDIDATES:
        stats = _score(runs, cfg, fx, oracle=True, lam=1.0)
        assert stats["adv_corr"] == pytest.approx(1.0, abs=1e-9)
        assert stats["adv_std_ratio"] == pytest.approx(1.0, abs=1e-9)


# ── the measurement ──────────────────────────────────────────────────────────


def test_fitted_critic_does_separate_candidates():
    """A gate that cannot separate anything measures nothing."""
    runs = _corpus(60, seed=11)
    fx = _fixture(runs)
    corrs = [_score(runs, cfg, fx, oracle=False, alpha=1e3)["adv_corr"] for cfg in CANDIDATES]
    assert max(corrs) - min(corrs) > 1e-3, corrs


def test_separation_shrinks_as_the_critic_improves():
    """Misfit is the ONLY channel, so a better critic must leave less room."""
    runs = _corpus(60, seed=11)
    fx = _fixture(runs)

    def spread(alpha: float) -> float:
        c = [_score(runs, cfg, fx, oracle=False, alpha=alpha)["adv_corr"] for cfg in CANDIDATES]
        return max(c) - min(c)

    assert spread(1e-4) < spread(1e4)


def test_reference_advantage_is_config_independent():
    """`a_ref` is built from base(T) alone — no candidate gets an easier target."""
    runs = _corpus(15, seed=13)
    _, _, _, a_ref = _fixture(runs)
    expected = terminal_targets(runs) - _fixture(runs)[2]
    assert np.allclose(a_ref, expected)


# ── GAE ──────────────────────────────────────────────────────────────────────


def test_gae_matches_the_trainer():
    """Same recursion as `trainer.gae_from_row`, without torch or a checkpoint."""
    trainer = pytest.importorskip("ppo.trainer")
    rng = np.random.default_rng(0)
    rewards, values = rng.normal(size=17), rng.normal(size=17)
    row = {
        "transitions": [
            {"feat": None, "action_idx": 0, "old_logprob": 0.0, "value": float(v), "metrics": {}}
            for v in values
        ],
        "final": {},
        "won": False,
    }
    cfg = trainer.PPOConfig(gamma=1.0, lam=0.9)
    original = trainer.shaped_rewards
    trainer.shaped_rewards = lambda *a, **k: (list(rewards), 0.0)
    try:
        got = [t.advantage for t in trainer.gae_from_row(row, cfg, ShapingConfig())]
    finally:
        trainer.shaped_rewards = original
    assert np.allclose(got, gae(rewards, values, lam=0.9, gamma=1.0))


def test_gae_zero_terminal_bootstrap():
    assert gae(np.array([1.0]), np.array([0.25]), lam=0.97)[0] == pytest.approx(0.75)


# ── baseline fitting ─────────────────────────────────────────────────────────


def test_kfold_splits_by_run_not_by_state():
    """Every state in a run shares its `base(T)`, so a state-wise split leaks."""
    rng = random.Random(5)
    # Outcome is a coin flip uncorrelated with any feature: an honest OOS fit
    # cannot beat predicting the mean, so a high R^2 can only be leakage.
    runs = [_run(rng, n=25, won=(i % 2 == 0)) for i in range(40)]
    X, rid = design_matrix(runs)
    y = terminal_targets(runs)
    pred = kfold_predict(X, y, rid, len(runs), 5, 1.0)
    r2 = 1.0 - float(((y - pred) ** 2).mean()) / float(y.var())
    assert r2 < 0.5, f"leak: OOS R^2 {r2:.3f} on an unpredictable target"


def test_kfold_single_run_falls_back_to_in_sample():
    runs = _corpus(1, seed=2)
    X, rid = design_matrix(runs)
    y = terminal_targets(runs)
    pred = kfold_predict(X, y, rid, 1, 5, 1.0)
    assert pred.shape == y.shape and np.isfinite(pred).all()


def test_calibrate_alpha_hits_the_target_ev():
    runs = _corpus(60, seed=17)
    X, rid = design_matrix(runs)
    alpha, gap, ev = calibrate_alpha(runs, ShapingConfig(), X, rid, 5, target_ev=0.5)
    assert alpha > 0
    assert gap == pytest.approx(abs(ev - 0.5))
    # A lower target must not pick a *sharper* critic.
    loose, _, loose_ev = calibrate_alpha(runs, ShapingConfig(), X, rid, 5, target_ev=0.0)
    assert loose >= alpha and loose_ev <= ev


def test_phi_helper_agrees_with_rewards_module():
    rng = random.Random(1)
    run = Run(metrics=[_metrics(7, rng)], final=_metrics(9, rng), won=False)
    cfg = ShapingConfig(hp_coef=0.2, upg_coef=0.01, relic_coef=0.03, offset=0.5)
    m = run.metrics[0]
    assert (
        phi(run, cfg)[0]
        == pytest.approx(base_value(m, False, cfg) + shape_value(m, cfg))
        == pytest.approx(
            cfg.progress_weight * min(1.0, 7 / 45)
            + cfg.hp_coef * (m["hp"] / m["max_hp"])
            + cfg.upg_coef * m["num_upgraded"]
            + cfg.relic_coef * m["num_relics"]
            - cfg.offset
        )
    )


# ── corpus loading / CLI ─────────────────────────────────────────────────────


def _write_corpus(path, n_runs: int = 30, seed: int = 0) -> None:
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as fh:
        for _ in range(n_runs):
            run = _run(rng)
            fh.write(
                json.dumps(
                    {
                        "outcome": "won" if run.won else "lost",
                        "won": run.won,
                        "final": run.final,
                        "decisions": [{"metrics": m} for m in run.metrics],
                    }
                )
                + "\n"
            )


def test_load_runs_skips_failed_and_empty_rows(tmp_path):
    p = tmp_path / "c.jsonl"
    good = {
        "outcome": "lost",
        "won": False,
        "final": {"total_floor": 3},
        "decisions": [{"metrics": {"total_floor": 1}}],
    }
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(good) + "\n")
        fh.write(json.dumps({**good, "outcome": "error"}) + "\n")
        fh.write(json.dumps({**good, "outcome": "timeout"}) + "\n")
        fh.write(json.dumps({**good, "decisions": []}) + "\n")
        fh.write("\n")
    assert len(load_runs(str(p))) == 1


def test_load_runs_respects_limit(tmp_path):
    p = tmp_path / "c.jsonl"
    _write_corpus(p, n_runs=10)
    assert len(load_runs(str(p), limit=4)) == 4


def test_parse_candidate():
    name, cfg = parse_candidate("mine=0.05,0.01,0.02,0.3")
    assert name == "mine"
    assert (cfg.hp_coef, cfg.upg_coef, cfg.relic_coef, cfg.offset) == (0.05, 0.01, 0.02, 0.3)


@pytest.mark.parametrize("spec", ["", "nameonly", "x=1,2,3", "x=1,2,3,4,5", "=1,2,3,4"])
def test_parse_candidate_rejects_junk(spec):
    with pytest.raises(ValueError):
        parse_candidate(spec)


def test_build_candidates_has_the_shipped_configs():
    runs = _corpus(10, seed=4)
    names = [n for n, _ in build_candidates(runs, [], with_fit=True)]
    assert names == ["priors", "none", "necro_fit", "corpus_fit"]
    assert [n for n, _ in build_candidates(runs, ["x=0,0,0,0"], with_fit=False)][-1] == "x"


def test_main_reports_both_tables_and_passes_the_null(tmp_path, capsys):
    corpus, out = tmp_path / "c.jsonl", tmp_path / "gate.json"
    _write_corpus(corpus, n_runs=40)
    assert main(["--corpus", str(corpus), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "ideal critic" in printed and "fitted critic" in printed
    assert "OK — matches the theorem" in printed
    payload = json.loads(out.read_text())
    assert payload["n_runs"] == 40
    assert payload["oracle"]["adv_corr_spread"] < 1e-9
    assert {c["name"] for c in payload["oracle"]["candidates"]} == {
        "priors",
        "none",
        "necro_fit",
        "corpus_fit",
    }


def test_main_calibrates_alpha(tmp_path, capsys):
    corpus = tmp_path / "c.jsonl"
    _write_corpus(corpus, n_runs=40)
    assert main(["--corpus", str(corpus), "--calibrate-ev", "0.5"]) == 0
    assert "calibrated alpha=" in capsys.readouterr().out


def test_main_on_an_unusable_corpus(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert main(["--corpus", str(p)]) == 1
