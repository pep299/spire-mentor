"""GH #830 G-D: the pre-registered n_eq metrics on synthetic reference groups.

Each group's members have true values q; a playout is q plus unit noise, so
the k-playout mean is the textbook estimator with SD 1/sqrt(k). An estimator
equal to q plus noise of SD 1/sqrt(n) must then score n_eq ~ n on both
metrics, a fresh playout ~ 1, and a constant 0 (calibrated) / 1 (its mix).
"""

from __future__ import annotations

import numpy as np
import pytest
from ppo.deepbets import value_neq as vn

N_R, N_POOL, N_FRESH = 256, 128, 8


def _synth(
    rng: np.random.Generator, n_groups: int, spread: float = 0.25, crn: bool = False
) -> tuple[vn.RefSet, np.ndarray]:
    groups, q = [], []
    n_tot = N_R + N_POOL + N_FRESH
    for _ in range(n_groups):
        n = int(rng.integers(2, 6))
        qg = rng.normal(0.0, spread, n) + rng.normal(0.0, 0.5)
        if crn:
            shared = rng.normal(0.0, 1.0, n_tot)
            own = rng.normal(0.0, 1.0, (n, n_tot))
            noise = 0.8 * shared[None, :] + 0.6 * own
        else:
            noise = rng.normal(0.0, 1.0, (n, n_tot))
        groups.append(qg[:, None] + noise)
        q.append(qg)
    clusters = [f"fight{g // 2}" for g in range(n_groups)]
    rs = vn.build_refset(groups, N_R, N_POOL, N_FRESH, clusters, [crn] * n_groups)
    qpad = np.zeros_like(rs.R)
    for gi, qg in enumerate(q):
        qpad[gi, : len(qg)] = qg
    return rs, qpad


@pytest.fixture(scope="module")
def sets() -> dict[str, object]:
    rng = np.random.default_rng(7)
    test, q_test = _synth(rng, 700)
    dev, q_dev = _synth(rng, 700)
    return {"test": test, "q_test": q_test, "dev": dev, "q_dev": q_dev, "rng": rng}


def _noisy(rng: np.random.Generator, q: np.ndarray, mask: np.ndarray, n: float) -> np.ndarray:
    return np.where(mask, q + rng.normal(0.0, 1.0 / np.sqrt(n), q.shape), 0.0)


@pytest.fixture(scope="module")
def report(sets: dict) -> dict:
    rng = np.random.default_rng(11)
    test, dev = sets["test"], sets["dev"]
    ests = {
        "const": (np.zeros_like(test.R), np.zeros_like(dev.R)),
        "n8": (
            _noisy(rng, sets["q_test"], test.mask, 8.0),
            _noisy(rng, sets["q_dev"], dev.mask, 8.0),
        ),
        "truth": (sets["q_test"], sets["q_dev"]),
    }
    return vn.evaluate_stratum(test, dev, ests, n_boot=60, seed=1)


def test_constant_v_scores_zero_calibrated_and_one_regret(report: dict) -> None:
    e = report["estimators"]["const"]
    assert e["neq_cal"] < 0.5
    assert e["R2_V"] == pytest.approx(0.0, abs=1e-9)
    # Any lam < 1 ranks a constant's mix by its playout alone: the k = 1 point.
    assert e["neq_regret_M"] == pytest.approx(1.0, abs=0.05)
    # On its own a constant is a random pick: k = 0.
    assert e["neq_regret_V"] == pytest.approx(0.0, abs=1e-9)
    assert report["sanity"]["checks"]["const_cal_lt_0.5"]
    assert report["sanity"]["checks"]["const_regret_M_approx_1"]


def test_fresh_single_playout_scores_about_one(report: dict) -> None:
    """The sanity block scores each fresh playout column as an estimator and
    averages: one column alone is noisy (its calibrated n_eq has an SD of about
    0.3 at 700 groups), the mean over columns is not."""
    s = report["sanity"]
    f = s["fresh_playout_mean_over_columns"]
    assert s["fresh_columns"] == N_FRESH
    assert 0.75 <= f["neq_cal"] <= 1.33
    assert 0.75 <= f["neq_regret_V"] <= 1.33
    # Mixed half and half with the pool playout it is a 2-playout mean.
    assert 1.5 <= f["neq_regret_M"] <= 2.7
    assert s["checks"]["fresh_cal_approx_1"] and s["checks"]["fresh_regret_V_approx_1"]
    assert s["ok"]


def test_an_estimator_worth_eight_playouts_scores_about_eight(report: dict) -> None:
    e = report["estimators"]["n8"]
    assert 5.0 <= e["neq_cal"] <= 13.0
    assert 4.0 <= e["neq_regret_V"] <= 16.0
    lo, hi = e["neq_cal_ci90"]
    assert lo <= e["neq_cal"] <= hi


def test_noise_correction_gives_the_truth_full_credit(report: dict) -> None:
    e = report["estimators"]["truth"]
    assert e["R2_V"] == pytest.approx(1.0, abs=0.08)
    assert e["neq_cal_censored"]
    assert e["neq_regret_V"] >= 32


def test_raw_mse_neq_rewards_a_constant(report: dict) -> None:
    """Why raw-MSE n_eq is ruled out: sigma^2 / E[d^2] = 1 / 0.0625 * ... >> 1."""
    assert report["raw_mse_neq_const"] > 10


def test_regret_curve_falls_and_is_informative(report: dict) -> None:
    c = report["regret_curve"]
    assert c["0"] > c["1"] > c["8"] > c["64"]
    assert not report["uninformative"]


def test_uninformative_when_every_member_is_equal() -> None:
    rng = np.random.default_rng(3)
    rs, _ = _synth(rng, 200, spread=0.0)
    dev, _ = _synth(rng, 200, spread=0.0)
    out = vn.evaluate_stratum(rs, dev, {}, n_boot=40, seed=2)
    assert out["uninformative"]


def test_crn_ratio() -> None:
    rng = np.random.default_rng(5)
    ind, _ = _synth(rng, 150)
    crn, _ = _synth(rng, 150, crn=True)
    r_ind, r_crn = vn.crn_ratio(ind), vn.crn_ratio(crn)
    assert r_ind["pairs"] == pytest.approx(1.0, abs=0.05)
    assert r_ind["centred"] == pytest.approx(1.0, abs=0.08)
    # Shared noise 0.8^2 of the variance: differences keep 0.36 of it.
    assert r_crn["pairs"] == pytest.approx(0.36, abs=0.05)
    assert r_crn["centred"] == pytest.approx(0.36, abs=0.06)


def test_group_regret_splits_ties() -> None:
    R = np.array([[1.0, 0.0, 0.5]])
    mask = np.ones((1, 3), bool)
    assert vn.group_regret(R, mask, np.zeros((1, 3)))[0] == pytest.approx(0.5)
    assert vn.group_regret(R, mask, np.array([[3.0, 1.0, 3.0]]))[0] == pytest.approx(0.25)
    padded = np.array([[1.0, 0.0, 0.0]])
    pmask = np.array([[True, True, False]])
    assert vn.group_regret(padded, pmask, np.array([[0.0, 1.0, 9.0]]))[0] == pytest.approx(1.0)


def test_neq_from_curve_interpolation() -> None:
    ys = np.array([0.0, 0.1, 0.2, 0.3, 0.4])  # k = 0..4, increasing
    assert vn.neq_from_curve(ys, 0.05, True) == (pytest.approx(0.5), False)
    k, cens = vn.neq_from_curve(ys, 0.15, True)
    assert k == pytest.approx(np.sqrt(2.0)) and not cens  # log-linear between 1 and 2
    assert vn.neq_from_curve(ys, 0.9, True) == (4.0, True)
    assert vn.neq_from_curve(ys, -1.0, True) == (0.0, False)
    # A decreasing curve (regret), with a noisy bump the running min removes.
    reg = np.array([1.0, 0.6, 0.65, 0.4])
    assert vn.neq_from_curve(reg, 0.6, False)[0] == pytest.approx(1.0)


def test_verdict_refuses_a_failed_sanity_check(report: dict) -> None:
    bad = {"strata": {"contested_boss": {**report, "sanity": {"ok": False}}}}
    assert vn.verdict(bad, 4.0)["verdict"] == "INVALID"


def test_verdict_bars(report: dict) -> None:
    def stratum(cal: float, reg: float, unin: bool = False) -> dict:
        e = {"neq_cal": cal, "neq_regret_M": reg, "lambda": 0.5}
        return {"sanity": {"ok": True}, "uninformative": unin, "estimators": {"V": e}}

    rep = {"strata": {"contested_boss": stratum(5, 6), "contested_elite": stratum(4.5, 9)}}
    assert vn.verdict(rep, 4.0)["verdict"] == "PASS"
    rep["strata"]["contested_elite"] = stratum(3, 9)
    assert vn.verdict(rep, 4.0)["verdict"] == "RETRY_BAND"
    rep["strata"]["contested_elite"] = stratum(1.5, 9)
    assert vn.verdict(rep, 4.0)["verdict"] == "KILL"
    # An uninformative regret curve leaves the calibrated metric to decide.
    rep["strata"]["contested_elite"] = stratum(6, 1.0, unin=True)
    assert vn.verdict(rep, 4.0)["verdict"] == "PASS"
    rep["strata"]["contested_boss"]["estimators"]["texel"] = {"neq_cal": 7.0}
    assert vn.verdict(rep, 4.0)["verdict"] == "KILL"


def test_refs_shards_round_trip(tmp_path) -> None:
    """`write_refs` -> `load_refs` keeps every group's playouts, member ids and
    tags; a group's values are a slice of ONE in-memory copy per shard (reading
    `NpzFile[key]` per group once held ~60 GB of decompressed copies)."""
    import json

    from ppo.deepbets import value_label as vl

    rng = np.random.default_rng(0)
    groups = []
    for gi, n in enumerate((2, 3, 4)):
        members = [{"id": f"g{gi}:m{j}", "kind": "child"} for j in range(n)]
        g = {"group": f"g:{gi}", "stratum": "contested_boss", "room": "boss",
             "fight": f"f{gi % 2}", "crn": False, "deep": False, "p_parent": 0.5,
             "members": members}  # fmt: skip
        groups.append({"group": g, "vals": rng.normal(size=(n, 12)).astype(np.float32)})
    vl.write_refs(str(tmp_path / "refs-00000.npz"), groups)
    (tmp_path / "manifest.json").write_text(json.dumps({"n_r": 8, "n_ref": 10, "n_fresh": 2}))
    _, out = vn.load_refs(str(tmp_path))
    d = out["contested_boss"]
    assert d["group"] == ["g:0", "g:1", "g:2"]
    assert d["members"][2] == ["g2:m0", "g2:m1", "g2:m2", "g2:m3"]
    for gi in range(3):
        np.testing.assert_array_equal(d["vals"][gi], groups[gi]["vals"])
    bases = {id(v.base) for v in d["vals"] if v.base is not None}
    assert len(bases) <= 1
