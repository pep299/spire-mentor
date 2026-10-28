"""`ppo.combat_bench` report + statistics (GH #830), on synthetic data.

Every expectation here is hand-computed or follows from an identity, so a
regression in an estimator shows up as a wrong number, not a crash.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ppo.combat_bench import report as R
from ppo.combat_bench import stats as S

# ── synthetic records ──────────────────────────────────────────────────


def _fight(
    key: str,
    player: str = "p1",
    run: str = "r1",
    *,
    room: str = "boss",
    act: int = 1,
    enc: str = "E1",
    hp: int = 50,
    dt: int = 10,
    died: bool = False,
    quit: bool = False,
    flags: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "key": key,
        "file": run,
        "floor": 1,
        "act": act,
        "room_type": room,
        "encounter": enc,
        "character": "Ironclad",
        "ascension": 10,
        "player": player,
        "cohort": "pro",
        "run_won": False,
        "run_abandoned": quit,
        "deck": [],
        "relics": [],
        "potions": [],
        "belt": None,
        "belt_unused": False,
        "monsters": None,
        "hp_before": hp,
        "max_hp": 80,
        "n_enchanted": 0,
        "human": {
            "hp_after": 0 if died or quit else hp - dt,
            "damage_taken": dt,
            "hp_healed": 0,
            "max_hp_gained": 0,
            "max_hp_lost": 0,
            "turns": 5,
            "died": died,
            "quit": quit,
        },
        "flags": list(flags),
    }


def _cell(
    key: str,
    rep: int,
    *,
    end: str = "won",
    dmg_c: int = 5,
    turns: int = 3,
    block: int = 0,
    sims: int | None = 100,
    cpu: float | None = 0.5,
    decisions: int = 10,
) -> dict[str, Any]:
    return {
        "key": key,
        "block": block,
        "rep": {
            "rep": rep,
            "seed": 1,
            "end": end,
            "died": end == "died",
            "hp_end_combat": 0,
            "hp_after": 0,
            "max_hp_after": 80,
            "dmg": dmg_c,
            "dmg_c": dmg_c,
            "turns": turns,
            "steps": 10,
            "decisions": decisions,
            "sims_spent": sims,
            "cpu_s": cpu,
            "potions_left": 0,
            "drunk": 0,
        },
    }


def _write_set(tmp: Path, fights: list[dict[str, Any]]) -> tuple[str, str]:
    p = tmp / "set.jsonl"
    p.write_text("".join(json.dumps(f) + "\n" for f in fights))
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


def _write_played(
    tmp: Path,
    name: str,
    sha: str,
    cells: list[dict[str, Any]],
    *,
    src: str = "src0",
    reps: int = 2,
    block_size: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    header = {
        "schema": R.PLAYED_SCHEMA,
        "set_name": "t",
        "set_sha256": sha,
        "spec": {"kind": "mcts", "name": name},
        "src_hash": src,
        "reps": reps,
        "rep_offset": 0,
        "caps": {},
    }
    if block_size is not None:
        header["block_size"] = block_size
    header |= extra or {}
    p = tmp / f"{name}.jsonl"
    p.write_text(
        json.dumps({"header": header}) + "\n" + "".join(json.dumps(c) + "\n" for c in cells)
    )
    return str(p)


# ── estimators ─────────────────────────────────────────────────────────


def test_cr1_hand_example() -> None:
    # mean 4, residuals [-3,-2,-1,0,6], cluster sums a=-5 b=-1 c=6, sum of squares 62,
    # g/(g-1) = 1.5 -> se = sqrt(93) / 5.
    est = S.cr1([1, 2, 3, 4, 10], ["a", "a", "b", "b", "c"])
    assert est.mean == pytest.approx(4.0)
    assert est.se == pytest.approx(math.sqrt(93) / 5)
    assert (est.n, est.g) == (5, 3)


def test_cr1_matches_the_explicit_sandwich() -> None:
    rng = np.random.default_rng(7)
    g = rng.integers(0, 9, size=60)
    x = rng.normal(size=60) + g * 0.3
    n, k, G = len(x), 1, len(set(g.tolist()))
    X = np.ones((n, 1))
    e = x - x.mean()
    bread = np.linalg.inv(X.T @ X)
    meat = sum(
        (X[g == c].T @ e[g == c])[:, None] @ (X[g == c].T @ e[g == c])[None, :] for c in set(g)
    )
    v = G / (G - 1) * (n - 1) / (n - k) * bread @ meat @ bread
    assert S.cr1(x, g).se == pytest.approx(math.sqrt(v[0, 0]), rel=1e-12)


def test_cr1_one_cluster_has_no_se() -> None:
    assert math.isnan(S.cr1([1.0, 2.0], ["a", "a"]).se)


def test_sign_flip_exact_small_cases() -> None:
    # |sum| = 5.5; extreme sign vectors: all +, all -, and flipping only the -0.5 (|6.5|)
    # either way -> 4 of 16.
    r = S.sign_flip_test([1, 2, -0.5, 3])
    assert r.exact and r.n_perm == 16
    assert r.p == pytest.approx(4 / 16)
    # Only the two constant sign vectors reach |3|.
    assert S.sign_flip_test([1, 1, 1]).p == pytest.approx(2 / 8)
    assert S.sign_flip_test([0.0, 0.0]).p == pytest.approx(1.0)


def test_sign_flip_random_approximates_exact() -> None:
    d = np.random.default_rng(3).normal(0.4, 1, size=12)
    exact = S.sign_flip_test(d, n_perm=5000)
    approx = S.sign_flip_test(d, n_perm=4000, seed=5)
    assert exact.exact and not approx.exact
    assert approx.p == pytest.approx(exact.p, abs=0.025)
    assert S.sign_flip_test(d, n_perm=4000, seed=5).p == approx.p


def test_holm() -> None:
    # sorted 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1 -> max(0.06, 0.04)
    assert S.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    out = S.holm([0.2, math.nan, 0.6])
    assert out[0] == pytest.approx(0.4) and out[2] == pytest.approx(0.6) and math.isnan(out[1])
    # 0.02*2 = 0.04, then 0.3*1 = 0.3; and 0.8*2 caps at 1, so 0.9 inherits the cap.
    assert S.holm([0.3, 0.02]) == pytest.approx([0.3, 0.04])
    assert S.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_eb_shrink_hand_example() -> None:
    # mu = 5, Q = 50, k = 2, sum w - sum w^2 / sum w = 1 -> tau2 = 49, weight 49/50.
    eb = S.eb_shrink([0.0, 10.0], [1.0, 1.0])
    assert eb.mu == pytest.approx(5.0) and eb.tau2 == pytest.approx(49.0)
    assert eb.shrunk == pytest.approx([0.1, 9.9])
    # No between-group spread beyond noise -> everything collapses onto the prior.
    eb = S.eb_shrink([4.9, 5.1], [1.0, 1.0], mu=3.0)
    assert eb.tau2 == pytest.approx(0.0) and eb.shrunk == pytest.approx([3.0, 3.0])


def test_webb_weights() -> None:
    w, exact = S.webb_weights(3, 9999, seed=0)
    assert exact and w.shape == (216, 3)
    assert w.mean(axis=0) == pytest.approx([0, 0, 0], abs=1e-12)
    assert (w**2).mean(axis=0) == pytest.approx([1, 1, 1])
    w, exact = S.webb_weights(12, 999, seed=0)
    assert not exact and w.shape == (999, 12)


def test_wild_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(11)
    cl = np.repeat(np.arange(10), rng.integers(3, 30, 10))
    x = rng.normal(1, 3, len(cl))
    a = S.wild_cluster_ci(x, cl, draws=999, seed=4)
    assert S.wild_cluster_ci(x, cl, draws=999, seed=4) == a
    assert S.wild_cluster_ci(x, cl, draws=999, seed=5) != a
    assert a.lo < a.mean < a.hi
    # 6^4 = 1296 <= draws: all weight vectors enumerated, the seed is irrelevant.
    cl4 = cl % 4
    e1 = S.wild_cluster_ci(x, cl4, draws=2000, seed=1)
    assert e1.exact and e1 == S.wild_cluster_ci(x, cl4, draws=2000, seed=2)


def test_wild_bootstrap_ci_covers_the_truth() -> None:
    """12 unbalanced clusters with a cluster effect: the CI covers mu about 95% of the time."""
    rng = np.random.default_rng(2026)
    mu, hits, widths, sims = 2.0, 0, [], 150
    for s in range(sims):
        cl = np.repeat(np.arange(12), rng.integers(5, 150, 12))
        x = mu + rng.normal(0, 3, 12)[cl] + rng.normal(0, 10, len(cl))
        ci = S.wild_cluster_ci(x, cl, draws=499, seed=s)
        hits += ci.lo <= mu <= ci.hi
        widths.append(ci.hi - ci.lo)
    assert 0.88 <= hits / sims <= 0.99
    assert all(math.isfinite(w) for w in widths)


def test_wild_bootstrap_degenerate_inputs() -> None:
    one = S.wild_cluster_ci([1.0, 2.0, 3.0], ["a", "a", "a"])
    assert one.mean == pytest.approx(2.0) and math.isnan(one.lo)
    flat = S.wild_cluster_ci([2.0, 2.0, 2.0], ["a", "b", "c"])
    assert (flat.lo, flat.hi) == (2.0, 2.0)


# ── report logic ───────────────────────────────────────────────────────


def test_loss_and_quit_interval() -> None:
    f = _fight("k", hp=30, dt=12)
    assert R.human_metrics(f, 65, "quit_excluded")["L"] == 12
    dead = _fight("k", hp=30, dt=40, died=True)
    # Overkill is clamped at hp_before.
    assert R.human_metrics(dead, 65, "quit_excluded")["L"] == 30 + 65
    q = _fight("k", hp=30, dt=12, quit=True)
    assert R.human_metrics(q, 65, "quit_excluded") is None
    assert R.human_metrics(q, 65, "quit_as_death") == {
        "L": 95,
        "died": 1.0,
        "dmg": 12.0,
        "turns": 5.0,
    }
    rep = _cell("k", 0, end="died", dmg_c=55)["rep"]
    assert R.rep_loss(rep, f, 65) == 30 + 65
    assert R.rep_loss(rep, f, 10) == 30 + 10
    # The quit bracket's low end: survived with the damage taken so far.
    assert R.human_metrics(q, 65, "quit_as_survived") == {
        "L": 12,
        "died": 0.0,
        "dmg": 12.0,
        "turns": 5.0,
    }


def test_a_death_counts_all_hp_whatever_the_damage_rows_say() -> None:
    # The Insatiable's Sandpit kills via CreatureCmd.Kill: 41 HP, only 31 of it
    # in DamageTaken. Both sides score a death as hp_before + death_hp.
    sandpit = _fight("k", hp=41, dt=31, died=True)
    assert R.human_metrics(sandpit, 65, "quit_excluded")["L"] == 41 + 65
    rep = _cell("k", 0, end="died", dmg_c=20)["rep"]
    assert R.rep_loss(rep, sandpit, 65) == 41 + 65
    # A win still clamps the damage at hp_before.
    assert R.rep_loss(_cell("k", 0, dmg_c=99)["rep"], sandpit, 65) == 41


def test_quit_interval_in_the_gap(tmp_path: Path) -> None:
    fights = [_fight("a", hp=50, dt=10), _fight("b", hp=40, dt=5, quit=True, run="r2")]
    set_path, sha = _write_set(tmp_path, fights)
    cells = [_cell("a", r, dmg_c=20) for r in (0, 1)] + [_cell("b", r, dmg_c=8) for r in (0, 1)]
    played = _write_played(tmp_path, "arm", sha, cells)
    out = tmp_path / "r.json"
    assert R.main(["--set", set_path, played, "--json", str(out), "--boot", "199"]) == 0
    g = json.loads(out.read_text())["human_gap"]["arm"]
    # excluded: only fight a, gap 20 - 10; as death: b adds 8 - (40 + 65);
    # as survived: b adds 8 - 5.
    assert g["quit_excluded"]["overall"]["gap"]["mean"] == pytest.approx(10.0)
    assert g["quit_as_death"]["overall"]["gap"]["mean"] == pytest.approx((10 + 8 - 105) / 2)
    surv = g["quit_as_survived"]["overall"]["gap"]["mean"]
    assert surv == pytest.approx((10 + 8 - 5) / 2)
    # The two ends bracket the truth; dropping the quit (10) is not an end:
    # here it lies outside the bracket [-43.5, 6.5].
    assert g["quit_as_death"]["overall"]["gap"]["mean"] < surv < 10.0


def test_pairing_keeps_the_intersection_of_scored_reps() -> None:
    fights = {k: _fight(k) for k in ("a", "b", "c")}
    a = {
        "a": {0: _cell("a", 0)["rep"], 1: _cell("a", 1, dmg_c=9)["rep"]},
        "b": {1: _cell("b", 1)["rep"]},
    }
    b = {
        "a": {1: _cell("a", 1, dmg_c=3)["rep"], 2: _cell("a", 2)["rep"]},
        "b": {0: _cell("b", 0)["rep"]},
        "c": {0: _cell("c", 0)["rep"]},
    }
    rows = R.pair_rows(fights, a, b, 65)
    # a: common rep {1} only; b: no common rep -> excluded; c: absent from arm a.
    assert [(r.key, r.reps) for r in rows] == [("a", (1,))]
    assert rows[0].d("L") == pytest.approx(9 - 3)


def test_cells_dropped_and_bounds(tmp_path: Path) -> None:
    fights = [_fight("a", hp=50, dt=10), _fight("b", hp=20, dt=0, run="r2")]
    set_path, sha = _write_set(tmp_path, fights)
    cells = [
        _cell("a", 0, dmg_c=10),
        _cell("a", 1, end="step_cap", dmg_c=99),
        _cell("b", 0, end="died", dmg_c=20),
        # b rep 1 is missing
    ]
    p = R.load_played(_write_played(tmp_path, "x", sha, cells), "x", R.load_set(set_path))
    c = R.cell_summary(p, R.load_set(set_path))
    assert (c["expected"], c["scored"], c["missing"], c["dropped"]) == (4, 2, 1, 2)
    assert c["ends"]["step_cap"] == 1 and c["ends"]["died"] == 1
    b = R.drop_bounds(p, R.load_set(set_path), 65)
    # scored-only fight means: a 10, b 85.  best: a (10+0)/2, b (85+0)/2;
    # worst: a (10+115)/2, b (85+85)/2.
    assert b["L_scored"] == pytest.approx((10 + 85) / 2)
    assert b["L_bound"] == pytest.approx([(5 + 42.5) / 2, (62.5 + 85) / 2])
    assert b["gap_bound_quit_excluded"] == pytest.approx(
        [((5 - 10) + (42.5 - 0)) / 2, ((62.5 - 10) + (85 - 0)) / 2]
    )


def test_complete_blocks_only(tmp_path: Path) -> None:
    fights = [_fight(k) for k in ("a", "b", "c")]
    set_path, sha = _write_set(tmp_path, fights)
    fs = R.load_set(set_path)
    # 3 fights x 2 reps = 6 cells in blocks of 2: block 0 full, block 1 half, block 2 full.
    cells = [
        _cell("a", 0, block=0),
        _cell("b", 1, block=0),
        _cell("c", 0, block=1),
        _cell("a", 1, block=2),
        _cell("c", 1, block=2),
    ]
    p = R.load_played(_write_played(tmp_path, "x", sha, cells, block_size=2), "x", fs)
    kept = R.complete_blocks_only(p, len(fs))
    assert sorted(kept.cells) == [("a", 0), ("a", 1), ("b", 1), ("c", 1)]
    assert kept.blocks_dropped == 1
    # Without a header block_size the largest block count (2) is used.
    p2 = R.load_played(_write_played(tmp_path, "y", sha, cells), "y", fs)
    assert sorted(R.complete_blocks_only(p2, len(fs)).cells) == sorted(kept.cells)


def test_refuses_header_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    set_path, sha = _write_set(tmp_path, [_fight("a"), _fight("b", run="r2")])
    cells = [_cell(k, r) for k in ("a", "b") for r in (0, 1)]
    a = _write_played(tmp_path, "a", sha, cells, src="engine1")
    b = _write_played(tmp_path, "b", sha, cells, src="engine2")
    assert R.main(["--set", set_path, a, b, "--boot", "99"]) == 2
    assert "src_hash differs" in capsys.readouterr().err
    assert R.main(["--set", set_path, a, b, "--boot", "99", "--allow-engine-mismatch"]) == 0
    assert "ENGINE MISMATCH" in capsys.readouterr().err
    other = _write_played(tmp_path, "c", "0" * 64, cells, src="engine1")
    assert R.main(["--set", set_path, a, other, "--boot", "99"]) == 2
    assert "set_sha256 mismatch" in capsys.readouterr().err


def test_retried_cells_keep_the_last_record(tmp_path: Path) -> None:
    fs = R.load_set(_write_set(tmp_path, [_fight("a")])[0])
    sha = R.file_sha256(str(tmp_path / "set.jsonl"))
    cells = [
        _cell("a", 0, end="watchdog"),
        _cell("a", 1, dmg_c=7),
        _cell("a", 0, dmg_c=9),  # `run --retry-ends watchdog` appended it
        _cell("a", 1, dmg_c=1),  # a plain repeat: the first record stands
    ]
    p = R.load_played(_write_played(tmp_path, "x", sha, cells), "x", fs)
    assert (p.cells["a", 0]["end"], p.cells["a", 0]["dmg_c"], p.cells["a", 1]["dmg_c"]) == (
        "won",
        9,
        7,
    )
    assert (p.retried, p.duplicates) == (1, 1)


def test_partial_block_warns_unless_complete_blocks_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_path, sha = _write_set(tmp_path, [_fight(k, run=k) for k in ("a", "b", "c")])
    # Block 1 holds 1 of its 2 planned cells: an interim read.
    cells = [_cell("a", 0, block=0), _cell("b", 1, block=0), _cell("c", 0, block=1)]
    cells += [_cell("a", 1, block=2), _cell("c", 1, block=2)]
    played = _write_played(tmp_path, "x", sha, cells, block_size=2)
    assert R.main(["--set", set_path, played, "--boot", "99"]) == 0
    assert "!!! PARTIAL BLOCK: x has 1 block(s)" in capsys.readouterr().err
    assert R.main(["--set", set_path, played, "--boot", "99", "--complete-blocks-only"]) == 0
    assert "PARTIAL BLOCK" not in capsys.readouterr().err


def test_sims_cap_ends_warn_loudly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    set_path, sha = _write_set(tmp_path, [_fight("a"), _fight("b", run="r2")])
    cells = [_cell("a", 0), _cell("a", 1, end="sims_cap"), _cell("b", 0), _cell("b", 1)]
    assert R.main(["--set", set_path, _write_played(tmp_path, "a", sha, cells)]) == 0
    assert "!!! SIMS CAP: a has 1 sims_cap ends" in capsys.readouterr().err


def test_cpu_ratio_needs_one_host_and_one_wheel_kind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_path, sha = _write_set(tmp_path, [_fight("a"), _fight("b", run="r2")])
    cells = [_cell(k, r, cpu=1.0) for k in ("a", "b") for r in (0, 1)]
    slow = [_cell(k, r, cpu=2.0) for k in ("a", "b") for r in (0, 1)]

    def cost(extra_b: dict[str, Any]) -> tuple[dict[str, Any], str]:
        base = _write_played(tmp_path, "base", sha, cells, extra={"pgo": True, "host": "h1"})
        arm = _write_played(tmp_path, "arm", sha, slow, extra={"pgo": True, "host": "h1"} | extra_b)
        out = tmp_path / "r.json"
        assert R.main(["--set", set_path, base, arm, "--boot", "99", "--json", str(out)]) == 0
        return json.loads(out.read_text())["cost"]["arm"]["vs_base"], capsys.readouterr().err

    ratio, err = cost({})
    assert ratio["cpu_s"] == pytest.approx(2.0) and "cpu x base is n/a" not in err
    ratio, err = cost({"pgo": False})  # a PGO wheel against a plain one: same src_hash
    assert ratio["cpu_s"] is None and ratio["sims_spent"] == pytest.approx(1.0)
    assert "!!! cpu x base is n/a: wheel kinds differ" in err
    ratio, err = cost({"host": "laptop"})
    assert ratio["cpu_s"] is None and "hosts differ" in err


def test_load_difference_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    set_path, sha = _write_set(tmp_path, [_fight("a")])
    idle, busy = [_cell("a", r) for r in (0, 1)], [_cell("a", r) for r in (0, 1)]
    for c in idle:
        c["rep"]["load1"] = 4.0
    for c in busy:
        c["rep"]["load1"] = 850.0
    hdr = {"cpus": 48}
    a = _write_played(tmp_path, "idle", sha, idle, extra=hdr)
    b = _write_played(tmp_path, "busy", sha, busy, extra=hdr)
    assert R.main(["--set", set_path, a, b, "--boot", "99"]) == 0
    assert "!!! LOAD differs between arms" in capsys.readouterr().err


def test_splits_carry_the_wild_player_ci(tmp_path: Path) -> None:
    """The split 'all' is the overall gap, with the same wild-by-player CI."""
    set_path, played, _, _ = _two_arm_bench(tmp_path)
    out = tmp_path / "r.json"
    assert R.main(["--set", set_path, played, "--boot", "199", "--json", str(out)]) == 0
    rep = json.loads(out.read_text())
    arm = rep["arms"][0]["name"]
    overall = rep["human_gap"][arm]["quit_excluded"]["overall"]
    split = next(s for s in rep["splits"] if s["split"] == "all")["gap"][arm]
    for k in ("mean", "ci_lo", "ci_hi"):
        assert split[k] == pytest.approx(overall["gap"][k])
    assert split["deaths_pp_ci_lo"] == pytest.approx(overall["deaths_pp"]["ci_lo"])
    stratum = rep["human_gap"][arm]["quit_excluded"]["strata"][0]
    assert stratum["deaths_pp"]["ci_lo"] == pytest.approx(overall["deaths_pp"]["ci_lo"])


def test_watchdog_warns_loudly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    set_path, sha = _write_set(tmp_path, [_fight("a"), _fight("b", run="r2")])
    cells = [_cell("a", 0), _cell("a", 1, end="watchdog"), _cell("b", 0), _cell("b", 1)]
    a = _write_played(tmp_path, "a", sha, cells)
    assert R.main(["--set", set_path, a, "--boot", "99"]) == 0
    assert "!!! WATCHDOG" in capsys.readouterr().err


def _two_arm_bench(tmp_path: Path, n: int = 24) -> tuple[str, str, str, dict[str, Any]]:
    """Two arms on n fights x 2 reps; arm `b` takes exactly 3 less dmg_c on every cell."""
    rng = np.random.default_rng(0)
    fights, ca, cb = [], [], []
    for i in range(n):
        k = f"f{i}"
        fights.append(
            _fight(k, player=f"p{i % 6}", run=f"r{i % 12}", hp=60, dt=int(rng.integers(0, 30)))
        )
        for r in (0, 1):
            d = int(rng.integers(5, 40))
            ca.append(_cell(k, r, dmg_c=d, sims=1000, cpu=1.0, decisions=10))
            cb.append(_cell(k, r, dmg_c=d - 3, sims=2000, cpu=3.0, decisions=10))
    set_path, sha = _write_set(tmp_path, fights)
    a = _write_played(tmp_path, "base", sha, ca)
    b = _write_played(tmp_path, "cand", sha, cb)
    out = tmp_path / "r.json"
    assert R.main(["--set", set_path, a, b, "--json", str(out), "--boot", "499"]) == 0
    return set_path, a, b, json.loads(out.read_text())


def test_paired_delta_holm_and_cost(tmp_path: Path) -> None:
    _, _, _, rep = _two_arm_bench(tmp_path)
    m = rep["paired"]["cand"]["metrics"]
    assert m["L"]["delta"] == pytest.approx(-3.0)
    assert m["L"]["cells"] == 48 and m["L"]["fights"] == 24
    # Every fight moves the same way: the sign-flip p is its floor 1 / (1 + perm).
    assert m["L"]["perm_p"] == pytest.approx(1 / 20001)
    assert m["L"]["holm_p"] == pytest.approx(m["L"]["perm_p"])
    assert m["deaths pp"]["delta"] == pytest.approx(0.0)
    cost = rep["cost"]
    assert cost["base"]["sims_per_decision"] == pytest.approx(100.0)
    assert cost["cand"]["cpu_s_per_fight"] == pytest.approx(3.0)
    assert cost["cand"]["vs_base"]["sims_spent"] == pytest.approx(2.0)
    assert cost["cand"]["vs_base"]["cpu_s"] == pytest.approx(3.0)


def test_pro_gap_via_baseline_equals_direct_gap_on_equal_reps(tmp_path: Path) -> None:
    """With identical scored reps, gap(B0) + delta(A - B0) == gap(A) fight by fight."""
    _, _, _, rep = _two_arm_bench(tmp_path)
    direct = rep["human_gap"]["cand"]["quit_excluded"]["overall"]["gap"]
    via = rep["pro_gap_via_baseline"]["cand"]["quit_excluded"]["gap"]
    assert via["mean"] == pytest.approx(direct["mean"])
    assert via["ci_lo"] == pytest.approx(direct["ci_lo"], abs=1e-3)
    assert via["ci_hi"] == pytest.approx(direct["ci_hi"], abs=1e-3)
    base = rep["pro_gap_via_baseline"]["base"]["quit_excluded"]["gap"]
    assert via["mean"] - base["mean"] == pytest.approx(-3.0)


def test_equal_player_weight_variant(tmp_path: Path) -> None:
    # p1 has three fights with gap 10, p2 one fight with gap 2: fight mean 8, player mean 6.
    fights = [_fight(f"a{i}", "p1", f"r{i}", dt=0) for i in range(3)]
    fights.append(_fight("b", "p2", "r9", dt=0))
    set_path, sha = _write_set(tmp_path, fights)
    cells = [_cell(f"a{i}", r, dmg_c=10) for i in range(3) for r in (0, 1)]
    cells += [_cell("b", r, dmg_c=2) for r in (0, 1)]
    played = _write_played(tmp_path, "arm", sha, cells)
    out = tmp_path / "r.json"
    assert R.main(["--set", set_path, played, "--json", str(out), "--boot", "99"]) == 0
    ov = json.loads(out.read_text())["human_gap"]["arm"]["quit_excluded"]["overall"]
    assert ov["gap"]["mean"] == pytest.approx(8.0)
    assert ov["gap_equal_player"]["mean"] == pytest.approx(6.0)


def test_self_null_is_about_zero(tmp_path: Path) -> None:
    """Exchangeable reps: one rep playing the human gives a gap whose CI covers 0."""
    rng = np.random.default_rng(42)
    fights, cells = [], []
    for i in range(240):
        k = f"f{i}"
        fights.append(_fight(k, player=f"p{i % 12}", run=f"r{i % 60}", hp=70, act=1 + i % 3))
        level = rng.uniform(5, 40)
        for r in range(4):
            dmg = int(max(0, rng.normal(level, 8)))
            cells.append(_cell(k, r, dmg_c=dmg, end="died" if dmg >= 70 else "won"))
    set_path, sha = _write_set(tmp_path, fights)
    played = _write_played(tmp_path, "sn", sha, cells, reps=4)
    out = tmp_path / "r.json"
    argv = ["--set", set_path, "--self-null", played, "--json", str(out), "--boot", "499"]
    assert R.main([*argv, "--self-null-coverage", "--self-null-trials", "6"]) == 0
    sn = json.loads(out.read_text())["self_null"]
    allrow = sn["strata"][0]
    assert allrow["stratum"] == "all" and allrow["fights"] == 240
    g = allrow["gap"]
    assert g["ci_lo"] <= 0.0 <= g["ci_hi"]
    assert abs(g["mean"]) < 3 * g["se_cr1"]
    cov = sn["coverage"]
    assert cov["trials"] == 4 + 6
    assert cov["coverage"]["wild_player"] >= 0.8
    assert abs(cov["gap_all_mean"]) < 1.0


def test_holm_across_arms_and_delta_bounds(tmp_path: Path) -> None:
    """Three arms: Holm runs over the two non-baseline arms; dropped cells widen the bound."""
    rng = np.random.default_rng(5)
    fights, base, a1, a2 = [], [], [], []
    for i in range(30):
        k = f"f{i}"
        fights.append(_fight(k, player=f"p{i % 5}", run=f"r{i % 10}", hp=60, dt=10))
        for r in (0, 1):
            d = int(rng.integers(10, 30))
            base.append(_cell(k, r, dmg_c=d))
            a1.append(_cell(k, r, dmg_c=d + int(rng.integers(-3, 4))))
            # a2: one cell capped, so it is dropped and only rep 1 pairs on f0.
            end = "round_cap" if (i, r) == (0, 0) else "won"
            a2.append(_cell(k, r, dmg_c=d - 2, end=end))
    set_path, sha = _write_set(tmp_path, fights)
    paths = [_write_played(tmp_path, n, sha, c) for n, c in (("b", base), ("x", a1), ("y", a2))]
    out = tmp_path / "r.json"
    argv = ["--set", set_path, *paths, "--baseline", "b", "--json", str(out), "--boot", "99"]
    assert R.main(argv) == 0
    rep = json.loads(out.read_text())
    px = rep["paired"]["x"]["metrics"]["L"]
    py = rep["paired"]["y"]["metrics"]["L"]
    assert [px["holm_p"], py["holm_p"]] == pytest.approx(S.holm([px["perm_p"], py["perm_p"]]))
    assert (py["fights"], py["cells"]) == (30, 59)
    assert py["delta"] == pytest.approx(-2.0)
    # f0 rep 0 of y is dropped: best case L=0 vs base L, worst case 60+65 vs base L.
    lb0 = next(c["rep"]["dmg_c"] for c in base if (c["key"], c["rep"]["rep"]) == ("f0", 0))
    lo, hi = rep["paired"]["y"]["delta_bound_L"]
    assert lo == pytest.approx(-2.0 + ((0 - lb0) - (lb0 - 2 - lb0)) / 2 / 30)
    assert hi == pytest.approx(-2.0 + ((125 - lb0) - (lb0 - 2 - lb0)) / 2 / 30)
    assert rep["cells"]["y"]["ends"]["round_cap"] == 1 and rep["cells"]["y"]["dropped"] == 1


def test_death_hp_and_arm_names(tmp_path: Path) -> None:
    set_path, sha = _write_set(tmp_path, [_fight("a", hp=40, dt=40, died=True)])
    path = _write_played(tmp_path, "file_stem", sha, [_cell("a", 0, end="died", dmg_c=50)])
    out = tmp_path / "r.json"
    argv = ["--set", set_path, f"mine={path}", "--death-hp", "30", "--json", str(out)]
    assert R.main(argv) == 0
    rep = json.loads(out.read_text())
    assert [a["name"] for a in rep["arms"]] == ["mine"]
    ov = rep["human_gap"]["mine"]["quit_excluded"]["overall"]
    assert ov["pilot"]["L"] == pytest.approx(40 + 30) and ov["human"]["L"] == pytest.approx(70)
    assert R._arm_spec(path) == ("file_stem", path)


def test_scored_cell_without_dmg_c_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    set_path, sha = _write_set(tmp_path, [_fight("a")])
    bad = _cell("a", 0)
    bad["rep"]["dmg_c"] = None
    path = _write_played(tmp_path, "x", sha, [bad])
    assert R.main(["--set", set_path, path]) == 2
    assert "has no dmg_c" in capsys.readouterr().err


def test_text_report_sections(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _two_arm_bench(tmp_path)
    text = capsys.readouterr().out
    for section in (
        "== cells",
        "== pilot vs human",
        "== arm vs arm",
        "== pro gap via the baseline",
        "== cost",
        "== splits",
        "per-player gap",
        "per-encounter gap",
    ):
        assert section in text


def test_expected_cells_follow_the_header_selection(tmp_path: Path) -> None:
    """A run on part of the set expects cells for its planned fights only."""
    fights = [
        _fight("a", room="boss"),
        _fight("b", room="elite", run="r2"),
        _fight("c", room="boss", run="r3"),
        _fight("d", room="boss", run="r4"),
    ]
    set_path, sha = _write_set(tmp_path, fights)
    fs = R.load_set(set_path)

    def sel(planned: list[str], **kw: Any) -> dict[str, Any]:
        return {
            "keys": None,
            "room": None,
            "limit": None,
            "fights_sha256": hashlib.sha256("\n".join(planned).encode()).hexdigest(),
            **kw,
        }

    assert R.selected_keys({}, fs) is None  # no select: the whole set
    assert R.selected_keys({"select": sel(["a", "c"], room=["boss"], limit=2)}, fs) == ["a", "c"]
    assert R.selected_keys({"select": sel(["b", "d"], keys=["d", "b"])}, fs) == ["b", "d"]
    with pytest.raises(R.BenchError, match="does not hash"):
        R.selected_keys({"select": sel(["a"], room=["boss"])}, fs)
    with pytest.raises(R.BenchError, match="does not have"):
        R.selected_keys({"select": sel(["z"], keys=["z"])}, fs)

    path = _write_played(tmp_path, "x", sha, [_cell(k, r) for k in ("b", "d") for r in (0, 1)])
    text = Path(path).read_text(encoding="utf-8").splitlines()
    head = json.loads(text[0])
    head["header"]["select"] = sel(["b", "d"], keys=["b", "d"])
    Path(path).write_text("\n".join([json.dumps(head), *text[1:]]) + "\n", encoding="utf-8")
    out = tmp_path / "r.json"
    assert R.main(["--set", set_path, path, "--json", str(out), "--boot", "99"]) == 0
    rep = json.loads(out.read_text())
    assert rep["cells"]["x"]["expected"] == 4 and rep["cells"]["x"]["dropped"] == 0
    assert (rep["set"]["fights"], rep["set"]["fights_in_set"]) == (2, 4)


def test_loss_counts_the_unlogged_hp_loss() -> None:
    """`dmg_unlogged` (Thorns, Inferno, ...) is part of the C# `DamageTaken`."""
    f = _fight("k", hp=30, dt=12)
    rep = _cell("k", 0, dmg_c=10)["rep"] | {"dmg_unlogged": 4}
    assert R.rep_loss(rep, f, 65) == 14
    assert R.rep_metrics(rep, f, 65)["dmg"] == 14
    rep["dmg_unlogged"] = 25  # clamped at hp_before like the human's
    assert R.rep_loss(rep, f, 65) == 30
