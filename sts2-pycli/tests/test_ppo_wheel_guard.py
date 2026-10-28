"""Gates for `ppo.wheel_guard`: training runs only on a fresh PGO `sts2_rs`,
and stays on the wheel it was checked on.

The user asked for every training run to use the PGO wheel. These pin the
guard without a PGO build: `wheel_build_info` / `wheel_staleness` are
monkeypatched (or a fake `sts2_rs` module is injected), so the suite stays
runnable on the plain worktree wheel it always has.

Covered: a fresh PGO wheel passes; a plain wheel is refused with the fix
named; the escape hatch (flag or env) admits a plain wheel loudly and records
`wheel_pgo=false`, and leaves a PGO wheel recorded as PGO; a stale wheel, and
pgo_wheel.sh's staged, unverified or instrumented builds, are refused even with
the escape hatch; both entry points (`ppo.trainer`'s CLI, `ppo.screen`) check
and pin before anything runs, while `train()` itself never checks or pins. The
pin: a content-addressed private copy that spawn workers import, a re-exec
onto it, a refusal when a pinned run imports anything else, and no pin (only
the check) where there is no exec (Windows). The per-iteration backstop: every
worker reports the build it ran, and `train()` aborts (naming how to resume)
when one differs from the run's, while an iteration with nothing to check
records a null wheel; a resume on another build says so. The screen's evals:
each row carries the build its run loaded, a fresh eval on another build
aborts, and reused arms and evals are listed in screen.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from ppo import wheel_guard
from ppo.wheel_guard import (
    ALLOW_PLAIN_ENV,
    FIX,
    IDENTITY_KEYS,
    PIN_DIRNAME,
    PIN_ENV,
    check_training_wheel,
    engine_change,
    pin_of,
    pin_wheel,
    prepare_training_wheel,
    python_source_digest,
    source_digest,
    wheel_identity,
    worker_source_mismatch,
    worker_wheel_mismatch,
)

PGO = {
    "pgo": True,
    "pgo_instrumented": False,
    "profdata_sha256": "ab" * 32,
    "target_cpu": None,
    "src_hash": "0123456789abcdef",
    "build_git": "deadbeef",
    "so_path": "/venv/site-packages/sts2_rs/sts2_rs.abi3.so",
}
PLAIN = {**PGO, "pgo": False, "profdata_sha256": None}


@pytest.fixture(autouse=True)
def _no_ambient_env(monkeypatch):
    """A developer shell with the opt-out (or a pin) exported must not turn the
    refusal tests green. setenv-then-delenv makes monkeypatch restore the
    original state even when the code under test sets the variable itself."""
    for var in (ALLOW_PLAIN_ENV, PIN_ENV):
        monkeypatch.setenv(var, "0")
        monkeypatch.delenv(var)


def _wheel(monkeypatch, info: dict, stale: str | None = None) -> None:
    monkeypatch.setattr(wheel_guard, "wheel_build_info", lambda: dict(info))
    monkeypatch.setattr(wheel_guard, "wheel_staleness", lambda rs_root: stale)


def test_a_fresh_pgo_wheel_passes_and_is_recorded(monkeypatch, tmp_path, capsys):
    _wheel(monkeypatch, PGO)
    rec = check_training_wheel(rs_root=tmp_path)
    assert rec["wheel_pgo"] is True
    assert rec["allow_plain_wheel"] is False
    assert rec["profdata_sha256"] == PGO["profdata_sha256"]
    assert rec["src_hash"] == PGO["src_hash"]
    assert capsys.readouterr().err == ""
    # The record is JSON-able: it goes into checkpoints, wandb and screen.json.
    json.dumps(rec)


def _pgo_script(rs_root: Path) -> None:
    """The checkout `just sim-engine-pgo` needs: sts2-rs/scripts/pgo_wheel.sh."""
    (rs_root / "scripts").mkdir(parents=True, exist_ok=True)
    (rs_root / "scripts" / "pgo_wheel.sh").write_text("#!/usr/bin/env bash\n")


def test_a_plain_wheel_is_refused_and_the_fix_is_named(monkeypatch, tmp_path):
    _wheel(monkeypatch, PLAIN)
    _pgo_script(tmp_path)
    with pytest.raises(SystemExit) as exc:
        check_training_wheel(rs_root=tmp_path)
    msg = str(exc.value.code)
    assert "plain build" in msg
    if os.name == "nt":
        # pgo_wheel.sh is bash-only: on Windows the refusal says so instead of
        # naming a build that cannot run there.
        assert "Windows" in msg
    else:
        assert FIX in msg
    # The host that cannot build PGO needs to be told its way out.
    assert "--allow-plain-wheel" in msg
    assert ALLOW_PLAIN_ENV in msg


@pytest.mark.parametrize("why", ["no-script", "windows"])
def test_a_host_that_cannot_build_pgo_is_sent_to_the_opt_out(monkeypatch, tmp_path, why):
    """No pgo_wheel.sh in the checkout (an export without it), or Windows,
    where the bash recipe cannot run: naming `just sim-engine-pgo` as the fix
    would send the user nowhere."""
    _wheel(monkeypatch, PLAIN)
    if why == "windows":
        _pgo_script(tmp_path)
        monkeypatch.setattr(wheel_guard.os, "name", "nt")
    with pytest.raises(SystemExit) as exc:
        check_training_wheel(rs_root=tmp_path)
    msg = str(exc.value.code)
    assert "This host cannot build it" in msg
    assert "--allow-plain-wheel" in msg and ALLOW_PLAIN_ENV in msg
    assert FIX not in msg


@pytest.mark.parametrize("via", ["flag", "env"])
def test_the_escape_hatch_admits_a_plain_wheel_loudly(monkeypatch, tmp_path, capsys, via):
    _wheel(monkeypatch, PLAIN)
    if via == "env":
        monkeypatch.setenv(ALLOW_PLAIN_ENV, "1")
    rec = check_training_wheel(allow_plain=via == "flag", rs_root=tmp_path)
    assert rec["wheel_pgo"] is False
    assert rec["allow_plain_wheel"] is True
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "wheel_pgo=false" in err
    assert FIX in err


@pytest.mark.parametrize("via", ["flag", "env"])
def test_the_escape_hatch_on_a_pgo_wheel_still_records_pgo(monkeypatch, tmp_path, capsys, via):
    """The opt-out only admits a plain wheel; it does not make a PGO run plain.
    `wheel_pgo` says what the run got, not what it was allowed."""
    _wheel(monkeypatch, PGO)
    if via == "env":
        monkeypatch.setenv(ALLOW_PLAIN_ENV, "1")
    rec = check_training_wheel(allow_plain=via == "flag", rs_root=tmp_path)
    assert rec["wheel_pgo"] is True
    assert rec["allow_plain_wheel"] is True
    assert capsys.readouterr().err == ""


def test_the_quiet_pre_pin_check_still_refuses_but_does_not_warn(monkeypatch, tmp_path, capsys):
    _wheel(monkeypatch, PLAIN)
    rec = check_training_wheel(allow_plain=True, rs_root=tmp_path, warn=False)
    assert rec["wheel_pgo"] is False
    assert capsys.readouterr().err == ""
    with pytest.raises(SystemExit):
        check_training_wheel(rs_root=tmp_path, warn=False)


@pytest.mark.parametrize("info", [PGO, PLAIN], ids=["pgo", "plain"])
@pytest.mark.parametrize("allow_plain", [False, True])
def test_a_stale_wheel_is_refused_even_with_the_escape_hatch(
    monkeypatch, tmp_path, info, allow_plain
):
    _wheel(monkeypatch, info, stale="loaded .so src_hash = aaaa, on-disk source hash = bbbb")
    with pytest.raises(SystemExit) as exc:
        check_training_wheel(allow_plain=allow_plain, rs_root=tmp_path)
    msg = str(exc.value.code)
    assert "not compiled from the current source" in msg
    assert "aaaa" in msg and "bbbb" in msg
    assert FIX in msg


@pytest.mark.parametrize(
    "info",
    [
        {**PGO, "so_path": "/rs/target-pgo-wheel-unverified/sts2_rs.abi3.so"},
        {**PGO, "so_path": "/rs/target-pgo-wheel-stage/use/sts2_rs/sts2_rs.abi3.so"},
        {**PLAIN, "pgo_instrumented": True},
    ],
    ids=["skip-identity", "staged", "instrumented"],
)
@pytest.mark.parametrize("allow_plain", [False, True])
def test_pgo_wheel_sh_s_own_builds_are_refused(monkeypatch, tmp_path, info, allow_plain):
    """A PGO module that has not passed (or skipped) pgo_wheel.sh's identity
    check reports pgo too; only the venv install it puts in after the check
    may train. The instrumented build would write .profraw from every worker."""
    _wheel(monkeypatch, info)
    with pytest.raises(SystemExit, match=r"one of pgo_wheel\.sh's own builds"):
        check_training_wheel(allow_plain=allow_plain, rs_root=tmp_path)


def test_an_ambient_opt_out_spelled_off_does_not_admit_a_plain_wheel(monkeypatch, tmp_path):
    _wheel(monkeypatch, PLAIN)
    monkeypatch.setenv(ALLOW_PLAIN_ENV, "0")
    with pytest.raises(SystemExit):
        check_training_wheel(rs_root=tmp_path)


def _fake_sts2_rs(monkeypatch, **attrs) -> None:
    mod = types.ModuleType("sts2_rs")
    mod.__file__ = "/fake/sts2_rs.abi3.so"
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, "sts2_rs", mod)


def test_staleness_is_the_baked_vs_recomputed_source_hash(monkeypatch, tmp_path):
    (tmp_path / "crates").mkdir()
    seen = []

    def recompute(root):
        seen.append(root)
        return "same"

    _fake_sts2_rs(monkeypatch, src_hash=lambda: "same", recompute_src_hash=recompute)
    assert wheel_guard.wheel_staleness(tmp_path) is None
    assert seen == [str(tmp_path)]

    _fake_sts2_rs(monkeypatch, src_hash=lambda: "old", recompute_src_hash=lambda root: "new")
    reason = wheel_guard.wheel_staleness(tmp_path)
    assert reason is not None and "old" in reason and "new" in reason


def test_staleness_fails_closed_when_it_cannot_be_shown(monkeypatch, tmp_path):
    # No source tree to compare against: refusing beats training blind.
    _fake_sts2_rs(monkeypatch, src_hash=lambda: "x", recompute_src_hash=lambda root: "x")
    assert "no sts2-rs source" in wheel_guard.wheel_staleness(tmp_path / "missing")
    # A wheel that predates the stamp cannot be shown fresh either.
    (tmp_path / "crates").mkdir()
    _fake_sts2_rs(monkeypatch)
    assert "predates" in wheel_guard.wheel_staleness(tmp_path)


def test_a_wheel_without_build_info_counts_as_plain(monkeypatch):
    _fake_sts2_rs(monkeypatch)
    info = wheel_guard.wheel_build_info()
    assert info["pgo"] is False
    assert info["so_path"] == "/fake/sts2_rs.abi3.so"

    _fake_sts2_rs(monkeypatch, build_info=lambda: dict(PGO, so_path=None))
    assert wheel_guard.wheel_build_info()["pgo"] is True


def test_the_banner_names_the_wheel():
    assert "PGO" in wheel_guard.wheel_banner({**PGO, "wheel_pgo": True})
    plain = wheel_guard.wheel_banner({**PLAIN, "wheel_pgo": False})
    assert "PLAIN" in plain and "wheel_pgo=false" in plain
    assert "pinned at /run/pin" in wheel_guard.wheel_banner(
        {**PGO, "wheel_pgo": True, "pinned": "/run/pin"}
    )


# ── the build a process ran, and the per-iteration check ─────────────────


def test_the_identity_is_the_build_and_the_file_not_the_commit():
    ident = wheel_identity(PGO)
    assert tuple(ident) == IDENTITY_KEYS
    assert ident["so_path"] == PGO["so_path"]
    # Two builds of one engine source at different commits are one engine.
    assert wheel_identity({**PGO, "build_git": "other"}) == ident
    for key in ("pgo", "profdata_sha256", "src_hash", "so_path"):
        assert wheel_identity({**PGO, key: "changed"}) != ident


def test_worker_wheel_mismatch():
    run = wheel_identity(PGO)
    ran = {"wheel": dict(run), "transitions": [1]}
    assert worker_wheel_mismatch([ran, ran], run) is None
    # A run that never reached the engine reports nothing and is not judged.
    never = {"seed": 3, "outcome": "timeout", "won": False}
    assert worker_wheel_mismatch([ran, never], run) is None

    plain = {"wheel": wheel_identity(PLAIN), "transitions": [1]}
    why = worker_wheel_mismatch([ran, plain, plain], run)
    assert why is not None and "2 run(s)" in why and '"pgo": false' in why
    # Even a run with no transitions is judged if it reports a different build.
    assert worker_wheel_mismatch([{"wheel": wheel_identity(PLAIN)}], run) is not None
    # A training row cannot skip the check by not reporting.
    assert worker_wheel_mismatch([{"transitions": [1]}], run) is not None


def test_engine_change_ignores_where_the_file_lives():
    assert engine_change(PGO, {**PGO, "so_path": "/run/sts2_rs-pin/x/sts2_rs.abi3.so"}) is None
    assert engine_change(None, PGO) is None and engine_change(PGO, None) is None
    moved = engine_change(PGO, {**PGO, "src_hash": "fedcba9876543210", "build_git": "x"})
    assert moved == "src_hash 0123456789abcdef -> fedcba9876543210"
    assert "pgo True -> False" in str(engine_change(PGO, PLAIN))


def test_play_one_reports_the_wheel_it_ran():
    from ppo import workers

    row = workers.play_one(seed=7, policy="random", max_steps=3)
    assert row["wheel"] == wheel_identity()
    json.dumps(row["wheel"])
    assert row["py_src"] == python_source_digest() == source_digest()


def _cfg(save, **kw):
    from ppo.trainer import PPOConfig

    return PPOConfig(
        **{
            "character": "Ironclad",
            "iterations": 1,
            "games_per_iter": 2,
            "sims": 4,
            "boss_sims": 4,
            "procs": 2,
            "device": "cpu",
            "plasticity_samples": 0,
            "save_every": 1,
            "save_path": str(save),
            **kw,
        }
    )


@pytest.mark.parametrize("async_collect", [False, True], ids=["sync", "async"])
def test_train_aborts_when_the_workers_ran_another_wheel(monkeypatch, tmp_path, async_collect):
    """`train(cfg)` with no record (the pytest path: no pin) compares the
    workers against its own import, not against PGO-ness, and aborts before
    training on the batch."""
    from ppo import trainer

    other = {**wheel_identity(), "so_path": "/elsewhere/sts2_rs.abi3.so"}
    rows = [{"seed": 1, "outcome": "died", "won": False, "transitions": [], "wheel": other}]
    monkeypatch.setattr(trainer, "collect_rows", lambda *a, **k: [dict(r) for r in rows])
    save = tmp_path / "run"
    with pytest.raises(SystemExit) as exc:
        trainer.train(_cfg(save, async_collect=async_collect))
    msg = str(exc.value.code)
    assert "ABORTING at iteration 0" in msg
    assert "/elsewhere/sts2_rs.abi3.so" in msg
    assert "no checkpoint was saved yet" in msg
    assert not os.path.exists(f"{save}.stats.jsonl")


def test_an_iteration_no_run_reached_the_engine_records_a_null_wheel(monkeypatch, tmp_path):
    """Nothing to check (every run an error or a timeout, as a deleted pin or a
    broken worker interpreter would look): the stats row says so instead of
    dropping the fields."""
    from ppo import trainer

    rows = [{"seed": s, "outcome": "error", "won": False} for s in (1, 2)]
    monkeypatch.setattr(trainer, "collect_rows", lambda *a, **k: [dict(r) for r in rows])
    save = tmp_path / "run"
    trainer.train(_cfg(save))
    with open(f"{save}.stats.jsonl", encoding="utf-8") as fh:
        [row] = [json.loads(line) for line in fh]
    assert row["errors"] == 2
    assert row["wheel"] is None and row["wheel_pgo"] is None


def test_the_abort_names_the_newest_checkpoint_to_resume_from(tmp_path):
    from ppo.trainer import resume_hint

    save = str(tmp_path / "run")
    assert "no checkpoint" in resume_hint(save, 5)
    for k in (2, 4, 7):
        Path(f"{save}.iter_{k}.pt").touch()
    hint = resume_hint(save, 5)
    assert "--resume-from-iter 4" in hint and f"{save}.iter_4.pt" in hint
    # With --save-every 1, iteration 4's update saved iter_5.pt, the state an
    # aborted iteration 5 started from: resume there, redoing only iteration 5.
    Path(f"{save}.iter_5.pt").touch()
    assert "--resume-from-iter 5" in resume_hint(save, 5)


def test_a_resume_on_another_build_says_so(monkeypatch, tmp_path, capsys):
    from ppo import trainer

    rows = [{"seed": 1, "outcome": "error", "won": False}]
    monkeypatch.setattr(trainer, "collect_rows", lambda *a, **k: [dict(r) for r in rows])
    save = tmp_path / "run"
    first = {**PGO, "wheel_pgo": True, "allow_plain_wheel": False}
    trainer.train(_cfg(save), wheel=first)
    resume = _cfg(save, iterations=2, resume_from_iter=1)
    capsys.readouterr()
    # The same build pinned elsewhere is the same engine.
    trainer.train(resume, wheel={**first, "so_path": "/elsewhere/sts2_rs.abi3.so"})
    assert "another sts2_rs build" not in capsys.readouterr().out
    trainer.train(resume, wheel={**first, "src_hash": "fedcba9876543210"})
    out = capsys.readouterr().out
    assert "WARNING: iteration 1's checkpoint was trained on another sts2_rs build" in out
    assert "src_hash 0123456789abcdef -> fedcba9876543210" in out
    assert "first-party Python" not in out
    # A resume on other first-party Python is a source change, not a build one.
    monkeypatch.setattr(trainer, "python_source_digest", lambda: "feedfacefeedface")
    trainer.train(resume, wheel=first)
    out = capsys.readouterr().out
    assert "another sts2_rs build" not in out
    assert "trained with other first-party Python" in out
    assert "-> feedfacefeedface" in out


# ── first-party Python under the run ─────────────────────────────────────


def _py_tree(root: Path) -> Path:
    for rel in ("ppo/__init__.py", "ppo/env.py", "ppo/sub/x.py", "champions.py", "other.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# {rel}\n")
    return root


def test_the_source_digest_covers_what_the_workers_import(tmp_path):
    root = _py_tree(tmp_path)
    base = source_digest(root)
    assert source_digest(root) == base
    for rel in ("ppo/env.py", "ppo/sub/x.py", "champions.py"):
        before = (root / rel).read_text()
        (root / rel).write_text(before + "x = 1\n")
        assert source_digest(root) != base, rel
        (root / rel).write_text(before)
    assert source_digest(root) == base
    # A new module counts; bytecode and modules outside the set do not.
    (root / "ppo" / "new.py").write_text("")
    assert source_digest(root) != base
    (root / "ppo" / "new.py").unlink()
    (root / "ppo" / "__pycache__").mkdir()
    (root / "ppo" / "__pycache__" / "env.cpython-313.pyc").write_bytes(b"\0")
    (root / "other.py").write_text("changed")
    assert source_digest(root) == base


def test_worker_source_mismatch():
    run = "0123456789abcdef"
    ran = {"py_src": run, "transitions": [1]}
    assert worker_source_mismatch([ran, ran], run) is None
    # An error row that never reached the engine reports nothing.
    assert worker_source_mismatch([ran, {"outcome": "error"}], run) is None
    why = worker_source_mismatch([ran, {"py_src": "fedcba9876543210", "transitions": [1]}], run)
    assert why is not None and "1 run(s) ran py_src fedcba9876543210" in why
    # A training row with no report is a mismatch, not a pass.
    assert worker_source_mismatch([{"transitions": [1]}], run) is not None


@pytest.mark.parametrize("async_collect", [False, True], ids=["sync", "async"])
def test_train_aborts_when_the_workers_ran_other_python(monkeypatch, tmp_path, async_collect):
    """A pull mid-campaign: the extension is pinned, but the workers import the
    new `python/ppo` from disk. The abort names it a source change."""
    from ppo import trainer

    rows = [
        {
            "seed": 1,
            "outcome": "died",
            "won": False,
            "transitions": [],
            "wheel": wheel_identity(),
            "py_src": "feedfacefeedface",
        }
    ]
    monkeypatch.setattr(trainer, "collect_rows", lambda *a, **k: [dict(r) for r in rows])
    save = tmp_path / "run"
    with pytest.raises(SystemExit) as exc:
        trainer.train(_cfg(save, async_collect=async_collect))
    msg = str(exc.value.code)
    assert "ABORTING at iteration 0" in msg
    assert "SOURCE change" in msg and "not an sts2_rs build change" in msg
    assert "feedfacefeedface" in msg
    assert not os.path.exists(f"{save}.stats.jsonl")


@pytest.mark.parametrize("with_record", [False, True], ids=["train(cfg)", "cli-record"])
def test_a_training_run_records_what_its_workers_ran(tmp_path, with_record):
    """End to end through one real iteration on whatever wheel the venv has
    (`train()` never checks it): the workers' report has to reach the files a
    campaign is read from, and the per-iteration check must pass unpinned."""
    import torch
    from ppo.trainer import train

    save = tmp_path / "run"
    me = wheel_identity()
    # The record's `wheel_pgo` says the opposite of what the workers run, on a
    # plain venv and a PGO one alike, so a stats row copied from it shows.
    rec = {**wheel_guard.wheel_build_info(), "wheel_pgo": not me["pgo"], "allow_plain_wheel": True}
    train(_cfg(save), wheel=rec if with_record else None)
    with open(f"{save}.stats.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    assert [r["wheel"] for r in rows] == [me]
    # What the workers ran, not what the record or the opt-out said.
    assert [r["wheel_pgo"] for r in rows] == [bool(me["pgo"])]
    ck = torch.load(f"{save}.iter_1.pt", map_location="cpu", weights_only=True)
    assert ck.get("wheel") == (rec if with_record else None)
    assert [r["py_src"] for r in rows] == [python_source_digest()]
    assert ck["py_src"] == python_source_digest()


# ── pinning ──────────────────────────────────────────────────────────────

_NO_PIN_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows runs do not pin (no exec; PIN_SUPPORTED)"
)


@_NO_PIN_ON_WINDOWS
def test_pin_wheel_makes_a_content_addressed_copy(tmp_path):
    so = Path(wheel_guard.wheel_build_info()["so_path"])
    pin = pin_wheel(tmp_path / PIN_DIRNAME)
    copy = pin / "sts2_rs" / so.name
    assert (pin / "sts2_rs" / "__init__.py").is_file() and copy.is_file()
    record = json.loads((pin / "pin.json").read_text())
    assert pin.name == record["sha256"][:16]
    assert pin_of(copy) == pin
    # The venv's own install is not a pin.
    assert pin_of(so) is None

    # Pinning the same bytes again reuses the dir and leaves no temp dir.
    assert pin_wheel(tmp_path / PIN_DIRNAME) == pin
    assert sorted(p.name for p in (tmp_path / PIN_DIRNAME).iterdir()) == [pin.name]

    # A pin whose file no longer has the recorded hash is no pin, and is not
    # silently reused.
    with open(copy, "ab") as f:
        f.write(b"tampered")
    assert pin_of(copy) is None
    with pytest.raises(SystemExit, match="is not a pin"):
        pin_wheel(tmp_path / PIN_DIRNAME)


def _fake_installed(monkeypatch, site: Path) -> Path:
    """A maturin-shaped `sts2_rs` package under `site`, loaded as the process's
    `sts2_rs`; returns the package dir."""
    pkg = site / "sts2_rs"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sts2_rs.abi3.so").write_bytes(b"the extension")
    mod = types.ModuleType("sts2_rs")
    mod.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "sts2_rs", mod)
    return pkg


@_NO_PIN_ON_WINDOWS
def test_a_pin_leaves_out_temp_files_a_killed_install_left_behind(monkeypatch, tmp_path):
    """pgo_wheel.sh copies the .so to `<so>.pgo-tmp` (and its rollback to
    `<so>.tmp`) before a rename; a SIGKILL in between leaves the file in the
    package dir. A pin needs neither."""
    pkg = _fake_installed(monkeypatch, tmp_path / "site")
    (pkg / "sts2_rs.abi3.so.pgo-tmp").write_bytes(b"half")
    (pkg / "sts2_rs.abi3.so.tmp").write_bytes(b"half")
    pin = pin_wheel(tmp_path / PIN_DIRNAME)
    assert sorted(p.name for p in (pin / "sts2_rs").iterdir()) == [
        "__init__.py",
        "sts2_rs.abi3.so",
    ]


@_NO_PIN_ON_WINDOWS
def test_a_pin_another_user_created_is_not_reused(monkeypatch, tmp_path):
    """The reuse check hashes only the .so; whoever created the dir also chose
    its `__init__.py`, which runs first. A pin root shared with another user
    (a world-writable `--out`) must not hand the run their copy."""
    _fake_installed(monkeypatch, tmp_path / "site")
    root = tmp_path / PIN_DIRNAME
    pin = pin_wheel(root)
    assert pin_wheel(root) == pin
    monkeypatch.setattr(wheel_guard.os, "getuid", lambda: pin.stat().st_uid + 1)
    with pytest.raises(SystemExit, match="another user created"):
        pin_wheel(root)


@_NO_PIN_ON_WINDOWS
def test_spawn_workers_import_the_pinned_copy(monkeypatch, tmp_path):
    """The mechanism the pin rests on: a spawn child takes the parent's
    `sys.path`, so with the pin in front it imports the pinned copy, not the
    venv's (which this test process has already loaded)."""
    pin = pin_wheel(tmp_path / PIN_DIRNAME)
    monkeypatch.syspath_prepend(str(pin))
    with mp.get_context("spawn").Pool(1) as pool:
        child = pool.apply(wheel_identity)
    assert child["so_path"].startswith(str(pin) + os.sep)
    assert pin_of(child["so_path"]) == pin
    parent = wheel_identity()
    assert {k: v for k, v in child.items() if k != "so_path"} == {
        k: v for k, v in parent.items() if k != "so_path"
    }


class _Execed(Exception):
    pass


def _prepare_env(
    monkeypatch, info, *, pinned=None, argv=("-m", "ppo.trainer", "--iterations", "3")
):
    calls: dict = {"checks": [], "pins": [], "exec": None}

    def check(**kw):
        calls["checks"].append(kw)
        if not info["pgo"] and not kw.get("allow_plain"):
            raise SystemExit("refused")
        return {**info, "wheel_pgo": info["pgo"], "allow_plain_wheel": kw.get("allow_plain")}

    def pin(root):
        calls["pins"].append(root)
        return Path("/runs/sts2_rs-pin/0123456789abcdef")

    def execve(path, args, env):
        calls["exec"] = (path, args, env)
        raise _Execed

    # The pin and the exec are faked, so the pinning path runs on any OS.
    monkeypatch.setattr(wheel_guard, "PIN_SUPPORTED", True)
    monkeypatch.setattr(wheel_guard, "wheel_build_info", lambda: dict(info))
    monkeypatch.setattr(wheel_guard, "pin_of", lambda so: pinned)
    monkeypatch.setattr(wheel_guard, "check_training_wheel", check)
    monkeypatch.setattr(wheel_guard, "pin_wheel", pin)
    monkeypatch.setattr(os, "execve", execve)
    monkeypatch.setattr(sys, "orig_argv", ["/venv/bin/python", *argv])
    return calls


def test_an_unpinned_launch_checks_quietly_pins_and_restarts_on_the_pin(monkeypatch):
    calls = _prepare_env(monkeypatch, PGO)
    monkeypatch.setenv("PYTHONPATH", "python")
    with pytest.raises(_Execed):
        prepare_training_wheel(
            pin_root="/runs/sts2_rs-pin", allow_plain=False, main_module="ppo.trainer"
        )
    assert calls["checks"] == [{"allow_plain": False, "warn": False}]
    assert calls["pins"] == ["/runs/sts2_rs-pin"]
    path, args, env = calls["exec"]
    assert path == sys.executable
    assert args == [sys.executable, "-m", "ppo.trainer", "--iterations", "3"]
    pin = str(Path("/runs/sts2_rs-pin/0123456789abcdef"))
    assert env["PYTHONPATH"] == os.pathsep.join([pin, "python"])
    assert env[PIN_ENV] == pin


def test_pinning_is_on_everywhere_but_windows():
    assert wheel_guard.PIN_SUPPORTED is (sys.platform != "win32")


@pytest.mark.parametrize("info", [PGO, PLAIN], ids=["pgo", "plain"])
def test_without_exec_a_launch_is_checked_loudly_and_not_pinned(monkeypatch, capsys, info):
    """Windows (`PIN_SUPPORTED` false): the run starts on the venv's import,
    with the full check, and records that it is unpinned; the opt-out still
    admits a plain wheel there. The per-iteration check is its backstop."""
    calls = _prepare_env(monkeypatch, info)
    monkeypatch.setattr(wheel_guard, "PIN_SUPPORTED", False)
    rec = prepare_training_wheel(pin_root="/runs/p", allow_plain=True, main_module="ppo.trainer")
    assert rec["pinned"] is None and rec["wheel_pgo"] is info["pgo"]
    assert calls["checks"] == [{"allow_plain": True}]
    assert calls["pins"] == [] and calls["exec"] is None
    assert "not pinned" in capsys.readouterr().err
    if not info["pgo"]:
        # Without the opt-out, a plain wheel is still refused.
        with pytest.raises(SystemExit, match="refused"):
            prepare_training_wheel(pin_root="/runs/p", allow_plain=False, main_module="ppo.trainer")


def test_a_refused_wheel_is_never_pinned(monkeypatch):
    calls = _prepare_env(monkeypatch, PLAIN)
    with pytest.raises(SystemExit, match="refused"):
        prepare_training_wheel(pin_root="/runs/p", allow_plain=False, main_module="ppo.trainer")
    assert calls["pins"] == [] and calls["exec"] is None


def test_a_pinned_launch_checks_loudly_and_records_the_pin(monkeypatch):
    pin = Path("/runs/sts2_rs-pin/0123456789abcdef")
    calls = _prepare_env(monkeypatch, PGO, pinned=pin)
    monkeypatch.setenv(PIN_ENV, str(pin))
    rec = prepare_training_wheel(pin_root="/runs/p", allow_plain=True, main_module="ppo.trainer")
    assert rec["pinned"] == str(pin) and rec["wheel_pgo"] is True
    assert calls["checks"] == [{"allow_plain": True}]
    assert calls["pins"] == [] and calls["exec"] is None


@pytest.mark.parametrize("pinned", [None, Path("/runs/other-pin")], ids=["venv", "other-pin"])
def test_a_pinned_run_refuses_an_import_from_anywhere_else(monkeypatch, pinned):
    calls = _prepare_env(monkeypatch, PGO, pinned=pinned)
    monkeypatch.setenv(PIN_ENV, "/runs/sts2_rs-pin/0123456789abcdef")
    with pytest.raises(SystemExit, match="pinned to /runs/sts2_rs-pin/0123456789abcdef"):
        prepare_training_wheel(pin_root="/runs/p", allow_plain=False, main_module="ppo.trainer")
    assert calls["pins"] == [] and calls["exec"] is None


def test_only_its_own_module_is_restarted(monkeypatch):
    """A process that was not started as `-m <module>` (pytest calling the CLI
    function) must never be exec'd into a re-run of itself."""
    calls = _prepare_env(monkeypatch, PGO, argv=("-m", "pytest", "ppo.trainer"))
    with pytest.raises(SystemExit, match=r"not started as `python -m ppo\.trainer`"):
        prepare_training_wheel(pin_root="/runs/p", allow_plain=False, main_module="ppo.trainer")
    assert calls["exec"] is None


# ── entry points ──────────────────────────────────────────────────────────


def test_the_trainer_cli_pins_and_checks_before_it_trains(monkeypatch, tmp_path):
    from ppo import trainer

    calls = []
    monkeypatch.setattr(trainer, "train", lambda cfg, wheel=None: calls.append((cfg, wheel)))

    def refuse(**kw):
        raise SystemExit("refused")

    monkeypatch.setattr(trainer, "prepare_training_wheel", refuse)
    with pytest.raises(SystemExit, match="refused"):
        trainer._cli(["--iterations", "3"])
    assert calls == []

    seen = []

    def accept(**kw):
        seen.append(kw)
        return {**PLAIN, "wheel_pgo": False, "allow_plain_wheel": True, "pinned": "p"}

    monkeypatch.setattr(trainer, "prepare_training_wheel", accept)
    save = tmp_path / "runs" / "ic"
    trainer._cli(["--iterations", "3", "--allow-plain-wheel", "--save-path", str(save)])
    # The pin lives next to the checkpoints.
    assert seen == [
        {
            "pin_root": str(tmp_path / "runs" / PIN_DIRNAME),
            "allow_plain": True,
            "main_module": "ppo.trainer",
        }
    ]
    [(cfg, wheel)] = calls
    assert cfg.iterations == 3
    # The flag is not a PPOConfig field: the run records what it got instead.
    assert not hasattr(cfg, "allow_plain_wheel")
    assert wheel["wheel_pgo"] is False


def _screen(monkeypatch, record=None):
    from ppo import screen

    launched = []
    prepared = []
    monkeypatch.setattr(screen, "run_arm", lambda *a, **k: launched.append(a) or "ckpt")
    monkeypatch.setattr(screen, "eval_arm", lambda *a, **k: evals.append(k) or {})
    monkeypatch.setattr(screen, "report", lambda *a, **k: {})
    evals: list[dict] = []
    screen.evals = evals  # type: ignore[attr-defined]  # what each eval was handed

    def prepare(**kw):
        prepared.append(kw)
        if record is None:
            raise SystemExit("refused")
        return dict(record)

    monkeypatch.setattr(screen, "prepare_training_wheel", prepare)
    return screen, launched, prepared


def test_the_screen_fails_fast_before_launching_an_arm(monkeypatch, tmp_path):
    screen, launched, _ = _screen(monkeypatch)
    with pytest.raises(SystemExit, match="refused"):
        screen.main(["--name", "t", "--out", str(tmp_path / "a"), "--init-ckpt", "x.pt"])
    assert launched == []

    # A dry run launches nothing, so it does not need the PGO wheel.
    assert screen.main(["--name", "t", "--out", str(tmp_path / "b"), "--dry-run"]) == 0


def test_the_screen_pins_once_for_both_arms_under_its_out_dir(monkeypatch, tmp_path):
    screen, launched, prepared = _screen(
        monkeypatch, {**PGO, "wheel_pgo": True, "allow_plain_wheel": False}
    )
    out = tmp_path / "s"
    screen.main(["--name", "t", "--out", str(out), "--init-ckpt", "x.pt"])
    assert prepared == [
        {
            "pin_root": os.path.join(str(out), PIN_DIRNAME),
            "allow_plain": False,
            "main_module": "ppo.screen",
        }
    ]
    assert len(launched) == 2
    # Both evals get the pinned record, or `eval_arm`'s wheel check is off.
    recorded = json.loads((out / "screen.json").read_text())["wheel"]
    assert [k.get("wheel") for k in screen.evals] == [recorded, recorded]
    assert recorded["wheel_pgo"] is True


def test_the_screen_hands_the_opt_out_to_its_trainers(monkeypatch, tmp_path):
    screen, _, _ = _screen(monkeypatch, {**PLAIN, "wheel_pgo": False, "allow_plain_wheel": True})
    out = tmp_path / "s"
    screen.main(["--name", "t", "--out", str(out), "--init-ckpt", "x.pt", "--allow-plain-wheel"])
    # `run_arm` builds each trainer's env from os.environ.
    assert os.environ[ALLOW_PLAIN_ENV] == "1"
    rec = json.loads((out / "screen.json").read_text())
    assert rec["wheel"]["wheel_pgo"] is False


def test_the_screen_does_not_export_the_opt_out_for_a_pgo_wheel(monkeypatch, tmp_path):
    """`--allow-plain-wheel` on a host that did build PGO: the arms must not be
    told plain is fine -- they would then accept a plain wheel that replaced
    the PGO one."""
    screen, _, _ = _screen(monkeypatch, {**PGO, "wheel_pgo": True, "allow_plain_wheel": True})
    out = tmp_path / "s"
    screen.main(["--name", "t", "--out", str(out), "--init-ckpt", "x.pt", "--allow-plain-wheel"])
    assert ALLOW_PLAIN_ENV not in os.environ
    rec = json.loads((out / "screen.json").read_text())
    assert rec["wheel"]["wheel_pgo"] is True
    assert "reused_arms" not in rec["wheel"]


def test_screen_json_says_which_arm_was_reused_and_what_it_ran(monkeypatch, tmp_path):
    screen, _, _ = _screen(monkeypatch, {**PGO, "wheel_pgo": True, "allow_plain_wheel": False})
    out = tmp_path / "s"
    (out / "a").mkdir(parents=True)
    (out / "a" / "main.pt").touch()
    earlier = wheel_identity(PLAIN)
    with open(out / "a" / "main.stats.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"iteration": 0, "wheel": wheel_identity(PGO)}) + "\n")
        f.write(json.dumps({"iteration": 1, "wheel": earlier}) + "\n")
    screen.main(["--name", "t", "--out", str(out), "--init-ckpt", "x.pt"])
    rec = json.loads((out / "screen.json").read_text())
    assert rec["wheel"]["reused_arms"] == {"a": earlier}


def test_screen_json_lists_reused_evals_and_what_they_ran(monkeypatch, tmp_path):
    screen, _, _ = _screen(monkeypatch, {**PGO, "wheel_pgo": True, "allow_plain_wheel": False})
    out = tmp_path / "s"
    out.mkdir()
    earlier = wheel_identity(PLAIN)
    (out / "eval-b.jsonl").write_text(
        json.dumps({"seed": 1, "won": True, "outcome": "won", "wheel": earlier}) + "\n"
    )
    (out / "eval-a.jsonl").write_text(json.dumps({"seed": 1, "won": True}) + "\n")
    screen.main(["--name", "t", "--out", str(out), "--init-ckpt", "x.pt"])
    rec = json.loads((out / "screen.json").read_text())
    # eval-a predates the per-row wheel record.
    assert rec["wheel"]["reused_evals"] == {"a": None, "b": earlier}
    assert "reused_arms" not in rec["wheel"]


@pytest.mark.parametrize("ran", ["pinned", "other", "pre-existing"])
def test_a_fresh_eval_must_have_run_the_pinned_wheel(monkeypatch, tmp_path, ran):
    """The evals inherit the screen's pin; their runs report what they loaded,
    and a fresh eval that ran anything else aborts the screen. An eval file
    from an earlier invocation is reused unchecked (screen.json lists it)."""
    import subprocess

    from ppo import screen

    pinned = {**PGO, "so_path": "/out/sts2_rs-pin/0123/sts2_rs/sts2_rs.abi3.so"}
    reported = wheel_identity(pinned if ran == "pinned" else PGO)
    rows = [
        {"seed": 1, "won": True, "outcome": "won", "wheel": reported},
        {"seed": 2, "won": False, "outcome": "error"},
    ]
    out = tmp_path / "s"
    out.mkdir()
    if ran == "pre-existing":
        (out / "eval-a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def run(cmd, env, check):
        with open(env["SCREEN_OUT"], "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(screen.subprocess, "run", run)
    cfg = {"character": "Ironclad", "ascension_weights": "", "sims": 4, "boss_sims": 4}
    arm = screen.Arm("a", features="")
    record = {**pinned, "wheel_pgo": True}
    if ran == "other":
        with pytest.raises(SystemExit, match="did not run the sts2_rs this screen pinned"):
            screen.eval_arm(arm, "x.pt", cfg, str(out), 2, False, wheel=record)
        # Moved aside, so the next invocation evaluates again (and checks again)
        # instead of reusing the file unchecked.
        assert not (out / "eval-a.jsonl").exists()
        assert len(list(out.glob("eval-a.jsonl.wheel-mismatch-*"))) == 1
        with pytest.raises(SystemExit, match="did not run the sts2_rs this screen pinned"):
            screen.eval_arm(arm, "x.pt", cfg, str(out), 2, False, wheel=record)
    else:
        # The error row is left out of the result: a crash is not a loss.
        assert screen.eval_arm(arm, "x.pt", cfg, str(out), 2, False, wheel=record) == {1: True}


def test_the_eval_snippet_writes_the_wheel_each_run_reported(monkeypatch, tmp_path):
    """What `eval_arm` checks has to reach the eval file: run the snippet's code
    in-process on canned `play_batch` rows."""
    from ppo import screen, workers

    ran = wheel_identity(PGO)
    rows = [
        {"seed": 11, "won": True, "outcome": "won", "final": {"total_floor": 50}, "wheel": ran},
        {"seed": 12, "won": False, "outcome": "error"},
    ]
    monkeypatch.setattr(workers, "play_batch", lambda *a, **k: [dict(r) for r in rows])
    out = tmp_path / "eval-a.jsonl"
    env = {
        "SCREEN_SEEDS": "2",
        "SCREEN_SEED_BASE": "11",
        "SCREEN_ASC_WEIGHTS": "",
        "SCREEN_CHARACTER": "Ironclad",
        "SCREEN_CKPT": "x.pt",
        "SCREEN_SIMS": "4",
        "SCREEN_BOSS_SIMS": "4",
        "SCREEN_COMBAT": "mcts",
        "SCREEN_SURROGATE": "",
        "SCREEN_PROCS": "0",
        "SCREEN_OUT": str(out),
        "SCREEN_PYTHON_DIR": os.path.join(os.path.dirname(__file__), "..", "python"),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(screen._EVAL_SNIPPET, {})
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r.get("wheel") for r in written] == [ran, None]
    assert [r["won"] for r in written] == [True, False]
