"""Background sweep jobs: one thread per sweep driving the process-pool runner.

State lives on disk (spec.json / progress.json / rows.jsonl /
summary.json under results_root/<sweep_id>/), so the API can list and
serve past sweeps across restarts; daemon threads run detached with no
in-memory registry.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from simlab.runner import CancelToken, run_sweep, write_spec
from simlab.specs import SweepSpec

#: How long `_atomic_write_json` keeps retrying a rename the OS refuses because
#: a reader holds the destination open. The contended window is one small
#: `read_text`, i.e. microseconds; a second of retries is far past pathological
#: while still surfacing a real permission problem instead of hanging.
_REPLACE_RETRIES = 50
_REPLACE_BACKOFF_S = 0.02


def _atomic_write_json(path: Path, obj: object) -> None:
    """Atomically write `obj` as JSON to `path` via a temp-file rename.

    A plain write_text can be half-read by a concurrent status poll; writing
    to a sibling .tmp and renaming it (Path.replace wraps os.replace, an atomic
    same-filesystem rename) makes the swap atomic.

    Three corrections, all of which only ever bit on Windows:

    * The temp name is per-writer. A fixed ``progress.tmp`` is shared by the
      progress callback, the watchdog and the final status write, so two of
      them overlapping leave one reading the other's half-written JSON.
    * ``os.replace`` refuses to overwrite a destination that ANY other handle
      has open, and Python's ``open`` does not pass ``FILE_SHARE_DELETE`` — so
      the status endpoint reading ``progress.json`` makes a concurrent write
      raise ``PermissionError``. POSIX renames straight over an open file, so
      this never fired on Linux; here it surfaced as a sweep intermittently
      dying with ``status: "error"`` mid-run (the exception is caught by the
      sweep thread and recorded as the sweep's outcome). Retry, then raise.
    * ``encoding="utf-8"``: the payload carries an ``error`` string built from
      an arbitrary exception message, and the locale default is cp1252.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    for attempt in range(_REPLACE_RETRIES):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(_REPLACE_BACKOFF_S)


def _read_json(path: Path) -> Any:
    """Read a JSON file that `_atomic_write_json` may be swapping right now.

    The reader-side twin of the rename retry: on Windows, opening the
    destination while a writer's ``os.replace`` is mid-swap raises
    ``PermissionError`` too (seen in the public CI on the status poll).
    Retry on the same budget, then raise.
    """
    for attempt in range(_REPLACE_RETRIES):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S)
    raise AssertionError("unreachable")


# Hard ceiling on a single web sweep's wall-clock. A runaway sweep would
# otherwise pin every core indefinitely — costly on always-CPU-allocated hosts
# (the deployed Cloud Run runs --no-cpu-throttling so background workers get full
# CPU). CLI sweeps (`just sim`) bypass JobManager and are intentionally uncapped.
DEFAULT_MAX_RUNTIME_S = 600  # 10 minutes


class JobManager:
    def __init__(
        self,
        results_root: Path,
        jobs: int = 0,
        max_runtime_s: float = 0,
        runner: Callable[..., object] = run_sweep,
    ) -> None:
        """
        Args:
            results_root: Directory sweeps are written under.
            jobs: Worker count; see :meth:`_resolve_jobs`.
            max_runtime_s: Per-sweep wall-clock cap; see
                :meth:`_resolve_max_runtime`.
            runner: The sweep implementation, injectable so a test can supply a
                stub. Sweeps run on DETACHED daemon threads that outlive the
                request (and, in a test, the test), so a stub installed by
                monkeypatching the module-global ``run_sweep`` is visible to
                every other sweep still in flight — it silently no-ops their
                runs, and they then hang at ``status: "running"``. Per-manager
                injection keeps a stub to the manager that asked for it.
        """
        self.results_root = Path(results_root)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.jobs = self._resolve_jobs(jobs)
        self.max_runtime_s = self._resolve_max_runtime(max_runtime_s)
        self._runner = runner
        # Live sweeps only: sweep_id -> CancelToken, so the stop endpoint can
        # reach the detached runner thread's executor. Entries are removed when
        # the run finishes (a restart kills the workers anyway, so nothing
        # cross-process to track). Guarded by _lock for the HTTP/runner threads.
        self._cancels: dict[str, CancelToken] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_jobs(jobs: int) -> int:
        """Resolve the per-sweep worker count.

        An explicit `jobs` (from `create_app(jobs=...)`) wins. Otherwise the
        `SIMLAB_JOBS` env var overrides the default — set it to use all cores
        (`SIMLAB_JOBS=$(nproc)`) when you run one sweep at a time and want full
        CPU, or to a small number when running many concurrent sweeps. With
        neither set we keep the conservative half-cores default: concurrent
        sweeps would otherwise each spawn cpu_count processes and oversubscribe.
        """
        if jobs > 0:
            return jobs
        env = os.environ.get("SIMLAB_JOBS")
        if env:
            try:
                n = int(env)
                if n > 0:
                    return n
            except ValueError:
                pass
        return max(1, (os.cpu_count() or 2) // 2)

    @staticmethod
    def _resolve_max_runtime(max_runtime_s: float) -> float:
        """Resolve the per-sweep wall-clock cap in seconds.

        An explicit `max_runtime_s` (from `create_app`) wins. Otherwise the
        `SIMLAB_MAX_RUNTIME_S` env var overrides the 10-minute default. A value
        <= 0 (explicit or via env) disables the cap (uncapped, for local dev).
        """
        if max_runtime_s:
            return max_runtime_s
        env = os.environ.get("SIMLAB_MAX_RUNTIME_S")
        if env:
            try:
                return float(env)
            except ValueError:
                pass
        return DEFAULT_MAX_RUNTIME_S

    def submit(self, sweep: SweepSpec) -> str:
        base_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{sweep.name}"
        # TOCTOU-safe: create the dir exclusively, retrying with a counter suffix
        # if two sweeps collide in the same second.
        counter = 0
        while True:
            sweep_id = base_id if counter == 0 else f"{base_id}-{counter}"
            out_dir = self.results_root / sweep_id
            try:
                out_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                counter += 1

        # Write the spec BEFORE returning. `list_sweeps` skips any directory
        # without a spec.json, and the runner used to be the only writer — from
        # inside the worker thread — so a client that POSTed a sweep and then
        # listed could legitimately not see it yet (the flake that made
        # `test_list_sweeps_sorted_by_mtime_desc` fail under load, and a real
        # gap in the UI). Writing it here also makes the listing's mtime sort
        # key the SUBMIT time rather than whenever the thread got scheduled.
        write_spec(out_dir, sweep)

        # Wall-clock start for the live combats/sec readout. Set just before the
        # run begins so the first poll's rate isn't skewed by spawn warmup.
        started = [0.0]

        def _progress(done: int, total: int) -> None:
            elapsed = time.perf_counter() - started[0] if started[0] else 0.0
            _atomic_write_json(
                out_dir / "progress.json",
                {
                    "done": done,
                    "total": total,
                    "status": "running",
                    "jobs": self.jobs,
                    "elapsed_s": round(elapsed, 2),
                    "combats_per_s": round(done / elapsed, 2) if elapsed > 0 else 0.0,
                },
            )

        # Counted off the materialised cells, not the axes: a start-state sweep
        # has empty encounter/deck axes, and the product read 0 until the first
        # chunk reported.
        total = len(sweep.cells()) * sweep.n_seeds
        # The first progress.json is written HERE, for the same reason as the
        # spec: written only by the worker thread, `status()` answered
        # "unknown" (a 404 from the API) to a client that POSTed and then
        # polled before the thread was scheduled — the flake that failed
        # `test_results_unknown_baseline_422` on CI.
        _progress(0, total)

        cancel = CancelToken()
        with self._lock:
            self._cancels[sweep_id] = cancel

        # Watchdog: trip the same hard-kill token the stop endpoint uses once the
        # sweep exceeds the wall-clock cap. `timed_out` distinguishes it from a
        # user-requested stop so the final status can say why.
        timed_out = threading.Event()

        def _on_timeout() -> None:
            timed_out.set()
            cancel.request()

        watchdog = (
            threading.Timer(self.max_runtime_s, _on_timeout) if self.max_runtime_s > 0 else None
        )

        def _run() -> None:
            try:
                started[0] = time.perf_counter()
                _progress(0, total)
                if watchdog is not None:
                    watchdog.start()
                self._runner(
                    sweep, out_dir=out_dir, jobs=self.jobs, progress=_progress, cancel=cancel
                )
                prog = _read_json(out_dir / "progress.json")
                # run_sweep is status-agnostic: a tripped token means the sweep
                # was stopped, so the partial result is "cancelled", not "done".
                # `stopped_reason` says whether it was the user or the runtime cap.
                if cancel.is_set():
                    prog["status"] = "cancelled"
                    prog["stopped_reason"] = "max_runtime" if timed_out.is_set() else "user"
                else:
                    prog["status"] = "done"
                _atomic_write_json(out_dir / "progress.json", prog)
            except Exception as exc:
                _atomic_write_json(
                    out_dir / "progress.json",
                    {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                )
            finally:
                if watchdog is not None:
                    watchdog.cancel()
                with self._lock:
                    self._cancels.pop(sweep_id, None)

        # Daemon thread runs detached; on-disk progress.json is the source of
        # truth, so we intentionally keep no reference to the thread object.
        threading.Thread(target=_run, daemon=True, name=f"sweep-{sweep_id}").start()
        return sweep_id

    def stop(self, sweep_id: str) -> bool:
        """Request cancellation of a running sweep; return False if not running.

        Hard-kills the sweep's worker processes immediately (via the token);
        the runner thread then writes the "cancelled" status. A sweep that has
        already finished (or never existed, or predates a restart) has no live
        token and returns False.
        """
        with self._lock:
            cancel = self._cancels.get(sweep_id)
        if cancel is None:
            return False
        cancel.request()
        return True

    def status(self, sweep_id: str) -> dict:
        sweep_dir = self.results_root / sweep_id
        p = sweep_dir / "progress.json"
        if not p.exists():
            # CLI sweeps (`just sim`) write no progress.json; if the summary is
            # there the sweep finished — keep them browsable in the UI.
            if (sweep_dir / "summary.json").exists():
                return {"status": "done"}
            return {"status": "unknown"}
        return _read_json(p)

    def list_sweeps(self) -> list[dict]:
        out = []
        for d in self.results_root.iterdir():
            spec_path = d / "spec.json"
            if not spec_path.exists():
                continue
            entry = {
                "sweep_id": d.name,
                "spec": json.loads(spec_path.read_text(encoding="utf-8")),
                "mtime": spec_path.stat().st_mtime,
            }
            entry.update(self.status(d.name))
            out.append(entry)
        # Sort newest-first by spec.json mtime. CLI sweeps have non-timestamped
        # names, so sorting by name (the old behavior) broke chronology.
        out.sort(key=lambda e: e["mtime"], reverse=True)
        return out
