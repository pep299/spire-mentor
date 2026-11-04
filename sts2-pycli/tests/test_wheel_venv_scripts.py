"""Run ``tests/test_wheel_venv_layout.sh`` under pytest (GH #506).

The script pins ``wheel-venv.sh``'s POSIX / Windows / MSYS2 venv-layout
probes. On its own it only ran through ``just test-wheel-venv``, which no gate
calls, so a regression in the Windows behaviour it exists for would pass
``just test`` and the public repo's Windows CI. It is hermetic (fabricated
venvs under a temp dir, nothing built), so it is cheap enough for every run.

``test_wheel_venv_ambiguity.sh`` stays out: it plants a marker under the real
checkout's ``.venv-h/`` and needs the main ``.venv``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("test_wheel_venv_layout.sh")


def _bash() -> str | None:
    bash = shutil.which("bash")
    # On Windows `bash` can resolve to the WSL launcher in System32, which runs
    # the script inside a Linux VM instead of the Git Bash this layout is for.
    if bash is None or (sys.platform == "win32" and "system32" in bash.lower()):
        return None
    return bash


def test_wheel_venv_layout_script_passes() -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("needs bash (Git Bash on Windows)")
    proc = subprocess.run([bash, SCRIPT.as_posix()], capture_output=True, check=False, timeout=120)
    out = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 0, out
    assert "test_wheel_venv_layout: PASS" in out, out
