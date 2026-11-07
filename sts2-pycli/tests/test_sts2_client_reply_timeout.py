"""The C# reply watchdog: a hung oracle fails fast and names the command (round-4 8084#b1)."""

from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest
import sts2_client
from sts2_client import Game, OracleHang

# A stand-in for the headless: answers the first command, then hangs on the second.
_FAKE = """
import sys, time
for i, line in enumerate(sys.stdin):
    if i == 0:
        print('{"type": "ok"}', flush=True)
    else:
        time.sleep(600)
"""


def _fake_game(timeout: float) -> Game:
    g = object.__new__(Game)
    g.proc = subprocess.Popen(
        [sys.executable, "-c", _FAKE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    g._lock = threading.Lock()
    g.step = 0
    g._reply_timeout = timeout
    g._hung = None
    return g


def test_hung_reply_raises_oracle_hang_naming_the_command():
    g = _fake_game(1.0)
    try:
        assert g.send({"cmd": "get_map"}) == {"type": "ok"}
        t0 = time.monotonic()
        with pytest.raises(
            OracleHang, match=r"^EOF from game process: no reply to step 2 \(action end_turn\)"
        ):
            g.send({"cmd": "action", "action": "end_turn"})
        assert time.monotonic() - t0 < 10
        assert g.proc.wait(timeout=5) is not None
    finally:
        g.proc.kill()


def test_oracle_hang_is_a_runtime_error():
    assert issubclass(OracleHang, RuntimeError)


def test_reply_timeout_reads_the_env(monkeypatch):
    monkeypatch.delenv(sts2_client.REPLY_TIMEOUT_ENV, raising=False)
    assert sts2_client.reply_timeout() == sts2_client.DEFAULT_REPLY_TIMEOUT
    monkeypatch.setenv(sts2_client.REPLY_TIMEOUT_ENV, "0")
    assert sts2_client.reply_timeout() == 0
