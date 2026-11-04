"""Two co-op seats share the seed and character, so their advisors' run logs
used to share a name when they started in the same second: both opened it
"w" and interleaved, tearing lines mid-record (2026-10-27)."""

import json
from pathlib import Path

import game_log
from game_log import GameLogger


def test_loggers_started_in_the_same_second_get_their_own_files(tmp_path, monkeypatch):
    class FrozenNow(game_log.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 27, 20, 18, 0)

    monkeypatch.setattr(game_log, "datetime", FrozenNow)
    host = GameLogger("Ironclad", "SEED", log_dir=str(tmp_path))
    seat = GameLogger("Ironclad", "SEED", log_dir=str(tmp_path))
    host.log_state({"seat": "host"})
    seat.log_state({"seat": "teammate"})
    host.close()
    seat.close()

    assert host.path != seat.path
    for logger, seat_name in ((host, "host"), (seat, "teammate")):
        records = [
            json.loads(line) for line in Path(logger.path).read_text(encoding="utf-8").splitlines()
        ]
        assert [r["data"]["seat"] for r in records if r["type"] == "state"] == [seat_name]
