"""
Game logging — saves every state + action per run as JSONL for debugging.

Log files are stored in <project_root>/logs/ with auto-cleanup of old files.
"""

import itertools
import json
import os
import pathlib
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")

# Default: keep logs for 7 days
LOG_RETENTION_DAYS = 7


def _json_default(o):
    """Fallback encoder for values ``json`` can't serialize natively.

    The Rust backend ships a combat ``ResolveChoice``'s ``selection`` field as a
    PyO3 ``Vec<u8>`` — Python ``bytes`` — inside the state dict at a
    ``combat_resolve_choice`` decision point. Convert it (and any stray bytes)
    to a list of ints, which is lossless and matches how the replay pipeline and
    ``reasoning._choice_key`` read ``selection``.
    """
    if isinstance(o, (bytes, bytearray)):
        return list(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _dumps(entry) -> str:
    """JSON-encode one log record, tolerating engine ``bytes`` values."""
    return json.dumps(entry, ensure_ascii=False, default=_json_default)


def cleanup_old_logs(max_age_days=LOG_RETENTION_DAYS):
    """Remove log files older than max_age_days."""
    if not os.path.isdir(LOG_DIR):
        return
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for fname in os.listdir(LOG_DIR):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(LOG_DIR, fname)
        if pathlib.Path(fpath).stat().st_mtime < cutoff:
            os.remove(fpath)
            removed += 1
    if removed:
        print(f"  [log] Cleaned up {removed} old log file(s)")


class GameLogger:
    """Logs every game step (state received + action sent) to a JSONL file."""

    def __init__(
        self,
        character: str,
        seed: str,
        enabled: bool = True,
        log_dir: str | None = None,
        ascension: int = 0,
        extra: dict | None = None,
    ):
        self.enabled = enabled
        self.character = character
        self.seed = seed
        self.ascension = ascension
        self._step = 0
        self._file = None
        if not enabled:
            return

        out_dir = log_dir or LOG_DIR
        os.makedirs(out_dir, exist_ok=True)
        if out_dir == LOG_DIR:
            cleanup_old_logs()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_seed = str(seed).replace("/", "_")
        # Exclusive create, suffixed on a clash: two co-op seats share the seed
        # and character, and advisors started in the same second both opened
        # one file "w" and tore each other's lines (2026-10-27).
        # utf-8 explicitly: Windows' default cp1252 cannot encode the arrows in
        # action labels ("Play Strike → Cultist"), which silently broke every
        # log_advice/log_action write in the live-advisor lane (2026-08-21).
        stem = f"{ts}_{character}_{safe_seed}"
        for n in itertools.count(1):
            suffix = "" if n == 1 else f"_{n}"
            self._path = os.path.join(out_dir, f"{stem}{suffix}.jsonl")
            try:
                self._file = open(self._path, "x", encoding="utf-8")
                break
            except FileExistsError:
                continue

        # First record is a self-describing header so a log identifies its own
        # run (character / seed / ascension) without depending on the start_run
        # command — workbench live/branching logs never emit one. Consumers that
        # only care about play (e.g. TimelineStore) ignore non-state/action types.
        entry = {
            "step": self._step,
            "ts": datetime.now().isoformat(),
            "type": "meta",
            "data": {
                "character": character,
                "seed": str(seed),
                "ascension": ascension,
                # Anything the writer needs to reproduce the run later
                # (``run_mode`` / ``act1`` / ``backend`` for the workbench's
                # replay-based restore). Merged, not nested, so a reader that
                # predates the field just ignores the extra keys.
                **(extra or {}),
            },
        }
        self._file.write(_dumps(entry) + "\n")
        self._file.flush()

    def log_state(self, state: dict):
        """Log a state/decision point received from the simulator."""
        if not self.enabled or not self._file:
            return
        self._step += 1
        entry = {
            "step": self._step,
            "ts": datetime.now().isoformat(),
            "type": "state",
            "data": state,
        }
        self._file.write(_dumps(entry) + "\n")
        self._file.flush()

    def log_action(self, action: dict):
        """Log an action/command sent to the simulator."""
        if not self.enabled or not self._file:
            return
        entry = {
            "step": self._step,
            "ts": datetime.now().isoformat(),
            "type": "action",
            "data": action,
        }
        self._file.write(_dumps(entry) + "\n")
        self._file.flush()

    def log_combat_log(self, rows: list):
        """Log the display-ready combat-log rows of the action just logged.

        Written between an ``action`` record and the state it produced, so a
        loader attributes the rows to the decision the action answered. Replay
        consumers that predate the record type ignore it.
        """
        if not self.enabled or not self._file:
            return
        entry = {
            "step": self._step,
            "ts": datetime.now().isoformat(),
            "type": "combat_log",
            "data": list(rows),
        }
        self._file.write(_dumps(entry) + "\n")
        self._file.flush()

    def log_meta_extra(self, extra: dict):
        """Append a ``type:"meta"`` record merging ``extra`` into its ``data``.

        Used by the workbench MCTS recorder to attach head-meta provenance
        (``fixture_stem`` / ``build_id`` / ``encounter`` / ``sims``) that
        ``workbench.report`` / ``workbench.aggregate`` read, without coupling
        the logger's constructor to those keys.
        """
        if not self.enabled or not self._file:
            return
        entry = {
            "step": self._step,
            "ts": datetime.now().isoformat(),
            "type": "meta",
            "data": dict(extra),
        }
        self._file.write(_dumps(entry) + "\n")
        self._file.flush()

    def log_summary(self, summary: dict):
        """Append the terminal summary as a ``type:"meta"`` record.

        ``data`` is ``{"summary": True, **summary}`` (keys ``final_won`` /
        ``final_turn`` / ``final_player_hp`` / ``final_player_max_hp``), which
        ``workbench.report._load`` recognizes as the run's summary head.
        """
        if not self.enabled or not self._file:
            return
        entry = {
            "step": self._step,
            "ts": datetime.now().isoformat(),
            "type": "meta",
            "data": {"summary": True, **summary},
        }
        self._file.write(_dumps(entry) + "\n")
        self._file.flush()

    def log_mcts(self, data: dict):
        """Log per-decision MCTS search stats for the most-recent state.

        Shares the current step number with the preceding ``log_state`` call,
        so :class:`workbench.timeline.TimelineStore` can attach it to that step.
        """
        if not self.enabled or not self._file:
            return
        entry = {
            "step": self._step,
            "ts": datetime.now().isoformat(),
            "type": "mcts",
            "data": data,
        }
        self._file.write(_dumps(entry) + "\n")
        self._file.flush()

    def log_coach(self, data: dict):
        """Log one live-coach record (``realgame_coach``): a graded turn or
        choice, or a ``withdraw`` tombstone for records an undo abandoned.

        Shares the current step number like :meth:`log_mcts`, but the record
        is self-contained (it names its ``node_id``); readers that predate the
        type ignore it.
        """
        if not self.enabled or not self._file:
            return
        entry = {
            "step": self._step,
            "ts": datetime.now().isoformat(),
            "type": "coach",
            "data": data,
        }
        self._file.write(_dumps(entry) + "\n")
        self._file.flush()

    def log_wpa(self, data: dict):
        """Log the per-decision WPA value record for the most-recent state.

        Shares the current step number with the preceding ``log_state`` call
        (same contract as :meth:`log_mcts`), so the workbench replay timeline
        can attach the PPO critic's value to that decision. ``data`` is
        ``{"value": float, "taken_label": str | None}``.
        """
        if not self.enabled or not self._file:
            return
        entry = {
            "step": self._step,
            "ts": datetime.now().isoformat(),
            "type": "wpa",
            "data": data,
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def reopen(self) -> bool:
        """Re-open a closed log for appending; ``True`` when it is open again.

        The live coach's death autopsy is built after the run end closed the
        file (it waits for the dying turn's grade), and it still belongs in
        that run's log. Records keep the last step number, so readers attach
        nothing new to any step.
        """
        if not self.enabled or self._file or not self.path:
            return False
        self._file = open(self.path, "a", encoding="utf-8")
        return True

    @property
    def path(self):
        return getattr(self, "_path", None)
