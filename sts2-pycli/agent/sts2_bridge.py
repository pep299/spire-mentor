#!/usr/bin/env python3
"""HTTP bridge for STS2 headless game. Lets non-MCP agents drive the game one
command at a time over a localhost JSON HTTP endpoint.

Usage:
    python3 agent/sts2_bridge.py [port] [--compact] [--log FILE]
    python3 agent/sts2_bridge.py replay <logfile> [--until STEP] [--port PORT]

Examples:
    # Normal play with logging
    python3 agent/sts2_bridge.py 9876 --compact --log /tmp/game.jsonl

    # Replay a logged game up to step 42, then continue interactively
    python3 agent/sts2_bridge.py replay /tmp/game.jsonl --until 42 --port 9876

The underlying ``Game`` process wrapper, ``sanitize_json``, and ``compact_json``
helpers live in ``python/sts2_client.py`` and are shared with the MCP server
and the pytest fixture.
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))

from sts2_client import Game, compact_json, sanitize_json  # noqa: E402

# --- Arg parsing ---
COMPACT = "--compact" in sys.argv
REPLAY_MODE = len(sys.argv) > 1 and sys.argv[1] == "replay"
LOG_FILE = None
REPLAY_FILE = None
REPLAY_UNTIL = None
PORT = 9876

if REPLAY_MODE:
    REPLAY_FILE = sys.argv[2] if len(sys.argv) > 2 else None
    for i, a in enumerate(sys.argv):
        if a == "--until" and i + 1 < len(sys.argv):
            REPLAY_UNTIL = int(sys.argv[i + 1])
        if a == "--port" and i + 1 < len(sys.argv):
            PORT = int(sys.argv[i + 1])
else:
    filtered = [a for a in sys.argv[1:] if not a.startswith("--")]
    PORT = int(filtered[0]) if filtered else 9876
    for i, a in enumerate(sys.argv):
        if a == "--log" and i + 1 < len(sys.argv):
            LOG_FILE = sys.argv[i + 1]


# --- Logging ---
_log_fh = None


def log_entry(step, request, response):
    global _log_fh
    if _log_fh is None and LOG_FILE:
        _log_fh = open(LOG_FILE, "w", encoding="utf-8")
    if _log_fh:
        entry = {"step": step, "ts": time.time(), "req": request, "resp": response}
        _log_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _log_fh.flush()


# --- Replay mode ---
def do_replay():
    if not REPLAY_FILE:
        print(
            "Usage: sts2_bridge.py replay <logfile> [--until STEP] [--port PORT]",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Replaying {REPLAY_FILE} until step {REPLAY_UNTIL or 'end'}...", file=sys.stderr)
    game = Game(forward_stderr=True)

    with open(REPLAY_FILE, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    replayed = 0
    last_resp = None
    for entry in entries:
        step = entry["step"]
        if REPLAY_UNTIL and step > REPLAY_UNTIL:
            break
        req = entry["req"]
        resp = sanitize_json(game.send(req))
        last_resp = resp
        replayed += 1

        dec = resp.get("decision", resp.get("type", "?"))
        hp = resp.get("hp", resp.get("player", {}).get("hp", "")) if isinstance(resp, dict) else ""
        print(
            f"  Step {step}: {req.get('cmd', '?')}/{req.get('action', '')} → {dec} (hp={hp})",
            file=sys.stderr,
        )

    print(f"\nReplayed {replayed} steps. Game is at step {replayed}.", file=sys.stderr)
    if last_resp:
        dec = last_resp.get("decision", last_resp.get("type", "?"))
        print(f"Current state: {dec}", file=sys.stderr)
        print(json.dumps(last_resp, ensure_ascii=False, indent=2)[:2000], file=sys.stderr)

    print(f"\nStarting interactive bridge on port {PORT}...", file=sys.stderr)

    class ReplayHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                result = sanitize_json(game.send(data))
            except RuntimeError as e:
                # Game subprocess has died (EOF). Match the legacy in-process
                # behavior of returning a JSON error envelope rather than
                # propagating an exception out of do_POST.
                result = {"type": "error", "message": str(e)}
            if COMPACT:
                result = compact_json(result)
            body = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    HTTPServer(("127.0.0.1", PORT), ReplayHandler).serve_forever()


# --- Normal mode ---
if REPLAY_MODE:
    do_replay()
    sys.exit(0)

game = Game(forward_stderr=True)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            raw_result = sanitize_json(game.send(data))
        except RuntimeError as e:
            # Game subprocess has died (EOF). Return a JSON error envelope to
            # match the pre-refactor behavior of Game._read returning a dict.
            raw_result = {"type": "error", "message": str(e)}
        # Log FULL response (before compact) so replay reproduces accurately.
        log_entry(game.step, data, raw_result)
        client_result = compact_json(raw_result) if COMPACT else raw_result
        body = json.dumps(client_result, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


mode = "compact" if COMPACT else "full"
log_info = f", logging to {LOG_FILE}" if LOG_FILE else ""
print(f"STS2 bridge on port {PORT} (mode={mode}{log_info})", file=sys.stderr)
HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
