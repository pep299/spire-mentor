"""ControlPlane — the advisor's localhost HTTP server for the rewind lane.

Ported from the offline smoke harness (fake-mod round trip incl. fork and
409 non-seekable) that validated the lane before it shipped 2026-08-21. A
FakeClient stands in for the SpireMentor mod: it answers the rewind verbs the
way the mod would, from another thread, exercising the waiter/round-trip
machinery for real over HTTP on an ephemeral port.
"""

from __future__ import annotations

import gzip
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from realgame_control import BugReportForward, ControlPlane, StaticSite
from realgame_livetree import LiveRunTree

ENV: dict[str, Any] = {
    "seed": "S1",
    "run_state": {"character": "IRONCLAD"},
    "act": 1,
    "floor": 1,
}


class FakeClient:
    """Answers rewind verbs the way the mod would, from another thread."""

    def __init__(self, control: ControlPlane) -> None:
        self.control = control
        self.sent: list[dict[str, Any]] = []
        #: Optional override: action dict -> reply dict (tests inject errors).
        self.reply_fn: Any = None

    def send_action(self, action: dict[str, Any], *, echo_state_id: bool = True) -> None:
        self.sent.append(action)
        if self.reply_fn is not None:
            reply = self.reply_fn(action)
        else:
            kind = action.get("action")
            if kind == "list_snapshots":
                reply = {
                    "type": "snapshots",
                    "run_start_time": 111,
                    "snapshots": [
                        {"id": "111-0", "seq": 0, "floor": 1, "post_combat": False},
                        {"id": "111-1", "seq": 1, "floor": 1, "post_combat": True},
                    ],
                }
            elif kind == "restore_snapshot":
                reply = {
                    "type": "snapshot_restored",
                    "snapshot_id": action["snapshot_id"],
                    "floor": 1,
                    "state_id": 999,
                }
            elif kind == "undo_turn":
                reply = {
                    "type": "combat_undone",
                    "floor": 1,
                    "turn": action["turn"],
                    "tape_id": action.get("tape_id"),
                    "replayed_events": 5,
                    "state_id": 1000,
                }
            else:
                raise AssertionError(kind)
        # The mod's replies are broadcasts routed back by the advisor loop —
        # from a different thread than the HTTP handler that is waiting.
        threading.Timer(0.02, lambda: self.control.on_bridge_message(reply)).start()


@pytest.fixture()
def plane():
    tree = LiveRunTree()
    control = ControlPlane(tree)
    client = FakeClient(control)
    control.set_client(client)
    control.start(0)  # ephemeral port
    assert control.port
    yield control, tree, client
    control.stop()


def _get(control: ControlPlane, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{control.port}{path}", timeout=5) as r:
        return json.loads(r.read())


def _post(control: ControlPlane, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{control.port}{path}",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status or 0, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _play_room(control: ControlPlane, tree: LiveRunTree) -> None:
    """Room entry snapshot, one combat decision, one reward decision."""
    control.on_bridge_message(
        {
            "type": "snapshot_saved",
            "snapshot_id": "111-0",
            "seq": 0,
            "floor": 1,
            "act": 1,
            "post_combat": False,
        }
    )
    tree.on_decision({**ENV, "turn": 1, "request_id": "r1"}, "combat_action")
    tree.on_human_action({"detail": "Strike -> Cultist", "advice_request_id": "r1"})
    control.on_bridge_message(
        {
            "type": "snapshot_saved",
            "snapshot_id": "111-1",
            "seq": 1,
            "floor": 1,
            "act": 1,
            "post_combat": True,
            "has_tape": True,
        }
    )
    tree.on_decision({**ENV, "request_id": "r2"}, "card_reward")


class TestReadEndpoints:
    def test_tree_and_snapshots(self, plane) -> None:
        control, tree, _ = plane
        _play_room(control, tree)
        t = _get(control, "/tree")
        assert [n["decision"] for n in t["nodes"]] == [
            "root",
            "room_entry",
            "combat_action",
            "card_reward",
        ]
        combat = next(n for n in t["nodes"] if n["decision"] == "combat_action")
        assert combat["label"] == "f1 · Strike -> Cultist"
        snaps = _get(control, "/snapshots")
        assert [s["id"] for s in snaps["snapshots"]] == ["111-0", "111-1"]

    def test_status_reports_connection(self, plane) -> None:
        control, _, _ = plane
        assert _get(control, "/status")["connected"] is True


#: A Copy-state-shaped decision message for the /state tests.
ADVICE_REQUEST: dict[str, Any] = {
    **ENV,
    "type": "advice_request",
    "advice_for": "combat_action",
    "state_id": 7,
    "hand": [{"id": "STRIKE_IRONCLAD"}],
}


class TestState:
    """GET /state: the latest advice_request, verbatim (what Copy state yields)."""

    def test_404_before_any_decision(self, plane) -> None:
        control, _, _ = plane
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(control, "/state")
        assert exc.value.code == 404
        assert json.loads(exc.value.read())["error"] == "no_state"

    def test_serves_latest_advice_request_verbatim(self, plane) -> None:
        control, _, _ = plane
        control.note_message({**ADVICE_REQUEST, "state_id": 6})
        control.note_message(ADVICE_REQUEST)
        # Other message types do not replace the position.
        control.note_message({**ENV, "type": "human_action", "action": "play_card"})
        assert _get(control, "/state") == ADVICE_REQUEST

    def test_stored_copy_is_isolated_from_later_mutation(self, plane) -> None:
        control, _, _ = plane
        msg = json.loads(json.dumps(ADVICE_REQUEST))
        control.note_message(msg)
        msg["hand"].append({"id": "DEFEND_IRONCLAD"})
        assert _get(control, "/state") == ADVICE_REQUEST

    @pytest.mark.parametrize("end", ["game_over", "run_complete"])
    def test_cleared_at_run_end(self, plane, end: str) -> None:
        control, _, _ = plane
        control.note_message(ADVICE_REQUEST)
        control.note_message({**ENV, "type": end})
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(control, "/state")
        assert exc.value.code == 404


class TestSeek:
    def test_room_entry_seek_round_trip_and_fork(self, plane) -> None:
        control, tree, client = plane
        _play_room(control, tree)
        room = next(n for n in _get(control, "/tree")["nodes"] if n["decision"] == "room_entry")
        code, reply = _post(control, "/restore", {"node_id": room["id"]})
        assert (code, reply["type"]) == (200, "snapshot_restored")
        assert client.sent[-1] == {"action": "restore_snapshot", "snapshot_id": "111-0"}
        assert _get(control, "/tree")["active_leaf"] == room["id"]
        # The next decision forks a sibling branch at the restored node.
        tree.on_decision({**ENV, "request_id": "r3"}, "combat_action")
        t = _get(control, "/tree")
        children = [n for n in t["nodes"] if n["parent_id"] == room["id"]]
        assert len(children) == 2

    def test_combat_node_seeks_via_undo_turn_with_tape(self, plane) -> None:
        control, tree, client = plane
        _play_room(control, tree)
        combat = next(
            n for n in _get(control, "/tree")["nodes"] if n["decision"] == "combat_action"
        )
        assert combat["turn"] == 1 and combat["tape_id"] == "111-1"
        code, reply = _post(control, "/restore", {"node_id": combat["id"]})
        assert (code, reply["type"]) == (200, "combat_undone")
        assert client.sent[-1] == {"action": "undo_turn", "turn": 1, "tape_id": "111-1"}
        # Undo to turn 1 = the fight's start: active lands on the room entry.
        t = _get(control, "/tree")
        room = next(n for n in t["nodes"] if n["decision"] == "room_entry")
        assert t["active_leaf"] == room["id"]

    def test_live_combat_node_seeks_without_tape(self, plane) -> None:
        control, tree, client = plane
        control.on_bridge_message(
            {
                "type": "snapshot_saved",
                "snapshot_id": "111-0",
                "seq": 0,
                "floor": 1,
                "act": 1,
                "post_combat": False,
            }
        )
        tree.on_decision({**ENV, "turn": 2, "request_id": "r1"}, "combat_action")
        combat = next(
            n for n in _get(control, "/tree")["nodes"] if n["decision"] == "combat_action"
        )
        code, reply = _post(control, "/restore", {"node_id": combat["id"]})
        assert (code, reply["type"]) == (200, "combat_undone")
        assert client.sent[-1] == {"action": "undo_turn", "turn": 2}

    def test_undo_error_is_a_409_not_a_timeout(self, plane) -> None:
        control, tree, client = plane
        control.on_bridge_message(
            {
                "type": "snapshot_saved",
                "snapshot_id": "111-0",
                "seq": 0,
                "floor": 1,
                "act": 1,
                "post_combat": False,
            }
        )
        tree.on_decision({**ENV, "turn": 1, "request_id": "r1"}, "combat_action")
        client.reply_fn = lambda action: {
            "type": "error",
            "error": "undo_not_in_combat",
            "detail": "no combat in progress",
        }
        combat = next(
            n for n in _get(control, "/tree")["nodes"] if n["decision"] == "combat_action"
        )
        code, reply = _post(control, "/restore", {"node_id": combat["id"]})
        assert (code, reply.get("error")) == (409, "undo_not_in_combat")

    def test_node_without_save_point_or_turn_is_409(self, plane) -> None:
        control, tree, _ = plane
        _play_room(control, tree)
        reward = next(n for n in _get(control, "/tree")["nodes"] if n["decision"] == "card_reward")
        code, reply = _post(control, "/restore", {"node_id": reward["id"]})
        assert (code, reply.get("error")) == (409, "not_seekable")

    def test_missing_body_is_400(self, plane) -> None:
        control, _, _ = plane
        code, _reply = _post(control, "/restore", {})
        assert code == 400


# ---------------------------------------------------------------------------
# Browser access (GH #709): the CORS + Private Network Access opt-in
# ---------------------------------------------------------------------------

ORIGIN = "https://your-workbench.example.com"


def _request(
    control: ControlPlane, method: str, path: str, headers: dict[str, str]
) -> tuple[int, dict[str, str]]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{control.port}{path}", method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status or 0, {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}


@pytest.fixture()
def browser_plane():
    tree = LiveRunTree()
    control = ControlPlane(tree)
    control.set_client(FakeClient(control))
    control.start(0, allow_origins=[ORIGIN + "/"])  # a trailing slash is tolerated
    assert control.port
    yield control
    control.stop()


class TestBrowserAccess:
    def test_default_sends_no_cors_headers(self, plane) -> None:
        control, _, _ = plane
        code, headers = _request(control, "GET", "/status", {"Origin": ORIGIN})
        assert code == 200
        assert "access-control-allow-origin" not in headers
        # The preflight is refused outright, so a browser never sends the POST.
        code, headers = _request(
            control,
            "OPTIONS",
            "/restore",
            {"Origin": ORIGIN, "Access-Control-Request-Method": "POST"},
        )
        assert code == 403
        assert "access-control-allow-origin" not in headers

    def test_allow_listed_origin_gets_cors_and_pna_headers(self, browser_plane) -> None:
        control = browser_plane
        code, headers = _request(
            control,
            "OPTIONS",
            "/restore",
            {
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert code == 204
        assert headers["access-control-allow-origin"] == ORIGIN
        assert "POST" in headers["access-control-allow-methods"]
        assert headers["access-control-allow-headers"] == "content-type"
        assert headers["access-control-allow-private-network"] == "true"
        assert headers["vary"] == "Origin"
        # …and on the real replies, echoed (never `*`).
        code, headers = _request(control, "GET", "/status", {"Origin": ORIGIN})
        assert code == 200
        assert headers["access-control-allow-origin"] == ORIGIN

    def test_other_origins_get_nothing(self, browser_plane) -> None:
        control = browser_plane
        code, headers = _request(control, "GET", "/status", {"Origin": "https://evil.example"})
        assert code == 200  # the same-machine proxy path is unaffected…
        assert "access-control-allow-origin" not in headers  # …but a page cannot read it
        code, _ = _request(
            control,
            "OPTIONS",
            "/restore",
            {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert code == 403
        # No Origin at all (curl, the proxy): plain replies, no CORS headers.
        code, headers = _request(control, "GET", "/status", {})
        assert code == 200
        assert "access-control-allow-origin" not in headers


@pytest.fixture()
def site_plane(tmp_path: Path):
    """A control plane hosting a fake static workbench build."""
    root = tmp_path / "workbench"
    (root / "wb-assets").mkdir(parents=True)
    (root / "assets" / "cards").mkdir(parents=True)
    (root / "ppo").mkdir()
    (root / "index.html").write_text("<!doctype html><title>wb</title>")
    (root / "wb-precache.json").write_text('{"build": "t", "files": []}')
    (root / "wb-assets" / "engine.wasm").write_bytes(b"\0asm")
    (root / "assets" / "cards" / "strike.png").write_bytes(b"RIFF....WEBP")
    (root / "ppo" / "index.json").write_text('{"files": ["champ.onnx"]}')
    ppo = tmp_path / "advisor-ppo"
    ppo.mkdir()
    (ppo / "champ.onnx").write_bytes(b"onnx")
    (tmp_path / "secret.txt").write_text("outside the build")
    control = ControlPlane(LiveRunTree())
    control.start(0, site=StaticSite(root, ppo_dir=ppo))
    assert control.port
    yield control
    control.stop()


def _fetch(control: ControlPlane, path: str) -> tuple[int, dict[str, str], bytes]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{control.port}{path}", timeout=5) as r:
            return r.status or 0, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()


class TestBundledWorkbench:
    def test_no_site_means_no_files_and_no_health(self, plane) -> None:
        control, _, _ = plane
        assert _fetch(control, "/")[0] == 404
        assert _fetch(control, "/api/health")[0] == 404  # the mod falls back to :8077

    def test_health_and_index(self, site_plane) -> None:
        assert json.loads(_fetch(site_plane, "/api/health")[2]) == {
            "ok": True,
            "workbench": "static",
        }
        code, headers, body = _fetch(site_plane, "/")
        assert code == 200 and body.startswith(b"<!doctype html>")
        assert headers["content-type"].startswith("text/html")
        assert headers["cache-control"] == "no-cache"
        # The endpoints win over the site.
        assert json.loads(_fetch(site_plane, "/status")[2]) == {"connected": False}

    def test_headers_follow_the_pages_rules(self, site_plane) -> None:
        _, headers, _ = _fetch(site_plane, "/wb-assets/engine.wasm")
        assert headers["content-type"] == "application/wasm"
        assert "immutable" in headers["cache-control"]
        _, headers, _ = _fetch(site_plane, "/assets/cards/strike.png")
        assert headers["content-type"] == "image/webp"  # WebP under the .png name

    def test_ppo_nets_come_from_the_advisor_bundle(self, site_plane) -> None:
        assert json.loads(_fetch(site_plane, "/ppo/index.json")[2]) == {"files": ["champ.onnx"]}
        code, headers, body = _fetch(site_plane, "/ppo/champ.onnx")
        assert (code, body) == (200, b"onnx")
        assert headers["content-type"] == "application/octet-stream"

    def test_spa_fallback_and_misses(self, site_plane) -> None:
        assert _fetch(site_plane, "/some/route")[2].startswith(b"<!doctype html>")
        assert _fetch(site_plane, "/missing.js")[0] == 404
        # No bug_reports configured: a 404 keeps the page's Report button hidden.
        assert _fetch(site_plane, "/api/bugreport")[0] == 404
        assert _fetch(site_plane, "/..%2fsecret.txt")[0] == 404
        assert _fetch(site_plane, "/ppo/..%2f..%2fsecret.txt")[0] == 404

    def test_is_static_build(self, tmp_path: Path) -> None:
        (tmp_path / "index.html").write_text("x")
        assert not StaticSite.is_static_build(tmp_path)  # a server build of the SPA
        (tmp_path / "wb-precache.json").write_text("{}")
        assert StaticSite.is_static_build(tmp_path)


class TestGzippedJson:
    """The mod's copy of the build ships every .json as .json.gz (the game
    reads each .json under a mod's folder as a mod manifest)."""

    @pytest.fixture
    def gz_plane(self, tmp_path: Path):
        root = tmp_path / "workbench"
        (root / "assets").mkdir(parents=True)
        (root / "index.html").write_text("<!doctype html><title>wb</title>")
        (root / "wb-precache.json.gz").write_bytes(gzip.compress(b'{"build": "t"}'))
        (root / "assets" / "card_text.json.gz").write_bytes(gzip.compress(b'{"STRIKE": "Deal 6"}'))
        assert StaticSite.is_static_build(root)
        control = ControlPlane(LiveRunTree())
        control.start(0, site=StaticSite(root))
        yield control
        control.stop()

    def test_sent_gzip_encoded_when_accepted(self, gz_plane) -> None:
        req = urllib.request.Request(
            f"http://127.0.0.1:{gz_plane.port}/assets/card_text.json",
            headers={"Accept-Encoding": "gzip, deflate, br"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            headers = {k.lower(): v for k, v in r.headers.items()}
            body = r.read()
        assert headers["content-encoding"] == "gzip"
        assert headers["content-type"] == "application/json"
        assert headers["vary"] == "accept-encoding"
        assert int(headers["content-length"]) == len(body)
        assert json.loads(gzip.decompress(body)) == {"STRIKE": "Deal 6"}

    def test_decompressed_for_a_client_without_gzip(self, gz_plane) -> None:
        code, headers, body = _fetch(gz_plane, "/assets/card_text.json")
        assert code == 200 and "content-encoding" not in headers
        assert json.loads(body) == {"STRIKE": "Deal 6"}
        assert int(headers["content-length"]) == len(body)

    def test_head_and_missing(self, gz_plane) -> None:
        req = urllib.request.Request(
            f"http://127.0.0.1:{gz_plane.port}/wb-precache.json", method="HEAD"
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 200 and r.read() == b""
        assert _fetch(gz_plane, "/assets/nope.json")[0] == 404


class TestLiveLog:
    def test_404_until_a_log_is_open(self, plane) -> None:
        control, _, _ = plane
        assert _fetch(control, "/log")[0] == 404

    def test_serves_complete_lines_and_a_stat(self, plane, tmp_path: Path) -> None:
        control, _, _ = plane
        log = tmp_path / "20261016-120000_IRONCLAD_ABC.jsonl"
        log.write_bytes(b'{"step": 0}\n{"step": 1}\n{"ste')  # the advisor mid-write
        control.set_log_path(str(log))
        code, _, body = _fetch(control, "/log")
        reply = json.loads(body)
        assert code == 200
        assert reply["name"] == log.name
        assert reply["text"] == '{"step": 0}\n{"step": 1}\n'  # the torn record waits
        assert reply["size"] == log.stat().st_size
        stat = json.loads(_fetch(control, "/log?stat=1")[2])
        assert "text" not in stat and stat["size"] == reply["size"]

    def test_name_is_a_bare_safe_filename(self, plane, tmp_path: Path) -> None:
        control, _, _ = plane
        log = tmp_path / "run with space.jsonl"
        log.write_text("{}\n")
        control.set_log_path(str(log))
        assert json.loads(_fetch(control, "/log?stat=1")[2])["name"] == "run_with_space.jsonl"

    def test_closed_log_still_served(self, plane, tmp_path: Path) -> None:
        # The autopsy's links arrive right after game_over closed the file:
        # the finished log stays served until the next run opens its own.
        from realgame_advisor import publish_log_path
        from realgame_workbench_log import WorkbenchRunLogger

        control, _, _ = plane
        env = {"seed": "S1", "floor": 1, "run_state": {"character": "IRONCLAD"}}
        wlog = WorkbenchRunLogger(str(tmp_path))
        wlog.log_decision({**env, "options": []}, "event")
        publish_log_path(control, wlog)
        first = wlog.path
        wlog.log_run_end({"type": "game_over", "victory": False, **env})
        publish_log_path(control, wlog)
        code, _, body = _fetch(control, "/log")
        assert code == 200 and json.loads(body)["name"] == Path(first or "").name
        wlog.log_decision({**env, "seed": "S2", "options": []}, "event")
        publish_log_path(control, wlog)
        assert wlog.path != first
        assert json.loads(_fetch(control, "/log?stat=1")[2])["name"] == Path(wlog.path or "").name


# -- bug reports (the bundled workbench's 🐞 Report button) -----------------


class _FakeStore:
    """The hosted site behind its sign-in, on a real loopback port: only
    POST /intake/bugreport answers signed out; everything else redirects to
    the sign-in page, as Cloudflare Access does."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        store = self
        self.posts: list[tuple[dict[str, str], dict[str, Any]]] = []
        self.reports: dict[str, bytes] = {}
        self.fail_with: int | None = None
        #: Set: the intake is NOT exempt from the sign-in either.
        self.gate_intake = False
        self.login_hits = 0

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def _reply(self, code: int, body: bytes) -> None:
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _to_login(self) -> None:
                self.send_response(302)
                self.send_header("Location", f"{store.site}/cdn-cgi/access/login?team=t")
                self.send_header("content-length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers["content-length"]))
                if self.path != "/intake/bugreport" or store.gate_intake:
                    self._to_login()
                    return
                store.posts.append(
                    ({k.lower(): v for k, v in self.headers.items()}, json.loads(raw))
                )
                if store.fail_with:
                    self._reply(store.fail_with, b'{"detail": "store says no"}')
                    return
                rid = "wb-00000000000000aa"
                store.reports[rid] = raw
                self._reply(
                    201, json.dumps({"id": rid, "url": f"{store.site}/?report={rid}"}).encode()
                )

            def do_GET(self) -> None:
                if self.path.startswith("/cdn-cgi/access/login"):
                    store.login_hits += 1
                    self._reply(200, b"<html>sign in with Google</html>")
                else:
                    self._to_login()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.site = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _dump(**extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "meta": {"mode": "replay", "host": "http://127.0.0.1:8079"},
        "repro": {"run_file": "advisor.jsonl", "line": 3, "run_log": "{}\n"},
        "state": {"game_state": {"floor": 4}, "map": None},
        "ui": {"screen": "combat_play", "overlay": None},
        "note": "intent shows the wrong damage",
        **extra,
    }


def _post_raw(
    control: ControlPlane, path: str, body: bytes, headers: dict[str, str]
) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{control.port}{path}", data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status or 0, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture()
def store():
    s = _FakeStore()
    yield s
    s.stop()


@pytest.fixture()
def report_plane(tmp_path: Path, store: _FakeStore):
    control = ControlPlane(LiveRunTree())
    control.start(0, bug_reports=BugReportForward(store.site, tmp_path / "bugreports", timeout=5))
    assert control.port
    yield control
    control.stop()


JSON = {"content-type": "application/json"}


class TestBugReports:
    def test_probe_answers_json_so_the_button_shows(self, report_plane, tmp_path: Path) -> None:
        assert _get(report_plane, "/api/bugreport") == {"ok": True, "store": "forward"}
        local = BugReportForward(None, tmp_path)
        assert local.probe() == {"ok": True, "store": "local"}

    def test_forwards_with_the_live_position_and_returns_the_store_reply(
        self, report_plane, store: _FakeStore
    ) -> None:
        report_plane.note_message({"type": "advice_request", "advice_for": "card_reward", **ENV})
        report_plane.set_log_path("C:/Users/alice/AppData/Local/SpireMentor/logs/a.jsonl")
        code, reply = _post_raw(report_plane, "/api/bugreport", json.dumps(_dump()).encode(), JSON)
        assert code == 201
        assert reply == {
            "id": "wb-00000000000000aa",
            "url": f"{store.site}/?report=wb-00000000000000aa",
        }
        headers, sent = store.posts[-1]
        # Not a browser: no Origin, but an agent Cloudflare's bot check accepts.
        assert "origin" not in headers
        assert headers["user-agent"].startswith("SpireMentor-advisor/")
        assert sent["state"]["live_game"]["advice_for"] == "card_reward"
        assert sent["state"]["game_state"] == {"floor": 4}  # the page's own view, untouched
        assert sent["meta"]["forwarded_by"] == "advisor"
        assert sent["meta"]["advisor"]["character"] == "IRONCLAD"
        assert "log_path" not in sent["meta"]["advisor"]  # it names the player's home
        assert sent["repro"]["run_log"] == "{}\n"

    def test_a_sent_report_is_read_on_the_site_not_through_here(
        self, report_plane, store: _FakeStore
    ) -> None:
        _, reply = _post_raw(report_plane, "/api/bugreport", json.dumps(_dump()).encode(), JSON)
        code, _, body = _fetch(report_plane, f"/api/bugreport/{reply['id']}")
        # Reading needs the site's sign-in, which this server cannot pass.
        assert code == 404
        assert f"{store.site}/?report={reply['id']}" in json.loads(body)["detail"]
        assert store.login_hits == 0
        assert _fetch(report_plane, "/api/bugreport/..%2f..%2fsecret")[0] == 404

    def test_a_gated_intake_is_reported_as_the_sign_in_not_followed(
        self, report_plane, store: _FakeStore, tmp_path: Path
    ) -> None:
        store.gate_intake = True
        code, reply = _post_raw(report_plane, "/api/bugreport", json.dumps(_dump()).encode(), JSON)
        assert code == 201 and Path(reply["path"]).is_file()
        assert "HTTP 302" in reply["detail"] and "sign-in" in reply["detail"]
        # The redirect was not followed (a followed POST lands on the login page).
        assert store.login_hits == 0

    def test_store_failure_saves_locally_and_says_so(
        self, report_plane, store: _FakeStore, tmp_path: Path
    ) -> None:
        store.fail_with = 503
        code, reply = _post_raw(report_plane, "/api/bugreport", json.dumps(_dump()).encode(), JSON)
        assert code == 201
        path = Path(reply["path"])
        assert path.parent == tmp_path / "bugreports" and path.name == f"{reply['id']}.json"
        assert "store says no" in reply["detail"] and str(path) in reply["detail"]
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["meta"]["id"] == reply["id"]
        assert saved["meta"]["received_via"] == "advisor-local"
        # ...and the page can open it from there.
        assert _get(report_plane, f"/api/bugreport/{reply['id']}")["meta"]["id"] == reply["id"]

    def test_no_site_saves_locally(self, tmp_path: Path) -> None:
        control = ControlPlane(LiveRunTree())
        control.start(0, bug_reports=BugReportForward(None, tmp_path / "br"))
        try:
            code, reply = _post_raw(control, "/api/bugreport", json.dumps(_dump()).encode(), JSON)
        finally:
            control.stop()
        assert code == 201 and Path(reply["url"]).is_file()
        assert "no bug-report address" in reply["detail"]

    def test_refuses_foreign_pages_non_json_and_bad_dumps(
        self, report_plane, store: _FakeStore
    ) -> None:
        body = json.dumps(_dump()).encode()
        own = f"http://127.0.0.1:{report_plane.port}"
        assert (
            _post_raw(
                report_plane, "/api/bugreport", body, {**JSON, "Origin": "https://evil.example"}
            )[0]
            == 403
        )
        # A cross-site page can send text/plain without a preflight; refuse it.
        assert (
            _post_raw(report_plane, "/api/bugreport", body, {"content-type": "text/plain"})[0]
            == 415
        )
        assert _post_raw(report_plane, "/api/bugreport", b"{nope", JSON)[0] == 400
        code, reply = _post_raw(
            report_plane, "/api/bugreport", json.dumps({"schema_version": 1}).encode(), JSON
        )
        assert (code, reply) == (422, {"detail": "meta must be an object"})
        assert store.posts == []
        # The page's own origin is fine.
        assert _post_raw(report_plane, "/api/bugreport", body, {**JSON, "Origin": own})[0] == 201
