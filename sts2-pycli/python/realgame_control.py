"""Localhost control plane inside the live-game advisor (rewind lane).

The advisor owns the single TCP connection to the SpireMentor mod, so anything
else that wants to talk to the live game — the workbench's ``/api/bridge/*``
proxies — goes through this tiny HTTP server (``127.0.0.1:<control-port>``,
default 8079; 8078 belongs to the standalone SimLab). One command at a time:
the bridge's rewind verbs are fire-and-forget with broadcast replies, so a
command registers a waiter, sends the verb, and blocks (bounded) until the
advisor's receive loop routes the matching reply back here.

Endpoints (all JSON):
    GET  /status       — {connected, character, seed, floor, act, log_path}
    GET  /tree         — LiveRunTree.timeline()  (BranchTree.tsx wire shape)
    GET  /snapshots    — the mod's snapshot list (list_snapshots round trip)
    GET  /state        — the latest ``advice_request`` from the mod: the same
                         message the mod's Copy state yields, so the
                         workbench's state import takes it as is (404 until
                         the first one, and again after game over)
    GET  /log          — the run log the advisor is writing (``log_path``):
                         ``{name, size, mtime, text}``, the text cut at the
                         last complete line; ``?stat=1`` leaves the text out.
                         The bundled workbench's Follow mode mirrors it into
                         the page (it cannot read the file); 404 until the
                         first message opens a log
    POST /restore      — {snapshot_id} or {node_id}; rewinds the live game
    POST /search_tree  — companion MCTS tree over the current combat board
    GET  /api/bugreport, POST /api/bugreport, GET /api/bugreport/<id>
                       — the bundled workbench's bug reports (below); only
                         when ``start(..., bug_reports=...)`` configures them

Browser access (GH #709). The Python workbench reaches this server through a
same-origin proxy; the browser-only workbench (``https://your-workbench.example.com``)
has no server to proxy for it and must call this loopback server from an
``https://`` page. That crosses three browser gates — mixed content (a
loopback ``http://`` target is exempt as a potentially trustworthy origin),
Chrome's Private Network Access preflight, and CORS — and the last two are
answered here, OPT-IN: ``start(port, allow_origins=[...])`` names the page
origins allowed to call in (``--control-allow-origin`` on the advisor). With
no origin listed the server sends no CORS headers at all, so no page anywhere
can read a reply, exactly as before. Same-origin proxies are unaffected.

The bundled workbench. The packaged mod ships the browser-only workbench
(``web/dist`` of ``just workbench-static``) next to the advisor, and this
server hosts it: every GET that is not one of the endpoints above is a file
of that build (``start(..., site=StaticSite(...))``). The page and the
control plane then share one origin, so the live-game views need neither the
CORS opt-in nor Chrome's local-network prompt. ``GET /api/health`` answers
200 only while a site is mounted: the mod's workbench button probes it to
choose between this page and the dev workbench on :8077.

Bug reports from the bundled workbench. Its 🐞 Report button posts to
``/api/bugreport`` on the origin that served it -- on the hosted site that is
the Pages Function (``web/functions/api/_bugreport.ts``), here it is this
server, which forwards the dump to the same store (``--bug-report-site``, the
site the mod's own in-game reports go to; the mod passes it) and answers with
the store's ``{id, url}``. It posts to the store's intake,
``<site>/intake/bugreport``: the hosted site is behind a sign-in (Cloudflare
Access) that only that path is exempt from, so the forward follows no
redirect -- a redirect is the sign-in page, and is reported as such. Reading a
report back (``GET /api/bugreport/<id>``) needs the sign-in, so here it only
finds reports this server saved itself. On the way it adds what only the advisor has: the
latest position the game sent (``state.live_game``, the same message as Copy
state, which the workbench's Import state rebuilds) and the run envelope
(``meta.advisor``). With no site, or when the store cannot be reached, the
dump is saved to ``<log dir>/../bugreports/<id>.json`` instead and the reply's
``url`` is that path. Only the page's own origin (or an allow-listed one) may
post: a foreign ``Origin`` is refused, and so is a body that is not
``application/json``, which a cross-site page could only send after a
preflight this server refuses.

The mod's copy of the build carries its ``.json`` files gzipped as
``<name>.json.gz``: the game reads every ``.json`` under a mod's folder as a
mod manifest. A request for ``x.json`` falls back to ``x.json.gz``, sent with
``content-encoding: gzip`` (decompressed here for a client that does not
accept gzip), so the page never sees the difference.
"""

from __future__ import annotations

import copy
import gzip
import io
import json
import logging
import re
import secrets
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from realgame_livetree import LiveRunTree
from realgame_workbench_log import run_log_name

logger = logging.getLogger(__name__)

#: Content types by extension. Explicit, not ``mimetypes``: Python on
#: Windows reads the registry and has no ``.wasm`` entry, and the wasm must go
#: out as ``application/wasm`` or ``WebAssembly.instantiateStreaming`` refuses it.
_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".webmanifest": "application/manifest+json",
    ".wasm": "application/wasm",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".mp3": "audio/mpeg",
    ".webm": "video/webm",
    ".ogg": "audio/ogg",
    ".txt": "text/plain; charset=utf-8",
    ".atlas": "text/plain; charset=utf-8",
}


class StaticSite:
    """A static workbench build served from disk (see the module docstring).

    Headers follow ``web/public/_headers``, the Cloudflare Pages rules the
    same build is deployed with: the art is WebP under ``.png`` names
    (``scripts/art_to_dist.py``), content-hashed ``wb-assets/`` are immutable,
    the shell is ``no-cache``.

    Args:
        root: The build (holds ``index.html`` and ``wb-precache.json``).
        ppo_dir: Where ``/ppo/<file>`` is looked up when the build does not
            carry it: the packaged mod strips the champion nets from the
            build and serves the copies the advisor bundles anyway.
    """

    def __init__(self, root: Path, ppo_dir: Path | None = None) -> None:
        self.root = root.resolve()
        self.ppo_dir = ppo_dir.resolve() if ppo_dir else None

    @staticmethod
    def is_static_build(root: Path) -> bool:
        """True for a browser-only build. ``wb-precache.json`` is written only
        by the static build (vite.config.ts ``swPrecache``); a server build of
        the same SPA would call ``/api/*`` here and fail on every request."""
        precache = root / "wb-precache.json"
        return (root / "index.html").is_file() and (
            precache.is_file() or precache.with_name(precache.name + ".gz").is_file()
        )

    def resolve(self, url_path: str) -> Path | None:
        """The file for a request path, or None. Never leaves the roots."""
        rel = url_path.lstrip("/") or "index.html"
        found = self._under(self.root, rel)
        if found is None and rel.endswith(".json"):
            found = self._under(self.root, rel + ".gz")
        if found is None and self.ppo_dir is not None and rel.startswith("ppo/"):
            found = self._under(self.ppo_dir, rel[len("ppo/") :])
        if found is None and "." not in rel.rsplit("/", 1)[-1]:
            found = self._under(self.root, "index.html")  # SPA fallback
        return found

    @staticmethod
    def _under(root: Path, rel: str) -> Path | None:
        try:
            path = (root / rel).resolve()
        except (OSError, ValueError):
            return None
        if path != root and root not in path.parents:
            return None
        return path if path.is_file() else None

    @staticmethod
    def gzipped(url_path: str, file: Path) -> bool:
        """True when ``file`` is the ``.json.gz`` stand-in for a ``.json``
        request (``resolve``): its bytes go out gzip-encoded."""
        return file.name.endswith(".json.gz") and not url_path.endswith(".gz")

    @staticmethod
    def headers(url_path: str, file: Path) -> dict[str, str]:
        ext = ".json" if StaticSite.gzipped(url_path, file) else file.suffix.lower()
        ctype = _MIME.get(ext, "application/octet-stream")
        if url_path.startswith("/assets/") and ext == ".png":
            ctype = "image/webp"
        if url_path.startswith("/ppo/") and ext != ".json":
            ctype = "application/octet-stream"
        if url_path.startswith(("/wb-assets/", "/ppo/")) and url_path != "/ppo/index.json":
            cache = "public, max-age=31536000, immutable"
        elif url_path.startswith(("/assets/", "/fonts/", "/icons/")):
            cache = "public, max-age=604800"
        else:
            cache = "no-cache"
        return {
            "content-type": ctype,
            "cache-control": cache,
            "x-content-type-options": "nosniff",
        }


#: The store's body cap (``web/functions/api/_bugreport.ts`` MAX_BODY_BYTES).
BUGREPORT_MAX_BYTES = 20 * 1024 * 1024
#: A report id (the store's ``ID_RE``): the only key a GET may name.
BUGREPORT_ID_RE = re.compile(r"^wb-[0-9a-f]{16}$")
#: Cloudflare's bot check answers 403 to urllib's default agent.
BUGREPORT_USER_AGENT = "SpireMentor-advisor/1 (bundled workbench bug-report forward)"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turn every redirect into an ``HTTPError`` instead of following it."""

    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any):
        return None


def _redirect_text(exc: urllib.error.HTTPError) -> str:
    host = urllib.parse.urlsplit(exc.headers.get("Location") or "").hostname or ""
    return (
        f"HTTP {exc.code}: the report store sent us to {host} (a sign-in page?) -- "
        "its /intake/bugreport is not exempt from the site's sign-in"
        if host
        else f"HTTP {exc.code}: the report store redirected instead of answering"
    )


#: The run envelope fields a forwarded report is stamped with (``/status``
#: minus ``log_path``, which names the player's home directory).
_STATUS_FIELDS = ("connected", "character", "seed", "floor", "act", "state_id")


class BugReportForward:
    """The bundled workbench's ``/api/bugreport`` (see the module docstring).

    Args:
        site: The hosted workbench origin whose ``/intake/bugreport`` stores
            reports (``https://...``, no trailing slash needed), or None.
        local_dir: Where a report is saved when there is no site or the
            store cannot be reached.
        urlopen: How to open a request (default: urllib following no
            redirects).
        timeout: Seconds per store call.
    """

    def __init__(
        self,
        site: str | None,
        local_dir: Path,
        *,
        urlopen: Callable[..., Any] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.site = site.strip().rstrip("/") if site and site.strip() else None
        self.local_dir = local_dir
        self._urlopen = urlopen or urllib.request.build_opener(_NoRedirect()).open
        self.timeout = timeout

    def probe(self) -> dict[str, Any]:
        """``GET /api/bugreport``: the JSON the page's probe wants (``ok``)."""
        return {"ok": True, "store": "forward" if self.site else "local"}

    def submit(
        self,
        dump: dict[str, Any],
        *,
        live_state: dict[str, Any] | None,
        status: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        """Enrich ``dump`` and store it; (HTTP status, reply)."""
        problem = _validate_dump(dump)
        if problem:
            return 422, {"detail": problem}
        meta = dump["meta"]
        meta["forwarded_by"] = "advisor"
        meta["advisor"] = {k: status.get(k) for k in _STATUS_FIELDS if k in status}
        state = dump["state"]
        if live_state is not None and "live_game" not in state:
            state["live_game"] = live_state
        body = json.dumps(dump, default=str).encode("utf-8")
        if len(body) > BUGREPORT_MAX_BYTES:
            return 413, {
                "detail": f"report too large ({len(body)} bytes; limit {BUGREPORT_MAX_BYTES})"
            }
        why = "this build has no bug-report address"
        if self.site:
            try:
                reply = self._post(body)
            except Exception as exc:
                why = f"could not reach the report store: {exc}"
                logger.warning("bug report forward to %s failed: %s", self.site, exc)
            else:
                logger.info("bug report forwarded: %s", reply.get("id"))
                return 201, reply
        return self._save_local(dump, why)

    def fetch(self, report_id: str) -> tuple[int, bytes]:
        """``GET /api/bugreport/<id>``: a report this server saved itself.

        The store's copies are behind the hosted site's sign-in, which a
        server cannot pass; a miss says where to read them instead.
        """
        if not BUGREPORT_ID_RE.match(report_id):
            return 404, _detail(f"unknown report: {report_id}")
        path = self.local_dir / f"{report_id}.json"
        if path.is_file():
            return 200, path.read_bytes()
        if self.site:
            return 404, _detail(
                f"{report_id} is not saved here; a sent report opens on the hosted "
                f"workbench (signed in): {self.site}/?report={report_id}"
            )
        return 404, _detail(f"unknown report: {report_id}")

    def _post(self, body: bytes) -> dict[str, Any]:
        # The intake, with no Origin header: the store takes a non-browser
        # client's post there, and the site's sign-in exempts that path.
        req = urllib.request.Request(
            f"{self.site}/intake/bugreport",
            data=body,
            method="POST",
            headers={"content-type": "application/json", "User-Agent": BUGREPORT_USER_AGENT},
        )
        try:
            with self._urlopen(req, timeout=self.timeout) as res:
                reply = json.loads(res.read())
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                raise RuntimeError(_redirect_text(exc)) from exc
            raise RuntimeError(f"HTTP {exc.code}: {_error_text(exc)}") from exc
        if not (isinstance(reply, dict) and isinstance(reply.get("id"), str) and reply.get("url")):
            raise RuntimeError("the store's reply was not {id, url}")
        return {"id": reply["id"], "url": reply["url"]}

    def _save_local(self, dump: dict[str, Any], why: str) -> tuple[int, dict[str, Any]]:
        report_id = f"wb-{secrets.token_hex(8)}"
        meta = dump["meta"]
        meta["id"] = report_id
        meta["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["received_via"] = "advisor-local"
        try:
            self.local_dir.mkdir(parents=True, exist_ok=True)
            path = self.local_dir / f"{report_id}.json"
            path.write_text(json.dumps(dump, default=str), encoding="utf-8")
        except OSError as exc:
            return 500, {"detail": f"{why}, and saving it failed: {exc}"}
        logger.info("bug report saved locally: %s (%s)", path, why)
        return 201, {
            "id": report_id,
            "url": str(path),
            "path": str(path),
            "detail": f"Not sent ({why}). Saved to {path} -- attach that file to your report.",
        }


def _validate_dump(dump: Any) -> str | None:
    """The store's schema check (``_bugreport.ts`` validateDump), here so a
    bad dump fails before it is enriched or saved."""
    if not isinstance(dump, dict):
        return "body must be a JSON object"
    if dump.get("schema_version") != 1:
        return "schema_version must be 1"
    for key in ("meta", "repro", "state", "ui"):
        if not isinstance(dump.get(key), dict):
            return f"{key} must be an object"
    if dump.get("note") is not None and not isinstance(dump.get("note"), str):
        return "note must be a string or null"
    return None


def _detail(msg: str) -> bytes:
    return json.dumps({"detail": msg}).encode("utf-8")


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        text = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return exc.reason or ""
    try:
        detail = json.loads(text).get("detail")
        if isinstance(detail, str):
            return detail
    except (ValueError, AttributeError):
        pass
    return text[:200]


#: Message types the advisor's receive loop must route to the control plane.
BRIDGE_MESSAGE_TYPES = (
    "snapshots",
    "snapshot_saved",
    "snapshot_restored",
    "combat_undone",
    "error",
)


class _Waiter:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.payload: dict[str, Any] | None = None

    def fulfill(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.event.set()


class ControlPlane:
    """State shared between the advisor loop and the HTTP handler threads."""

    def __init__(self, tree: LiveRunTree) -> None:
        self.tree = tree
        #: Page origins allowed to call in from a browser (``start`` sets it).
        self.allow_origins: tuple[str, ...] = ()
        self._lock = threading.Lock()
        self._client: Any = None
        self._status: dict[str, Any] = {"connected": False}
        self._waiters: dict[str, _Waiter] = {}
        self._last_state: dict[str, Any] | None = None
        self._server: ThreadingHTTPServer | None = None
        #: The bundled workbench build, if any (``start`` sets it).
        self.site: StaticSite | None = None
        #: The bundled workbench's bug-report forward, if any (``start``).
        self.bug_reports: BugReportForward | None = None
        self.search_tree_fn: Any = None  # set by the advisor: (sims|None) -> dict

    # -- called from the advisor loop ----------------------------------

    def set_client(self, client: Any) -> None:
        with self._lock:
            self._client = client
            self._status["connected"] = client is not None

    def note_message(self, msg: dict[str, Any]) -> None:
        """Track the latest run envelope for /status (any message type), and
        the latest decision position for /state."""
        rs = msg.get("run_state") or {}
        mtype = msg.get("type")
        # Copied: the advisor's handlers may annotate the message they get.
        state = copy.deepcopy(msg) if mtype == "advice_request" else None
        with self._lock:
            if state is not None:
                self._last_state = state
            elif mtype in ("game_over", "run_complete"):
                self._last_state = None
            self._status.update(
                {
                    "connected": True,
                    "character": rs.get("character") or self._status.get("character"),
                    "seed": msg.get("seed") or self._status.get("seed"),
                    "floor": msg.get("floor", self._status.get("floor")),
                    "act": msg.get("act", self._status.get("act")),
                    "state_id": msg.get("state_id", self._status.get("state_id")),
                }
            )

    def last_state(self) -> dict[str, Any] | None:
        """The latest ``advice_request`` (see /state), or None."""
        with self._lock:
            return self._last_state

    def set_log_path(self, path: str | None) -> None:
        with self._lock:
            self._status["log_path"] = path

    def on_bridge_message(self, msg: dict[str, Any]) -> None:
        """Route a rewind-lane broadcast from the mod (see BRIDGE_MESSAGE_TYPES)."""
        mtype = msg.get("type")
        if mtype == "snapshot_saved":
            self.tree.on_snapshot_saved(msg)
            return
        if mtype == "snapshots":
            self._fulfill("snapshots", msg)
            return
        if mtype == "snapshot_restored":
            self.tree.on_snapshot_restored(msg)
            self._fulfill("restore", msg)
            return
        if mtype == "combat_undone":
            self.tree.on_combat_undone(msg)
            self._fulfill("restore", msg)
            return
        if mtype == "error" and (
            msg.get("snapshot_id") is not None or str(msg.get("error") or "").startswith("undo_")
        ):
            self._fulfill("restore", msg)

    def _fulfill(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            waiter = self._waiters.pop(kind, None)
        if waiter is not None:
            waiter.fulfill(payload)

    # -- command round trips (HTTP handler threads) --------------------

    def _round_trip(
        self, kind: str, action: dict[str, Any], timeout: float
    ) -> dict[str, Any] | None:
        with self._lock:
            client = self._client
            if client is None:
                return None
            waiter = _Waiter()
            self._waiters[kind] = waiter
        try:
            client.send_action(action, echo_state_id=False)
        except Exception as exc:
            with self._lock:
                self._waiters.pop(kind, None)
            logger.warning("control command %s failed: %s", action.get("action"), exc)
            return None
        if not waiter.event.wait(timeout):
            with self._lock:
                self._waiters.pop(kind, None)
            return {"type": "error", "error": "timeout", "detail": f"{kind} timed out"}
        return waiter.payload

    def list_snapshots(self) -> dict[str, Any] | None:
        return self._round_trip("snapshots", {"action": "list_snapshots"}, timeout=5.0)

    def restore(self, snapshot_id: str) -> dict[str, Any] | None:
        return self._round_trip(
            "restore",
            {"action": "restore_snapshot", "snapshot_id": snapshot_id},
            timeout=20.0,
        )

    def undo_turn(self, turn: int, tape_id: str | None = None) -> dict[str, Any] | None:
        action: dict[str, Any] = {"action": "undo_turn", "turn": int(turn)}
        if tape_id:
            action["tape_id"] = tape_id
        # A persisted-tape replay reloads a room AND fast-forwards a whole
        # combat prefix; give it more room than a plain restore.
        return self._round_trip("restore", action, timeout=60.0)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def live_log(self, *, stat_only: bool) -> tuple[int, dict[str, Any]]:
        """``GET /log``: the current run log, as (HTTP status, payload)."""
        with self._lock:
            path = self._status.get("log_path")
        if not path:
            return 404, {"error": "no_log", "detail": "the advisor has not opened a run log yet"}
        file = Path(path)
        try:
            st = file.stat()
            raw = b"" if stat_only else file.read_bytes()
        except OSError as exc:
            return 404, {"error": "no_log", "detail": str(exc)}
        # The bundled page stores it under this key (its run store takes a
        # bare filename, engineRoutes.ts RUN_FILENAME); coach records name
        # their file by the same rule.
        name = run_log_name(str(file)) or ""
        payload: dict[str, Any] = {"name": name, "size": st.st_size, "mtime": st.st_mtime}
        if not stat_only:
            # The advisor appends while we read: a torn last record would
            # fail the page's parse, so it waits for the next poll.
            payload["text"] = raw[: raw.rfind(b"\n") + 1].decode("utf-8", errors="replace")
        return 200, payload

    # -- HTTP server ---------------------------------------------------

    def start(
        self,
        port: int,
        allow_origins: Any = (),
        site: StaticSite | None = None,
        bug_reports: BugReportForward | None = None,
    ) -> None:
        """Serve on ``127.0.0.1:port``.

        Args:
            port: TCP port (``0`` = ephemeral; see :attr:`port`).
            allow_origins: Page origins (``https://your-workbench.example.com``)
                allowed to call this server from a browser. Each gets the CORS
                headers and the Private Network Access opt-in on every reply
                and on the preflight; anything else gets none. Default: none.
            site: The bundled workbench build to host at ``/`` (see the
                module docstring). Default: none, every other path is a 404.
            bug_reports: Serve ``/api/bugreport`` through this forward (see
                the module docstring). Default: none -- the route is a 404,
                which keeps the page's Report button hidden.
        """
        plane = self
        self.allow_origins = tuple(
            o.strip().rstrip("/") for o in (allow_origins or ()) if str(o).strip()
        )
        self.site = site
        self.bug_reports = bug_reports

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("control: " + format, *args)

            def _cors_headers(self) -> None:
                """CORS + Private Network Access, for an allow-listed Origin only.

                The allow-list is exact-match on the page origin; the reply
                echoes it (never ``*``), with ``Vary: Origin`` so a cache keeps
                the two answers apart. ``Access-Control-Allow-Private-Network``
                is Chrome's PNA opt-in: a public-origin page fetching a
                loopback address preflights and needs this on the preflight
                reply (https://wicg.github.io/private-network-access/).
                """
                origin = self.headers.get("Origin")
                if not origin or origin.rstrip("/") not in plane.allow_origins:
                    return
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "content-type")
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Access-Control-Max-Age", "600")
                self.send_header("Vary", "Origin")

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, default=str).encode("utf-8")
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self._cors_headers()
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:
                """The CORS / PNA preflight. 204 with the headers for an
                allow-listed origin; 403 (no CORS headers) for any other, so
                the browser refuses the real request before it is sent."""
                origin = self.headers.get("Origin")
                if origin and origin.rstrip("/") in plane.allow_origins:
                    self.send_response(204)
                    self.send_header("content-length", "0")
                    self._cors_headers()
                    self.end_headers()
                else:
                    self._send(403, {"error": "origin not allowed"})

            def _send_bytes(self, code: int, body: bytes) -> None:
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.send_header("cache-control", "no-store")
                self._cors_headers()
                self.end_headers()
                self.wfile.write(body)

            def _post_bug_report(self, forward: BugReportForward) -> None:
                """``POST /api/bugreport`` (see the module docstring)."""
                origin = self.headers.get("Origin")
                own = f"http://{self.headers.get('Host', '')}"
                if origin and origin.rstrip("/") not in (own, *plane.allow_origins):
                    self._send(
                        403, {"detail": "bug reports are accepted from the workbench page only"}
                    )
                    return
                ctype = (self.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
                if ctype != "application/json":
                    self._send(415, {"detail": "a bug report is application/json"})
                    return
                length = int(self.headers.get("content-length") or 0)
                if length > BUGREPORT_MAX_BYTES:
                    self._send(413, {"detail": f"report too large ({length} bytes)"})
                    return
                try:
                    dump = json.loads(self.rfile.read(length)) if length > 0 else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self._send(400, {"detail": "body is not JSON"})
                    return
                code, reply = forward.submit(
                    dump, live_state=plane.last_state(), status=plane.status()
                )
                self._send(code, reply)

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("content-length") or 0)
                if length <= 0:
                    return {}
                try:
                    return json.loads(self.rfile.read(length)) or {}
                except json.JSONDecodeError:
                    return {}

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/status":
                    self._send(200, plane.status())
                elif path == "/tree":
                    self._send(200, plane.tree.timeline())
                elif path == "/state":
                    state = plane.last_state()
                    if state is None:
                        self._send(
                            404,
                            {
                                "error": "no_state",
                                "detail": "no decision from the game yet "
                                "(the mod sends one at each combat turn or choice screen)",
                            },
                        )
                    else:
                        self._send(200, state)
                elif path == "/snapshots":
                    reply = plane.list_snapshots()
                    if reply is None:
                        self._send(503, {"error": "bridge not connected"})
                    else:
                        self._send(200, reply)
                elif path == "/log":
                    code, payload = plane.live_log(stat_only="stat=1" in self.path)
                    self._send(code, payload)
                elif path == "/api/bugreport" and plane.bug_reports is not None:
                    self._send(200, plane.bug_reports.probe())
                elif path.startswith("/api/bugreport/") and plane.bug_reports is not None:
                    code, body = plane.bug_reports.fetch(unquote(path[len("/api/bugreport/") :]))
                    self._send_bytes(code, body)
                elif path == "/api/health" and plane.site is not None:
                    self._send(200, {"ok": True, "workbench": "static"})
                elif not self._send_file(path, head=False):
                    self._send(404, {"error": f"unknown path {path}"})

            def do_HEAD(self) -> None:
                if not self._send_file(self.path.split("?", 1)[0], head=True):
                    self.send_response(404)
                    self.send_header("content-length", "0")
                    self.end_headers()

            def _send_file(self, path: str, *, head: bool) -> bool:
                """Serve a file of the bundled workbench; False if there is none."""
                site = plane.site
                if site is None or path.startswith("/api/"):
                    return False
                file = site.resolve(unquote(path))
                if file is None:
                    return False
                encoding = None
                try:
                    if site.gzipped(path, file):
                        accept = self.headers.get("accept-encoding", "")
                        if "gzip" in accept.lower():
                            encoding = "gzip"
                            fh = file.open("rb")
                        else:
                            fh = io.BytesIO(gzip.decompress(file.read_bytes()))
                    else:
                        fh = file.open("rb")
                    size = fh.seek(0, io.SEEK_END)
                    fh.seek(0)
                except (OSError, EOFError):
                    return False
                with fh:
                    self.send_response(200)
                    for name, value in site.headers(path, file).items():
                        self.send_header(name, value)
                    if site.gzipped(path, file):
                        self.send_header("vary", "accept-encoding")
                    if encoding:
                        self.send_header("content-encoding", encoding)
                    self.send_header("content-length", str(size))
                    self.end_headers()
                    if not head:
                        try:
                            shutil.copyfileobj(fh, self.wfile, 256 * 1024)
                        except (BrokenPipeError, ConnectionResetError):
                            pass  # the browser cancelled (navigation, SW abort)
                return True

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/api/bugreport" and plane.bug_reports is not None:
                    self._post_bug_report(plane.bug_reports)
                    return
                body = self._body()
                if path == "/restore":
                    snapshot_id = body.get("snapshot_id")
                    node_id = body.get("node_id")
                    if snapshot_id is None and node_id is not None:
                        plan = plane.tree.seek_plan(int(node_id))
                        if plan is None:
                            self._send(
                                409,
                                {
                                    "error": "not_seekable",
                                    "detail": "that node carries no game save point "
                                    "(the game only saves at room entry and "
                                    "post-combat rewards)",
                                },
                            )
                            return
                        if plan["action"] == "undo_turn":
                            reply = plane.undo_turn(plan["turn"], plan.get("tape_id"))
                            if reply is None:
                                self._send(503, {"error": "bridge not connected"})
                            elif reply.get("type") == "combat_undone":
                                self._send(200, reply)
                            else:
                                self._send(409, reply)
                            return
                        snapshot_id = plan["snapshot_id"]
                    if not snapshot_id:
                        self._send(400, {"error": "missing snapshot_id/node_id"})
                        return
                    reply = plane.restore(str(snapshot_id))
                    if reply is None:
                        self._send(503, {"error": "bridge not connected"})
                    elif reply.get("type") == "snapshot_restored":
                        self._send(200, reply)
                    else:
                        self._send(409, reply)
                elif path == "/search_tree":
                    fn = plane.search_tree_fn
                    if fn is None:
                        self._send(503, {"error": "search unavailable"})
                        return
                    try:
                        tree = fn(body.get("sims"))
                    except LookupError as exc:
                        self._send(409, {"error": "no_combat_state", "detail": str(exc)})
                        return
                    except Exception as exc:
                        self._send(500, {"error": "search_failed", "detail": str(exc)})
                        return
                    self._send(200, tree)
                else:
                    self._send(404, {"error": f"unknown path {path}"})

        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError as exc:
            logger.warning("control plane could not bind 127.0.0.1:%d: %s", port, exc)
            return
        self._server = server
        thread = threading.Thread(target=server.serve_forever, name="advisor-control", daemon=True)
        thread.start()
        logger.info(
            "control plane listening on 127.0.0.1:%d (browser origins: %s)",
            self.port or port,
            ", ".join(self.allow_origins) or "none",
        )

    @property
    def port(self) -> int | None:
        """The bound port (differs from the requested one when it was 0)."""
        return self._server.server_address[1] if self._server else None

    def stop(self) -> None:
        """Shut the HTTP server down (tests; the advisor never stops it)."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
