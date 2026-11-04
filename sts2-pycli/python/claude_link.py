"""Claude link: lets a terminal Claude Code session see and drive the workbench.

The workbench (browser build or the server-backed one) holds the run you are
looking at — in the browser build it lives only in that tab's Web Worker and
IndexedDB, so no process on the desk can read it. This module bridges the gap:

    Claude Code ──stdio MCP──► this relay ◄──WebSocket (loopback)── workbench tab

It is an MCP server (Claude Code starts it from ``.mcp.json``) that also
listens on ``ws://127.0.0.1:<port>`` (default 8091, ``CLAUDE_LINK_PORT``).
The page connects to it once the user turns on Tools → "🤖 Claude link"
(``web/src/claude/``). Every tool call becomes one request to the page, which
answers it through the same ``apiFetch`` door its own panels use — so Claude
sees exactly the run on screen, and its analyses run in the same engine.

Tools:
    wb_status   — is a page connected; which port; what URL.
    wb_context  — what the user is looking at (mode + the shown step).
    wb_call     — any workbench ``/api`` route against the live session.
    wb_show     — put a chart / explanation into the workbench's Claude dock.

Security: a page must PAIR before it can attach. The relay keeps a random
token in ``~/.config/spire-mentor/claude-link-token`` (0600, created on first
use; ``CLAUDE_LINK_TOKEN`` overrides it) and closes any connection whose hello
does not carry it (code 4003). Without that, any local process — or a page
on any localhost port — could attach as "the page", feed Claude a made-up
run (prompt injection) and receive what Claude shows (``wb_show(file=…)``
reads local files). The token persists, so a browser pairs once per
workbench origin: Claude hands it over (``wb_status``), the user pastes it at
Tools → "🤖 Claude link" or opens the workbench with ``#claude-link=<token>``.
On top of that the socket binds loopback only and refuses a browser
``Origin`` other than a localhost page or one listed in
``CLAUDE_LINK_ORIGINS`` (comma-separated, e.g. a hosted workbench origin);
a sandboxed iframe — the dock renders everything Claude shows in one — has
the opaque origin ``null`` and is refused too.

Only one relay can hold the port. A second Claude Code session's relay keeps
retrying the bind every few seconds and its tools say so, so the link moves
to it once the first session exits.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import itertools
import json
import logging
import os
import secrets
import stat
import sys
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from mcp.server.fastmcp import FastMCP
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

log = logging.getLogger("claude_link")

DEFAULT_PORT = 8091
PROTOCOL_VERSION = 1
BIND_RETRY_S = 5.0
HELLO_TIMEOUT_S = 5.0
#: Close code for a hello without the pairing token.
BAD_TOKEN_CODE = 4003
#: Page replies can be whole run trees or sweep results.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
#: Above this, a tool result is truncated (pass ``save_to`` for the whole).
MAX_RESULT_CHARS = 60_000
#: A call the relay stopped waiting for still holds the page's engine (the
#: browser build's worker cannot interrupt a search). New calls are refused
#: while it runs, but not forever: a page that never answers is given up on.
ABANDONED_TTL_S = 1800.0
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

#: Render-only fields of the workbench wire: art paths, rig geometry, the
#: rich-text twin of every description. Dropped unless ``raw=True``.
_DROP_KEYS = frozenset(
    {
        "image",
        "rig",
        "rig_bounds_w",
        "rig_bounds_h",
        "rig_scale",
        "description_rich",
        "assets_base",
        "size_jitter",
        "hue_jitter",
    }
)

_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}


class LinkError(RuntimeError):
    """The page is not reachable, or it answered with an error."""


def origin_allowed(origin: str | None, extra: frozenset[str]) -> bool:
    """Whether a WebSocket handshake's ``Origin`` may attach as the page.

    No ``Origin`` is a non-browser client on this machine, which already has
    local access. ``null`` (a sandboxed iframe, a ``file:`` page) is refused.
    """
    if origin is None:
        return True
    if origin in extra:
        return True
    parts = urlsplit(origin)
    return parts.scheme in ("http", "https") and parts.hostname in _LOCAL_HOSTS


def extra_origins_from_env() -> frozenset[str]:
    raw = os.environ.get("CLAUDE_LINK_ORIGINS", "")
    return frozenset(o.strip().rstrip("/") for o in raw.split(",") if o.strip())


def token_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "spire-mentor" / "claude-link-token"


def load_or_create_token(path: Path | None = None) -> str:
    """The pairing token: ``CLAUDE_LINK_TOKEN``, else the token file (created
    0600 on first use). Persistent, so a browser pairs once, not per session."""
    env = os.environ.get("CLAUDE_LINK_TOKEN", "").strip()
    if env:
        return env
    path = path or token_path()
    existing = _read_token(path)
    if existing:
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a private temp file, then hard-linked into place: the link
    # fails if the file exists, so two relays starting together agree on ONE
    # token (the loser reads the winner's) and no reader sees a partial write.
    token = secrets.token_urlsafe(18)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token + "\n")
        try:
            os.link(tmp, path)
        except FileExistsError:
            return _read_token(path) or token
    finally:
        tmp.unlink(missing_ok=True)
    return token


def _read_token(path: Path) -> str | None:
    """The token in an existing file, tightening a loose mode to 0600 first."""
    if not path.is_file():
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        path.chmod(0o600)
    return path.read_text(encoding="utf-8").strip() or None


def compact(value: Any) -> Any:
    """Strip render-only fields (art, rigs, rich text) from a wire payload."""
    if isinstance(value, dict):
        return {
            k: compact(v)
            for k, v in value.items()
            if k not in _DROP_KEYS and not k.endswith("_image")
        }
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def compact_stream(frames: list[Any], keep_progress: int = 2) -> dict[str, Any]:
    """An NDJSON advisor stream, minus all but the last few progress frames."""
    final = [f for f in frames if isinstance(f, dict) and f.get("type") in ("result", "error")]
    progress = [f for f in frames if f not in final]
    return {
        "progress_frames": len(progress),
        "last_progress": progress[-keep_progress:],
        "final": final,
    }


def render_result(value: Any, *, raw: bool, save_to: str | None) -> str:
    """JSON text for a tool result: compacted, saved to disk, or truncated."""
    if save_to:
        path = Path(save_to).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(value, indent=1, ensure_ascii=False)
        path.write_text(text, encoding="utf-8")
        keys = list(value) if isinstance(value, dict) else None
        return json.dumps(
            {"saved_to": str(path.resolve()), "bytes": len(text.encode()), "top_level_keys": keys}
        )
    shown = value if raw else compact(value)
    text = json.dumps(shown, separators=(",", ":"), ensure_ascii=False)
    if len(text) > MAX_RESULT_CHARS:
        return (
            text[:MAX_RESULT_CHARS]
            + f"\n…[truncated: {len(text)} chars total — call again with save_to=<path> and read the file]"
        )
    return text


def show_payload(
    title: str,
    *,
    html: str | None = None,
    markdown: str | None = None,
    plotly: dict[str, Any] | None = None,
    file: str | None = None,
) -> dict[str, Any]:
    """The dock item for ``wb_show``: exactly one content source → html | plotly."""
    given = [
        k
        for k, v in (("html", html), ("markdown", markdown), ("plotly", plotly), ("file", file))
        if v
    ]
    if len(given) != 1:
        raise LinkError(
            f"wb_show needs exactly one of html / markdown / plotly / file (got {given or 'none'})"
        )
    if file:
        path = Path(file).expanduser()
        if not path.is_file():
            raise LinkError(f"no such file: {path}")
        suffix = path.suffix.lower()
        if suffix in (".html", ".htm"):
            html = path.read_text(encoding="utf-8")
        elif suffix in (".md", ".markdown", ".txt"):
            markdown = path.read_text(encoding="utf-8")
        elif suffix == ".json":
            plotly = json.loads(path.read_text(encoding="utf-8"))
        elif suffix in _IMAGE_TYPES:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            html = (
                '<!doctype html><html><body style="margin:0;background:#fff">'
                f'<img alt="{path.name}" style="max-width:100%;display:block" '
                f'src="data:{_IMAGE_TYPES[suffix]};base64,{data}"></body></html>'
            )
        else:
            raise LinkError(f"wb_show cannot display {suffix or 'extension-less'} files")
    if plotly is not None:
        if not isinstance(plotly, dict) or "data" not in plotly:
            raise LinkError('plotly must be a figure dict: {"data": [...], "layout": {...}}')
        return {"title": title, "kind": "plotly", "content": plotly}
    if markdown is not None:
        return {"title": title, "kind": "markdown", "content": markdown}
    return {"title": title, "kind": "html", "content": html}


@dataclass
class PageHub:
    """The one workbench page attached to this relay, and its open requests."""

    token: str
    port: int = DEFAULT_PORT
    origins: frozenset[str] = frozenset()
    bound: bool = False
    bind_error: str | None = None
    hello: dict[str, Any] = field(default_factory=dict)
    _ws: ServerConnection | None = None
    _pending: dict[int, tuple[ServerConnection, asyncio.Future[Any]]] = field(default_factory=dict)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))
    #: Timed-out calls the page is still working on: id → (page, route, when).
    _abandoned: dict[int, tuple[ServerConnection, str, float]] = field(default_factory=dict)

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def status(self) -> dict[str, Any]:
        return {
            "relay": f"ws://127.0.0.1:{self.port}",
            "listening": self.bound,
            "bind_error": self.bind_error,
            "page_connected": self.connected,
            "page": self.hello or None,
            "page_busy_with": self.busy_with(),
            "pairing": self.pairing_hint(),
        }

    def busy_with(self) -> str | None:
        """A timed-out call the attached page has not finished yet, if any."""
        now = time.monotonic()
        for rid, (owner, route, since) in list(self._abandoned.items()):
            if owner is not self._ws or now - since > ABANDONED_TTL_S:
                self._abandoned.pop(rid, None)
                continue
            return f"{route} (timed out {now - since:.0f}s ago)"
        return None

    def pairing_hint(self) -> str:
        return (
            f"pairing token: {self.token} — in the workbench, Tools → '🤖 Claude link' asks for it once "
            f"(or open the workbench URL with #claude-link={self.token} appended)"
        )

    def not_connected_reason(self) -> str:
        if not self.bound:
            return (
                f"this session's relay cannot listen on 127.0.0.1:{self.port} ({self.bind_error}). "
                "Most likely another Claude Code session holds the Claude link; it moves here when that "
                "session exits. CLAUDE_LINK_PORT picks another port (the page's Claude link URL must match)."
            )
        return (
            "no workbench page is connected. Open the workbench and turn on Tools → '🤖 Claude link' "
            f"(it connects to ws://127.0.0.1:{self.port}). If it has not paired yet, give the user the "
            f"{self.pairing_hint()}."
        )

    def process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        origin = request.headers.get("Origin")
        if origin_allowed(origin, self.origins):
            return None
        log.warning("refused a page from origin %s", origin)
        return connection.respond(403, f"origin {origin} is not allowed (CLAUDE_LINK_ORIGINS)\n")

    async def handler(self, ws: ServerConnection) -> None:
        try:
            first = json.loads(await asyncio.wait_for(ws.recv(), HELLO_TIMEOUT_S))
        except (TimeoutError, ConnectionClosed, json.JSONDecodeError):
            await ws.close(4001, "expected a hello message")
            return
        if not isinstance(first, dict) or first.get("type") != "hello":
            await ws.close(4001, "expected a hello message")
            return
        offered = first.get("token")
        if not isinstance(offered, str) or not hmac.compare_digest(
            offered.encode(), self.token.encode()
        ):
            log.warning("refused a page with a missing or wrong pairing token")
            await ws.close(BAD_TOKEN_CODE, "bad pairing token")
            return
        previous, self._ws = self._ws, ws
        self.hello = {k: first.get(k) for k in ("url", "mode", "static", "protocol")}
        log.info("page attached: %s", self.hello.get("url"))
        if previous is not None:
            with contextlib.suppress(Exception):
                await previous.close(4000, "replaced by a newer workbench tab")
        try:
            async for message in ws:
                self._on_message(message)
        except ConnectionClosed:
            pass
        finally:
            if self._ws is ws:
                self._ws = None
                self.hello = {}
            for rid, (owner, _, _) in list(self._abandoned.items()):
                if owner is ws:
                    self._abandoned.pop(rid, None)
            for rid, (owner, fut) in list(self._pending.items()):
                if owner is ws and not fut.done():
                    fut.set_exception(LinkError("the workbench page disconnected before answering"))
                    self._pending.pop(rid, None)

    def _on_message(self, message: str | bytes) -> None:
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict):
            return
        if msg.get("type") == "hello":  # mode changes re-announce themselves
            self.hello.update({k: msg.get(k) for k in ("url", "mode", "static") if k in msg})
            return
        if msg.get("type") != "res":
            return
        rid = msg.get("id")
        if isinstance(rid, int) and self._abandoned.pop(rid, None) is not None:
            log.info("the page finished timed-out request %d", rid)
        entry = self._pending.pop(rid, None) if isinstance(rid, int) else None
        if entry is None:
            return
        _, fut = entry
        if fut.done():
            return
        if msg.get("ok"):
            fut.set_result(msg.get("result"))
        else:
            fut.set_exception(LinkError(str(msg.get("error") or "the page reported an error")))

    async def request(self, op: str, *, timeout: float, **payload: Any) -> Any:
        ws = self._ws
        if ws is None:
            raise LinkError(self.not_connected_reason())
        if op == "call" and (busy := self.busy_with()):
            # It would only queue behind the abandoned one in the page's
            # single engine worker, and so would the user's own panels.
            raise LinkError(
                f"the page is still working on an earlier call that timed out: {busy}. "
                "Its result is discarded when it lands; new calls are refused until then. "
                "Wait and retry (wb_status shows it), or ask the user to reload the workbench tab "
                "to abort it."
            )
        rid = next(self._ids)
        fut: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[rid] = (ws, fut)
        try:
            await ws.send(json.dumps({"type": "req", "id": rid, "op": op, **payload}))
            return await asyncio.wait_for(fut, timeout)
        except TimeoutError as e:
            route = str(payload.get("route") or op)
            self._abandoned[rid] = (ws, route, time.monotonic())
            with contextlib.suppress(ConnectionClosed):
                await ws.send(json.dumps({"type": "cancel", "id": rid}))
            raise LinkError(
                f"the page did not answer '{route}' within {timeout:.0f}s. It was asked to cancel, "
                "but a search already running in the browser's engine finishes first; further "
                "wb_call requests are refused until it does."
            ) from e
        except ConnectionClosed as e:
            raise LinkError("the workbench page disconnected") from e
        finally:
            self._pending.pop(rid, None)

    async def serve_forever(self, host: str = "127.0.0.1") -> None:
        """Hold the port; while another process has it, keep retrying."""
        while True:
            try:
                async with serve(
                    self.handler,
                    host,
                    self.port,
                    process_request=self.process_request,
                    max_size=MAX_MESSAGE_BYTES,
                ):
                    self.bound, self.bind_error = True, None
                    log.info("listening on ws://%s:%d", host, self.port)
                    # Not `server.serve_forever()`: cancelled, that one waits for the
                    # open connections to drain BEFORE the websockets layer closes
                    # them, so a relay with a page attached never shut down. Parked
                    # here, a cancel exits `async with`, which closes the page's
                    # connection (1001) and then the server.
                    await asyncio.get_running_loop().create_future()
            except OSError as e:
                if self.bound or self.bind_error != str(e):
                    log.warning("cannot listen on %s:%d: %s (retrying)", host, self.port, e)
                self.bound, self.bind_error = False, str(e)
                await asyncio.sleep(BIND_RETRY_S)


INSTRUCTIONS = """\
The Claude link connects you to the Slay the Spire 2 workbench the user has open in their
browser (a web app over the Rust engine: live runs with a branch tree, advisors, replays,
SimLab). The run on screen exists only in that tab, so reach it through these tools.

- wb_context: what the user is looking at — `mode` (play / replay / mcts) and the shown step:
  `decision` (combat_play, card_reward, map_select, shop, event_choice, rest_site, …), `state`
  (hand, piles, enemies + intents, player HP/deck/relics/potions, options), AI `scores`.
  Start here when the user says "this fight", "this run", "here".
- wb_call(route, method, body): any workbench route, against the LIVE run. Useful ones:
    GET  /api/live/state                 the live step (same shape as wb_context's step)
    GET  /api/live/timeline              the run's branch tree (nodes, labels, active path)
    GET  /api/live/map                   the act's map graph
    GET  /api/live/deck-analysis         deck grades (damage / block / velocity)
    POST /api/live/hint {"config": {"sims": 2000}}         MCTS scores for each combat move
    POST /api/live/search_tree {"config": {"sims": 2000}}  the MCTS tree (combat only)
    POST /api/live/card-advisor/stream {"config": {}}      card-reward EV rollouts (slow)
    POST /api/live/decision-advisor/stream {"config": {}}  shop / event EV rollouts (slow)
    POST /api/live/ppo-policy            the PPO champion's distribution (run decisions)
    POST /api/live/wpa-review            win-probability curve over the run's decisions
    GET  /api/live/position              the current fight as a simlab-position/1 document
    GET  /api/live/persist               (browser build) the run tree: identity + action line
    POST /api/live/action {"action": verb, "args": {...}}  take an action (CHANGES the run)
    POST /api/live/seek {"node_id": n}  jump the branch tree (CHANGES what the user sees)
  Search configs take MctsConfig keys: sims, c_puct, depth, seed, hidden_info ("beliefdraw"
  is what the deployed agent runs), threads (the browser build runs 1). The route table
  lives in sts2-rs/crates/sts2-workbench/src/api.rs and web/src/api.ts.
  Ask before mutating the user's run (action / seek / start / quit); after a mutating call
  the page refreshes itself.
- wb_show(title, html | markdown | plotly | file): put a result in the workbench's Claude dock,
  next to the board. Prefer plotly figures ({"data": [...], "layout": {...}}) for charts.
  Everything renders in a sandboxed iframe (scripts allowed, no access to the page; plotly and
  markdown load their renderer from a CDN); file takes .html / .md / .png / .svg / .json
  (plotly). Showing a title again replaces that item, so iterate on one chart instead of
  stacking copies.
- If no page is connected, wb_status carries the pairing token to give the user (one-time
  per browser and workbench origin).

Big payloads: pass save_to=<path> to wb_context / wb_call and read the file. For experiments
heavier than the browser's single-threaded engine, export the position (save_to on
/api/live/position or /api/live/persist) and run the repo's Python / Rust tooling on it, then
wb_show the result.
"""


def build_server(hub: PageHub) -> FastMCP:
    @contextlib.asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncGenerator[None]:
        task = asyncio.create_task(hub.serve_forever())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    mcp = FastMCP("workbench", instructions=INSTRUCTIONS, lifespan=lifespan)

    @mcp.tool()
    def wb_status() -> dict[str, Any]:
        """Whether a workbench page is attached to this session, on which port, at what URL."""
        return hub.status()

    @mcp.tool()
    async def wb_context(raw: bool = False, save_to: str | None = None) -> str:
        """What the user is looking at in the workbench right now.

        Returns the view mode (play / replay / mcts) and the shown step: decision kind,
        full board / run state, the AI scores on screen. In replay mode also the run log's
        path and the viewed step index.

        Args:
            raw: keep render-only fields (art paths, rigs, rich text). Default strips them.
            save_to: write the full JSON to this path and return only a summary.
        """
        try:
            result = await hub.request("context", timeout=30)
        except LinkError as e:
            return f"error: {e}"
        return render_result(result, raw=raw, save_to=save_to)

    @mcp.tool()
    async def wb_call(
        route: str,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        raw: bool = False,
        save_to: str | None = None,
        timeout_s: float = 600,
    ) -> str:
        """Call a workbench /api route through the user's page (same engine, same live run).

        Streaming routes (…/stream) return their final frame plus a progress count.
        A non-GET call to /api/live/* makes the page refresh its view afterwards.

        Args:
            route: e.g. "/api/live/hint" (query string allowed).
            method: GET / POST / PUT / DELETE.
            body: JSON body for POST routes, e.g. {"config": {"sims": 4000}}.
            raw: keep render-only fields. Default strips them.
            save_to: write the full response (every stream frame) to this path and
                return only a summary.
            timeout_s: how long to wait for the page (rollout advisors can take minutes).
        """
        if not route.startswith("/api/") and route != "/api":
            return "error: route must start with /api/"
        try:
            result = await hub.request(
                "call", timeout=timeout_s, method=method.upper(), route=route, body=body
            )
        except LinkError as e:
            return f"error: {e}"
        if (
            isinstance(result, dict)
            and isinstance(result.get("frames"), list)
            and not raw
            and not save_to
        ):
            result = {"status": result.get("status"), **compact_stream(result["frames"])}
        return render_result(result, raw=raw, save_to=save_to)

    @mcp.tool()
    async def wb_show(
        title: str,
        html: str | None = None,
        markdown: str | None = None,
        plotly: dict[str, Any] | None = None,
        file: str | None = None,
    ) -> str:
        """Show a chart, table or explanation in the workbench's Claude dock.

        Give exactly one of: html (a full document; sandboxed iframe, scripts allowed),
        markdown, plotly (a figure dict {"data": [...], "layout": {...}}), or file (a path
        to .html / .md / .png / .jpg / .svg / .json-plotly). Showing the same title again
        replaces that item.
        """
        try:
            item = show_payload(title, html=html, markdown=markdown, plotly=plotly, file=file)
            await hub.request("show", timeout=30, item=item)
        except LinkError as e:
            return f"error: {e}"
        return f"shown in the workbench dock: {title}"

    return mcp


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="claude_link: %(message)s")
    port = int(os.environ.get("CLAUDE_LINK_PORT", DEFAULT_PORT))
    hub = PageHub(token=load_or_create_token(), port=port, origins=extra_origins_from_env())
    build_server(hub).run("stdio")


if __name__ == "__main__":
    main()
