"""The Claude link relay: origin gate, payload shaping, and the page round-trip.

The page side is played by a plain ``websockets`` client speaking the same
hello / req / res protocol as ``web/src/claude/link.ts``.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import claude_link
import pytest
from claude_link import (
    LinkError,
    PageHub,
    build_server,
    compact,
    origin_allowed,
    render_result,
    show_payload,
)
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- origin gate


@pytest.mark.parametrize(
    ("origin", "ok"),
    [
        (None, True),  # a local non-browser client
        ("http://localhost:5173", True),
        ("http://127.0.0.1:4173", True),
        ("http://[::1]:8079", True),
        ("https://localhost", True),
        ("null", False),  # sandboxed iframe — the dock's own Claude HTML
        ("https://evil.example", False),
        ("http://localhost.evil.example", False),
        ("file://", False),
    ],
)
def test_origin_gate(origin: str | None, ok: bool) -> None:
    assert origin_allowed(origin, frozenset()) is ok


def test_origin_gate_extra_allowlist() -> None:
    hosted = "https://wb.example.dev"
    assert not origin_allowed(hosted, frozenset())
    assert origin_allowed(hosted, frozenset({hosted}))


def test_extra_origins_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_LINK_ORIGINS", " https://a.dev/ ,https://b.dev,, ")
    assert claude_link.extra_origins_from_env() == frozenset({"https://a.dev", "https://b.dev"})


# ------------------------------------------------------------ payload shaping


def test_compact_strips_render_fields_only() -> None:
    step = {
        "assets_base": "/assets",
        "state": {
            "energy_image": "/x.png",
            "hand": [
                {
                    "name": "Strike",
                    "image": "/s.png",
                    "description": "Deal 6",
                    "description_rich": "[b]6",
                }
            ],
            "enemies": [
                {"name": "Nibbit", "hp": 44, "rig": "r", "rig_scale": 1.0, "intents": [{"dmg": 12}]}
            ],
        },
    }
    assert compact(step) == {
        "state": {
            "hand": [{"name": "Strike", "description": "Deal 6"}],
            "enemies": [{"name": "Nibbit", "hp": 44, "intents": [{"dmg": 12}]}],
        }
    }


def test_render_result_truncates_and_saves(tmp_path: Path) -> None:
    big = {"rows": ["x" * 100] * 1000}
    text = render_result(big, raw=False, save_to=None)
    assert len(text) < claude_link.MAX_RESULT_CHARS + 200
    assert "truncated" in text
    out = tmp_path / "sub" / "big.json"
    summary = json.loads(render_result(big, raw=False, save_to=str(out)))
    assert summary["top_level_keys"] == ["rows"]
    assert json.loads(out.read_text(encoding="utf-8")) == big


def test_render_result_save_keeps_raw_fields(tmp_path: Path) -> None:
    out = tmp_path / "s.json"
    render_result({"image": "/a.png", "hp": 3}, raw=False, save_to=str(out))
    assert json.loads(out.read_text(encoding="utf-8")) == {"image": "/a.png", "hp": 3}


def test_show_payload_needs_exactly_one_source() -> None:
    with pytest.raises(LinkError, match="exactly one"):
        show_payload("t")
    with pytest.raises(LinkError, match="exactly one"):
        show_payload("t", html="<p>", markdown="x")


def test_show_payload_kinds(tmp_path: Path) -> None:
    assert show_payload("a", markdown="# hi") == {
        "title": "a",
        "kind": "markdown",
        "content": "# hi",
    }
    fig = {"data": [{"y": [1, 2]}], "layout": {}}
    assert show_payload("b", plotly=fig)["kind"] == "plotly"
    with pytest.raises(LinkError, match="figure dict"):
        show_payload("c", plotly={"layout": {}})
    png = tmp_path / "chart.png"
    png.write_bytes(b"\x89PNG fake")
    item = show_payload("d", file=str(png))
    assert item["kind"] == "html"
    assert "data:image/png;base64," in item["content"]
    fig_file = tmp_path / "fig.json"
    fig_file.write_text(json.dumps(fig), encoding="utf-8")
    assert show_payload("e", file=str(fig_file)) == {"title": "e", "kind": "plotly", "content": fig}
    blob = tmp_path / "x.bin"
    blob.write_bytes(b"")
    with pytest.raises(LinkError, match="cannot display"):
        show_payload("f", file=str(blob))


# ------------------------------------------------------------- round trip


Answer = Callable[[dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]


TOKEN = "test-token"


async def _page(
    port: int,
    answer: Answer,
    *,
    origin: str | None = "http://localhost:5173",
    token: str | None = TOKEN,
) -> ClientConnection:
    ws = await connect(f"ws://127.0.0.1:{port}", origin=origin)  # ty: ignore[invalid-argument-type]
    hello: dict[str, Any] = {
        "type": "hello",
        "url": "http://localhost:5173/",
        "mode": "play",
        "static": True,
    }
    if token is not None:
        hello["token"] = token
    await ws.send(json.dumps(hello))

    async def serve() -> None:
        async for raw in ws:
            msg = json.loads(raw)
            out = answer(msg)
            reply: dict[str, Any] = await out if asyncio.iscoroutine(out) else out  # ty: ignore[invalid-assignment]
            await ws.send(json.dumps({"type": "res", "id": msg["id"], **reply}))

    asyncio.get_running_loop().create_task(serve())
    return ws


async def _with_hub(body: Callable[[PageHub, int], Awaitable[None]]) -> None:
    port = _free_port()
    hub = PageHub(token=TOKEN, port=port)
    task = asyncio.create_task(hub.serve_forever())
    try:
        for _ in range(100):
            if hub.bound:
                break
            await asyncio.sleep(0.01)
        assert hub.bound
        await body(hub, port)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def _wait_connected(hub: PageHub) -> None:
    for _ in range(100):
        if hub.connected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("page never attached")


def test_request_round_trip() -> None:
    async def body(hub: PageHub, port: int) -> None:
        with pytest.raises(LinkError, match="no workbench page is connected"):
            await hub.request("context", timeout=1)
        ws = await _page(
            port, lambda m: {"ok": True, "result": {"op": m["op"], "route": m.get("route")}}
        )
        await _wait_connected(hub)
        assert hub.status()["page"]["mode"] == "play"
        assert await hub.request("context", timeout=2) == {"op": "context", "route": None}
        assert await hub.request("call", timeout=2, method="GET", route="/api/live/state") == {
            "op": "call",
            "route": "/api/live/state",
        }
        await ws.close()

    asyncio.run(_with_hub(body))


def test_page_error_and_disconnect_fail_the_request() -> None:
    async def body(hub: PageHub, port: int) -> None:
        ws = await _page(port, lambda m: {"ok": False, "error": "HTTP 409 not in combat"})
        await _wait_connected(hub)
        with pytest.raises(LinkError, match="409 not in combat"):
            await hub.request("call", timeout=2, method="POST", route="/api/live/hint")
        await ws.close()

        hang = asyncio.Event()

        async def never(_m: dict[str, Any]) -> dict[str, Any]:
            await hang.wait()
            return {"ok": True}

        ws2 = await _page(port, never)
        await _wait_connected(hub)
        pending = asyncio.create_task(hub.request("context", timeout=5))
        await asyncio.sleep(0.05)
        await ws2.close()
        with pytest.raises(LinkError, match="disconnected"):
            await pending

    asyncio.run(_with_hub(body))


def test_newer_tab_replaces_older() -> None:
    async def body(hub: PageHub, port: int) -> None:
        first = await _page(port, lambda m: {"ok": True, "result": "first"})
        await _wait_connected(hub)
        await _page(port, lambda m: {"ok": True, "result": "second"})
        await asyncio.wait_for(first.wait_closed(), 2)
        assert first.close_code == 4000
        assert await hub.request("context", timeout=2) == "second"

    asyncio.run(_with_hub(body))


def test_foreign_origin_refused() -> None:
    async def body(_hub: PageHub, port: int) -> None:
        with pytest.raises(InvalidStatus) as err:
            await _page(port, lambda m: {"ok": True}, origin="https://evil.example")
        assert err.value.response.status_code == 403

    asyncio.run(_with_hub(body))


@pytest.mark.parametrize("token", [None, "wrong-token", ""])
def test_unpaired_page_is_refused_and_cannot_evict_the_paired_one(token: str | None) -> None:
    async def body(hub: PageHub, port: int) -> None:
        await _page(port, lambda m: {"ok": True, "result": "paired"})
        await _wait_connected(hub)
        intruder = await _page(
            port, lambda m: {"ok": True, "result": "intruder"}, origin=None, token=token
        )
        await asyncio.wait_for(intruder.wait_closed(), 2)
        assert intruder.close_code == claude_link.BAD_TOKEN_CODE
        assert await hub.request("context", timeout=2) == "paired"

    asyncio.run(_with_hub(body))


def test_token_file_is_created_private_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLAUDE_LINK_TOKEN", raising=False)
    path = tmp_path / "cfg" / "claude-link-token"
    first = claude_link.load_or_create_token(path)
    assert len(first) >= 20
    assert claude_link.load_or_create_token(path) == first
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    monkeypatch.setenv("CLAUDE_LINK_TOKEN", " from-env ")
    assert claude_link.load_or_create_token(path) == "from-env"


def test_status_carries_the_pairing_token() -> None:
    hub = PageHub(token="abc123", bound=True)
    assert "#claude-link=abc123" in hub.status()["pairing"]
    assert "abc123" in hub.not_connected_reason()


def test_second_relay_reports_the_busy_port() -> None:
    async def body(hub: PageHub, port: int) -> None:
        other = PageHub(token=TOKEN, port=port)
        task = asyncio.create_task(other.serve_forever())
        await asyncio.sleep(0.1)
        assert not other.bound
        with pytest.raises(LinkError, match="another Claude Code session"):
            await other.request("context", timeout=1)
        task.cancel()

    asyncio.run(_with_hub(body))


# ------------------------------------------------------------- MCP tools


def _tool_text(result: Any) -> str:
    content = result[0] if isinstance(result, tuple) else result
    return "".join(getattr(c, "text", "") for c in content)


def test_tools_end_to_end() -> None:
    seen: list[dict[str, Any]] = []

    def answer(m: dict[str, Any]) -> dict[str, Any]:
        seen.append(m)
        if m["op"] == "context":
            return {
                "ok": True,
                "result": {"mode": "play", "step": {"decision": "combat_play", "image": "/a.png"}},
            }
        if m["op"] == "call":
            frames = [{"type": "tick", "done": i} for i in range(50)] + [
                {"type": "result", "candidates": [1]}
            ]
            return {"ok": True, "result": {"status": 200, "frames": frames}}
        return {"ok": True, "result": None}

    async def body(hub: PageHub, port: int) -> None:
        mcp = build_server(hub)
        await _page(port, answer)
        await _wait_connected(hub)

        ctx = json.loads(_tool_text(await mcp.call_tool("wb_context", {})))
        assert ctx == {"mode": "play", "step": {"decision": "combat_play"}}

        out = json.loads(
            _tool_text(
                await mcp.call_tool(
                    "wb_call",
                    {
                        "route": "/api/live/card-advisor/stream",
                        "method": "post",
                        "body": {"config": {}},
                    },
                )
            )
        )
        assert out["progress_frames"] == 50
        assert out["final"] == [{"type": "result", "candidates": [1]}]
        assert seen[-1]["method"] == "POST"

        bad = _tool_text(await mcp.call_tool("wb_call", {"route": "http://evil/x"}))
        assert bad.startswith("error:")

        shown = _tool_text(await mcp.call_tool("wb_show", {"title": "EV", "markdown": "**hi**"}))
        assert shown == "shown in the workbench dock: EV"
        assert seen[-1]["item"] == {"title": "EV", "kind": "markdown", "content": "**hi**"}

    asyncio.run(_with_hub(body))


def test_timed_out_call_is_cancelled_and_blocks_new_calls_until_it_lands() -> None:
    async def body(hub: PageHub, port: int) -> None:
        release = asyncio.Event()
        cancels: list[int] = []
        ws = await connect(f"ws://127.0.0.1:{port}", origin="http://localhost:5173")  # ty: ignore[invalid-argument-type]
        await ws.send(json.dumps({"type": "hello", "token": TOKEN, "mode": "play"}))

        async def late(rid: int) -> None:
            await release.wait()
            await ws.send(json.dumps({"type": "res", "id": rid, "ok": True, "result": 1}))

        async def serve() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "cancel":
                    cancels.append(msg["id"])
                    continue
                if msg.get("route") == "/api/live/card-advisor/stream":
                    asyncio.get_running_loop().create_task(late(msg["id"]))
                    continue
                await ws.send(
                    json.dumps({"type": "res", "id": msg["id"], "ok": True, "result": "ok"})
                )

        task = asyncio.get_running_loop().create_task(serve())
        await _wait_connected(hub)
        with pytest.raises(LinkError, match="did not answer '/api/live/card-advisor/stream'"):
            await hub.request(
                "call", timeout=0.1, method="POST", route="/api/live/card-advisor/stream"
            )
        await asyncio.sleep(0.05)
        assert len(cancels) == 1
        assert "card-advisor" in (hub.status()["page_busy_with"] or "")
        with pytest.raises(LinkError, match="still working on an earlier call"):
            await hub.request("call", timeout=1, method="GET", route="/api/live/state")
        # The page's main thread still answers context / show meanwhile.
        assert await hub.request("context", timeout=1) == "ok"
        release.set()
        for _ in range(100):
            if hub.busy_with() is None:
                break
            await asyncio.sleep(0.01)
        assert await hub.request("call", timeout=1, method="GET", route="/api/live/state") == "ok"
        await ws.close()
        task.cancel()

    asyncio.run(_with_hub(body))


def test_token_creation_race_agrees_on_one_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    monkeypatch.delenv("CLAUDE_LINK_TOKEN", raising=False)
    path = tmp_path / "cfg" / "claude-link-token"
    out: list[str] = []
    barrier = threading.Barrier(8)

    def make() -> None:
        barrier.wait()
        out.append(claude_link.load_or_create_token(path))

    threads = [threading.Thread(target=make) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(out)) == 1
    assert path.read_text(encoding="utf-8").strip() == out[0]
    assert [p.name for p in path.parent.iterdir()] == [path.name]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_existing_loose_token_file_is_tightened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLAUDE_LINK_TOKEN", raising=False)
    path = tmp_path / "claude-link-token"
    path.write_text("old-token\n", encoding="utf-8")
    path.chmod(0o644)
    assert claude_link.load_or_create_token(path) == "old-token"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_wb_call_save_to_keeps_every_stream_frame(tmp_path: Path) -> None:
    frames = [{"type": "tick", "done": i} for i in range(10)] + [{"type": "result", "x": 1}]

    async def body(hub: PageHub, port: int) -> None:
        mcp = build_server(hub)
        await _page(port, lambda m: {"ok": True, "result": {"status": 200, "frames": frames}})
        await _wait_connected(hub)
        dest = tmp_path / "s.json"
        await mcp.call_tool(
            "wb_call",
            {"route": "/api/live/card-advisor/stream", "method": "POST", "save_to": str(dest)},
        )
        assert json.loads(dest.read_text(encoding="utf-8"))["frames"] == frames

    asyncio.run(_with_hub(body))
