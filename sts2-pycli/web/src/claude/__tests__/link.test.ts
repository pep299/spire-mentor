import { afterEach, describe, expect, it, vi } from "vitest";
import {
  answer, BAD_TOKEN_CODE, CLAUDE_LINK_KEY, CLAUDE_LINK_TOKEN_KEY, ClaudeLinkClient, consumePairingFragment,
  DEFAULT_CLAUDE_LINK_URL, isMutating, REPLACED_CODE, type LinkHost, type LinkStatus,
} from "../link";
import { upsertItem, type DockItem } from "../useClaudeLink";
import { frameDoc, nextFrameHeight } from "../frameDoc";

function host(over: Partial<LinkHost> = {}): LinkHost & { shown: unknown[]; mutated: string[]; requests: unknown[] } {
  const shown: unknown[] = [];
  const mutated: string[] = [];
  const requests: unknown[] = [];
  return {
    shown,
    mutated,
    requests,
    context: () => ({ mode: "play" }),
    afterMutation: async (r, _res, req) => { mutated.push(r); requests.push(req); },
    show: (it) => { shown.push(it); },
    ...over,
  };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

describe("isMutating", () => {
  it.each([
    ["GET", "/api/live/state", false],
    ["POST", "/api/live/hint", false],
    ["POST", "/api/live/card-advisor/stream", false],
    ["POST", "/api/live/action", true],
    ["POST", "/api/live/seek?x=1", true],
    ["POST", "/api/live/start", true],
    ["POST", "/api/simlab/position/build", false],
    ["POST", "/api/live/save", false],
    ["POST", "/api/live/run-mode", false],
    ["POST", "/api/saves/load", true],
    ["HEAD", "/api/live/action", false],
  ])("%s %s → %s", (method, route, want) => {
    expect(isMutating(method, route)).toBe(want);
  });
});

describe("answer", () => {
  it("context reads the host at request time", async () => {
    expect(await answer({ type: "req", id: 1, op: "context" }, host(), vi.fn())).toEqual({ mode: "play" });
  });

  it("call: JSON body, method, status; read-only routes do not refresh", async () => {
    const h = host();
    const fetcher = vi.fn(async () => json({ scores: [1] }));
    const out = await answer(
      { type: "req", id: 1, op: "call", method: "post", route: "/api/live/hint", body: { config: { sims: 9 } } },
      h, fetcher,
    );
    expect(out).toEqual({ status: 200, body: { scores: [1] } });
    expect(fetcher).toHaveBeenCalledWith("/api/live/hint", {
      method: "POST", body: '{"config":{"sims":9}}', headers: { "content-type": "application/json" },
    });
    expect(h.mutated).toEqual([]);
  });

  it("a successful mutating call refreshes; a failed one does not", async () => {
    const h = host();
    await answer({ type: "req", id: 1, op: "call", method: "POST", route: "/api/live/action", body: {} }, h,
      async () => json({ step: {} }));
    expect(h.mutated).toEqual(["/api/live/action"]);
    const out = await answer({ type: "req", id: 2, op: "call", method: "POST", route: "/api/live/seek", body: {} }, h,
      async () => json({ detail: "no such node" }, 404));
    expect(out).toEqual({ status: 404, body: { detail: "no such node" } });
    expect(h.mutated).toEqual(["/api/live/action"]);
  });

  it("a GET drops a stray body instead of failing in fetch()", async () => {
    const fetcher = vi.fn(async () => json({ ok: 1 }));
    await answer({ type: "req", id: 1, op: "call", method: "GET", route: "/api/live/state", body: {} }, host(), fetcher);
    expect(fetcher).toHaveBeenCalledWith("/api/live/state", expect.not.objectContaining({ body: expect.anything() }));
  });

  it("a mutation hands the host its route, reply and request body", async () => {
    const h = host();
    const afterMutation = vi.fn(async () => {});
    await answer({ type: "req", id: 1, op: "call", method: "POST", route: "/api/live/quit" },
      { ...h, afterMutation }, async () => json({ ok: true, log_path: "/l.jsonl" }));
    expect(afterMutation).toHaveBeenCalledWith("/api/live/quit", { status: 200, body: { ok: true, log_path: "/l.jsonl" } }, null);
    await answer({ type: "req", id: 2, op: "call", method: "POST", route: "/api/live/action", body: { action: "end_turn" } },
      h, async () => json({ step: {} }));
    expect(h.requests).toEqual([{ action: "end_turn" }]);
  });

  it("ndjson streams come back as frames", async () => {
    const resp = new Response('{"type":"tick"}\n{"type":"result","x":1}\n', {
      headers: { "content-type": "application/x-ndjson" },
    });
    const out = await answer({ type: "req", id: 1, op: "call", method: "POST", route: "/api/live/card-advisor/stream" },
      host(), async () => resp);
    expect(out).toEqual({ status: 200, frames: [{ type: "tick" }, { type: "result", x: 1 }] });
  });

  it("refuses non-/api routes and bad show items", async () => {
    await expect(answer({ type: "req", id: 1, op: "call", route: "https://evil/x" }, host(), vi.fn()))
      .rejects.toThrow(/not an \/api route/);
    await expect(answer({ type: "req", id: 1, op: "show", item: { title: "t", kind: "exe" } }, host(), vi.fn()))
      .rejects.toThrow(/expected/);
    const h = host();
    await answer({ type: "req", id: 1, op: "show", item: { title: "t", kind: "plotly", content: { data: [] } } }, h, vi.fn());
    expect(h.shown).toEqual([{ title: "t", kind: "plotly", content: { data: [] } }]);
  });
});

/** Just enough of a browser WebSocket for the client. */
class FakeWs {
  static last: FakeWs | null = null;
  static count = 0;
  readyState = 0;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: ((e: { code: number }) => void) | null = null;
  constructor(public url: string) { FakeWs.last = this; FakeWs.count += 1; }
  send(s: string) { this.sent.push(s); }
  close() { this.readyState = 3; }
  open() { this.readyState = 1; this.onopen?.(); }
  serverClose(code: number) { this.readyState = 3; this.onclose?.({ code }); }
}

afterEach(() => {
  vi.useRealTimers();
  FakeWs.last = null;
  FakeWs.count = 0;
});

function client(statuses: LinkStatus[]) {
  return new ClaudeLinkClient({
    url: "ws://127.0.0.1:8091",
    token: "tok",
    host: host(),
    fetcher: vi.fn(),
    onStatus: (s) => statuses.push(s),
    hello: () => ({ mode: "play" }),
    WebSocketImpl: FakeWs as unknown as typeof WebSocket,
  });
}

describe("ClaudeLinkClient", () => {
  it("says hello on open and answers a request", async () => {
    const statuses: LinkStatus[] = [];
    client(statuses).start();
    const ws = FakeWs.last!;
    ws.open();
    expect(statuses).toEqual(["connected"]);
    expect(JSON.parse(ws.sent[0])).toEqual({ type: "hello", mode: "play", token: "tok" });
    ws.onmessage!({ data: JSON.stringify({ type: "req", id: 7, op: "context" }) });
    await vi.waitFor(() => expect(ws.sent).toHaveLength(2));
    expect(JSON.parse(ws.sent[1])).toEqual({ type: "res", id: 7, ok: true, result: { mode: "play" } });
    ws.onmessage!({ data: JSON.stringify({ type: "req", id: 8, op: "bogus" }) });
    await vi.waitFor(() => expect(ws.sent).toHaveLength(3));
    expect(JSON.parse(ws.sent[2])).toMatchObject({ id: 8, ok: false, error: "unknown op: bogus" });
  });

  it("reconnects with backoff after a drop, but not after being replaced", () => {
    vi.useFakeTimers();
    const statuses: LinkStatus[] = [];
    const c = client(statuses);
    c.start();
    FakeWs.last!.serverClose(1006);
    expect(statuses).toEqual(["waiting"]);
    vi.advanceTimersByTime(1000);
    expect(FakeWs.count).toBe(2);
    FakeWs.last!.open();
    FakeWs.last!.serverClose(REPLACED_CODE);
    expect(statuses[statuses.length - 1]).toBe("replaced");
    vi.advanceTimersByTime(60_000);
    expect(FakeWs.count).toBe(2);
  });

  it("a refused pairing token stops retrying", () => {
    vi.useFakeTimers();
    const statuses: LinkStatus[] = [];
    client(statuses).start();
    FakeWs.last!.serverClose(BAD_TOKEN_CODE);
    expect(statuses).toEqual(["unauthorized"]);
    vi.advanceTimersByTime(60_000);
    expect(FakeWs.count).toBe(1);
  });

  it("a relay cancel aborts the request; its late result is dropped, not refreshed", async () => {
    let resolve!: (r: Response) => void;
    let signal: AbortSignal | undefined;
    const fetcher = vi.fn((_: string, init?: RequestInit) => {
      signal = init?.signal ?? undefined;
      return new Promise<Response>((r) => { resolve = r; });
    });
    const h = host();
    new ClaudeLinkClient({
      url: "ws://x", token: "tok", host: h, fetcher, onStatus: () => {}, hello: () => ({}),
      WebSocketImpl: FakeWs as unknown as typeof WebSocket,
    }).start();
    const ws = FakeWs.last!;
    ws.open();
    ws.onmessage!({ data: JSON.stringify({ type: "req", id: 5, op: "call", method: "POST", route: "/api/live/action", body: {} }) });
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalled());
    ws.onmessage!({ data: JSON.stringify({ type: "cancel", id: 5 }) });
    expect(signal?.aborted).toBe(true);
    resolve(json({ step: {} }));
    await vi.waitFor(() => expect(ws.sent).toHaveLength(2));
    expect(JSON.parse(ws.sent[1])).toMatchObject({ id: 5, ok: false, error: expect.stringMatching(/cancelled/) });
    expect(h.mutated).toEqual([]);
  });

  it("stop() ends the retry loop", () => {
    vi.useFakeTimers();
    const c = client([]);
    c.start();
    FakeWs.last!.serverClose(1006);
    c.stop();
    vi.advanceTimersByTime(60_000);
    expect(FakeWs.count).toBe(1);
  });
});

describe("pairing link", () => {
  afterEach(() => localStorage.clear());

  it("#claude-link=<token> stores the token, turns the link on, and strips the fragment", () => {
    const replace = vi.fn();
    const loc = { hash: "#claude-link=a%2Bb", pathname: "/", search: "?x=1" };
    expect(consumePairingFragment(loc, replace)).toBe(true);
    expect(localStorage.getItem(CLAUDE_LINK_TOKEN_KEY)).toBe("a+b");
    expect(localStorage.getItem(CLAUDE_LINK_KEY)).toBe(DEFAULT_CLAUDE_LINK_URL);
    expect(replace).toHaveBeenCalledWith("/?x=1");
  });

  it("a malformed pairing link neither throws nor stores anything, and is stripped", () => {
    const replace = vi.fn();
    expect(consumePairingFragment({ hash: "#claude-link=%E0", pathname: "/", search: "" }, replace)).toBe(false);
    expect(replace).toHaveBeenCalledWith("/");
    expect(localStorage.getItem(CLAUDE_LINK_TOKEN_KEY)).toBeNull();
  });

  it("ignores other fragments", () => {
    const replace = vi.fn();
    expect(consumePairingFragment({ hash: "#report=1", pathname: "/", search: "" }, replace)).toBe(false);
    expect(replace).not.toHaveBeenCalled();
    expect(localStorage.getItem(CLAUDE_LINK_TOKEN_KEY)).toBeNull();
  });
});

describe("dock items", () => {
  const item = (title: string, id: number): DockItem => ({ title, kind: "markdown", content: "", id, at: 0 });

  it("a title shown again replaces its item and moves it to the top", () => {
    const items = [item("a", 1), item("b", 2)];
    const out = upsertItem(items, { title: "b", kind: "html", content: "<p>" }, 3);
    expect(out.map((i) => [i.title, i.id])).toEqual([["b", 3], ["a", 1]]);
  });

  it("frameDoc injects the height reporter and escapes markdown into the fallback", () => {
    const html = frameDoc({ ...item("t", 1), kind: "html", content: "<html><body><p>x</p></body></html>" }, "tok");
    expect(html.indexOf("claudeDock")).toBeLessThan(html.indexOf("</body>"));
    const md = frameDoc({ ...item("t", 1), content: "<b>hi</b></script><script>alert(1)" }, "tok");
    expect(md).toContain("&lt;b&gt;hi&lt;/b&gt;");
    expect(md).not.toContain("</script><script>alert(1)");
  });

  it("markdown rides as JSON: </script> keeps no backslash, <!-- cannot swallow the renderer", () => {
    const src = "a `</script>` b <!--<script> c";
    const doc = frameDoc({ ...item("t", 1), content: src }, "tok");
    const json = doc.match(/<script id="md" type="application\/json">(.*?)<\/script>/)![1];
    expect(json).not.toMatch(/<(!--|\/script)/i);
    expect(JSON.parse(json)).toBe(src);
    expect(doc).toContain("marked.parse(JSON.parse(");
  });

  it("frame height: no padding, and a viewport-tied document stops growing", () => {
    // Plain content: fits exactly, and a repeat report is a fixed point.
    expect(nextFrameHeight(160, 300, null)).toEqual({ height: 300, growth: 140, loop: false });
    expect(nextFrameHeight(300, 300, 140)).toEqual({ height: 300, growth: null, loop: false });
    // 100vh + 20px margin: every height it is given comes back 20px taller.
    const a = nextFrameHeight(160, 180, null);
    const b = nextFrameHeight(a.height, a.height + 20, a.growth);
    expect(b).toMatchObject({ height: 180, loop: true });
    expect(nextFrameHeight(160, 99_999, null).height).toBe(2000);
  });

  it("plotly figures render inside the frame, their JSON unable to close the script", () => {
    const evil = "</script><script>alert(1)</script>";
    const doc = frameDoc({ ...item("t", 1), kind: "plotly", content: { data: [{ y: [1], name: evil }] } }, "tok");
    expect(doc).toContain("Plotly.newPlot");
    expect(doc).not.toContain("</script><script>alert(1)");
    const json = doc.match(/<script id="fig" type="application\/json">(.*?)<\/script>/)![1];
    expect(JSON.parse(json).data[0].name).toBe(evil);
  });
});
