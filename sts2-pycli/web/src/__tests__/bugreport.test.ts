import { beforeEach, describe, expect, it } from "vitest";
import type { BugReportDump } from "../api";
import {
  buildBrowserDump,
  bugReportEndpointAvailable,
  captureBrowserBugReport,
  fetchBrowserBugReport,
  importReportPosition,
  loadReportIntoWorkbench,
  reportGamePosition,
  reportLoadKind,
  reportRunFilename,
  resetBugReportProbe,
} from "../bugreport";

const jsonRes = (body: unknown, status = 200, type = "application/json") =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": type } });

/** A worker-bound `apiFetch` double: `METHOD /path` → reply, recording calls. */
function fakeApi(routes: Record<string, unknown | ((body: unknown) => unknown)>) {
  const calls: { key: string; body: unknown }[] = [];
  const api = async (input: string, init?: RequestInit): Promise<Response> => {
    const method = (init?.method ?? "GET").toUpperCase();
    const key = `${method} ${input.split("?")[0]}`;
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : null;
    calls.push({ key: `${method} ${input}`, body });
    if (!(key in routes)) return jsonRes({ detail: `not available: ${key}` }, 501);
    const r = routes[key];
    const val = typeof r === "function" ? (r as (b: unknown) => unknown)(body) : r;
    if (val instanceof Response) return val;
    return jsonRes(val);
  };
  return { api, calls };
}

const identity = { origin: "https://wb.test", userAgent: "UA/1", buildId: "deadbeef" };
const persist = { character: "Silent", seed: "ABC", ascension: 3, nodes: [{ id: 0 }], active_leaf: 0 };

describe("buildBrowserDump", () => {
  it("live: the step, the map and the persisted tree, stamped with the host identity", async () => {
    const { api } = fakeApi({
      "GET /api/live/state": { idx: 4, decision: "combat_play", state: { floor: 2, player: { name: "Silent" } } },
      "GET /api/live/map": { nodes: [1, 2] },
      "GET /api/live/persist": { persist },
    });
    const d = await buildBrowserDump({ mode: "live", screen: "combat_play", note: "hi" }, { api, ...identity });
    expect(d).toEqual({
      schema_version: 1,
      meta: {
        mode: "live", character: "Silent", seed: "ABC", ascension: 3, backend: "wasm",
        engine_build: "v0.107.1", workbench_commit: "deadbeef", host: "https://wb.test", user_agent: "UA/1",
      },
      repro: { persist },
      state: { game_state: { floor: 2, player: { name: "Silent" } }, map: { nodes: [1, 2] } },
      ui: { screen: "combat_play", overlay: null },
      note: "hi",
    });
  });

  it("live with no run is 'nothing to capture' (the Python 409)", async () => {
    const { api } = fakeApi({ "GET /api/live/state": jsonRes({ detail: "no live run" }, 400) });
    await expect(buildBrowserDump({ mode: "live" }, { api, ...identity })).rejects.toThrow("nothing to capture");
  });

  it("replay: the viewed step and the run log's text", async () => {
    const { api, calls } = fakeApi({
      "GET /api/step/7": { idx: 7, state: { floor: 9, player: { name: "Ironclad" } } },
      "GET /api/runs/read": { path: "20260903_Ironclad_S.jsonl", text: '{"type":"meta"}\n', size: 16, mtime: 1 },
    });
    const d = await buildBrowserDump(
      { mode: "replay", step: 7, run_path: "20260903_Ironclad_S.jsonl", screen: "map" },
      { api, ...identity },
    );
    expect(d.meta).toMatchObject({ mode: "replay", character: "Ironclad", backend: null });
    expect(d.repro).toEqual({ run_file: "20260903_Ironclad_S.jsonl", line: 7, run_log: '{"type":"meta"}\n' });
    expect(d.state).toEqual({ game_state: { floor: 9, player: { name: "Ironclad" } }, map: null });
    expect(calls.map((c) => c.key)).toContain("GET /api/runs/read?path=20260903_Ironclad_S.jsonl");
    await expect(buildBrowserDump({ mode: "replay", step: null }, { api, ...identity })).rejects.toThrow("nothing to capture");
    await expect(buildBrowserDump({ mode: "bogus" }, { api, ...identity })).rejects.toThrow("unknown mode");
  });
});

describe("capture / fetch / probe", () => {
  beforeEach(() => resetBugReportProbe());

  it("posts the dump to /api/bugreport and returns the id + url", async () => {
    const { api } = fakeApi({
      "GET /api/live/state": { state: { floor: 1 } },
      "GET /api/live/map": null,
      "GET /api/live/persist": { persist },
    });
    const posted: { url: string; init: RequestInit }[] = [];
    const fetch = async (url: RequestInfo | URL, init?: RequestInit) => {
      posted.push({ url: String(url), init: init ?? {} });
      return jsonRes({ id: "wb-0123456789abcdef", url: "https://wb.test/?report=wb-0123456789abcdef" }, 201);
    };
    const r = await captureBrowserBugReport({ mode: "live", note: null }, { api, fetch, ...identity });
    expect(r).toEqual({ id: "wb-0123456789abcdef", url: "https://wb.test/?report=wb-0123456789abcdef" });
    expect(posted).toHaveLength(1);
    expect(posted[0].url).toBe("/api/bugreport");
    expect(posted[0].init.method).toBe("POST");
    const body = JSON.parse(posted[0].init.body as string) as BugReportDump;
    expect(body.repro).toEqual({ persist });
    // A refused post surfaces the Function's detail.
    const refuse = async () => jsonRes({ detail: "report too large" }, 413);
    await expect(captureBrowserBugReport({ mode: "live" }, { api, fetch: refuse, ...identity })).rejects.toThrow("413 report too large");
  });

  it("fetches a stored report by id", async () => {
    const dump = { schema_version: 1, meta: { id: "wb-1" }, repro: {}, state: {}, ui: { screen: null, overlay: null }, note: null };
    const fetch = async (url: RequestInfo | URL) =>
      String(url) === "/api/bugreport/wb-1" ? jsonRes(dump) : jsonRes({ detail: "unknown report" }, 404);
    expect(await fetchBrowserBugReport("wb-1", { fetch })).toEqual(dump);
    await expect(fetchBrowserBugReport("wb-2", { fetch })).rejects.toThrow("404 unknown report");
  });

  it("the probe is yes only for the Function's JSON, and is asked once", async () => {
    let n = 0;
    const yes = async () => (n++, jsonRes({ ok: true, store: "kv" }));
    expect(await bugReportEndpointAvailable({ fetch: yes })).toBe(true);
    expect(await bugReportEndpointAvailable({ fetch: yes })).toBe(true);
    expect(n).toBe(1);
    resetBugReportProbe();
    // A plain static host rewrites unknown routes to index.html: a 200 that is not JSON.
    const html = async () => new Response("<!doctype html>", { status: 200, headers: { "content-type": "text/html" } });
    expect(await bugReportEndpointAvailable({ fetch: html })).toBe(false);
    resetBugReportProbe();
    const missing = async () => new Response("", { status: 404 });
    expect(await bugReportEndpointAvailable({ fetch: missing })).toBe(false);
    resetBugReportProbe();
    const offline = async () => { throw new TypeError("Failed to fetch"); };
    expect(await bugReportEndpointAvailable({ fetch: offline })).toBe(false);
  });
});

describe("loading a report into the reader's workbench", () => {
  const base = (repro: Record<string, unknown>, meta: Record<string, unknown> = { id: "wb-abcdef0123456789" }): BugReportDump => ({
    schema_version: 1, meta, repro, state: {}, ui: { screen: null, overlay: null }, note: null,
  });

  it("names the stored log after the report and the original file", () => {
    expect(reportRunFilename(base({ run_log: "{}", run_file: "20260903_Ironclad_S.jsonl" }))).toBe("wb-abcdef0123456789_20260903_Ironclad_S.jsonl");
    expect(reportRunFilename(base({ run_log: "{}", run_file: "/home/x/logs/run one.jsonl" }))).toBe("wb-abcdef0123456789_run_one.jsonl");
    expect(reportRunFilename(base({ run_log: "{}", run_file: "../../evil.txt" }))).toBe("wb-abcdef0123456789_evil.jsonl");
    expect(reportRunFilename(base({ run_log: "{}" }))).toBe("wb-abcdef0123456789_run.jsonl");
  });

  it("a live tree is imported; a run log is stored; anything else is refused", async () => {
    const { api, calls } = fakeApi({
      "POST /api/live/import": { idx: 3 },
      "POST /api/runs/put": (b: unknown) => ({ path: (b as { path: string }).path, size: 1 }),
    });
    expect(reportLoadKind(base({ persist }))).toBe("live");
    expect(await loadReportIntoWorkbench(base({ persist }), { api })).toEqual({ kind: "live" });
    expect(calls[0]).toEqual({ key: "POST /api/live/import", body: { persist } });
    expect(reportLoadKind(base({ run_log: '{"a":1}\n', run_file: "r.jsonl" }))).toBe("replay");
    expect(await loadReportIntoWorkbench(base({ run_log: '{"a":1}\n', run_file: "r.jsonl" }), { api })).toEqual({
      kind: "replay", path: "wb-abcdef0123456789_r.jsonl",
    });
    expect(calls[1]).toEqual({ key: "POST /api/runs/put", body: { path: "wb-abcdef0123456789_r.jsonl", text: '{"a":1}\n' } });
    expect(reportLoadKind(base({ timeline: [] }))).toBeNull();
    await expect(loadReportIntoWorkbench(base({ timeline: [] }), { api })).rejects.toThrow("no run to load");
    // The engine's refusal comes through.
    const { api: refusing } = fakeApi({ "POST /api/live/import": jsonRes({ detail: "persisted session is empty" }, 409) });
    await expect(loadReportIntoWorkbench(base({ persist }), { api: refusing })).rejects.toThrow("409 persisted session is empty");
  });
});

describe("the game position a report carries", () => {
  const base = { schema_version: 1, repro: {}, ui: { screen: null, overlay: null }, note: null };
  const position = { type: "advice_request", advice_for: "card_reward", game_version: "v0.111.0" };

  it("a mod report's game_state; a bundled-workbench report's live_game; nothing else", () => {
    const mod = { ...base, meta: { source: "mod" }, state: { game_state: position } } as BugReportDump;
    expect(reportGamePosition(mod)).toBe(position);
    const bundled = { ...base, meta: { mode: "replay" }, state: { game_state: { floor: 3 }, live_game: position } } as BugReportDump;
    expect(reportGamePosition(bundled)).toBe(position);
    // A workbench step is not a mod message, whoever claims it.
    const plain = { ...base, meta: { mode: "live" }, state: { game_state: { floor: 3 } } } as BugReportDump;
    expect(reportGamePosition(plain)).toBeNull();
    const fake = { ...base, meta: { source: "mod" }, state: { game_state: { floor: 3 } } } as BugReportDump;
    expect(reportGamePosition(fake)).toBeNull();
  });

  it("imports it through the state import, in the build it was taken in", async () => {
    const { api, calls } = fakeApi({ "POST /api/live/import-state": { idx: 0, import: { lane: "run" } } });
    const mod = { ...base, meta: { source: "mod" }, state: { game_state: position } } as BugReportDump;
    expect(await importReportPosition(mod, { api })).toEqual({ lane: "run" });
    expect(calls[0].body).toEqual({ payload: JSON.stringify(position), lane: "auto", game_version: "v0.111.0" });
    const none = { ...base, meta: {}, state: {} } as BugReportDump;
    await expect(importReportPosition(none, { api })).rejects.toThrow("no game position");
  });
});
