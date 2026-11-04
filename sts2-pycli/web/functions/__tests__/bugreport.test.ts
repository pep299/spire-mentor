import { describe, expect, it } from "vitest";
import {
  ID_RE,
  KEY_PREFIX,
  MAX_BODY_BYTES,
  TTL_SECONDS,
  handleGet,
  handleIntakeMiss,
  handleIntakeProbe,
  handlePost,
  handleProbe,
  metadataOf,
  newId,
  validateDump,
} from "../api/_bugreport";
import type { Env, KvNamespace } from "../env";

/** A Map-backed KV double recording the put options. */
function fakeKv() {
  const values = new Map<string, string>();
  const options = new Map<string, { expirationTtl?: number; metadata?: unknown }>();
  const kv: KvNamespace = {
    async get(key) {
      return values.get(key) ?? null;
    },
    async put(key, value, opts) {
      values.set(key, value);
      options.set(key, opts ?? {});
    },
    async list({ prefix } = {}) {
      const keys = [...values.keys()].filter((k) => !prefix || k.startsWith(prefix)).map((name) => ({ name }));
      return { keys, list_complete: true };
    },
  };
  return { kv, values, options, env: { BUGREPORTS: kv } as Env };
}

const ORIGIN = "https://your-workbench.example.com";

function dump(extra: Record<string, unknown> = {}) {
  return {
    schema_version: 1,
    meta: { mode: "live", character: "Silent", workbench_commit: "abc1234" },
    repro: { persist: { nodes: [] } },
    state: { game_state: { floor: 3 } },
    ui: { screen: "combat_play", overlay: null },
    note: "the enemy's block never dropped",
    ...extra,
  };
}

function post(body: unknown, headers: Record<string, string> = {}) {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return new Request(`${ORIGIN}/api/bugreport`, {
    method: "POST",
    headers: { origin: ORIGIN, "content-type": "application/json", "user-agent": "Mozilla/5.0 (Linux; Android 14) Chrome/128", ...headers },
    body: text,
  });
}

const fixed = { now: () => new Date("2026-09-03T12:00:00.123Z"), random: (n: number) => new Uint8Array(n).fill(0xab) };

describe("bug-report Function — validation", () => {
  it("ids are wb- + 16 hex, from the random source", () => {
    const id = newId(fixed.random);
    expect(id).toBe("wb-abababababababab");
    expect(ID_RE.test(id)).toBe(true);
    expect(ID_RE.test(newId())).toBe(true);
    expect(ID_RE.test("../etc/passwd")).toBe(false);
  });

  it("accepts the schema-1 dump and names what is wrong otherwise", () => {
    expect(validateDump(dump())).toBeNull();
    expect(validateDump(dump({ note: null }))).toBeNull();
    expect(validateDump([])).toMatch(/object/);
    expect(validateDump(dump({ schema_version: 2 }))).toMatch(/schema_version/);
    expect(validateDump(dump({ state: "x" }))).toMatch(/state/);
    expect(validateDump(dump({ note: 3 }))).toMatch(/note/);
  });

  it("listing metadata is the short header, note clipped", () => {
    const d = dump({ note: "x".repeat(500) });
    const m = metadataOf(d, "2026-09-03T12:00:00Z");
    expect(m).toEqual({
      captured_at: "2026-09-03T12:00:00Z",
      mode: "live",
      character: "Silent",
      screen: "combat_play",
      build: "abc1234",
      note: "x".repeat(200),
    });
    expect(JSON.stringify(m).length).toBeLessThan(1024);
  });
});

describe("bug-report Function — routes", () => {
  it("the probe answers JSON so the SPA can tell a Functions host from a plain one", async () => {
    const r = handleProbe();
    expect(r.status).toBe(200);
    expect(r.headers.get("content-type")).toBe("application/json");
    expect(await r.json()).toEqual({ ok: true, store: "kv" });
  });

  it("stores a report, stamping id and captured_at like the Python route layer", async () => {
    const { env, values, options } = fakeKv();
    const r = await handlePost(post(dump()), env, fixed);
    expect(r.status).toBe(201);
    const body = (await r.json()) as { id: string; url: string };
    expect(body).toEqual({ id: "wb-abababababababab", url: `${ORIGIN}/?report=wb-abababababababab` });
    const key = KEY_PREFIX + body.id;
    const stored = JSON.parse(values.get(key)!);
    expect(stored.meta).toMatchObject({
      id: body.id,
      captured_at: "2026-09-03T12:00:00Z",
      received_via: "pages",
      character: "Silent",
      user_agent: expect.stringContaining("Android 14"),
    });
    expect(stored.state).toEqual({ game_state: { floor: 3 } });
    expect(options.get(key)).toEqual({
      expirationTtl: TTL_SECONDS,
      metadata: expect.objectContaining({ character: "Silent", captured_at: "2026-09-03T12:00:00Z" }),
    });
    // …and reads back byte-identical.
    const g = await handleGet(body.id, env);
    expect(g.status).toBe(200);
    expect(await g.text()).toBe(values.get(key));
  });

  it("rejects cross-origin, oversized, non-JSON and off-schema posts without touching the store", async () => {
    const { env, values } = fakeKv();
    expect((await handlePost(post(dump(), { origin: "https://evil.example" }), env)).status).toBe(403);
    // A loopback page (the mod-bundled workbench) must go through its control
    // plane's forward, never post here itself.
    expect((await handlePost(post(dump(), { origin: "http://127.0.0.1:8079" }), env)).status).toBe(403);
    // /api/ is the page's: a client without an Origin uses the intake.
    const noOrigin = new Request(`${ORIGIN}/api/bugreport`, { method: "POST", body: "{}" });
    expect((await handlePost(noOrigin, env)).status).toBe(403);
    expect((await handlePost(post(dump(), { "content-length": String(MAX_BODY_BYTES + 1) }), env)).status).toBe(413);
    expect((await handlePost(post("{not json"), env)).status).toBe(400);
    const off = await handlePost(post({ schema_version: 1, meta: {} }), env);
    expect(off.status).toBe(422);
    expect(await off.json()).toEqual({ detail: "repro must be an object" });
    expect(values.size).toBe(0);
  });

  it("the intake takes a report from a client that sends no Origin (the mod, the advisor's forward)", async () => {
    const { env, values, options } = fakeKv();
    const modDump = {
      schema_version: 1,
      meta: { source: "mod", mode: "game", mod_version: "0.8.2", character: "Defect", user_agent: "SpireMentor/0.8.2 (Slay the Spire 2 mod)" },
      repro: {},
      state: { game_state: { type: "advice_request", advice_for: "combat_action" }, map: null },
      ui: { screen: "combat_action", overlay: null },
      logs: { mod: ["a"], game: "b", advisor: null },
      note: "orb evoked twice",
    };
    const req = new Request(`${ORIGIN}/intake/bugreport`, {
      method: "POST",
      headers: { "content-type": "application/json", "user-agent": "SpireMentor/0.8.2 (Slay the Spire 2 mod)" },
      body: JSON.stringify(modDump),
    });
    const r = await handlePost(req, env, fixed, "intake");
    expect(r.status).toBe(201);
    const { id, url } = (await r.json()) as { id: string; url: string };
    // The link is to the site's report view, where reading needs the sign-in.
    expect(url).toBe(`${ORIGIN}/?report=${id}`);
    const stored = JSON.parse(values.get(KEY_PREFIX + id)!);
    // The mod's own fields ride through untouched.
    expect(stored.logs).toEqual(modDump.logs);
    expect(stored.meta).toMatchObject({ source: "mod", user_agent: "SpireMentor/0.8.2 (Slay the Spire 2 mod)", received_via: "pages" });
    expect(options.get(KEY_PREFIX + id)?.metadata).toMatchObject({ source: "mod", build: "0.8.2", mode: "game", screen: "combat_action" });
  });

  it("the intake still refuses other sites' pages, and the page itself may use it", async () => {
    const { env, values } = fakeKv();
    const at = (headers: Record<string, string>) =>
      new Request(`${ORIGIN}/intake/bugreport`, { method: "POST", headers, body: JSON.stringify(dump()) });
    expect((await handlePost(at({ origin: "https://evil.example" }), env, {}, "intake")).status).toBe(403);
    expect(values.size).toBe(0);
    expect((await handlePost(at({ origin: ORIGIN }), env, fixed, "intake")).status).toBe(201);
  });

  it("the intake's probe says so, and nothing else under /intake/ answers", async () => {
    expect(await handleIntakeProbe().json()).toEqual({ ok: true, store: "kv", intake: true });
    const miss = handleIntakeMiss();
    expect(miss.status).toBe(404);
    expect(miss.headers.get("content-type")).toBe("application/json");
  });

  it("an unknown or malformed id is a 404, never a KV lookup outside the prefix", async () => {
    const { env, values } = fakeKv();
    values.set("other/secret", "{}");
    expect((await handleGet("wb-0000000000000000", env)).status).toBe(404);
    expect((await handleGet("../other/secret", env)).status).toBe(404);
    expect((await handleGet("secret", env)).status).toBe(404);
  });
});
