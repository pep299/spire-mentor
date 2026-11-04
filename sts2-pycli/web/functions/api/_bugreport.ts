/**
 * Bug reports for the browser-only workbench — the one server-side piece of
 * the static site, a Cloudflare Pages Function over a Workers KV namespace
 * (binding `BUGREPORTS`, wrangler.toml).
 *
 * The Python workbench captures a report server-side (workbench/bugreport.py
 * builds the dump, bugreport_store.py puts it in GCS). Here the PAGE builds
 * the dump (src/bugreport.ts: it has the engine in its worker) and posts the
 * finished JSON; this side only validates, stamps `meta.id` /
 * `meta.captured_at` the way the Python route layer does, stores, and serves
 * it back. Same schema, same id format (`wb-` + 16 hex), so `?report=<id>`
 * opens either kind and `scripts/bug_reports.py` reads both.
 *
 * Routes (functions/api/bugreport.ts, functions/api/bugreport/[id].ts,
 * functions/intake/bugreport.ts):
 *   GET  /api/bugreport        {ok, store}  — the probe the SPA uses to show
 *                              the button (a host without Functions answers
 *                              HTML or 404, never this JSON).
 *   POST /api/bugreport        the page's dump → {id, url}; same origin only
 *   GET  /api/bugreport/<id>   the stored dump
 *   POST /intake/bugreport     a non-browser client's dump → {id, url}
 *   GET  /intake/bugreport     {ok, store, intake: true} — the same probe,
 *                              for checking the intake is reachable
 *
 * Why a second path for posting. The deployed site sits behind a sign-in
 * (Cloudflare Access, Google accounts), and the clients that are not the
 * page cannot sign in: the Spire Mentor mod's in-game "Report a bug"
 * (SimBugReport.cs, `meta.source = "mod"`) and the advisor's control plane
 * forwarding the mod-bundled workbench's reports (realgame_control.py).
 * Access exempts a path together with everything under it and has no
 * exact-path rule, so exempting `/api/bugreport` would also open
 * `/api/bugreport/<id>` — every stored report — to anyone. `/intake/` holds
 * nothing to read: its one route only writes, and every other path under it
 * is a 404 (functions/intake/[[path]].ts), so the Access exemption there
 * (deploy/DEPLOY.md, "Bug-report intake") exposes no page, asset or report.
 *
 * Abuse budget: there is no rate-limit binding on Pages' free plan. The
 * cap is the body size (a flood of 20 MiB bodies still costs nothing — KV
 * writes are free up to 1,000 a day and then fail, which is the worst
 * case: reports lost for a day) plus an origin check on POST: a request
 * that names a foreign `Origin` is refused on both paths, which stops
 * drive-by posts from other sites' pages. The intake also takes a request
 * with NO `Origin` — browsers always send one on a POST, so that is a
 * non-browser client, which could forge the header anyway; `/api/` keeps
 * requiring the page's own origin.
 */
import type { Env, KvNamespace } from "../env";

export const ID_RE = /^wb-[0-9a-f]{16}$/;
export const KEY_PREFIX = "bugreports/";
/** KV values cap at 25 MiB; leave room for the stamped fields. */
export const MAX_BODY_BYTES = 20 * 1024 * 1024;
/** Reports self-clean after this; a bug older than that has been fixed or
 *  re-reported (the free tier's 1 GB is the reason to clean at all). */
export const TTL_SECONDS = 180 * 24 * 3600;
const NOTE_META_CHARS = 200;

export type Json = Record<string, unknown>;

/** `wb-` + 16 hex chars (64 random bits) — the id is the only barrier to
 *  reading a report, so it must be unguessable. */
export function newId(random: (n: number) => Uint8Array = randomBytes): string {
  return `wb-${[...random(8)].map((b) => b.toString(16).padStart(2, "0")).join("")}`;
}

function randomBytes(n: number): Uint8Array {
  return crypto.getRandomValues(new Uint8Array(n));
}

export function isoNow(now: () => Date = () => new Date()): string {
  return now().toISOString().replace(/\.\d{3}Z$/, "Z");
}

export function json(body: unknown, status = 200, extra: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store", ...extra },
  });
}

export function detail(status: number, msg: string): Response {
  return json({ detail: msg }, status);
}

const isObject = (v: unknown): v is Json => typeof v === "object" && v !== null && !Array.isArray(v);

/** The dump shape src/bugreport.ts builds (schema 1, the Python one). */
export function validateDump(body: unknown): string | null {
  if (!isObject(body)) return "body must be a JSON object";
  if (body.schema_version !== 1) return "schema_version must be 1";
  for (const k of ["meta", "repro", "state", "ui"]) {
    if (!isObject(body[k])) return `${k} must be an object`;
  }
  if (body.note != null && typeof body.note !== "string") return "note must be a string or null";
  return null;
}

/** What a listing shows without fetching the value (KV metadata ≤ 1024 B).
 *  `build` is the workbench commit for a page's report, the mod version
 *  for the mod's; `source` is set only on reports that say where they came
 *  from (the mod's: "mod"). */
export function metadataOf(dump: Json, capturedAt: string): Json {
  const meta = dump.meta as Json;
  const ui = dump.ui as Json;
  const note = typeof dump.note === "string" ? dump.note.slice(0, NOTE_META_CHARS) : null;
  const out: Json = {
    captured_at: capturedAt,
    mode: meta.mode ?? null,
    character: meta.character ?? null,
    screen: ui.screen ?? null,
    build: meta.workbench_commit ?? meta.mod_version ?? null,
    note,
  };
  if (typeof meta.source === "string") out.source = meta.source.slice(0, 16);
  return out;
}

export interface StoreDeps {
  now?: () => Date;
  random?: (n: number) => Uint8Array;
}

/** Stamp + store a validated dump; returns the id. */
export async function storeDump(
  kv: KvNamespace,
  dump: Json,
  request: Request,
  deps: StoreDeps = {},
): Promise<string> {
  const id = newId(deps.random);
  const capturedAt = isoNow(deps.now);
  const meta = dump.meta as Json;
  meta.id = id;
  meta.captured_at = capturedAt;
  meta.received_via = "pages";
  const ua = request.headers.get("user-agent");
  if (ua && meta.user_agent == null) meta.user_agent = ua.slice(0, 300);
  await kv.put(KEY_PREFIX + id, JSON.stringify(dump), {
    expirationTtl: TTL_SECONDS,
    metadata: metadataOf(dump, capturedAt),
  });
  return id;
}

/** Who may post: the page's own origin always; a request with no `Origin`
 *  (a non-browser client) only on the intake. */
export type OriginPolicy = "same-origin" | "intake";

/** `POST /api/bugreport` (`"same-origin"`) and `POST /intake/bugreport`
 *  (`"intake"`). */
export async function handlePost(
  request: Request,
  env: Env,
  deps: StoreDeps = {},
  policy: OriginPolicy = "same-origin",
): Promise<Response> {
  const url = new URL(request.url);
  const origin = request.headers.get("origin");
  const allowed = origin === url.origin || (policy === "intake" && origin === null);
  if (!allowed) {
    return detail(
      403,
      policy === "intake"
        ? "the intake takes reports from the mod and its advisor, not from other sites' pages"
        : "bug reports are accepted from the workbench page only",
    );
  }
  const declared = Number(request.headers.get("content-length") ?? "0");
  if (declared > MAX_BODY_BYTES) return detail(413, `report too large (${declared} bytes; limit ${MAX_BODY_BYTES})`);
  const text = await request.text();
  if (text.length > MAX_BODY_BYTES) return detail(413, `report too large (${text.length} chars; limit ${MAX_BODY_BYTES})`);
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return detail(400, "body is not JSON");
  }
  const problem = validateDump(body);
  if (problem) return detail(422, problem);
  const id = await storeDump(env.BUGREPORTS, body as Json, request, deps);
  return json({ id, url: `${url.origin}/?report=${id}` }, 201);
}

/** `GET /api/bugreport` — the probe. */
export function handleProbe(): Response {
  return json({ ok: true, store: "kv" });
}

/** `GET /intake/bugreport` — the probe, saying it is the intake: a
 *  signed-out `curl` of it proves the Access exemption works. */
export function handleIntakeProbe(): Response {
  return json({ ok: true, store: "kv", intake: true });
}

/** Any other path under `/intake/`: nothing lives there. */
export function handleIntakeMiss(): Response {
  return detail(404, "the intake has one route: POST /intake/bugreport");
}

/** `GET /api/bugreport/<id>`. */
export async function handleGet(id: string, env: Env): Promise<Response> {
  if (!ID_RE.test(id)) return detail(404, `unknown report: ${id}`);
  const text = await env.BUGREPORTS.get(KEY_PREFIX + id, "text");
  if (text == null) return detail(404, `unknown report: ${id}`);
  return new Response(text, {
    status: 200,
    // Immutable once written; a reader may cache it.
    headers: { "content-type": "application/json", "cache-control": "public, max-age=3600" },
  });
}
