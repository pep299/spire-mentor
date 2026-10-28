import { describe, it, expect, beforeEach, vi } from "vitest";
import { eventAssetUrl, resolveMonsterAttack, type MonsterSfx, type SfxMap } from "../sfxMap";

const OWL: MonsterSfx = {
  moves: [
    { name: "MAGISTRATE_SCRUTINY", intent: "Attack", hits: 1, event: "event:/sfx/e/scrutiny" },
    { name: "PECK_ASSAULT", intent: "Attack", hits: 6, event: "event:/sfx/e/peck" },
    { name: "JUDICIAL_FLIGHT", intent: "Buff", hits: 0, event: "event:/sfx/e/takeoff" },
  ],
  defaultAttack: "event:/sfx/e/scrutiny",
};

describe("eventAssetUrl", () => {
  it("strips the event scheme and appends .webm", () => {
    expect(eventAssetUrl("event:/sfx/block_gain")).toBe("/assets/audio/sfx/block_gain.webm");
  });
  it("handles an event outside the sfx tree", () => {
    // temp_sfx.bank events live at event:/temp/sfx/…, not event:/sfx/….
    expect(eventAssetUrl("event:/temp/sfx/game_over")).toBe("/assets/audio/temp/sfx/game_over.webm");
  });

  // The game's SECOND sound scheme: plain Godot streams under
  // `res://debug_audio/`, which is where combat victory, relic pickups and
  // every card hit sound live. `encode_sfx` transcodes them into the same
  // tree, so the SOURCE extension must be replaced, not kept — a URL ending
  // `victory.mp3.webm` 404s, and the cue goes silent with only a console
  // warning.
  it("maps a debug-audio stream into the same tree, dropping its extension", () => {
    expect(eventAssetUrl("res://debug_audio/victory.mp3"))
      .toBe("/assets/audio/debug_audio/victory.webm");
  });
  it("drops a non-mp3 debug-audio extension too", () => {
    // `regent_intro.wav` and `ui_click.wav` ship alongside the mp3s.
    expect(eventAssetUrl("res://debug_audio/ui_click.wav"))
      .toBe("/assets/audio/debug_audio/ui_click.webm");
  });

  // These two live here rather than under `loadSfxMap` because the thing
  // under test is the URL, and the module-level rev is only observable
  // through it. `vi.resetModules()` in each is what makes the "before a map
  // loads" case reachable at all — the rev is module state, so without a
  // fresh import it would carry over from whichever test ran last.
  it("appends no query string before a map has loaded", async () => {
    // Mutation caught: `?v=${audioRev}` unconditionally, which would ship a
    // literal `?v=` on every request made before the map resolves.
    vi.resetModules();
    const fresh = await import("../sfxMap");
    expect(fresh.eventAssetUrl("event:/sfx/block_gain")).toBe("/assets/audio/sfx/block_gain.webm");
  });

  it("busts the cache with the loaded map's audioRev", async () => {
    // The whole point: these URLs are served `immutable, max-age=1y` but are
    // NOT content-addressed, so a re-encode is invisible to an already-primed
    // browser unless the URL moves. Mutation caught: `loadSfxMap` not setting
    // the rev, or setting it after resolving.
    vi.resetModules();
    const fresh = await import("../sfxMap");
    const map: SfxMap = {
      version: "v1", audioRev: "deadbeefcafe0001",
      generic: {}, powers: {}, cards: {}, relics: {}, monsters: {},
    };
    await fresh.loadSfxMap(vi.fn().mockResolvedValue({ ok: true, json: async () => map }));
    expect(fresh.eventAssetUrl("event:/sfx/block_gain"))
      .toBe("/assets/audio/sfx/block_gain.webm?v=deadbeefcafe0001");
  });
});

describe("resolveMonsterAttack", () => {
  it("matches the move whose hit count matches the intent", () => {
    // Mutation caught: exact-match filter dropped, or compared against the wrong field.
    expect(resolveMonsterAttack(OWL, { hits: 6 }, "event:/sfx/block_hit")).toBe("event:/sfx/e/peck");
  });

  it("falls back to the monster's default attack when no move's hit count matches", () => {
    // Mutation caught: a `?? 1`/`?? 2` default on the incoming hits that happens to
    // coincide with a real hit count. NOTE: `exact.length` here is 0 either way (no
    // move has hits:2), so this does NOT exercise `exact.length === 1` vs `>= 1` —
    // see "falls back ... when two moves genuinely tie" below for that branch.
    expect(resolveMonsterAttack(OWL, { hits: 2 }, "event:/sfx/block_hit")).toBe("event:/sfx/e/scrutiny");
  });

  it("falls back to the monster's default attack when two moves genuinely tie", () => {
    // Verified against the actual mutation (checked outside this tree, see task-5
    // fix report): `exact.length === 1` weakened to `exact.length >= 1` returns
    // "event:/sfx/e/swipe_a" here instead of the default — this test catches it,
    // where the single-match/no-match tests above and below do not (their
    // `exact.length` is never 2).
    const monster: MonsterSfx = {
      moves: [
        { name: "SWIPE_A", intent: "Attack", hits: 2, event: "event:/sfx/e/swipe_a" },
        { name: "SWIPE_B", intent: "Attack", hits: 2, event: "event:/sfx/e/swipe_b" },
      ],
      defaultAttack: "event:/sfx/e/default",
    };
    expect(resolveMonsterAttack(monster, { hits: 2 }, "event:/sfx/block_hit")).toBe("event:/sfx/e/default");
  });

  it("never returns a non-attack move", () => {
    // Mutation caught: the `intent === "Attack"` filter removed, letting a Buff move
    // with a coincidentally-matching hit count (0) win.
    expect(resolveMonsterAttack(OWL, { hits: 0 }, "event:/sfx/block_hit")).toBe("event:/sfx/e/scrutiny");
  });

  it("degrades to the caller's fallback for an unknown monster", () => {
    // Mutation caught: the `!m` guard removed or the fallback param ignored.
    expect(resolveMonsterAttack(undefined, { hits: 1 }, "event:/sfx/block_hit")).toBe("event:/sfx/block_hit");
  });

  it("treats a missing intent.hits as unable to discriminate, not as a specific count", () => {
    // Real data: many moves report a hit count the intent side never sends. Defaulting
    // the missing value to 0 or 1 would silently pick whichever move happens to have
    // that count, e.g. `intent?.hits ?? 1` mismatching this fixture's hits:1 move.
    // Mutation caught: any `?? 0` / `?? 1` default replacing "skip the exact match".
    const monster: MonsterSfx = {
      moves: [
        { name: "JAB", intent: "Attack", hits: 1, event: "event:/sfx/e/jab" },
        { name: "FLURRY", intent: "Attack", hits: 3, event: "event:/sfx/e/flurry" },
      ],
      defaultAttack: "event:/sfx/e/flurry",
    };
    expect(resolveMonsterAttack(monster, {}, "event:/sfx/block_hit")).toBe("event:/sfx/e/flurry");
    expect(resolveMonsterAttack(monster, undefined, "event:/sfx/block_hit")).toBe("event:/sfx/e/flurry");
  });

  it("never matches a move whose own hits could not be resolved from the decomp", () => {
    // Real fixture: TestSubject's MULTI_CLAW_MOVE is Attack with damage but no hits.
    // Mutation caught: `mv.hits === hits` replaced by a loose/coerced comparison
    // (e.g. `(mv.hits ?? 0) === (hits ?? 0)`) that lets an unresolved move match.
    const monster: MonsterSfx = {
      moves: [
        { name: "MULTI_CLAW_MOVE", intent: "Attack", event: "event:/sfx/e/claw" },
        { name: "SLASH", intent: "Attack", hits: 2, event: "event:/sfx/e/slash" },
      ],
      defaultAttack: "event:/sfx/e/slash",
    };
    expect(resolveMonsterAttack(monster, { hits: 2 }, "event:/sfx/block_hit")).toBe("event:/sfx/e/slash");
  });

  it("does not let an unresolved-hits move stand in for an explicit hits:0 move", () => {
    // Verified against the actual mutation (checked outside this tree, see task-5
    // fix report): `mv.hits === hits` loosened to `(mv.hits ?? 0) === (hits ?? 0)`
    // makes UNRESOLVED_CLAW (no hits) coerce to 0 and tie with ZERO_HIT_JAB
    // (hits:0) — two "matches" instead of one, so it falls to the default instead
    // of the correct single move. The earlier MULTI_CLAW_MOVE test above does NOT
    // catch this: its default coincides with the wrongly-resolved event, so a
    // coerced comparison is invisible there.
    const monster: MonsterSfx = {
      moves: [
        { name: "UNRESOLVED_CLAW", intent: "Attack", event: "event:/sfx/e/claw" },
        { name: "ZERO_HIT_JAB", intent: "Attack", hits: 0, event: "event:/sfx/e/jab0" },
      ],
      defaultAttack: "event:/sfx/e/default",
    };
    expect(resolveMonsterAttack(monster, { hits: 0 }, "event:/sfx/block_hit")).toBe("event:/sfx/e/jab0");
  });

  it("falls back to the caller's fallback when the monster has no attack moves and no default", () => {
    // Mutation caught: `attacks[0]?.event ?? fallback` weakened to always return
    // `attacks[0]?.event` (crashing or returning undefined instead of the fallback).
    const monster: MonsterSfx = {
      moves: [{ name: "HIDE", intent: "Hidden", event: "event:/sfx/e/hide" }],
    };
    expect(resolveMonsterAttack(monster, { hits: 1 }, "event:/sfx/block_hit")).toBe("event:/sfx/block_hit");
  });

  it("falls back to the first attack move when there is no explicit default", () => {
    // Mutation caught: `m.defaultAttack ?? attacks[0]?.event` collapsed to just
    // `m.defaultAttack`, losing the attacks[0] fallback tier entirely.
    const monster: MonsterSfx = {
      moves: [
        { name: "ONLY_MOVE", intent: "Attack", hits: 5, event: "event:/sfx/e/only" },
      ],
    };
    expect(resolveMonsterAttack(monster, { hits: 2 }, "event:/sfx/block_hit")).toBe("event:/sfx/e/only");
  });
});

describe("loadSfxMap", () => {
  // `cached` is private module state, shared across every test in this file.
  // Reset the module between tests so each one starts from a clean slate.
  beforeEach(() => {
    vi.resetModules();
  });

  it("memoises a successful fetch — later calls don't fetch again", async () => {
    // Mutation caught: dropping the `cached ??=` memoisation (or memoising per
    // call) would make fetchFn get called more than once here.
    const { loadSfxMap: freshLoadSfxMap } = await import("../sfxMap");
    const map: SfxMap = {
      version: "v1",
      generic: {},
      powers: {},
      cards: {},
      relics: {},
      monsters: {},
    };
    const fetchFn = vi.fn().mockResolvedValue({ ok: true, json: async () => map });

    const first = await freshLoadSfxMap(fetchFn);
    const second = await freshLoadSfxMap(fetchFn);

    expect(first).toEqual(map);
    expect(second).toEqual(map);
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it("does not memoise a failed fetch — a later call retries and can succeed", async () => {
    // Regression test for the bug this fix addresses: `cached ??=` alone leaves
    // a REJECTED promise in `cached` forever (a rejection is neither null nor
    // undefined, so `??=` never reassigns it) — every later call replayed the
    // same rejection instead of retrying, wedging sound off for the session.
    const { loadSfxMap: freshLoadSfxMap } = await import("../sfxMap");
    const map: SfxMap = {
      version: "v1",
      generic: {},
      powers: {},
      cards: {},
      relics: {},
      monsters: {},
    };
    const fetchFn = vi
      .fn()
      .mockRejectedValueOnce(new Error("network blip"))
      .mockResolvedValueOnce({ ok: true, json: async () => map });

    await expect(freshLoadSfxMap(fetchFn)).rejects.toThrow("network blip");
    await expect(freshLoadSfxMap(fetchFn)).resolves.toEqual(map);
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  // GH #877: each game build has its own map, built from its own decompile.
  it("loads a non-default build's own map, memoised apart from the default's", async () => {
    const { loadSfxMap: freshLoadSfxMap } = await import("../sfxMap");
    const mapOf = (version: string): SfxMap => ({ version, generic: {}, powers: {}, cards: {}, relics: {}, monsters: {} });
    const fetchFn = vi.fn(async (url: string) => ({
      ok: true,
      json: async () => mapOf(url === "/v0.111.0/sfx_map.json" ? "v0.111.0" : "v0.107.1"),
    }));
    const f = fetchFn as unknown as typeof fetch;

    expect((await freshLoadSfxMap(f, "v0.111.0")).version).toBe("v0.111.0");
    expect((await freshLoadSfxMap(f)).version).toBe("v0.107.1");
    expect((await freshLoadSfxMap(f, "v0.107.1")).version).toBe("v0.107.1");
    expect((await freshLoadSfxMap(f, "v0.111.0")).version).toBe("v0.111.0");
    expect(fetchFn.mock.calls.map((c) => c[0])).toEqual(["/v0.111.0/sfx_map.json", "/sfx_map.json"]);
  });

  it("falls back to the default build's map when a build has none (the public snapshot)", async () => {
    const { loadSfxMap: freshLoadSfxMap } = await import("../sfxMap");
    const map: SfxMap = { version: "v0.107.1", generic: {}, powers: {}, cards: {}, relics: {}, monsters: {} };
    const fetchFn = vi.fn(async (url: string) =>
      url === "/sfx_map.json" ? { ok: true, json: async () => map } : { ok: false, status: 404, json: async () => ({}) });

    await expect(freshLoadSfxMap(fetchFn as unknown as typeof fetch, "v0.111.0")).resolves.toEqual(map);
    expect(fetchFn.mock.calls.map((c) => c[0])).toEqual(["/v0.111.0/sfx_map.json", "/sfx_map.json"]);
  });
});
