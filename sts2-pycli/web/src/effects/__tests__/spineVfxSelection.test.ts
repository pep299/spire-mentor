import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { TextureAtlas } from "@esotericsoftware/spine-webgl";
import { describe, it, expect } from "vitest";
import {
  buildLiveEffect,
  spineExpired,
  SPINE_MAX_LIFETIME_MS,
  type LiveParticles,
  type LiveSpine,
} from "../../components/vfxLive";
import type {
  DissolveEffect,
  FlipbookEffect,
  ParticlesEffect,
  SpineEffect,
  SpriteShaderEffect,
  TransformAnimEffect,
  VfxManifest,
} from "../manifest";
import { effectsForEvent, type Bindings } from "../bindings";
import type { TargetRef } from "../types";

// `buildLiveEffect` is the pure routing decision `VfxLayer`'s sink makes —
// given a resolved manifest entry, which draw path (`Live` record) it
// becomes, or `null` for a kind with no renderer yet. jsdom has no WebGL, so
// this — not the render loop itself — is the seam `anchorRect` already
// established for testing `sink.play` without a canvas mock: assert
// selection and lifetime shape, never pixels.

const SPINE_EFFECT: SpineEffect = {
  kind: "spine",
  rig: "vfx_rigs/vfx_bite/vfx_bite",
  scale: 0.75,
  clip: "bite",
};

const FLIPBOOK_EFFECT: FlipbookEffect = {
  kind: "flipbook",
  frames: ["vfx/attack_slash/0.png", "vfx/attack_slash/1.png"],
  fps: 15,
  loop: false,
  scale: 1,
  randomizeRotation: false,
  rotationRange: [0, 0],
};

// Minimal well-formed fixture (Task 15 lands a real renderer for this kind,
// so `buildLiveEffect` now actually walks `emitters`/`mirrorX`/etc. — an
// empty composite is enough to exercise the ROUTING decision this suite
// checks, without needing a real emitter).
const PARTICLES_EFFECT: ParticlesEffect = {
  kind: "particles",
  lifetime: 1,
  emitters: [],
  mirrorX: null,
  pivotScale: null,
  tinted: false,
};

// Task 13 gives both of these a real draw path (`LiveSpriteShader` /
// `LiveTransformAnim`). Every kind now has one — `dissolve`'s is the debris
// burst, routed through the particle path (see `DISSOLVE_EFFECT` below).
const SPRITE_SHADER_EFFECT: SpriteShaderEffect = {
  kind: "sprite_shader",
  shader: "power_flash_vfx",
  blendMode: "blend_add",
  strength: 1,
  scale: 0.4,
  modulateAlpha: 1,
  tween: [[{ property: "scale", to: 0.4, duration: 0.4, ease: null, trans: null }]],
  lifetime: 0.4,
};

const TRANSFORM_ANIM_EFFECT: TransformAnimEffect = {
  kind: "transform_anim",
  lifetime: 0.4,
  sprites: [
    {
      name: "BlockBreakLeft",
      texture: "vfx/block_broken/left.png",
      positionX: null,
      positionY: null,
      rotation: null,
      restPositionX: -1,
      restPositionY: -10,
      restRotation: 0,
    },
  ],
};

// The REAL `vfx_monster_death` entry, transcribed from the committed manifest
// (one `CPUParticles2D` debris emitter) with two deliberate edits:
//
//  * `selfModulate` alpha is 0.8 rather than the scene's 1.0, so a build that
//    REPLACED the alpha with `modulateAlpha` instead of multiplying by it
//    fails — at 1.0 the two are indistinguishable.
//  * `emissionSphereRadius` is left at the manifest's 0, which is the whole
//    point: `extract_vfx.py` blanks the scene's 99.23 because
//    `NMonsterDeathVfx.cs:231` overwrites it per creature.
const DISSOLVE_EFFECT: DissolveEffect = {
  kind: "dissolve",
  shader: "dissolve",
  blendMode: "blend_premul_alpha",
  gradient1: "vfx/vfx_monster_death/gradient1.png",
  gradient2: "vfx/vfx_monster_death/gradient2.png",
  modulateAlpha: 0.466667,
  thresholdFrom: 1,
  thresholdTo: 0,
  dissolveEase: 1,
  dissolveTrans: 1,
  dissolveSeconds: 2.5,
  lifetime: 2.5,
  emissionRadiusPerCreatureExtent: 0.5,
  emitters: [
    {
      name: "Particles",
      texture: "vfx/monsters/decimillipede/decimillipede_blue_flipflake.png",
      material: null,
      slot: null,
      blendAdd: false,
      trailEnabled: false,
      trailLifetime: 0.3,
      amount: 500,
      cpu: true,
      explosiveness: 0.08,
      fixedFps: 0,
      lifetime: 1.8,
      localCoords: false,
      offset: [0, 0],
      oneShot: true,
      randomness: 0,
      rotation: 0,
      selfModulate: [1, 1, 1, 0.8],
      startDelay: 0,
      underMirrorX: false,
      underPivotScale: false,
      zIndex: 0,
      process: {
        alphaCurve: null,
        angleCurve: null,
        angleMax: 180,
        angleMin: -180,
        angularVelocityCurve: null,
        angularVelocityMax: 100.46,
        angularVelocityMin: -108.83,
        animOffsetMax: 1,
        animOffsetMin: 0,
        animSpeedCurve: null,
        animSpeedMax: 1.5,
        animSpeedMin: 0.5,
        colorInitialRamp: null,
        colorRamp: null,
        dampingCurve: null,
        dampingMax: 0,
        dampingMin: 0,
        direction: [1, -0.5, 0],
        emissionBoxExtents: [1, 1, 1],
        emissionRingAxis: [0, 0, 1],
        emissionRingConeAngle: 90,
        emissionRingHeight: 1,
        emissionRingInnerRadius: 0,
        emissionRingRadius: 1,
        emissionShape: 1,
        emissionShapeOffset: [0, 0, 0],
        emissionShapeScale: [1, 1, 1],
        emissionSphereRadius: 0,
        flatness: 0,
        gravity: [0, -100, 0],
        initialVelocityMax: 150,
        initialVelocityMin: 100,
        lifetimeRandomness: 0.75,
        particleFlagAlignY: false,
        particleFlagDampingAsFriction: false,
        particleFlagDisableZ: false,
        particleFlagRotateY: false,
        radialVelocityCurve: null,
        radialVelocityMax: 0,
        radialVelocityMin: 0,
        scaleCurve: "vfx_particles/0b4426afdf8ac9c9.json",
        scaleMax: 0.07,
        scaleMin: 0.05,
        scaleOverVelocityCurve: null,
        scaleOverVelocityMax: 0,
        spread: 30,
        turbulenceEnabled: false,
        turbulenceInfluenceMax: 0.1,
        turbulenceInfluenceMin: 0.1,
        turbulenceInfluenceOverLifeCurve: null,
        turbulenceInitialDisplacementMax: 0,
        turbulenceInitialDisplacementMin: 0,
        turbulenceNoiseScale: 9,
        turbulenceNoiseSpeed: [0, 0, 0],
        turbulenceNoiseSpeedRandom: 0.2,
        turbulenceNoiseStrength: 1,
      },
    },
  ],
};

describe("buildLiveEffect", () => {
  it("routes a spine-kind manifest entry to the spine draw path", () => {
    const live = buildLiveEffect(SPINE_EFFECT, 10, 20, "rev1");
    expect(live?.kind).toBe("spine");
  });

  it("a spine live record starts with no skeleton/state/track — built lazily once its rig loads", () => {
    const live = buildLiveEffect(SPINE_EFFECT, 10, 20, "rev1");
    expect(live).toMatchObject({ kind: "spine", skeleton: null, state: null, track: null, x: 10, y: 20 });
  });

  it("still routes a flipbook-kind entry to the flipbook draw path", () => {
    const live = buildLiveEffect(FLIPBOOK_EFFECT, 0, 0, "rev1");
    expect(live?.kind).toBe("flipbook");
  });

  it("routes a particles-kind entry to the particle-composite draw path", () => {
    const live = buildLiveEffect(PARTICLES_EFFECT, 0, 0, "rev1");
    expect(live?.kind).toBe("particles");
  });

  it("routes a sprite_shader-kind entry (vfx_power_flash_vfx) to its own draw path", () => {
    const live = buildLiveEffect(SPRITE_SHADER_EFFECT, 0, 0, "rev1");
    expect(live?.kind).toBe("sprite_shader");
  });

  it("carries the icon URL onto the sprite_shader live record — the manifest has no texture for this kind", () => {
    const live = buildLiveEffect(SPRITE_SHADER_EFFECT, 0, 0, "rev1", null, "https://example.test/icon.png");
    expect(live).toMatchObject({ kind: "sprite_shader", iconUrl: "https://example.test/icon.png" });
  });

  it("a sprite_shader live record with no icon on record carries iconUrl: null, not a crash", () => {
    const live = buildLiveEffect(SPRITE_SHADER_EFFECT, 0, 0, "rev1");
    expect(live).toMatchObject({ kind: "sprite_shader", iconUrl: null });
  });

  it("routes a transform_anim-kind entry (vfx_block_broken) to its own draw path", () => {
    const live = buildLiveEffect(TRANSFORM_ANIM_EFFECT, 0, 0, "rev1");
    expect(live?.kind).toBe("transform_anim");
  });

  it("plays NOTHING for a `dissolve` with no per-creature measurements", () => {
    // `buildLiveEffect` used to end in an unconditional
    // `return { kind: "spine", ... }` fall-through, which typed a
    // `transform_anim`/`sprite_shader` entry AS a Spine effect and then read
    // `entry.rig`/`entry.clip` off it in the draw loop. `null` is what the
    // sink already handles by pushing nothing.
    //
    // `dissolve` reaches that answer for its OWN reason now that it has a
    // draw path: the burst's radius is assigned per creature by the game
    // (`NMonsterDeathVfx.cs:231`) and `extract_vfx.py` blanks the scene's
    // placeholder to 0, so an unmeasurable creature has no radius and the
    // honest outcome is silence — not a burst collapsed onto one point.
    expect(buildLiveEffect(DISSOLVE_EFFECT, 0, 0, "rev1")).toBeNull();
  });

  it("routes a `dissolve` WITH measurements to the particle draw path", () => {
    // The debris burst is the half of `vfx_monster_death` that lives on the
    // shared overlay; the dissolve SPRITE is drawn per creature inside
    // `SpineSprite` and never reaches here. Routing through `particles` is
    // also what gives the burst the default alpha-mix blend rather than the
    // sprite's `blend_premul_alpha` — a `CanvasItem`'s material does not
    // descend to its children, so the game draws it that way too.
    const live = buildLiveEffect(
      DISSOLVE_EFFECT, 0, 0, "rev1", null, null, { emissionRadius: 48, lifetime: 1.25 },
    );
    expect(live?.kind).toBe("particles");
    expect((live as LiveParticles).composite.emitters).toHaveLength(1);
    // The composite's life is the DISSOLVE TWEEN's, narrowed per creature —
    // NOT the manifest's 2.5 s ceiling, and not the emitter's own 1.8 s.
    expect((live as LiveParticles).composite.lifetime).toBe(1.25);
  });

  it("gives the burst the caller's radius and the parent node's alpha", () => {
    // Two things the manifest cannot bake in, both verified on the built
    // effect rather than on the drawn pixels:
    //
    // 1. `emission_sphere_radius`. The manifest's is 0 on purpose — a burst
    //    that used it would put all 500 particles on one point.
    // 2. `Visual`'s `modulate` alpha. The burst node is `parent="Visual"` in
    //    the scene and Godot's `modulate` descends to children, so the game
    //    draws the debris at 0.466667 too. Applying it to the sprite alone
    //    renders the burst ~2x too bright.
    const live = buildLiveEffect(
      DISSOLVE_EFFECT, 0, 0, "rev1", null, null, { emissionRadius: 48, lifetime: 2.5 },
    ) as LiveParticles;
    const em = live.composite.emitters[0]!.emitter;
    expect(em.process.emissionSphereRadius).toBe(48);
    // 0.8 (the fixture's own self_modulate alpha) * 0.466667.
    expect(em.selfModulate[3]).toBeCloseTo(0.8 * 0.466667, 10);
    // ...and the RGB is untouched: `modulate` here is an alpha-only Color.
    expect(em.selfModulate.slice(0, 3)).toEqual([1, 1, 1]);
  });

  it("does not mutate the manifest entry it was handed", () => {
    // `dissolveBurstEffect` folds a parent's alpha into a child's field, which
    // is exactly the kind of edit that must not be in-place: the manifest is
    // shared, long-lived and read by `SpineSprite` too, so a second death
    // would otherwise apply 0.466667 again — and a third again after that.
    const before = structuredClone(DISSOLVE_EFFECT);
    buildLiveEffect(DISSOLVE_EFFECT, 0, 0, "rev1", null, null, { emissionRadius: 48, lifetime: 2.5 });
    buildLiveEffect(DISSOLVE_EFFECT, 0, 0, "rev1", null, null, { emissionRadius: 48, lifetime: 2.5 });
    expect(DISSOLVE_EFFECT).toEqual(before);
  });

  it("a spine live record is not expired the moment it's created", () => {
    const live = buildLiveEffect(SPINE_EFFECT, 0, 0, "rev1") as LiveSpine;
    expect(spineExpired(live, live.startedAt)).toBe(false);
  });

  it("a LiveSpine whose rig never finishes loading is eventually dropped rather than retained forever", () => {
    // The draw loop's "still loading, keep waiting" branch has no other exit
    // for a rig that never resolves (a real network failure past the point
    // `AssetManagerBase` ever retries writes only to its own `errors` map,
    // never to `cache.assets`) — `spineExpired`'s wall-clock cap is the
    // backstop that frees the instance regardless, so a stuck load can never
    // accumulate in `liveRef.current` forever.
    const live = buildLiveEffect(SPINE_EFFECT, 0, 0, "rev1") as LiveSpine;
    expect(spineExpired(live, live.startedAt + SPINE_MAX_LIFETIME_MS - 1)).toBe(false);
    expect(spineExpired(live, live.startedAt + SPINE_MAX_LIFETIME_MS + 1)).toBe(true);
  });
});

// The EXTRACTED (and committed) manifest + binding table, read for real —
// same pattern as `manifestShape.test.ts` / `bindingsRealTable.test.ts`, and
// for the same reason: the synthetic fixtures above prove this code handles
// a spine entry SHAPED the way we assume it is; they don't prove the real
// table still selects one, or that `vfx_adrenaline` is still unreachable
// through it. Read at RUNTIME, not `import`ed — `sts2-wiki/` is
// `.dockerignore`d and the web build stage copies only `sts2-pycli/web/`, so
// a static import would fail `npm run build` inside the container. Absent
// tree => the suite skips; present tree => it checks.
const WEB_ROOT = resolve(__dirname, "../../..");
const MANIFEST_PATH = resolve(WEB_ROOT, "../../sts2-wiki/site/public/images/vfx/manifest.json");

function readManifest(): VfxManifest | null {
  try {
    return JSON.parse(readFileSync(MANIFEST_PATH, "utf8")) as VfxManifest;
  } catch {
    return null;
  }
}

const manifest = readManifest();
const manifestSuite = manifest ? describe : describe.skip;

manifestSuite("the four newly-reachable Spine effects, against the real committed manifest", () => {
  const loaded = manifest as VfxManifest;

  it.each(["vfx_bite", "vfx_scratch", "vfx_flying_slash", "vfx_gaze"])(
    "%s is a spine-kind entry that routes to the spine draw path",
    (name) => {
      const entry = loaded.effects[name];
      expect(entry?.kind).toBe("spine");
      const live = buildLiveEffect(entry!, 0, 0, loaded.rev);
      expect(live?.kind).toBe("spine");
    },
  );

  it("still ignores vfx_adrenaline for selection", () => {
    // Full-screen, played by `VfxCmd.PlayFullScreenInCombat` — sub-project 5.
    // The manifest still carries it (sub-project 1 extracted all six Spine
    // rigs), and this renderer would happily draw it if asked — the
    // exemption lives entirely in the binding table never naming it, which
    // is what the assertion below actually pins.
    expect(loaded.effects.vfx_adrenaline?.kind).toBe("spine");
  });
});

const BINDINGS_PATH = resolve(WEB_ROOT, "../../sts2-wiki/site/public/images/vfx/vfx_bindings.json");

function readBindings(): Bindings | null {
  try {
    return JSON.parse(readFileSync(BINDINGS_PATH, "utf8")) as Bindings;
  } catch {
    return null;
  }
}

const bindings = readBindings();
const bindingsSuite = bindings ? describe : describe.skip;

const E: TargetRef = { side: "enemy", index: 0 };

bindingsSuite("vfx_adrenaline stays unselectable through the real binding table", () => {
  const table = bindings as Bindings;

  it("no bound row names vfx_adrenaline as its effect", () => {
    expect(table.bindings.some((row) => row.effect === "vfx_adrenaline")).toBe(false);
  });

  it("effectsForEvent never selects it for an ordinary hit, even with no matching source", () => {
    // No `source` at all -> the `<unbound>` fallback path (`vfx_attack_slash`),
    // not a full-screen effect that happens to share a fallback chain with it.
    const picks = effectsForEvent({ kind: "hurt", target: E, amount: 6 }, table);
    expect(picks.map((p) => p.effect)).not.toContain("vfx_adrenaline");
  });
});

// `VfxLayer`'s `fileAlias` cache-bust (`ensureSpineRigLoading`) aliases
// exactly one atlas page, named `<rig-basename>.png`. That assumption holds
// for every Spine rig the binding table can select TODAY, but it is not a
// property of the manifest shape — `vfx_adrenaline` (still unselectable,
// exempted above) already declares a SECOND page
// (`vfx_adrenaline_2.png`) that convention doesn't cover, and it broke
// exactly the way you'd expect the first time this renderer's eager preload
// touched it (see task-10-report.md's corrected root cause). A future
// binding could name a rig like that before anyone remembers to generalize
// `fileAlias` — this guard catches that at the table-extraction boundary
// instead of at a silent broken-texture render.
const bothSuite = manifest && bindings ? describe : describe.skip;

bothSuite("every Spine rig the binding table can select declares exactly one atlas page", () => {
  // `describe.skip` still runs this collector, so the absent-tree case must
  // bail here rather than dereference the nulls (a bring-your-own-game
  // checkout without extracted VFX).
  if (!manifest || !bindings) return;
  const m = manifest as VfxManifest;
  const b = bindings as Bindings;
  const boundEffects = new Set(b.bindings.map((row) => row.effect));
  const selectableSpineNames = Object.entries(m.effects)
    .filter(([name, e]) => e.kind === "spine" && boundEffects.has(name))
    .map(([name]) => name);

  it("has at least one selectable spine rig to check — a regressed table would make the check below vacuous", () => {
    expect(selectableSpineNames.length).toBeGreaterThan(0);
  });

  it.each(selectableSpineNames)("%s's atlas declares exactly one page", (name) => {
    const entry = m.effects[name] as SpineEffect;
    const atlasPath = resolve(WEB_ROOT, "../../sts2-wiki/site/public/images", `${entry.rig}.atlas`);
    const atlas = new TextureAtlas(readFileSync(atlasPath, "utf8"));
    expect(
      atlas.pages.length,
      `${name} (${entry.rig}.atlas) declares ${atlas.pages.length} pages: ` +
        `[${atlas.pages.map((p) => p.name).join(", ")}]. VfxLayer's fileAlias ` +
        `cache-bust only aliases "<rig-basename>.png" — extend ` +
        `ensureSpineRigLoading to alias every declared page before this rig ` +
        `can load correctly.`,
    ).toBe(1);
  });
});
