import { useEffect, useRef, useState, type RefObject } from "react";
import * as spine from "@esotericsoftware/spine-webgl";
import {
  effectAssetUrl,
  type DissolveEffect,
  type FlipbookEffect,
  type ParticlesEffect,
  type SpineEffect,
  type SpriteShaderEffect,
  type TransformAnimEffect,
  type VfxEffect,
  type VfxManifest,
} from "../effects/manifest";
import type { EffectSink, PlayedEffect } from "../effects/sink";
import type { TargetRef } from "../effects/types";
import { startComposite, isDone, type LiveComposite } from "../effects/particles/composite";
import { makeRng } from "../effects/particles/rng";
import { advanceTo, createSimState, readout, type SimState } from "../effects/particles/simulate";
import { sampleTrack } from "../effects/transformAnim";
import {
  appendPowerFlashQuadVertices,
  appendQuadVertices,
  buildParticleSystemSpec,
  createLutTexture,
  createParticleProgram,
  createPowerFlashProgram,
  drawParticleBatch,
  drawPowerFlashQuad,
  flipbookCell,
  maskChannelOf,
  particleDrawAngle,
  particleVertexColor,
  scalarCurvePathsOf,
  trailQuadGeometry,
  type BatchMaterialState,
  type LutTexture,
  type ParticleProgram,
  type PowerFlashProgram,
} from "../effects/particles/shaders";

/**
 * The DOM box an effect spawns on, or `null` when that creature is not on the
 * board. Anchors are the ones `Combatant.tsx` already emits, and *arena* is
 * `.gs-actors` — the container they are drawn in (see `Board.tsx`).
 *
 * Osty deliberately returns `null`: it has no `data-combatant` anchor, and
 * sub-project 1 leaves it to the existing CSS effects.
 */
export function anchorRect(target: TargetRef, arena: HTMLElement): DOMRect | null {
  const sel =
    target.side === "enemy"
      ? `[data-enemy-index="${target.index}"]`
      : target.side === "player"
        ? `[data-combatant="player"]`
        : null;
  if (!sel) return null;
  const el = arena.querySelector(sel);
  return el ? (el as HTMLElement).getBoundingClientRect() : null;
}

/**
 * The block BADGE's own box, or `null` when it is not on screen —
 * `vfx_block_broken`'s anchor (`sink.ts`'s `PlayedEffect.anchor` doc:
 * `NHealthBar.cs:575`'s `_blockContainer.GlobalPosition +
 * _blockContainer.Size * 0.5`), which is a DIFFERENT element than the
 * creature `anchorRect` resolves — `block_spark_vfx` anchors to the creature
 * itself, `vfx_block_broken` does not, and the two must not be unified (see
 * that doc for the full C# citation).
 *
 * Looked up as a DESCENDANT of the creature's own anchor element
 * (`Combatant.tsx`'s `data-block-badge` div), not a standalone selector — the
 * badge has no target-identifying attribute of its own, only the creature
 * container does. Returns `null` only when the creature itself is off-board
 * (a rig still loading, or a slot with nothing in it) — `Combatant.tsx` keeps
 * the badge div ALWAYS mounted once the rig is up (only its number/opacity
 * toggle with `c.block`), specifically so this never returns null for the one
 * case that matters: a `blockBreak` event, which by construction only ever
 * fires when block was just non-zero (`combatDiff.ts`: `prev.block > 0 &&
 * next.block === 0`), so the badge was genuinely on screen right up to this
 * transition. See `Combatant.tsx`'s doc on that div for why a
 * conditionally-mounted badge doesn't work here: `Board.tsx`'s `blockBreak`
 * handling runs in a `useEffect` AFTER `c.block` has already committed to 0,
 * so gating the anchor's OWN presence on `c.block > 0` (or even a same-tick
 * `Fx` flag meant to bridge the gap) reads the same too-late DOM — it looked
 * like a fix and wasn't, verified by tracing React's render-then-effect
 * ordering, not assumed.
 */
export function badgeRect(target: TargetRef, arena: HTMLElement): DOMRect | null {
  const sel =
    target.side === "enemy"
      ? `[data-enemy-index="${target.index}"]`
      : target.side === "player"
        ? `[data-combatant="player"]`
        : null;
  if (!sel) return null;
  const creature = arena.querySelector(sel);
  if (!creature) return null;
  const badge = creature.querySelector("[data-block-badge]");
  return badge ? (badge as HTMLElement).getBoundingClientRect() : null;
}

interface LiveFlipbook {
  kind: "flipbook";
  effect: FlipbookEffect;
  texturesRev: string;
  /** Canvas-space centre, Y-up. */
  x: number;
  y: number;
  /** Godot's own `Node2D.RotationDegrees`, straight from the scene's authored
   *  `_rotationRange` — NOT a draw angle. Convert with
   *  {@link drawAngleFromGodotDegrees} at the draw call. */
  angleDeg: number;
  startedAt: number;
}

/**
 * A Godot 2D rotation, in DEGREES, as the `angle` argument of `spine-webgl`'s
 * `SceneRenderer.drawTextureRotated`.
 *
 * Two conversions, and both are needed — either one alone is wrong:
 *
 * 1. **Unit.** `drawTextureRotated`'s `angle` is DEGREES: `SceneRenderer.js:247`
 *    hands it to `MathUtils.cosDeg`/`sinDeg`, which multiply by `Math.PI / 180`
 *    themselves (`Utils.js:170`). Handing it radians makes it re-convert an
 *    already-converted value, which is what the flipbook branch used to do —
 *    an authored 0-25° jitter came out at 1/57th of a degree, i.e. inert.
 * 2. **Sign.** Godot's 2D rotation is visually CLOCKWISE in its Y-down space;
 *    the rotation matrix `drawTextureRotated` builds is counter-clockwise in a
 *    Y-up one, and this canvas is Y-up. The same axis flip that already negates
 *    particle and sprite Y for position inverts rotational chirality too. The
 *    flipbooks' `_rotationRange` is `(0, 25)` — asymmetric — so fixing only the
 *    unit would spin all 14 shipped composites the wrong way.
 *
 * `NSpriteAnimator._Ready` (`NSpriteAnimator.cs:91-93`) is the authored source:
 * `RotationDegrees = new Random().Next((int)_rotationRange.X, (int)_rotationRange.Y)`.
 */
export function drawAngleFromGodotDegrees(deg: number): number {
  return -deg;
}

/** {@link drawAngleFromGodotDegrees} for a Godot rotation in RADIANS — the unit
 *  `Animation` rotation tracks (and `Node2D.Rotation`) are authored in. */
export function drawAngleFromGodotRadians(rad: number): number {
  return drawAngleFromGodotDegrees((rad * 180) / Math.PI);
}

export interface LiveSpine {
  kind: "spine";
  effect: SpineEffect;
  /** Canvas-space centre, Y-up — same convention as `LiveFlipbook`. */
  x: number;
  y: number;
  /** Built lazily on the first frame whose rig assets have finished loading
   *  (the atlas/skeleton binary race the first hits of a fight, same as a
   *  flipbook frame still in flight). `null` until then. */
  skeleton: spine.Skeleton | null;
  state: spine.AnimationState | null;
  /** The one-shot track `state.setAnimation` returned — `isComplete()` on
   *  this, not a timer, is what frees the instance under the normal path, so
   *  a clip's own duration (never modeled separately here) is exactly how
   *  long it lives. `null` until `skeleton`/`state` are built, or if the rig
   *  has no `clip` at all. See `spineExpired` for the ABNORMAL path (a rig
   *  whose assets never finish loading). */
  track: spine.TrackEntry | null;
  /** When this instance was created (`performance.now()`), for
   *  `spineExpired`'s wall-clock backstop — independent of load success or
   *  `track`, unlike `LiveFlipbook`'s `startedAt` (which drives its actual
   *  frame index, not just a safety cap). */
  startedAt: number;
  /** Slot-parented emitters this rig has STARTED, keyed by emitter name, with
   *  the `performance.now()` its rig event fired at and its own simulation
   *  (GH #669). Empty until the rig raises the event — the burst is not part
   *  of the effect's opening frame, it is `NKaiserCrabBossExplosionVfx`
   *  reacting to `left_embers_start` partway through the clip. Absent
   *  entirely for a rig with no `emitters`. */
  slotEmitters?: Map<string, { startedAt: number; sim: SimState | null; seed: number }>;
}

/** A started particle composite (`kind: "particles"`, Task 15). The heavy
 *  lifting — which emitters exist, when each starts, the resolved tint, the
 *  bespoke-script randomiser — lives in `startComposite`/`isDone`
 *  (`effects/particles/composite.ts`), which is what's unit-tested; this is
 *  just the wrapper the draw loop's `Live` union needs (anchor position +
 *  wall-clock start), the same shape `LiveFlipbook`/`LiveSpine` already have. */
export interface LiveParticles {
  kind: "particles";
  composite: LiveComposite;
  /** One resumable simulation per emitter, positionally aligned with
   *  `composite.emitters`, each advanced once per display frame. `null` until
   *  the draw loop can build that emitter's spec — its curve LUTs load
   *  asynchronously, so an emitter may be several frames late; whenever it
   *  does start, `advanceTo` catches it up in one jump, which is the same
   *  frame sequence it would have run had it started on time.
   *
   *  Held here rather than on `LiveComposite` to keep `composite.ts` pure
   *  data: this is renderer state, and it is mutable by design. */
  sims: (SimState | null)[];
  /** Canvas-space centre, Y-up — same convention as `LiveFlipbook`/`LiveSpine`. */
  x: number;
  y: number;
  startedAt: number;
}

/** A started `transform_anim` effect (`vfx_block_broken`'s two halves, Task
 *  13). Purely a wall-clock playhead into `effect.sprites`' baked tracks —
 *  `sampleTrack` (a pure function, unit-tested on its own) does the actual
 *  sampling; this is only the anchor + start-time wrapper the draw loop's
 *  `Live` union needs, the same shape every other kind here has. */
interface LiveTransformAnim {
  kind: "transform_anim";
  effect: TransformAnimEffect;
  /** Canvas-space spawn point, Y-up — the block BADGE's centre for
   *  `vfx_block_broken` (`sink.ts`'s `anchor: "badge"`), NOT the creature. */
  x: number;
  y: number;
  startedAt: number;
}

/** A started `sprite_shader` effect (`vfx_power_flash_vfx`, Task 13) — one
 *  additive sprite over a power's icon, animated by `effect.tween`
 *  (`sampleSpriteShaderTween`, a pure function tested on its own below). */
interface LiveSpriteShader {
  kind: "sprite_shader";
  effect: SpriteShaderEffect;
  x: number;
  y: number;
  /** The flashing power's own icon URL (`PlayedEffect.iconUrl` — see that
   *  field's doc for why the manifest itself carries no texture for this
   *  kind). `null` when the power had no icon on record: nothing draws for
   *  an instance with no texture, same as every other "still loading, keep
   *  waiting" texture path in this file — except this one never resolves,
   *  since a `null` icon URL has nothing to eventually load. */
  iconUrl: string | null;
  startedAt: number;
}

type Live = LiveFlipbook | LiveSpine | LiveParticles | LiveTransformAnim | LiveSpriteShader;

/**
 * Wall-clock lifetime cap for a `LiveSpine` instance, independent of
 * `TrackEntry.isComplete()` or whether its rig ever finished loading.
 *
 * `AssetManagerBase.error()` (the vendored spine-core loader) records a
 * failed fetch only in its own `errors` map — never in `cache.assets` — so a
 * transient failure (a real network error, not the by-design "never asked
 * for" case `vfx_adrenaline` sits in) leaves `getSkeletonData` returning
 * `null` FOREVER for that rig: nothing here ever retries. Without this cap,
 * the draw loop's "still loading, keep waiting" branch would keep every
 * `LiveSpine` pushed for that rig in `liveRef.current` for the life of the
 * `manifest` reference — an unbounded leak, and a direct violation of the
 * "freed when the clip ends" contract this renderer exists to satisfy.
 *
 * `getSkeletonData` failing to resolve is ALSO checked directly (via
 * `assets.getErrors()`) so a load failure drops its instances promptly
 * rather than waiting out the full cap — but this constant is the
 * load-bearing half: it holds even for a failure mode neither check
 * anticipates. Generous relative to every real clip here (well under a
 * second once playing) and to a legitimately slow-but-successful load, so it
 * never cuts off a real instance — it only bounds a genuinely stuck one.
 */
export const SPINE_MAX_LIFETIME_MS = 8000;

/** Pure wall-clock check backing the cap above — no DOM/canvas/GL, so it's
 *  directly unit-testable under jsdom without reproducing a real network
 *  failure (see `spineVfxSelection.test.ts`). */
export function spineExpired(l: LiveSpine, now: number): boolean {
  return now - l.startedAt > SPINE_MAX_LIFETIME_MS;
}

/**
 * Which overlay draw path a manifest entry routes to, as a fresh `Live`
 * record ready to push. Pure: touches no DOM/canvas/GL (the particle branch
 * only calls `startComposite`, itself pure — see its own doc), which is what
 * makes the ROUTING decision unit-testable under jsdom (no WebGL there) —
 * the same seam `anchorRect` above already established for the anchor half
 * of `sink.play`. See `spineVfxSelection.test.ts` / `composite.test.ts`.
 *
 * `tint` is the binding table's tint (`PlayedEffect.tint`) — consumed only by
 * the particle branch (design spec §1.4/§5), `null` by default so every
 * existing flipbook/Spine call site (and every pre-Task-15 test) keeps
 * working unchanged. `iconUrl` (`PlayedEffect.iconUrl`) is consumed only by
 * the `sprite_shader` branch (Task 13's `vfx_power_flash_vfx`), `null` by
 * default for the same reason.
 */
export function buildLiveEffect(
  entry: VfxEffect,
  x: number,
  y: number,
  manifestRev: string,
  tint: string | null = null,
  iconUrl: string | null = null,
  dissolve: { emissionRadius: number; lifetime: number } | null = null,
): Live | null {
  if (entry.kind === "particles") {
    // Math.random(), not a seeded stream: this is the ONE instancing draw
    // that picks which random composite a live event gets (mirroring how
    // LiveFlipbook's randomizeRotation already draws below) — the
    // determinism contract ("no Math.random() in the simulator") is about
    // `simulate.ts`'s own particle physics, not the top-level UI choosing a
    // fresh seed per play.
    const seed = Math.floor(Math.random() * 0xffffffff);
    return {
      kind: "particles",
      composite: startComposite(entry, makeRng(seed), tint),
      sims: entry.emitters.map(() => null),
      x,
      y,
      startedAt: performance.now(),
    };
  }
  if (entry.kind === "flipbook") {
    const [lo, hi] = entry.rotationRange;
    const angle = entry.randomizeRotation ? lo + Math.random() * (hi - lo) : 0;
    return { kind: "flipbook", effect: entry, texturesRev: manifestRev, x, y, angleDeg: angle, startedAt: performance.now() };
  }
  if (entry.kind === "spine") {
    // `skeleton`/`state`/`track` are built lazily by the draw loop once the
    // rig's assets finish loading — see the module doc comment.
    return { kind: "spine", effect: entry, x, y, skeleton: null, state: null, track: null, startedAt: performance.now() };
  }
  if (entry.kind === "transform_anim") {
    return { kind: "transform_anim", effect: entry, x, y, startedAt: performance.now() };
  }
  if (entry.kind === "sprite_shader") {
    // `iconUrl` rides in on the play call, not the manifest entry — see
    // `LiveSpriteShader`'s doc for why the manifest carries no texture here.
    return { kind: "sprite_shader", effect: entry, x, y, iconUrl, startedAt: performance.now() };
  }
  if (entry.kind === "dissolve") {
    // The dying creature's DEBRIS BURST. The dissolve SPRITE is not drawn
    // here at all — it lives per-creature inside `SpineSprite`, which owns
    // the FBO the rig renders into (`effects/dissolve.ts`'s module doc says
    // why it cannot be shared). This branch is the other half of
    // `vfx_monster_death`, and it is an ordinary particle composite.
    //
    // `dissolve` is required, not optional-with-a-default: the manifest's own
    // `emissionSphereRadius` is deliberately blanked to 0 (`extract_vfx.py`),
    // because the game overwrites it per creature. No `dissolve` means no
    // measurable creature, and playing nothing is the honest outcome — a
    // burst at the blanked value would be every particle on one point.
    if (!dissolve) return null;
    return {
      kind: "particles",
      composite: startComposite(dissolveBurstEffect(entry, dissolve), makeRng(Math.floor(Math.random() * 0xffffffff)), tint),
      sims: entry.emitters.map(() => null),
      x,
      y,
      startedAt: performance.now(),
    };
  }
  // Returning `null` — which the sink already handles by not pushing
  // anything — plays nothing, which is the honest outcome for a kind with no
  // draw path. This used to be an unconditional `return { kind: "spine", ... }`
  // fall-through, which typed a `transform_anim` entry AS a Spine effect and
  // then read `entry.rig`/`entry.clip` off it at draw time.
  //
  // The annotation is the guard, and the reason this is not simply a bare
  // `return null`: a kind added to `VfxEffect` without a draw path fails to
  // COMPILE here rather than silently joining the do-nothing set. `never`
  // because every kind now HAS a branch above — add one to `VfxEffect` and
  // this stops compiling.
  const unrendered: never = entry;
  void unrendered;
  return null;
}

/**
 * `vfx_monster_death`'s debris burst as a `ParticlesEffect` the ordinary
 * composite path can start, with the two things the C# does to it that the
 * manifest cannot bake in.
 *
 * 1. **The radius**, and with it the composite's **lifetime** — both
 *    per-creature, both from `live` (see `PlayedEffect.dissolve`).
 *    `NMonsterDeathVfx.cs:231` assigns
 *    `EmissionSphereRadius = node.Size.Y / 4f` per creature, so the scene's
 *    authored 99.23 never reaches the screen and `extract_vfx.py` blanks it
 *    to 0.
 *
 * 2. **The parent's `modulate`.** The burst's node is `parent="Visual"` in
 *    the scene, and `Visual` carries `modulate = Color(1, 1, 1, 0.466667)`.
 *    Godot's `modulate` descends to children (`self_modulate` is the one that
 *    does not), so the game draws the debris at that alpha too. Folded into
 *    the emitter's `selfModulate` HERE, at play time, rather than in the
 *    extractor: `selfModulate` is the scene node's own verbatim value, and
 *    baking a parent's alpha into it would make the field a lie for every
 *    other reader (`DissolveEffect.modulateAlpha`'s doc makes the same point
 *    from the other side).
 *
 * The third item rides along for free: routing through the particle path
 * draws the burst at that path's alpha-mix blend rather than the dissolve
 * sprite's `blend_premul_alpha`, which is what the game does — a
 * `CanvasItem`'s `material` does NOT descend to children.
 *
 * `mirrorX`/`pivotScale` are null and `tinted` false because the death scene
 * has neither — those are `NVfxParticleSystem` authoring concepts and this
 * is a bespoke `NMonsterDeathVfx`.
 */
export function dissolveBurstEffect(
  entry: DissolveEffect,
  live: { emissionRadius: number; lifetime: number },
): ParticlesEffect {
  return {
    kind: "particles",
    // The composite's life is the DISSOLVE TWEEN's, not the debris'. The C#
    // frees the whole death node once that tween finishes
    // (`NMonsterDeathVfx.cs:240-241`), and the tween is per-creature — so a
    // small monster's burst really is cut off before its 1.8 s particles
    // would have expired. `entry.lifetime` is the 2.5 s CEILING; the caller
    // narrows it, the same way `Board`'s `dissolveEffectFor` narrows
    // `dissolveSeconds` for the sprite half.
    lifetime: live.lifetime,
    emitters: entry.emitters.map((e) => ({
      ...e,
      selfModulate: [
        e.selfModulate[0],
        e.selfModulate[1],
        e.selfModulate[2],
        e.selfModulate[3] * entry.modulateAlpha,
      ] as [number, number, number, number],
      process: { ...e.process, emissionSphereRadius: live.emissionRadius },
    })),
    mirrorX: null,
    pivotScale: null,
    tinted: false,
  };
}

/** `TransitionType`s reported by `evaluateEase` already, so a tween that
 *  names one this port has never seen logs once rather than every frame. */
const reportedUnknownTrans = new Set<number>();

/**
 * One `SpriteShaderTweener`'s eased progress at local fraction `t` (clamped
 * to [0, 1], not wrapped — a tween runs once, it does not loop).
 *
 * Godot's `Tween` defaults to `TRANS_LINEAR` / `EASE_IN_OUT` when a tweener
 * calls neither `SetEase` nor `SetTrans` (`ease`/`trans` both `null` in the
 * manifest) — confirmed against Godot's own `Tween` class reference ("Before
 * this method is called, the default transition type is TRANS_LINEAR" / "...
 * the default ease type is EASE_IN_OUT"), not assumed. A LINEAR transition's
 * curve is `t` regardless of ease direction, so `null`/`TRANS_LINEAR` (0)
 * never needs the ease branch at all — which is also every `scale` tweener in
 * the committed `power_flash_vfx` data (`ease`/`trans` both `null`).
 *
 * `TRANS_SINE` (1) is the only other transition the committed tween data
 * uses (`modulateAlpha`'s two tweeners, `ease` 1/EASE_OUT and 0/EASE_IN), so
 * that is the only one this port implements — Godot's own Penner-style sine
 * curves (`math_funcs.cpp`). A `trans` this has never seen reports once and
 * falls back to linear rather than silently guessing a curve shape.
 */
function evaluateEase(t: number, trans: number | null, ease: number | null): number {
  const clamped = t < 0 ? 0 : t > 1 ? 1 : t;
  if (trans === null || trans === 0) return clamped; // TRANS_LINEAR (or the unset default)
  if (trans === 1) {
    // TRANS_SINE.
    if (ease === 0) return 1 - Math.cos((clamped * Math.PI) / 2); // EASE_IN
    if (ease === 1) return Math.sin((clamped * Math.PI) / 2); // EASE_OUT
    return -(Math.cos(Math.PI * clamped) - 1) / 2; // EASE_IN_OUT (ease null/2/unrecognized)
  }
  if (!reportedUnknownTrans.has(trans)) {
    reportedUnknownTrans.add(trans);
    console.error(`power flash tween: unrecognized TransitionType ${trans}; using linear`);
  }
  return clamped;
}

/**
 * `vfx_power_flash_vfx`'s scale/alpha at `elapsed` seconds into its life —
 * `NPowerFlashVfx.cs:97-102`'s tween, transcribed as data by Task 10
 * (`SpriteShaderEffect.tween`) and evaluated here. Pure: no DOM/canvas/GL, so
 * it is unit-testable without a WebGL context, the same seam every other
 * pure piece of this file already gets.
 *
 * Steps run in SEQUENCE; the tweeners inside one step run in PARALLEL, each
 * over its OWN duration — `modulateAlpha`'s first tweener (0.25s) finishes
 * and holds while `scale`'s sibling in the same step (0.4s) keeps animating,
 * and the NEXT step does not start until the step's longest tweener does
 * (`SpriteShaderTweener`'s own doc: "outer array = STEPS run in sequence;
 * inner = tweeners run in parallel within their step"). `remaining` tracks
 * how far past each step's start `elapsed` is — negative once a step has not
 * started yet, `evaluateEase`'s clamp then holds that property at its
 * pre-step value, matching a `Tween` that has not reached that leg.
 */
export function sampleSpriteShaderTween(
  effect: SpriteShaderEffect,
  elapsed: number,
): { scale: number; modulateAlpha: number } {
  let scale = effect.scale;
  let modulateAlpha = effect.modulateAlpha;
  let remaining = elapsed;
  for (const step of effect.tween) {
    const fromScale = scale;
    const fromAlpha = modulateAlpha;
    for (const tw of step) {
      const t = tw.duration > 0 ? remaining / tw.duration : remaining >= 0 ? 1 : 0;
      const eased = evaluateEase(t, tw.trans, tw.ease);
      if (tw.property === "scale") scale = fromScale + (tw.to - fromScale) * eased;
      else if (tw.property === "modulateAlpha") modulateAlpha = fromAlpha + (tw.to - fromAlpha) * eased;
    }
    const stepDuration = step.reduce((m, tw) => Math.max(m, tw.duration), 0);
    remaining -= stepDuration;
  }
  return { scale, modulateAlpha };
}

/**
 * The combat VFX overlay: one arena-sized canvas above the combatants.
 *
 * Reuses `spine.SceneRenderer` (the app already runs one per combatant inside
 * `SpineSprite`) both as a textured-quad batcher for flipbook frames AND to
 * `drawSkeleton` live Spine rigs (`kind: "spine"`) — one `Skeleton` +
 * `AnimationState` per in-flight instance, built lazily once its rig assets
 * finish loading and dropped once its one-shot clip completes
 * (`TrackEntry.isComplete()`). That costs no new dependency and no
 * hand-rolled batcher. Degrades to a no-op wherever WebGL is unavailable —
 * notably jsdom, which is why only selection/lifetime, never pixels, is
 * unit-tested here (see `spineVfxSelection.test.ts`).
 *
 * `kind: "particles"` composites (Task 15) draw through a THIRD path: their
 * own tiny WebGL program (`effects/particles/shaders.ts`), not
 * `spine.SceneRenderer`'s batcher — a custom per-material shader swap isn't
 * something that batcher exposes. Each frame, every live composite's
 * emitters are ADVANCED (`advanceTo`) — one `SimState` per emitter, held in
 * `LiveParticles.sims`, stepped forward to the current clock. It used to
 * re-run `simulate(spec, elapsed, ...)` from t=0 every frame, which is
 * O(frames²) and measured at 96% of a core for a single composite; advancing
 * is byte-identical to that replay (`simulate.ts`'s increments-equal-a-jump
 * test) and additionally makes the work independent of the display refresh
 * rate. Particles are drawn as textured, tinted
 * quads AFTER `renderer.end()` flushes the flipbook/Spine batch, grouped by
 * texture (design spec §4.2). See `shaders.ts`'s module doc for the shader
 * fidelity gap this draw path carries.
 */
export default function VfxLayer({
  arenaRef,
  manifest,
  onSink,
  sceneScale = 1,
}: {
  /** The canvas's own container, `.gs-actors` — the box the combatants are
   *  drawn in, which is what positions have to be measured against. */
  arenaRef: RefObject<HTMLElement | null>;
  manifest: VfxManifest | null;
  onSink: (sink: EffectSink) => void;
  /** Stage px per scene px, camera and aspect fit included — `Board`'s
   *  `actorScale`, i.e. exactly what every combatant is drawn at. Effects
   *  scale WITH the scene because the game parents them into the creature's
   *  own container (`VfxCmd.PlayOnCreature*`); sized in constant screen px
   *  they grew relative to their target as the window shrank. */
  sceneScale?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const liveRef = useRef<Live[]>([]);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  // Read by the draw loop, never a dependency of it: the renderer effect below
  // depends ONLY on the manifest, so a resize must not be able to re-fetch a
  // texture (the "bosses shift for a second" bug).
  const sceneScaleRef = useRef(sceneScale);
  sceneScaleRef.current = sceneScale;

  // Size, tracked on its own. Kept separate from the loading effect below so a
  // resize can never re-fetch a texture — the "bosses shift for a second" bug
  // was exactly a live measurement wired into an asset-loading dependency.
  useEffect(() => {
    const el = arenaRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      setSize((p) => (p.w === r.width && p.h === r.height ? p : { w: r.width, h: r.height }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [arenaRef]);

  // Renderer + textures. Depends ONLY on the manifest — never on `size` — so
  // an arena resize resizes the canvas without re-fetching a single texture.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !manifest) return;
    let gl: WebGLRenderingContext | null = null;
    try {
      gl = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: false });
    } catch {
      gl = null;
    }
    // No GL (jsdom, a blocked context, an old browser): the overlay is a
    // silent no-op and the rest of the board keeps working on its existing
    // CSS effects (damage numbers, shake, death fade).
    if (!gl) return;

    const renderer = new spine.SceneRenderer(canvas, gl);
    const assets = new spine.AssetManager(gl);
    const particleProgram: ParticleProgram = createParticleProgram(gl);
    const powerFlashProgram: PowerFlashProgram = createPowerFlashProgram(gl);
    // Scalar (`Curve`) and RGBA (`colorRamp`, a `Gradient`) baked LUTs,
    // fetched as plain JSON — outside `spine.AssetManager`, which only knows
    // textures/atlases/binaries. Populated eagerly below, same reasoning as
    // flipbook frames: a handful of small files, all eligible from the first
    // hit of a fight. `undefined` = not requested yet / still in flight;
    // `get`/`has` in the draw loop treat that as "still loading", the same
    // "keep waiting" contract every other asset in this file already has.
    const scalarLuts = new Map<string, readonly number[]>();
    // Colourisation LUTs uploaded to GL, keyed by manifest path. Built lazily
    // on first use rather than at load time: `createLutTexture` needs the
    // fetched JSON, which arrives asynchronously.
    const lutTextures = new Map<string, LutTexture>();
    function lutTextureFor(path: string): LutTexture | null {
      const existing = lutTextures.get(path);
      if (existing) return existing;
      const baked = colorLuts.get(path);
      if (!baked) return null;
      const created = createLutTexture(gl!, baked);
      lutTextures.set(path, created);
      return created;
    }
    const colorLuts = new Map<string, readonly (readonly [number, number, number, number])[]>();
    const lutFetchStarted = new Set<string>();
    function ensureLutLoading(path: string, isColor: boolean): void {
      if (lutFetchStarted.has(path)) return;
      lutFetchStarted.add(path);
      fetch(effectAssetUrl(path, manifest!.rev))
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`vfx lut ${r.status} ${path}`))))
        .then((json) => {
          if (isColor) colorLuts.set(path, json as (readonly [number, number, number, number])[]);
          else scalarLuts.set(path, json as number[]);
        })
        .catch((err: unknown) => console.error(`vfx particle lut fetch failed: ${path}`, err));
    }
    // Flipbook frames are cheap (a handful of small PNGs per effect) and
    // every one is eligible to play from the first hit of a fight, so they
    // are all preloaded eagerly, same as before this task. Particle textures
    // and their emitters' curve/gradient LUTs get the same eager treatment —
    // they're reused heavily across composites (design spec §2.1: the same
    // leaf scenes appear in up to six of the 13) and are individually small.
    for (const e of Object.values(manifest.effects)) {
      if (e.kind === "flipbook") for (const f of e.frames) assets.loadTexture(effectAssetUrl(f, manifest.rev));
      // `dissolve` alongside `particles`: `vfx_monster_death`'s debris burst
      // is an ordinary particle emitter and reaches the screen through the
      // same draw path (`buildLiveEffect`'s `dissolve` branch), so it needs
      // the same texture/LUT preloading. Omitting it left the burst's texture
      // unrequested by anything, and the draw path SKIPS a frame whose
      // texture is missing rather than fetching it — so the burst would have
      // drawn nothing, silently, forever.
      if (e.kind === "particles" || e.kind === "dissolve") {
        for (const em of e.emitters) {
          assets.loadTexture(effectAssetUrl(em.texture, manifest.rev));
          for (const path of scalarCurvePathsOf(em)) ensureLutLoading(path, false);
          if (em.process.colorRamp) ensureLutLoading(em.process.colorRamp, true);
          // The material's own colourisation ramp and flipbook curve — the
          // two things that decide what COLOUR a particle is and which frame
          // of its sheet it shows.
          if (em.material?.lut) ensureLutLoading(em.material.lut, true);
          if (em.material?.flipbookCurve) ensureLutLoading(em.material.flipbookCurve, false);
        }
      }
      // `transform_anim` sprites are as cheap as a flipbook frame — one PNG
      // per sprite (two, for `vfx_block_broken`) — so they get the same eager
      // treatment, textures AND the baked position/rotation tracks alike
      // (`scalarLuts` is a generic path -> number[] cache; it does not care
      // which consumer a path belongs to).
      if (e.kind === "transform_anim") {
        for (const s of e.sprites) {
          assets.loadTexture(effectAssetUrl(s.texture, manifest.rev));
          if (s.positionX) ensureLutLoading(s.positionX, false);
          if (s.positionY) ensureLutLoading(s.positionY, false);
          if (s.rotation) ensureLutLoading(s.rotation, false);
        }
      }
      // A spine effect's RIG loads lazily (see `ensureSpineRigLoading` below),
      // but its slot-parented emitters' textures and LUTs do not: they are a
      // glow dot and a handful of baked curves, the same size class as the
      // composite textures above, and the draw path SKIPS a frame whose
      // texture is missing rather than fetching it — the exact silent-nothing
      // failure the `dissolve` note above records (GH #669).
      if (e.kind === "spine") {
        for (const em of e.emitters ?? []) {
          assets.loadTexture(effectAssetUrl(em.texture, manifest.rev));
          for (const path of scalarCurvePathsOf(em)) ensureLutLoading(path, false);
          if (em.process.colorRamp) ensureLutLoading(em.process.colorRamp, true);
        }
      }
      // `kind === "spine"` otherwise loads lazily — see `ensureSpineRigLoading`.
      // `kind === "sprite_shader"` (`vfx_power_flash_vfx`) has no texture to
      // preload from the manifest at all — its texture is a POWER's icon,
      // resolved per-instance at play time (`LiveSpriteShader.iconUrl`) — see
      // `ensureIconLoading` below.
    }

    // Spine rigs are NOT preloaded eagerly: a rig's `.skel`/`.atlas`/pages are
    // heavier than a flipbook's frames, and — unlike flipbook — fetching only
    // the rig an actual selection names means nothing ever asks for
    // `vfx_adrenaline`'s assets at all (full-screen, sub-project 5, still
    // unrendered here) — which is exactly right: it stays unreachable, not
    // merely unselected. Started on first request (`ensureSpineRigLoading`,
    // called from `getSkeletonData`) and tracked here so a second `vfx_bite`
    // hit doesn't re-issue the fetch.
    const spineLoadStarted = new Set<string>();
    function ensureSpineRigLoading(entry: SpineEffect): void {
      if (spineLoadStarted.has(entry.rig)) return;
      spineLoadStarted.add(entry.rig);
      // `entry.rig` is the extension-less base path; the page-image name
      // inside the `.atlas` text is `<basename>.png` for every rig this
      // renderer actually loads (verified against every committed VFX rig
      // this table can select — `spineVfxSelection.test.ts` gates that fact:
      // `AssetManager.loadTextureAtlas` resolves `fileAlias[page.name]` per
      // DECLARED page, so a rig with a second page not covered here — like
      // `vfx_adrenaline`, which is why it must never reach this function —
      // would resolve that page's fetch to `undefined` and throw), which is
      // what lets `fileAlias` give it the SAME `?v=` cache-bust as the
      // `.atlas`/`.skel` fetches below — the art bucket serves everything
      // `immutable` at non-content-addressed URLs, so a bare page path would
      // keep serving a stale PNG after a redeploy even once the
      // atlas/skeleton themselves were busted.
      const base = entry.rig.slice(entry.rig.lastIndexOf("/") + 1);
      assets.loadBinary(effectAssetUrl(`${entry.rig}.skel`, manifest!.rev));
      assets.loadTextureAtlas(
        effectAssetUrl(`${entry.rig}.atlas`, manifest!.rev),
        undefined,
        undefined,
        { [`${base}.png`]: effectAssetUrl(`${entry.rig}.png`, manifest!.rev) },
      );
    }

    // True once `AssetManagerBase` has recorded a failed fetch for this
    // rig's `.skel` or `.atlas` — it keys `errors` by the exact path passed
    // to `loadBinary`/`loadTextureAtlas` (including a failed ATLAS PAGE:
    // the vendored loader attributes that failure to the atlas's own path,
    // not the page's), so these are the same URLs `ensureSpineRigLoading`
    // requested. A failure here never resolves on its own — `error()` writes
    // only to `errors`, never to `cache.assets` — so `getSkeletonData` would
    // otherwise return `null` forever; this lets the draw loop drop the
    // instance immediately instead of only via `spineExpired`'s slower cap.
    function rigLoadFailed(entry: SpineEffect): boolean {
      const errors = assets.getErrors();
      return (
        effectAssetUrl(`${entry.rig}.skel`, manifest!.rev) in errors ||
        effectAssetUrl(`${entry.rig}.atlas`, manifest!.rev) in errors
      );
    }

    // One parsed `SkeletonData` per rig, built lazily once both its `.skel`
    // and `.atlas` finish loading (a fight's first hits race that load, same
    // as a flipbook frame still in flight) and cached for the life of this
    // effect — every `vfx_bite` hit in a fight shares one parse, not one per
    // instance. Keyed by `rig` (the manifest key IS the rig base path here,
    // not a display name).
    const skeletonDataCache = new Map<string, spine.SkeletonData>();
    function getSkeletonData(entry: SpineEffect): spine.SkeletonData | null {
      const cached = skeletonDataCache.get(entry.rig);
      if (cached) return cached;
      ensureSpineRigLoading(entry);
      const skelBin = assets.get(effectAssetUrl(`${entry.rig}.skel`, manifest!.rev)) as
        | Uint8Array
        | undefined;
      const atlas = assets.get(effectAssetUrl(`${entry.rig}.atlas`, manifest!.rev)) as
        | spine.TextureAtlas
        | undefined;
      if (!skelBin || !atlas) return null;
      const data = new spine.SkeletonBinary(new spine.AtlasAttachmentLoader(atlas)).readSkeletonData(
        skelBin,
      );
      skeletonDataCache.set(entry.rig, data);
      return data;
    }

    // `vfx_power_flash_vfx` instances each carry their own texture URL — a
    // power's icon, resolved at PLAY time (`LiveSpriteShader.iconUrl`), not a
    // manifest-relative path the eager preload loop above can know in
    // advance. `AssetManager` caches by URL string already, so a second flash
    // of the same power is a cache hit; this set only avoids re-issuing the
    // fetch call itself on every frame while the first one is still in
    // flight — the same idea as `spineLoadStarted` above.
    const iconLoadStarted = new Set<string>();
    function ensureIconLoading(url: string): void {
      if (iconLoadStarted.has(url)) return;
      iconLoadStarted.add(url);
      assets.loadTexture(url);
    }

    let raf = 0;
    // Wall-clock delta for `AnimationState.update` — flipbook timing indexes
    // frames off `startedAt` directly and needs no per-tick delta, but a
    // Spine `AnimationState` advances incrementally like the per-combatant
    // one `SpineSprite` already drives.
    let lastTime = performance.now();
    const frame = () => {
      const canvasH = canvas.height;
      // OrthoCamera is Y-up, origin bottom-left; the viewport is the whole
      // canvas so canvas pixels map 1:1 to camera units.
      renderer.camera.position.set(canvas.width / 2, canvasH / 2, 0);
      renderer.camera.setViewport(canvas.width, canvasH);
      renderer.camera.update();
      gl!.clearColor(0, 0, 0, 0);
      gl!.clear(gl!.COLOR_BUFFER_BIT);
      renderer.begin();
      const now = performance.now();
      const dt = Math.max(0, (now - lastTime) / 1000);
      lastTime = now;
      // Particle quads are gathered here (grouped by texture, design spec
      // §4.2) while the filter below walks every live instance, then drawn
      // in one pass AFTER `renderer.end()` flushes the flipbook/Spine batch
      // — see the module doc comment for why this is a separate GL pass
      // rather than routed through `spine.SceneRenderer`'s own batcher.
      const particleBatches = new Map<
        string,
        {
          tex: spine.GLTexture;
          material: BatchMaterialState;
          verts: number[];
        }
      >();
      // Every live `vfx_power_flash_vfx` instance's quad, one entry each (no
      // texture-keyed batching like `particleBatches` above — each instance
      // draws a different power's icon at a different tween-animated colour,
      // so there is nothing two instances would ever share a draw call for).
      // Drawn after `renderer.end()`, same reasoning as the particle pass.
      const powerFlashDraws: {
        tex: spine.GLTexture;
        strength: number;
        color: [number, number, number, number];
        verts: number[];
      }[] = [];
      liveRef.current = liveRef.current.filter((l) => {
        if (l.kind === "particles") {
          const elapsed = (now - l.startedAt) / 1000;
          if (isDone(l.composite, elapsed)) return false;
          for (let ei = 0; ei < l.composite.emitters.length; ei++) {
            const le = l.composite.emitters[ei];
            if (elapsed < le.startedAt) continue; // this emitter hasn't started yet (staggered composite)
            const emitterElapsed = elapsed - le.startedAt;
            const texUrl = effectAssetUrl(le.emitter.texture, manifest!.rev);
            const tex = assets.get(texUrl) as spine.GLTexture | undefined;
            if (!tex) continue; // texture still loading — keep waiting, same as flipbook/Spine
            const spec = buildParticleSystemSpec(le.emitter, scalarLuts);
            if (!spec) continue; // a curve LUT this emitter needs hasn't loaded yet
            const colorRampPath = le.emitter.process.colorRamp;
            if (colorRampPath && !colorLuts.has(colorRampPath)) continue; // still loading
            const colorRampLut = colorRampPath ? (colorLuts.get(colorRampPath) ?? null) : null;
            const alphaCurveLut = le.emitter.process.alphaCurve ? (scalarLuts.get(le.emitter.process.alphaCurve) ?? null) : null;

            // Advance this emitter's own simulation to the current clock
            // rather than replaying it from t=0 (which cost O(frames^2) and
            // put one composite at 96% of a core). `advanceTo` steps whole
            // fixed-timestep frames, so the result is identical to the replay
            // AND independent of the display refresh rate — see
            // `simulate.ts`'s `advanceTo` doc and its increments-equal-a-jump
            // test.
            let sim = l.sims[ei];
            if (!sim) {
              sim = createSimState(spec, makeRng(le.seed));
              l.sims[ei] = sim;
            }
            advanceTo(sim, emitterElapsed);
            const result = readout(sim);
            const img = tex.getImage();
            const k = sceneScaleRef.current;
            // Renderer-level support for the two bespoke, named-node
            // transforms the extractor records but cannot fully compose
            // (Task 3's carry-forward note: an EmitterSpec only exposes the
            // composed TRANSLATION, never an intermediate container's own
            // rotation/scale). WHICH emitters each applies to is resolved at
            // EXTRACTION time, from the real scene graph
            // (`underPivotScale`/`underMirrorX`). `vfx_poison_impact` is the
            // only composite either applies to today.
            const pivotScale = l.composite.effect.pivotScale;
            const sx = le.emitter.underPivotScale && pivotScale ? pivotScale[1] : 1;
            const sy = le.emitter.underPivotScale && pivotScale ? pivotScale[2] : 1;
            const mirrorSign = l.composite.mirrored && le.emitter.underMirrorX ? -1 : 1;

            const material = le.emitter.material;
            const materialLut = material?.lut ? lutTextureFor(material.lut) : null;
            const flipbookCurveLut = material?.flipbookCurve
              ? (scalarLuts.get(material.flipbookCurve) ?? null)
              : null;
            const batchMaterial = {
              lut: materialLut,
              mask: maskChannelOf(material),
              plain: material === null,
            };

            for (const particle of result.particles) {
              if (!particle.alive) continue;
              // Godot's `INSTANCE_CUSTOM.y / INSTANCE_CUSTOM.w`, which the
              // simulator now reports directly. Deriving it here as
              // `age / spec.lifetime` ignored each particle's own randomized
              // lifetime and ended its alpha curve early, popping it out at
              // partial opacity — 44 of 64 emitters author that randomness.
              const lifetimeFraction = particle.lifetimeFraction;
              const color = particleVertexColor(le.emitter, l.composite.tint, colorRampLut, alphaCurveLut, lifetimeFraction);
              const cell = material
                ? flipbookCell(material, lifetimeFraction, particle.animOffset, flipbookCurveLut)
                : undefined;
              // Godot 2D is Y-down; the ortho camera here is Y-up (see
              // `anchorRect`'s doc), so particle Y is negated on the way in.
              const cx = l.x + particle.x * k * sx * mirrorSign;
              const cy = l.y - particle.y * k * sy;
              // `mirrorSign` belongs on the WIDTH as well as the position:
              // `NPoisonImpactVfx.cs:101` sets `Scale = (-1, 1)` on a node
              // that PARENTS the emitters, and a negative parent scale flips
              // what its children draw, not only where they sit. A negative
              // width swaps this quad's x corners while its UVs stay put,
              // which is that flip (`appendQuadVertices`, pinned by test).
              const w = img.width * particle.scale * k * sx * mirrorSign;
              const h = img.height * particle.scale * k * sy;
              const batchKey = `${texUrl}::${material?.path ?? "plain"}`;
              let batch = particleBatches.get(batchKey);
              if (!batch) {
                batch = { tex, material: batchMaterial, verts: [] };
                particleBatches.set(batchKey, batch);
              }
              // ...and on the ANGLE too: a parent mirror applied after the
              // particle's own rotation is `M·R(a) = R(-a)·M`, so the negated
              // width alone mirrors the quad in its LOCAL frame and leaves a
              // rotated (or velocity-aligned) quad pointing the wrong way.
              const angle = particleDrawAngle(le.emitter.process, particle) * mirrorSign;
              appendQuadVertices(batch.verts, { cx, cy, w, h, color, cell, angle }, canvas.width, canvas.height);
            }
          }
          return true;
        }
        if (l.kind === "flipbook") {
          const idx = Math.floor(((now - l.startedAt) / 1000) * l.effect.fps);
          if (idx >= l.effect.frames.length) return false;
          const url = effectAssetUrl(l.effect.frames[idx]!, l.texturesRev);
          // `get`, not `require`: a texture still in flight must skip this
          // frame, not throw and kill the render loop — the first frames of a
          // fight race the load.
          const tex = assets.get(url) as spine.GLTexture | undefined;
          if (!tex) return true;
          const img = tex.getImage();
          // The manifest's `scale` is the game's authored one, in SCENE px; the
          // scene factor turns that into the stage px the fighters are drawn at.
          const k = l.effect.scale * sceneScaleRef.current;
          const w = img.width * k;
          const h = img.height * k;
          renderer.drawTextureRotated(
            tex,
            l.x - w / 2,
            l.y - h / 2,
            w,
            h,
            w / 2,
            h / 2,
            drawAngleFromGodotDegrees(l.angleDeg),
          );
          return true;
        }
        if (l.kind === "transform_anim") {
          const elapsed = (now - l.startedAt) / 1000;
          const t = l.effect.lifetime > 0 ? elapsed / l.effect.lifetime : 1;
          if (t > 1) return false;
          const k = sceneScaleRef.current;
          for (const s of l.effect.sprites) {
            const url = effectAssetUrl(s.texture, manifest!.rev);
            const tex = assets.get(url) as spine.GLTexture | undefined;
            if (!tex) continue; // texture still loading — keep waiting, same as a flipbook frame in flight
            const img = tex.getImage();
            // A track LUT still in flight reads as "no track yet" for THIS
            // frame only, and both that and a channel the animation genuinely
            // never touches resolve to the sprite's own authored transform
            // (`rest*` — `sampleTrack`'s doc: "hold the sprite's authored
            // value, do not default it to zero"; the manifest carries it
            // precisely so this caller has it). Unlike a missing texture,
            // neither blocks the draw: an untouched channel is a legitimate
            // steady state, and an unresolved LUT looks identical to it for
            // one frame, which is harmless at 0.4s/~24 frames of life.
            const offX = sampleTrack(s.positionX ? (scalarLuts.get(s.positionX) ?? null) : null, t, s.restPositionX);
            const offY = sampleTrack(s.positionY ? (scalarLuts.get(s.positionY) ?? null) : null, t, s.restPositionY);
            // An `Animation` rotation track is authored in RADIANS; the unit
            // and sign this needs to become a draw angle are both
            // `drawAngleFromGodotRadians`'s to explain.
            const rotDeg = drawAngleFromGodotRadians(
              sampleTrack(s.rotation ? (scalarLuts.get(s.rotation) ?? null) : null, t, s.restRotation),
            );
            const w = img.width * k;
            const h = img.height * k;
            const cx = l.x + offX * k;
            const cy = l.y - offY * k;
            renderer.drawTextureRotated(tex, cx - w / 2, cy - h / 2, w, h, w / 2, h / 2, rotDeg);
          }
          return true;
        }
        if (l.kind === "sprite_shader") {
          const elapsed = (now - l.startedAt) / 1000;
          if (elapsed > l.effect.lifetime) return false;
          if (l.iconUrl === null) return true; // no icon on record for this power — nothing to draw, ever
          ensureIconLoading(l.iconUrl);
          const tex = assets.get(l.iconUrl) as spine.GLTexture | undefined;
          if (!tex) return true; // icon still loading — keep waiting
          const img = tex.getImage();
          const { scale, modulateAlpha } = sampleSpriteShaderTween(l.effect, elapsed);
          const k = scale * sceneScaleRef.current;
          const verts: number[] = [];
          appendPowerFlashQuadVertices(
            verts,
            { cx: l.x, cy: l.y, w: img.width * k, h: img.height * k },
            canvas.width,
            canvas.height,
          );
          powerFlashDraws.push({ tex, strength: l.effect.strength, color: [1, 1, 1, modulateAlpha], verts });
          return true;
        }

        // kind === "spine". Checked FIRST, unconditionally, every frame, for
        // every spine instance — see `spineExpired`'s doc comment for why
        // this must never be reachable-but-skipped by any branch below it.
        if (spineExpired(l, now)) return false;
        if (!l.skeleton || !l.state) {
          // A failed fetch never resolves on its own (see `rigLoadFailed`) —
          // drop promptly rather than idling out the wall-clock cap above.
          if (rigLoadFailed(l.effect)) return false;
          const data = getSkeletonData(l.effect);
          if (!data) return true; // rig still loading — keep waiting, same as a flipbook texture in flight
          const clip = l.effect.clip;
          // No authored clip (or one this rig doesn't actually have): there is
          // nothing to play and no completion event will ever fire, so drop
          // it immediately rather than leaking a permanently-idle instance.
          if (!clip || !data.findAnimation(clip)) return false;
          const skeleton = new spine.Skeleton(data);
          const state = new spine.AnimationState(new spine.AnimationStateData(data));
          skeleton.setToSetupPose();
          l.skeleton = skeleton;
          l.state = state;
          // Slot-parented emitters start on one of this rig's OWN animation
          // events, not when the effect does (GH #669) —
          // `NKaiserCrabBossExplosionVfx.OnAnimationEvent` restarts the ember
          // burst on `left_embers_start`, 0.100s into a 0.67s clip. The
          // listener goes on BEFORE `setAnimation` so an event at t=0 is not
          // missed.
          const emitters = l.effect.emitters;
          if (emitters && emitters.length > 0) {
            const started = new Map<
              string,
              { startedAt: number; sim: SimState | null; seed: number }
            >();
            l.slotEmitters = started;
            state.addListener({
              event: (_entry, ev) => {
                for (const em of emitters) {
                  // `Restart()`, not "start once": a rig that raised the event
                  // twice would restart the burst, so a repeat REPLACES the
                  // run rather than being ignored.
                  if (ev.data.name === em.event) {
                    started.set(em.name, {
                      startedAt: performance.now(),
                      sim: null,
                      seed: (l.startedAt * 1000) | 0,
                    });
                  }
                }
              },
            });
          }
          l.track = state.setAnimation(0, clip, false);
        }
        l.state.update(dt);
        l.state.apply(l.skeleton);
        l.skeleton.x = l.x;
        l.skeleton.y = l.y;
        // Same SCENE-px -> stage-px convention as the flipbook branch above.
        const k = l.effect.scale * sceneScaleRef.current;
        l.skeleton.scaleX = k;
        l.skeleton.scaleY = k;
        l.skeleton.updateWorldTransform(spine.Physics.update);
        // Slot-parented emitters (GH #669). Appended to the same
        // `particleBatches` the composite branch fills, so they ride the
        // particle GL pass that runs after the spine batcher flushes — a
        // burst drawn through `drawSkeleton` would be batched with the rig's
        // own quads and take its blend mode.
        //
        // Position comes from the SLOT's bone every frame rather than from a
        // fixed offset: that is what parenting to a `SpineSlotNode` means, and
        // a rig whose slot moves during the burst has to be followed. The
        // skeleton is already placed and scaled below, so `bone.worldX/Y` is in
        // canvas space (Y-up) with no further conversion.
        //
        // `explosion_slot` itself does NOT move: its bone is `root`, at the
        // origin, unrotated and at world scale 1.0 for all 0.667 s of the clip
        // (measured off the shipped `.skel` through spine-core). That is worth
        // recording, because it is what makes `ek` below the WHOLE scale
        // factor: were the slot's bone scaled, the emitter's Godot global
        // transform would carry that too and the embers would be sized by it.
        for (const em of l.effect.emitters ?? []) {
          const run = l.slotEmitters?.get(em.name);
          if (!run) continue; // its rig event hasn't fired yet
          const bone = l.skeleton.findSlot(em.slot)?.bone;
          if (!bone) continue; // a slot this rig doesn't have — nothing to anchor to
          const texUrl = effectAssetUrl(em.texture, manifest!.rev);
          const tex = assets.get(texUrl) as spine.GLTexture | undefined;
          if (!tex) continue; // texture still loading, same as everywhere else
          const spec = buildParticleSystemSpec(em, scalarLuts);
          if (!spec) continue; // a curve LUT this emitter needs hasn't loaded
          const colorRampPath = em.process.colorRamp;
          if (colorRampPath && !colorLuts.has(colorRampPath)) continue;
          const colorRampLut = colorRampPath ? (colorLuts.get(colorRampPath) ?? null) : null;
          const alphaCurveLut = em.process.alphaCurve
            ? (scalarLuts.get(em.process.alphaCurve) ?? null)
            : null;
          if (!run.sim) run.sim = createSimState(spec, makeRng(run.seed));
          advanceTo(run.sim, (now - run.startedAt) / 1000);
          const result = readout(run.sim);
          const img = tex.getImage();
          // The SAME factor the skeleton itself is drawn at (`k` below):
          // in Godot the emitter is a child of the `SpineSprite`, so its
          // authored offsets and sizes live under that node's 0.25 scale, and
          // its 2500-5000 initial velocity and 300px emission ring are large
          // precisely because of it. Converting with the scene factor alone
          // flung every ember off-screen within a frame — which reads exactly
          // like "the burst never drew".
          const ek = l.effect.scale * sceneScaleRef.current;
          const batchKey = `${texUrl}::slot:${em.slot}::${em.blendAdd ? "add" : "mix"}`;
          for (const particle of result.particles) {
            if (!particle.alive) continue;
            const color = particleVertexColor(
              em,
              // No composite tint here: `PlayedEffect.tint` is the binding
              // table's colourisation of a particle COMPOSITE, and a Spine
              // effect has none — `crabRigEventVfx` plays this with
              // `tint: null`.
              [1, 1, 1, 1],
              colorRampLut,
              alphaCurveLut,
              particle.lifetimeFraction,
            );
            const baseHeight = img.height * particle.scale;
            // A trail-enabled emitter's particle is a RIBBON along its path,
            // not a quad at a point. For these embers that IS the effect: the
            // quad alone is a 24x24 dot at 0.1-0.4 scale under this effect's
            // 0.25, measured on screen at 0.5-1.7 px, while the segment its
            // 2500-5000 initial velocity sweeps out over `trailLifetime`
            // measured 20-46 px. Untrailed they render as a few stray pixels;
            // trailed, as the radiating spark burst the game shows.
            const trail = em.trailEnabled
              ? trailQuadGeometry(particle, baseHeight, em.trailLifetime)
              : null;
            const cx = bone.worldX + (trail?.cx ?? particle.x) * ek;
            const cy = bone.worldY - (trail?.cy ?? particle.y) * ek;
            const w = img.width * particle.scale * ek;
            const h = (trail?.h ?? baseHeight) * ek;
            let batch = particleBatches.get(batchKey);
            if (!batch) {
              batch = {
                tex,
                material: { lut: null, mask: maskChannelOf(null), plain: true, additive: em.blendAdd },
                verts: [],
              };
              particleBatches.set(batchKey, batch);
            }
            appendQuadVertices(
              batch.verts,
              { cx, cy, w, h, color, angle: trail?.angle ?? particleDrawAngle(em.process, particle) },
              canvas.width,
              canvas.height,
            );
          }
        }
        renderer.drawSkeleton(l.skeleton, false); // false = straight (non-premultiplied) alpha
        return !(l.track?.isComplete() ?? false);
      });
      renderer.end();
      // Particle pass — its own program/buffers, drawn as a separate GL pass
      // after the spine batcher has flushed (see the module doc comment).
      // `spine.SceneRenderer.end()` leaves blend state enabled with straight
      // alpha (matching `drawSkeleton`'s own `false` argument above), so no
      // extra state restore is needed before `drawParticleBatch` sets its own.
      for (const batch of particleBatches.values()) {
        drawParticleBatch(gl!, particleProgram, batch.tex, batch.material, batch.verts);
      }
      // Power-flash pass — its own program (Task 13), same "separate GL pass
      // after the batcher flushed" reasoning as the particle pass above.
      for (const draw of powerFlashDraws) {
        drawPowerFlashQuad(gl!, powerFlashProgram, draw.tex, draw.strength, draw.color, draw.verts);
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      renderer.dispose();
      assets.dispose();
      liveRef.current = [];
    };
  }, [manifest]);

  // Publish the sink once the layer exists, so callers can start playing
  // effects immediately — the sink itself checks readiness per-call.
  useEffect(() => {
    const sink: EffectSink = {
      play(p: PlayedEffect) {
        const arena = arenaRef.current;
        const canvas = canvasRef.current;
        if (!arena || !canvas || !manifest) return;
        const entry = manifest.effects[p.effect];
        if (!entry) return;
        // `anchor: "badge"` (`vfx_block_broken` alone) resolves against the
        // block BADGE, a descendant of the creature, not the creature's own
        // rect — see `badgeRect`'s doc for why (and `sink.ts`'s
        // `PlayedEffect.anchor` for the C# citation). Returns `null` exactly
        // when the real game's guard would too: the badge not mounted.
        // `anchor: "point"` carries its own spawn in viewport px and never
        // looks up a creature rect — see `sink.ts`'s `PlayedEffect.point` for
        // why the Kaiser Crab's explosion cannot use one. A point play with
        // no point draws nothing, matching the `!rect` guard below.
        const arenaRect = arena.getBoundingClientRect();
        const spawn = ((): { x: number; y: number } | null => {
          if (p.anchor === "point") {
            return p.point
              ? { x: p.point.x - arenaRect.left, y: p.point.y - arenaRect.top }
              : null;
          }
          const r = p.anchor === "badge" ? badgeRect(p.target, arena) : anchorRect(p.target, arena);
          if (!r) return null;
          return {
            x: r.left - arenaRect.left + r.width / 2,
            // `anchor: "center"` is `NCreature.VfxSpawnPosition` (the rect
            // centre); `anchor: "badge"` is likewise the resolved rect's own
            // centre (the badge's `GlobalPosition + Size * 0.5`, already the
            // whole of what `badgeRect` returns); `anchor: "origin"` is the
            // creature node's own origin, which for a ground-standing rig
            // sits at its feet (rect bottom) — the ONLY anchor that reads
            // `bottom` instead of the rect's centre.
            y: p.anchor === "origin" ? r.bottom - arenaRect.top : r.top - arenaRect.top + r.height / 2,
          };
        })();
        if (!spawn) return;
        const xDom = spawn.x;
        const yDom = spawn.y;
        // DOM rects are Y-down from the top; the ortho camera is Y-up.
        const live = buildLiveEffect(
          entry, xDom, canvas.height - yDom, manifest.rev, p.tint, p.iconUrl ?? null, p.dissolve ?? null,
        );
        if (live) liveRef.current.push(live);
      },
    };
    onSink(sink);
    // No cleanup: on unmount the retained sink no-ops on its own, because
    // `canvasRef.current` is `null` by then and `play` returns early. Adding
    // one would only race a caller that still holds it.
  }, [arenaRef, manifest, onSink]);

  return (
    <canvas
      ref={canvasRef}
      width={size.w}
      height={size.h}
      aria-hidden="true"
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        // The scene places its actors with `slotStyle`, at z-index 1 for
        // enemies and 2 for the player and Osty, so 3 puts impact VFX over
        // every creature — which is what the game does: `VfxCmd` adds them to
        // `CombatVfxContainer` / the creature's own vfx container, drawn
        // after the creature containers. It stays below the stage's HUD
        // furniture (`.gs-endturn`, `.gs-pileplaque`) at 6.
        //
        // This does NOT contradict `arenaForegroundZIndex.test.ts`'s rule
        // that nothing in the room may cover a combatant's readouts. That
        // rule is about persistent room SCENERY (`.gs-arena-fg`), which sat
        // over the HP bars permanently. This canvas is fully transparent
        // between effects, and each flipbook lasts only 4-6 frames at 15fps
        // (~270-400ms) — never a persistent occlusion.
        zIndex: 3,
      }}
    />
  );
}
