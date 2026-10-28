/**
 * Task 14: the particle integrator — steps a particle system's `amount`
 * particles forward from spawn to retirement, one fixed-timestep frame at a
 * time. Ported from Godot 4.5.1's `ParticleProcessMaterial` compute shader
 * (`scene/resources/particle_process_material.cpp`, tag `4.5.1-stable`),
 * transcribed from `task-14-godot-source.txt`'s four extracted regions:
 *
 * - **Spawn-time parameter draws** (lines 565-650): `calculate_initial_physical_params`
 *   (tail, `linear_accel`/`radial_accel`/`tangent_accel`/`damping`),
 *   `calculate_initial_dynamics_params`, and `calculate_initial_display_params`.
 *   Each is explicitly commented in the source `// DO NOT REORDER OPERATIONS,
 *   IT BREAKS VISUAL COMPATIBILITY` — the `mixRand()` calls below run in
 *   EXACTLY that source order *within* each function, including drawing (and
 *   discarding) an RNG sample for every field the extracted manifest never
 *   populates (`linear_accel`, `radial_accel`, `tangent_accel`, `angle`,
 *   `angular_velocity`, `directional_velocity`, `orbit_velocity`,
 *   `hue_rotation`, `animation_speed`, `animation_offset`) — those fields are
 *   always 0 for every emitter this game's `extract_vfx.py` has ever produced
 *   (confirmed at the Python dataclass level: `particle_scene.py`'s
 *   `ProcessSpec` declares no field for any of them), so their *value* is
 *   inert, but Godot's shader still burns one `rand_from_seed(alt_seed)` call
 *   per line, unconditionally, and skipping that call would shift every
 *   subsequent draw to a different point in the stream. Keeping the draw and
 *   discarding the result is what preserves stream-position fidelity without
 *   fabricating unmodeled physics.
 *
 *   **Cross-function call order, now CONFIRMED by source** (lines 537-627,
 *   `start()`'s tail and `process()`'s head — see "previously-missed
 *   functions" below): both `start()` and `process()` call, in this exact
 *   order, `calculate_initial_display_params`, then
 *   `calculate_initial_dynamics_params`, then
 *   `calculate_initial_physical_params` — all three drawing sequentially
 *   from the SAME `alt_seed`. `spawnParticle` follows this order exactly.
 *   (An earlier round of this task had display params confirmed first but
 *   guessed at the other two's relative order; this region settles it.)
 *
 *   `process()` recomputes all three from scratch EVERY FRAME, via a fresh
 *   `alt_seed = hash(NUMBER + 1 + RANDOM_SEED)` — but `NUMBER` (the
 *   particle's stable slot) and `RANDOM_SEED` never change across a
 *   particle's life, so this hash — and therefore every draw sequenced from
 *   it — is IDENTICAL every frame. Real Godot's per-frame recompute is a
 *   redundant-but-consistent re-derivation, not fresh randomization; caching
 *   these three functions' results once at spawn (`spawnParticle`, below)
 *   and reusing them every `stepParticle` call reproduces the same
 *   observable values without literally redrawing every frame — which this
 *   module's shared, sequentially-advancing `rng()` stream (a project-wide
 *   simplification of Godot's actual per-particle-independent hash seeding —
 *   see `emit.ts`'s module doc for the same tradeoff) could not do anyway:
 *   redrawing from a shared stream every frame would produce a DIFFERENT
 *   value each frame, not the frame-invariant one Godot's particles actually
 *   have.
 *
 * - **Per-frame process** (lines 940-1150): age/retirement, the
 *   radial-displacement + force + damping + turbulence velocity pipeline,
 *   position integration, then the lifetime-bound scale evaluation
 *   (`process_display_param`) and the scale-over-velocity stage.
 *
 * - **Previously-missed functions, part 1** (lines 638-860, appended after
 *   the first extract sliced `calculate_initial_position` in half and
 *   dropped its three neighbors entirely): `calculate_initial_position`,
 *   `get_random_direction_from_spread`, `process_radial_displacement`, and
 *   `process_physical_parameters` — all four now transcribed literally
 *   (`getRandomDirectionFromSpread`, the radial-displacement block in
 *   `stepParticle`, and `process_physical_parameters`'s damping-curve line)
 *   rather than reconstructed from their call sites. See the Task 14 report
 *   for whether the earlier reconstructions matched (one did; one did not).
 *
 * - **Previously-missed functions, part 2** (lines 860-940, `start()`'s tail
 *   and `process()`'s head — the hole between parts 1 and the per-frame
 *   region): beyond confirming the cross-function call order above, this
 *   region adds two spawn-time steps `spawnParticle` previously missed
 *   entirely — the `AMOUNT_RATIO` rejection draw (`rng()`, unconditional but
 *   inert: `amount_ratio` is not an `EmitterSpec` field, so Godot's own
 *   default of 1.0 means `rand_from_seed(alt_seed) > 1.0` is always false)
 *   and the turbulence-gated initial-position nudge
 *   (`initial_turbulence_displacement = mix(turbulenceInitialDisplacementMin,
 *   Max, rng())`, then `position += noiseDirection(position, spawnTime) *
 *   that`) — a REAL manifest field (`turbulenceInitialDisplacementMin/Max`,
 *   confirmed present in `ParticleEmitterProcess`) this module's first two
 *   rounds never implemented despite the original brief explicitly asking
 *   for it ("implement the paths faithfully anyway" — measured ~1.49e-6 for
 *   every real turbulence-enabled emitter, so it's inert in practice, but
 *   its `rng()` draw still shifts the stream when turbulence is enabled).
 *   `process_display_param(params, 0.0)` — called TWICE in `start()` — is
 *   NOT modeled: it's pure curve sampling (zero `rand_from_seed` calls,
 *   confirmed from its full body in the first extract), and neither call's
 *   result (`start()`'s own local `params.scale`) is ever written to a
 *   register `process()` reads back — `process()` recomputes its own
 *   `params` from scratch each frame (see the recompute note above).
 *   Genuinely dead computation for this integrator's purposes; skipping it
 *   changes neither the RNG stream nor any observable output.
 *
 * - **Two review-found RNG bugs, both fixed:** (1) `spawnParticle` never
 *   drew `calculate_initial_display_params`'s `color_initial_ramp` sample
 *   (source line 33, right after the `lifetime` draw) — see
 *   `ParticleSystemSpec.colorInitialRamp`'s doc; 13 of 64 real emitters set
 *   it, including this module's own `BLUNT_SPECKS` test fixture. (2)
 *   `process_radial_displacement`'s "particle sits on its pivot" fallback
 *   direction was drawn LIVE from `stepParticle`, once per frame the
 *   condition held — wrong, since real Godot's `alt_seed` is frame-invariant
 *   per particle (see the recompute note above), so that direction never
 *   changes frame to frame. Worse, a live per-frame draw makes a particle's
 *   total draw count depend on its own runtime trajectory (how many frames
 *   it happens to linger near the pivot) rather than being a pure function
 *   of emitter config — the exact invariant this module's shared,
 *   sequentially-advancing RNG stream needs every particle to respect. Now
 *   drawn ONCE at spawn, unconditionally (every particle draws it, whether
 *   or not it's ever used), and cached — see `InternalParticle.radialFallbackDir`'s
 *   doc and `simulate.test.ts`'s draw-count invariant test, which directly
 *   asserts the property that let this bug through two review rounds.
 *
 * **What is still NOT modeled, and why:**
 * - `process_orbit_displacement` / `process_directional_displacement`:
 *   `orbitVelocityMin/Max` and `directionalVelocityMin/Max` are not fields of
 *   `particle_scene.py`'s `ProcessSpec` dataclass at all — not merely zero,
 *   structurally absent — so no real emitter can ever drive either, and
 *   nothing here approximates them as always-zero stand-ins.
 * - `EMISSION_SHAPE_POINTS` / `EMISSION_SHAPE_DIRECTED_POINTS`: both read a
 *   baked point-cloud TEXTURE (`emission_texture_points`), which is a resource
 *   the extractor does not carry into the manifest, so `ProcessSpec` has no
 *   field for it and no real emitter in the manifest selects either shape.
 *   `calculateInitialPosition` leaves them at the origin like `POINT`. Note
 *   the draw these two shapes DO cost sits upstream, in
 *   `calculate_initial_display_params` (`ppm451.cpp:600`, one
 *   `rand_from_seed` picking the point index) — the position branch itself
 *   only `texelFetch`es (`ppm451.cpp:661-666`). So were a POINTS emitter ever
 *   to reach this integrator it would desync the stream by one draw, on top
 *   of having no texture to place particles with; the extractor not carrying
 *   the resource is what keeps that unreachable.
 * - `velocity_pivot`: also not a `ProcessSpec` field (Godot default: the
 *   origin) — `process_radial_displacement`'s `global_pivot` is `EMISSION_TRANSFORM`
 *   directly, not a separate always-zero offset field.
 * - Collision (`COLLISION_RIGID`/`COLLISION_HIDE_ON_CONTACT`), the velocity
 *   limit curve, sub-emitters, and all color/hue/animation display state:
 *   no manifest field ever configures collision or a velocity limit curve,
 *   and color/animation are a rendering concern, not this integrator's.
 *
 * **Unguarded `normalize()`, same policy as `turbulence.ts`:** both the
 * damping-friction branch (`VELOCITY = normalize(VELOCITY) * v`) and
 * `process_radial_displacement`'s negative-radial-velocity correction
 * (`normalize(radial_displacement) * min(...)`) divide by a vector length
 * with no zero-length guard in the source — 0/0 is NaN, matching GLSL.
 * Transcribed unguarded here, on the same reasoning `turbulence.ts` documents
 * for `curl_3d`: real emitters don't hit either in practice (nonzero initial
 * velocity; friction damping decays asymptotically rather than reaching
 * exactly zero in finitely many frames; the radial correction only fires
 * when `radial_velocity` is already nonzero and negative, so the vector it
 * normalizes is never the zero vector), and adding a guard the source
 * doesn't have would be an unfaithful port.
 */

import type { TurbulenceParams } from "./turbulence";
import { noiseDirection, turbulenceDisplacement } from "./turbulence";
import { sampleLut } from "./lut";
import { emissionTimes } from "./emit";
import { makeRng } from "./rng";

type Vec3 = readonly [number, number, number];

/** GLSL's unclamped `mix(a, b, t)` — see `turbulence.ts`'s module doc for why
 *  this must NOT clamp `t`. */
function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** `mix(min, max, rand_from_seed(alt_seed))` — draws exactly one RNG sample,
 *  in the caller's stated order, whether or not `min === max`. */
function mixRand(min: number, max: number, rng: () => number): number {
  return mix(min, max, rng());
}

/** `rand_from_seed_m1_p1`: a `[-1, 1)` draw, used only by
 *  `get_random_direction_from_spread`. */
function randM1P1(rng: () => number): number {
  return rng() * 2 - 1;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function length3(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/**
 * `get_random_direction_from_spread` (`task-14-godot-source.txt` lines
 * 445-475, `ppm451.cpp:775`), transcribed literally including its 2D/3D
 * branch split (gated on `particleFlagDisableZ`, exactly as the source gates
 * on `particle_flags[PARTICLE_FLAG_DISABLE_Z]`). Called twice by Godot: once
 * for the initial velocity direction with the material's own `spread` (this
 * specific call site is not itself in the extract — `start()` is truncated
 * before it — but the function's existence, its `spread_angle` parameter,
 * and every real emitter's `particleFlagDisableZ: true` make the intent
 * unambiguous; see `spawnParticle`'s doc), and again by
 * `process_radial_displacement` with a hardcoded 360° when a particle sits
 * exactly on its radial pivot.
 */
function getRandomDirectionFromSpread(spreadAngleDeg: number, spec: ParticleSystemSpec, rng: () => number): Vec3 {
  // `float pi = 3.14159;` — the source's own literal, not Math.PI, per the
  // task's literal-transcription standard (the ~1e-6 rad difference is
  // numerically irrelevant either way).
  const pi = 3.14159;
  const spreadRad = (spreadAngleDeg * pi) / 180;
  if (spec.particleFlagDisableZ) {
    // Spread calculation for 2D.
    let angle1Rad = randM1P1(rng) * spreadRad;
    angle1Rad +=
      spec.direction[0] !== 0
        ? Math.atan2(spec.direction[1], spec.direction[0])
        : Math.sign(spec.direction[1]) * (pi / 2);
    return [Math.cos(angle1Rad), Math.sin(angle1Rad), 0];
  }

  // Spread calculation for 3D — unexercised by any real emitter
  // (particleFlagDisableZ is true on every one this task was measured
  // against) but transcribed in full now that the literal source is given.
  const angle1Rad = randM1P1(rng) * spreadRad;
  const angle2Rad = randM1P1(rng) * spreadRad * (1 - spec.flatness);
  const directionXz: Vec3 = [Math.sin(angle1Rad), 0, Math.cos(angle1Rad)];
  const directionYzRaw: Vec3 = [0, Math.sin(angle2Rad), Math.cos(angle2Rad)];
  // "Better uniform distribution."
  const directionYz: Vec3 = [
    directionYzRaw[0],
    directionYzRaw[1],
    directionYzRaw[2] / Math.max(0.0001, Math.sqrt(Math.abs(directionYzRaw[2]))),
  ];
  const spreadDirectionLocal: Vec3 = [
    directionXz[0] * directionYz[2],
    directionYz[1],
    directionXz[2] * directionYz[2],
  ];
  const dirLen = length3(spec.direction);
  const directionNrm: Vec3 =
    dirLen > 0 ? [spec.direction[0] / dirLen, spec.direction[1] / dirLen, spec.direction[2] / dirLen] : [0, 0, 1];
  // Rotate spread to direction.
  let binormal = cross3([0, 1, 0], directionNrm);
  if (length3(binormal) < 0.0001) {
    // Direction is parallel to Y. Choose Z as the binormal.
    binormal = [0, 0, 1];
  }
  const binormalLen = length3(binormal);
  binormal = [binormal[0] / binormalLen, binormal[1] / binormalLen, binormal[2] / binormalLen];
  const normal = cross3(binormal, directionNrm);
  const spreadDirection: Vec3 = [
    binormal[0] * spreadDirectionLocal[0] + normal[0] * spreadDirectionLocal[1] + directionNrm[0] * spreadDirectionLocal[2],
    binormal[1] * spreadDirectionLocal[0] + normal[1] * spreadDirectionLocal[1] + directionNrm[1] * spreadDirectionLocal[2],
    binormal[2] * spreadDirectionLocal[0] + normal[2] * spreadDirectionLocal[1] + directionNrm[2] * spreadDirectionLocal[2],
  ];
  const sdLen = length3(spreadDirection);
  return [spreadDirection[0] / sdLen, spreadDirection[1] / sdLen, spreadDirection[2] / sdLen];
}

/**
 * The flat, manifest-shaped input this module simulates. Field names and
 * types track `manifest.ts`'s `ParticleEmitter`/`ParticleEmitterProcess` —
 * this is that pair flattened to one level, with curve fields as already
 * loaded/baked LUTs (`readonly number[] | null`, Task 2's `bake_curve`
 * output) rather than the manifest's `vfx_particles/<hash>.json` path
 * strings, exactly the substitution `turbulence.ts`'s `TurbulenceParams`
 * already makes for the turbulence curve. A future wiring task can build this
 * from a real `ParticleEmitter` with `{ ...emitter, ...emitter.process }`
 * plus loading each `*Curve` path into its LUT array.
 *
 * Fields Godot's material always carries but `particle_scene.py`'s
 * `ProcessSpec` dataclass never declares (`linearAccel*`, `radialAccel*`,
 * `tangentAccel*`, `initialAngle*`, `angularVelocity*`, `directionalVelocity*`,
 * `orbitVelocity*`, `velocityLimitCurve`, `collisionMode`, `velocityPivot`,
 * emission shape, hue/animation speed/offset) are intentionally absent from
 * this interface — see the module doc's "what is still NOT modeled" section.
 */
export interface ParticleSystemSpec {
  /** Number of particles in the system (`ParticleEmitter.amount`). */
  amount: number;
  /** `ParticleEmitter.explosiveness`. */
  explosiveness: number;
  /** `ParticleEmitter.randomness`. */
  randomness: number;
  /** `ParticleEmitter.fixedFps`. `0` (Godot's "unset" sentinel) falls back
   *  to Godot's own default of 30 — see `simulate`'s doc. */
  fixedFps: number;
  /** System cycle length in seconds (`ParticleEmitter.lifetime`), Godot's
   *  `LIFETIME` shader global. */
  lifetime: number;
  /** `ParticleEmitter.localCoords` — see `simulate`'s `opts.moveBy` doc. */
  localCoords: boolean;
  /** Constant spawn-position offset (`ParticleEmitter.offset`) — the only
   *  spawn-position field the manifest carries; see module doc. */
  offset: readonly [number, number];

  /** `process.direction` — initial-velocity direction before `spread`
   *  jitter (`getRandomDirectionFromSpread`). */
  direction: readonly [number, number, number];
  /** `process.spread`, degrees. */
  spread: number;
  /** `process.flatness` — only consulted by the 3D branch of
   *  `getRandomDirectionFromSpread`, unexercised by any real (2D,
   *  `particleFlagDisableZ: true`) emitter. */
  flatness: number;
  /** `process.gravity`. */
  gravity: readonly [number, number, number];
  /** `process.initialVelocityMin`. */
  initialVelocityMin: number;
  /** `process.initialVelocityMax`. */
  initialVelocityMax: number;
  /** `process.lifetimeRandomness`. */
  lifetimeRandomness: number;
  animOffsetMin: number;
  animOffsetMax: number;
  /** `process.animSpeed*` and its curve. Together with the offset these are the
   *  flipbook PHASE: `CUSTOM.z = animation_offset + lifetime_percent *
   *  animation_speed` (`ppm451.cpp:1147`), with the offset curve ADDED and the
   *  speed curve MULTIPLIED by `process_display_param` (`:611-637`). */
  animSpeedMin: number;
  animSpeedMax: number;
  animSpeedCurve: readonly number[] | null;
  /** `process.angle*` — the particle's initial rotation, DEGREES. */
  angleMin: number;
  angleMax: number;
  /** MULTIPLIES the initial angle (`base_angle *= texture(...)`, `:1076`). */
  angleCurve: readonly number[] | null;
  /** `process.emissionShape` — Godot's `EmissionShape` enum, VERBATIM:
   *  0 Point, 1 Sphere, 2 Sphere Surface, 3 Box, 4 Points, 5 Directed Points,
   *  6 Ring (`particle_process_material.h:81-90`).
   *
   *  This is the `ParticleProcessMaterial` enum, which `CPUParticles2D`'s only
   *  partly shares — 3 is RECTANGLE there and 6 does not exist
   *  (`cpu_particles_2d.cpp:1547`). Every emitter that reaches this integrator
   *  today is a GPU one; the extractor carries the divergence note. */
  emissionShape: number;
  /** `ParticleEmitter.cpu` — whether this emitter is a `CPUParticles2D`
   *  rather than a `GPUParticles2D`. Only `calculateInitialPosition` reads
   *  it today, and it selects a DIFFERENT formula there rather than tweaking
   *  one (see that function's doc); the rest of this integrator remains the
   *  GPU shader's math, which is a real approximation for the pck's one CPU
   *  emitter and is documented as such on `DissolveEffect.emitters`. */
  cpu: boolean;
  emissionSphereRadius: number;
  emissionBoxExtents: readonly [number, number, number];
  emissionRingAxis: readonly [number, number, number];
  emissionRingHeight: number;
  emissionRingRadius: number;
  emissionRingInnerRadius: number;
  emissionRingConeAngle: number;
  emissionShapeOffset: readonly [number, number, number];
  emissionShapeScale: readonly [number, number, number];
  /** `process.angularVelocity*` — spin, DEGREES PER SECOND. */
  angularVelocityMin: number;
  angularVelocityMax: number;
  /** Multiplies the whole accumulated spin term rather than being integrated
   *  (`base_angle += CUSTOM.y * LIFETIME * omega * curve(t)`, `:1079`). */
  angularVelocityCurve: readonly number[] | null;
  /** `process.colorInitialRamp`. Never READ here — color is a rendering
   *  concern, out of this integrator's scope — but its PRESENCE gates a real
   *  spawn-time RNG draw (`rand_from_seed(alt_seed)` inside
   *  `texture(color_initial_ramp, vec2(rand_from_seed(alt_seed)))`,
   *  `calculate_initial_display_params`) that shifts every subsequent draw.
   *  13 of 64 real particle emitters set this, including `vfx_attack_blunt`'s
   *  `vfx_common_specks` (this module's own `BLUNT_SPECKS` test fixture) —
   *  not a hypothetical, unlike the fields under "still NOT modeled" in the
   *  module doc. */
  colorInitialRamp: readonly (readonly [number, number, number, number])[] | null;
  /** `process.dampingMin`. */
  dampingMin: number;
  /** `process.dampingMax`. */
  dampingMax: number;
  /** Baked `process.dampingCurve`, or `null`. */
  dampingCurve: readonly number[] | null;
  /** `process.particleFlagDampingAsFriction`. */
  particleFlagDampingAsFriction: boolean;
  /** `process.particleFlagDisableZ`. */
  particleFlagDisableZ: boolean;
  /** `process.radialVelocityMin`. */
  radialVelocityMin: number;
  /** `process.radialVelocityMax`. */
  radialVelocityMax: number;
  /** Baked `process.radialVelocityCurve`, or `null`. */
  radialVelocityCurve: readonly number[] | null;
  /** `process.scaleMin`. */
  scaleMin: number;
  /** `process.scaleMax`. */
  scaleMax: number;
  /** Baked `process.scaleCurve`, or `null`. */
  scaleCurve: readonly number[] | null;
  /** `process.scaleOverVelocityMax`. The manifest never carries a `Min`
   *  counterpart (every real emitter's is 0), so it's not modeled here. */
  scaleOverVelocityMax: number;
  /** Baked `process.scaleOverVelocityCurve`, or `null`. */
  scaleOverVelocityCurve: readonly number[] | null;
  /** `process.turbulenceEnabled`. */
  turbulenceEnabled: boolean;
  /** `process.turbulenceInfluenceMin`. */
  turbulenceInfluenceMin: number;
  /** `process.turbulenceInfluenceMax`. */
  turbulenceInfluenceMax: number;
  /** Baked `process.turbulenceInfluenceOverLifeCurve`, or `null`. */
  turbulenceInfluenceOverLifeCurve: readonly number[] | null;
  /** `process.turbulenceInitialDisplacementMin` — spawn-position nudge along
   *  the noise direction, drawn once per particle. Measured ~1.49e-6 for
   *  every real turbulence-enabled emitter (inert in practice), but a real
   *  manifest field — see module doc. */
  turbulenceInitialDisplacementMin: number;
  /** `process.turbulenceInitialDisplacementMax`. */
  turbulenceInitialDisplacementMax: number;
  /** `process.turbulenceNoiseScale`. */
  turbulenceNoiseScale: number;
  /** `process.turbulenceNoiseSpeed`. */
  turbulenceNoiseSpeed: readonly [number, number, number];
  /** `process.turbulenceNoiseSpeedRandom`. */
  turbulenceNoiseSpeedRandom: number;
  /** `process.turbulenceNoiseStrength`. */
  turbulenceNoiseStrength: number;
}

/** One simulated particle's observable state — position, velocity, scale,
 *  life. Rendering-only state (rotation, color, animation frame) is out of
 *  this integrator's scope; see module doc. */
export interface Particle {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  scale: number;
  /** Seconds since spawn. */
  age: number;
  /** Godot's `INSTANCE_CUSTOM.y / INSTANCE_CUSTOM.w` — age as a fraction of
   *  THIS particle's own randomized lifetime, so it reaches 1.0 at
   *  retirement whatever `lifetimeRandomness` drew.
   *
   *  Every shader in the family opens by computing exactly this, and it is
   *  the sample point for `colorRamp`, `alphaCurve` and the flipbook curve.
   *  It is NOT `age / spec.lifetime`: 44 of 64 real emitters author
   *  `lifetimeRandomness > 0`, and for those a particle retires at
   *  `ageFraction == lifetimeFrac`, so sampling by the system lifetime ends
   *  the alpha curve early and the particle pops out at partial opacity
   *  instead of fading. */
  lifetimeFraction: number;
  /** Godot's `INSTANCE_CUSTOM.z`, resolved — `animation_offset +
   *  lifetime_percent * animation_speed`, both already curve-modulated
   *  (`ppm451.cpp:1147`). Named `animOffset` because for every emitter that
   *  leaves `anim_speed` at Godot's default 0 it IS just the offset, which is
   *  most of them; where speed is authored this now advances through the sheet
   *  instead of holding one frame. */
  animOffset: number;
  /** Godot's `CUSTOM.x`: this particle's rotation in RADIANS, initial angle
   *  plus accumulated spin (`ppm451.cpp:1074-1081`). The 2D transform Godot
   *  builds from it is `[cos, -sin; sin, cos]` (`:1090-1092`). */
  angle: number;
  /** The unit direction `particle_flag_align_y` points the particle's local
   *  +Y along, in the emitter's own (Y-down) space: Godot's
   *  `TRANSFORM[1] = normalize(final_velocity)` (`ppm451.cpp:1085-1092`).
   *  Tracked for every particle, but only an emitter that sets the flag draws
   *  with it — and then INSTEAD of `angle`, which that branch never reads.
   *  It is history, not a function of the current velocity: a frame whose
   *  `final_velocity` is exactly zero (linear damping clamps it there) keeps
   *  the previous frame's direction, and a particle that has never moved holds
   *  the spawn identity `(0, 1)`. */
  heading: [number, number];
  /** `false` before spawn, `false` again once retired — `true` only while
   *  live and rendered. */
  alive: boolean;
}

export interface SimulateOpts {
  /** Where the emitter node ends up by the end of the simulated `duration`,
   *  relative to its start at the origin. Applied only at read-out, as a
   *  step (not tracked continuously frame-by-frame): with `localCoords`,
   *  every particle's reported position gets this offset added, because a
   *  local-coordinates particle system is a child of the emitter node and
   *  moves with it; with world coordinates a particle keeps the world
   *  position it already has, since it left the emitter's local space at
   *  spawn. See module doc / Task 14 report for why this is a discrete
   *  read-out adjustment rather than a continuously-tracked transform. */
  moveBy?: readonly [number, number];
}

export interface SimResult {
  /** Number of fixed-timestep frames actually simulated. */
  steps: number;
  particles: Particle[];
  /** Count of particles both spawned and not yet retired at the end of the
   *  simulated `duration`. */
  alive: number;
}

interface InternalParticle {
  spawned: boolean;
  alive: boolean;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  scale: number;
  /** Godot's `params.animation_offset`, drawn once at spawn. */
  animOffset: number;
  /** Godot's `params.animation_speed`, drawn once at spawn. */
  animSpeed: number;
  /** Godot's `dynamic_params.angle`, DEGREES, drawn once at spawn. */
  angleBase: number;
  /** Godot's `dynamic_params.angular_velocity`, DEGREES/SECOND. */
  angularVelocity: number;
  /** `Particle.heading` — carried frame to frame because Godot's zero-velocity
   *  branch re-reads the previous `TRANSFORM[1]`. */
  heading: [number, number];
  /** Post-sign-clamp base scale drawn at spawn, before lifetime curves. */
  baseScale: number;
  /** Godot's `CUSTOM.y`: age as a fraction of the *system* `lifetime`. */
  ageFraction: number;
  /** Godot's `CUSTOM.w`: this particle's own randomized lifetime fraction
   *  (`1 - lifetimeRandomness * rand`) — the threshold `ageFraction` is
   *  compared against for retirement, and the divisor for `lifetime_percent`. */
  lifetimeFrac: number;
  dampingBase: number;
  radialVelocityDraw: number;
  turbInfluenceDraw: number;
  /** `process_radial_displacement`'s "particle sits on its pivot" fallback
   *  direction (`get_random_direction_from_spread(alt_seed, 360.0)`).
   *  Godot re-derives the same `alt_seed` every `process()` call, so this is
   *  frame-invariant for a given particle — drawn ONCE here at spawn and
   *  reused every frame, never drawn live from `stepParticle`. Drawn
   *  unconditionally (every particle, whether or not it ever actually visits
   *  the pivot) so that draw count per particle stays a pure function of
   *  emitter config, not of the runtime trajectory a particle's own random
   *  velocity happens to produce — see module doc and
   *  `simulate.test.ts`'s draw-count invariant test. */
  radialFallbackDir: Vec3;
}

/**
 * `calculate_initial_position` (`ppm451.cpp:638-694`), the emission volume.
 *
 * Godot compiles ONE shape's body into the shader — the C++ emits the branch
 * for the configured `emission_shape` and nothing else — so the runtime switch
 * here is a reshaping, not a behaviour change. What must survive that reshaping
 * is the DRAW COUNT AND ORDER, because `alt_seed` is one sequential stream and
 * the spread direction is drawn right after this:
 *
 *     Point            0 draws
 *     Sphere           3  (s, t, p)
 *     Sphere Surface   2  (s, t)
 *     Box              3  (x, y, z)
 *     Points/Directed  0  (the draw happens in the display params instead)
 *     Ring             3  (y_pos, ring_spawn_angle, ring_random_radius)
 *
 * `Points`/`Directed Points` need an emission TEXTURE, which the extractor has
 * no reader for; they fall through to the origin here, which is what a missing
 * texture would sample to anyway, and no emitter in scope authors either.
 *
 * The trailing `pos * emission_shape_scale + emission_shape_offset` applies to
 * every shape including Point, which is why an emitter with a plain offset and
 * a non-default scale is not a no-op.
 *
 * **`spec.cpu` selects a different function entirely, not a variant of this
 * one.** `CPUParticles2D` is a separate implementation that happens to expose
 * the same property names; for the emission sphere it computes a genuinely
 * different distribution from a different number of draws
 * (`cpu_particles_2d.cpp:893-902`), and it has no `emission_shape_scale`/
 * `_offset` at all (neither string appears anywhere in that file). Only
 * `vfx_monster_death`'s debris burst takes that path today, and running it
 * through the GPU formula would be the kind of plausible-looking invented
 * physics this port exists to avoid.
 */
function calculateInitialPosition(spec: ParticleSystemSpec, rng: () => number): Vec3 {
  const pi = 3.14159; // The shader's own literal, not Math.PI — see module doc.
  let pos: Vec3 = [0, 0, 0];
  if (spec.cpu) {
    switch (spec.emissionShape) {
      case 1: {
        // `cpu_particles_2d.cpp:893-897`. TWO draws, angle first, and a flat
        // 2D disc — no z, and uniform in RADIUS rather than in area, so it
        // is denser toward the centre than the GPU ball's projection is.
        const t = 2 * pi * rng();
        const radius = spec.emissionSphereRadius * rng();
        pos = [Math.cos(t) * radius, Math.sin(t) * radius, 0];
        break;
      }
      case 2: {
        // `cpu_particles_2d.cpp:898-902`. Also two draws, but `s` FIRST and
        // the angle second — the opposite order to the sphere above. Still
        // flat, and `sqrt(1 - s*s)` means it is a filled disc despite the
        // `SPHERE_SURFACE` name: the 3D shell's projection, not a ring.
        const s = rng();
        const t = 2 * pi * rng();
        const radius = spec.emissionSphereRadius * Math.sqrt(1 - s * s);
        pos = [Math.cos(t) * radius, Math.sin(t) * radius, 0];
        break;
      }
      // Every other CPU shape is deliberately unreachable: `_parse_cpu_process`
      // reads only `emission_shape` and `emission_sphere_radius`, because the
      // two enums diverge past 2 (3 is Rectangle here, Box there; 6 does not
      // exist here) and mapping the rest by value would silently corrupt them.
      // A CPU emitter authoring one of those would need the extractor extended
      // first, so falling through to the origin is the honest answer.
    }
    // No trailing `* scale + offset`: `CPUParticles2D` has no such property.
    return pos;
  }
  switch (spec.emissionShape) {
    case 1: {
      // Sphere: a point in the BALL, not on it — `mix(0, surface, p)` walks in
      // from the centre.
      const s = rng() * 2 - 1;
      const tAng = rng() * 2 * pi;
      const pw = rng();
      const radius = spec.emissionSphereRadius * Math.sqrt(1 - s * s);
      pos = [
        radius * Math.cos(tAng) * pw,
        radius * Math.sin(tAng) * pw,
        spec.emissionSphereRadius * s * pw,
      ];
      break;
    }
    case 2: {
      // Sphere surface: the same two draws, no third — always on the shell.
      const s = rng() * 2 - 1;
      const tAng = rng() * 2 * pi;
      const radius = spec.emissionSphereRadius * Math.sqrt(1 - s * s);
      pos = [radius * Math.cos(tAng), radius * Math.sin(tAng), spec.emissionSphereRadius * s];
      break;
    }
    case 3: {
      // Box. GLSL does not specify argument evaluation order inside a
      // `vec3(...)`, so x/y/z is a convention every real driver follows — and
      // the order matters here only because these are three stream draws.
      const bx = rng() * 2 - 1;
      const by = rng() * 2 - 1;
      const bz = rng() * 2 - 1;
      pos = [
        bx * spec.emissionBoxExtents[0],
        by * spec.emissionBoxExtents[1],
        bz * spec.emissionBoxExtents[2],
      ];
      break;
    }
    case 6: {
      // Ring. The `skew`/`pow` pair biases height toward the narrow end of a
      // cone, so a cone-angle of 90 (every emitter in scope) leaves
      // `top_radius == radius_clamped`, `skew == 1` and the distribution even.
      const radiusClamped = Math.max(0.001, spec.emissionRingRadius);
      const topRadius = Math.max(
        radiusClamped - Math.tan(((90 - spec.emissionRingConeAngle) * pi) / 180) * spec.emissionRingHeight,
        0,
      );
      let yPos = rng();
      const skew = Math.max(
        Math.min(radiusClamped, topRadius) / Math.max(radiusClamped, topRadius),
        0.5,
      );
      yPos = radiusClamped < topRadius ? Math.pow(yPos, skew) : 1 - Math.pow(yPos, skew);
      const ringSpawnAngle = rng() * 2 * pi;
      const inner = spec.emissionRingInnerRadius;
      let ringRandomRadius = Math.sqrt(
        rng() * (radiusClamped * radiusClamped - inner * inner) + inner * inner,
      );
      ringRandomRadius +=
        (ringRandomRadius * (topRadius / radiusClamped) - ringRandomRadius) * yPos;
      const a = spec.emissionRingAxis;
      const axis: Vec3 =
        a[0] === 0 && a[1] === 0 && a[2] === 0 ? [0, 0, 1] : normalizeVec3([a[0], a[1], a[2]]);
      // `abs(axis) == vec3(1, 0, 0)` picks the fallback cross partner.
      let ortho: Vec3 =
        Math.abs(axis[0]) === 1 && axis[1] === 0 && axis[2] === 0
          ? cross(axis, [0, 1, 0])
          : cross(axis, [1, 0, 0]);
      ortho = normalizeVec3(ortho);
      // Rodrigues rotation of `ortho` about `axis` by `ringSpawnAngle`. Written
      // out as the shader's own mat3 so the transcription stays checkable.
      const sn = Math.sin(ringSpawnAngle);
      const cs = Math.cos(ringSpawnAngle);
      const oc = 1 - cs;
      const r0: Vec3 = [
        cs + axis[0] * axis[0] * oc,
        axis[0] * axis[1] * oc - axis[2] * sn,
        axis[0] * axis[2] * oc + axis[1] * sn,
      ];
      const r1: Vec3 = [
        axis[0] * axis[1] * oc + sn * axis[2],
        cs + axis[1] * axis[1] * oc,
        axis[1] * axis[2] * oc - axis[0] * sn,
      ];
      const r2: Vec3 = [
        axis[2] * axis[0] * oc - axis[1] * sn,
        axis[2] * axis[1] * oc + axis[0] * sn,
        cs + axis[2] * axis[2] * oc,
      ];
      // GLSL `mat3(c0, c1, c2)` takes COLUMNS, and `m * v` is
      // `c0*v.x + c1*v.y + c2*v.z`.
      ortho = normalizeVec3([
        r0[0] * ortho[0] + r1[0] * ortho[1] + r2[0] * ortho[2],
        r0[1] * ortho[0] + r1[1] * ortho[1] + r2[1] * ortho[2],
        r0[2] * ortho[0] + r1[2] * ortho[1] + r2[2] * ortho[2],
      ]);
      const h = yPos * spec.emissionRingHeight - spec.emissionRingHeight / 2;
      pos = [
        ortho[0] * ringRandomRadius + h * axis[0],
        ortho[1] * ringRandomRadius + h * axis[1],
        ortho[2] * ringRandomRadius + h * axis[2],
      ];
      break;
    }
    default:
      // Point (0) and the two texture-driven shapes: no draws, origin.
      break;
  }
  return [
    pos[0] * spec.emissionShapeScale[0] + spec.emissionShapeOffset[0],
    pos[1] * spec.emissionShapeScale[1] + spec.emissionShapeOffset[1],
    pos[2] * spec.emissionShapeScale[2] + spec.emissionShapeOffset[2],
  ];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalizeVec3(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  return len > 0 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, 0];
}

/**
 * `start()` (`task-14-godot-source.txt` lines 537-605, `ppm451.cpp:860-940`),
 * transcribed in source order — see module doc for the (now CONFIRMED)
 * cross-function call order, which draws are real values vs. discarded
 * stream-position placeholders, and why `process_display_param(params, 0.0)`
 * (called twice in the real `start()`) is deliberately not modeled.
 * `simTime` is the simulation clock at the moment this particle spawns,
 * needed only for the turbulence initial-displacement nudge's
 * `noiseDirection` call.
 */
function spawnParticle(
  p: InternalParticle,
  spec: ParticleSystemSpec,
  simTime: number,
  turbParams: TurbulenceParams,
  rng: () => number,
): void {
  // calculate_initial_display_params — "DO NOT REORDER OPERATIONS". Called
  // first by start() (confirmed).
  let scale = mixRand(spec.scaleMin, spec.scaleMax, rng);
  scale = Math.sign(scale) * Math.max(Math.abs(scale), 0.001);
  rng(); // hue_rotation
  // `params.animation_speed`, drawn HERE — before the offset, which is the
  // order `calculate_initial_display_params` uses (`ppm451.cpp:585-608`). The
  // draw was always taken for stream position; the VALUE is now kept, because
  // `CUSTOM.z` is offset PLUS `lifetime_percent * speed`, and dropping the
  // speed term froze every sheet on its starting frame.
  p.animSpeed = mixRand(spec.animSpeedMin, spec.animSpeedMax, rng);
  // `params.animation_offset = mix(anim_offset_min, anim_offset_max,
  // rand_from_seed(alt_seed))`. The draw was always taken here for stream
  // position; the VALUE is now kept, because the flipbook shaders read it as
  // `INSTANCE_CUSTOM.z` to pick each particle's starting frame. Discarding it
  // made every particle of an emitter animate in lockstep.
  p.animOffset = mixRand(spec.animOffsetMin, spec.animOffsetMax, rng);
  p.lifetimeFrac = 1.0 - spec.lifetimeRandomness * rng();
  // params.color = color_value (no draw) then, gated on the ramp being
  // valid: params.color *= texture(color_initial_ramp, vec2(rand_from_seed(
  // alt_seed))) — a REAL RNG-consuming draw even though this integrator
  // never reads the resulting color. 13 of 64 real emitters set this,
  // including BLUNT_SPECKS below — see ParticleSystemSpec.colorInitialRamp's
  // doc for why round 2 missed it.
  if (spec.colorInitialRamp) rng();
  p.baseScale = scale;
  p.scale = scale;

  // calculate_initial_dynamics_params — "DO NOT REORDER OPERATIONS". Called
  // second (confirmed).
  // The first two draws of `calculate_initial_dynamics_params`
  // (`ppm451.cpp:571-582`), in its order. Both were taken and discarded; both
  // are now kept, which moves no draw — the particles simply stop rendering
  // unrotated. DEGREES: `dynamic_params.angle` is a straight `mix()` of the
  // degree-valued uniforms, and `process()` converts once at the end
  // (`CUSTOM.x = base_angle * degree_to_rad`, `:1081`).
  p.angleBase = mixRand(spec.angleMin, spec.angleMax, rng);
  p.angularVelocity = mixRand(spec.angularVelocityMin, spec.angularVelocityMax, rng);
  const initialVelocityMultiplier = mixRand(spec.initialVelocityMin, spec.initialVelocityMax, rng);
  rng(); // directional_velocity
  p.radialVelocityDraw = mixRand(spec.radialVelocityMin, spec.radialVelocityMax, rng);
  rng(); // orbit_velocity
  p.turbInfluenceDraw = spec.turbulenceEnabled
    ? mixRand(spec.turbulenceInfluenceMin, spec.turbulenceInfluenceMax, rng)
    : 0;

  // calculate_initial_physical_params (tail): linear_accel, radial_accel,
  // tangent_accel, damping — in that order. Called third (confirmed).
  rng(); // linear_accel
  rng(); // radial_accel
  rng(); // tangent_accel
  p.dampingBase = mixRand(spec.dampingMin, spec.dampingMax, rng);

  // process_display_param(params, 0.0), called here and again after
  // RESTART_VELOCITY below — not modeled, see module doc.

  // AMOUNT_RATIO rejection — draws unconditionally, always inert (see
  // module doc).
  rng();

  // RESTART_POSITION: calculate_initial_position (= plain `offset` for
  // every real emitter — see module doc) plus the turbulence-gated initial
  // displacement nudge. EMISSION_TRANSFORM is identity in our model (see
  // SimulateOpts.moveBy's doc), so the final `TRANSFORM = EMISSION_TRANSFORM
  // * TRANSFORM` line is a no-op here.
  const emitted = calculateInitialPosition(spec, rng);
  p.x = spec.offset[0] + emitted[0];
  p.y = spec.offset[1] + emitted[1];
  p.z = emitted[2];
  if (spec.turbulenceEnabled) {
    const initialTurbulenceDisplacement = mixRand(
      spec.turbulenceInitialDisplacementMin,
      spec.turbulenceInitialDisplacementMax,
      rng,
    );
    const noiseDir = noiseDirection([p.x, p.y, p.z], simTime, turbParams);
    p.x += noiseDir[0] * initialTurbulenceDisplacement;
    p.y += noiseDir[1] * initialTurbulenceDisplacement;
    p.z += noiseDir[2] * initialTurbulenceDisplacement;
  }
  // `start()`'s tail: `TRANSFORM[3].z = 0.0` under PARTICLE_FLAG_DISABLE_Z
  // (`ppm451.cpp`), AFTER the turbulence nudge — which is why this is here and
  // not folded into the shape. Every emitter in scope sets the flag, so before
  // emission shapes existed a hardcoded `p.z = 0` was indistinguishable from
  // this; a sphere's z is the first thing that could have made it visible.
  if (spec.particleFlagDisableZ) p.z = 0;

  // RESTART_VELOCITY: get_random_direction_from_spread(alt_seed, spread) *
  // initial_velocity_multiplier (this call site, only inferred in earlier
  // rounds, is now literally confirmed by the source). The subsequent
  // `VELOCITY = (EMISSION_TRANSFORM * vec4(VELOCITY, 0)).xyz` (rotation-only;
  // no-op, our EMISSION_TRANSFORM carries no rotation) and `VELOCITY +=
  // EMITTER_VELOCITY * inherit_emitter_velocity_ratio` (no-op: the emitter
  // is stationary throughout a simulated span, and
  // inheritEmitterVelocityRatio is not a ProcessSpec field) draw nothing
  // from alt_seed.
  const dir = getRandomDirectionFromSpread(spec.spread, spec, rng);
  p.vx = dir[0] * initialVelocityMultiplier;
  p.vy = dir[1] * initialVelocityMultiplier;
  p.vz = dir[2] * initialVelocityMultiplier;

  // process_radial_displacement's "particle sits on its pivot" fallback
  // direction — start() and this spawn frame's first process() call fuse
  // into one invocation in real Godot (see module doc's recompute note), so
  // this corresponds to that first process_radial_displacement call. Drawn
  // HERE, unconditionally, rather than live inside stepParticle's pivot
  // check — see InternalParticle.radialFallbackDir's doc for why (frame
  // invariance in real Godot; a config-fixed draw count per particle here).
  p.radialFallbackDir = getRandomDirectionFromSpread(360, spec, rng);

  // `RESTART_ROT_SCALE` resets the basis to identity (`ppm451.cpp:878-882`),
  // so a respawned particle does not inherit its previous life's heading.
  p.heading = [0, 1];
  p.ageFraction = 0;
  p.spawned = true;
  p.alive = true;
}

/**
 * The per-frame process (`task-14-godot-source.txt` lines 940-1150),
 * transcribed in source order. `emissionPos` is `EMISSION_TRANSFORM[3].xyz`
 * — held at the origin for the whole simulated `duration` (see
 * `SimulateOpts.moveBy`'s doc for why the emitter's motion is a read-out
 * adjustment, not a per-frame one); it also stands in for
 * `process_radial_displacement`'s `global_pivot` directly, since
 * `velocity_pivot` is never a real field (see module doc).
 */
function stepParticle(
  p: InternalParticle,
  spec: ParticleSystemSpec,
  dt: number,
  simTime: number,
  emissionPos: Vec3,
  turbParams: TurbulenceParams,
): void {
  // CUSTOM.y += DELTA / LIFETIME; INTERPOLATE_TO_END is always 0 for a
  // one-shot workbench render, so `mix(CUSTOM.y, 1.0, 0.0)` is a no-op and is
  // dropped.
  p.ageFraction += dt / spec.lifetime;
  const lifetimePercent = p.ageFraction / p.lifetimeFrac;
  if (p.ageFraction > p.lifetimeFrac) {
    p.alive = false;
  }

  // process_radial_displacement (task-14-godot-source.txt lines 477-497,
  // ppm451.cpp:807), transcribed literally.
  let controlledDisplacement: Vec3 = [0, 0, 0];
  if (dt >= 0.001) {
    let radialDisplacementMultiplier = 1.0;
    if (spec.radialVelocityCurve) radialDisplacementMultiplier = sampleLut(spec.radialVelocityCurve, lifetimePercent);
    const toPivot: Vec3 = [p.x - emissionPos[0], p.y - emissionPos[1], p.z - emissionPos[2]];
    const toPivotLen = length3(toPivot);
    if (toPivotLen > 0.01) {
      const inv = radialDisplacementMultiplier * p.radialVelocityDraw;
      controlledDisplacement = [(toPivot[0] / toPivotLen) * inv, (toPivot[1] / toPivotLen) * inv, (toPivot[2] / toPivotLen) * inv];
    } else {
      // Cached at spawn (InternalParticle.radialFallbackDir's doc) — NOT
      // drawn live here. Godot's alt_seed is frame-invariant per particle,
      // so this direction never changes frame to frame; drawing it live
      // from this module's shared stream would (a) give a DIFFERENT
      // direction every frame a particle lingers near its pivot, visible as
      // jitter Godot doesn't have, and (b) make a particle's total draw
      // count depend on its own runtime trajectory rather than emitter
      // config — see simulate.test.ts's draw-count invariant test.
      controlledDisplacement = [
        p.radialFallbackDir[0] * p.radialVelocityDraw,
        p.radialFallbackDir[1] * p.radialVelocityDraw,
        p.radialFallbackDir[2] * p.radialVelocityDraw,
      ];
    }
    if (radialDisplacementMultiplier * p.radialVelocityDraw < 0) {
      // Prevent inwards velocity to flicker once the point is reached.
      const mag = Math.min(Math.abs(radialDisplacementMultiplier * p.radialVelocityDraw), toPivotLen / dt);
      const cdLen = length3(controlledDisplacement);
      controlledDisplacement = [
        (controlledDisplacement[0] / cdLen) * mag,
        (controlledDisplacement[1] / cdLen) * mag,
        (controlledDisplacement[2] / cdLen) * mag,
      ];
    }
  }
  // process_directional_displacement / process_orbit_displacement: no
  // ProcessSpec field ever drives either (see module doc) — omitted, not
  // approximated as zero-contribution stand-ins.

  // process_physical_parameters(lifetime_percent) (task-14-godot-source.txt
  // lines 509-522, ppm451.cpp:839), transcribed literally: only `damping`
  // has a manifest-carried curve (linear/radial/tangent accel curves are
  // never real — see module doc — so their `*=` lines are structurally
  // no-ops on an always-0 baseline and are not written out as dead code).
  let damping = p.dampingBase;
  if (spec.dampingCurve) damping *= sampleLut(spec.dampingCurve, lifetimePercent);

  // Force block (lines 115-142): gravity is the only manifest-carried
  // contributor — linear/radial/tangent accel are always 0 (see module doc),
  // so their `force +=` lines are structurally no-ops and are not written
  // out as dead arithmetic.
  const force: Vec3 = [spec.gravity[0], spec.gravity[1], spec.particleFlagDisableZ ? 0 : spec.gravity[2]];
  p.vx += force[0] * dt;
  p.vy += force[1] * dt;
  p.vz += force[2] * dt;

  // Damping block (lines 143-162) — unguarded normalize(), see module doc.
  if (damping > 0) {
    let v = length3([p.vx, p.vy, p.vz]);
    if (!spec.particleFlagDampingAsFriction) {
      v -= damping * dt;
    } else if (v > 0.001) {
      const damp = v * v * damping * 0.05 * dt;
      v -= damp;
    }
    if (v < 0) {
      p.vx = 0;
      p.vy = 0;
      p.vz = 0;
    } else {
      const cur = length3([p.vx, p.vy, p.vz]);
      p.vx = (p.vx / cur) * v;
      p.vy = (p.vy / cur) * v;
      p.vz = (p.vz / cur) * v;
    }
  }

  // Collision (lines 164-180): no manifest field ever sets a collision mode
  // — omitted.

  // Turbulence before limiting (lines 182-205), gated on turbulence_enabled.
  if (spec.turbulenceEnabled) {
    const noiseDir = noiseDirection([p.x, p.y, p.z], simTime, turbParams);
    const noiseDirLen = length3(noiseDir);
    const noiseDirUnit: Vec3 = [noiseDir[0] / noiseDirLen, noiseDir[1] / noiseDirLen, noiseDir[2] / noiseDirLen];
    const velInfl = turbulenceDisplacement(lifetimePercent, turbParams, p.turbInfluenceDraw);

    let velMag = length3([p.vx, p.vy, p.vz]);
    let target: Vec3 = [
      noiseDirUnit[0] * velMag * (1 + (1 - velInfl) * 0.2),
      noiseDirUnit[1] * velMag * (1 + (1 - velInfl) * 0.2),
      noiseDirUnit[2] * velMag * (1 + (1 - velInfl) * 0.2),
    ];
    p.vx = mix(p.vx, target[0], velInfl);
    p.vy = mix(p.vy, target[1], velInfl);
    p.vz = mix(p.vz, target[2], velInfl);

    velMag = length3(controlledDisplacement);
    target = [
      noiseDirUnit[0] * velMag * (1 + (1 - velInfl) * 0.2),
      noiseDirUnit[1] * velMag * (1 + (1 - velInfl) * 0.2),
      noiseDirUnit[2] * velMag * (1 + (1 - velInfl) * 0.2),
    ];
    controlledDisplacement = [
      mix(controlledDisplacement[0], target[0], velInfl),
      mix(controlledDisplacement[1], target[1], velInfl),
      mix(controlledDisplacement[2], target[2], velInfl),
    ];
  }

  let finalVelocity: Vec3 = [
    controlledDisplacement[0] + p.vx,
    controlledDisplacement[1] + p.vy,
    controlledDisplacement[2] + p.vz,
  ];

  // Velocity limit curve (lines 208-213): no manifest field — omitted.

  if (spec.particleFlagDisableZ) finalVelocity = [finalVelocity[0], finalVelocity[1], 0];

  p.x += finalVelocity[0] * dt;
  p.y += finalVelocity[1] * dt;
  p.z += finalVelocity[2] * dt;

  // `particle_flag_align_y` (`ppm451.cpp:1085-1092`): the post-limit,
  // Z-flattened `final_velocity` — radial displacement included, which a
  // heading taken from `VELOCITY` alone would miss — or, when that is zero,
  // the previous direction re-normalized.
  const headingLen = Math.hypot(finalVelocity[0], finalVelocity[1]);
  if (headingLen > 0) p.heading = [finalVelocity[0] / headingLen, finalVelocity[1] / headingLen];

  // process_display_param (lines 47-72 of the source, called at line 221):
  // only the scale curve is modeled — color/hue/animation are a rendering
  // concern, out of this integrator's scope (see module doc).
  let scale = p.baseScale;
  if (spec.scaleCurve) scale *= sampleLut(spec.scaleCurve, lifetimePercent);

  // Angle / TRANSFORM orientation (lines 223-278): no manifest field ever
  // drives rotation — omitted (rendering concern; see module doc).

  // scale_over_velocity (lines 280-286).
  if (spec.scaleOverVelocityCurve) {
    const finalVelocityLen = length3(finalVelocity);
    if (finalVelocityLen > 0.001) {
      const t = spec.scaleOverVelocityMax !== 0 ? clamp01(finalVelocityLen / spec.scaleOverVelocityMax) : 0;
      scale *= sampleLut(spec.scaleOverVelocityCurve, t);
    } else {
      scale *= sampleLut(spec.scaleOverVelocityCurve, 0);
    }
  }
  p.scale = scale;
}

/**
 * A particle system caught mid-flight: everything the integrator needs to
 * take the next fixed-timestep frame. Held by a caller that draws the same
 * system over and over — one state per live emitter, advanced once per
 * display frame — instead of replaying the whole system from t=0 each time,
 * which made the draw cost quadratic in frames (one composite measured at
 * 96% of a core to render its own one-second life).
 *
 * Every field is private to this module in spirit: `advanceTo` and `readout`
 * are the whole interface. `rng` lives here because spawns happen mid-flight,
 * so the stream must survive between calls — which is also why a state can
 * never be rewound or forked.
 */
export interface SimState {
  readonly spec: ParticleSystemSpec;
  /** Seconds per fixed-timestep frame, `1 / fixedFps`. */
  readonly dt: number;
  readonly fixedFps: number;
  /** Frames integrated so far. The simulation clock is `framesDone * dt` —
   *  quantized, deliberately: the caller's display clock never sets the
   *  step size, so a 144 Hz monitor does the same work as a 60 Hz one for
   *  the identical output. */
  framesDone: number;
  readonly particles: InternalParticle[];
  readonly spawnTimes: number[];
  readonly turbParams: TurbulenceParams;
  readonly rng: () => number;
}

/**
 * Builds a system at t=0. Consumes `rng` immediately for the emission
 * schedule (Godot draws it once, up front) and then holds it for the spawn
 * draws `advanceTo` makes later.
 */
export function createSimState(spec: ParticleSystemSpec, rng: () => number): SimState {
  const fixedFps = spec.fixedFps > 0 ? spec.fixedFps : 30;
  const spawnTimes = emissionTimes(
    { amount: spec.amount, lifetime: spec.lifetime, explosiveness: spec.explosiveness, randomness: spec.randomness },
    rng,
  );

  const particles: InternalParticle[] = spawnTimes.map(() => ({
    spawned: false,
    alive: false,
    x: 0,
    y: 0,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    scale: 0,
    baseScale: 0,
    ageFraction: 0,
    lifetimeFrac: 1,
    animOffset: 0,
    animSpeed: 0,
    angleBase: 0,
    angularVelocity: 0,
    dampingBase: 0,
    radialVelocityDraw: 0,
    turbInfluenceDraw: 0,
    radialFallbackDir: [0, 0, 0],
    heading: [0, 1],
  }));

  return {
    spec,
    dt: 1 / fixedFps,
    fixedFps,
    framesDone: 0,
    particles,
    spawnTimes,
    turbParams: {
      strength: spec.turbulenceNoiseStrength,
      scale: spec.turbulenceNoiseScale,
      speed: spec.turbulenceNoiseSpeed,
      speedRandom: spec.turbulenceNoiseSpeedRandom,
      influenceMin: spec.turbulenceInfluenceMin,
      influenceMax: spec.turbulenceInfluenceMax,
      influenceOverLifeLut: spec.turbulenceInfluenceOverLifeCurve,
    },
    rng,
  };
}

/**
 * Integrates forward until the system's clock reaches `duration` seconds,
 * one whole `dt` frame at a time.
 *
 * FORWARD ONLY: a `duration` already passed is a no-op, never a rewind. The
 * frame sequence, spawn order and RNG draw order are therefore identical to
 * running the whole span in one call — the caller's step size is invisible
 * to the result. `simulate.test.ts` asserts exactly that, and it is the
 * property that makes holding a state across frames equivalent to the replay
 * this replaced.
 */
export function advanceTo(state: SimState, duration: number): void {
  const target = Math.round(duration * state.fixedFps);
  if (target <= state.framesDone) return;

  // EMISSION_TRANSFORM held at the origin for the whole simulated span — see
  // SimulateOpts.moveBy's doc.
  const emissionPos: Vec3 = [0, 0, 0];
  const { spec, dt, particles, spawnTimes, turbParams, rng } = state;

  for (let frame = state.framesDone + 1; frame <= target; frame++) {
    const simTime = frame * dt;
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      if (!p.spawned && simTime >= spawnTimes[i]) {
        spawnParticle(p, spec, simTime, turbParams, rng);
      }
      if (p.spawned && p.alive) {
        stepParticle(p, spec, dt, simTime, emissionPos, turbParams);
      }
    }
  }
  state.framesDone = target;
}

/**
 * Projects the current state into the renderer-facing `SimResult`. Pure: the
 * caller reads out once per display frame from a state it keeps, so nothing
 * here may mutate — `moveBy` in particular is applied to the copy, never
 * banked into the particle.
 */
export function readout(state: SimState, opts?: SimulateOpts): SimResult {
  const { spec, particles } = state;
  const moveBy = opts?.moveBy ?? [0, 0];
  const readoutOffset: readonly [number, number] = spec.localCoords ? moveBy : [0, 0];

  const particlesOut: Particle[] = particles.map((p) => {
    // Godot's `lifetime_percent` (`ppm451.cpp:947`), the sample point for
    // every curve below.
    const lifetimeFraction = p.lifetimeFrac > 0 ? p.ageFraction / p.lifetimeFrac : 0;
    // `process_display_param(params, lifetime_percent)` (`:611-637`). The two
    // anim curves are asymmetric IN THE SOURCE and that is not a typo here:
    // the offset curve is `+=`, the speed curve is `*=`.
    const animSpeed =
      p.animSpeed * (spec.animSpeedCurve ? sampleLut(spec.animSpeedCurve, lifetimeFraction) : 1);
    // `CUSTOM.x` (`:1074-1081`). The angle curve MULTIPLIES the initial angle;
    // the angular-velocity curve multiplies the whole accumulated spin term
    // rather than being integrated, so elapsed seconds is `CUSTOM.y * LIFETIME`
    // — the RAW normalized age times the system lifetime, NOT
    // `lifetime_percent`, which is that age divided by this particle's own
    // randomized lifetime.
    let baseAngleDeg = p.angleBase * (spec.angleCurve ? sampleLut(spec.angleCurve, lifetimeFraction) : 1);
    const spinCurve = spec.angularVelocityCurve
      ? sampleLut(spec.angularVelocityCurve, lifetimeFraction)
      : 1;
    baseAngleDeg += p.ageFraction * spec.lifetime * p.angularVelocity * spinCurve;
    return {
      x: p.x + readoutOffset[0],
      y: p.y + readoutOffset[1],
      z: p.z,
      vx: p.vx,
      vy: p.vy,
      vz: p.vz,
      scale: p.scale,
      age: p.ageFraction * spec.lifetime,
      lifetimeFraction,
      // `CUSTOM.z = params.animation_offset + lifetime_percent *
      // params.animation_speed` (`:1147`).
      animOffset: p.animOffset + lifetimeFraction * animSpeed,
      angle: (baseAngleDeg * Math.PI) / 180,
      heading: [p.heading[0], p.heading[1]],
      alive: p.spawned && p.alive,
    };
  });

  return {
    steps: state.framesDone,
    particles: particlesOut,
    alive: particlesOut.filter((p) => p.alive).length,
  };
}

/**
 * Steps `spec.amount` particles forward for `duration` seconds at
 * `spec.fixedFps` (Godot's own default of 30 when `fixedFps` is `0`/unset —
 * matches the measured manifest distribution: 24 scenes author 60, one 25,
 * eleven leave it unset). Fully determined by `spec`, `duration` and `rng` —
 * never touches `Math.random`/`Date`.
 *
 * The whole-span form of the three functions above, and deliberately nothing
 * more than their composition: this module's test suite drives `simulate`,
 * so keeping it a thin wrapper makes that suite the oracle for the resumable
 * path the draw loop actually runs.
 */
export function simulate(spec: ParticleSystemSpec, duration: number, rng: () => number, opts?: SimulateOpts): SimResult {
  const state = createSimState(spec, rng);
  advanceTo(state, duration);
  return readout(state, opts);
}

/** Test/debug convenience: simulates a single particle (`amount: 1`,
 *  immediate spawn) for exactly `n` fixed-timestep frames and returns its
 *  final state. Uses an internal fixed seed — every field this module reads
 *  that participates in `n`-frame physics is exercised through explicit
 *  `min`/`max` overrides in each call site, so the seed choice doesn't
 *  affect the numeric result, only the discarded stream-position draws (see
 *  module doc). */
export function stepN(spec: Omit<ParticleSystemSpec, "amount"> & Partial<Pick<ParticleSystemSpec, "amount">>, n: number): Particle {
  const full: ParticleSystemSpec = { ...spec, amount: spec.amount ?? 1 };
  const duration = n / (full.fixedFps > 0 ? full.fixedFps : 30);
  return simulate(full, duration, makeRng(1)).particles[0];
}
