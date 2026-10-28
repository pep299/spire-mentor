import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, it, expect } from "vitest";

// simulate.ts's `ParticleSystemSpec` is NOT a `Pick<>` of `manifest.ts`'s
// `ParticleEmitterProcess` — it's an independently hand-written interface
// covering the fields the integrator actually consumes (plus a few
// explicitly-skipped rendering-only ones: alpha/color curves,
// particleFlagRotateY, and particleFlagAlignY, which the renderer reads).`particle_scene.py`'s `ProcessSpec` dataclass
// today declares no field for orbit/directional velocity, linear/radial/
// tangent accel, initial angle/angular velocity, a velocity-limit curve, or
// collision — Task 4's "every dataclass field reaches the manifest" guard
// means that's a promise about what's already there, not a bound on what
// could be added. If `ProcessSpec` ever grows one of these fields,
// `extract_vfx.py` starts serializing it automatically (same guard,
// forwards), and `manifestShape.test.ts`'s exact-key check would force
// `manifest.ts`'s TYPE to be updated to match — but nothing forces
// `simulate.ts` to be updated too. An effect that then authors a real
// (non-default) value for one of these would silently render with that
// physics missing, and no existing test would catch it: manifestShape only
// compares the manifest against manifest.ts's type, never against what the
// integrator implements.
//
// This suite closes that specific gap: it reads the real committed manifest
// and asserts every particle emitter's `process` object carries none of the
// fields this integrator doesn't implement. It passes today (those keys
// don't exist yet) and is the tripwire for the day they do.
//
// Read at RUNTIME, not `import`ed — same reason as `manifestShape.test.ts`:
// the extracted art tree is gitignored and `.dockerignore`d, and the image's
// web build stage copies only `sts2-pycli/web/`, so a static import would
// fail `npm run build` inside the container. Absent tree => the suite
// skips; present tree => it checks.
const WEB_ROOT = resolve(__dirname, "../../../..");
const MANIFEST_PATH = resolve(WEB_ROOT, "../../sts2-wiki/site/public/images/vfx/manifest.json");

interface RawManifest {
  effects: Record<string, { kind: string; emitters?: { name: string; process?: Record<string, unknown> }[] }>;
}

function readManifest(): RawManifest | null {
  try {
    return JSON.parse(readFileSync(MANIFEST_PATH, "utf8")) as RawManifest;
  } catch {
    return null;
  }
}

const loaded = readManifest();
const suite = loaded ? describe : describe.skip;
const manifest = loaded as RawManifest;

// Every field simulate.ts's ParticleSystemSpec implements, or explicitly
// skips as a rendering-only concern (see that interface's module doc) — the
// complete set of `ParticleEmitterProcess` keys `manifestShape.test.ts`
// checks today. Anything outside this set showing up on a real emitter is
// physics simulate.ts does not implement.
const IMPLEMENTED_OR_INTENTIONALLY_SKIPPED = new Set([
  "direction",
  "spread",
  "flatness",
  "gravity",
  "initialVelocityMin",
  "initialVelocityMax",
  "lifetimeRandomness",
  "dampingMin",
  "dampingMax",
  "dampingCurve",
  "particleFlagDampingAsFriction",
  "particleFlagDisableZ",
  "radialVelocityMin",
  "radialVelocityMax",
  "radialVelocityCurve",
  "scaleMin",
  "scaleMax",
  "scaleCurve",
  "scaleOverVelocityMax",
  "scaleOverVelocityCurve",
  "turbulenceEnabled",
  "turbulenceInfluenceMin",
  "turbulenceInfluenceMax",
  "turbulenceInfluenceOverLifeCurve",
  "turbulenceInitialDisplacementMin", // measured ~0 for real emitters, but
  "turbulenceInitialDisplacementMax", // a real, implemented field (spawn nudge)
  "turbulenceNoiseScale",
  "turbulenceNoiseSpeed",
  "turbulenceNoiseSpeedRandom",
  "turbulenceNoiseStrength",
  // `colorInitialRamp`'s color IS rendering-only and unread here, same as
  // alphaCurve/colorRamp below — but unlike those two, its PRESENCE gates a
  // real spawn-time RNG draw this integrator must consume for stream
  // fidelity (see ParticleSystemSpec.colorInitialRamp's doc). It belongs in
  // this "implemented" set, not the rendering-only skip list below — round 2
  // of this task grouped it there and missed the draw as a result.
  "colorInitialRamp",
  // `animOffset*` is Godot's PARAM_ANIM_OFFSET: which frame of a flipbook
  // sheet a particle STARTS on. Like `colorInitialRamp` it costs a spawn-time
  // RNG draw, and `simulate.ts` already consumes that draw in stream position
  // (the literal `rng(); // animation_offset` line) — so stream fidelity is
  // not at risk here. Unlike `colorInitialRamp`, its VALUE is then a pure
  // rendering input (`INSTANCE_CUSTOM.z` in the flipbook shaders), which is
  // why it sits here rather than in the physics set above.
  //
  // NOT a blessing of the gap: until the renderer surfaces this value, every
  // particle of an emitter starts on the same frame and they animate in
  // lockstep. That is the flipbook half of task 15b, and it is the reason
  // this comment names the shader input rather than just saying "rendering".
  "animOffsetMin",
  "animOffsetMax",
  // `animSpeed*` is the OTHER half of that same shader input, and the reason
  // the note above is no longer a live gap: `CUSTOM.z` is
  // `animation_offset + lifetime_percent * animation_speed`
  // (`ppm451.cpp:1147`), so the offset alone picks a frame and never leaves
  // it. Its draw was likewise already consumed in stream position; the value
  // is now kept and resolved at readout, curve included (`*=`, where the
  // offset curve is `+=` — the asymmetry is Godot's).
  "animSpeedMin",
  "animSpeedMax",
  "animSpeedCurve",
  // `angle*` / `angularVelocity*` — PARAM_ANGLE and PARAM_ANGULAR_VELOCITY,
  // both DEGREES, both drawn in `calculate_initial_dynamics_params` as its
  // first two draws. Those draws were always consumed here; keeping the values
  // moved nothing in the stream. `readout` resolves them into `Particle.angle`
  // (Godot's `CUSTOM.x`, radians) and the draw path rotates the quad.
  "angleMin",
  "angleMax",
  "angleCurve",
  "angularVelocityMin",
  "angularVelocityMax",
  "angularVelocityCurve",
  // The emission VOLUME — `calculate_initial_position` (`ppm451.cpp:638-694`).
  // Every one of these was previously structurally absent from `ProcessSpec`,
  // which made `EMISSION_SHAPE_POINT` the only reachable branch and pinned
  // every particle of every emitter to the emitter origin. The manifest says
  // that was wrong for 29 of 69 emitters (12 Sphere, 3 Sphere Surface, 2 Box,
  // 12 Ring), so this is corrected physics, not a bookkeeping change.
  //
  // Each shape also draws a DIFFERENT number of times (Point 0, Sphere 3,
  // Sphere Surface 2, Box 3, Ring 3), so this is the one place in the spawn
  // path where the field VALUE selects the stream length rather than just
  // being consumed by it — `simulate.test.ts`'s per-shape draw-count test
  // pins that.
  "emissionShape",
  "emissionSphereRadius",
  "emissionBoxExtents",
  "emissionRingAxis",
  "emissionRingHeight",
  "emissionRingRadius",
  "emissionRingInnerRadius",
  "emissionRingConeAngle",
  // The trailing `pos * scale + offset` applies to EVERY shape including
  // Point, so these two are live even on a point emitter (10 and 7 real
  // emitters author them non-default respectively).
  "emissionShapeScale",
  "emissionShapeOffset",
  // Rendering-only — intentionally out of this integrator's scope, not
  // unimplemented physics, and unlike colorInitialRamp above, carries no RNG
  // side effect at all.
  "alphaCurve",
  "colorRamp",
  // Implemented, but by the RENDERER (`particleDrawAngle`, GH #689): the
  // integrator tracks `Particle.heading` for every particle, so the flag
  // itself gates no physics and never needs to reach the spec.
  "particleFlagAlignY",
  "particleFlagRotateY",
]);

suite("simulate.ts's ParticleSystemSpec coverage of the real manifest", () => {
  it("no emitter's process carries a field the integrator doesn't implement", () => {
    let checked = 0;
    for (const [effectName, effect] of Object.entries(manifest.effects)) {
      if (effect.kind !== "particles" || !effect.emitters) continue;
      for (const emitter of effect.emitters) {
        if (!emitter.process) continue;
        checked++;
        for (const field of Object.keys(emitter.process)) {
          expect(
            IMPLEMENTED_OR_INTENTIONALLY_SKIPPED.has(field),
            `${effectName}.${emitter.name}.process.${field}: simulate.ts's ` +
              `ParticleSystemSpec does not implement this field yet — a ` +
              `real emitter now authors it, so this effect will render with ` +
              `that physics silently missing until simulate.ts is extended.`,
          ).toBe(true);
        }
      }
    }
    // A manifest with zero particle emitters would make the loop above
    // vacuously true.
    expect(checked).toBeGreaterThan(0);
  });
});
