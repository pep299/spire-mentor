import { describe, it, expect } from "vitest";
import { advanceTo, createSimState, readout, simulate, stepN } from "../simulate";
import type { ParticleSystemSpec } from "../simulate";
import { makeRng } from "../rng";

/** A flat, constant LUT — `sampleLut` returns this value for every `t`. */
function flat(value: number): number[] {
  return [value, value];
}

/** Every field at Godot's own zero/neutral default: a single particle,
 *  spawning immediately at the origin, moving along +x, subject to no
 *  forces, no curves, no turbulence. Individual tests override only the
 *  fields their scenario needs. */
const BARE: ParticleSystemSpec = {
  amount: 1,
  cpu: false,
  explosiveness: 0,
  randomness: 0,
  fixedFps: 60,
  lifetime: 1,
  localCoords: false,
  offset: [0, 0],
  direction: [1, 0, 0],
  spread: 0,
  flatness: 0,
  gravity: [0, 0, 0],
  initialVelocityMin: 0,
  initialVelocityMax: 0,
  lifetimeRandomness: 0,
  animOffsetMin: 0,
  animOffsetMax: 0,
  animSpeedMin: 0,
  animSpeedMax: 0,
  animSpeedCurve: null,
  angleMin: 0,
  angleMax: 0,
  angleCurve: null,
  angularVelocityMin: 0,
  angularVelocityMax: 0,
  angularVelocityCurve: null,
  emissionShape: 0,
  emissionSphereRadius: 1,
  emissionBoxExtents: [1, 1, 1],
  emissionRingAxis: [0, 0, 1],
  emissionRingHeight: 1,
  emissionRingRadius: 1,
  emissionRingInnerRadius: 0,
  emissionRingConeAngle: 90,
  emissionShapeOffset: [0, 0, 0],
  emissionShapeScale: [1, 1, 1],
  colorInitialRamp: null,
  dampingMin: 0,
  dampingMax: 0,
  dampingCurve: null,
  particleFlagDampingAsFriction: true,
  particleFlagDisableZ: true,
  radialVelocityMin: 0,
  radialVelocityMax: 0,
  radialVelocityCurve: null,
  scaleMin: 1,
  scaleMax: 1,
  scaleCurve: null,
  scaleOverVelocityMax: 0,
  scaleOverVelocityCurve: null,
  turbulenceEnabled: false,
  turbulenceInfluenceMin: 0,
  turbulenceInfluenceMax: 0,
  turbulenceInfluenceOverLifeCurve: null,
  turbulenceInitialDisplacementMin: 0,
  turbulenceInitialDisplacementMax: 0,
  turbulenceNoiseScale: 1,
  turbulenceNoiseSpeed: [0, 0, 0],
  turbulenceNoiseSpeedRandom: 0,
  turbulenceNoiseStrength: 1,
};

// vfx_attack_blunt's "vfx_common_specks" emitter (sts2-wiki/site/public/
// images/vfx/manifest.json) — the real authored parameter set turbulence.ts
// and emit.ts were transcribed against. Curve fields are representative
// fixture LUTs, not the manifest's actual baked bytes (this module consumes
// already-loaded LUTs, not `vfx_particles/<hash>.json` paths — see
// simulate.ts's module doc), matching turbulence.test.ts's own convention.
const BLUNT_SPECKS: ParticleSystemSpec = {
  amount: 6,
  cpu: false,
  explosiveness: 1.0,
  randomness: 0.5,
  fixedFps: 60,
  lifetime: 1.0,
  localCoords: false,
  offset: [0, 0],
  direction: [-1, 0, 0],
  spread: 180,
  flatness: 1.0,
  gravity: [0, 0, 0],
  initialVelocityMin: 500,
  initialVelocityMax: 2000,
  lifetimeRandomness: 0.8,
  animOffsetMin: 0,
  animOffsetMax: 0,
  animSpeedMin: 0,
  animSpeedMax: 0,
  animSpeedCurve: null,
  angleMin: 0,
  angleMax: 0,
  angleCurve: null,
  angularVelocityMin: 0,
  angularVelocityMax: 0,
  angularVelocityCurve: null,
  emissionShape: 0,
  emissionSphereRadius: 1,
  emissionBoxExtents: [1, 1, 1],
  emissionRingAxis: [0, 0, 1],
  emissionRingHeight: 1,
  emissionRingRadius: 1,
  emissionRingInnerRadius: 0,
  emissionRingConeAngle: 90,
  emissionShapeOffset: [0, 0, 0],
  emissionShapeScale: [1, 1, 1],
  // The real emitter's is a 256-stop [1,1,1,1] gradient (a no-op tint) —
  // presence is what matters here (it gates a real spawn-time RNG draw this
  // module never uses the resulting color for), not the exact stops.
  colorInitialRamp: [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
  ],
  dampingMin: 0.26,
  dampingMax: 0.26,
  dampingCurve: [1, 0.6, 0.2, 0],
  particleFlagDampingAsFriction: true,
  particleFlagDisableZ: true,
  radialVelocityMin: 150,
  radialVelocityMax: 250,
  radialVelocityCurve: [0, 0.5, 1, 0.5, 0],
  scaleMin: 0.15,
  scaleMax: 0.25,
  scaleCurve: [1, 0.8, 0.4, 0],
  scaleOverVelocityMax: 670.71,
  scaleOverVelocityCurve: [0.5, 1, 1.5],
  turbulenceEnabled: true,
  turbulenceInfluenceMin: 0.65,
  turbulenceInfluenceMax: 0.85,
  turbulenceInfluenceOverLifeCurve: null,
  // Measured (see turbulence.ts's module doc): ~1.49e-6 for every real
  // turbulence-enabled emitter — effectively zero, but a real field.
  turbulenceInitialDisplacementMin: 1.49012e-6,
  turbulenceInitialDisplacementMax: 1.49012e-6,
  turbulenceNoiseScale: 10.1,
  turbulenceNoiseSpeed: [0.1, 0.1, 0.1],
  turbulenceNoiseSpeedRandom: 1.0,
  turbulenceNoiseStrength: 5.0,
};

/** Deep-clone through JSON so `toEqual` compares plain data, matching how
 *  Task 13's tests compare structured results. */
function snapshot<T>(value: T): unknown {
  return JSON.parse(JSON.stringify(value));
}

describe("simulate / stepN", () => {
  it("a particle with no forces travels at its initial velocity", () => {
    const p = stepN({ ...BARE, initialVelocityMin: 100, initialVelocityMax: 100 }, 30);
    expect(p.x).toBeCloseTo(50, 0); // 100 px/s for 0.5s at 60fps
  });

  it("damping as friction decelerates rather than clamping", () => {
    const p = stepN(
      { ...BARE, initialVelocityMin: 1000, initialVelocityMax: 1000, dampingMin: 0.26, dampingMax: 0.26 },
      30,
    );
    expect(p.vx).toBeLessThan(1000);
    expect(p.vx).toBeGreaterThan(0);
  });

  it("scale multiplies base, curve and over-velocity together", () => {
    // Three independent factors. A test that only checks the product is 1 at
    // t=0 would pass with any two of them dropped.
    const p = stepN(
      { ...BARE, scaleMin: 2, scaleMax: 2, scaleCurve: flat(0.5), scaleOverVelocityCurve: flat(3) },
      1,
    );
    expect(p.scale).toBeCloseTo(3, 5);
  });

  it("get_random_direction_from_spread covers the full range at spread=180 (vfx_common_specks's authored value)", () => {
    // spread=180 is a real authored value (vfx_attack_blunt's specks) against
    // Godot's default of 45 — every other test here uses spread: 0, which a
    // wrong (too-narrow, ignored, or mis-centered) spread implementation
    // would still pass. This exercises the real transcribed
    // get_random_direction_from_spread with a spread wide enough that a
    // wrong distribution is visible: at 180 degrees (a full circle in
    // Godot's 2D convention — see the module doc's angle1_rad formula),
    // roughly half of a large sample should fly backward relative to
    // `direction`.
    const N = 200;
    const result = simulate(
      {
        ...BARE,
        amount: N,
        explosiveness: 1,
        spread: 180,
        direction: [1, 0, 0],
        initialVelocityMin: 100,
        initialVelocityMax: 100,
      },
      1 / 60, // one frame: velocity direction is set at spawn and BARE has no
      // forces/damping/turbulence to perturb it before this reads vx/vy.
      makeRng(7),
    );
    const angles = result.particles.map((p) => Math.atan2(p.vy, p.vx));
    // direction=[1,0,0] with no spread would put every angle at 0 — a
    // too-narrow or ignored spread would fail these bounds.
    expect(Math.min(...angles)).toBeLessThan(-Math.PI / 2);
    expect(Math.max(...angles)).toBeGreaterThan(Math.PI / 2);
    const backward = result.particles.filter((p) => p.vx < 0).length;
    expect(backward).toBeGreaterThan(N * 0.25);
    expect(backward).toBeLessThan(N * 0.75);
  });

  it("respects each emitter's own fixed_fps", () => {
    // Measured: 24 scenes at 60, one at 25, 11 unset (Godot default 30).
    const at30 = simulate({ ...BARE, fixedFps: 30 }, 1.0, makeRng(1));
    const at60 = simulate({ ...BARE, fixedFps: 60 }, 1.0, makeRng(1));
    expect(at60.steps).toBe(2 * at30.steps);
  });

  it("is fully determined by its seed", () => {
    const a = simulate(BLUNT_SPECKS, 0.5, makeRng(9));
    const b = simulate(BLUNT_SPECKS, 0.5, makeRng(9));
    expect(snapshot(a)).toEqual(snapshot(b));
    // A weak-check guard: JSON round-tripping turns NaN into null, which
    // would make a broken (NaN-producing) run look "deterministic" too.
    for (const particle of a.particles) {
      for (const v of [particle.x, particle.y, particle.z, particle.vx, particle.vy, particle.vz, particle.scale]) {
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  it("draws a config-fixed number of RNG samples per particle, independent of its runtime trajectory", () => {
    // The invariant this module's shared, sequentially-advancing RNG stream
    // depends on: two particles under the SAME emitter config must consume
    // the SAME number of draws, even if their own (randomly drawn) physics
    // send them down different runtime paths. A particle that stays exactly
    // on process_radial_displacement's pivot for its whole life used to draw
    // one extra RNG sample EVERY FRAME it lingered there (fixed in this
    // round) — this test would have failed against that pre-fix behavior:
    // a stationary particle draws ~30 extra samples over 0.5s at 60fps,
    // while a fast-moving one draws only the one unavoidable at spawn.
    function countDraws(overrides: Partial<ParticleSystemSpec>): number {
      let count = 0;
      const base = makeRng(3);
      const counting = (): number => {
        count++;
        return base();
      };
      simulate({ ...BARE, ...overrides }, 0.5, counting);
      return count;
    }
    // Stays exactly at the pivot for the whole run: zero initial velocity,
    // no gravity/damping/turbulence to move it — process_radial_displacement's
    // "at pivot" branch re-triggers every single frame.
    const pivotBound = countDraws({ initialVelocityMin: 0, initialVelocityMax: 0 });
    // Leaves the pivot on frame 1 (100px/s at 60fps moves it ~1.7px, well
    // past the 0.01 threshold) and never returns.
    const movingAway = countDraws({ initialVelocityMin: 100, initialVelocityMax: 100 });
    expect(pivotBound).toBe(movingAway);
  });

  it("keeping the angle/anim_speed draws moves NOTHING else in the stream", () => {
    // The load-bearing claim behind implementing tumble and anim rate at all:
    // `calculate_initial_dynamics_params` always drew `angle` and
    // `angular_velocity` as its first two samples, and
    // `calculate_initial_display_params` always drew `anim_speed`, and this
    // module always consumed those draws to hold stream position. Keeping the
    // VALUES therefore cannot shift any later draw. Pinned by asserting that
    // an emitter authoring all three lands its particle in exactly the same
    // PLACE as one authoring none — position is downstream of every draw that
    // follows, so it is the sensitive witness.
    const authored = simulate(
      { ...BARE, angleMin: -180, angleMax: 180, angularVelocityMin: -360, angularVelocityMax: 360, animSpeedMin: 1, animSpeedMax: 6 },
      0.5,
      makeRng(7),
    );
    const bare = simulate(BARE, 0.5, makeRng(7));
    expect(authored.particles[0]!.x).toBeCloseTo(bare.particles[0]!.x, 12);
    expect(authored.particles[0]!.y).toBeCloseTo(bare.particles[0]!.y, 12);
    expect(authored.particles[0]!.scale).toBeCloseTo(bare.particles[0]!.scale, 12);
  });

  it("draws exactly as many samples as its emission shape's branch does", () => {
    // `calculate_initial_position` (`ppm451.cpp:638-694`) is the ONE place
    // where a spec VALUE picks how long the RNG stream is, rather than just
    // being consumed by it: each shape's branch is emitted with a different
    // number of `rand_from_seed` calls. Get one wrong and every draw after
    // it — spread, damping, scale, radial — shifts for that emitter only,
    // which reads as "this one effect looks scrambled" rather than as a
    // stream bug.
    function countDraws(overrides: Partial<ParticleSystemSpec>): number {
      let count = 0;
      const base = makeRng(3);
      const counting = (): number => {
        count++;
        return base();
      };
      simulate({ ...BARE, ...overrides }, 0.5, counting);
      return count;
    }
    const point = countDraws({ emissionShape: 0 });
    // Sphere: s, t, p (`ppm451.cpp:645-647`).
    expect(countDraws({ emissionShape: 1 })).toBe(point + 3);
    // Sphere Surface: s, t — no `mix` factor (`ppm451.cpp:654-655`).
    expect(countDraws({ emissionShape: 2 })).toBe(point + 2);
    // Box: one per axis (`ppm451.cpp:660`).
    expect(countDraws({ emissionShape: 3 })).toBe(point + 3);
    // Points / Directed Points sample a texture the extractor never carries,
    // so their branch is unreachable here and draws nothing — see
    // `calculateInitialPosition`'s doc for the upstream draw they'd owe.
    expect(countDraws({ emissionShape: 4 })).toBe(point);
    expect(countDraws({ emissionShape: 5 })).toBe(point);
    // Ring: y_pos, ring_spawn_angle, ring_random_radius
    // (`ppm451.cpp:670,674,675`).
    expect(countDraws({ emissionShape: 6 })).toBe(point + 3);
  });

  it("spawns particles across the emission volume, not all on the origin", () => {
    // The regression this whole shape port exists to fix: with no shape field
    // at all, `EMISSION_SHAPE_POINT` was the only reachable branch, so all 29
    // of the manifest's non-point emitters stacked every particle on the
    // emitter origin.
    const N = 200;
    const r = simulate(
      { ...BARE, amount: N, explosiveness: 1, emissionShape: 1, emissionSphereRadius: 40 },
      1 / 60,
      makeRng(5),
    );
    expect(r.particles.length).toBe(N);
    const radii = r.particles.map((q) => Math.hypot(q.x, q.y, q.z));
    // Not a point: something is out near the rim, and the cloud is spread.
    expect(Math.max(...radii)).toBeGreaterThan(20);
    expect(Math.min(...radii)).toBeLessThan(10);
    // And nothing escapes the sphere. `mix(vec3(0), rim, p)` with p in [0,1)
    // is a strict interior sample, so the radius is a hard bound.
    expect(Math.max(...radii)).toBeLessThanOrEqual(40);
  });

  it("puts every SPHERE_SURFACE particle ON the sphere, not inside it", () => {
    // The `mix(..., p)` third draw is exactly what separates shape 1 from
    // shape 2 (`ppm451.cpp:648` vs `ppm451.cpp:657`). Dropping it from the
    // Sphere branch — or adding it to the Surface branch — would leave the
    // draw COUNT test above passing while silently swapping the two volumes.
    const N = 100;
    const r = simulate(
      {
        ...BARE,
        amount: N,
        explosiveness: 1,
        particleFlagDisableZ: false, // else `TRANSFORM[3].z = 0` flattens the
        // shell onto XY and the radius stops being constant — see the
        // projection test below, which is the case 28 of 29 real emitters hit.
        emissionShape: 2,
        emissionSphereRadius: 40,
      },
      1 / 60,
      makeRng(5),
    );
    for (const q of r.particles) {
      expect(Math.hypot(q.x, q.y, q.z)).toBeCloseTo(40, 4);
    }
  });

  it("applies emission_shape_scale and _offset even to a POINT emitter", () => {
    // `return pos * emission_shape_scale + emission_shape_offset` is OUTSIDE
    // the shape switch (`ppm451.cpp:697`), so it is live on the 40 point
    // emitters too — 7 of the manifest's emitters author a non-default offset
    // and 10 a non-default scale.
    const r = simulate(
      { ...BARE, emissionShape: 0, emissionShapeOffset: [7, -3, 0], emissionShapeScale: [2, 2, 2] },
      1 / 60,
      makeRng(5),
    );
    // pos is the origin for POINT, so scale is a no-op and offset is the whole
    // displacement.
    expect(r.particles[0]!.x).toBeCloseTo(7, 6);
    expect(r.particles[0]!.y).toBeCloseTo(-3, 6);
  });

  it("scales the emission volume before offsetting it, not after", () => {
    // Order matters: `pos * scale + offset`. Swapping them would move a
    // radius-40 sphere's centre to (14, 0) instead of (7, 0), which the
    // POINT test above cannot see (its `pos` is zero, so both orders agree).
    const N = 200;
    const spec = {
      ...BARE,
      amount: N,
      explosiveness: 1,
      particleFlagDisableZ: false,
      emissionShape: 2,
      emissionSphereRadius: 40,
      emissionShapeScale: [2, 2, 2] as const,
      emissionShapeOffset: [7, 0, 0] as const,
    };
    const r = simulate(spec, 1 / 60, makeRng(5));
    for (const q of r.particles) {
      // Every particle sits on a radius-80 shell CENTRED on the offset.
      expect(Math.hypot(q.x - 7, q.y, q.z)).toBeCloseTo(80, 3);
    }
  });

  it("flattens the emission volume onto XY when disable_z is set", () => {
    // `TRANSFORM[3].z = 0.0` at the end of the start block
    // (`ppm451.cpp:921`) — so the 3D volume is PROJECTED, not intersected.
    // This is the case that actually ships: 28 of the manifest's 29
    // non-point emitters set the flag, so their spheres and rings reach the
    // screen as filled discs and annuli. A port that instead re-normalised
    // the position back onto the sphere would look right in isolation and
    // wrong here.
    const N = 200;
    const r = simulate(
      { ...BARE, amount: N, explosiveness: 1, emissionShape: 2, emissionSphereRadius: 40 },
      1 / 60,
      makeRng(5),
    );
    const radii = r.particles.map((q) => Math.hypot(q.x, q.y));
    for (const q of r.particles) expect(q.z).toBe(0);
    // A projected shell fills the disc rather than hugging its rim.
    expect(Math.max(...radii)).toBeGreaterThan(35);
    expect(Math.min(...radii)).toBeLessThan(15);
    expect(Math.max(...radii)).toBeLessThanOrEqual(40 + 1e-9);
  });

  it("lays a RING out in the plane its axis is normal to", () => {
    // The rotation-matrix half of the ring branch (`ppm451.cpp:676-694`) is
    // the most intricate thing in `calculate_initial_position`, and a wrong
    // ortho_axis still produces a plausible-looking ring — just the wrong
    // one. With a zero-height ring on the default z axis, every particle must
    // land at z = 0 with its radius between the inner and outer radii.
    const N = 100;
    const r = simulate(
      {
        ...BARE,
        amount: N,
        explosiveness: 1,
        particleFlagDisableZ: false, // otherwise z is forced to 0 regardless
        emissionShape: 6,
        emissionRingAxis: [0, 0, 1],
        emissionRingRadius: 30,
        emissionRingInnerRadius: 20,
        emissionRingHeight: 0,
        emissionRingConeAngle: 90,
      },
      1 / 60,
      makeRng(5),
    );
    for (const q of r.particles) {
      expect(q.z).toBeCloseTo(0, 6);
      const radius = Math.hypot(q.x, q.y);
      expect(radius).toBeGreaterThanOrEqual(20 - 1e-6);
      expect(radius).toBeLessThanOrEqual(30 + 1e-6);
    }
    // And it is a ring, not a single spoke.
    const angles = r.particles.map((q) => Math.atan2(q.y, q.x));
    expect(Math.max(...angles) - Math.min(...angles)).toBeGreaterThan(Math.PI);
  });

  it("swings the RING into the plane of a non-default axis", () => {
    // Same ring, axis rotated onto x: the flat plane must follow the axis
    // (x = 0 now), which is what actually exercises the Rodrigues rotation.
    const r = simulate(
      {
        ...BARE,
        amount: 60,
        explosiveness: 1,
        particleFlagDisableZ: false,
        emissionShape: 6,
        emissionRingAxis: [1, 0, 0],
        emissionRingRadius: 30,
        emissionRingInnerRadius: 30,
        emissionRingHeight: 0,
        emissionRingConeAngle: 90,
      },
      1 / 60,
      makeRng(5),
    );
    for (const q of r.particles) {
      expect(q.x).toBeCloseTo(0, 6);
      expect(Math.hypot(q.y, q.z)).toBeCloseTo(30, 4);
    }
  });

  it("uses CPUParticles2D's own emission formula, not the shader's", () => {
    // `CPUParticles2D` is a separate implementation that merely shares the
    // property NAMES. Its sphere is two draws of a FLAT disc, uniform in
    // radius (`cpu_particles_2d.cpp:893-897`); the shader's is three draws of
    // a 3D ball (`ppm451.cpp:645-648`). `vfx_monster_death`'s debris burst is
    // the pck's one `CPUParticles2D`, and running it through the GPU formula
    // would be plausible-looking invented physics — the exact failure this
    // port exists to avoid.
    function countDraws(overrides: Partial<ParticleSystemSpec>): number {
      let count = 0;
      const base = makeRng(3);
      const counting = (): number => {
        count++;
        return base();
      };
      simulate({ ...BARE, ...overrides }, 0.5, counting);
      return count;
    }
    const point = countDraws({ cpu: true, emissionShape: 0 });
    expect(countDraws({ cpu: true, emissionShape: 1 })).toBe(point + 2); // t, radius
    expect(countDraws({ cpu: true, emissionShape: 2 })).toBe(point + 2); // s, t
    // ...against the GPU node's three for the same shape.
    expect(countDraws({ cpu: false, emissionShape: 1 })).toBe(point + 3);
  });

  it("keeps a CPU emitter's particles flat, whatever disable_z says", () => {
    // `p.transform[2]` is a `Vector2` — a `CPUParticles2D` has no z to place a
    // particle on, so the flatness is structural rather than the GPU node's
    // post-hoc `TRANSFORM[3].z = 0`. Asserted with `disable_z` OFF, which is
    // how the real burst is authored: under the GPU formula that same config
    // scatters particles through a 3D ball.
    const r = simulate(
      {
        ...BARE,
        cpu: true,
        amount: 100,
        explosiveness: 1,
        particleFlagDisableZ: false,
        emissionShape: 1,
        emissionSphereRadius: 48,
      },
      1 / 60,
      makeRng(5),
    );
    for (const q of r.particles) {
      expect(q.z).toBe(0);
      expect(Math.hypot(q.x, q.y)).toBeLessThanOrEqual(48);
    }
    // A real disc, not a point: the burst's whole purpose is to spread over
    // the dying creature.
    expect(Math.max(...r.particles.map((q) => Math.hypot(q.x, q.y)))).toBeGreaterThan(24);
  });

  it("ignores emission_shape_scale/_offset on a CPU emitter", () => {
    // Neither string appears anywhere in `cpu_particles_2d.cpp` — the node has
    // no such property, so the trailing `pos * scale + offset` the shader
    // applies has no counterpart here. Carrying it over would displace the
    // burst by a field the game cannot read.
    const r = simulate(
      {
        ...BARE,
        cpu: true,
        emissionShape: 0,
        emissionShapeOffset: [500, 500, 0],
        emissionShapeScale: [9, 9, 9],
      },
      1 / 60,
      makeRng(5),
    );
    expect(r.particles[0]!.x).toBeCloseTo(0, 6);
    expect(r.particles[0]!.y).toBeCloseTo(0, 6);
  });

  it("spins a particle by angular_velocity, in degrees per second", () => {
    // `base_angle += CUSTOM.y * LIFETIME * angular_velocity` then
    // `CUSTOM.x = base_angle * degree_to_rad` (`ppm451.cpp:1079-1081`). A fixed
    // 90 deg/s over half of a 1s lifetime is 45 degrees.
    const r = simulate(
      { ...BARE, angularVelocityMin: 90, angularVelocityMax: 90 },
      0.5,
      makeRng(3),
    );
    expect(r.particles[0]!.angle).toBeCloseTo((45 * Math.PI) / 180, 6);
  });

  it("starts a particle at its authored angle, converted to radians", () => {
    // `angle_min == angle_max` removes the draw's randomness and
    // `angular_velocity` stays 0, so nothing accumulates and this pins the
    // UNIT alone: 180 degrees must arrive as PI radians, not as 180. Sampled
    // one frame in rather than at 0 — a particle has not spawned at 0.
    const r = simulate({ ...BARE, angleMin: 180, angleMax: 180 }, 1 / 60, makeRng(3));
    expect(r.particles[0]!.angle).toBeCloseTo(Math.PI, 6);
  });

  it("leaves every particle unrotated when nothing authors a rotation", () => {
    // Godot's default for both params is 0 (`ppm451.cpp:2372-2390`), and most
    // emitters leave them there — those must stay axis-aligned exactly as they
    // drew before rotation existed.
    const r = simulate(BARE, 0.5, makeRng(3));
    expect(r.particles[0]!.angle).toBe(0);
  });

  it("advances the flipbook phase by anim_speed instead of holding one frame", () => {
    // `CUSTOM.z = animation_offset + lifetime_percent * animation_speed`
    // (`ppm451.cpp:1147`). With offset pinned to 0 and speed to 4, the phase at
    // half life is 2 — where before it was 0 for the particle's whole life.
    const spec = { ...BARE, animOffsetMin: 0, animOffsetMax: 0, animSpeedMin: 4, animSpeedMax: 4 };
    const half = simulate(spec, 0.5, makeRng(3));
    expect(half.particles[0]!.animOffset).toBeCloseTo(0.5 * 4, 6);
    // ...and it is genuinely advancing, not a constant: one frame in, the
    // phase is one frame's worth, not half a lifetime's.
    const oneFrame = simulate(spec, 1 / 60, makeRng(3));
    expect(oneFrame.particles[0]!.animOffset).toBeCloseTo((1 / 60) * 4, 6);
  });

  it("holds the phase at the offset when anim_speed is left at zero", () => {
    // The default, and most emitters — the sheet is a per-particle variety
    // PICK, not an animation, and must not start creeping.
    const spec = { ...BARE, animOffsetMin: 0.25, animOffsetMax: 0.25 };
    expect(simulate(spec, 1 / 60, makeRng(3)).particles[0]!.animOffset).toBeCloseTo(0.25, 6);
    expect(simulate(spec, 0.5, makeRng(3)).particles[0]!.animOffset).toBeCloseTo(0.25, 6);
  });

  it("local_coords moves live particles with the emitter; world coords does not", () => {
    // Set on 20 of 36 emitters, unset (Godot default false) on the rest — a real
    // authored distinction, and only observable once the emitter moves.
    const local = simulate({ ...BARE, localCoords: true }, 0.5, makeRng(1), { moveBy: [100, 0] });
    const world = simulate({ ...BARE, localCoords: false }, 0.5, makeRng(1), { moveBy: [100, 0] });
    expect(local.particles[0].x).toBeCloseTo(world.particles[0].x + 100, 5);
  });

  it("retires a particle at the end of its life", () => {
    expect(simulate({ ...BARE, lifetime: 0.5 }, 0.6, makeRng(1)).alive).toBe(0);
  });
});

describe("heading (particle_flag_align_y)", () => {
  it("follows the direction of travel", () => {
    const p = stepN({ ...BARE, direction: [0.6, -0.8, 0], initialVelocityMin: 100, initialVelocityMax: 100 }, 5);
    expect(p.heading[0]).toBeCloseTo(0.6, 9);
    expect(p.heading[1]).toBeCloseTo(-0.8, 9);
  });

  it("turns with gravity rather than keeping the launch direction", () => {
    const p = stepN(
      { ...BARE, gravity: [0, 400, 0], initialVelocityMin: 100, initialVelocityMax: 100 },
      30,
    );
    // vx stays 100, vy has grown to ~200 — the heading is the CURRENT velocity.
    expect(p.heading[1] / p.heading[0]).toBeCloseTo(p.vy / p.vx, 6);
    expect(p.heading[1]).toBeGreaterThan(0.8);
  });

  it("holds the last direction once damping has stopped the particle", () => {
    // Linear damping clamps the speed to exactly zero (`v < 0` -> 0); Godot's
    // zero-velocity branch re-normalizes the previous TRANSFORM[1] rather than
    // snapping back to identity, so a speck that skids to a halt stays
    // pointing the way it was going.
    const p = stepN(
      {
        ...BARE,
        direction: [1, 0, 0],
        initialVelocityMin: 50,
        initialVelocityMax: 50,
        dampingMin: 1000,
        dampingMax: 1000,
        particleFlagDampingAsFriction: false,
      },
      30,
    );
    expect(p.vx).toBe(0);
    expect(p.heading).toEqual([1, 0]);
  });

  it("is the spawn identity for a particle that never moved", () => {
    const p = stepN(BARE, 5);
    expect(p.heading).toEqual([0, 1]);
  });

  it("counts radial displacement, which lives in final_velocity, not VELOCITY", () => {
    // A point-emitted particle offset from the pivot with no initial velocity moves
    // ONLY by radial displacement; a heading taken from `vx/vy` would stay at
    // identity while the particle visibly flies outward.
    const p = stepN(
      {
        ...BARE,
        emissionShapeOffset: [10, 0, 0],
        radialVelocityMin: 100,
        radialVelocityMax: 100,
      },
      5,
    );
    expect(p.vx).toBe(0);
    expect(p.heading[0]).toBeCloseTo(1, 6);
    expect(p.heading[1]).toBeCloseTo(0, 6);
  });
});

describe("createSimState / advanceTo / readout", () => {
  // THE property this refactor rests on. `simulate()` used to be re-run from
  // t=0 on every display frame; now one `SimState` per emitter is advanced
  // instead. That is only safe if advancing is exactly a replay -- same
  // frames, same order, same RNG draws -- so it is asserted directly rather
  // than inferred from the physics tests above happening to still pass.
  it("advancing in many small increments equals one jump to the end", () => {
    const jumped = createSimState(BLUNT_SPECKS, makeRng(9));
    advanceTo(jumped, 0.5);

    const stepped = createSimState(BLUNT_SPECKS, makeRng(9));
    // Deliberately NOT frame-aligned: 1/144 is a 144 Hz display clock driving
    // a 60 Hz simulation, which is exactly what the real draw loop does.
    for (let i = 1; i <= 72; i++) advanceTo(stepped, i / 144);

    expect(snapshot(readout(stepped))).toEqual(snapshot(readout(jumped)));
  });

  it("reproduces `simulate` exactly, which is what keeps the suite above an oracle", () => {
    const state = createSimState(BLUNT_SPECKS, makeRng(9));
    advanceTo(state, 0.5);
    expect(snapshot(readout(state))).toEqual(snapshot(simulate(BLUNT_SPECKS, 0.5, makeRng(9))));
  });

  it("is forward-only: rewinding is a no-op, not a re-simulation", () => {
    // The draw loop's clock is monotonic, but a stale `elapsed` must never
    // silently produce a DIFFERENT past than the one already integrated.
    const state = createSimState(BLUNT_SPECKS, makeRng(9));
    advanceTo(state, 0.5);
    const after = snapshot(readout(state));
    advanceTo(state, 0.2);
    expect(snapshot(readout(state))).toEqual(after);
  });

  it("carries `moveBy` at read-out, not into the integration", () => {
    // Read-out must stay a pure projection of the state: the draw loop calls
    // it once per frame on a state it keeps, so an offset that accumulated
    // would walk every particle off screen.
    const state = createSimState({ ...BARE, localCoords: true }, makeRng(1));
    advanceTo(state, 0.5);
    const moved = readout(state, { moveBy: [100, 0] });
    const still = readout(state);
    expect(moved.particles[0].x - still.particles[0].x).toBeCloseTo(100, 6);
    expect(snapshot(readout(state))).toEqual(snapshot(still));
  });
});
