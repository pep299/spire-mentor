/**
 * The particle draw path: turning a `readout()` result into textured, tinted
 * quads on screen.
 *
 * ## The shader family, and what is and is not ported
 *
 * Design spec §4.1: the materials in use are one idiom — a grayscale texture
 * used as a mask, colourised through a 1-D LUT, times the particle's vertex
 * colour:
 *
 * ```glsl
 * COLOR = vec4((lut(tex.r) * vertex_color).rgb, vertex_color.a * tex.r);
 * ```
 *
 * `lut` is the MATERIAL's own colorization gradient — a `.tres` shader
 * parameter, distinct from the per-particle-system `Curve`/`Gradient`
 * resources on `ParticleEmitterProcess`. It IS extracted now
 * (`particle_scene.py`'s `parse_material`, surfaced as
 * `ParticleEmitter.material`), and the fragment shader below samples it for
 * real, per fragment. 14 of the 47 LUTs in use exceed 1.0 — one reaches
 * `(12, 8, 8)` — so each is normalised by its peak into an 8-bit texture and
 * scaled back in the shader (`createLutTexture`/`lutScaleOf`): this canvas is
 * WebGL1, where float textures need an extension nothing guarantees, and
 * clamping would flatten exactly the glow the overbright exists for.
 *
 * The mask channel is keyed on the shader NAME, not guessed, and it is a
 * per-shader FACT rather than a two-family split: of the 11 shaders the
 * committed manifest actually uses, 5 mask on RED, 3 on ALPHA, 2 on GREEN
 * and 1 on GREEN×ALPHA (`MASK_CHANNELS`/`maskChannelOf`, each entry quoting
 * the `.gdshader` line it came from). The
 * flipbook family's sheet indexing is ported too — `flipbook.gdshaderinc`'s
 * UV math reduces algebraically to a row-major cell lookup for UVs in
 * [0, 1], so `flipbookCell` computes the cell CPU-side and the quad carries
 * that sub-rect. Six of the twelve flipbook materials assign no
 * `flipbook_curve`; the shader declares it `hint_default_black`, so an
 * unassigned sampler reads 0 and those sheets pick ONE frame from the
 * particle's own anim offset and hold it — a variety picker, not a missing
 * animation, and `flipbookCell` renders it as such.
 *
 * **What still does NOT render, measured across the 64 emitters and 20
 * distinct materials in use:**
 *
 * - `use_erosion_texture`: enabled on **0** emitters. The uniform is
 *   extracted so a scene that turns it on becomes visible rather than
 *   silently wrong.
 * - `use_hue_shift`: **3** emitters, all in `vfx_fire_burst`, which render
 *   without the per-particle hue rotation.
 * - `vfx_ring_polar_shader`: **1** emitter. It carries no LUT at all and
 *   uses a polar erosion smoothstep this port does not implement.
 * - `process.colorInitialRamp`'s per-particle multiplier (13 of 64 emitters
 *   set it): `simulate.ts` consumes the RNG draw that would pick a sample
 *   from it — for stream-position fidelity — but never surfaces the
 *   resulting index, so no particle-level seed survives for a renderer to
 *   sample the ramp with. See `ParticleSystemSpec.colorInitialRamp`'s doc.
 *
 * `material === null` ("a plain textured quad with no material override" —
 * `manifest.ts`'s own doc on `ParticleEmitter.material`) is not a gap: those
 * emitters use the texture's own RGBA untouched, tinted by the vertex
 * colour, which is what the real game does for them too, since there is no
 * material or shader in play at all.
 */

import { lutSamplePoint, sampleLut } from "./lut";
import type { ParticleEmitter, ParticleEmitterProcess, ParticleMaterial } from "../manifest";
import type { Particle, ParticleSystemSpec } from "./simulate";

// ---------------------------------------------------------------------------
// Vertex colour (CPU-side). This is the `vertex_color` the fragment shader
// multiplies the material's own LUT sample by — two different gradients, one
// per particle and one per fragment, not one applied twice.
// ---------------------------------------------------------------------------

/** Bilinearly-interpolated RGBA sample from a baked `colorRamp` LUT (256
 *  `[r, g, b, a]` stops, Task 2's `bake_gradient`) — the colour analogue of
 *  `lut.ts`'s `sampleLut`, sharing its endpoint-inclusive index math. */
export function sampleColorLut(
  lut: readonly (readonly [number, number, number, number])[],
  t: number,
): [number, number, number, number] {
  if (lut.length === 0) return [1, 1, 1, 1];
  const { i0, i1, f } = lutSamplePoint(lut.length, t);
  const a = lut[i0]!;
  const b = lut[i1]!;
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f];
}

/** Design spec §3.2: "colour from `color_ramp` over lifetime ... ×
 *  `self_modulate` × the composite's tint; then `alpha_curve`." (See module
 *  doc for why `color_initial_ramp` is not a factor here.) `lifetimeFraction`
 *  is Godot's `CUSTOM.y / CUSTOM.w`, which `readout()` reports per particle —
 *  do NOT re-derive it as `age / spec.lifetime`, which ignores each
 *  particle's own randomized lifetime and ends the curve early. */
export function particleVertexColor(
  emitter: ParticleEmitter,
  tint: readonly [number, number, number, number],
  colorRampLut: readonly (readonly [number, number, number, number])[] | null,
  alphaCurveLut: readonly number[] | null,
  lifetimeFraction: number,
): [number, number, number, number] {
  let [r, g, b, a] = emitter.selfModulate;
  r *= tint[0];
  g *= tint[1];
  b *= tint[2];
  a *= tint[3];
  if (colorRampLut) {
    const [cr, cg, cb, ca] = sampleColorLut(colorRampLut, lifetimeFraction);
    r *= cr;
    g *= cg;
    b *= cb;
    a *= ca;
  }
  if (alphaCurveLut) a *= sampleLut(alphaCurveLut, lifetimeFraction);
  return [r, g, b, a];
}

// ---------------------------------------------------------------------------
// Manifest -> simulate.ts's ParticleSystemSpec, resolving curve paths against
// a pre-loaded LUT cache.
// ---------------------------------------------------------------------------

/** Every scalar-curve field `ParticleSystemSpec` reads from a manifest path
 *  (as opposed to `colorRamp`, which is RGBA-shaped — see `sampleColorLut`
 *  above — or `colorInitialRamp`, which the manifest already inlines). */
function scalarCurvePaths(p: ParticleEmitterProcess): (string | null)[] {
  return [
    p.dampingCurve, p.radialVelocityCurve, p.scaleCurve, p.scaleOverVelocityCurve,
    p.turbulenceInfluenceOverLifeCurve, p.angleCurve, p.angularVelocityCurve, p.animSpeedCurve,
  ];
}

/** Builds the flattened spec `simulate()` consumes from one manifest
 *  `ParticleEmitter`, resolving every `*Curve` path against `curveLuts`.
 *  Returns `null` when a needed curve hasn't finished loading yet — the same
 *  "still loading, skip this frame" contract `VfxLayer.tsx` already uses for
 *  a flipbook texture or Spine rig in flight, not an error. */
export function buildParticleSystemSpec(
  emitter: ParticleEmitter,
  curveLuts: ReadonlyMap<string, readonly number[]>,
): ParticleSystemSpec | null {
  const p = emitter.process;
  for (const path of scalarCurvePaths(p)) {
    if (path !== null && !curveLuts.has(path)) return null;
  }
  const lut = (path: string | null): readonly number[] | null => (path === null ? null : (curveLuts.get(path) ?? null));
  return {
    amount: emitter.amount,
    cpu: emitter.cpu,
    explosiveness: emitter.explosiveness,
    randomness: emitter.randomness,
    fixedFps: emitter.fixedFps,
    lifetime: emitter.lifetime,
    localCoords: emitter.localCoords,
    offset: emitter.offset,
    direction: p.direction,
    spread: p.spread,
    flatness: p.flatness,
    gravity: p.gravity,
    initialVelocityMin: p.initialVelocityMin,
    initialVelocityMax: p.initialVelocityMax,
    lifetimeRandomness: p.lifetimeRandomness,
    animOffsetMin: p.animOffsetMin,
    animOffsetMax: p.animOffsetMax,
    animSpeedMin: p.animSpeedMin,
    animSpeedMax: p.animSpeedMax,
    animSpeedCurve: lut(p.animSpeedCurve),
    angleMin: p.angleMin,
    angleMax: p.angleMax,
    angleCurve: lut(p.angleCurve),
    emissionShape: p.emissionShape,
    emissionSphereRadius: p.emissionSphereRadius,
    emissionBoxExtents: p.emissionBoxExtents,
    emissionRingAxis: p.emissionRingAxis,
    emissionRingHeight: p.emissionRingHeight,
    emissionRingRadius: p.emissionRingRadius,
    emissionRingInnerRadius: p.emissionRingInnerRadius,
    emissionRingConeAngle: p.emissionRingConeAngle,
    emissionShapeOffset: p.emissionShapeOffset,
    emissionShapeScale: p.emissionShapeScale,
    angularVelocityMin: p.angularVelocityMin,
    angularVelocityMax: p.angularVelocityMax,
    angularVelocityCurve: lut(p.angularVelocityCurve),
    colorInitialRamp: p.colorInitialRamp,
    dampingMin: p.dampingMin,
    dampingMax: p.dampingMax,
    dampingCurve: lut(p.dampingCurve),
    particleFlagDampingAsFriction: p.particleFlagDampingAsFriction,
    particleFlagDisableZ: p.particleFlagDisableZ,
    radialVelocityMin: p.radialVelocityMin,
    radialVelocityMax: p.radialVelocityMax,
    radialVelocityCurve: lut(p.radialVelocityCurve),
    scaleMin: p.scaleMin,
    scaleMax: p.scaleMax,
    scaleCurve: lut(p.scaleCurve),
    scaleOverVelocityMax: p.scaleOverVelocityMax,
    scaleOverVelocityCurve: lut(p.scaleOverVelocityCurve),
    turbulenceEnabled: p.turbulenceEnabled,
    turbulenceInfluenceMin: p.turbulenceInfluenceMin,
    turbulenceInfluenceMax: p.turbulenceInfluenceMax,
    turbulenceInfluenceOverLifeCurve: lut(p.turbulenceInfluenceOverLifeCurve),
    turbulenceInitialDisplacementMin: p.turbulenceInitialDisplacementMin,
    turbulenceInitialDisplacementMax: p.turbulenceInitialDisplacementMax,
    turbulenceNoiseScale: p.turbulenceNoiseScale,
    turbulenceNoiseSpeed: p.turbulenceNoiseSpeed,
    turbulenceNoiseSpeedRandom: p.turbulenceNoiseSpeedRandom,
    turbulenceNoiseStrength: p.turbulenceNoiseStrength,
  };
}

/** Every path a `curveLuts` cache must resolve before `buildParticleSystemSpec`
 *  can succeed for this emitter — what a caller should eagerly `fetch`. */
export function scalarCurvePathsOf(emitter: ParticleEmitter): string[] {
  return scalarCurvePaths(emitter.process).filter((p): p is string => p !== null);
}

// ---------------------------------------------------------------------------
// Shader classification — the one real draw-path fork (which channel masks).
// ---------------------------------------------------------------------------

/** Which texture channel a shader's alpha comes out of. `"g*a"` is the one
 *  shader that multiplies two of them. */
export type MaskChannel = "r" | "g" | "a" | "g*a";

/**
 * Per-shader mask channel, read off each `.gdshader`'s own fragment stage.
 *
 * This was a one-entry deny-list ("everything except the grayscale shader
 * masks on alpha") for as long as the port believed the materials in use
 * were two families. They are 11 shaders, and the split does not run along
 * the flipbook/grayscale line at all: `vfx_poof_shader`,
 * `vfx_round_smoke_shader` and `vfx_ring_polar_shader` mask on RED like the
 * grayscale one, and their textures (`common_poof.png`,
 * `common_horizontal_smoke.png`, `common_round_smoke_flipbook.png`,
 * `common_ring_polar_a.png`) carry a CONSTANT 255 alpha with the whole
 * gradient in red — so an alpha rule makes `mask = 1.0` everywhere and draws
 * a soft puff as an opaque rectangle. That was live for 8 of the 47
 * material-bearing emitters, across `vfx_fire_burst`, `vfx_gaseous_impact`,
 * `vfx_goopy_impact`, `vfx_poison_impact` and `vfx_spore_impact`.
 *
 * Each entry quotes the line the channel comes from (v0.107.1 pck,
 * `shaders/vfx/**`); an unlisted shader is NOT given a default channel
 * silently — see `maskChannelOf`.
 */
const MASK_CHANNELS: Readonly<Record<string, MaskChannel>> = {
  // COLOR = vec4(color_sub.rgb, vertex_color.a * grayscale.r)
  vfx_grayscale_particle_shader: "r",
  // erosion_from_factors(..., texture_color.r), alpha = that
  vfx_poof_shader: "r",
  vfx_round_smoke_shader: "r",
  vfx_heavy_blunt_erosion_shader: "r",
  // smoothstep(..., texture_color.r); no LUT, rgb straight from vertex_color
  vfx_ring_polar_shader: "r",
  // alpha = flipbook_texture_color.a * vertex_color.a * erosion
  vfx_flipbook_shader: "a",
  vfx_row_flipbook_shader: "a",
  // erosion_from_factors(..., texture_color.a)
  vfx_potion_splash_shader: "a",
  // erosion_from_factors(..., flipbook_texture.g)
  vfx_goopy_impact_bits_shader: "g",
  // COLOR = vec4(lut_color.rgb, erosion_val * texture_color.a) * vertex_color
  // with erosion_val taken from texture_color.g — the only shader in use
  // that multiplies two channels.
  vfx_poison_skull_shader: "g*a",
  // erosion_from_factors(..., texture_color.a) via the shared flipbook path
  vfx_common_particle_shader: "a",
};

/** Shaders already reported by `maskChannelOf` as unknown, so a per-frame
 *  draw path logs each one once instead of every frame. */
const reportedUnknownShaders = new Set<string>();

/** The mask channel for one material, or `"a"` for `material === null` (the
 *  plain path ignores the mask entirely — see `BatchMaterialState.plain`).
 *
 *  A shader `MASK_CHANNELS` has never seen is LOUD rather than silently
 *  defaulted: getting the channel wrong is invisible for a texture whose
 *  channels agree and draws an opaque rectangle for one where they do not,
 *  which is exactly the failure that hid here before. The manifest-driven
 *  test in `shaders.test.ts` is the real gate — this is the runtime tell if
 *  a new material ships ahead of it. */
export function maskChannelOf(material: ParticleMaterial | null): MaskChannel {
  if (material === null) return "a";
  const channel = MASK_CHANNELS[material.shader];
  if (channel) return channel;
  if (!reportedUnknownShaders.has(material.shader)) {
    reportedUnknownShaders.add(material.shader);
    console.error(
      `particle material ${material.path}: unknown shader ${material.shader}; ` +
        "masking on alpha, which may be wrong — add it to MASK_CHANNELS",
    );
  }
  return "a";
}

// ---------------------------------------------------------------------------
// LUT textures and flipbook cells.
// ---------------------------------------------------------------------------

/** A colourisation LUT uploaded as a 1-D (N x 1) RGBA texture, plus the scale
 *  its values were normalised by — see `PARTICLE_FRAGMENT_SHADER`'s doc for
 *  why an 8-bit texture plus a scalar rather than a float texture. */
export interface LutTexture {
  texture: WebGLTexture;
  scale: number;
}

/** Normalising divisor for a baked LUT: its peak RGB channel, or 1 when the
 *  ramp is already inside [0, 1]. Split out so the quantisation behaviour is
 *  testable without a GL context. */
export function lutScaleOf(lut: readonly (readonly [number, number, number, number])[]): number {
  let peak = 1;
  for (const stop of lut) {
    peak = Math.max(peak, stop[0], stop[1], stop[2]);
  }
  return peak;
}

/** Uploads a baked RGBA LUT as an N x 1 texture with CLAMP_TO_EDGE + LINEAR,
 *  matching how Godot samples a `GradientTexture1D`. */
export function createLutTexture(
  gl: WebGLRenderingContext,
  lut: readonly (readonly [number, number, number, number])[],
): LutTexture {
  const scale = lutScaleOf(lut);
  const bytes = new Uint8Array(lut.length * 4);
  const q = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255)));
  for (let i = 0; i < lut.length; i++) {
    const stop = lut[i]!;
    // Alpha is NOT scaled: only RGB carries the authored overbright, and
    // dividing alpha by an RGB peak would make an opaque ramp translucent.
    bytes[i * 4 + 0] = q(stop[0] / scale);
    bytes[i * 4 + 1] = q(stop[1] / scale);
    bytes[i * 4 + 2] = q(stop[2] / scale);
    bytes[i * 4 + 3] = q(stop[3]);
  }
  const texture = gl.createTexture();
  if (!texture) throw new Error("particle lut: createTexture failed");
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, lut.length, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return { texture, scale };
}

/** UV sub-rect of one sprite-sheet cell: `[u0, v0, u1, v1]`. */
export type FlipbookCell = readonly [number, number, number, number];

/**
 * Which cell of a flipbook sheet a particle shows.
 *
 * Godot computes this in the VERTEX stage via `flipbook.gdshaderinc`:
 *
 * ```text
 * progress = floor(mod(progress, size.x * size.y));
 * frame_size = 1.0 / size;
 * frame = fract(uv / size) + frame_size;
 * frame.x += ((progress / size.x) - frame_size.x * floor(progress / size.x) * size.x) - frame_size.x;
 * frame.y += (frame_size.y * floor(progress / size.x)) - frame_size.y;
 * ```
 *
 * For `uv` in [0, 1] that algebra collapses exactly to a row-major cell
 * lookup — `frame.x = (uv.x + col) / size.x`, `frame.y = (uv.y + row) / size.y`
 * with `col = progress mod size.x` and `row = floor(progress / size.x)` —
 * because `frame_size.x * floor(p / size.x) * size.x` is just
 * `floor(p / size.x)`, and `progress` is already floored so
 * `fract(progress / size.x)` is `col / size.x`. So computing the cell here and
 * emitting sub-rect UVs is equivalent to porting the GLSL, not an
 * approximation of it.
 *
 * The two shader families derive `progress` differently, and both are real:
 *
 *  - `vfx_flipbook_shader` (17 emitters): the per-particle offset and the
 *    lifetime curve ADD into one linear frame index —
 *    `(frame_count * offset + frame_count * curve) * 0.99`. The `0.99` is
 *    verbatim; it keeps the index off the exact end of the sheet.
 *  - `vfx_row_flipbook_shader` (1 emitter): the offset picks a ROW and the
 *    curve walks along it, so each particle animates through its own row.
 */
export function flipbookCell(
  material: ParticleMaterial,
  lifetimeFraction: number,
  animOffset: number,
  flipbookCurveLut: readonly number[] | null,
): FlipbookCell {
  const size = material.flipbookSize;
  const frameCount = material.frameCount;
  if (size === null || frameCount === null) return [0, 0, 1, 1];
  const [cols, rows] = size;
  if (cols <= 0 || rows <= 0) return [0, 0, 1, 1];

  const curve = flipbookCurveLut ? sampleLut(flipbookCurveLut, lifetimeFraction) : 0;

  let progress: number;
  if (material.shader === "vfx_row_flipbook_shader") {
    const frameX = Math.floor(animOffset * rows) / rows;
    const frameY = Math.floor(curve * cols) / cols / rows;
    progress = Math.floor((frameX + frameY) * frameCount);
  } else {
    progress = (frameCount * animOffset + frameCount * curve) * 0.99;
  }

  const cells = cols * rows;
  progress = Math.floor(((progress % cells) + cells) % cells);
  const col = progress % cols;
  const row = Math.floor(progress / cols);
  return [col / cols, row / rows, (col + 1) / cols, (row + 1) / rows];
}

// ---------------------------------------------------------------------------
// WebGL: one shared program for every particle quad.
// ---------------------------------------------------------------------------

export const PARTICLE_VERTEX_SHADER = `
attribute vec2 a_position;
attribute vec2 a_uv;
attribute vec4 a_color;
varying vec2 v_uv;
varying vec4 v_color;
void main() {
  v_uv = a_uv;
  v_color = a_color;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

/**
 * Transcribed from the 11 `.gdshader`s that cover every material in use
 * (v0.107.1 pck, `shaders/vfx/**`). All of them reduce to the same two
 * lines:
 *
 * ```glsl
 * vec4 lut_color = texture(lut, tex.rr);          // colourisation
 * COLOR = vec4((lut_color * vertex_color).rgb, vertex_color.a * MASK);
 * ```
 *
 * with exactly two forks between them, both real:
 *
 *  - **MASK** is one texture channel per shader — RED, GREEN, ALPHA, or (for
 *    `vfx_poison_skull_shader` alone) GREEN×ALPHA. `u_maskSelect` picks it
 *    with a dot product and `u_maskTimesAlpha` folds in that one product,
 *    rather than branching. The channel itself is chosen CPU-side from the
 *    shader name (`MASK_CHANNELS`); using `.r` for everything (the first
 *    cut) or `.a` for everything-but-grayscale (the second) each gets a
 *    different third of the emitters wrong.
 *  - the flipbook families sample a sub-rect of a sprite sheet rather than the
 *    whole texture. That fork is resolved CPU-side (`flipbookCell`) and
 *    arrives here as ordinary UVs, so it needs no branch.
 *
 * The per-lifetime EROSION SMOOTHSTEP those shaders wrap the mask in is still
 * not ported (module doc); the channel it reads is what this fixes.
 *
 * `u_lutScale` exists because the authored LUTs are HDR: 14 of the 47 in use
 * exceed 1.0 and one reaches 12.0 (`vfx_dagger_impact_core`'s `12, 8, 8`
 * overbright core). This canvas is WebGL1, where float textures need an
 * extension that is not guaranteed, so each LUT is normalised by its own peak
 * into an 8-bit texture and multiplied back here. That preserves the
 * overbright exactly up to 8-bit quantisation of the normalised ramp, rather
 * than clamping it away — clamping would flatten precisely the glows the
 * overbright exists for.
 */
export const PARTICLE_FRAGMENT_SHADER = `
precision mediump float;
uniform sampler2D u_tex;
uniform sampler2D u_lut;
uniform float u_hasLut;
uniform float u_lutScale;
uniform vec4 u_maskSelect;
uniform float u_maskTimesAlpha;
uniform float u_plain;
varying vec2 v_uv;
varying vec4 v_color;
void main() {
  vec4 tex = texture2D(u_tex, v_uv);
  vec3 lut = texture2D(u_lut, vec2(tex.r, 0.5)).rgb * u_lutScale;
  vec3 tinted = mix(vec3(1.0), lut, u_hasLut) * v_color.rgb;
  float mask = dot(tex, u_maskSelect) * mix(1.0, tex.a, u_maskTimesAlpha);
  vec4 masked = vec4(tinted, v_color.a * mask);
  vec4 plain = tex * v_color;
  gl_FragColor = mix(masked, plain, u_plain);
}
`;

export interface ParticleProgram {
  program: WebGLProgram;
  aPosition: number;
  aUv: number;
  aColor: number;
  uTex: WebGLUniformLocation | null;
  uLut: WebGLUniformLocation | null;
  uHasLut: WebGLUniformLocation | null;
  uLutScale: WebGLUniformLocation | null;
  uMaskSelect: WebGLUniformLocation | null;
  uMaskTimesAlpha: WebGLUniformLocation | null;
  uPlain: WebGLUniformLocation | null;
  /** 1x1 opaque white, bound to `u_lut` whenever a material has no LUT so the
   *  sampler is never left reading an unbound unit (undefined in WebGL1). */
  whiteLut: WebGLTexture;
  buffer: WebGLBuffer;
}

function compileShader(gl: WebGLRenderingContext, type: number, src: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("particle shader: createShader failed");
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`particle shader compile failed: ${info ?? "unknown error"}`);
  }
  return shader;
}

/** Compiles + links the one shared particle program. Every emitter's quads —
 *  masked or plain, whatever texture — draw through this program; the
 *  per-draw state is the bound texture and the LUT/mask uniforms, not a
 *  different program. */
export function createParticleProgram(gl: WebGLRenderingContext): ParticleProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, PARTICLE_VERTEX_SHADER);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, PARTICLE_FRAGMENT_SHADER);
  const program = gl.createProgram();
  if (!program) throw new Error("particle program: createProgram failed");
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`particle program link failed: ${info ?? "unknown error"}`);
  }
  const buffer = gl.createBuffer();
  if (!buffer) throw new Error("particle program: createBuffer failed");
  return {
    program,
    aPosition: gl.getAttribLocation(program, "a_position"),
    aUv: gl.getAttribLocation(program, "a_uv"),
    aColor: gl.getAttribLocation(program, "a_color"),
    uTex: gl.getUniformLocation(program, "u_tex"),
    uLut: gl.getUniformLocation(program, "u_lut"),
    uHasLut: gl.getUniformLocation(program, "u_hasLut"),
    uLutScale: gl.getUniformLocation(program, "u_lutScale"),
    uMaskSelect: gl.getUniformLocation(program, "u_maskSelect"),
    uMaskTimesAlpha: gl.getUniformLocation(program, "u_maskTimesAlpha"),
    uPlain: gl.getUniformLocation(program, "u_plain"),
    whiteLut: createLutTexture(gl, [[1, 1, 1, 1]]).texture,
    buffer,
  };
}

/** Floats per vertex in the interleaved buffer `appendQuadVertices` writes:
 *  clip-space xy, uv, rgba. */
export const PARTICLE_FLOATS_PER_VERTEX = 8;

/** One textured, tinted quad, given in CANVAS PIXEL space (Y-up, origin
 *  bottom-left — the same convention `VfxLayer.tsx` already uses for
 *  flipbook/Spine draws) and centred at `(cx, cy)`. */
export interface ParticleQuad {
  cx: number;
  cy: number;
  w: number;
  h: number;
  color: readonly [number, number, number, number];
  /** Which part of the texture to sample, as `[u0, v0, u1, v1]`. The whole
   *  texture for a plain material; one sprite-sheet cell for a flipbook
   *  (`flipbookCell`). Defaults to the whole texture when omitted. */
  cell?: FlipbookCell;
  /** The particle's rotation in RADIANS (`Particle.angle`, Godot's
   *  `CUSTOM.x`). Omitted or 0 leaves the quad axis-aligned, which is what
   *  every emitter that authors no `angle`/`angular_velocity` wants — and is
   *  the exact shape this built before rotation existed. */
  angle?: number;
}

/** Appends one quad (2 triangles, 6 vertices, no index buffer — batch sizes
 *  here are small, design spec §2.2: "under 200 particles") to `out`,
 *  converting pixel space to clip space so no caller has to think about NDC.
 *
 *  **V runs opposite Y.** Positions are Y-UP (`y1` is the top edge on
 *  screen); texture V is Y-DOWN (`v = 0` is the texture's FIRST row, i.e. its
 *  top, because nothing here sets `UNPACK_FLIP_Y_WEBGL` — Spine's
 *  `GLTexture` does not, and neither does any app code). So the top corner
 *  takes `v0` and the bottom corner takes `v1`. That is the same pairing
 *  Spine's own `SceneRenderer.drawTexture`/`drawTextureRotated` use for the
 *  flipbook and rig draws sharing this canvas (`v = 1` written against the
 *  `y` corner, `v = 0` against `y + height`), and `flipbookCell` emits its
 *  rows in the same Y-down space — row 0 is the sheet's TOP row. Pairing
 *  `v0` with the bottom corner instead drew every particle, and every
 *  flipbook cell, upside down relative to the game. */
export function appendQuadVertices(out: number[], quad: ParticleQuad, canvasW: number, canvasH: number): void {
  const { cx, cy, w, h, color } = quad;
  // Half-extents, so the corners below are offsets FROM the centre — which is
  // what lets the rotation be a plain 2x2 about the origin.
  const hw = w / 2;
  const hh = h / 2;
  const toClipX = (x: number) => (x / canvasW) * 2 - 1;
  const toClipY = (y: number) => (y / canvasH) * 2 - 1;
  const [u0, v0, u1, v1] = quad.cell ?? [0, 0, 1, 1];
  const corners: [number, number, number, number][] = [
    [-hw, -hh, u0, v1],
    [hw, -hh, u1, v1],
    [hw, hh, u1, v0],
    [-hw, -hh, u0, v1],
    [hw, hh, u1, v0],
    [-hw, hh, u0, v0],
  ];
  // Godot's own 2D particle transform is `TRANSFORM[0] = (cos, -sin)`,
  // `TRANSFORM[1] = (sin, cos)` (`ppm451.cpp:1090-1091`) — counter-clockwise in
  // ITS space. This canvas is Y-up where Godot 2D is Y-down, the same chirality
  // flip `VfxLayer`'s `drawAngleFromGodotDegrees` documents for sprites, so the
  // sign is inverted here to keep a positive `angular_velocity` spinning the
  // way the game shows it.
  const angle = quad.angle ?? 0;
  const cos = angle === 0 ? 1 : Math.cos(-angle);
  const sin = angle === 0 ? 0 : Math.sin(-angle);
  for (const [dx, dy, u, v] of corners) {
    const x = cx + (cos * dx - sin * dy);
    const y = cy + (sin * dx + cos * dy);
    out.push(toClipX(x), toClipY(y), u, v, color[0], color[1], color[2], color[3]);
  }
}

/** The stretched quad that stands in for a trail-enabled particle's ribbon —
 *  centre and rotation in the EMITTER's own space (Godot's, Y-down), in the
 *  emitter's own units. The caller scales and flips it exactly as it would the
 *  unstretched quad. */
export interface TrailQuadGeometry {
  cx: number;
  cy: number;
  /** Extent along the quad's LOCAL Y axis, i.e. along the path: the particle's
   *  own height plus the distance it covered over `trailLifetime`. */
  h: number;
  /** Rotation in the same sense `ParticleQuad.angle` takes (Godot's), putting
   *  the quad's local +Y along the direction of travel. */
  angle: number;
}

/**
 * Godot draws a `trail_enabled` particle as a RIBBON through the transforms it
 * held over the last `trail_lifetime` seconds, rather than as the single quad
 * every other emitter in scope gets. This returns the straight segment that
 * ribbon spans: from where the particle was `trailLifetime` ago to where it is
 * now, `baseHeight` longer so a stalled particle still draws its own quad.
 *
 * **Why a straight segment rather than the real history.** The ribbon's spine
 * is the trajectory, and the only emitter in scope authors `trail_lifetime =
 * 0.05` under a 30 fps process rate — about one and a half process frames of
 * history. A subdivided ribbon and a straight one differ by the trajectory's
 * curvature over those 50 ms: at that emitter's 5000 peak `linear_accel` that
 * is `0.5*a*t^2` = 6 emitter units, ~1.5 screen px, against a segment of
 * 30-60 px. Reproducing it exactly would mean carrying a per-particle position
 * history `simulate.ts` does not keep, to move a corner by a pixel.
 *
 * The rotation is the PATH's, not `particle_flag_align_y`'s — a ribbon follows
 * the trajectory whether or not the emitter also aligns its particles to
 * velocity, so this is correct for a trail-enabled emitter that leaves that
 * flag off. The untrailed quads take the flag through `particleDrawAngle`.
 *
 * Returns `null` when there is nothing to stretch — no trail lifetime, or a
 * particle that is not moving and so has no direction to stretch ALONG.
 */
export function trailQuadGeometry(
  particle: { x: number; y: number; vx: number; vy: number },
  baseHeight: number,
  trailLifetime: number,
): TrailQuadGeometry | null {
  const { x, y, vx, vy } = particle;
  const speed = Math.hypot(vx, vy);
  if (!(speed > 0) || !(trailLifetime > 0)) return null;
  const length = speed * trailLifetime;
  // The particle's position is the ribbon's HEAD, so the segment's centre sits
  // half its added length back along the path.
  const backX = (vx / speed) * (length / 2);
  const backY = (vy / speed) * (length / 2);
  return {
    cx: x - backX,
    cy: y - backY,
    h: baseHeight + length,
    angle: angleAlongLocalY(vx, vy),
  };
}

/** The `ParticleQuad.angle` that puts a quad's local +Y axis along `(dx, dy)`,
 *  a direction in the emitter's own Y-down space. `appendQuadVertices` rotates
 *  local (0, 1) to (-sin, cos) in Godot's sense (it applies the Y-up
 *  chirality flip itself), so that wants `sin = -dx/len`, `cos = dy/len`. */
export function angleAlongLocalY(dx: number, dy: number): number {
  return Math.atan2(-dx, dy);
}

/** The rotation one untrailed particle quad is drawn at. With
 *  `particle_flag_align_y` set, Godot's 2D branch REPLACES the basis with one
 *  built from the velocity (`ppm451.cpp:1085-1092`) and never reads
 *  `CUSTOM.x`, so the authored `angle`/`angular_velocity` do not compose with
 *  it — the flag wins outright. Otherwise it is the particle's own spin. */
export function particleDrawAngle(
  process: Pick<ParticleEmitterProcess, "particleFlagAlignY">,
  particle: Pick<Particle, "angle" | "heading">,
): number {
  return process.particleFlagAlignY ? angleAlongLocalY(particle.heading[0], particle.heading[1]) : particle.angle;
}

/** How one batch of quads is coloured: which LUT (if any) colourises the
 *  grayscale texture, and which channel carries the mask. */
export interface BatchMaterialState {
  lut: LutTexture | null;
  /** Which texture channel this material's shader takes its alpha from —
   *  `maskChannelOf`, keyed on the shader name. */
  mask: MaskChannel;
  /** `true` when the emitter has no material at all — the texture's own RGBA
   *  is used untouched, which is what the real game does for those too. */
  plain: boolean;
  /** `CanvasItemMaterial.blend_mode == 1` (Add) on the emitter, from the
   *  manifest's `blendAdd`. Every SHADER material in the family is
   *  `render_mode blend_mix`, so this only ever comes from a blend-mode-only
   *  `CanvasItemMaterial` sub-resource, which arrives with `material: null`
   *  and would otherwise draw as a straight-alpha mix — grey dots where the
   *  game glows (GH #669). Optional so the 14 shipped composites' call sites
   *  read exactly as they did. */
  additive?: boolean;
}

/** `u_maskSelect` for one mask channel: the `dot(tex, select)` that picks it.
 *  `"g*a"` selects GREEN here and folds the ALPHA factor in through
 *  `u_maskTimesAlpha`, which is the only product any shader in use takes. */
export function maskSelectVector(mask: MaskChannel): [number, number, number, number] {
  switch (mask) {
    case "r":
      return [1, 0, 0, 0];
    case "g":
    case "g*a":
      return [0, 1, 0, 0];
    case "a":
      return [0, 0, 0, 1];
  }
}

/** Draws one texture's worth of already-appended quad vertices. Callers group
 *  draws by texture AND material state (design spec 4.2). No-op on an empty
 *  batch. */
export function drawParticleBatch(
  gl: WebGLRenderingContext,
  prog: ParticleProgram,
  texture: { bind(unit?: number): void },
  material: BatchMaterialState,
  vertexData: number[],
): void {
  if (vertexData.length === 0) return;
  gl.useProgram(prog.program);
  texture.bind(0);
  gl.uniform1i(prog.uTex, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, material.lut ? material.lut.texture : prog.whiteLut);
  gl.uniform1i(prog.uLut, 1);
  gl.uniform1f(prog.uHasLut, material.lut ? 1 : 0);
  gl.uniform1f(prog.uLutScale, material.lut ? material.lut.scale : 1);
  gl.activeTexture(gl.TEXTURE0);

  const [mr, mg, mb, ma] = maskSelectVector(material.mask);
  gl.uniform4f(prog.uMaskSelect, mr, mg, mb, ma);
  gl.uniform1f(prog.uMaskTimesAlpha, material.mask === "g*a" ? 1 : 0);
  gl.uniform1f(prog.uPlain, material.plain ? 1 : 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, prog.buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexData), gl.DYNAMIC_DRAW);
  const stride = PARTICLE_FLOATS_PER_VERTEX * 4;
  gl.enableVertexAttribArray(prog.aPosition);
  gl.vertexAttribPointer(prog.aPosition, 2, gl.FLOAT, false, stride, 0);
  gl.enableVertexAttribArray(prog.aUv);
  gl.vertexAttribPointer(prog.aUv, 2, gl.FLOAT, false, stride, 8);
  gl.enableVertexAttribArray(prog.aColor);
  gl.vertexAttribPointer(prog.aColor, 4, gl.FLOAT, false, stride, 16);
  // `render_mode blend_mix` on every SHADER in the family: straight alpha.
  // `additive` is Godot's `CanvasItemMaterial` blend_mode 1 — the same
  // `(SRC_ALPHA, ONE)` `drawPowerFlashQuad` uses for `blend_add`, and NOT the
  // premultiplied `(ONE, ONE_MINUS_SRC_ALPHA)`; see that function for why.
  gl.enable(gl.BLEND);
  if (material.additive) {
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  } else {
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }
  gl.drawArrays(gl.TRIANGLES, 0, vertexData.length / PARTICLE_FLOATS_PER_VERTEX);
}

// ---------------------------------------------------------------------------
// `vfx_power_flash_vfx`: the one additive sprite-shader effect (Task 13).
// Not a particle system — one quad, drawn through its own tiny program rather
// than folded into the particle one above, since it needs a DIFFERENT blend
// mode (`gl.SRC_ALPHA, gl.ONE` — Godot's `blend_add` — vs the particle
// family's straight-alpha mix; see `drawPowerFlashQuad` for why that is NOT
// the premultiplied `gl.ONE, gl.ONE_MINUS_SRC_ALPHA`) and takes its colour as a
// single uniform (one quad per instance, so there is nothing for a
// per-vertex attribute to vary).
// ---------------------------------------------------------------------------

export const POWER_FLASH_VERTEX_SHADER = `
attribute vec2 a_position;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() {
  v_uv = a_uv;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

/**
 * Ported from `shaders/power_flash_vfx.gdshader` (v0.107.1 pck),
 * `render_mode blend_add`:
 *
 * ```text
 * shader_type canvas_item;
 * render_mode blend_add;
 * uniform float strength = 3.0;
 * void fragment() {
 *     vec4 col = texture(TEXTURE, UV);
 *     vec4 ModularCol = COLOR / col;          // dead -- never read
 *     vec4 pixelCol = texture(TEXTURE, UV);
 *     pixelCol.rgb *= strength;
 *     COLOR *= pixelCol;
 * }
 * ```
 *
 * A literal one-texture-factor transcription (`COLOR = vertex_color *
 * vec4(tex.rgb * strength, tex.a)`) is WRONG: in a Godot `canvas_item`
 * shader, `COLOR` on ENTRY to `fragment()` is already `vertex_color *
 * texture(TEXTURE, UV)` — confirmed against Godot's own canvas_item shader
 * reference ("In the `fragment()` function, the input value `COLOR` is that
 * same value multiplied by the color from the default `TEXTURE`"), not taken
 * on the addendum's word alone. So `COLOR *= pixelCol` samples and
 * multiplies the texture TWICE: rgb comes out `v_color.rgb * tex.rgb^2 *
 * strength`, alpha `v_color.a * tex.a^2`. The dead `ModularCol = COLOR / col`
 * line is itself evidence for this reading — dividing `COLOR` by a fresh
 * texture sample only recovers the original modulate colour if `COLOR`
 * already carried that sample; the line is still not transcribed (it is
 * genuinely dead, `ModularCol` is never read again), only its implication
 * about the line below it is kept.
 *
 * `u_strength` is read from the manifest's `strength` field (1.0 for the
 * committed `power_flash_vfx.tscn` scene, overriding the shader's own
 * `= 3.0` default — verified directly against both the committed manifest
 * and `power_flash_vfx.tscn` itself, not the `power_applied_vfx.tscn`
 * echo-panel scene, which is a DIFFERENT node using the same shader at a
 * DIFFERENT override, `1.5`) — never hardcoded, so a future scene using this
 * shader without a material override still gets the shader's real 3.0.
 *
 * `u_color` stands in for `vertex_color`: one quad per instance, animated by
 * `NPowerFlashVfx`'s scale/alpha tween (`SpriteShaderEffect.tween`), so there
 * is nothing for a per-vertex attribute to vary and a uniform is the exact
 * fit — rgb white (the scene's own alpha, `0.65098`, never reaches a
 * rendered frame: `NPowerFlashVfx.cs:95` sets `Modulate = Colors.White`
 * before the tween's first `await`), alpha animated by the tween.
 */
export const POWER_FLASH_FRAGMENT_SHADER = `
precision mediump float;
uniform sampler2D u_tex;
uniform float u_strength;
uniform vec4 u_color;
varying vec2 v_uv;
void main() {
  vec4 pixel = texture2D(u_tex, v_uv);
  gl_FragColor = u_color * pixel * vec4(pixel.rgb * u_strength, pixel.a);
}
`;

export interface PowerFlashProgram {
  program: WebGLProgram;
  aPosition: number;
  aUv: number;
  uTex: WebGLUniformLocation | null;
  uStrength: WebGLUniformLocation | null;
  uColor: WebGLUniformLocation | null;
  buffer: WebGLBuffer;
}

/** Compiles + links the power-flash program. One shared program for every
 *  live `vfx_power_flash_vfx` instance, same idiom as `createParticleProgram`
 *  above. */
export function createPowerFlashProgram(gl: WebGLRenderingContext): PowerFlashProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, POWER_FLASH_VERTEX_SHADER);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, POWER_FLASH_FRAGMENT_SHADER);
  const program = gl.createProgram();
  if (!program) throw new Error("power flash program: createProgram failed");
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`power flash program link failed: ${info ?? "unknown error"}`);
  }
  const buffer = gl.createBuffer();
  if (!buffer) throw new Error("power flash program: createBuffer failed");
  return {
    program,
    aPosition: gl.getAttribLocation(program, "a_position"),
    aUv: gl.getAttribLocation(program, "a_uv"),
    uTex: gl.getUniformLocation(program, "u_tex"),
    uStrength: gl.getUniformLocation(program, "u_strength"),
    uColor: gl.getUniformLocation(program, "u_color"),
    buffer,
  };
}

/** Floats per vertex in the power-flash quad buffer: clip-space xy, uv. No
 *  per-vertex colour — see the module doc above for why a uniform covers it. */
export const POWER_FLASH_FLOATS_PER_VERTEX = 4;

/** Appends one power-flash quad, in the same CANVAS PIXEL / Y-up / V-down
 *  convention as `appendQuadVertices` (see its doc for why V pairs with the
 *  corners the way it does — this draws the whole texture, so `cell` is
 *  always the default full-rect). */
export function appendPowerFlashQuadVertices(
  out: number[],
  quad: { cx: number; cy: number; w: number; h: number },
  canvasW: number,
  canvasH: number,
): void {
  const { cx, cy, w, h } = quad;
  const x0 = cx - w / 2;
  const x1 = cx + w / 2;
  const y0 = cy - h / 2;
  const y1 = cy + h / 2;
  const toClipX = (x: number) => (x / canvasW) * 2 - 1;
  const toClipY = (y: number) => (y / canvasH) * 2 - 1;
  const corners: [number, number, number, number][] = [
    [x0, y0, 0, 1],
    [x1, y0, 1, 1],
    [x1, y1, 1, 0],
    [x0, y0, 0, 1],
    [x1, y1, 1, 0],
    [x0, y1, 0, 0],
  ];
  for (const [x, y, u, v] of corners) {
    out.push(toClipX(x), toClipY(y), u, v);
  }
}

/** Draws one power-flash quad. `render_mode blend_add` is Godot's ordinary
 *  additive blend — `gl.SRC_ALPHA, gl.ONE` — NOT the premultiplied-alpha
 *  `blend_premul_alpha` mode `dissolve.gdshader` uses (`gl.ONE,
 *  gl.ONE_MINUS_SRC_ALPHA`, Task 14); the two are easy to conflate because
 *  both composite "additively" in casual speech, but they are different blend
 *  functions over different source colour conventions. */
export function drawPowerFlashQuad(
  gl: WebGLRenderingContext,
  prog: PowerFlashProgram,
  texture: { bind(unit?: number): void },
  strength: number,
  color: readonly [number, number, number, number],
  vertexData: number[],
): void {
  if (vertexData.length === 0) return;
  gl.useProgram(prog.program);
  texture.bind(0);
  gl.uniform1i(prog.uTex, 0);
  gl.uniform1f(prog.uStrength, strength);
  gl.uniform4f(prog.uColor, color[0], color[1], color[2], color[3]);

  gl.bindBuffer(gl.ARRAY_BUFFER, prog.buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertexData), gl.DYNAMIC_DRAW);
  const stride = POWER_FLASH_FLOATS_PER_VERTEX * 4;
  gl.enableVertexAttribArray(prog.aPosition);
  gl.vertexAttribPointer(prog.aPosition, 2, gl.FLOAT, false, stride, 0);
  gl.enableVertexAttribArray(prog.aUv);
  gl.vertexAttribPointer(prog.aUv, 2, gl.FLOAT, false, stride, 8);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  gl.drawArrays(gl.TRIANGLES, 0, vertexData.length / POWER_FLASH_FLOATS_PER_VERTEX);
}
