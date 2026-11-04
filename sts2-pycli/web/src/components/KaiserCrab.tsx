import { useEffect, useRef, useState } from "react";
import * as spine from "@esotericsoftware/spine-webgl";

/**
 * Renders the Kaiser Crab boss — a SINGLE big Spine rig whose two targetable
 * arms (Crusher = left, Rocket = right) are driven on separate animation tracks,
 * exactly as the game does (see `NKaiserCrabBossBackground`: body idle on the
 * base track, `left/*` on track 1, `right/*` on track 2). The workbench used to
 * alias both arms to this rig and draw a whole crab PER arm; this draws it once.
 *
 * Idle: `body/idle_loop` (track 0) + `left/idle_loop` (track 1) +
 * `right/idle_loop` (track 2). A one-shot on an arm (`{nonce, clip}` for
 * `leftAction` / `rightAction`) plays `<side>/<clip>` on that arm's track, then
 * resumes that arm's idle (unless `hold`, e.g. death, which freezes the last
 * frame).
 *
 * SIZE AND PLACE. The crab is not fitted to a box: like every other creature it
 * is drawn at its authored scene scale (0.5 px per rig unit — `creature_scales`
 * `kaiser_crab`) with its ROOT on the combat scene's ground line, which is what
 * makes it 2765 x 1230 scene px, i.e. wider than the 1920 scene and most of its
 * height. That is the fight: the game mounts this rig on the combat-room
 * BACKGROUND and pulls the camera back to 0.75 (`KaiserCrabBoss`), so a claw
 * reaches in past each edge of the screen with the player standing between
 * them. Fitting it into the arena's box instead — which is what the workbench
 * did — produced a tidy little crab in the middle of the room.
 *
 * The canvas is placed relative to the rig ROOT (reported through `onMeasure`),
 * not centred in a box, so the parent can pin it to the ground line and pin
 * each arm's HP/intent overlay over the matching claw.
 *
 * Like SpineSprite, STS2 atlases are straight (non-premultiplied) alpha.
 */
/** Overscan margins (fractions of the idle body box) so a claw lunge is never
 *  clipped by the canvas edge. */
const OVER_TOP = 0.16; // headroom (rocket launch / raised claw)
const OVER_SIDE = 0.08; // each side (claw slam lunge)
const OVER_BOT = 0.04; // ground room below the body

/** Fallback px per rig unit if the caller has no authored scale. */
const DEFAULT_PX_PER_UNIT = 0.5;

const BODY_TRACK = 0;
const LEFT_TRACK = 1;
const RIGHT_TRACK = 2;

// Slots that are attack-only VFX (flashes, kicked-up dirt, ground shadows) and
// would otherwise dangle around the idle crab. Hidden at idle, restored while an
// action one-shot plays so a real slam still throws dirt.
const HIDE = [/zap/i, /sparkle/i, /dirt/i, /plow/i, /shadow/i, /^liquid/i, /glow/i];

export interface ArmAction {
  nonce: number;
  clip: string; // bare clip name without the side prefix, e.g. "attack", "hurt", "die"
  hold?: boolean; // freeze last frame (death) instead of resuming idle
}

/** One claw's painted box, px relative to the rig's ROOT (+x right, +y UP). */
export interface ClawBox { cx: number; left: number; right: number; top: number; bottom: number }

/** Geometry the parent needs to frame the crab and lay the arm overlays out.
 *  Every number is px measured from the rig's ROOT: +x right, +y UP. */
export interface CrabMeasure {
  canvasW: number; // rendered canvas CSS width (px)
  canvasH: number; // rendered canvas CSS height (px)
  /** Canvas left edge, px right of the root (negative = left of it). */
  canvasLeft: number;
  /** Canvas bottom edge, px above the root (negative = below it). */
  canvasBottom: number;
  /** The idle silhouette's top, px above the root — what the parent flushes to
   *  the top of the frame, since this boss IS the room. */
  top: number;
  bodyH: number; // drawn body height (px)
  /** Each claw's painted box. The arm readouts hang off these: the game gives
   *  each arm its own creature node with its own HP bar and intent, and they
   *  belong under and over the claw that arm actually is — measured from the
   *  claw's ART, not from its bone, which sits nowhere near the middle of it. */
  claw: { [part: string]: ClawBox };
}

export default function KaiserCrab({
  rigBase,
  pxPerUnit = DEFAULT_PX_PER_UNIT,
  leftAction,
  rightAction,
  onMeasure,
  onRigEvent,
}: {
  rigBase: string;
  /** Rendered px per rig unit — the creature's authored scene scale times the
   *  stage's scene factor. There is no box fit: this alone decides the size. */
  pxPerUnit?: number;
  leftAction?: ArmAction;
  rightAction?: ArmAction;
  // Reports the rendered crab geometry once measured, so the parent can anchor
  // each arm overlay over its claw. null = not yet / failed.
  onMeasure?: (m: CrabMeasure | null) => void;
  /** Fires for every SPINE EVENT the rig emits, with a resolver for any bone's
   *  live position.
   *
   *  This rig carries ten authored events, and the game drives real VFX off
   *  them — `NKaiserCrabBossVfx` listens for all ten. The one wired today is
   *  `claw_explode_l` (t=1.53s of `left/die`), which spawns the Crusher's
   *  death explosion at `%LeftArmExplosionPosition`, a `SpineBoneNode`
   *  following the `claw_upper` bone (GH #642).
   *
   *  `boneAt` is passed rather than a fixed anchor because the bone must be
   *  read AT THE MOMENT THE EVENT FIRES: 1.53 s into the death clip the claw
   *  is high in the air, nowhere near its setup pose, and nowhere near the
   *  ground-line overlay that arm's `TargetRef` resolves to. It returns
   *  VIEWPORT px (what `getBoundingClientRect` reports), which is what
   *  `PlayedEffect.point` takes, or `null` for an unknown bone / unmeasured
   *  rig. */
  onRigEvent?: (
    name: string,
    boneAt: (bone: string) => { x: number; y: number } | null,
  ) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<spine.AnimationState | null>(null);
  const dataRef = useRef<spine.SkeletonData | null>(null);
  // Read from inside the load effect, which must not re-run when either
  // changes — `pxPerUnit` moves on every stage resize (see the `world` doc
  // below), and `onRigEvent` is a fresh closure on every parent render.
  const onRigEventRef = useRef(onRigEvent); onRigEventRef.current = onRigEvent;
  const pxPerUnitRef = useRef(pxPerUnit); pxPerUnitRef.current = pxPerUnit;
  const inActionRef = useRef<{ [track: number]: boolean }>({});
  /** Arm one-shots that arrived while the rig was not loaded, per track.
   *
   *  The crab REMOUNTS at the moment its arms die — the whole 5 MB rig is
   *  disposed and re-fetched — and the `die` one-shot lands in that window.
   *  `playArm` used to consume the nonce (`prev.current = action.nonce`)
   *  BEFORE checking that the rig was ready, so the action was swallowed and
   *  never replayed: the Kaiser Crab's death animation has never actually
   *  played here, and neither have the spine events on it (GH #642's
   *  `claw_explode_l` among them). Stashing instead, and replaying from
   *  `initialize`, is what makes a one-shot survive the reload. */
  const pendingArmRef = useRef<{
    [track: number]: { side: string; action: ArmAction; prev: React.MutableRefObject<number | undefined> };
  }>({});
  /** `initialize` runs inside the load effect, which closes over the first
   *  render; `playArm` is redefined every render. A ref is how the replay
   *  reaches the current one. */
  const playArmRef = useRef<((
    track: number, side: string, action?: ArmAction,
    prev?: React.MutableRefObject<number | undefined>,
  ) => void) | null>(null);
  // The latest `onMeasure` and arm actions, for effects that must not re-run
  // when only those props' identities change (the load effect, the per-nonce
  // arm one-shots).
  const measureRef = useRef(onMeasure); measureRef.current = onMeasure;
  const leftActionRef = useRef(leftAction); leftActionRef.current = leftAction;
  const rightActionRef = useRef(rightAction); rightActionRef.current = rightAction;
  const prevLeft = useRef<number | undefined>(undefined);
  const prevRight = useRef<number | undefined>(undefined);
  const frameRef = useRef<{ sideMw: number; topMw: number; botMw: number } | null>(null);
  // The measured crab, in RIG UNITS relative to the rig root — never px. How big
  // the crab is drawn changes while the fight is on screen (the stage is
  // re-measured on every resize), and keeping the measurement scale-free makes
  // that a pure canvas resize; when the px sizes lived here, the size had to be
  // a dependency of the load effect, and a re-fit disposed the canvas and
  // re-fetched all four atlas pages (~5 MB) — a second of the boss gone, then
  // back at a new size with both arm overlays landing somewhere else.
  const [world, setWorld] = useState<{
    x: number; y: number; w: number; h: number; sideMw: number; topMw: number; botMw: number;
    claw: { [part: string]: ClawBox };
  } | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const canvasEl = canvasRef.current;
    if (!canvasEl) return;
    setWorld(null);
    setFailed(false);
    frameRef.current = null;

    let skeleton: spine.Skeleton | null = null;
    const off = new spine.Vector2(), size = new spine.Vector2();

    const hideFx = (sk: spine.Skeleton) => {
      for (const slot of sk.slots) {
        const a = slot.getAttachment();
        if (!a) continue;
        if (!(a instanceof spine.RegionAttachment || a instanceof spine.MeshAttachment)) continue;
        if (HIDE.some((p) => p.test(slot.data.name) || p.test(a.name ?? ""))) slot.setAttachment(null);
      }
    };

    // Set the three idle tracks, tolerating a rig that names them differently.
    const setIdle = (st: spine.AnimationState, data: spine.SkeletonData) => {
      const has = (n: string) => !!data.findAnimation(n);
      if (has("body/idle_loop")) st.setAnimation(BODY_TRACK, "body/idle_loop", true);
      else if (has("idle_loop")) st.setAnimation(BODY_TRACK, "idle_loop", true);
      if (has("left/idle_loop")) st.setAnimation(LEFT_TRACK, "left/idle_loop", true);
      if (has("right/idle_loop")) st.setAnimation(RIGHT_TRACK, "right/idle_loop", true);
    };

    /** A bone's CURRENT position in viewport px, for `onRigEvent`.
     *
     *  World rig units -> canvas CSS px is exact rather than approximate,
     *  because `render()` pins the camera to a known box: viewport
     *  `vw x vh` = the measured body plus this file's overscan margins, drawn
     *  into a canvas whose CSS size is that same box times `pxPerUnit`. So the
     *  mapping is the one linear transform below — origin at the box's
     *  top-left, Y flipped (Spine is Y-up, the DOM is Y-down).
     *
     *  `render()` sets `viewportWidth`/`viewportHeight` AFTER its
     *  `r.resize(Expand)` call, so the explicit box is the one that survives;
     *  reading the camera here instead would race the next frame.
     *
     *  `null` before the rig is measured (`size.y <= 0`), for an unknown bone,
     *  or once the canvas is gone. */
    const boneViewportPoint = (boneName: string): { x: number; y: number } | null => {
      const fr = frameRef.current;
      if (!skeleton || !fr || size.y <= 0) return null;
      const bone = skeleton.findBone(boneName);
      if (!bone) return null;
      const rect = canvasEl.getBoundingClientRect();
      const k = pxPerUnitRef.current ?? DEFAULT_PX_PER_UNIT;
      return {
        x: rect.left + (bone.worldX - (off.x - fr.sideMw)) * k,
        y: rect.top + ((off.y + size.y + fr.topMw) - bone.worldY) * k,
      };
    };

    const app: spine.SpineCanvasApp = {
      loadAssets(c) {
        c.assetManager.loadBinary(`${rigBase}.skel`);
        c.assetManager.loadTextureAtlas(`${rigBase}.atlas`);
      },
      initialize(c) {
        const am = c.assetManager;
        if (am.hasErrors()) { setFailed(true); return; }
        try {
          const atlas = am.require(`${rigBase}.atlas`);
          const data = new spine.SkeletonBinary(new spine.AtlasAttachmentLoader(atlas))
            .readSkeletonData(am.require(`${rigBase}.skel`));
          dataRef.current = data;
          skeleton = new spine.Skeleton(data);
          const st = new spine.AnimationState(new spine.AnimationStateData(data));
          setIdle(st, data);
          skeleton.setToSetupPose();
          st.apply(skeleton);
          hideFx(skeleton);
          skeleton.updateWorldTransform(spine.Physics.update);
          skeleton.getBounds(off, size, []);
          stateRef.current = st;
          // Spine events, straight off the runtime's own listener API. The
          // rig authors ten (`claw_explode_l`, `plow_chunks_*`, `death_spit_*`,
          // `charge_steam_*`, `regen_splats_*`, `rocket_thrust_*`,
          // `rocket_throttle`) and the game drives real VFX off all of them;
          // the parent decides which it cares about. See `onRigEvent`'s doc.
          st.addListener({
            event: (_entry, ev) => { onRigEventRef.current?.(ev.data.name, boneViewportPoint); },
          });
          // Anything that arrived while this rig was (re)loading — see
          // `pendingArmRef`. Replayed AFTER the idle tracks are set, so the
          // one-shot lands on top of the idle rather than under it.
          const pending = pendingArmRef.current;
          pendingArmRef.current = {};
          for (const [track, p] of Object.entries(pending)) {
            playArmRef.current?.(Number(track), p.side, p.action, p.prev);
          }
          // The idle bounds plus generous cosmetic overscan (the per-arm attack
          // lunges extend past idle, but framing every track's full envelope is
          // overkill here — the overscan covers it).
          const sideMw = OVER_SIDE * size.x;
          const topMw = OVER_TOP * size.y;
          const botMw = OVER_BOT * size.y;
          frameRef.current = { sideMw, topMw, botMw };
          // Each claw's PAINTED box, in rig units relative to the root. Bone
          // positions are not enough: a claw's bone sits at its shoulder end,
          // so pinning the readouts to it parked the Crusher's HP bar in the
          // middle of its own pincer. Region and mesh attachments both hand
          // back world vertices, which is what the renderer draws.
          const verts = new Float32Array(1024);
          const clawBox = (prefix: string): ClawBox | null => {
            let l = Infinity, r = -Infinity, b = Infinity, t = -Infinity;
            for (const sl of skeleton!.slots) {
              if (!sl.data.name.startsWith(prefix)) continue;
              const a = sl.getAttachment();
              let n = 0;
              if (a instanceof spine.RegionAttachment) {
                a.computeWorldVertices(sl, verts, 0, 2); n = 8;
              } else if (a instanceof spine.MeshAttachment) {
                n = Math.min(a.worldVerticesLength, verts.length);
                a.computeWorldVertices(sl, 0, n, verts, 0, 2);
              } else continue;
              for (let i = 0; i < n; i += 2) {
                l = Math.min(l, verts[i]); r = Math.max(r, verts[i]);
                b = Math.min(b, verts[i + 1]); t = Math.max(t, verts[i + 1]);
              }
            }
            if (!Number.isFinite(l)) return null;
            return { cx: (l + r) / 2, left: l, right: r, top: t, bottom: b };
          };
          // Fall back to a quarter/three-quarter split of the body if a claw has
          // no slots to measure.
          const half = (sign: number): ClawBox => ({
            cx: off.x + size.x * (sign < 0 ? 0.2 : 0.8),
            left: off.x + size.x * (sign < 0 ? 0.05 : 0.65),
            right: off.x + size.x * (sign < 0 ? 0.35 : 0.95),
            top: off.y + size.y * 0.5, bottom: off.y,
          });
          setWorld({
            x: off.x, y: off.y, w: size.x, h: size.y, sideMw, topMw, botMw,
            claw: {
              left: clawBox("claw_l") ?? half(-1),
              right: clawBox("claw_r") ?? half(1),
            },
          });
        } catch { setFailed(true); }
      },
      update(_c, delta) {
        const st = stateRef.current;
        if (!st || !skeleton) return;
        st.update(delta);
        st.apply(skeleton);
        if (!inActionRef.current[LEFT_TRACK] && !inActionRef.current[RIGHT_TRACK]) hideFx(skeleton);
        skeleton.update(delta);
        skeleton.updateWorldTransform(spine.Physics.update);
      },
      render(c) {
        if (!skeleton || size.y <= 0) return;
        const r = c.renderer, gl = c.gl;
        gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
        r.resize(spine.ResizeMode.Expand);
        const fr = frameRef.current;
        const sideMw = fr ? fr.sideMw : OVER_SIDE * size.x;
        const topMw = fr ? fr.topMw : OVER_TOP * size.y;
        const botMw = fr ? fr.botMw : OVER_BOT * size.y;
        const vw = size.x + 2 * sideMw;
        const vh = size.y + topMw + botMw;
        r.camera.position.x = off.x + size.x / 2;
        r.camera.position.y = (off.y - botMw) + vh / 2;
        r.camera.viewportWidth = vw; r.camera.viewportHeight = vh; r.camera.update();
        r.begin();
        r.drawSkeleton(skeleton, false);
        r.end();
      },
      error() { setFailed(true); measureRef.current?.(null); },
    };

    let canvas: spine.SpineCanvas | null = null;
    try {
      canvas = new spine.SpineCanvas(canvasEl, { app, webglConfig: { alpha: true, premultipliedAlpha: false } });
    } catch { setFailed(true); }
    return () => { canvas?.dispose(); stateRef.current = null; dataRef.current = null; };
    // `pxPerUnit` is NOT a dependency: everything measured above is in rig
    // units, so re-fitting the crab to a resized stage is a render-time rescale
    // (see below), not a reload of four atlas pages.
  }, [rigBase]);

  // Per-arm one-shot driver: plays "<side>/<clip>" on the arm's track, then
  // resumes that arm's idle (unless hold = death).
  const playArm = (track: number, side: string, action?: ArmAction, prev?: React.MutableRefObject<number | undefined>) => {
    if (!action || !prev) return;
    if (prev.current === action.nonce) return;
    const st = stateRef.current, data = dataRef.current;
    if (!st || !data) {
      // Rig not loaded yet. Hold the action rather than marking it played —
      // see `pendingArmRef` for the death this used to eat.
      pendingArmRef.current[track] = { side, action, prev };
      return;
    }
    prev.current = action.nonce;
    const names = data.animations.map((a) => a.name);
    const want = `${side}/${action.clip}`;
    // Exact, then prefix (so "attack" → "attack_heavy"/"attack_med" variants).
    const clip = names.find((n) => n === want)
      ?? names.find((n) => n.startsWith(want))
      ?? names.find((n) => n.startsWith(`${side}/${action.clip.split("_")[0]}`));
    if (!clip) return;
    inActionRef.current[track] = true;
    st.setAnimation(track, clip, false);
    if (!action.hold) {
      const idle = names.includes(`${side}/idle_loop`) ? `${side}/idle_loop` : undefined;
      if (idle) st.addAnimation(track, idle, true, 0);
    }
    const dur = (data.findAnimation(clip)?.duration ?? 0.5) * 1000;
    window.setTimeout(() => { inActionRef.current[track] = false; }, dur + 50);
  };

  playArmRef.current = playArm;

  const leftNonce = leftAction?.nonce, rightNonce = rightAction?.nonce;
  useEffect(() => { playArmRef.current?.(LEFT_TRACK, "left", leftActionRef.current, prevLeft); }, [leftNonce]);
  useEffect(() => { playArmRef.current?.(RIGHT_TRACK, "right", rightActionRef.current, prevRight); }, [rightNonce]);

  const dropShadow = "drop-shadow(0 12px 10px rgba(0,0,0,.55))";
  // Rig units → px, per render. No box fit: the authored scale IS the size.
  const k = pxPerUnit;
  const canvasW = world ? (world.w + 2 * world.sideMw) * k : 300;
  const canvasH = world ? (world.h + world.topMw + world.botMw) * k : 300;
  // Canvas edges relative to the rig ROOT, in px (+x right, +y up).
  const canvasLeft = world ? (world.x - world.sideMw) * k : -150;
  const canvasBottom = world ? (world.y - world.botMw) * k : 0;
  // Reported from an effect so a re-fit re-reports without the rig reloading.
  // `onMeasure` is a parent setState; keying on the numbers (not the object)
  // keeps that from looping. (`measureRef` itself is declared above the load
  // effect, which reports a failed load through it.)
  const scaleBox = (b: ClawBox): ClawBox => ({
    cx: b.cx * k, left: b.left * k, right: b.right * k, top: b.top * k, bottom: b.bottom * k,
  });
  const measure: CrabMeasure | null = world && !failed ? {
    canvasW, canvasH, canvasLeft, canvasBottom,
    top: (world.y + world.h) * k, bodyH: world.h * k,
    claw: { left: scaleBox(world.claw.left), right: scaleBox(world.claw.right) },
  } : null;
  const measureRest = useRef(measure); measureRest.current = measure;
  const measureKey = measure
    ? `${canvasW},${canvasH},${canvasLeft},${canvasBottom},${measure.claw.left.cx},${measure.claw.right.cx}`
    : "";
  useEffect(() => { measureRef.current?.(measureRest.current); }, [measureKey]);
  return (
    <canvas
      ref={canvasRef}
      width={Math.max(64, Math.round(canvasW * 1.3))}
      height={Math.max(64, Math.round(canvasH * 1.3))}
      style={{
        position: "absolute", left: canvasLeft, bottom: canvasBottom,
        width: canvasW, height: canvasH,
        opacity: world ? 1 : 0, transition: "opacity 140ms ease-out",
        display: failed ? "none" : "block", pointerEvents: "none", filter: dropShadow,
      }}
    />
  );
}
