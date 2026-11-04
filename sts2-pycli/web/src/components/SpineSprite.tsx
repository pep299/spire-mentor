import { useEffect, useRef, useState } from "react";
import * as spine from "@esotericsoftware/spine-webgl";
import { effectAssetUrl, type DissolveEffect } from "../effects/manifest";
import {
  createDissolveProgram, createGradientTexture, createRigFramebuffer,
  dissolveThreshold, drawDissolveQuad, loadImage,
  type DissolveProgram, type RigFramebuffer,
} from "../effects/dissolve";

/**
 * Renders a live Spine rig (the real in-game character) into a WebGL canvas,
 * looping `animation` (default idle). Falls back to a static image while loading
 * or if WebGL / the rig fails. STS2 atlases are straight (non-premultiplied)
 * alpha, so we render with premultipliedAlpha=false (else region edges halo).
 *
 * `rigBase` is the URL base without extension; we append .skel/.atlas/.png.
 *
 * Proportions: every combatant (player + enemies) is rendered at
 * `worldScale * pxPerScene` pixels per rig-unit, where `worldScale` is the
 * creature's authored SCENE scale (e.g. Ironclad 0.28, Nibbit 0.28 but a much
 * smaller rig, Defect 0.125) and `pxPerScene` maps the game's 1920x1080 scene
 * onto the box we actually got (see src/scene.ts). That product is the game's
 * own sizing rule, not an approximation of it: the authored scale IS pixels per
 * rig unit in the game's scene, so a rig drawn this way is the size the game
 * draws it, and every creature is to scale against every other. The rigs are
 * authored already facing the player (the game never flips them), so we don't
 * flip either.
 *
 * Framing: the canvas is sized to *contain* the whole idle body (width too, so
 * wide monsters' tails aren't clipped) plus an overscan margin for the attack
 * lunge / raised weapon, bottom-anchored at the feet so it doesn't cover the
 * HP bar below.
 *
 * One-shot actions: when `actionNonce` changes, `actionAnim` (e.g. "attack",
 * "hurt") plays once, then idle resumes.
 */
// Overscan margins (fractions of the body box) for the canvas around the idle.
const OVER_TOP = 0.34;  // headroom (raised weapons / jumps)
const OVER_SIDE = 0.28; // each side (lunge)
const OVER_BOT = 0.05;  // ground room below the feet

// Some rigs keep their body (or parts of it) in NAMED skins rather than the
// default skin, and the game composes a skin at spawn instead of rendering the
// bare default: Scroll of Biting holds its whole body in skin1/skin2 (the
// default skin is just the shadow/FX), and Cubex Construct combines a RANDOM
// eye + moss skin over its body (see CubexConstruct.SetupSkins). A viewer that
// never sets a skin renders those setup slots empty — Scroll shows nothing;
// Cubex shows only its floating orb, which reads as a "pearl". Detect every
// setup slot the default skin can't provide and compose a skin that fills each
// from the first named skin that supplies it (one eye, one moss, one body
// variant — never stacking duplicates). No-op for the vast majority of rigs,
// whose default skin already covers every slot.
function composeSetupSkin(skeleton: spine.Skeleton, data: spine.SkeletonData): void {
  const def = data.defaultSkin;
  const missing: Array<{ index: number; name: string }> = [];
  for (let i = 0; i < data.slots.length; i++) {
    const attName = data.slots[i].attachmentName;
    if (attName && !(def && def.getAttachment(i, attName))) missing.push({ index: i, name: attName });
  }
  if (!missing.length) return;
  const named = data.skins.filter((s) => s !== def);
  if (!named.length) return;
  const composed = new spine.Skin("__composed");
  if (def) composed.addSkin(def);
  for (const { index, name } of missing) {
    // Already provided by a skin we've merged in (e.g. one moss skin covers both
    // its top+bottom slots) — don't merge a second skin for the same visual.
    if (composed.getAttachment(index, name)) continue;
    const src = named.find((s) => s.getAttachment(index, name));
    if (src) composed.addSkin(src);
  }
  skeleton.setSkin(composed);
  skeleton.setSlotsToSetupPose();
}

export default function SpineSprite({ rigBase, animation = "idle_loop", holdSetup = false, worldScale, pxPerScene = 1, boundsH, maxBodyH, fallbackHeight = 200, fallback, alt, actionAnim, actionNonce, actionAnimCandidates, holdLast, onClips, onMeasure, onFrame, flipX, skin, hueShift, dissolve }: {
  rigBase: string; animation?: string;
  /** Hold the rig's SETUP pose instead of looping an idle, until an
   *  `actionNonce` fires. The idle picker falls through to `animations[0]` when
   *  nothing matches, so a rig with a single clip has no other way to sit
   *  still: the treasure chest's one clip is its OPENING, and looping it left
   *  the chest permanently open on a screen whose whole point is opening it. */
  holdSetup?: boolean;
  worldScale?: number | null;
  /** Stage px per SCENE px — the factor that maps the game's 1920x1080 combat
   *  scene onto the stage this rig is drawn in (`scene.fitScene().k`). Both
   *  `worldScale` and `boundsH` are in scene units, so this is the only place
   *  the rendering knows how big the window is. 1 outside the arena (the
   *  preview harnesses, whose props are already in px). */
  pxPerScene?: number;
  boundsH?: number | null;
  /** Tallest body, in px, the arena has room for from the ground line up (see
   *  Board's `maxBodyH`). A creature bigger than this is drawn scaled DOWN to
   *  it rather than with its head cut off above the stage: the game's scene is
   *  ~1005px tall and ours is ~818, so the biggest rigs — the Bygone Effigy's
   *  measured body is 503px against the ~392px the arena can give it — simply
   *  do not fit at the authored scale, and nothing used to notice. Its layout box is capped with
   *  it, which is also what stops three tall enemies (box width is 0.7 × box
   *  height) from running off the side of the stage.
   *
   *  Left undefined outside the arena — the preview harnesses and the treasure
   *  chest have their own boxes and no ground line to fit against. */
  maxBodyH?: number | null;
  fallbackHeight?: number; fallback?: string | null; alt?: string;
  // Named Spine skin to wear (monster VARIANTS share a rig and differ only by
  // skin — the game's MonsterModel.SetupSkins; see src/monsterSkins.ts). The
  // default skin still backs every slot the named skin doesn't override, same
  // as the game's SetSkin. Missing/unknown names fall back silently to the
  // default composed look — never an error.
  skin?: string;
  // Mirror the rig horizontally. Most STS2 rigs are authored already facing the
  // player so we never flip — but some (the Decimillipede segments) are authored
  // facing away and the game renders them with a negative X scale.
  flipX?: boolean;
  actionAnim?: string; actionNonce?: number;
  actionAnimCandidates?: string[];  // first available is played for the one-shot
  holdLast?: boolean;               // don't resume idle after the one-shot (death)
  onClips?: (has: (n: string) => boolean) => void;  // report rig's clip set once loaded
  // Reports the y-offset (px, from the layout box's TOP) of the loaded rig's
  // visual top (its head). The reserved box is sized from the authored boundsH,
  // which is not a reliable proxy for the rig's measured body, so the box is
  // usually taller (sometimes shorter) than the rig. Overlays (intent, block,
  // hotkey) that the parent pins to the box top would otherwise float detached
  // above small enemies; the parent uses this offset to pin them to the rig
  // head instead. `null` = not measured yet / failed (parent treats as 0).
  // Called TWICE per load: first with the analytic bounds value at initialize,
  // then again once the first frame has been rendered and its alpha scanned for
  // the top row that actually carries paint (see render()). The refinement is
  // what makes the anchor land on the VISIBLE body rather than on the rig's
  // (often much larger) attachment bounds.
  onMeasure?: (headTopPx: number | null) => void;
  // Reports the rendered footprint (px) once measured, so a parent that composes
  // several rigs (the Decimillipede segment strip) can lay them out by their real
  // sizes. `bodyW`/`bodyH` are the measured body (no overscan); `canvasW`/
  // `canvasH` include the overscan margin. `null` = not measured / failed.
  onFrame?: (f: { canvasW: number; canvasH: number; bodyW: number; bodyH: number } | null) => void;
  // Per-instance hue offset in GODOT hue units (0..1), from
  // `NCombatRoom.RandomizeEnemyScalesAndHues`: repeated monsters of one type are
  // tinted apart so a row of five reads as five animals rather than one stamped
  // five times. Absent/0 leaves the rig's own colours alone.
  hueShift?: number | null;
  // The dying-creature dissolve (Task 14): `vfx_monster_death`'s manifest
  // entry plus WHEN this instance's dissolve tween started
  // (`performance.now()` at the `death` event, held by the caller — see
  // `Board.tsx`'s `dying` state's `fadedAt`). `rev` is the manifest's own
  // cache-busting revision, needed to fetch the two baked gradient PNGs via
  // `effectAssetUrl`, same as every other manifest-named asset.
  //
  // `null`/`undefined` (not currently dying, or the manifest hasn't loaded /
  // lacks the entry) leaves `render()`'s plain path untouched — no
  // framebuffer is ever allocated for a sprite that never dissolves.
  // Deliberately NOT a dependency of the load effect below: this changes far
  // more often (every death) than a rig reload should ever be triggered by,
  // so it is read from a ref inside `render()` instead — the same pattern
  // `sceneScaleRef` uses in `VfxLayer.tsx` for a per-frame value a draw loop
  // needs but a load effect must not restart over.
  dissolve?: { effect: DissolveEffect; rev: string; startedAt: number } | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<spine.AnimationState | null>(null);
  const idleRef = useRef<string>(animation);
  // The load effect deliberately does not re-run on these (see its deps);
  // it reads their latest values when the rig finishes loading.
  const actionAnimRef = useRef(actionAnim); actionAnimRef.current = actionAnim;
  const onClipsRef = useRef(onClips); onClipsRef.current = onClips;
  const prevNonce = useRef<number | undefined>(undefined);
  const inActionRef = useRef(false);
  // Frame margins (WORLD units) around the idle body, sized to contain EVERY
  // animation the rig can play (idle/attack/hurt/die) so no pose is ever
  // cropped. Symmetric horizontally → the idle body stays centred. render()
  // reads this; it is computed once on load.
  const frameRef = useRef<{ sideMw: number; topMw: number; botMw: number } | null>(null);
  // What the visible-ink scan in render() needs, as a SCALE-FREE ratio (the
  // body's share of the canvas width). render() is built inside the load effect
  // and can never read React state, hence a ref — and keeping it scale-free is
  // what lets a re-fit leave the loaded rig alone.
  const inkRef = useRef<{ bodyFrac: number } | null>(null);
  const visDoneRef = useRef(false);   // visible top already reported for this rig
  const visTriesRef = useRef(0);      // give up rather than scan every frame forever
  const visBufRef = useRef<Uint8Array | null>(null);  // reused readback buffer
  const visSkipRef = useRef(0);       // scan every Nth frame, not every frame
  // Measured idle body size + the same margins, in WORLD units, set on load.
  // Deliberately NOT in px: `maxBodyH` is a live measurement of the arena (Board
  // re-measures it on every arena resize — the hand strip's height feeds it), so
  // the px factor `k` below changes while a rig is on screen. Storing the world
  // box and scaling it at render time makes a re-fit a pure canvas resize; when
  // the px sizes lived here, `maxBodyH` had to be a dependency of the load
  // effect, and every re-fit disposed the canvas and re-fetched, re-parsed and
  // re-scanned the whole rig. On the giants — the only rigs `maxBodyH` actually
  // clamps — that was the boss blinking out for a second and coming back at a
  // new size, with its layout box moving the column under it.
  const [world, setWorld] = useState<{ x: number; y: number; w: number; h: number; sideMw: number; topMw: number; botMw: number } | null>(null);
  // Where the rendered rig's visible ink starts, as a fraction of the canvas
  // height measured DOWN from its top. Scale-free for the same reason.
  const [inkTop, setInkTop] = useState<number | null>(null);
  const [failed, setFailed] = useState(false);
  // Synced every render (not a load-effect dependency — see the `dissolve`
  // prop's own doc): `render()`, built once inside the load effect below,
  // reads this fresh each frame instead of closing over a stale prop value.
  const dissolveRef = useRef<{ effect: DissolveEffect; rev: string; startedAt: number } | null>(null);
  dissolveRef.current = dissolve ?? null;

  useEffect(() => {
    const canvasEl = canvasRef.current;
    if (!canvasEl) return;
    setWorld(null); setInkTop(null); setFailed(false); frameRef.current = null;
    inkRef.current = null; visDoneRef.current = false; visTriesRef.current = 0;
    visBufRef.current = null; visSkipRef.current = 0;

    let skeleton: spine.Skeleton | null = null;
    // The dissolve post-pass's own GL resources (Task 14) — created lazily,
    // the first frame `dissolveRef.current` is non-null, and reused for
    // every subsequent dissolving frame of THIS rig load. `null` means
    // either "never dissolved yet" or "not created yet"; either way
    // `render()` below allocates on first need and never for a rig that
    // never dies. Tied to this effect's own closure (same lifetime as
    // `skeleton`/`canvas`/`gl`), not a ref — nothing outside `render()` ever
    // reads it.
    let dissolveGl: {
      program: DissolveProgram;
      fbo: RigFramebuffer | null;
      gradientTex: readonly [WebGLTexture, WebGLTexture] | null;
      gradientKey: string | null;
    } | null = null;
    // Set in this effect's cleanup so an in-flight gradient-image fetch
    // (below) never writes into a `dissolveGl` whose rig has already been
    // torn down or reloaded.
    let dissolveDisposed = false;
    const off = new spine.Vector2(), size = new spine.Vector2();
    // Big VFX / ground-shadow slots are hidden (keeps framing tight on the
    // body+weapon and avoids huge slash flashes). `^mega` covers oversized
    // attack-only overlays (Vantom's megatail/megablade/megashine) that the rig
    // parks far above the body in setup pose and only swings into frame during
    // its multi-track attack — which this single-track viewer never reproduces,
    // so they'd otherwise just dangle from the top of the canvas.
    //
    // Soul Nexus is the same kind of bug: its "trail" souls — the `straight
    // soul`/`straight wave` meshes, their `soulface` overlays, the `light squig`
    // wisps and the dim off-screen `glowie2` duplicate — are parked ~900-2200u
    // off the body in setup/idle and only travel along their path constraints
    // during cast/attack (the path_N_start trail events). Idle leaves them at
    // ~0.85 alpha out in space, so they render as detached purple shapes around
    // the eye. These are slot-name-specific (the on-body souls are `soul N` /
    // `collar soul N` / `collarface N` / the centred `glowie`, none of which
    // match), and hiding only the rendered mesh leaves the path constraints
    // intact. Like `^mega`, hiding is skipped during an action one-shot, so a
    // real cast/attack still swings the souls into frame.
    const HIDE = [/slash/i, /zap/i, /whoosh/i, /windpath/i, /path\b/i, /projectile/i, /^shadow/i, /chadow/i, /ground/i, /^mega/i, /^straight /i, /^soulface/i, /^light squig/i, /^glowie2/i];
    const hideFx = (sk: spine.Skeleton) => {
      for (const slot of sk.slots) {
        const a = slot.getAttachment();
        if (!a) continue;
        // Only hide RENDERED visuals (region/mesh). Path/clipping/bounding/point
        // attachments are invisible guides that drive constraints — nulling one
        // silently breaks the constraint that follows it. Vantom's tail and whip
        // are positioned by PathConstraints whose guide slots ("tail_path",
        // "whip_path") match the /path\b/ VFX filter, so hiding them froze the
        // tail bones at their off-body setup coords (the detached/hanging tail).
        if (!(a instanceof spine.RegionAttachment || a instanceof spine.MeshAttachment)) continue;
        if (HIDE.some((p) => p.test(slot.data.name) || p.test(a.name ?? ""))) slot.setAttachment(null);
      }
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
          skeleton = new spine.Skeleton(data);
          // Variant skin (the game's SetupSkins pick) when requested AND the
          // rig actually has it; otherwise the generic setup-slot compose.
          // Not both: composeSetupSkin diffs against the DEFAULT skin and
          // would overwrite a variant with the first named skin it finds.
          const variantSkin = skin ? data.findSkin(skin) : null;
          if (variantSkin) {
            skeleton.setSkin(variantSkin);
            skeleton.setSlotsToSetupPose();
          } else {
            composeSetupSkin(skeleton, data);
          }
          // Exact idle candidates first, then a prefix match — some rigs only
          // have per-phase/variant idles ("idle_loop1/2/3" on the Test
          // Subject's three-form skeleton), and falling through to
          // animations[0] lands on an arbitrary clip (Test Subject:
          // "--knockout3", a sprawled knocked-out pose).
          const names = data.animations.map((a) => a.name);
          const idleCandidates = [animation, "idle_loop", "idle", "Idle_loop", "Idle"];
          const idle = holdSetup
            ? undefined
            : idleCandidates.find((n) => names.includes(n))
              ?? idleCandidates.map((c) => names.find((n) => n.startsWith(c))).find(Boolean)
              ?? names[0];
          idleRef.current = idle ?? "";
          onClipsRef.current?.((n: string) => !!data.findAnimation(n));
          // Measure the IDLE pose (FX hidden) to know the body box for framing.
          const st = new spine.AnimationState(new spine.AnimationStateData(data));
          if (idle) st.setAnimation(0, idle, true);
          skeleton.setToSetupPose();
          st.apply(skeleton);
          hideFx(skeleton);
          skeleton.updateWorldTransform(spine.Physics.update);
          skeleton.getBounds(off, size, []);
          stateRef.current = st;
          // Union the AABB across every animation this rig can play, so the
          // canvas/camera contains the FULL motion envelope — a lunge (attack)
          // or recoil (hurt) can move the skeleton well past the idle bounds,
          // which would otherwise be cropped at the canvas edge.
          let uMinX = off.x, uMinY = off.y, uMaxX = off.x + size.x, uMaxY = off.y + size.y;
          const scan = new spine.AnimationState(new spine.AnimationStateData(data));
          const o = new spine.Vector2(), sz = new spine.Vector2();
          // Resolve each scan candidate exact-then-prefix, mirroring the
          // one-shot clip resolution below — so a rig whose clips are
          // numbered variants ("attack_big1", "hurt2") still gets its real
          // action envelope contained instead of silently skipping the scan.
          const scanNames = Array.from(new Set(
            [idleRef.current, actionAnimRef.current, "attack", "hurt", "hit", "die"]
              .filter((c): c is string => !!c)
              .map((c) => names.find((n) => n === c || n.startsWith(c)))
              .filter((n): n is string => !!n)
          ));
          for (const nm of scanNames) {
            const anim = data.findAnimation(nm);
            if (!anim) continue;
            // A death clip sinks the body into the ground (the sandpit collapses,
            // a boss falls through the floor). That huge downward travel must NOT
            // count toward the bottom margin: botMw anchors the *idle* feet above
            // the viewport bottom, so inflating it lifts the resting body off the
            // ground line — The Insatiable (die sinks ~1632u) and Infested
            // Guardian (~2652u) floated near the top, off-screen. The sink is
            // meant to disappear into the floor, so let it clip there instead.
            const isDeath = /(^|[_/])(die|death|defeat)/i.test(nm);
            scan.setAnimation(0, nm, false);
            const dur = Math.max(anim.duration, 0.0001), STEPS = 14;
            for (let i = 0; i <= STEPS; i++) {
              skeleton.setToSetupPose();
              scan.apply(skeleton);
              hideFx(skeleton);
              skeleton.updateWorldTransform(spine.Physics.update);
              skeleton.getBounds(o, sz, []);
              if (sz.x > 0 && sz.y > 0) {
                uMinX = Math.min(uMinX, o.x);
                if (!isDeath) uMinY = Math.min(uMinY, o.y);
                uMaxX = Math.max(uMaxX, o.x + sz.x); uMaxY = Math.max(uMaxY, o.y + sz.y);
              }
              scan.update(dur / STEPS);
            }
          }
          // Restore the live idle pose (the scan mutated the skeleton).
          skeleton.setToSetupPose(); st.apply(skeleton); hideFx(skeleton);
          skeleton.updateWorldTransform(spine.Physics.update);

          // Margins (world units) the motion needs beyond the idle box, floored
          // at the cosmetic overscan. Symmetric L/R keeps the idle centred while
          // still containing an asymmetric lunge.
          const leftExt = Math.max(0, off.x - uMinX);
          const rightExt = Math.max(0, uMaxX - (off.x + size.x));
          const topExt = Math.max(0, uMaxY - (off.y + size.y));
          const botExt = Math.max(0, off.y - uMinY);
          const sideMw = Math.max(OVER_SIDE * size.x, leftExt, rightExt);
          const topMw = Math.max(OVER_TOP * size.y, topExt);
          const botMw = Math.max(OVER_BOT * size.y, botExt);
          frameRef.current = { sideMw, topMw, botMw };
          // World units only — the px sizes, the head offset the parent pins its
          // overlays to, and the footprint reported to `onFrame` are all derived
          // from this below, so they follow a re-fit without a reload.
          setWorld({ x: off.x, y: off.y, w: size.x, h: size.y, sideMw, topMw, botMw });
          // …and hand render() what its visible-ink scan needs, likewise scale
          // free: the body's share of the canvas width.
          inkRef.current = { bodyFrac: size.x / Math.max(1, size.x + 2 * sideMw) };
        } catch { setFailed(true); }
      },
      update(_c, delta) {
        const st = stateRef.current;
        if (!st || !skeleton) return;
        st.update(delta); st.apply(skeleton);
        if (!inActionRef.current) hideFx(skeleton);
        // Advance the skeleton's physics clock — without this, spring bones
        // (ears, tails) see a wrong timestep and oscillate way too fast.
        skeleton.update(delta);
        skeleton.updateWorldTransform(spine.Physics.update);
      },
      render(c) {
        if (!skeleton || size.y <= 0) return;
        const sk = skeleton; // stable non-null binding for the nested draw helper below
        const r = c.renderer, gl = c.gl;
        // Renders the live rig into WHATEVER framebuffer is currently bound,
        // using the same camera framing either way — the plain path below
        // binds nothing (the default framebuffer) and the dissolve path
        // binds its own FBO first, so the captured frame lines up pixel-for-
        // pixel with what the plain path would have drawn straight to the
        // canvas. Viewport (world units) = canvas size / pxPerUnit; since the
        // canvas CSS is the body box * (1 + overscan), the world viewport is
        // exactly the body bounds plus the same overscan — so nothing is
        // distorted or cropped regardless of the body's aspect ratio.
        // Margins from the load-time animation scan (contain every pose);
        // fall back to the cosmetic overscan before the scan has run.
        const drawSkeletonToCurrentTarget = () => {
          const fr = frameRef.current;
          const sideMw = fr ? fr.sideMw : OVER_SIDE * size.x;
          const topMw = fr ? fr.topMw : OVER_TOP * size.y;
          const botMw = fr ? fr.botMw : OVER_BOT * size.y;
          const vw = size.x + 2 * sideMw;
          const vh = size.y + topMw + botMw;
          r.camera.position.x = off.x + size.x / 2;          // centre horizontally
          r.camera.position.y = (off.y - botMw) + vh / 2;     // feet near the bottom
          r.camera.viewportWidth = vw; r.camera.viewportHeight = vh; r.camera.update();
          r.begin();
          r.drawSkeleton(sk, false); // false = straight (non-premultiplied) alpha
          r.end();
        };

        const dissolving = dissolveRef.current;
        if (!dissolving) {
          // Untouched plain path: no framebuffer is ever allocated for a
          // sprite that never dissolves.
          gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
          r.resize(spine.ResizeMode.Expand);
          drawSkeletonToCurrentTarget();
        } else {
          // Task 14: the dying-creature dissolve — see `effects/dissolve.ts`'s
          // module doc for the shader transcription and why `threshold` runs
          // 1 -> 0.
          if (!dissolveGl) {
            dissolveGl = { program: createDissolveProgram(gl), fbo: null, gradientTex: null, gradientKey: null };
          }
          const dGl = dissolveGl;
          r.resize(spine.ResizeMode.Expand); // sizes canvas.width/height + gl.viewport
          // Fetch + upload both baked gradients once per effect instance.
          // Keyed by the path pair so a (never-expected, but possible)
          // manifest change to `vfx_monster_death` doesn't wedge on the
          // first pair forever. Same leak-on-supersede risk as the FBO
          // above — delete the OLD pair (if the previous key ever finished
          // loading one) before dropping the reference to it.
          const gradKey = `${dissolving.effect.gradient1}|${dissolving.effect.gradient2}`;
          if (dGl.gradientKey !== gradKey) {
            const oldGradientTex = dGl.gradientTex;
            dGl.gradientKey = gradKey;
            dGl.gradientTex = null;
            if (oldGradientTex) {
              gl.deleteTexture(oldGradientTex[0]);
              gl.deleteTexture(oldGradientTex[1]);
            }
            Promise.all([
              loadImage(effectAssetUrl(dissolving.effect.gradient1, dissolving.rev)),
              loadImage(effectAssetUrl(dissolving.effect.gradient2, dissolving.rev)),
            ])
              .then(([img1, img2]) => {
                // A rig reload/unmount, or a newer gradient pair already
                // requested, between the fetch starting and finishing —
                // drop this result, the same "still loading, skip it"
                // contract every other async asset load in this app
                // follows (see this file's own atlas/skeleton race).
                if (dissolveDisposed || dGl.gradientKey !== gradKey) return;
                dGl.gradientTex = [createGradientTexture(gl, img1), createGradientTexture(gl, img2)];
              })
              .catch((err: unknown) => console.error("dissolve gradient load failed", err));
          }

          const gradientTex = dGl.gradientTex;
          if (!gradientTex) {
            // Gradients still in flight. The fetch above starts on the FIRST
            // dissolving frame and nothing preloads it, so this covers a real
            // network round-trip at the very start of the very first death of
            // a page session — and at that point the dissolve has barely
            // begun (`threshold` ~= 1, i.e. NOTHING eaten away yet), so the
            // frame the viewer is owed is the plain, fully-visible rig. Draw
            // exactly that, on the untouched no-dissolve path below: skipping
            // the draw instead blinks the corpse out of existence for the
            // round-trip and then pops it back in already half-gone.
            //
            // No FBO is allocated on this path either — a dissolve that never
            // gets its gradients (a 404 on the pair) then costs nothing.
            gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
            drawSkeletonToCurrentTarget();
          } else {
            // Snapshot the live rig into an off-screen texture sized to the
            // drawing buffer, then eat it away with the ported
            // `shaders/dissolve.gdshader` as a full-canvas post-pass.
            //
            // (Re)allocate the FBO to the current drawing buffer — the canvas
            // can resize (an arena re-fit) while a corpse is mid-dissolve.
            // Delete the SUPERSEDED framebuffer/texture first (review round 2,
            // finding 2): unlike the "no explicit disposal at unmount"
            // convention this file's rig load otherwise follows (bounded by
            // GC once per rig), this can fire repeatedly within one ~2.5s
            // dissolve and would otherwise leak an FBO/texture per re-fit.
            const dw = gl.drawingBufferWidth, dh = gl.drawingBufferHeight;
            let fbo = dGl.fbo;
            if (!fbo || fbo.width !== dw || fbo.height !== dh) {
              if (fbo) {
                gl.deleteFramebuffer(fbo.framebuffer);
                gl.deleteTexture(fbo.texture);
              }
              fbo = createRigFramebuffer(gl, dw, dh);
              dGl.fbo = fbo;
            }
            gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.framebuffer);
            gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
            drawSkeletonToCurrentTarget();
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);

            const elapsed = (performance.now() - dissolving.startedAt) / 1000;
            const threshold = dissolveThreshold(elapsed, dissolving.effect.dissolveSeconds);
            drawDissolveQuad(gl, dGl.program, fbo.texture, gradientTex[0], gradientTex[1], {
              threshold,
              time: elapsed,
              texelSize: [1 / fbo.width, 1 / fbo.height],
              modulateAlpha: dissolving.effect.modulateAlpha,
            });
          }
        }

        // ---- one-shot VISIBLE-ink measurement (refines the analytic --rig-top)
        // `getBounds` measures ATTACHMENT QUADS, not painted pixels: the atlas
        // regions carry transparent padding, and rigs park alpha-0 / off-body
        // meshes that the FX filter above doesn't name. On the slimes that
        // overscan is ~15px of empty air on top of a ~45-110px body, so the
        // intent/block/hotkey overlays — which pin to `--rig-top` — floated
        // detached above the creature (the r5 gauntlet's rank-2 combat tell).
        // Reading the frame's alpha back once gives the top row that actually
        // has paint in it, for any rig shape, with no per-monster tuning.
        // Same-frame readPixels, so no preserveDrawingBuffer is needed.
        // A rig whose early frames never clear the ink threshold spends its
        // whole retry budget here, and each try is a synchronous `readPixels`
        // stall inside the rAF callback plus a `bw*bh*4` throwaway (~2.5MB at
        // 400x400 @ DPR 2). With five enemies mounting at once that was ~12MB
        // of garbage per frame on combat entry. So: reuse ONE buffer, and
        // spread the 20 tries over every 4th frame — which also gives the pose
        // time to settle, making an early try more likely to find real ink.
        const g = inkRef.current;
        if (g && !visDoneRef.current && visTriesRef.current < 20
            && (visSkipRef.current++ & 3) === 0) {
          visTriesRef.current++;
          try {
            const bw = gl.drawingBufferWidth, bh = gl.drawingBufferHeight;
            const need = bw * bh * 4;
            let buf = visBufRef.current;
            if (!buf || buf.length !== need) { buf = new Uint8Array(need); visBufRef.current = buf; }
            gl.readPixels(0, 0, bw, bh, gl.RGBA, gl.UNSIGNED_BYTE, buf);
            // A row counts as "body" only once it carries ~8% of the rig's own
            // body width in ink. Anchoring to the literal first inked row lands
            // on whatever thinnest appendage points highest — the Twig Slime's
            // twig tips, the Mawler's tail tip — and the overlay then reads as
            // floating, because the eye anchors to the creature's MASS, not to
            // a 3px whisker. Scaling the rule by the rig's own width keeps it
            // shape-agnostic: a wide blob and a tall rig both resolve to the
            // top of their silhouette proper.
            const minInk = Math.max(3, Math.round(0.08 * g.bodyFrac * bw));
            // …and alpha 96, not "any non-zero alpha": the slimes are translucent
            // goo whose silhouette fades out over ~15px, so a 12% floor put the
            // anchor in air that composites to nothing on screen. 96/255 is
            // roughly where a pixel starts contributing to the read.
            const MIN_A = 96;
            // readPixels' origin is BOTTOM-left, so the visual top is the LAST row.
            let topRow = -1;
            for (let row = bh - 1; row >= 0 && topRow < 0; row--) {
              const base = row * bw * 4;
              let ink = 0;
              for (let x = 0; x < bw; x++) {
                if (buf[base + x * 4 + 3] >= MIN_A && ++ink >= minInk) { topRow = bh - 1 - row; break; }
              }
            }
            if (topRow >= 0) {
              visDoneRef.current = true;
              // As a fraction of the canvas height, so the head anchor derived
              // from it below survives a re-fit. (px would not: the canvas is
              // `k` times the world box, and `k` moves with the arena.)
              setInkTop(topRow / bh);
            }
          } catch { visDoneRef.current = true; }  // keep the analytic value
          // Done, or out of tries: drop the buffer rather than hold it for the
          // life of the rig.
          if (visDoneRef.current || visTriesRef.current >= 20) visBufRef.current = null;
        }
      },
      error() { setFailed(true); },
    };

    let canvas: spine.SpineCanvas | null = null;
    try {
      canvas = new spine.SpineCanvas(canvasEl, { app, webglConfig: { alpha: true, premultipliedAlpha: false } });
    } catch { setFailed(true); }
    return () => { canvas?.dispose(); stateRef.current = null; dissolveDisposed = true; };
    // NOTE: actionAnim is intentionally NOT a dependency — the load effect
    // disposes/recreates the whole rig, so reloading every time the action clip
    // flips (hurt↔attack↔die) would race with (and cancel) the one-shot below.
    // The bounds scan already frames the common action clips.
    //
    // Nor are the SIZING props (worldScale / pxPerScene / fallbackHeight /
    // boundsH / maxBodyH). Everything measured here is in world units, so how
    // big the rig is DRAWN is a render-time factor (`k` below), not something
    // the rig has to be rebuilt for. `pxPerScene` in particular is a live
    // measurement of the stage that moves while a fight is on screen.
  }, [rigBase, animation, holdSetup, skin]);

  // Play a one-shot action animation when actionNonce changes. Picks the first
  // available clip from the candidate list (or [actionAnim]); shows in-rig FX
  // during the clip; on death (holdLast) freezes the last frame instead of
  // resuming idle.
  useEffect(() => {
    const isNew = prevNonce.current !== actionNonce;
    prevNonce.current = actionNonce;
    if (!isNew || actionNonce == null) return;
    const st = stateRef.current;
    if (!st) return;
    const names = st.data.skeletonData.animations.map((a) => a.name);
    const candidates = actionAnimCandidates && actionAnimCandidates.length ? actionAnimCandidates : actionAnim ? [actionAnim] : [];
    // Exact match first, then a prefix match so a request for "attack" resolves
    // to a rig's numbered/variant clip ("attack2", "attack_heavy", …) — clip
    // naming is inconsistent across the extracted rigs.
    const clip = candidates.find((c) => names.includes(c))
      ?? candidates.map((c) => names.find((n) => n === c || n.startsWith(c))).find(Boolean);
    if (!clip) return;
    inActionRef.current = true;
    st.setAnimation(0, clip, false);
    // On death (holdLast), setAnimation(loop=false) freezes the last frame —
    // do not queue idle. Otherwise resume the idle loop after the one-shot.
    // On `holdSetup` there is no idle to go back to — the setup pose is the
    // rest state and the one-shot's last frame replaces it.
    if (!holdLast && idleRef.current) {
      st.addAnimation(0, idleRef.current, true, 0);
    }
    const dur = (st.data.skeletonData.findAnimation(clip)?.duration ?? 0.4) * 1000;
    const t = window.setTimeout(() => { inActionRef.current = false; }, dur + 50);
    return () => window.clearTimeout(t);
  }, [actionNonce, actionAnim, actionAnimCandidates, holdLast]);

  const dropShadow = "drop-shadow(0 10px 8px rgba(0,0,0,.5))";
  // Godot hue units are turns; CSS wants degrees.
  const spriteFilter = hueShift
    ? `${dropShadow} hue-rotate(${(hueShift * 360).toFixed(1)}deg)`
    : dropShadow;
  // Rig units → px. Proportional sizing (worldScale known) preserves player↔enemy
  // size; with no scale available (a rig with no creature_scales entry) the body
  // is normalized to `fallbackHeight` so it renders at a sane size instead of
  // full scale. …unless the creature is too big for the room, in which case it is
  // fitted: the alternative is the head clipped off above the stage, which is
  // what a Bygone Effigy did at every window size, desktop included.
  //
  // Computed per RENDER, not per load — see the `world` state above.
  let k = 1;
  if (world) {
    k = worldScale != null ? worldScale * pxPerScene : fallbackHeight / Math.max(1, world.h);
    if (maxBodyH != null && maxBodyH > 0 && world.h * k > maxBodyH) k = maxBodyH / Math.max(1, world.h);
  }
  const body = world && {
    w: world.w * k, h: world.h * k,
    sideM: world.sideMw * k, topM: world.topMw * k, botM: world.botMw * k,
  };
  // Layout box: the slot the combatant column reserves in flow. It is sized ONCE
  // from a stable pre-load hint (authored Bounds height boundsH * K, else the
  // generic fallback) and DELIBERATELY does not switch to the measured `body`
  // once the Spine rig loads. Tying it to the async measurement made the box
  // resize on load (typ. shrink), and because the arena is bottom-anchored
  // (`align-items: flex-end`) and content-sized, that shrink reflowed the whole
  // board below it — the hand/end-turn visibly jumped up by ~100px as each rig
  // finished loading. The canvas (below) is absolutely positioned and bottom-
  // anchored, so it renders the measured sprite — overscanning this box freely —
  // without perturbing layout. boundsH*K ≈ the visible canvas extent, so it is
  // also the more honest amount of space to reserve for the sprite.
  // Capped at `maxBodyH` for the same reason the rig is (above): an
  // uncapped box for a rig that no longer draws that tall reserves a slot
  // taller than the stage AND — via boxW — wider than its share of it.
  const boxH = Math.min(
    boundsH != null ? boundsH * pxPerScene : fallbackHeight,
    maxBodyH != null && maxBodyH > 0 ? maxBodyH : Infinity,
  );
  const boxW = Math.round(boxH * 0.7);
  // The fallback image is shown ONLY on hard failure (WebGL/rig load error), NOT
  // while the rig is loading. The fallback is sized to the layout box (height
  // 100%, ≈ boundsH*K), but the loaded rig renders at its MEASURED proportional
  // body height, and the caller's pre-load hint need not agree with that —
  // combatants pass the rig's own box (`Combatant`'s `spriteBodyH`), room
  // furniture passes a hand-placed `fallbackHeight`. So showing the
  // box-sized fallback during load and then swapping to the measured rig made
  // every combatant visibly jump on load: feet stay put (both bottom-anchored)
  // but the head snaps to the new height — small enemies appeared to "fall"
  // (box taller than rig), big ones to grow. Suppressing the load-time fallback
  // removes the before/after pair entirely: the slot stays empty (reserved at a
  // stable box height, so the board never reflows) and the rig simply fades in
  // at its correct size. The fallback is still the real failure path.
  const showFallback = failed && fallback;
  const canvasW = body ? body.w + 2 * body.sideM : boxW;
  const canvasH = body ? body.h + body.topM + body.botM : boxH;
  // Anchor the rig's ROOT — the point the game stands a creature on — at the
  // layout box's bottom-centre, so the box bottom IS the scene's ground line and
  // the box centre IS the creature's slot.
  //
  // Vertically: the canvas holds a `botM` overscan band below the idle body
  // (room for a downward slam so it isn't clipped) and the idle body itself
  // starts at world y = `world.y`, which is usually a little UNDER the root (a
  // rig's dust/contact art). So the canvas's bottom edge belongs
  // `(world.y - botM)` rig units below the root — negative, i.e. hanging below
  // the box, where a slam renders BEHIND the later-painted nameplate/HP
  // siblings instead of over them. Bottom-anchoring the canvas at 0 instead
  // pushed the resting feet a whole overscan band UP off the box bottom, so
  // tall action-heavy rigs (the Knowledge Demon: botM≈200px) floated far above
  // their own nameplate.
  //
  // ROOT-ANCHORING IS FOR RIGS THE SCENE PLACES — the combatants, which arrive
  // with a `worldScale` (their authored scene scale) and stand on a slot the
  // room's own layout computed. It is WRONG for a rig the caller fits into a
  // box it placed by hand (`fallbackHeight`, no `worldScale`): those rigs are
  // room furniture, and a room rig's root is wherever its scene happened to put
  // it, not its ground contact. `shop_merchant_top`'s art sits 2385..2938 rig
  // units RIGHT of its root and 1551 units BELOW it, so root-anchoring threw the
  // merchant a full frame off the tent and out of the shop — the screen rendered
  // with no merchant in it at all. `treasure_act_1` (541, -2401) is the same
  // shape of offset. For those callers the rig's BOUNDING BOX is the thing the
  // hand-placed box is about, so centre it and hang the overscan below, exactly
  // as this did before the scene port.
  const rootAnchored = worldScale != null;
  const canvasBottom = body && world ? (rootAnchored ? (world.y - world.botMw) * k : -body.botM) : 0;
  // Horizontal anchor: the rig's ROOT sits on the layout box's centre line, not
  // the rig's bounding box. The game places a creature by its node origin —
  // `NCombatRoom` derives the x from `Visuals.Bounds` but the rig hangs off the
  // node — and rigs are routinely asymmetric about it (the Ironclad's box runs
  // -529..+346 rig units; the Decimillipede's tail segment lies ENTIRELY to one
  // side of its root). Centring the box instead slid every creature off its
  // slot by half that asymmetry, and it is what stopped the three Decimillipede
  // segments — three pieces of one animal authored around a shared origin —
  // from tiling into one body.
  //
  // `flipX` mirrors about the root too: the canvas content is flipped in place
  // (scaleX(-1) about its own centre), so the canvas's own box moves to the
  // mirrored span.
  const canvasLeft = body && world && rootAnchored
    ? (flipX ? -(world.x + world.w + world.sideMw) : world.x - world.sideMw) * k
    : -canvasW / 2;
  // Where the rig's head sits inside the layout box, so box-top-anchored overlays
  // (intent, block, hotkey) can be pinned to the rig instead of floating above a
  // creature shorter than its box. The analytic body box first; once the ink scan
  // in render() has found the top row that actually carries paint, that instead —
  // the canvas top is `boxH - canvasBottom - canvasH`, because the canvas is
  // anchored at `bottom: canvasBottom`. Negative when the rig is taller than the
  // box (its head pokes above the box top); the parent allows that.
  const headTop = !body || failed || !world ? null
    : inkTop != null ? boxH - canvasBottom - canvasH + inkTop * canvasH
      : rootAnchored ? boxH - (world.y + world.h) * k : boxH - body.h;
  const footprint = !body || failed ? null
    : { canvasW, canvasH, bodyW: body.w, bodyH: body.h };
  // Reported from effects rather than inline, so a re-fit re-reports both without
  // the rig having reloaded. Callback identity is deliberately not a dependency:
  // parents pass inline closures, and re-reporting the same numbers every render
  // would loop through their setState.
  const measureRef = useRef(onMeasure); measureRef.current = onMeasure;
  const frameCbRef = useRef(onFrame); frameCbRef.current = onFrame;
  const footprintRef = useRef(footprint); footprintRef.current = footprint;
  useEffect(() => { measureRef.current?.(headTop); }, [headTop]);
  // Keyed by value, not identity: `footprint` is a fresh object every render.
  const footprintKey = footprint
    ? `${footprint.canvasW},${footprint.canvasH},${footprint.bodyW},${footprint.bodyH}`
    : "";
  useEffect(() => { frameCbRef.current?.(footprintRef.current); }, [footprintKey]);
  return (
    <div
      /* The rig failed to load, so the `<img>` fallback below is standing in
         for it — and an `<img>` cannot be dissolved by the WebGL post-pass.
         `game-skin.css` keys the dying-creature CSS fade back ON for this
         subtree; `Combatant` has no way to see this state from outside. */
      className={failed ? "gs-rig-failed" : undefined}
      style={{ position: "relative", height: Math.round(boxH), width: Math.round(boxW) }}>
      <canvas
        ref={canvasRef}
        width={Math.max(64, Math.round(canvasW * 1.3))} height={Math.max(64, Math.round(canvasH * 1.3))}
        style={{ position: "absolute", bottom: canvasBottom, left: `calc(50% + ${canvasLeft}px)`,
          transform: flipX ? "scaleX(-1)" : undefined,
          width: canvasW, height: canvasH,
          // Fade the rig in once it's measured (body set) instead of popping it
          // in — there is no load-time fallback underneath it anymore.
          opacity: body ? 1 : 0, transition: "opacity 140ms ease-out",
          display: failed ? "none" : "block", pointerEvents: "none", filter: spriteFilter }}
      />
      {showFallback && (
        <img src={fallback!} alt={alt}
          style={{ position: "absolute", bottom: 0, left: "50%",
            transform: `translateX(-50%)${flipX ? " scaleX(-1)" : ""}`,
            height: "100%", width: "auto", maxWidth: "100%", objectFit: "contain", filter: spriteFilter }} />
      )}
    </div>
  );
}
