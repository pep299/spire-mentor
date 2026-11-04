// The targeting arrow's curve and segment geometry — kept out of the component
// file so it exports only the component (react-refresh/only-export-components).

/* The card-targeting arrow, ported from the game's own
 * `MegaCrit.Sts2.Core.Nodes.Combat/NTargetingArrow.cs` (v0.107.1). It is not a
 * drawn path: the game lays 19 copies of one small chevron sprite along a
 * quadratic Bézier and caps them with a bigger head sprite. Both sprites ship
 * with the art tree, so this renders the real thing rather than approximating
 * it with strokes.
 *
 * Every pixel constant below is measured against the game's 1080-tall design
 * viewport and scaled by `boardHeight / DESIGN_HEIGHT`, so the arrow keeps its
 * proportions on a workbench board of any size. */

export interface Point { x: number; y: number }

/** Godot design viewport height. The decomp's `540f` half-screen test and its
 *  88/40 px offsets are all relative to this. */
export const DESIGN_HEIGHT = 1080;
export const SEGMENT_COUNT = 19;

/** Intrinsic sprite sizes, in px. */
const SEGMENT_PX = 148;
const HEAD_W = 183;
const HEAD_H = 203;

/** How far back along the shaft the head sprite's CENTRE sits, so its drawn
 *  point lands on the thing you are aiming at rather than overshooting it. */
const HEAD_SETBACK = 88;
/** Where the Bézier stops — inside the head sprite, which covers the join. */
const CURVE_END = 40;

const SCALE_START = 0.28;
const SCALE_END = 0.42;
const HEAD_SCALE = 0.95;
const HEAD_SCALE_HOVER = 1.05;
/** Applied as a CSS scale on top of the base size, so only the hover pop
 *  transitions and the arrow itself still tracks the cursor without lag. */
export const HEAD_HOVER_RATIO = HEAD_SCALE_HOVER / HEAD_SCALE;

/** `StsColors.targetingArrowEnemy` / `targetingArrowAlly`. White is the
 *  untinted default — the sprites are greyscale and the game tints them with
 *  `Modulate`, which multiplies. */
export const TINTS = {
  none: "#ffffff",
  enemy: "#e61e1b",
  ally: "#36c78a",
} as const;
export type ArrowTint = keyof typeof TINTS;

export interface ArrowSegment extends Point {
  /** The sprite's Godot scale factor, kept for tests and debugging. */
  scale: number;
  /** Drawn edge length in board px. */
  size: number;
  /** Radians. 0 draws the sprite pointing up (-Y), as the art is authored. */
  rotation: number;
}

export interface ArrowGeometry {
  segments: ArrowSegment[];
  head: Point & { rotation: number };
  headSize: { w: number; h: number };
}

export function rotate(x: number, y: number, r: number): Point {
  const c = Math.cos(r), s = Math.sin(r);
  return { x: x * c - y * s, y: x * s + y * c };
}

/** `MathHelper.BezierCurve(v0, v1, c0, t)` — quadratic, control point LAST. */
export function bezier(a: Point, b: Point, c: Point, t: number): Point {
  const u = 1 - t;
  return {
    x: u * u * a.x + 2 * u * t * c.x + t * t * b.x,
    y: u * u * a.y + 2 * u * t * c.y + t * t * b.y,
  };
}

/** The Bézier's control point. Pulling X back *past* the origin is what gives
 *  the shaft its overhand throw; the Y branch is the decomp's `From.Y > 540f`
 *  test — a card in the bottom half of the screen (the usual case, the hand
 *  lives there) swings the curve out below itself, one higher up hangs it. */
function controlPoint(from: Point, head: Point, boardHeight: number): Point {
  return {
    x: from.x - (head.x - from.x) * 0.25,
    y: from.y > boardHeight / 2
      ? head.y + (head.y - from.y) * 0.5
      : head.y * 0.75 + from.y * 0.25,
  };
}

/**
 * Place the head and the 19 shaft segments for one arrow.
 *
 * `from` is where the arrow hangs off the card, `tip` is what it points at —
 * both in board-relative px, as is the returned geometry.
 *
 * The game's head placement is circular: `UpdateArrowPosition` positions the
 * head using its rotation, then derives a new rotation from where it landed.
 * Godot gets away with it because `_Process` runs every frame and the loop is
 * one frame behind. A single React render has no previous frame, so settle the
 * loop here instead — it converges well inside four passes.
 */
export function arrowGeometry(from: Point, tip: Point, boardHeight: number): ArrowGeometry {
  const k = boardHeight / DESIGN_HEIGHT;

  let rotation = Math.atan2(tip.y - from.y, tip.x - from.x) + Math.PI / 2;
  const place = (back: number, r: number): Point => {
    const off = rotate(0, back * k, r);
    return { x: tip.x + off.x, y: tip.y + off.y };
  };
  for (let pass = 0; pass < 4; pass++) {
    const c = controlPoint(from, place(HEAD_SETBACK, rotation), boardHeight);
    rotation = Math.atan2(tip.y - c.y, tip.x - c.x) + Math.PI / 2;
  }
  const head = place(HEAD_SETBACK, rotation);
  const end = place(CURVE_END, rotation);
  const control = controlPoint(from, head, boardHeight);

  const segments: ArrowSegment[] = [];
  for (let i = 0; i < SEGMENT_COUNT; i++) {
    // `(float)i / 20f` — the shaft covers only the first 90% of the curve.
    const p = bezier(from, end, control, i / (SEGMENT_COUNT + 1));
    // `Mathf.Lerp` does NOT clamp, and this weight passes 1 at i≈9.5, so the
    // game extrapolates past SCALE_END on purpose: the blocks nearest the head
    // are the fattest. Clamping here flattens the taper the arrow reads by.
    const scale = SCALE_START + (SCALE_END - SCALE_START) * ((i * 2) / SEGMENT_COUNT);
    segments.push({ ...p, scale, size: SEGMENT_PX * scale * k, rotation: 0 });
  }
  for (let i = 1; i < SEGMENT_COUNT; i++) {
    const d = { x: segments[i].x - segments[i - 1].x, y: segments[i].y - segments[i - 1].y };
    segments[i].rotation = Math.atan2(d.y, d.x) + Math.PI / 2;
  }
  // Segment 0 has no predecessor to difference against, so it borrows the
  // 0->1 direction. The decomp also assigns it a flat 0/π inside the loop, but
  // overwrites that immediately afterwards — that branch is dead, and porting
  // it would leave the tail block sitting upright.
  const d0 = { x: segments[0].x - segments[1].x, y: segments[0].y - segments[1].y };
  segments[0].rotation = Math.atan2(d0.y, d0.x) - Math.PI / 2;

  return {
    segments,
    head: { ...head, rotation },
    headSize: { w: HEAD_W * HEAD_SCALE * k, h: HEAD_H * HEAD_SCALE * k },
  };
}
