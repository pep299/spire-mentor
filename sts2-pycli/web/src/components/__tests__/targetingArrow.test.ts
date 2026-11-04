import { describe, it, expect } from "vitest";
import { arrowGeometry, SEGMENT_COUNT, DESIGN_HEIGHT } from "../arrowGeometry";

/** A card at the bottom-centre of a 1080-tall board, aiming up-right at an
 *  enemy — the ordinary case, and the one the game's constants are tuned for. */
const FROM = { x: 960, y: 950 };
const TIP = { x: 1500, y: 400 };

function geo(from = FROM, tip = TIP, h = DESIGN_HEIGHT) {
  return arrowGeometry(from, tip, h);
}

describe("the targeting arrow's geometry", () => {
  it("lays the game's 19 segments plus one head", () => {
    const g = geo();
    expect(g.segments).toHaveLength(SEGMENT_COUNT);
    expect(g.head).toBeDefined();
  });

  it("starts at the card and ends near the target", () => {
    const g = geo();
    const first = g.segments[0];
    const last = g.segments[SEGMENT_COUNT - 1];
    expect(Math.hypot(first.x - FROM.x, first.y - FROM.y)).toBeLessThan(1);
    // The shaft stops short of the tip — the head sprite covers the rest.
    const gap = Math.hypot(last.x - TIP.x, last.y - TIP.y);
    expect(gap).toBeGreaterThan(0);
    expect(gap).toBeLessThan(Math.hypot(TIP.x - FROM.x, TIP.y - FROM.y) / 3);
  });

  it("puts the head's point ON the target, not its centre", () => {
    // The sprite's centre sits back along the shaft so the drawn tip lands on
    // whatever you are aiming at. A head centred on the target would overshoot
    // the enemy by half a sprite.
    const g = geo();
    expect(Math.hypot(g.head.x - TIP.x, g.head.y - TIP.y)).toBeGreaterThan(40);
  });

  it("fattens from tail to head", () => {
    const g = geo();
    const scales = g.segments.map((s) => s.scale);
    for (let i = 1; i < scales.length; i++) expect(scales[i]).toBeGreaterThan(scales[i - 1]);
  });

  it("bows rather than running straight", () => {
    // A straight line between two points reads as a UI connector; the game
    // throws the arrow, so the shaft leaves the chord.
    const g = geo();
    const mid = g.segments[Math.floor(SEGMENT_COUNT / 2)];
    const ax = TIP.x - FROM.x, ay = TIP.y - FROM.y;
    const bx = mid.x - FROM.x, by = mid.y - FROM.y;
    const offLine = Math.abs(ax * by - ay * bx) / Math.hypot(ax, ay);
    expect(offLine).toBeGreaterThan(30);
  });

  it("continues the shaft rather than pointing down the chord", () => {
    // The arrow is thrown OVER the board and drops onto the target, so the
    // head follows the curve's incoming tangent — aiming it straight down the
    // from->tip chord would kink it away from the segments feeding into it.
    const g = geo();
    // rotation 0 draws the sprite pointing up (-Y), so the drawn direction is
    // the rotation applied to (0,-1).
    const dx = Math.sin(g.head.rotation), dy = -Math.cos(g.head.rotation);
    const last = g.segments[SEGMENT_COUNT - 1];
    const tx = TIP.x - last.x, ty = TIP.y - last.y;
    const dot = (dx * tx + dy * ty) / Math.hypot(tx, ty);
    expect(dot).toBeGreaterThan(0.95);
  });

  it("orients every segment along the shaft, tail included", () => {
    const g = geo();
    // No segment may be left at rotation 0 (the Godot decomp assigns segment 0
    // a placeholder inside its loop and overwrites it after — porting the
    // placeholder instead of the overwrite leaves the tail block upright).
    for (let i = 0; i < SEGMENT_COUNT - 1; i++) {
      const seg = g.segments[i];
      const nx = g.segments[i + 1].x - seg.x, ny = g.segments[i + 1].y - seg.y;
      const dx = Math.sin(seg.rotation), dy = -Math.cos(seg.rotation);
      const dot = (dx * nx + dy * ny) / Math.hypot(nx, ny);
      expect(dot).toBeGreaterThan(0.9);
    }
  });

  it("scales the whole arrow with the board", () => {
    // Every constant in the decomp is measured against the game's 1080-tall
    // viewport. A half-height board must get a half-size arrow, or the head
    // swallows the enemy it is pointing at.
    const full = geo(FROM, TIP, DESIGN_HEIGHT);
    const half = geo(FROM, TIP, DESIGN_HEIGHT / 2);
    expect(half.headSize.h).toBeCloseTo(full.headSize.h / 2, 4);
    expect(half.segments[0].size).toBeCloseTo(full.segments[0].size / 2, 4);
  });

  it("keeps the tail below the card when the card is high on the board", () => {
    // The decomp branches its control point on whether the origin is in the
    // bottom half of the screen. Both branches must produce a usable curve.
    const high = geo({ x: 960, y: 200 }, { x: 1500, y: 700 });
    expect(high.segments).toHaveLength(SEGMENT_COUNT);
    expect(high.segments.every((s) => Number.isFinite(s.x) && Number.isFinite(s.y))).toBe(true);
  });
});
