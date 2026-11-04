import { describe, it, expect } from "vitest";
import { MathUtils } from "@esotericsoftware/spine-webgl";
import {
  anchorRect,
  badgeRect,
  drawAngleFromGodotDegrees,
  drawAngleFromGodotRadians,
  sampleSpriteShaderTween,
} from "../vfxLive";
import type { TargetRef } from "../../effects/types";
import type { SpriteShaderEffect } from "../../effects/manifest";

function arenaWith(html: string): HTMLElement {
  const arena = document.createElement("div");
  arena.innerHTML = html;
  document.body.appendChild(arena);
  return arena;
}

describe("anchorRect", () => {
  it("finds the player by its combatant anchor", () => {
    const arena = arenaWith(`<div data-combatant="player"></div>`);
    expect(anchorRect({ side: "player" }, arena)).not.toBeNull();
  });

  it("finds an enemy by its encounter-slot index", () => {
    const arena = arenaWith(`<div data-enemy-index="2"></div>`);
    const t: TargetRef = { side: "enemy", index: 2 };
    expect(anchorRect(t, arena)).not.toBeNull();
  });

  it("returns null for a slot that is not on the board", () => {
    const arena = arenaWith(`<div data-enemy-index="0"></div>`);
    expect(anchorRect({ side: "enemy", index: 5 }, arena)).toBeNull();
  });

  it("returns null for osty, which sub-project 1 does not anchor", () => {
    const arena = arenaWith(`<div data-combatant="player"></div>`);
    expect(anchorRect({ side: "osty" }, arena)).toBeNull();
  });
});

describe("badgeRect", () => {
  it("finds the block badge nested under the player's creature anchor", () => {
    const arena = arenaWith(
      `<div data-combatant="player"><div data-block-badge="true">7</div></div>`,
    );
    expect(badgeRect({ side: "player" }, arena)).not.toBeNull();
  });

  it("finds the block badge nested under an enemy's creature anchor", () => {
    const arena = arenaWith(
      `<div data-enemy-index="2"><div data-block-badge="true">3</div></div>`,
    );
    expect(badgeRect({ side: "enemy", index: 2 }, arena)).not.toBeNull();
  });

  it("returns null when the creature itself is not on the board", () => {
    const arena = arenaWith(
      `<div data-enemy-index="0"><div data-block-badge="true">3</div></div>`,
    );
    expect(badgeRect({ side: "enemy", index: 5 }, arena)).toBeNull();
  });

  it("returns null (not a fallback to the creature's own rect) when the badge div isn't in the DOM at all", () => {
    // `Combatant.tsx` keeps `[data-block-badge]` ALWAYS mounted once the rig
    // is up (see its doc) — this covers the one case it is genuinely absent:
    // a rig still loading, before `Combatant` has rendered any chrome at all.
    const arena = arenaWith(`<div data-enemy-index="0"></div>`);
    expect(badgeRect({ side: "enemy", index: 0 }, arena)).toBeNull();
  });

  it("does not match a badge belonging to a DIFFERENT creature", () => {
    const arena = arenaWith(
      `<div data-enemy-index="0"><div data-block-badge="true">1</div></div>` +
        `<div data-enemy-index="1"></div>`,
    );
    expect(badgeRect({ side: "enemy", index: 1 }, arena)).toBeNull();
  });
});

describe("drawAngleFromGodotDegrees", () => {
  // Not a restatement of the conversion — this replays `spine-webgl`'s OWN
  // quad rotation (`SceneRenderer.js:246-249`) through its OWN trig helpers, so
  // what these assertions pin is where a corner actually lands. Both halves of
  // the bug were invisible to any test that only compared numbers: the unit
  // error hid inside `MathUtils.cosDeg`, and the sign error is only a sign.
  function rotate(px: number, py: number, angle: number): { x: number; y: number } {
    const cos = MathUtils.cosDeg(angle);
    const sin = MathUtils.sinDeg(angle);
    return { x: cos * px - sin * py, y: sin * px + cos * py };
  }

  // The flipbooks' authored `_rotationRange` upper bound, and the sprite's top
  // edge in this Y-up canvas at a round half-height.
  const AUTHORED_DEG = 25;
  const TOP_EDGE = { x: 0, y: 100 };

  it("turns the sprite by the authored magnitude", () => {
    const p = rotate(TOP_EDGE.x, TOP_EDGE.y, drawAngleFromGodotDegrees(AUTHORED_DEG));
    expect(Math.hypot(p.x, p.y)).toBeCloseTo(100);
    expect(Math.abs(p.x)).toBeCloseTo(100 * Math.sin((AUTHORED_DEG * Math.PI) / 180));
  });

  it("spins a positive Godot rotation clockwise on screen", () => {
    // Godot 2D rotates clockwise as seen by the player, which carries the top
    // edge to the RIGHT. This canvas is Y-up, so that is +x.
    expect(rotate(TOP_EDGE.x, TOP_EDGE.y, drawAngleFromGodotDegrees(AUTHORED_DEG)).x).toBeGreaterThan(0);
  });

  it("is what the pre-fix radian conversion was not, on both counts", () => {
    // `(deg * Math.PI) / 180` handed to a parameter `cosDeg` converts AGAIN.
    const inert = rotate(TOP_EDGE.x, TOP_EDGE.y, (AUTHORED_DEG * Math.PI) / 180);
    expect(Math.abs(inert.x)).toBeLessThan(1); // 42px of throw collapses to 0.76px
    expect(inert.x).toBeLessThan(0); // and what little there is goes the wrong way
  });

  it("agrees with itself across the two authored units", () => {
    expect(drawAngleFromGodotRadians(Math.PI / 4)).toBeCloseTo(drawAngleFromGodotDegrees(45));
  });
});

describe("sampleSpriteShaderTween", () => {
  // The committed `vfx_power_flash_vfx` manifest entry (Task 10):
  // scale 0.4 -> 0.4 (0.4s, linear) -> 0.45 (0.4s, linear); modulateAlpha
  // 1.0 -> 1.0 (0.25s, EASE_OUT/SINE, parallel with the first scale step)
  // -> 0.0 (0.25s, EASE_IN/SINE), lifetime 1.05s.
  const EFFECT: SpriteShaderEffect = {
    kind: "sprite_shader",
    shader: "power_flash_vfx",
    blendMode: "blend_add",
    strength: 1.0,
    scale: 0.4,
    modulateAlpha: 1.0,
    tween: [
      [
        { property: "scale", to: 0.4, duration: 0.4, ease: null, trans: null },
        { property: "modulateAlpha", to: 1.0, duration: 0.25, ease: 1, trans: 1 },
      ],
      [{ property: "scale", to: 0.45, duration: 0.4, ease: null, trans: null }],
      [{ property: "modulateAlpha", to: 0.0, duration: 0.25, ease: 0, trans: 1 }],
    ],
    lifetime: 1.05,
  };

  it("starts at the manifest's authored scale/alpha", () => {
    expect(sampleSpriteShaderTween(EFFECT, 0)).toEqual({ scale: 0.4, modulateAlpha: 1.0 });
  });

  it("holds scale flat through step 0 — its target equals its start", () => {
    expect(sampleSpriteShaderTween(EFFECT, 0.4).scale).toBeCloseTo(0.4);
  });

  it("grows scale to 0.45 by the end of step 1, alpha still at its peak", () => {
    const { scale, modulateAlpha } = sampleSpriteShaderTween(EFFECT, 0.8);
    expect(scale).toBeCloseTo(0.45);
    expect(modulateAlpha).toBeCloseTo(1.0);
  });

  it("eases alpha out through step 2 on TRANS_SINE/EASE_IN, not linearly", () => {
    // Halfway through the final 0.25s fade (elapsed 0.8 + 0.125 = 0.925):
    // EASE_IN sine is `1 - cos(t * PI/2)`, so the eased fraction at t=0.5 is
    // `1 - cos(PI/4)` ~= 0.29289, NOT the linear 0.5 a flat lerp would give.
    const { modulateAlpha, scale } = sampleSpriteShaderTween(EFFECT, 0.925);
    expect(modulateAlpha).toBeCloseTo(1 - (1 - Math.cos(Math.PI / 4)), 5);
    expect(scale).toBeCloseTo(0.45); // holds — step 1 already finished
  });

  it("ends fully faded out at the authored lifetime", () => {
    const { scale, modulateAlpha } = sampleSpriteShaderTween(EFFECT, 1.05);
    expect(scale).toBeCloseTo(0.45);
    expect(modulateAlpha).toBeCloseTo(0, 5);
  });

  it("clamps rather than overshooting past the authored lifetime", () => {
    const { scale, modulateAlpha } = sampleSpriteShaderTween(EFFECT, 9);
    expect(scale).toBeCloseTo(0.45);
    expect(modulateAlpha).toBeCloseTo(0, 5);
  });
});
