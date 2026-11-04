import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import Board from "../Board";
import { spriteBodyH } from "../combatantMetrics";
import { instanceScale } from "../../scene";
import fixture from "./fixture-step.json";
import type { Step } from "../../api";

/**
 * `NCombatRoom.RandomizeEnemyScalesAndHues` sizes every member of a group of
 * two or more monsters of one type by where its rolled max HP landed in that
 * monster's initial-HP band, so three Exoskeletons at 26/24/25 HP are three
 * visibly different sizes in the game. The board drew them as identical clones.
 *
 * The multiplier is computed server-side (it needs the HP band, which no wire
 * carries) and arrives as `size_jitter`. What is pinned here is that the DRAWN
 * rig takes it and the ROW does not: the game scales the creature only after
 * `PositionEnemies` has spaced the row by the authored `Visuals.Bounds`, so a
 * bigger instance must not push its neighbours along.
 */
const enemy = (index: number, jitter?: number) => ({
  index, name: "Nibbit", hp: 30, max_hp: 30, block: 0,
  image: null, rig: "/assets/monster_rigs/nibbit/nibbit", rig_scale: 0.28,
  rig_bounds_h: 146, rig_bounds_w: 240,
  intents: [{ type: "Attack", damage: 6, hits: 1 }],
  ...(jitter === undefined ? {} : { size_jitter: jitter }),
});

function stepWith(enemies: unknown[]): Step {
  const base = fixture as unknown as Step;
  return { ...base, state: { ...base.state, enemies } as Step["state"] };
}

function slotXs(c: Element): Record<string, number> {
  const out: Record<string, number> = {};
  for (const el of c.querySelectorAll<HTMLElement>(".gs-side")) {
    const idx = el.querySelector("[data-enemy-index]")?.getAttribute("data-enemy-index");
    if (idx != null) out[idx] = parseFloat(el.style.left);
  }
  return out;
}

describe("per-instance creature scale", () => {
  it("defaults to 1 for a creature the rule did not fire on", () => {
    expect(instanceScale({})).toBe(1);
    expect(instanceScale({ size_jitter: null })).toBe(1);
    // A nonsense value must not collapse a rig to nothing.
    expect(instanceScale({ size_jitter: 0 })).toBe(1);
  });

  it("scales the reserved body with the rig", () => {
    const plain = spriteBodyH(enemy(0), true);
    const big = spriteBodyH(enemy(0, 1.1), true);
    expect(big).toBeCloseTo(plain * 1.1, 6);
  });

  it("does not move the row", () => {
    // The row is spaced by the authored Bounds width, which the jitter never
    // touches — otherwise a high-HP instance would shove its neighbours right.
    const plain = slotXs(render(<Board step={stepWith([enemy(0), enemy(1)])} />).container);
    const jittered = slotXs(
      render(<Board step={stepWith([enemy(0, 1.15), enemy(1, 1.1)])} />).container,
    );
    expect(jittered["0"]).toBeCloseTo(plain["0"], 6);
    expect(jittered["1"]).toBeCloseTo(plain["1"], 6);
  });
});
