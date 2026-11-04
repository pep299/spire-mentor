import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import Combatant from "../Combatant";
import { doomAmount } from "../combatantMetrics";

/**
 * Doom on the health bar.
 *
 * Doom kills its owner at the end of their side's turn once
 * `CurrentHp <= Amount` (decomp `DoomPower.IsOwnerDoomed`), and the game paints
 * that kill line ONTO the bar — `NHealthBar`'s `%DoomForeground` runs from the
 * bar's low end out to `doom/maxHp`, and once Doom reaches current HP the red
 * fill is switched off entirely and the numerals recolour. We had none of it:
 * Doom was a status pip like any other, so the one thing it tells you — how
 * close the kill line is — had to be worked out in your head.
 */
const enemy = (hp: number, powers?: { name: string; id?: string; amount: number }[]) => ({
  name: "Test Subject", hp, max_hp: 100, block: 0, powers,
});

const bar = (c: HTMLElement) => ({
  doom: c.querySelector(".gs-hpdoom") as HTMLElement | null,
  fill: c.querySelector(".gs-hpfill") as HTMLElement | null,
  text: c.querySelector(".gs-hptext") as HTMLElement,
});

describe("doomAmount", () => {
  it("matches on the engine id, and on the title when that is all there is", () => {
    expect(doomAmount([{ name: "Doom", id: "DOOM", amount: 12 }])).toBe(12);
    expect(doomAmount([{ name: "Doom", amount: 7 }])).toBe(7);
  });

  it("is 0 for no powers, other powers, or a spent counter", () => {
    expect(doomAmount(undefined)).toBe(0);
    expect(doomAmount([])).toBe(0);
    expect(doomAmount([{ name: "Strength", id: "STRENGTH", amount: 3 }])).toBe(0);
    expect(doomAmount([{ name: "Doom", id: "DOOM", amount: 0 }])).toBe(0);
  });
});

describe("the Doom band on the health bar", () => {
  it("is absent when nothing is doomed", () => {
    const { container } = render(<Combatant c={enemy(80)} isEnemy />);
    expect(bar(container).doom).toBeNull();
    expect(bar(container).fill).not.toBeNull();
  });

  it("paints the low end of the bar out to doom/maxHp, under the live fill", () => {
    const { container } = render(
      <Combatant c={enemy(80, [{ name: "Doom", id: "DOOM", amount: 25 }])} isEnemy />,
    );
    const { doom, fill, text } = bar(container);
    expect(doom!.style.width).toBe("25%");
    expect(fill!.style.width).toBe("80%");
    expect(doom!.className).not.toContain("gs-hpdoom-lethal");
    expect(text.className).not.toContain("gs-hptext-doomed");
  });

  it("never overhangs the live fill", () => {
    // 20 HP left, Doom 25 — which is already lethal, but the clamp is what
    // stops a band from drawing over empty track in any near case.
    const { container } = render(
      <Combatant c={enemy(20, [{ name: "Doom", id: "DOOM", amount: 25 }])} isEnemy />,
    );
    expect(bar(container).doom!.style.width).toBe("20%");
  });

  it("takes the whole bar and recolours the numerals once Doom is lethal", () => {
    const { container } = render(
      <Combatant c={enemy(20, [{ name: "Doom", id: "DOOM", amount: 20 }])} isEnemy />,
    );
    const { doom, fill, text } = bar(container);
    // The game hides the red foreground outright in this state.
    expect(fill).toBeNull();
    expect(doom!.className).toContain("gs-hpdoom-lethal");
    expect(text.className).toContain("gs-hptext-doomed");
  });

  it("shows a floor-width sliver for a tiny Doom on a big pool", () => {
    const { container } = render(
      <Combatant c={{ ...enemy(300), max_hp: 300, hp: 300,
        powers: [{ name: "Doom", id: "DOOM", amount: 1 }] }} isEnemy />,
    );
    // 1/300 is a third of a percent — invisible. The game floors the bar at
    // 12px for the same reason (NHealthBar.GetFgWidth).
    expect(bar(container).doom!.style.width).toBe("4%");
  });

  it("draws nothing on a dead combatant", () => {
    const { container } = render(
      <Combatant c={enemy(0, [{ name: "Doom", id: "DOOM", amount: 20 }])} isEnemy />,
    );
    expect(bar(container).doom).toBeNull();
  });
});
