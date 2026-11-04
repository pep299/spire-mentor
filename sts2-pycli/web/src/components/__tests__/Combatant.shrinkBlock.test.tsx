import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import Combatant from "../Combatant";
import { bodyScale } from "../combatantMetrics";

/**
 * Two pieces of creature chrome the game draws and we didn't:
 *
 * - Shrink resizes the creature (`ShrinkPower.AfterApplied` →
 *   `NCreature.ScaleTo(0.5f, 0.75)`), the player included — only the sprite
 *   (`Visuals`), not the health bar.
 * - Block lives ON the health bar (`NHealthBar.RefreshBlockUi`): the bar turns
 *   blue and the shield sits on its left end, not up by the creature's head.
 */
const who = (over: Record<string, unknown> = {}) => ({
  name: "Ironclad", hp: 70, max_hp: 80, block: 0, ...over,
});

describe("bodyScale", () => {
  it("halves the body under Shrink, by engine id or by title", () => {
    expect(bodyScale([{ name: "Shrink", id: "SHRINK", amount: 1 }])).toBe(0.5);
    expect(bodyScale([{ name: "Shrink", amount: -1 }])).toBe(0.5);
  });

  it("is 1 for no powers or other powers", () => {
    expect(bodyScale(undefined)).toBe(1);
    expect(bodyScale([{ name: "Vulnerable", id: "VULNERABLE", amount: 2 }])).toBe(1);
  });
});

describe("Shrink on the sprite", () => {
  it("scales the player's body, and leaves the health bar outside it", () => {
    const { container } = render(<Combatant isEnemy={false}
      c={who({ image: "x.png", powers: [{ name: "Shrink", id: "SHRINK", amount: 1 }] })} />);
    const body = container.querySelector(".gs-sprite-body") as HTMLElement;
    expect(body.style.scale).toBe("0.5");
    expect(body.querySelector(".gs-sprite")).not.toBeNull();
    expect(body.querySelector(".gs-hpwrap")).toBeNull();
  });

  it("sets no scale at all when unshrunk", () => {
    const { container } = render(<Combatant isEnemy={false} c={who({ image: "x.png" })} />);
    expect((container.querySelector(".gs-sprite-body") as HTMLElement).style.scale).toBe("");
  });
});

describe("block on the health bar", () => {
  it("attaches the shield to the bar and turns the bar blue while blocked", () => {
    const { container } = render(<Combatant isEnemy={false} c={who({ block: 9 })} />);
    const wrap = container.querySelector(".gs-hpwrap") as HTMLElement;
    expect(wrap.className).toContain("gs-hp-blocked");
    const badge = wrap.querySelector("[data-block-badge]") as HTMLElement;
    expect(badge.textContent).toBe("9");
    expect(badge.querySelector("img")?.getAttribute("src")).toMatch(/ui\/combat\/block\.png$/);
  });

  it("drops the blue state when block is gone", () => {
    const { container } = render(<Combatant isEnemy={false} c={who({ block: 0 })} />);
    expect((container.querySelector(".gs-hpwrap") as HTMLElement).className).not.toContain("gs-hp-blocked");
  });

  it("drops the shield in only on a gain from zero (AnimateInBlock's oldBlock == 0)", () => {
    const fx = { blockPop: true, nonce: 1 };
    const { container, rerender } = render(<Combatant isEnemy={false} c={who({ block: 0 })} />);
    rerender(<Combatant isEnemy={false} c={who({ block: 5 })} effects={fx} />);
    expect(container.querySelector(".gs-block")!.className).toContain("gs-block-pop");
    rerender(<Combatant isEnemy={false} c={who({ block: 8 })} effects={{ ...fx, nonce: 2 }} />);
    expect(container.querySelector(".gs-block")!.className).not.toContain("gs-block-pop");
  });
});
