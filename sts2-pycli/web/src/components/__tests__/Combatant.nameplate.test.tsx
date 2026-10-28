import { render } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import Combatant, { nameplateText } from "../Combatant";
import { enemyLabeller } from "../../enemyLabels";

vi.mock("../SpineSprite", () => ({
  default: () => <div data-testid="spine-mock" />,
}));

/**
 * Three Two-Tailed Rats must say WHICH one: the nameplate shows the board
 * label's " #k" (the same numbering as "Strike → Two-Tailed Rat #2"), still
 * without the debug size suffix; the tooltip / accessible name carry the full
 * label.
 */
describe("nameplateText", () => {
  it("is the plain name with no label or a unique one", () => {
    expect(nameplateText("Louse")).toEqual({ text: "Louse", full: undefined });
    expect(nameplateText("Louse", "Louse")).toEqual({ text: "Louse", full: undefined });
  });

  it("adds the duplicate number", () => {
    expect(nameplateText("Two-Tailed Rat", "Two-Tailed Rat #2")).toEqual({
      text: "Two-Tailed Rat #2", full: undefined,
    });
  });

  it("strips the size suffix but keeps the number, full label on the tooltip", () => {
    expect(nameplateText("Twig Slime (M)")).toEqual({ text: "Twig Slime", full: "Twig Slime (M)" });
    expect(nameplateText("Twig Slime (M)", "Twig Slime (M) #3")).toEqual({
      text: "Twig Slime #3", full: "Twig Slime (M) #3",
    });
  });
});

describe("the enemy nameplate", () => {
  it("numbers same-named enemies by the board labeller", () => {
    const enemies = [
      { index: 0, name: "Two-Tailed Rat" },
      { index: 1, name: "Louse" },
      { index: 3, name: "Two-Tailed Rat" },
    ];
    const labelOf = enemyLabeller(enemies);
    const plate = (index: number, name: string) => {
      const c = { name, hp: 10, max_hp: 10, block: 0 };
      const { container } = render(<Combatant c={c} isEnemy label={labelOf(index)} />);
      return container.querySelector(".gs-nameplate")!.textContent;
    };
    expect(plate(0, "Two-Tailed Rat")).toBe("Two-Tailed Rat #1");
    expect(plate(1, "Louse")).toBe("Louse");
    expect(plate(3, "Two-Tailed Rat")).toBe("Two-Tailed Rat #2");
  });
});
