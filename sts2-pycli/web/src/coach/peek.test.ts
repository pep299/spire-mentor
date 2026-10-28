import { describe, expect, it } from "vitest";
import type { Score } from "../api";
import { peekLabelFromAction, peekLabelFromScores } from "./peek";

const row = (o: Partial<Score>): Score => ({
  stable_idx: 0, action: "Strike", win_prob: 0.5, q: 0, visits: 100, is_best: false,
  q_saturated: false, low_confidence: false, target_index: null, ...o,
});

describe("peekLabelFromScores", () => {
  it("names only the best row", () => {
    expect(peekLabelFromScores([row({}), row({ stable_idx: 2, action: "Bash", is_best: true })])).toBe("Bash");
  });
  it("names the target when there is more than one enemy", () => {
    const scores = [row({ action: "Bash", is_best: true, target_index: 1 })];
    const enemies = [{ index: 0, name: "Cultist" }, { index: 1, name: "Jaw Worm" }];
    expect(peekLabelFromScores(scores, enemies)).toBe("Bash (Jaw Worm)");
    expect(peekLabelFromScores(scores, [enemies[1]])).toBe("Bash");
  });
  it("numbers a target whose name is shared", () => {
    const scores = [row({ action: "Bash", is_best: true, target_index: 3 })];
    const rats = [
      { index: 0, name: "Two-Tailed Rat" }, { index: 1, name: "Louse" }, { index: 3, name: "Two-Tailed Rat" },
    ];
    expect(peekLabelFromScores(scores, rats)).toBe("Bash (Two-Tailed Rat #2)");
  });
  it("reads the end-turn row as end turn", () => {
    expect(peekLabelFromScores([row({ stable_idx: -1, action: "End Turn", is_best: true })])).toBe("end turn");
  });
  it("is null without a best row", () => {
    expect(peekLabelFromScores([row({})])).toBeNull();
  });
});

describe("peekLabelFromAction", () => {
  const options = [
    { action: "choose_card", args: { card_index: 0 }, label: "Anger" },
    { action: "choose_card", args: { card_index: 1 }, label: "Inflame" },
    { action: "skip", args: {}, label: "Skip" },
  ];
  it("finds the policy's option by action and args", () => {
    expect(peekLabelFromAction({ action: "choose_card", args: { card_index: 1 } }, options)).toBe("Inflame");
    expect(peekLabelFromAction({ action: "skip", args: {} }, options)).toBe("Skip");
  });
  it("is null when nothing matches or there is no pick", () => {
    expect(peekLabelFromAction({ action: "choose_card", args: { card_index: 5 } }, options)).toBeNull();
    expect(peekLabelFromAction(null, options)).toBeNull();
  });
});
