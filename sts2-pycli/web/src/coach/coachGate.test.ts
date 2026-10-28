import { describe, expect, it } from "vitest";
import type { ChoiceGrade } from "../api";
import { choiceRecordOf, isFresh } from "./coachGate";

describe("isFresh", () => {
  it("drops a grade once the player acted after asking", () => {
    let gen = 0;
    const sent = ++gen; // End turn
    expect(isFresh(sent, gen)).toBe(true);
    ++gen; // a card played in the next turn before the grade returned
    expect(isFresh(sent, gen)).toBe(false);
  });
});

describe("choiceRecordOf", () => {
  it("maps a live choice grade to the logged record shape", () => {
    const g: ChoiceGrade = {
      node_id: 12, parent_id: 11, decision: "card_reward", kind: "Card reward", verdict: "flag",
      strength: "strong preference", played: { label: "Anger", prob: 0.1, rank: 3 },
      top: { label: "Inflame", prob: 0.7 }, alternatives: [], n_options: 3, checkable: true,
    };
    const r = choiceRecordOf(g, true);
    expect(r).toMatchObject({ kind: "choice", id: "choice:12", label_kind: "Card reward", verdict: "flag", peeked: true });
  });
});
