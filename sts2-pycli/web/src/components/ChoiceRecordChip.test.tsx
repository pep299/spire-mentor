import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { CoachChoiceRecord } from "../api";
import { choiceChipText } from "../coach/coachText";
import ChoiceRecordChip from "./ChoiceRecordChip";

const rec = (o: Partial<CoachChoiceRecord> = {}): CoachChoiceRecord => ({
  kind: "choice", id: "choice:4", node_id: 4, decision: "card_reward", label_kind: "Card reward",
  verdict: "flag", strength: "strong preference", played: { label: "Anger", prob: 0.05, rank: 3 },
  top: { label: "Inflame", prob: 0.7 }, alternatives: [], peeked: false, ...o,
});

describe("choiceChipText", () => {
  it("reads like the mod's chip, with the rank's ordinal", () => {
    expect(choiceChipText(rec())).toEqual({
      title: "Card reward · you: Anger",
      lines: ["AI would take: Inflame (strong preference) · Anger was its 3rd choice"],
    });
    const eleventh = rec({ played: { label: "Anger", prob: 0.01, rank: 11 } });
    expect(choiceChipText(eleventh)?.lines[0]).toMatch(/11th choice$/);
  });
  it("adds the route line when both HPs are known", () => {
    const map = rec({
      label_kind: "Map", played: { label: "Elite", prob: 0.1, rank: 2 },
      path: { you: [{ hp: 50 }, { hp: 31.6 }], ai: [{ hp: 52 }, { hp: 44.2 }] },
    });
    expect(choiceChipText(map)?.lines[1]).toBe("route HP: you ≈32 · AI ≈44");
    expect(choiceChipText({ ...map, path: { you: [{ hp: 50 }], ai: [{}] } })?.lines).toHaveLength(1);
  });
  it("is silent on agreement and toss-ups", () => {
    expect(choiceChipText(rec({ verdict: "agree" }))).toBeNull();
    expect(choiceChipText(rec({ verdict: "tossup" }))).toBeNull();
  });
});

describe("ChoiceRecordChip", () => {
  it("renders the flagged choice", () => {
    render(<ChoiceRecordChip record={rec()} />);
    expect(screen.getByText("Card reward · you: Anger")).toBeTruthy();
  });
});
