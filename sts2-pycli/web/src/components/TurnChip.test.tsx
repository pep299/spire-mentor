import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { CoachTurnRecord } from "../api";
import { turnChipText } from "../coach/coachText";
import TurnChip from "./TurnChip";

const rec = (o: Partial<CoachTurnRecord> = {}): CoachTurnRecord => ({
  kind: "turn", id: "turn:7", node_id: 7, turn: 3, floor: null, verdict: "blunder",
  you: ["Strike (Cultist)", "end turn"], ai: ["Bash (Cultist)", "Defend", "end turn"],
  hp_drop: 9.2, win_before: 0.96, win_after: 0.94, peeked: false, ungraded_actions: 0,
  ...o,
});

describe("turnChipText", () => {
  it("reads a blunder with HP the way the mod's chip does", () => {
    expect(turnChipText(rec())).toEqual({
      title: "⚠ blunder",
      lines: [
        "Turn 3 · you: Strike (Cultist) → end turn",
        "AI: Bash (Cultist) → Defend → end turn",
        "≈ 9 HP more lost · win 96% → 94%",
      ],
    });
  });
  it("leaves the HP out of a slip without HP samples", () => {
    const t = turnChipText(rec({ verdict: "slip", hp_drop: null }));
    expect(t?.title).toBe("▽ suboptimal");
    expect(t?.lines[2]).toBe("win 96% → 94%");
  });
  it("ends with the reason line when there is one (GH #888)", () => {
    const why = "Your line wakes Lagavulin Matriarch on turn 3";
    expect(turnChipText(rec({ why }))?.lines).toEqual([
      "Turn 3 · you: Strike (Cultist) → end turn",
      "AI: Bash (Cultist) → Defend → end turn",
      "≈ 9 HP more lost · win 96% → 94%",
      why,
    ]);
  });
  it("says (peeked)", () => {
    expect(turnChipText(rec({ peeked: true }))?.lines[2]).toMatch(/\(peeked\)$/);
  });
  it("is silent on good, tied and ungraded turns", () => {
    for (const verdict of ["best", "tie", "ungraded"] as const) {
      expect(turnChipText(rec({ verdict }))).toBeNull();
    }
  });
});

describe("TurnChip", () => {
  it("renders nothing for a toss-up", () => {
    const { container } = render(<TurnChip record={rec({ verdict: "tie" })} />);
    expect(container).toBeEmptyDOMElement();
  });
  it("seeks and dismisses", () => {
    const onSeek = vi.fn();
    const onDismiss = vi.fn();
    render(<TurnChip record={rec()} onSeek={onSeek} onDismiss={onDismiss} />);
    fireEvent.click(screen.getByTitle(/jump to the turn/));
    fireEvent.click(screen.getByLabelText("Dismiss"));
    expect(onSeek).toHaveBeenCalled();
    expect(onDismiss).toHaveBeenCalled();
  });
});
