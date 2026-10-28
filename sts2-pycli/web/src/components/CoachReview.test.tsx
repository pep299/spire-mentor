import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { CoachChoiceRecord, CoachRecapRecord, CoachTurnRecord } from "../api";
import CoachReview from "./CoachReview";

const ctx = { hand: ["Strike", "Defend"], energy: 3, max_energy: 3, hp: 50, max_hp: 80, block: 0,
  enemies: [{ name: "Cultist", hp: 40, intent: "attack 6×2 + debuff" }] };
const turn = (id: string, verdict: CoachTurnRecord["verdict"], hp: number | null, log_idx = 1): CoachTurnRecord => ({
  kind: "turn", id, node_id: id, turn: 2, floor: 3, verdict, you: ["Strike", "end turn"],
  ai: ["Defend", "end turn"], hp_drop: hp, win_before: 0.9, win_after: 0.85, peeked: false,
  ungraded_actions: 0, log_idx, severity: hp ?? 5, context: ctx,
});
const choice: CoachChoiceRecord = {
  kind: "choice", id: "choice:9", node_id: 9, decision: "card_reward", label_kind: "Card reward",
  verdict: "flag", strength: "strong preference", played: { label: "Anger", prob: 0.05, rank: 3 },
  top: { label: "Inflame", prob: 0.7 }, alternatives: [], peeked: false, log_idx: 4,
};
const recap: CoachRecapRecord = {
  kind: "recap", id: "recap:1", node_id: 1, log_name: null, log_idx: 0, floor: 3, room_type: "monster",
  hp_start: 70, hp_end: 55, hp_lost: 15, ai_hp_lost: null, turns: 4, turns_graded: 3, turns_ungraded: 1,
  unfinished_turns: 0, costliest: [turn("turn:a", "blunder", 9)],
};

describe("CoachReview", () => {
  it("All decisions: turns by HP, choices apart, the ungraded count", () => {
    const records = [turn("turn:small", "slip", 3), turn("turn:big", "blunder", 11), turn("turn:u", "ungraded", null), choice];
    render(<CoachReview records={records} autopsy={null} onClose={() => {}} />);
    const rows = screen.getAllByRole("button", { name: /suboptimal|blunder/ });
    expect(rows[0]).toHaveTextContent("≈ 11 HP more lost");
    expect(rows[1]).toHaveTextContent("≈ 3 HP more lost");
    expect(screen.getByText(/1 turn ungraded/)).toBeTruthy();
    expect(screen.getByText(/AI would take: Inflame/)).toBeTruthy();
    expect(screen.getAllByText("Cultist intends: attack 6×2 + debuff")).toHaveLength(2);
  });

  it("a row seeks to its record", () => {
    const onSeek = vi.fn();
    render(<CoachReview records={[choice]} autopsy={null} onSeek={onSeek} onClose={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: /Card reward · you: Anger/ }));
    expect(onSeek).toHaveBeenCalledWith(choice);
  });

  it("Fights: the recap's HP and its costliest turn, no AI comparison", () => {
    render(<CoachReview records={[recap]} autopsy={null} initialTab="fights" onClose={() => {}} />);
    expect(screen.getByText("−15 HP · 1 turn ungraded")).toBeTruthy();
    expect(screen.queryByText(/AI line typically/)).toBeNull();
    expect(screen.getByText(/≈ 9 HP more lost/)).toBeTruthy();
  });

  it("opens on the Autopsy tab when there is one", () => {
    const autopsy = {
      items: [{ kind: "turn" as const, record: turn("turn:a", "blunder", 9) }, { kind: "choice" as const, record: choice }],
      floor: 7, turns_ungraded: 2, decisions_graded: 14,
    };
    render(<CoachReview records={[]} autopsy={autopsy} onClose={() => {}} />);
    expect(screen.getByText("Run autopsy · died on floor 7")).toBeTruthy();
    expect(screen.getByText(/1\. Combat turn · floor 3/)).toBeTruthy();
    expect(screen.getByText(/2\. Card reward · you: Anger/)).toBeTruthy();
    expect(screen.getByText(/14 decisions graded/)).toBeTruthy();
  });

  it("Escape closes", () => {
    const onClose = vi.fn();
    render(<CoachReview records={[]} autopsy={null} onClose={onClose} />);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });
});
