import { describe, expect, it } from "vitest";
import type { CoachChoiceRecord, CoachLog, CoachTurnRecord } from "../api";
import { recordsForStep, stepTones } from "./coachSelectors";

const turn = (id: string, verdict: CoachTurnRecord["verdict"], log_idx: number): CoachTurnRecord => ({
  kind: "turn", id, node_id: id, turn: 1, floor: 1, verdict, you: [], ai: [], hp_drop: null,
  win_before: null, win_after: null, peeked: false, ungraded_actions: 0, log_idx,
});
const choice = (id: string, verdict: CoachChoiceRecord["verdict"], log_idx: number): CoachChoiceRecord => ({
  kind: "choice", id, node_id: id, decision: "card_reward", label_kind: "Card reward", verdict,
  played: { label: "a", prob: 0.1, rank: 2 }, top: { label: "b", prob: 0.6 }, alternatives: [],
  peeked: false, log_idx,
});

const log: CoachLog = {
  records: [turn("turn:1", "slip", 0), choice("choice:2", "flag", 0), turn("turn:3", "best", 4)],
  // "turn:9" has no record (withdrawn): the id must resolve to nothing.
  by_step: { "0": ["turn:1", "choice:2"], "4": ["turn:3"], "6": ["turn:9"] },
  autopsy: null,
};

describe("recordsForStep", () => {
  it("returns the step's records in log order", () => {
    expect(recordsForStep(log, 0).map((r) => r.id)).toEqual(["turn:1", "choice:2"]);
  });
  it("skips ids whose record was withdrawn", () => {
    expect(recordsForStep(log, 6)).toEqual([]);
  });
  it("is empty without a log or for an unmarked step", () => {
    expect(recordsForStep(null, 0)).toEqual([]);
    expect(recordsForStep(log, 1)).toEqual([]);
  });
});

describe("stepTones", () => {
  it("ticks costly steps by their worst record, and only those", () => {
    expect([...stepTones(log)]).toEqual([[0, "slip"]]);
    const worse: CoachLog = {
      ...log,
      records: [...log.records, turn("turn:5", "blunder", 4)],
      by_step: { ...log.by_step, "4": ["turn:3", "turn:5"] },
    };
    expect(stepTones(worse).get(4)).toBe("blunder");
  });
});
