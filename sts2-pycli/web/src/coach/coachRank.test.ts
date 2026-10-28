import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { CoachChoiceRecord, CoachTurnRecord } from "../api";
import { rankAutopsy } from "./coachRank";

// The fixture lives with the Python twin's tests (realgame_coach.rank_autopsy).
const CASES = JSON.parse(
  readFileSync(resolve(__dirname, "../../../tests/fixtures/coach/rank_autopsy_cases.json"), "utf8"),
) as {
  cases: {
    name: string;
    turns: CoachTurnRecord[];
    choices: CoachChoiceRecord[];
    k: number;
    expect: [string, string][];
  }[];
};

describe("rankAutopsy (shared fixture with realgame_coach.rank_autopsy)", () => {
  for (const c of CASES.cases) {
    it(c.name, () => {
      const items = rankAutopsy(c.turns, c.choices, c.k);
      expect(items.map((i) => [i.kind, i.record.id])).toEqual(c.expect);
    });
  }
});
