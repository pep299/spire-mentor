// The death autopsy's ranking for LIVE play in the browser, where no advisor
// writes an autopsy record. Follow mode shows the advisor's own autopsy.
//
// Twin of `realgame_coach.rank_autopsy`; both are pinned by
// `sts2-pycli/tests/fixtures/coach/rank_autopsy_cases.json`. Combat turns and
// run-level choices share no scale (spec, "Autopsy ranking"): turns rank by
// expected HP lost (win-chance points when HP is unknown), choices by how far
// the AI's top pick out-weighed yours, interleaved turns first.
import type { CoachChoiceRecord, CoachTurnRecord } from "../api";

export const AUTOPSY_ITEMS = 3;

export type AutopsyItem =
  | { kind: "turn"; record: CoachTurnRecord }
  | { kind: "choice"; record: CoachChoiceRecord };

/** A graded turn worth naming: a slip or blunder with its position. */
function costly(t: CoachTurnRecord): boolean {
  return (t.verdict === "slip" || t.verdict === "blunder") && !!t.context;
}

export function turnCost(t: CoachTurnRecord): number {
  return t.hp_drop != null ? t.hp_drop : t.severity ?? 0;
}

export function choiceGap(c: CoachChoiceRecord): number {
  return (c.top?.prob ?? 0) - (c.played?.prob ?? 0);
}

export function rankAutopsy(
  turns: CoachTurnRecord[],
  choices: CoachChoiceRecord[],
  k: number = AUTOPSY_ITEMS,
): AutopsyItem[] {
  const ts = turns.filter(costly).sort((a, b) => turnCost(b) - turnCost(a));
  const cs = choices
    .filter((c) => c.verdict === "flag")
    .sort((a, b) => choiceGap(b) - choiceGap(a) || (b.top?.prob ?? 0) - (a.top?.prob ?? 0));
  const out: AutopsyItem[] = [];
  let ti = 0;
  let ci = 0;
  while (out.length < k && (ti < ts.length || ci < cs.length)) {
    if (ti < ts.length && (out.length % 2 === 0 || ci >= cs.length)) {
      out.push({ kind: "turn", record: ts[ti++] });
    } else {
      out.push({ kind: "choice", record: cs[ci++] });
    }
  }
  return out;
}
