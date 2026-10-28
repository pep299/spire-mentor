// Late coach replies. Every action and seek bumps a generation counter
// (`coachGenRef` in App); a grade requested at one generation describes the
// position of that moment, so it shows only if nothing has moved since —
// never late, on a turn or branch it no longer describes.
import type { ChoiceGrade, CoachChoiceRecord } from "../api";

export function isFresh(sentAt: number, now: number): boolean {
  return sentAt === now;
}

/** A live choice grade in the logged record's shape, for the Coach review. */
export function choiceRecordOf(g: ChoiceGrade, peeked: boolean): CoachChoiceRecord {
  return {
    kind: "choice",
    id: `choice:${g.node_id}`,
    node_id: g.node_id,
    decision: g.decision,
    label_kind: g.kind,
    verdict: g.verdict,
    strength: g.strength,
    played: g.played,
    top: g.top,
    alternatives: g.alternatives,
    peeked,
  };
}
