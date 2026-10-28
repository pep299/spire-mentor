// What follow / replay show from a run log's coach records (`GET /api/coach`).
// The log already applied withdrawals, so a withdrawn record is simply absent.
import type { CoachLog, CoachRecord } from "../api";

export type CoachTone = "blunder" | "slip" | "flag";

/** The graded turn / choice records on replay step `idx`, in log order. */
export function recordsForStep(log: CoachLog | null, idx: number): CoachRecord[] {
  if (!log) return [];
  const ids = log.by_step[String(idx)] ?? [];
  return ids
    .map((id) => log.records.find((r) => r.id === id))
    .filter((r): r is CoachRecord => !!r && r.kind !== "recap");
}

const RANK: Record<CoachTone, number> = { blunder: 3, slip: 2, flag: 1 };

function toneOf(r: CoachRecord): CoachTone | null {
  if (r.kind === "turn" && (r.verdict === "blunder" || r.verdict === "slip")) return r.verdict;
  if (r.kind === "choice" && r.verdict === "flag") return "flag";
  return null;
}

/** Timeline ticks: each step with a costly turn or a flagged choice, by its
 *  worst record. */
export function stepTones(log: CoachLog | null): Map<number, CoachTone> {
  const out = new Map<number, CoachTone>();
  if (!log) return out;
  for (const key of Object.keys(log.by_step)) {
    const idx = Number(key);
    for (const r of recordsForStep(log, idx)) {
      const t = toneOf(r);
      const prev = out.get(idx);
      if (t && (!prev || RANK[t] > RANK[prev])) out.set(idx, t);
    }
  }
  return out;
}
