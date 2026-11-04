// The rollout check behind ChoiceChip's "check it": its budget and its
// one-line summary. Kept out of ChoiceChip.tsx so that file exports only the
// component (react-refresh/only-export-components).
import type { ChoiceCheck } from "../api";

/** The rollout check's one-line summary: win rates and HP, you vs the AI.
 *  A win-rate gap inside two standard errors reads "within noise" — a
 *  dozen rollouts rarely separate two options, and the chip must not crown
 *  a winner from noise. */
export function checkSummary(check: ChoiceCheck): string | null {
  const you = check.candidates.find((c) => c.key === "you");
  const ai = check.candidates.find((c) => c.key === "ai");
  if (!you || !ai) return null;
  const pct = (p: number) => `${Math.round(p * 100)}%`;
  const se = Math.sqrt((you.win_rate * (1 - you.win_rate)) / Math.max(you.n, 1)
    + (ai.win_rate * (1 - ai.win_rate)) / Math.max(ai.n, 1));
  const gap = Math.abs(you.win_rate - ai.win_rate);
  const close = gap === 0 || gap < 2 * se ? " · within noise" : "";
  return `rollouts to end of act (n=${you.n}): you ${pct(you.win_rate)} · AI ${pct(ai.win_rate)}${close}`
    + ` · avg HP ${Math.round(you.avg_final_hp)} vs ${Math.round(ai.avg_final_hp)}`;
}

/** A light rollout budget: the check runs in the engine worker and every
 *  move waits behind it, so it trades precision for ~a few seconds. */
export const CHECK_CONFIG = { rollouts: 12, sims: 32 };
