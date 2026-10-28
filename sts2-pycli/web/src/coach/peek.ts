// Peek (P, Coach style): the AI's pick for the CURRENT decision, as one line —
// the next action only, never the ranked list or its scores.
import type { Score } from "../api";
import { enemyLabeller } from "../enemyLabels";

interface NamedEnemy {
  index?: number;
  name?: string;
}

/** The best row of a combat search as a label: `"Bash (Cultist)"` when the
 *  fight has more than one enemy, `"end turn"` for the end-turn row. */
export function peekLabelFromScores(scores: Score[], enemies: NamedEnemy[] = []): string | null {
  const best = scores.find((s) => s.is_best) ?? null;
  if (!best) return null;
  const label = best.stable_idx === -1 ? "end turn" : best.action;
  if (best.target_index == null || enemies.length < 2) return label;
  const target = enemies.find((e) => e.index === best.target_index);
  if (!target?.name) return label;
  // Numbered like the nameplate: "Bash (Two-Tailed Rat #2)" says WHICH rat.
  const named = enemies.filter(
    (e): e is { index: number; name: string } => e.index != null && e.name != null,
  );
  return `${label} (${enemyLabeller(named)(best.target_index)})`;
}

interface Option {
  action: string;
  args: Record<string, unknown>;
  label?: string;
}

function sameArgs(a: Record<string, unknown>, b: Record<string, unknown>): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) {
    if (JSON.stringify(a[k] ?? null) !== JSON.stringify(b[k] ?? null)) return false;
  }
  return true;
}

/** The screen option the run-level policy picked, by its own label. */
export function peekLabelFromAction(
  rec: { action: string; args: Record<string, unknown> } | null,
  options: Option[] = [],
): string | null {
  if (!rec) return null;
  const o = options.find((x) => x.action === rec.action && sameArgs(x.args ?? {}, rec.args ?? {}));
  return o?.label ?? null;
}
