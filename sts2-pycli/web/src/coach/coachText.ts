// The words on a coach chip, from one `coach` record: the mod's copy
// (`CoachChipText.cs`), so the workbench and the game say the same thing.
import type { CoachChoiceRecord, CoachTurnRecord } from "../api";

const pct = (p: number | null | undefined) => (p == null ? "?" : `${Math.round(p * 100)}%`);

/** The lines a turn record reads as. `null` = render nothing:
 *  a good turn, a toss-up, an ungraded one. */
export function turnChipText(rec: CoachTurnRecord): { title: string; lines: string[] } | null {
  if (rec.verdict !== "slip" && rec.verdict !== "blunder") return null;
  const hp = rec.hp_drop != null && Math.round(rec.hp_drop) >= 1
    ? `≈ ${Math.round(rec.hp_drop)} HP more lost · ` : "";
  return {
    title: rec.verdict === "blunder" ? "⚠ blunder" : "▽ suboptimal",
    lines: [
      `Turn ${rec.turn ?? "?"} · you: ${rec.you.join(" → ")}`,
      `AI: ${rec.ai.join(" → ")}`,
      `${hp}win ${pct(rec.win_before)} → ${pct(rec.win_after)}${rec.peeked ? " (peeked)" : ""}`,
      ...(rec.why ? [rec.why] : []),
    ],
  };
}

/** A turn as the recap / autopsy show it (the mod's `CoachPanelText.TurnBlock`):
 *  the position it started from, then your line, the AI's, the cost, and
 *  the reason when there is one. */
export function turnBlockLines(rec: CoachTurnRecord): string[] {
  const lines: string[] = [];
  const n = rec.turn ?? 0;
  const ctx = rec.context;
  if (ctx) {
    const hand = ctx.hand.join(", ");
    lines.push(`Turn ${n} — hand: ${hand || "(empty)"} · ${ctx.energy ?? 0} energy`);
    for (const e of ctx.enemies ?? []) {
      if (e.intent) lines.push(`${e.name} intends: ${e.intent}`);
    }
  } else {
    lines.push(`Turn ${n}`);
  }
  lines.push(`You: ${rec.you.join(" → ")}`);
  lines.push(`AI: ${rec.ai.join(" → ")}`);
  const peeked = rec.peeked ? " (peeked)" : "";
  if (rec.hp_drop != null && Math.round(rec.hp_drop) >= 1) {
    lines.push(`≈ ${Math.round(rec.hp_drop)} HP more lost${peeked}`);
  } else {
    lines.push(`win ${pct(rec.win_before)} → ${pct(rec.win_after)}${peeked}`);
  }
  if (rec.why) lines.push(rec.why);
  return lines;
}

/** "1 turn", "3 turns". */
export function plural(n: number, noun: string): string {
  return `${n} ${n === 1 ? noun : `${noun}s`}`;
}

/** "1st", "2nd", "3rd", "11th", "22nd" (the mod's `Ordinal`). */
export function ordinal(n: number): string {
  const m = Math.abs(n);
  const teen = m % 100 >= 11 && m % 100 <= 13;
  const suffix = teen ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[m % 10] ?? "th";
  return `${n}${suffix}`;
}

/** The projected HP at the far end of a map route: its last node with one. */
function routeHp(route: unknown): number | null {
  if (!Array.isArray(route)) return null;
  let hp: number | null = null;
  for (const step of route) {
    const h = (step as { hp?: unknown } | null)?.hp;
    if (typeof h === "number") hp = h;
  }
  return hp;
}

/** A logged choice record's lines. `null` = render nothing (agreed / toss-up).
 *  Non-combat wording is agreement, never a cost or win chance. */
export function choiceChipText(rec: CoachChoiceRecord): { title: string; lines: string[] } | null {
  if (rec.verdict !== "flag" || !rec.played || !rec.top) return null;
  const you = rec.played.label;
  const rank = rec.played.rank > 0 ? ` · ${you} was its ${ordinal(rec.played.rank)} choice` : "";
  const lines = [
    `AI would take: ${rec.top.label} (${rec.strength ?? "preference"})${rank}${rec.peeked ? " (peeked)" : ""}`,
  ];
  const youHp = routeHp(rec.path?.you);
  const aiHp = routeHp(rec.path?.ai);
  if (youHp != null && aiHp != null) lines.push(`route HP: you ≈${Math.round(youHp)} · AI ≈${Math.round(aiHp)}`);
  return { title: `${rec.label_kind} · you: ${you}`, lines };
}
