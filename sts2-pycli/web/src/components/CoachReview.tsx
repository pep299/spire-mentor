import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { CoachChoiceRecord, CoachRecapRecord, CoachRecord, CoachTurnRecord } from "../api";
import { choiceGap, turnCost, type AutopsyItem } from "../coach/coachRank";
import { choiceChipText, plural, turnBlockLines } from "../coach/coachText";

/**
 * Coach review — every graded decision of a run (follow / replay: what the
 * real-game advisor logged) or of this live session, as a bottom dock shaped
 * like the regret review (fixed, Escape closes, reports its height).
 *
 * Three tabs:
 *   - Autopsy: the run's costliest decisions, when the run ended in a death.
 *   - Fights: one recap per won fight (logged by the advisor).
 *   - All decisions: costly and tied turns by HP (else win-chance points),
 *     choices by how far the AI's pick out-weighed yours — two groups, never
 *     merged onto one scale.
 *
 * Every row seeks to its decision (`onSeek`) without closing the dock.
 */

export type CoachSeekTarget = CoachTurnRecord | CoachChoiceRecord | CoachRecapRecord;
export type CoachTab = "autopsy" | "fights" | "all";

export interface CoachAutopsyView {
  items: AutopsyItem[];
  floor?: number | null;
  turns_ungraded?: number;
  decisions_graded?: number;
}

function Row({ heading, lines, tone, onSeek }: {
  heading: string; lines: string[]; tone: string; onSeek?: () => void;
}) {
  return (
    <button type="button" onClick={onSeek} title={onSeek ? "jump to this decision" : undefined}
      style={{ display: "block", width: "100%", textAlign: "left", cursor: onSeek ? "pointer" : "default",
        background: "var(--field-bg)", border: "1px solid var(--border)", borderRadius: 8,
        padding: "8px 12px", marginBottom: 8, color: "var(--text)", font: "inherit" }}>
      <div style={{ color: tone, fontWeight: 700 }}>{heading}</div>
      {lines.map((l, i) => <div key={i} style={{ fontSize: 12, marginTop: 2 }}>{l}</div>)}
    </button>
  );
}

const TONE: Record<string, string> = {
  blunder: "var(--hp)",
  slip: "var(--warn, #c79a3a)",
  tie: "var(--muted)",
  flag: "var(--warn, #c79a3a)",
  agree: "var(--accent-strong)",
  tossup: "var(--muted)",
};

function choiceLines(c: CoachChoiceRecord): string[] {
  const text = choiceChipText(c);
  if (text) return text.lines;
  const agree = c.verdict === "agree" ? "AI agrees" : "toss-up";
  return [`${agree} · you: ${c.played?.label ?? "?"}${c.peeked ? " (peeked)" : ""}`];
}

export default function CoachReview({
  records, autopsy, initialTab, live = false, onSeek, onClose, onHeight,
}: {
  records: CoachRecord[];
  autopsy: CoachAutopsyView | null;
  initialTab?: CoachTab;
  live?: boolean;
  onSeek?: (r: CoachSeekTarget) => void;
  onClose: () => void;
  /** The dock's rendered height (px), so the parent can pad scroll regions
   *  out from under it; called with 0 on unmount. */
  onHeight?: (px: number) => void;
}) {
  const [tab, setTab] = useState<CoachTab>(initialTab ?? (autopsy ? "autopsy" : "all"));
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") { e.preventDefault(); onClose(); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const rootRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = rootRef.current;
    if (!el || !onHeight) return;
    const report = () => onHeight(el.offsetHeight);
    report();
    const ro = new ResizeObserver(report);
    ro.observe(el);
    return () => { ro.disconnect(); onHeight(0); };
  }, [onHeight]);

  const turns = records.filter((r): r is CoachTurnRecord => r.kind === "turn");
  const choices = records.filter((r): r is CoachChoiceRecord => r.kind === "choice");
  const recaps = records.filter((r): r is CoachRecapRecord => r.kind === "recap");
  const shownTurns = turns
    .filter((t) => t.verdict === "slip" || t.verdict === "blunder" || t.verdict === "tie")
    .sort((a, b) => turnCost(b) - turnCost(a));
  const shownChoices = choices.filter((c) => c.verdict !== "ungraded").sort((a, b) => choiceGap(b) - choiceGap(a));
  const ungraded = turns.filter((t) => t.verdict === "ungraded").length;
  const seek = (r: CoachSeekTarget) => (onSeek ? () => onSeek(r) : undefined);

  const tabs: [CoachTab, string][] = [
    ...(autopsy ? [["autopsy", "Autopsy"] as [CoachTab, string]] : []),
    ["fights", `Fights (${recaps.length})`],
    ["all", `All decisions (${shownTurns.length + shownChoices.length})`],
  ];

  return (
    <div ref={rootRef} className="mono" role="region" aria-label="Coach Review"
      style={{ position: "fixed", left: 0, right: 0, bottom: 0, zIndex: 50,
        display: "flex", flexDirection: "column", maxHeight: "46vh",
        background: "var(--bg)", borderTop: "1px solid var(--border)",
        boxShadow: "0 -8px 28px oklch(30% 0.04 255 / .35)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 16px",
        borderBottom: "1px solid var(--border)" }}>
        <b>Coach review{live ? " · live run" : ""}</b>
        <span role="tablist" style={{ display: "flex", gap: 4 }}>
          {tabs.map(([id, label]) => (
            <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)}
              style={{ font: "inherit", fontWeight: tab === id ? 700 : 400 }}>{label}</button>
          ))}
        </span>
        <span style={{ marginLeft: "auto" }} />
        <button onClick={onClose} aria-label="Close">✕</button>
      </div>
      <div style={{ padding: 16, overflow: "auto" }}>
        {tab === "autopsy" && autopsy && (
          <>
            <div style={{ marginBottom: 8 }}>
              <b>Run autopsy{autopsy.floor != null ? ` · died on floor ${autopsy.floor}` : ""}</b>
            </div>
            {autopsy.items.length === 0 && <div>No costly decisions found.</div>}
            {autopsy.items.map((item, i) => item.kind === "turn"
              ? <Row key={item.record.id} tone={TONE[item.record.verdict] ?? "var(--text)"}
                  heading={`${i + 1}. Combat turn${item.record.floor != null ? ` · floor ${item.record.floor}` : ""}`}
                  lines={turnBlockLines(item.record)} onSeek={seek(item.record)} />
              : <Row key={item.record.id} tone={TONE.flag}
                  heading={`${i + 1}. ${item.record.label_kind} · you: ${item.record.played?.label ?? "?"}`}
                  lines={choiceLines(item.record)} onSeek={seek(item.record)} />)}
            {(autopsy.decisions_graded != null || autopsy.turns_ungraded != null) && (
              <div style={{ color: "var(--muted)", fontSize: 11 }}>
                {plural(autopsy.decisions_graded ?? 0, "decision")} graded
                {" · "}{plural(autopsy.turns_ungraded ?? 0, "turn")} ungraded
              </div>
            )}
          </>
        )}
        {tab === "fights" && (
          <>
            {recaps.length === 0 && (
              <div style={{ color: "var(--muted)" }}>
                {live ? "Fight recaps come from the real-game advisor (follow mode)." : "No fight recaps in this run."}
              </div>
            )}
            {recaps.map((r) => (
              <div key={r.id} style={{ marginBottom: 12 }}>
                <Row tone="var(--text)" heading={`Fight recap · floor ${r.floor ?? "?"}`}
                  lines={[`−${r.hp_lost} HP${r.turns_ungraded > 0 ? ` · ${plural(r.turns_ungraded, "turn")} ungraded` : ""}`]}
                  onSeek={seek(r)} />
                {r.costliest.map((t) => (
                  <div key={t.id} style={{ marginLeft: 16 }}>
                    <Row tone={TONE[t.verdict] ?? "var(--text)"} heading={t.verdict === "blunder" ? "⚠ blunder" : "▽ suboptimal"}
                      lines={turnBlockLines(t)} onSeek={seek(t)} />
                  </div>
                ))}
              </div>
            ))}
          </>
        )}
        {tab === "all" && (
          <>
            <div style={{ fontWeight: 700, margin: "0 0 6px" }}>Combat turns · by HP lost</div>
            {shownTurns.length === 0 && <div style={{ color: "var(--muted)", marginBottom: 8 }}>No costly turns.</div>}
            {shownTurns.map((t) => (
              <Row key={t.id} tone={TONE[t.verdict] ?? "var(--text)"}
                heading={t.verdict === "blunder" ? "⚠ blunder" : t.verdict === "slip" ? "▽ suboptimal" : "≈ toss-up"}
                lines={turnBlockLines(t)} onSeek={seek(t)} />
            ))}
            {ungraded > 0 && (
              <div style={{ color: "var(--muted)", fontSize: 11, marginBottom: 8 }}>
                {plural(ungraded, "turn")} ungraded (too close to call, or not gradable)
              </div>
            )}
            <div style={{ fontWeight: 700, margin: "8px 0 6px" }}>Run choices · by how strongly the AI disagreed</div>
            {shownChoices.length === 0 && <div style={{ color: "var(--muted)" }}>No graded choices.</div>}
            {shownChoices.map((c) => (
              <Row key={c.id} tone={TONE[c.verdict] ?? "var(--text)"}
                heading={`${c.label_kind} · you: ${c.played?.label ?? "?"}`}
                lines={choiceLines(c)} onSeek={seek(c)} />
            ))}
          </>
        )}
      </div>
    </div>
  );
}
