import type { CoachTurnRecord } from "../api";
import { turnChipText } from "../coach/coachText";

/** Coach chip for one graded combat turn — live (the wasm's
 *  `last-turn-grade`) and in follow / replay (the advisor's logged record). */
export default function TurnChip({ record, onDismiss, onSeek }: {
  record: CoachTurnRecord;
  onDismiss?: () => void;
  /** "show me": jump to the turn's first decision. */
  onSeek?: () => void;
}) {
  const text = turnChipText(record);
  if (!text) return null;
  const tone = record.verdict === "blunder" ? "var(--hp)" : "var(--warn, #c79a3a)";
  return (
    <div className="mono" data-testid="turn-chip" style={{ marginBottom: 8, padding: "6px 10px", borderRadius: 4,
      color: tone, border: `1px solid ${tone}`, background: "var(--field-bg)",
      fontSize: 11, display: "flex", gap: 8, alignItems: "baseline" }}>
      <div style={{ flex: 1 }}>
        <div><b>{text.title}</b></div>
        {text.lines.map((l, i) => <div key={i}>{l}</div>)}
      </div>
      {onSeek && (
        <button type="button" onClick={onSeek} title="jump to the turn's first decision"
          style={{ font: "inherit", background: "none", border: "none", color: "var(--muted)", cursor: "pointer" }}>
          ↗
        </button>
      )}
      {onDismiss && (
        <button type="button" onClick={onDismiss} aria-label="Dismiss"
          style={{ font: "inherit", background: "none", border: "none", color: "var(--muted)", cursor: "pointer" }}>
          ✕
        </button>
      )}
    </div>
  );
}
