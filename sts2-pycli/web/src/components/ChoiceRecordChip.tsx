import type { CoachChoiceRecord } from "../api";
import { choiceChipText } from "../coach/coachText";

/** A logged run-level choice grade (follow / replay): the mod's chip copy.
 *  No "check it" button — the rollout check needs a live session. */
export default function ChoiceRecordChip({ record, onSeek }: {
  record: CoachChoiceRecord;
  onSeek?: () => void;
}) {
  const text = choiceChipText(record);
  if (!text) return null;
  const tone = "var(--warn, #c79a3a)";
  return (
    <div className="mono" data-testid="choice-record-chip" style={{ marginBottom: 8, padding: "6px 10px",
      borderRadius: 4, color: tone, border: `1px solid ${tone}`, background: "var(--field-bg)", fontSize: 11,
      display: "flex", gap: 8, alignItems: "baseline" }}>
      <div style={{ flex: 1 }}>
        <div><b>{text.title}</b></div>
        {text.lines.map((l, i) => <div key={i}>{l}</div>)}
      </div>
      {onSeek && (
        <button type="button" onClick={onSeek} title="jump to this decision"
          style={{ font: "inherit", background: "none", border: "none", color: "var(--muted)", cursor: "pointer" }}>
          ↗
        </button>
      )}
    </div>
  );
}
