/** Peek's one line: the AI's next action, nothing else (no scores, no list).
 *  `text` is the full line, so the Hints-style notice renders the same way. */
export default function PeekBanner({ text, onDismiss }: { text: string; onDismiss: () => void }) {
  return (
    <div className="mono" role="status" data-testid="peek-banner"
      style={{ marginBottom: 8, padding: "4px 8px", borderRadius: 4, fontSize: 11,
        border: "1px solid var(--accent)", display: "flex", gap: 8, alignItems: "baseline" }}>
      <span style={{ flex: 1 }}>👁 {text}</span>
      <button type="button" onClick={onDismiss} aria-label="dismiss peek"
        style={{ border: "none", background: "none", cursor: "pointer", font: "inherit", color: "var(--muted)" }}>
        ✕
      </button>
    </div>
  );
}
