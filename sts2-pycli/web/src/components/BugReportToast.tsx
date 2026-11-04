import { useEffect, useRef, useState } from "react";

export interface BugReportResult {
  id: string;
  /** The shareable link — or, when the report could not be sent (the
   *  mod-bundled workbench with no store to reach), the saved file's path. */
  url: string;
  /** Set when the report was saved locally instead of sent: why, and where. */
  detail?: string;
}

/**
 * The outcome of a bug-report capture, shown OUTSIDE the Tools menu.
 *
 * The 🐞 Report button is a row in a dropdown, and the dropdown unmounts its
 * rows the moment it closes — which the `window.prompt` for the note is
 * usually enough to do. So the id and its "copy link" button, rendered inline
 * beside the button, were gone before anyone could read them: a report was
 * captured and the link never reached the clipboard (user report 2026-09-12).
 * `App` owns the result instead and renders this, which outlives the menu.
 *
 * The copy is attempted once on mount (a clipboard write can be refused when
 * the gesture that started the capture is too far behind), and the URL stays
 * on screen — selectable, with an explicit Copy button — either way.
 */
export default function BugReportToast({
  result,
  error,
  onDismiss,
}: {
  result: BugReportResult | null;
  error: string | null;
  onDismiss: () => void;
}) {
  const [copied, setCopied] = useState<boolean | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") { e.preventDefault(); onDismiss(); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onDismiss]);

  // Best-effort auto-copy, so the common case needs no second click. A refusal
  // is not an error worth shouting about — the Copy button is right there.
  useEffect(() => {
    if (!result) return;
    let live = true;
    void Promise.resolve(navigator.clipboard?.writeText(result.url))
      .then(() => { if (live) setCopied(true); })
      .catch(() => { if (live) setCopied(false); });
    return () => { live = false; };
  }, [result]);

  if (!result && !error) return null;

  async function copy() {
    if (!result) return;
    try {
      await navigator.clipboard?.writeText(result.url);
      setCopied(true);
    } catch {
      // No clipboard permission: select the text so ctrl-C still works.
      setCopied(false);
      input.current?.select();
    }
  }

  return (
    <div role="dialog" aria-label="Bug report"
      style={{ position: "fixed", right: 16, bottom: 16, zIndex: 60, maxWidth: "min(560px, 92vw)",
        background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 10,
        boxShadow: "0 12px 40px oklch(30% 0.04 255 / .4)", padding: "10px 12px",
        display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
        <b>{error ? "🐞 Report failed" : "🐞 Report captured"}</b>
        <button type="button" onClick={onDismiss} aria-label="dismiss">✕</button>
      </div>
      {error ? (
        <span className="bugreport-error">{error}</span>
      ) : (
        <>
          <code>{result?.id}</code>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input ref={input} className="mono" readOnly value={result?.url ?? ""}
              aria-label="report URL" onFocus={(e) => e.currentTarget.select()}
              style={{ flex: "1 1 auto", minWidth: 0 }} />
            <button type="button" onClick={() => void copy()} title="Copy report URL">
              {copied ? "copied ✓" : /^https?:/.test(result?.url ?? "") ? "copy link" : "copy path"}
            </button>
          </div>
          {result?.detail && <span className="wb-hint" data-testid="bugreport-detail">{result.detail}</span>}
          {copied === false && (
            <span className="wb-hint">clipboard refused — the link is selected, copy it with ctrl-C</span>
          )}
        </>
      )}
    </div>
  );
}
