import { useEffect, useMemo, useRef, useState } from "react";
import type { LinkStatus } from "./link";
import { frameDoc, nextFrameHeight } from "./frameDoc";
import type { ClaudeLinkState, DockItem } from "./useClaudeLink";

/**
 * The Claude dock: what a linked Claude Code session put on screen with
 * `wb_show` — charts, tables, explanations — floating over the board's corner.
 *
 * Every item — HTML, markdown, Plotly figure — renders in a sandboxed iframe
 * (`allow-scripts`, no `allow-same-origin`): its scripts run in an opaque
 * origin, so they cannot reach this page, its worker, its IndexedDB, or the
 * relay (which refuses the `null` origin and an unpaired hello).
 */

const STATUS_LABEL: Record<LinkStatus, string> = {
  off: "off",
  waiting: "waiting for Claude Code",
  connected: "connected",
  replaced: "taken over by another tab",
  unauthorized: "pairing token refused",
};

function FrameView({ item }: { item: DockItem }) {
  const ref = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(160);
  const heightRef = useRef(160);
  const growthRef = useRef<number | null>(null);
  const frozenRef = useRef(false);
  const token = `item-${item.id}`;
  // A dock re-render (status flip, new item) must not rebuild every srcDoc —
  // an image item's is a multi-MB base64 string.
  const doc = useMemo(() => frameDoc(item, token), [item, token]);
  useEffect(() => {
    function onMessage(e: MessageEvent) {
      if (e.source !== ref.current?.contentWindow || frozenRef.current) return;
      const d = e.data as { claudeDock?: string; h?: unknown } | null;
      if (d?.claudeDock === token && typeof d.h === "number" && Number.isFinite(d.h)) {
        const next = nextFrameHeight(heightRef.current, d.h, growthRef.current);
        if (next.loop) frozenRef.current = true;
        growthRef.current = next.growth;
        heightRef.current = next.height;
        setHeight(next.height);
      }
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [token]);
  return (
    <iframe ref={ref} title={item.title} sandbox="allow-scripts" srcDoc={doc}
      style={{ width: "100%", height, border: 0, background: "#fff", borderRadius: 4, display: "block" }} />
  );
}

export default function ClaudeDock({ link }: { link: ClaudeLinkState }) {
  const [open, setOpen] = useState(true);
  const [wide, setWide] = useState(false);
  // A new item pops the dock open again.
  useEffect(() => {
    if (link.lastShownId > 0) setOpen(true);
  }, [link.lastShownId]);

  if (link.status === "off" && link.items.length === 0) return null;
  const dot = link.status === "connected" ? "oklch(72% 0.17 150)" : link.status === "waiting" ? "var(--gold)" : "var(--muted)";

  if (!open) {
    return (
      <button onClick={() => setOpen(true)} title={`Claude link: ${STATUS_LABEL[link.status]}`}
        style={{ position: "fixed", right: 16, bottom: 16, zIndex: 55, display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ width: 8, height: 8, borderRadius: 4, background: dot }} />
        🤖 Claude{link.items.length ? ` · ${link.items.length}` : ""}
      </button>
    );
  }
  return (
    <aside aria-label="Claude dock"
      style={{ position: "fixed", right: 16, bottom: 16, zIndex: 55,
        width: wide ? "min(960px, 94vw)" : "min(480px, 92vw)", maxHeight: "78vh",
        display: "flex", flexDirection: "column", background: "var(--panel)", color: "var(--text)",
        border: "1px solid var(--border)", borderRadius: 8, boxShadow: "var(--shadow)" }}>
      <header style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px",
        borderBottom: "1px solid var(--border)" }}>
        <span style={{ width: 8, height: 8, borderRadius: 4, background: dot }} />
        <strong>🤖 Claude</strong>
        <span style={{ color: "var(--muted)", fontSize: 12 }}>{STATUS_LABEL[link.status]}</span>
        <span style={{ flex: 1 }} />
        {link.status === "unauthorized" && <button onClick={link.pair} title="enter the token Claude gives you">pair…</button>}
        {link.items.length > 0 && <button onClick={link.clear} title="remove every item">clear</button>}
        <button onClick={() => setWide((w) => !w)} title={wide ? "narrow" : "wide"}>{wide ? "⇥" : "⇤"}</button>
        <button onClick={() => setOpen(false)} title="minimise">—</button>
      </header>
      <div style={{ overflowY: "auto", padding: 10, display: "flex", flexDirection: "column", gap: 12 }}>
        {link.items.length === 0 && (
          <div style={{ color: "var(--muted)", fontSize: 13 }}>
            {link.status === "connected"
              ? "Linked. Ask Claude Code in your terminal about this run — what it shows lands here."
              : link.status === "replaced"
                ? "Another workbench tab took the link. Turn it off and on in Tools to take it back."
                : link.status === "unauthorized"
                  ? "The relay refused this page's pairing token. Ask Claude for the workbench pairing token, then press pair…."
                : "Waiting for a Claude Code session started in the repo (it runs the relay from .mcp.json)."}
          </div>
        )}
        {link.items.map((item) => (
          <section key={item.id}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
              <strong style={{ flex: 1, fontSize: 13 }}>{item.title}</strong>
              <span style={{ color: "var(--muted)", fontSize: 11 }}>
                {new Date(item.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </span>
              <button onClick={() => link.dismiss(item.id)} title="remove" aria-label={`remove ${item.title}`}>✕</button>
            </div>
            <FrameView item={item} />
          </section>
        ))}
      </div>
    </aside>
  );
}

/** The Tools-menu row: turns the link on / off and shows its state. */
export function ClaudeLinkToggle({ link }: { link: ClaudeLinkState }) {
  const on = link.url != null;
  return (
    <button aria-pressed={on} onClick={() => link.setEnabled(!on)}
      title={on ? `Claude link to ${link.url}: ${STATUS_LABEL[link.status]} — click to turn off`
        : "let a Claude Code session in your terminal read this run, run analyses on it, and show charts here"}>
      🤖 Claude link{on ? ` · ${STATUS_LABEL[link.status]}` : ""}
    </button>
  );
}
