import { useEffect, useRef } from "react";
import type { LiveNode, LiveTimeline } from "../api";
import { hasReadableRow } from "../combatlog";
import { endedSelection } from "../textSelection";
import { DetailToggle, StepEvents, TurnHeader } from "./turnLedger";
import { turnChips, useDetailMode } from "./turnLedgerState";

/**
 * The live-run event log, rendered from the server's branch tree.
 *
 * Each row is one decision and is **clickable to seek** the run to that point.
 * The current position is marked. Where the run was rewound and replayed
 * differently, a fork exists: the alternative line is offered as an inline
 * "other line" chip so you can jump between branches. The original line is
 * always preserved — acting after a rewind forks rather than discards.
 *
 * Seeking back doesn't truncate the log: the steps *after* the current position
 * stay visible (de-emphasized) so you can jump forward again, exactly like the
 * branch tree keeps the whole branch on screen and only de-highlights what's
 * off the active path. Acting from an earlier step forks a new line; the new
 * leaf has no descendants, so the log naturally ends there and the old
 * continuation drops to an "other line" chip.
 *
 * Combat stretches render as a TURN LEDGER: steps group under "TURN N"
 * headers summarized by chips (damage dealt / block / HP lost / kills), each
 * step carrying its action's slice of the engine combat log as the same
 * coloured sentences the arena panel shows. A node's own `turn` is its
 * post-action round, so the round its action was *taken* in — what the ledger
 * groups by — is its parent's turn.
 */
export default function LiveLog({
  timeline,
  onSeek,
  busy,
}: {
  timeline: LiveTimeline | null;
  onSeek: (nodeId: number) => void;
  busy?: boolean;
}) {
  const [mode, setMode] = useDetailMode();
  // Keep the current step in view as the log grows (e.g. each card played in
  // combat appends a row), so the latest action is always visible.
  const leafRef = useRef<HTMLDivElement>(null);
  const leaf = timeline?.active_leaf ?? null;
  useEffect(() => {
    leafRef.current?.scrollIntoView({ block: "nearest" });
  }, [leaf, timeline?.nodes.length]);

  if (!timeline || timeline.nodes.length === 0) return null;
  const byId = new Map(timeline.nodes.map((n) => [n.id, n]));
  const path = timeline.active_path;
  const hasForks = timeline.nodes.some((n) => n.child_ids.length > 1);

  // Extend the active path (root → current) forward to the tip of the line so
  // seeking back doesn't hide the later steps. At a fork we follow the most
  // recently created branch; the others stay reachable via the "other line"
  // chips rendered per step. A freshly forked leaf has no children, so this adds
  // nothing and the log ends at the new step.
  const forward: number[] = [];
  if (leaf != null) {
    const seen = new Set<number>(path);
    let cur = byId.get(leaf);
    while (cur && cur.child_ids.length > 0) {
      const nextId = cur.child_ids[cur.child_ids.length - 1];
      if (seen.has(nextId)) break; // defensive: trees are acyclic, but never loop
      seen.add(nextId);
      forward.push(nextId);
      cur = byId.get(nextId);
    }
  }
  const displayPath = [...path, ...forward];
  const futureFrom = path.length; // indices >= this are steps after the current one

  // The turn each step's action was taken in: the parent's (pre-action) round.
  const actingTurn = (n: LiveNode): number | null => {
    const parent = n.parent_id != null ? byId.get(n.parent_id) : undefined;
    return parent?.turn ?? null;
  };
  // Chips aggregate per turn group over the displayed line.
  const chipsByStart = new Map<number, string[]>();
  {
    let start = -1;
    let lastTurn: number | null = null;
    let rowsAcc: LiveNode["combat_log"] = [];
    let detAcc: string[] = [];
    const flush = () => {
      if (start >= 0 && lastTurn != null) {
        chipsByStart.set(start, turnChips(rowsAcc ?? [], detAcc));
      }
    };
    displayPath.forEach((id, i) => {
      const n = byId.get(id);
      if (!n) return;
      const t = actingTurn(n);
      if (t !== lastTurn) {
        flush();
        start = i;
        lastTurn = t;
        rowsAcc = [];
        detAcc = [];
      }
      rowsAcc = [...(rowsAcc ?? []), ...(n.combat_log ?? [])];
      detAcc = [...detAcc, ...n.details];
    });
    flush();
  }
  // `hasReadableRow`, not the raw slice: the toggle it gates only changes which
  // READABLE rows show, so a step carrying nothing but view-only rows has no
  // tape as far as this is concerned.
  const anyRows = displayPath.some((id) => hasReadableRow(byId.get(id)?.combat_log ?? []));

  // Track the act/floor as we walk the active path so we can drop a labelled
  // separator the moment either changes — an "Act N" divider when the act
  // advances, a lighter "Floor N" rule when only the floor ticks over. This
  // makes it obvious where one act/floor ends and the next begins.
  let lastAct: number | null | undefined;
  let lastFloor: number | null | undefined;
  let lastTurnShown: number | null = null;

  return (
    // Header ("history · click a step to rewind") is rendered by the parent in a
    // pinned region so only this list scrolls; LiveLog is the list itself.
    <div className="mono">
      {anyRows && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 4 }}>
          <DetailToggle mode={mode} onChange={setMode} />
        </div>
      )}
      {displayPath.map((id, i) => {
        const n = byId.get(id);
        if (!n) return null;
        const parent = n.parent_id != null ? byId.get(n.parent_id) : undefined;
        // Alternative branches at this step: the parent's other children.
        const siblings = parent ? parent.child_ids.filter((c) => c !== n.id) : [];
        const isLeaf = id === leaf;
        // Steps after the current position: shown so you can jump forward, but
        // de-emphasized since they're "ahead" of where the run sits now.
        const isFuture = i >= futureFrom;

        const act = n.act ?? null;
        const floor = n.floor ?? null;
        const showAct = act != null && act !== lastAct;
        // A new act always implies a new floor; only label the floor on its own
        // when the act is unchanged, to avoid stacking two separators.
        const showFloor = !showAct && floor != null && floor !== lastFloor;
        lastAct = act;
        lastFloor = floor;

        const turn = actingTurn(n);
        const showTurn = turn != null && turn !== lastTurnShown;
        lastTurnShown = turn;

        return (
          <div key={id} style={{ display: "flex", flexDirection: "column" }}>
            {showAct && (
              <div
                aria-label={`Act ${act}`}
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  margin: "12px 0 5px", fontSize: 10, letterSpacing: ".08em",
                  textTransform: "uppercase", color: "var(--accent, #6aa0ff)",
                }}
              >
                <span aria-hidden style={{ height: 1, flex: 1, background: "currentColor", opacity: 0.4 }} />
                <span style={{ fontWeight: 700 }}>
                  Act {act}{floor != null ? ` · floor ${floor}` : ""}
                </span>
                <span aria-hidden style={{ height: 1, flex: 1, background: "currentColor", opacity: 0.4 }} />
              </div>
            )}
            {showFloor && (
              <div
                aria-label={`Floor ${floor}`}
                style={{
                  fontSize: 9.5, color: "var(--muted)", letterSpacing: ".04em",
                  margin: "8px 0 2px", paddingTop: 5, borderTop: "1px solid var(--border)",
                }}
              >
                floor {floor}
              </div>
            )}
            {showTurn && (
              <TurnHeader turn={turn as number} chips={chipsByStart.get(i) ?? []} />
            )}
            {/* A div, not a <button>: browsers will not start a text selection
                inside a button, so the step labels could not be copied. It
                keeps the button role, focus and Enter/Space activation, and a
                drag that selects text does not seek. */}
            <div
              ref={isLeaf ? leafRef : undefined}
              role="button"
              tabIndex={0}
              aria-disabled={busy || undefined}
              onClick={(e) => { if (!busy && !endedSelection(e.currentTarget)) onSeek(id); }}
              onKeyDown={(e) => {
                if (busy || (e.key !== "Enter" && e.key !== " ")) return;
                e.preventDefault();
                onSeek(id);
              }}
              aria-current={isLeaf ? "step" : undefined}
              title={isLeaf ? "current position" : isFuture ? "jump forward to this step" : "rewind the run to this step"}
              style={{
                display: "flex", alignItems: "baseline", gap: 6, width: "100%", textAlign: "left",
                border: "none", borderLeft: `2px solid ${isLeaf ? "var(--accent, #6aa0ff)" : "transparent"}`,
                background: isLeaf ? "var(--bg)" : "transparent", borderRadius: "0 4px 4px 0",
                padding: "2px 6px", cursor: busy ? "wait" : "pointer", font: "inherit",
                color: n.error ? "var(--hp)" : isFuture ? "var(--muted)" : "var(--text)",
                fontWeight: isLeaf ? 650 : 400, opacity: isFuture ? 0.85 : 1,
              }}
            >
              {/* The glyph and the "now" tag are chrome: keep them out of a copy. */}
              <span aria-hidden style={{ color: "var(--muted)", fontSize: 10, flexShrink: 0, userSelect: "none" }}>
                {isLeaf ? "▸" : isFuture ? "▹" : "↺"}
              </span>
              <span style={{ flex: 1 }}>{n.error ? `⚠ ${n.label}` : n.label}</span>
              {isLeaf && <span style={{ fontSize: 9.5, color: "var(--muted)", flexShrink: 0, userSelect: "none" }}>now</span>}
            </div>

            <StepEvents rows={n.combat_log ?? []} details={n.details} mode={mode} dim={isFuture} />

            {siblings.length > 0 && (
              <div style={{ paddingLeft: 24, display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center", margin: "1px 0 2px" }}>
                <span style={{ fontSize: 10, color: "var(--muted)" }}>
                  other line{siblings.length > 1 ? "s" : ""}:
                </span>
                {siblings.map((sid) => (
                  <button
                    key={sid}
                    type="button"
                    disabled={busy}
                    onClick={() => onSeek(sid)}
                    title={`switch to this branch: ${byId.get(sid)?.label ?? ""}`}
                    style={{
                      fontSize: 10, padding: "0 7px", border: "1px dashed var(--border)",
                      borderRadius: 10, background: "transparent", cursor: busy ? "wait" : "pointer",
                      color: "var(--muted)", maxWidth: 180, overflow: "hidden",
                      textOverflow: "ellipsis", whiteSpace: "nowrap",
                    }}
                  >
                    ↳ {byId.get(sid)?.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {(hasForks || forward.length > 0) && (
        <div style={{ marginTop: 6, fontSize: 9.5, color: "var(--muted)", lineHeight: 1.4 }}>
          ↺ rewind · ▸ current · ▹ jump forward{hasForks ? " · dashed = an alternate branch you can switch to" : ""}
        </div>
      )}
    </div>
  );
}
