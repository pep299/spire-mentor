import type { TimelineEntry } from "../api";
import { hasReadableRow } from "../combatlog";
import { endedSelection } from "../textSelection";
import { DetailToggle, StepEvents, TurnHeader } from "./turnLedger";
import { turnChips, useDetailMode } from "./turnLedgerState";

/**
 * The loaded-run history: one clickable row per recorded decision.
 *
 * Combat stretches render as a TURN LEDGER (shared with LiveLog): steps group
 * under "TURN N" headers summarized by chips, and each step shows its
 * action's combat-log slice as coloured sentences when the recording carried
 * one — falling back to the state-diff detail lines (also coloured) for runs
 * recorded without the engine log. An entry's `turn` is its pre-action state's
 * round, i.e. already the round the action was taken in.
 */

function headerFor(e: TimelineEntry): string {
  const act = e.act_name ?? (e.act != null ? `Act ${e.act}` : "?");
  return e.floor != null ? `${act} · floor ${e.floor}` : act;
}

export default function EventLog({ entries, current, onPick }: {
  entries: TimelineEntry[];
  current: number;
  onPick: (idx: number) => void;
}) {
  const [mode, setMode] = useDetailMode();
  // `hasReadableRow`, not the raw slice — see the same line in LiveLog.
  const anyRows = entries.some((e) => hasReadableRow(e.combat_log ?? []));

  // Chips aggregate over each turn group (consecutive same-turn entries).
  const chipsByStart = new Map<number, string[]>();
  {
    let start = -1;
    let lastTurn: number | null = null;
    let rowsAcc: TimelineEntry["combat_log"] = [];
    let detAcc: string[] = [];
    const flush = () => {
      if (start >= 0 && lastTurn != null) {
        chipsByStart.set(start, turnChips(rowsAcc ?? [], detAcc));
      }
    };
    entries.forEach((e, i) => {
      if (e.turn !== lastTurn) {
        flush();
        start = i;
        lastTurn = e.turn;
        rowsAcc = [];
        detAcc = [];
      }
      rowsAcc = [...(rowsAcc ?? []), ...(e.combat_log ?? [])];
      detAcc = [...detAcc, ...e.details];
    });
    flush();
  }

  let lastHeader = "";
  let lastTurn: number | null = null;
  return (
    <div>
      <div className="mono" style={{ display: "flex", justifyContent: "space-between",
        alignItems: "baseline", fontSize: 10, color: "var(--muted)",
        borderBottom: "1px solid var(--border)", paddingBottom: 5, margin: "18px 0 9px" }}>
        <span>Event log</span>
        {anyRows && <DetailToggle mode={mode} onChange={setMode} />}
      </div>
      {entries.length === 0 && (
        <div className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>load a run to see its history</div>
      )}
      {entries.map((e, i) => {
        const header = headerFor(e);
        const showHeader = header !== lastHeader;
        lastHeader = header;
        const showTurn = e.turn != null && e.turn !== lastTurn;
        lastTurn = e.turn;
        const isCurrent = e.idx === current;
        return (
          <div key={e.idx}>
            {showHeader && (
              <div className="mono" style={{ fontSize: 9.5, color: "var(--muted)", letterSpacing: ".02em",
                margin: "11px 0 3px", borderTop: "1px solid var(--border)", paddingTop: 6 }}>
                {header}
              </div>
            )}
            {showTurn && (
              <TurnHeader turn={e.turn as number} chips={chipsByStart.get(i) ?? []} />
            )}
            <div className="mono" onClick={(ev) => { if (!endedSelection(ev.currentTarget)) onPick(e.idx); }}
              style={{ display: "flex", gap: 7, fontSize: 11.5, lineHeight: 1.65, cursor: "pointer",
                padding: "1px 4px", borderRadius: 4,
                background: isCurrent ? "var(--accent-weak, var(--hp-track))" : "transparent",
                color: isCurrent ? "var(--accent-strong)" : "var(--text)",
                fontWeight: isCurrent ? 600 : 400 }}>
              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {e.label || e.decision}
              </span>
            </div>
            <div className="mono">
              <StepEvents rows={e.combat_log ?? []} details={e.details} mode={mode} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
