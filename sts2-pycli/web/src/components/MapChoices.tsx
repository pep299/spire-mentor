import type { Step, ChooseHandler, MapChoiceNode } from "../api";
import { nodeMeta } from "./mapArt";
// For the gsm-* classes below.
import "../skins/map.css";

/**
 * Play-mode map node picker. The live ``map_select`` state exposes only the
 * immediately reachable ``choices`` (no full graph), so this renders them as a
 * row of node buttons. Replay uses the richer connected ``MapView`` instead.
 */
export default function MapChoices({ step, onChoose }: { step: Step; onChoose?: ChooseHandler }) {
  // Sort left-to-right (by column) so the button order matches the map layout.
  const choices: MapChoiceNode[] = [...(step.state.choices ?? [])].sort(
    (a, b) => a.col - b.col || a.row - b.row,
  );
  return (
    <div className="gs-map gsm-choices">
      <h2>Choose your path</h2>
      <div className="gsm-choice-row">
        {choices.map((c, i) => {
          const m = nodeMeta(c.type ?? null);
          return (
            <button key={i} className="gsm-choice" disabled={!onChoose}
              onClick={() => onChoose?.("select_map_node", { col: c.col, row: c.row })}>
              {c.icon
                ? <img src={c.icon} alt={c.type ?? "node"} width={44} height={44}
                    style={{ objectFit: "contain" }} />
                : <span style={{ fontSize: 28, color: m.color }} aria-hidden="true">{m.glyph}</span>}
              <span className="gsm-choice-type">{c.type ?? "Node"}</span>
              <span className="gsm-choice-coord">({c.col},{c.row})</span>
            </button>
          );
        })}
        {choices.length === 0 && <span style={{ color: "var(--muted)" }}>no reachable nodes</span>}
      </div>
    </div>
  );
}
