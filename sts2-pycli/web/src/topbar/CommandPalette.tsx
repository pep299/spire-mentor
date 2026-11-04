import { useEffect, useMemo, useRef, useState } from "react";
import { filterEntries, type PaletteEntry } from "./paletteEntries";

/**
 * Ctrl+K / ⌘K: every top-bar command in one filterable list.
 *
 * The menus are the discoverable route; this is the fast one, for the people
 * who use the workbench daily and know the command by name. It is a pure view
 * of the same entries the menus render (see `TopBar.tsx`), so it can never
 * offer something the bar does not.
 */

export default function CommandPalette({ open, entries, onClose }: {
  open: boolean;
  entries: PaletteEntry[];
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const shown = useMemo(() => filterEntries(entries, query), [entries, query]);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setCursor(0);
    requestAnimationFrame(() => input.current?.focus());
  }, [open]);
  useEffect(() => { setCursor(0); }, [query]);

  if (!open) return null;

  function runAt(i: number) {
    const e = shown[i];
    if (!e || e.disabledReason) return;
    onClose();
    e.run();
  }
  function onKey(e: React.KeyboardEvent) {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); onClose(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); setCursor((c) => Math.min(c + 1, shown.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)); }
    else if (e.key === "Enter") { e.preventDefault(); runAt(cursor); }
  }

  let lastGroup = "";
  return (
    <div data-testid="palette-backdrop" className="wb-palette-backdrop" onMouseDown={onClose}>
      <div role="dialog" aria-label="Command palette" className="mono wb-palette"
        onMouseDown={(e) => e.stopPropagation()} onKeyDown={onKey}>
        <input ref={input} type="text" value={query} placeholder="type a command…"
          aria-label="command" role="combobox" aria-expanded="true" aria-controls="wb-palette-list"
          aria-activedescendant={shown[cursor] ? `wb-palette-${shown[cursor].id}` : undefined}
          onChange={(e) => setQuery(e.target.value)} />
        <div id="wb-palette-list" role="listbox" className="wb-palette-list">
          {shown.length === 0 && <div className="wb-palette-empty">no command matches</div>}
          {shown.map((e, i) => {
            const header = e.group !== lastGroup ? e.group : null;
            lastGroup = e.group;
            return (
              <div key={e.id}>
                {header && <div className="wb-palette-group">{header}</div>}
                <div id={`wb-palette-${e.id}`} role="option" aria-selected={i === cursor}
                  aria-disabled={e.disabledReason ? true : undefined}
                  title={e.disabledReason ?? e.title}
                  className={`wb-palette-item${i === cursor ? " wb-palette-item--cursor" : ""}${e.disabledReason ? " wb-palette-item--disabled" : ""}`}
                  onMouseEnter={() => setCursor(i)} onClick={() => runAt(i)}>
                  <span className="wb-palette-label">
                    {e.label}{e.checked !== undefined && <span className="wb-palette-state">{e.checked ? " · on" : " · off"}</span>}
                  </span>
                  {e.disabledReason
                    ? <span className="wb-palette-reason">{e.disabledReason}</span>
                    : e.shortcut && <kbd className="wb-palette-kbd">{e.shortcut}</kbd>}
                </div>
              </div>
            );
          })}
        </div>
        <div className="wb-palette-hint">↑↓ move · Enter run · Esc close</div>
      </div>
    </div>
  );
}
