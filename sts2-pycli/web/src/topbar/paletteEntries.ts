// The command palette's entries, filter and Ctrl+K chord — kept out of the
// component file so it exports only components (react-refresh/only-export-
// components).
import { useEffect } from "react";

export interface PaletteEntry {
  id: string;
  label: string;
  /** Section header in the list ("Run", "Analyze", "View", "Tools", "AI"). */
  group: string;
  title?: string;
  shortcut?: string;
  disabledReason?: string;
  /** For toggles: current state, shown after the label. */
  checked?: boolean;
  run: () => void;
}

export function filterEntries(entries: PaletteEntry[], query: string): PaletteEntry[] {
  const q = query.trim().toLowerCase();
  if (!q) return entries;
  const words = q.split(/\s+/);
  return entries.filter((e) => {
    const hay = `${e.group} ${e.label} ${e.title ?? ""}`.toLowerCase();
    return words.every((w) => hay.includes(w));
  });
}

/** True for the keystroke that opens the palette. Exported so the global
 *  listener and the help overlay describe the same chord. */
export function isPaletteChord(e: KeyboardEvent): boolean {
  return (e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && (e.key === "k" || e.key === "K");
}

/** Global hook: Ctrl+K / ⌘K toggles the palette from anywhere, even from a
 *  text field (the chord is unambiguous; the browser's own Ctrl+K is the
 *  address-bar search, which the workbench has no use for). */
export function usePaletteChord(toggle: () => void): void {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!isPaletteChord(e)) return;
      e.preventDefault();
      toggle();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);
}
