/**
 * True when the click that just landed on `el` ended a text selection inside
 * it: the user dragged to select (to copy) rather than clicked.
 *
 * The history logs make whole rows clickable (rewind / scrub to that step).
 * A drag that starts and ends inside one row still fires `click` on mouseup,
 * so without this check selecting a log line to copy it rewound the run. A
 * plain click collapses any earlier selection on mousedown, so it never
 * reads as one.
 */
export function endedSelection(el: Element): boolean {
  const sel = globalThis.getSelection?.();
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return false;
  return sel.getRangeAt(0).intersectsNode(el);
}
