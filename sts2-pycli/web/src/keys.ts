/** True when the event target is a text-entry control, so global shortcuts
 *  should not hijack the keystroke (e.g. typing a seed). */
export function isTextInput(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName);
}

/** True for a Ctrl / ⌘ / Alt chord. Those belong to the browser and the OS
 *  (Ctrl+A select all, Ctrl+F find, Ctrl+1 switch tab, Alt+← back), so the
 *  single-key shortcuts must let them through: Ctrl+A used to start autoplay
 *  instead of selecting the log. Shift is not a chord (it is how "?" is
 *  typed), and neither is AltGr, which reports Ctrl+Alt on Windows and types
 *  ordinary characters on many layouts. */
export function isBrowserChord(e: KeyboardEvent): boolean {
  if (e.getModifierState?.("AltGraph")) return false;
  return e.ctrlKey || e.metaKey || e.altKey;
}
