import { useEffect, useMemo, useState, type ReactNode } from "react";
import { HintsCtx, type AdviceStyle, type HintsValue } from "./useHints";

// Supplies the advice style (Hints / Coach) — what it means and the hook that
// reads it live in useHints.ts.

const STORAGE_KEY = "adviceStyle";
/** The old boolean switch: `"1"` (hints on) migrates to Hints, anything else
 *  (including absent) to Coach. */
const LEGACY_KEY = "hintsOn";

function readStored(): AdviceStyle {
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    if (s === "hints" || s === "coach") return s;
    return localStorage.getItem(LEGACY_KEY) === "1" ? "hints" : "coach";
  } catch {
    return "coach"; // private mode / storage disabled — default to hidden
  }
}

function valueOf(style: AdviceStyle, setStyle: (s: AdviceStyle) => void): HintsValue {
  return {
    style,
    setStyle,
    on: style === "hints",
    setOn: (v) => setStyle(v ? "hints" : "coach"),
    toggle: () => setStyle(style === "hints" ? "coach" : "hints"),
  };
}

export function HintsProvider({ children }: { children: ReactNode }) {
  const [style, setStyle] = useState<AdviceStyle>(readStored);
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, style);
    } catch {
      /* storage unavailable — the style still works for this session */
    }
  }, [style]);
  const value = useMemo(() => valueOf(style, setStyle), [style]);
  return <HintsCtx.Provider value={value}>{children}</HintsCtx.Provider>;
}

/** Test/story helper: force a style (or `on` = Hints) without touching storage. */
export function HintsFixture({
  on,
  style,
  children,
}: {
  on?: boolean;
  style?: AdviceStyle;
  children: ReactNode;
}) {
  const s: AdviceStyle = style ?? (on ? "hints" : "coach");
  const value = useMemo(() => valueOf(s, () => {}), [s]);
  return <HintsCtx.Provider value={value}>{children}</HintsCtx.Provider>;
}
