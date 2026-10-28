import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

/**
 * The advice STYLE of live play: **Hints** or **Coach** (the default).
 *
 * - **Hints** shows the machine's opinion WITHOUT being asked.
 * - **Coach** shows nothing unprompted before you act. The press-first tools
 *   still work, and peeking (P) reveals only the AI's next action. After you
 *   act, the coach compares your turn or choice with the AI's.
 *
 * After-the-fact feedback runs in BOTH styles. `on` (`style === "hints"`)
 * is what every gated surface reads.
 *
 * The workbench grew a lot of advice surfaces (the PPO policy panel, MCTS
 * combat hints, the EV analyzers on card rewards / shops / events, synergy and
 * archetype chips, the map's planned route). Every one of them used to render
 * on its own condition, so a human sitting down to play was told the answer
 * before deciding — and autobattle mode, whose whole point is to withhold the
 * champion's pick until after you commit, leaked it through the policy panel
 * sitting right under the options.
 *
 * **What `on` governs is UNPROMPTED advice, not access to the tools.**
 * The line is whether a surface can tell you something you did not ask for:
 *
 * - GATED (renders an opinion on its own): the PPO policy panel, the auto
 *   combat hint and its per-card win-prob overlays, the card-reward / shop /
 *   event EV analyzers, the synergy + archetype chips, and the map's green
 *   planned-route overlay — a route drawn across the map is the AI answering
 *   the map decision for you.
 * - UNGATED (press-first — nothing appears until you click): the hint button
 *   and H, peek (P), play-best (G), autoplay (A) and play-to-end (F), the
 *   search-tree inspector (S), the regret review and run review docks, the
 *   Coach review, and the deck analyzer. Reaching for one of these IS the
 *   deliberate act the style exists to require, so making you switch style
 *   first only costs a click. In Coach, a press-first reveal of the AI's pick
 *   for the current decision marks that decision peeked.
 *
 * Coach = play the game as a human: cards, prices, HP, deck, map, and nothing
 * that evaluates your options until you ask it to.
 *
 * Scope: **live play only.** Replay is for reviewing a recorded run, so its
 * surfaces stay visible — there is no drill to spoil there.
 */

export type AdviceStyle = "hints" | "coach";

const STORAGE_KEY = "adviceStyle";
/** The old boolean switch: `"1"` (hints on) migrates to Hints, anything else
 *  (including absent) to Coach. */
const LEGACY_KEY = "hintsOn";

interface HintsValue {
  style: AdviceStyle;
  setStyle: (s: AdviceStyle) => void;
  /** True when unprompted AI advice may be displayed (`style === "hints"`). */
  on: boolean;
  setOn: (v: boolean) => void;
  /** Switch between Hints and Coach. */
  toggle: () => void;
}

// The default is Coach, so a component rendered without a provider (a unit
// test, a stray subtree) hides advice rather than leaking it.
const Ctx = createContext<HintsValue>({
  style: "coach", setStyle: () => {}, on: false, setOn: () => {}, toggle: () => {},
});

export function useHints(): HintsValue {
  return useContext(Ctx);
}

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
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
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
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
