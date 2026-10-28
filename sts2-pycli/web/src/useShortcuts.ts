import { useEffect, useRef } from "react";
import type { Step } from "./api";
import { isTextInput } from "./keys";

export interface ShortcutContext {
  mode: "replay" | "play" | "mcts";
  live: Step | null;
  step: Step | null;
  count: number;
  idx: number;
  onScrub: (delta: number | "home" | "end") => void;
  onRecommend: () => void;
  onChoose: (action: string, args: Record<string, unknown>) => void;
  onPlayBest: () => void;
  onPlayToEnd: () => void;
  /** P: peek at the AI's pick for this decision (Coach style). */
  onPeek?: () => void;
  autoRunning: boolean;
  onStart: () => void;
  onToggleHelp: () => void;
}

/** Global keyboard shortcuts for replay scrubbing, play choice screens, the
 *  new-run form, and the help overlay. Play-combat keys are owned by Board, so
 *  this hook stays inert during `combat_play`. */
export function useShortcuts(ctx: ShortcutContext): void {
  const ref = useRef(ctx);
  ref.current = ctx;

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const c = ref.current;
      if (e.key === "?") { e.preventDefault(); c.onToggleHelp(); return; }
      // Enter starts a run even from a focused form field.
      if (e.key === "Enter" && c.mode === "play" && !c.live) { e.preventDefault(); c.onStart(); return; }
      if (isTextInput(e.target)) return;

      if (c.mode === "replay") {
        if (e.key === "ArrowRight") { e.preventDefault(); c.onScrub(1); }
        else if (e.key === "ArrowLeft") { e.preventDefault(); c.onScrub(-1); }
        else if (e.key === "Home") { e.preventDefault(); c.onScrub("home"); }
        else if (e.key === "End") { e.preventDefault(); c.onScrub("end"); }
        else if ((e.key === "h" || e.key === "H") && c.step?.decision === "combat_play") { e.preventDefault(); c.onRecommend(); }
        return;
      }

      // play mode
      if (!c.live) return;
      // A loop is running (autoplay / play-to-end): swallow keys so a stray
      // press can't start a second action. Esc (a global listener) interrupts.
      if (c.autoRunning) return;
      if (c.live.decision === "combat_play" || c.live.decision === "game_over") return; // Board owns combat
      // F plays the whole run out to a terminal state — map, rewards, fights and
      // all — from any non-combat screen (it's also wired in Board for combat).
      if (e.key === "f" || e.key === "F") { e.preventDefault(); c.onPlayToEnd(); return; }
      // G / A take the run-level policy's best option for this non-combat
      // decision (card reward, event, path, …) — the off-combat twin of combat's
      // play-best (G) / autoplay (A). On a single screen both reduce to one pick.
      if (e.key === "a" || e.key === "A" || e.key === "g" || e.key === "G") { e.preventDefault(); c.onPlayBest(); return; }
      if (e.key === "p" || e.key === "P") { e.preventDefault(); c.onPeek?.(); return; }
      const n = Number(e.key);
      if (!Number.isInteger(n) || n < 1 || n > 9) return;
      const enabled = (c.live.state.options ?? []).filter((o) => o.enabled);
      const o = enabled[n - 1];
      if (o) { e.preventDefault(); c.onChoose(o.action, o.args); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
