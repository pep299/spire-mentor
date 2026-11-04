// The "Analyze options (EV)" advisor's state hook (see DecisionAdvisor.tsx) —
// kept out of the component file so it exports only components (react-
// refresh/only-export-components).
import { useEffect, useRef, useState } from "react";
import {
  api,
  type DecisionAdvice,
  type DecisionAdvisorCandidate,
  type DecisionAdvisorProgress,
} from "../api";
import { useHints } from "../hints/useHints";

/** Live streaming progress: total/done playouts plus per-option completed counts. */
export type DecProgress = {
  done: number;
  total: number;
  rollouts: number;
  perCand: Map<string, number>;
  cands: { key: string; label: string }[];
};

export interface DecisionAdvisorState {
  enabled: boolean;
  advising: boolean;
  advice: DecisionAdvice | null;
  prog: DecProgress | null;
  elapsed: number;
  byKey: Map<string, DecisionAdvisorCandidate>;
  /** True when at least one option cleared the act in ≥1 rollout. Act-win is a
   *  coarse, often-all-zero signal (the act boss is a wall + a deliberately
   *  cheap rollout policy), so the chips lead with it only when it actually
   *  discriminates; otherwise they fall back to the depth/HP signal the ranking
   *  already uses. */
  anyWin: boolean;
  run: () => void;
}

/** Run the streaming decision advisor for the current screen. ``resetKey``
 *  (e.g. the step index) invalidates prior advice when the screen changes. */
export function useDecisionAdvisor(resetKey: unknown, enabled: boolean): DecisionAdvisorState {
  // The advisor ranks the options in front of you, so it is advice like any
  // other: available in live play only when the global hints switch is on.
  const hints = useHints();
  const active = enabled && hints.on;
  const [advice, setAdvice] = useState<DecisionAdvice | null>(null);
  const [advising, setAdvising] = useState(false);
  const [prog, setProg] = useState<DecProgress | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(0);

  // A fresh screen invalidates any prior advice.
  // Turning hints off mid-screen must also drop advice already on screen.
  useEffect(() => { setAdvice(null); setAdvising(false); setProg(null); setElapsed(0); }, [resetKey, active]);
  // Tick the elapsed-seconds readout while running.
  useEffect(() => {
    if (!advising) return;
    const id = setInterval(() => setElapsed((Date.now() - startRef.current) / 1000), 100);
    return () => clearInterval(id);
  }, [advising]);

  const byKey = new Map<string, DecisionAdvisorCandidate>(
    (advice?.candidates ?? []).map((c) => [c.key, c]),
  );
  const anyWin = (advice?.candidates ?? []).some((c) => c.win_rate > 0);

  async function run() {
    if (!active) return; // belt and braces: no advice without the switch
    startRef.current = Date.now();
    setElapsed(0);
    setProg(null);
    setAdvice(null);
    setAdvising(true);
    try {
      const result = await api.live.decisionAdvisorStream({}, (ev: DecisionAdvisorProgress) => {
        if (ev.type === "start") {
          setProg({
            done: 0,
            total: ev.total,
            rollouts: ev.rollouts,
            perCand: new Map(ev.candidates.map((c) => [c.key, 0])),
            cands: ev.candidates.map((c) => ({ key: c.key, label: c.label })),
          });
        } else {
          setProg((p) => {
            if (!p) return p;
            const perCand = new Map(p.perCand);
            perCand.set(ev.candidate_key, (perCand.get(ev.candidate_key) ?? 0) + 1);
            return { ...p, done: ev.done, perCand };
          });
        }
      });
      setAdvice(result);
    } catch (e) {
      setAdvice({ candidates: [], error: e instanceof Error ? e.message : String(e) });
    } finally {
      setAdvising(false);
      setProg(null);
    }
  }

  return { enabled: active, advising, advice, prog, elapsed, byKey, anyWin, run };
}
