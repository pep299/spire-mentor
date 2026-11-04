import { useEffect, useMemo, useState, type ReactNode } from "react";
import { resolveMode, EffectsCtx, type EffectsMode } from "./useEffectsMode";

export function EffectsProvider({ autoplaying, children }: { autoplaying: boolean; children: ReactNode }) {
  const [toggle, setToggle] = useState<EffectsMode>("full");
  const [reducedMotion, setReducedMotion] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!mq) return;
    const on = () => setReducedMotion(mq.matches);
    on();
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, []);
  const mode = resolveMode({ toggle, autoplaying, reducedMotion });
  const value = useMemo(() => ({ mode, toggle, setToggle }), [mode, toggle]);
  return <EffectsCtx.Provider value={value}>{children}</EffectsCtx.Provider>;
}
