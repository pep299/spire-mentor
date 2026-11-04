import { useState } from "react";
import { isStaticMode } from "../transport";
import { runBench } from "./benchRun";

export function BenchButton({ inCombat }: { inCombat: boolean }) {
  const [busy, setBusy] = useState(false);
  if (!isStaticMode() || !inCombat) return null;
  const run = async () => {
    setBusy(true);
    try {
      const report = await runBench();
      try {
        await navigator.clipboard?.writeText(report);
      } catch {
        /* clipboard needs a secure context + gesture; the alert still shows it */
      }
      window.alert(`${report}\n\n(copied to the clipboard)`);
    } finally {
      setBusy(false);
    }
  };
  return (
    <button onClick={run} disabled={busy}
      title="time the combat search at the advisor presets (200/800/1000/4000 sims) on THIS device and copy the numbers">
      {busy ? "⏱ …" : "⏱ bench"}
    </button>
  );
}
