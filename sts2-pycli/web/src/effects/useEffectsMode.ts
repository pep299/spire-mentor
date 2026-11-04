// The effects mode (full / reduced / off): its type, how it resolves, and the
// hook that reads it. EffectsProvider (EffectsContext.tsx) supplies the value.
import { createContext, useContext } from "react";

export type EffectsMode = "full" | "reduced" | "off";

export function resolveMode(opts: { toggle: EffectsMode; autoplaying: boolean; reducedMotion: boolean }): EffectsMode {
  if (opts.toggle === "off") return "off";
  // Autoplay/fast-step: keep attack/hurt/death cues playing but calm the big
  // motion (screen shake) — the loop is paced by the per-action server round
  // trip, so the one-shots are watchable. Set the fx toggle to "off" for a
  // truly silent fast-step.
  if (opts.autoplaying) return "reduced";
  if (opts.reducedMotion) return "reduced"; // still show informative cues; motion is calmed
  return opts.toggle;
}

export interface EffectsValue {
  mode: EffectsMode;
  toggle: EffectsMode;
  setToggle: (m: EffectsMode) => void;
}

export const EffectsCtx = createContext<EffectsValue>({ mode: "full", toggle: "full", setToggle: () => {} });

export function useEffectsMode(): EffectsValue {
  return useContext(EffectsCtx);
}
