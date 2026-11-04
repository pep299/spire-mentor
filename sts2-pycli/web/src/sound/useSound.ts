// The sound context's value, context and hook. SoundProvider (SoundContext.tsx)
// supplies the value.
import { createContext, useContext } from "react";
import type { SfxMap } from "./sfxMap";
import type { SoundCue } from "./types";

export interface SoundValue {
  on: boolean;
  volume: number;
  setOn: (v: boolean) => void;
  setVolume: (v: number) => void;
  /** Play a derived cue list, mode-filtered. No-op while sound is off. */
  emit: (cues: SoundCue[]) => void;
  map: SfxMap | null;
}

export const SoundCtx = createContext<SoundValue>({
  on: false, volume: 0.7, setOn: () => {}, setVolume: () => {}, emit: () => {}, map: null,
});

export function useSound(): SoundValue {
  return useContext(SoundCtx);
}
