import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useEffectsMode } from "../effects/EffectsContext";
import { loadSfxMap, type SfxMap } from "./sfxMap";
import { SoundPlayer } from "./player";
import { createWebAudioBackend } from "./webAudioBackend";
import { filterForMode } from "./triggers";
import type { SoundCue } from "./types";

/**
 * Game sound effects — **on by default** (user decision 2026-09-09; the
 * hints switch stays opt-in). A stored "0" written by a human toggling the
 * switch off is the only way it starts off — a "0" left behind by the
 * pre-#716 default-off build is retired on first load (see CHOICE_KEY).
 *
 * The AudioContext therefore usually comes to life at page load, with no
 * user gesture behind it, and browsers hand it back suspended: the backend
 * (`webAudioBackend.ts`) resumes it on the first pointer / key gesture and
 * again from any play that finds it still suspended. Nothing plays before
 * that gesture, which is also the earliest a fight could start.
 *
 * The effective loudness of the *set* is `useEffectsMode().mode`: `reduced`
 * (autoplay, play-to-end, fast-step, prefers-reduced-motion) drops chatter and
 * keeps hit / block / death, exactly as the visual layer does.
 */
const STORAGE_KEY = "soundOn";
const VOLUME_KEY = "soundVolume";
/**
 * Marks a browser whose stored `soundOn` reflects an actual human choice.
 *
 * Builds before #716 defaulted sound OFF *and* wrote that default back to
 * storage on every mount, so every browser that ever opened the workbench
 * carries a `soundOn="0"` nobody ever chose — and "on by default" reached
 * exactly nobody. The first load after this key lands retires that legacy
 * value once; from then on `soundOn` is a real choice and is obeyed.
 */
const CHOICE_KEY = "soundChoiceV2";

export interface SoundValue {
  on: boolean;
  volume: number;
  setOn: (v: boolean) => void;
  setVolume: (v: number) => void;
  /** Play a derived cue list, mode-filtered. No-op while sound is off. */
  emit: (cues: SoundCue[]) => void;
  map: SfxMap | null;
}

const Ctx = createContext<SoundValue>({
  on: false, volume: 0.7, setOn: () => {}, setVolume: () => {}, emit: () => {}, map: null,
});

export function useSound(): SoundValue {
  return useContext(Ctx);
}

function readStored(): boolean {
  // Absent (a first visit) and "1" both mean on; only an explicit "0" is off.
  try {
    if (localStorage.getItem(CHOICE_KEY) === null) {
      // Legacy storage: drop the pre-#716 "0" (see CHOICE_KEY) and start on.
      // Idempotent, so React's double-invoked initializer is harmless.
      localStorage.setItem(CHOICE_KEY, "1");
      localStorage.removeItem(STORAGE_KEY);
      return true;
    }
    return localStorage.getItem(STORAGE_KEY) !== "0";
  } catch { return true; }
}

function readVolume(): number {
  try {
    const v = Number(localStorage.getItem(VOLUME_KEY));
    return Number.isFinite(v) && v > 0 && v <= 1 ? v : 0.7;
  } catch { return 0.7; }
}

/** `gameVersion` is the build the shown run plays (GH #877): its own
 *  `sfx_map.json` is loaded, and a change of build reloads the map. */
export function SoundProvider({ children, gameVersion }: {
  children: ReactNode;
  gameVersion?: string | null;
}) {
  const [on, setOn] = useState<boolean>(readStored);
  const [volume, setVolume] = useState<number>(readVolume);
  const [map, setMap] = useState<SfxMap | null>(null);
  // The build `map` was loaded for; a different build drops it so the
  // effect below fetches that build's map.
  const [mapVersion, setMapVersion] = useState<string | null>(gameVersion ?? null);
  if ((gameVersion ?? null) !== mapVersion) {
    setMapVersion(gameVersion ?? null);
    setMap(null);
  }
  const playerRef = useRef<SoundPlayer | null>(null);
  const { mode } = useEffectsMode();

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, on ? "1" : "0"); } catch { /* storage off */ }
    if (!on) return;
    // First enable (at mount, usually): construct the backend. When this runs
    // from the toggle click, that click is the gesture the context needs; at
    // mount it is not, and the backend waits for one.
    if (!playerRef.current) {
      try {
        playerRef.current = new SoundPlayer(createWebAudioBackend());
      } catch (err) {
        // `new AudioContext()` can throw -- an environment already at its
        // concurrent-AudioContext cap is the realistic case. Uncaught here it
        // would escape the effect into React with no error boundary in
        // sight, so a first click on the toggle could take down the whole
        // workbench tree. Degrade instead: warn, leave the player unset (the
        // assignment above never completed), and flip the switch back off so
        // it never reads "on" while no audio can ever play.
        console.warn("[sound] failed to construct the audio backend — sound will stay off", err);
        setOn(false);
        return;
      }
    }
    if (!map) {
      // A failed load does NOT auto-retry here: `setMap(null)` when `map` is
      // already `null` is a same-value update, so React bails and this
      // effect does not re-fire on its own. That's deliberate, not a wedge --
      // `loadSfxMap` (Task 5) clears its cached promise on rejection, so
      // toggling sound off and back on retries cleanly.
      loadSfxMap(fetch, mapVersion)
        .then(setMap)
        .catch((err: unknown) => {
          // An audio feature's only symptom of breakage IS silence — there is
          // no visual tell. Swallowing this would leave the switch reading
          // "on" with nothing to show for it and nothing in devtools either.
          console.warn("[sound] failed to load sfx_map.json — sound will stay silent", err);
          setMap(null);
        });
    }
  }, [on, map, mapVersion]);

  useEffect(() => {
    try { localStorage.setItem(VOLUME_KEY, String(volume)); } catch { /* storage off */ }
  }, [volume]);

  const emit = useCallback((cues: SoundCue[]) => {
    const player = playerRef.current;
    if (!on || !player) return;
    const now = performance.now();
    for (const cue of filterForMode(cues, mode)) void player.play(cue, volume, now);
  }, [on, mode, volume]);

  const value = useMemo(
    () => ({ on, volume, setOn, setVolume, emit, map }),
    [on, volume, emit, map],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

/**
 * Test/story helper: force a value without touching storage or audio.
 *
 * `emit` and `map` default to the inert no-op/`null` pair — a caller that
 * only needs `on` (most of them) gets the old zero-config behaviour. A test
 * that needs to OBSERVE emitted cues passes its own `vi.fn()` for `emit` and
 * a real `map` (Board's effect only calls `sound.emit` when `sound.map` is
 * truthy), rather than reaching for `Ctx.Provider` directly — `Ctx` itself
 * stays unexported so this fixture remains the one sanctioned way in.
 */
export function SoundFixture({ on, emit, map, children }: {
  on: boolean;
  emit?: (cues: SoundCue[]) => void;
  map?: SfxMap | null;
  children: ReactNode;
}) {
  const value = useMemo(
    () => ({ on, volume: 0.7, setOn: () => {}, setVolume: () => {}, emit: emit ?? (() => {}), map: map ?? null }),
    [on, emit, map],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
