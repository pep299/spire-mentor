import { useEffect, useRef, type MutableRefObject } from "react";
import type { Step } from "../api";
import { useSound, type SoundValue } from "./useSound";
import { arrivalCues, goldGainCue, neowArrivalCue, potionGainCue, relicGainCue, runEndCue } from "./appSoundCues";
import { cueForTreasure } from "./runCues";

/**
 * Bridges the sound context down to `AppBody`'s handlers, which sit OUTSIDE
 * `<SoundProvider>` — `AppBody` is the component that renders it, so it
 * can't call `useSound()` itself. `<SoundBridge>` is rendered as a child of
 * the provider and writes the live `sound` value into `soundRef` on every
 * render, the same way `Board.tsx` holds its own ref (see that component's
 * comment): `AppBody`'s card-play / map-select / map-open handlers read
 * `soundRef.current` rather than depending on the context value directly.
 *
 * It also owns the two run-level cues that are state TRANSITIONS rather than
 * a single handler call — gold gained and the run ending — as real effects,
 * for the same reason Board.tsx's combat diff is a real effect: they need to
 * react to `live` changing regardless of which of App's many code paths
 * changed it (a normal play, autobattle-mode autoplay, "play to end", …), not
 * just a handful of named button clicks.
 *
 * Per the sound/emit dependency rule (see Board.tsx): neither effect lists
 * `sound`/`emit` in its dependency array — each reads `soundRef.current`
 * (refreshed unconditionally above, every render) instead, so a volume tick
 * or the sfx map finishing its fetch never re-runs these.
 */
export default function SoundBridge({ soundRef, live, mode }: {
  soundRef: MutableRefObject<SoundValue | null>;
  live: Step | null;
  mode: "replay" | "play" | "mcts";
}) {
  const sound = useSound();
  soundRef.current = sound;

  const prevGoldRef = useRef<number | null>(null);
  useEffect(() => {
    if (mode !== "play" || !live) { prevGoldRef.current = null; return; }
    const gold = live.state.player?.gold;
    if (gold == null) { prevGoldRef.current = null; return; }
    const prev = prevGoldRef.current;
    prevGoldRef.current = gold;
    const s = soundRef.current;
    if (!s?.map) return;
    const cue = goldGainCue(s.map, prev, gold);
    if (cue) s.emit([cue]);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- soundRef is a ref, deliberately excluded (see comment above)
  }, [mode, live]);

  // Every relic entering the inventory, whoever handed it over — the Neow
  // boon, a boss drop, a shop buy, a chest — and the same for potions.
  // Both lists are optional on the wire, so an absent one resets the baseline
  // rather than reading as "lost them all"; see `relicGainCue`.
  const prevRelicsRef = useRef<number | null>(null);
  const prevPotionsRef = useRef<number | null>(null);
  useEffect(() => {
    if (mode !== "play" || !live) {
      prevRelicsRef.current = null;
      prevPotionsRef.current = null;
      return;
    }
    const relics = live.state.player?.relics;
    const potions = live.state.player?.potions;
    const prevRelics = prevRelicsRef.current;
    const prevPotions = prevPotionsRef.current;
    prevRelicsRef.current = relics?.length ?? null;
    prevPotionsRef.current = potions?.length ?? null;
    const s = soundRef.current;
    if (!s?.map) return;
    const cues = [
      relics ? relicGainCue(s.map, prevRelics, relics.length) : null,
      potions ? potionGainCue(s.map, prevPotions, potions.length) : null,
    ].filter((c) => c !== null);
    if (cues.length) s.emit(cues);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- soundRef is a ref, deliberately excluded (see comment above)
  }, [mode, live]);

  // Arriving somewhere the game voices on open — the rewards screen (you won
  // a fight), the start of a fight, the shop, and Neow's room. All are
  // ARRIVALS, so all diff against the previous step rather than reading the
  // current one: a background poll re-delivering the same step must not
  // re-announce the moment. One effect, because they read the same two fields
  // off the same transition.
  const prevDecisionRef = useRef<string | null>(null);
  const prevEventNameRef = useRef<string | null>(null);
  useEffect(() => {
    if (mode !== "play" || !live) {
      prevDecisionRef.current = null;
      prevEventNameRef.current = null;
      return;
    }
    const prevDecision = prevDecisionRef.current;
    const prevEventName = prevEventNameRef.current;
    prevDecisionRef.current = live.decision;
    prevEventNameRef.current = (live.state.event_name ?? "").trim().toLowerCase() || null;
    const s = soundRef.current;
    if (!s?.map) return;
    const cues = [
      ...arrivalCues(s.map, prevDecision, live.decision, {
        holdsPotion: (live.state.player?.potions?.length ?? 0) > 0,
      }),
      neowArrivalCue(s.map, prevEventName, live.decision, live.state.event_name),
    ].filter((c) => c !== null);
    if (cues.length) s.emit(cues);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- soundRef is a ref, deliberately excluded (see comment above)
  }, [mode, live]);

  // Opening a treasure chest. The engine auto-resolves the room, so there is
  // no chest DECISION to hang this on — the reveal rides on the next
  // decision's state (see `RoomReveal`), which is exactly the moment the
  // panel showing it appears. Keyed on the reveal's identity (act + relic)
  // rather than on the decision, since two chests in a row would otherwise
  // read as one.
  const prevRevealRef = useRef<string | null>(null);
  useEffect(() => {
    if (mode !== "play" || !live) { prevRevealRef.current = null; return; }
    const reveal = live.state.room_reveal;
    const id = reveal ? `${reveal.act}/${reveal.relic?.name ?? "-"}/${reveal.gold}` : null;
    const prev = prevRevealRef.current;
    prevRevealRef.current = id;
    if (!reveal || id === prev) return;
    const s = soundRef.current;
    if (!s?.map) return;
    const cue = cueForTreasure(s.map, reveal.act);
    if (cue) s.emit([cue]);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- soundRef is a ref, deliberately excluded (see comment above)
  }, [mode, live]);

  // Latches once per game_over so any later `live` update that still reports
  // decision === "game_over" (e.g. a background poll re-fetching the same
  // step) doesn't replay the cue on every one of them.
  const gameOverCuedRef = useRef(false);
  useEffect(() => {
    if (mode !== "play" || !live || live.decision !== "game_over") {
      gameOverCuedRef.current = false;
      return;
    }
    if (gameOverCuedRef.current) return;
    gameOverCuedRef.current = true;
    const s = soundRef.current;
    if (!s?.map) return;
    const cue = runEndCue(s.map, live.decision, live.state.victory);
    if (cue) s.emit([cue]);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- soundRef is a ref, deliberately excluded (see comment above)
  }, [mode, live]);

  return null;
}
