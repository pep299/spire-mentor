import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { SoundProvider } from "../SoundContext";
import { useSound } from "../useSound";
import { SoundPlayer } from "../player";
import { EffectsProvider } from "../../effects/EffectsContext";
import * as sfxMap from "../sfxMap";

function Probe() {
  const { on, setOn, emit } = useSound();
  return (
    <div>
      <button onClick={() => setOn(!on)}>{on ? "on" : "off"}</button>
      <button
        onClick={() =>
          emit([
            { event: "event:/sfx/chatter1", cls: "chatter" },
            { event: "event:/sfx/info1", cls: "informative" },
          ])
        }
      >
        emit
      </button>
    </div>
  );
}

function toggleBtn() {
  return screen.getByText(/^(on|off)$/);
}

// Volume-specific probe: separate from Probe above so the volume tests don't
// have to thread setVolume/volume through every existing test's markup.
function VolumeProbe() {
  const { on, setOn, volume, setVolume, emit } = useSound();
  return (
    <div>
      <button onClick={() => setOn(!on)}>{on ? "on" : "off"}</button>
      <span data-testid="volume">{volume}</span>
      <button onClick={() => setVolume(0.3)}>set-volume</button>
      <button onClick={() => emit([{ event: "event:/sfx/info1", cls: "informative" }])}>emit</button>
    </div>
  );
}

/** jsdom here exposes a `localStorage` object with no working methods
 *  (`typeof globalThis.localStorage === "object"`, but `getItem`/`clear`
 *  are `undefined`), so the suite supplies its own in-memory store -- same
 *  mechanism hints/__tests__/hintsContext.test.tsx uses. */
function memoryStorage(seed: Record<string, string> = {}) {
  const map = new Map(Object.entries(seed));
  return {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    clear: () => map.clear(),
  };
}

describe("SoundProvider", () => {
  let originalAudioContext: typeof AudioContext | undefined;

  beforeEach(() => {
    // Seeded OFF: sound is on by default (see "defaults to on" below), and
    // every test here that clicks the toggle once wants to be turning it ON.
    // The soundChoiceV2 marker says that "0" is a human's choice, not the
    // pre-#716 default (which is retired on sight -- see the migration test).
    vi.stubGlobal("localStorage", memoryStorage({ soundOn: "0", soundChoiceV2: "1" }));
    originalAudioContext = globalThis.AudioContext;
    // jsdom has no AudioContext; a minimal stub so createWebAudioBackend()
    // (called on first enable) does not throw. Tests that care about
    // construction *counting* install their own counting stub below.
    // @ts-expect-error jsdom has no AudioContext
    globalThis.AudioContext = class {};
  });

  afterEach(() => {
    // Restore what beforeEach (and any test-local override) installed.
    // Defect fixed here vs. the brief's reference test: it assigned
    // globalThis.AudioContext and never restored it, leaking a stub into
    // every later test in the same worker.
    globalThis.AudioContext = originalAudioContext as typeof AudioContext;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("defaults to on; only a stored \"0\" starts it off", () => {
    vi.stubGlobal("localStorage", memoryStorage());
    const { unmount } = render(
      <SoundProvider>
        <Probe />
      </SoundProvider>,
    );
    expect(toggleBtn()).toHaveTextContent("on");
    unmount();
    vi.stubGlobal("localStorage", memoryStorage({ soundOn: "0", soundChoiceV2: "1" }));
    render(
      <SoundProvider>
        <Probe />
      </SoundProvider>,
    );
    expect(toggleBtn()).toHaveTextContent("off");
  });

  it("retires a pre-#716 stored \"0\" once, then obeys the human's choice", () => {
    // Builds before #716 defaulted OFF *and* persisted that default on every
    // mount, so every returning browser carries a "0" nobody chose and the
    // new default reached nobody. Mutation caught: dropping the CHOICE_KEY
    // branch from readStored -- this first render would come up "off".
    vi.stubGlobal("localStorage", memoryStorage({ soundOn: "0" })); // legacy: no marker
    const { unmount } = render(
      <SoundProvider>
        <Probe />
      </SoundProvider>,
    );
    expect(toggleBtn()).toHaveTextContent("on");
    expect(localStorage.getItem("soundChoiceV2")).toBe("1");
    unmount();

    // ... and the retirement is one-shot: a deliberate off now sticks.
    localStorage.setItem("soundOn", "0");
    render(
      <SoundProvider>
        <Probe />
      </SoundProvider>,
    );
    expect(toggleBtn()).toHaveTextContent("off");
  });

  it("persists the choice", async () => {
    const { unmount } = render(
      <SoundProvider>
        <Probe />
      </SoundProvider>,
    );
    await act(async () => {
      toggleBtn().click();
    });
    expect(localStorage.getItem("soundOn")).toBe("1");
    unmount();
    render(
      <SoundProvider>
        <Probe />
      </SoundProvider>,
    );
    expect(toggleBtn()).toHaveTextContent("on");
  });

  it("does not construct an AudioContext until sound is switched on", async () => {
    let constructed = 0;
    // @ts-expect-error jsdom has no AudioContext; count constructions.
    globalThis.AudioContext = class {
      constructor() {
        constructed += 1;
      }
    };
    render(
      <SoundProvider>
        <Probe />
      </SoundProvider>,
    );
    expect(constructed).toBe(0);
    await act(async () => {
      toggleBtn().click();
    });
    expect(constructed).toBe(1);
  });

  it("warns and stays silent when sfx_map.json fails to load, instead of failing silently", async () => {
    // Defect fixed here vs. the brief's reference code: it swallowed the
    // rejection whole (`.catch(() => setMap(null))`) -- the switch reads
    // "on", nothing plays, and no signal reaches devtools. An audio
    // feature's only symptom of breakage IS silence.
    // Mutation caught: deleting the console.warn call (or the whole .catch
    // handler) from SoundContext's loadSfxMap().then/.catch chain.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const err = new Error("network blip");
    vi.spyOn(sfxMap, "loadSfxMap").mockRejectedValueOnce(err);

    render(
      <SoundProvider>
        <Probe />
      </SoundProvider>,
    );
    await act(async () => {
      toggleBtn().click();
    });

    expect(warn).toHaveBeenCalled();
    const call = warn.mock.calls.find((c) => String(c[0]).includes("sfx_map"));
    expect(call).toBeDefined();
    expect(call?.[1]).toBe(err);
  });

  it("emit is a no-op while sound is off", async () => {
    // The user's explicit scope decision -- mirror the existing fx modes
    // rather than adding a separate audio switch -- is implemented by the
    // filterForMode(cues, mode) call inside emit; this test and the next
    // one are what makes that line load-bearing.
    // Mutation caught: removing the `if (!on) return` guard in emit -- the
    // player, once constructed by an earlier enable, would keep playing
    // even after the human switches sound back off.
    const play = vi.spyOn(SoundPlayer.prototype, "play").mockResolvedValue(undefined);
    render(
      <SoundProvider>
        <Probe />
      </SoundProvider>,
    );
    await act(async () => {
      toggleBtn().click(); // on: constructs the player
    });
    await act(async () => {
      toggleBtn().click(); // off again
    });
    await act(async () => {
      screen.getByText("emit").click();
    });
    expect(play).not.toHaveBeenCalled();
  });

  it("drops a chatter cue but keeps an informative one when the fx mode is reduced", async () => {
    // Mutation caught: dropping the filterForMode(cues, mode) call from
    // emit -- both cues would reach the player regardless of mode.
    const play = vi.spyOn(SoundPlayer.prototype, "play").mockResolvedValue(undefined);
    render(
      <EffectsProvider autoplaying>
        <SoundProvider>
          <Probe />
        </SoundProvider>
      </EffectsProvider>,
    );
    await act(async () => {
      toggleBtn().click();
    });
    await act(async () => {
      screen.getByText("emit").click();
    });
    const events = play.mock.calls.map((c) => (c[0] as { event: string }).event);
    expect(events).toContain("event:/sfx/info1");
    expect(events).not.toContain("event:/sfx/chatter1");
  });

  it("degrades to off when the audio backend fails to construct, instead of crashing the tree", async () => {
    // `new AudioContext()` can throw (e.g. an environment already at its
    // concurrent-AudioContext cap). Uncaught, that would escape the effect
    // into React with no error boundary in sight -- the first click on the
    // toggle taking down the whole workbench tree. If that happens here,
    // `act()` re-throws and this test fails with an uncaught exception; the
    // assertions below only run because it didn't.
    // Mutation caught: removing the try/catch around
    // `new SoundPlayer(createWebAudioBackend())` in the effect.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    // @ts-expect-error jsdom has no AudioContext; simulate construction failure.
    globalThis.AudioContext = class {
      constructor() {
        throw new Error("AudioContext cap reached");
      }
    };
    render(
      <SoundProvider>
        <Probe />
      </SoundProvider>,
    );
    await act(async () => {
      toggleBtn().click();
    });
    expect(toggleBtn()).toHaveTextContent("off");
    expect(warn).toHaveBeenCalled();
    const call = warn.mock.calls.find((c) => String(c[0]).includes("audio backend"));
    expect(call).toBeDefined();
  });

  // Final review finding 7 (2026-08-23): the volume control had zero test
  // coverage -- three mutations each left all 549 tests green (hardcoding
  // `player.play(cue, 1, now)`, dropping the persistence write, and making
  // readVolume always return 0.7). The three tests below each pin exactly
  // one of those.
  describe("volume", () => {
    it("passes the current volume to player.play, not a hardcoded value", async () => {
      // Mutation caught: `emit`'s `player.play(cue, volume, now)` weakened
      // to `player.play(cue, 1, now)`. 0.3 (set below) is unambiguous
      // against both the hardcoded 1 and the 0.7 default.
      const play = vi.spyOn(SoundPlayer.prototype, "play").mockResolvedValue(undefined);
      render(
        <SoundProvider>
          <VolumeProbe />
        </SoundProvider>,
      );
      await act(async () => {
        toggleBtn().click();
      });
      await act(async () => {
        screen.getByText("set-volume").click();
      });
      await act(async () => {
        screen.getByText("emit").click();
      });
      expect(play).toHaveBeenCalled();
      const [, volumeArg] = play.mock.calls[0];
      expect(volumeArg).toBe(0.3);
    });

    it("persists a volume change to storage", async () => {
      // Mutation caught: dropping the `localStorage.setItem(VOLUME_KEY, ...)`
      // effect entirely -- the slider would move but never be remembered.
      render(
        <SoundProvider>
          <VolumeProbe />
        </SoundProvider>,
      );
      await act(async () => {
        screen.getByText("set-volume").click();
      });
      expect(localStorage.getItem("soundVolume")).toBe("0.3");
    });

    it("restores the persisted volume on mount", () => {
      // Mutation caught: `readVolume` weakened to always return 0.7 -- a
      // fresh mount would ignore whatever was stored and silently reset the
      // slider every reload.
      vi.stubGlobal("localStorage", memoryStorage({ soundVolume: "0.42" }));
      render(
        <SoundProvider>
          <VolumeProbe />
        </SoundProvider>,
      );
      expect(screen.getByTestId("volume")).toHaveTextContent("0.42");
    });
  });
});
