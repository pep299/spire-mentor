import { describe, it, expect, vi } from "vitest";
import { render, act } from "@testing-library/react";
import { useRef } from "react";
import SoundBridge from "../SoundBridge";
import { SoundFixture } from "../SoundContext";
import type { SoundValue } from "../useSound";
import type { CombatState, Step } from "../../api";
import type { SfxMap } from "../sfxMap";

const MAP: SfxMap = {
  version: "test",
  generic: {
    gold: "event:/sfx/gold",
    defeat: "event:/sfx/defeat",
    victory: "res://debug_audio/victory.mp3",
    relicGet: "res://debug_audio/relic_get.mp3",
    neowWelcome: "event:/sfx/npcs/neow/neow_welcome",
  },
  cards: {}, relics: {}, monsters: {}, powers: {},
};

function state(over: Partial<CombatState> = {}): CombatState {
  return {
    decision: "combat_play", round: 1, energy: 3, max_energy: 3,
    draw_pile_count: 0, discard_pile_count: 0, exhaust_pile_count: 0,
    player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0 },
    player_powers: [], enemies: [], hand: [], ...over,
  } as CombatState;
}

function step(decision: string, over: Partial<CombatState>): Step {
  return {
    idx: 0, decision, scores: [], pv: null, chosen_action: null,
    source: "live", meta: { turn: 1, step_no: 0 },
    state: state({ decision, ...over }),
  };
}

/** Renders SoundBridge with a ref it never reads directly — only `emit`
 *  observations matter for these tests, same as boardSound.test.tsx. */
function Harness({ live, mode }: { live: Step | null; mode: "replay" | "play" | "mcts" }) {
  const soundRef = useRef<SoundValue | null>(null);
  return <SoundBridge soundRef={soundRef} live={live} mode={mode} />;
}

describe("SoundBridge — gold-gain cue", () => {
  it("emits the gold cue when live gold increases", () => {
    const emit = vi.fn();
    const a = step("card_reward", { player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0, gold: 99 } } as Partial<CombatState>);
    const b = step("card_reward", { player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0, gold: 130 } } as Partial<CombatState>);
    const { rerender } = render(
      <SoundFixture on map={MAP} emit={emit}><Harness live={a} mode="play" /></SoundFixture>,
    );
    act(() => {
      rerender(<SoundFixture on map={MAP} emit={emit}><Harness live={b} mode="play" /></SoundFixture>);
    });
    expect(emit).toHaveBeenCalledTimes(1);
    expect((emit.mock.calls[0][0] as { event: string }[])[0].event).toBe("event:/sfx/gold");
  });

  // Mutation caught: deleting the `sound.emit(...)` call (or the whole gold
  // effect) from SoundBridge.
  it("does not emit when gold decreases (a shop purchase)", () => {
    const emit = vi.fn();
    const a = step("shop", { player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0, gold: 130 } } as Partial<CombatState>);
    const b = step("shop", { player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0, gold: 55 } } as Partial<CombatState>);
    const { rerender } = render(
      <SoundFixture on map={MAP} emit={emit}><Harness live={a} mode="play" /></SoundFixture>,
    );
    act(() => {
      rerender(<SoundFixture on map={MAP} emit={emit}><Harness live={b} mode="play" /></SoundFixture>);
    });
    expect(emit).not.toHaveBeenCalled();
  });

  it("does not emit while replaying (mode !== 'play')", () => {
    const emit = vi.fn();
    const a = step("card_reward", { player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0, gold: 10 } } as Partial<CombatState>);
    const b = step("card_reward", { player: { name: "Ironclad", hp: 80, max_hp: 80, block: 0, gold: 50 } } as Partial<CombatState>);
    const { rerender } = render(
      <SoundFixture on map={MAP} emit={emit}><Harness live={a} mode="replay" /></SoundFixture>,
    );
    act(() => {
      rerender(<SoundFixture on map={MAP} emit={emit}><Harness live={b} mode="replay" /></SoundFixture>);
    });
    expect(emit).not.toHaveBeenCalled();
  });
});

describe("SoundBridge — run-end cue", () => {
  it("emits the defeat cue on reaching game_over undefeated", () => {
    const emit = vi.fn();
    const a = step("combat_play", {});
    const b = step("game_over", { victory: false } as Partial<CombatState>);
    const { rerender } = render(
      <SoundFixture on map={MAP} emit={emit}><Harness live={a} mode="play" /></SoundFixture>,
    );
    act(() => {
      rerender(<SoundFixture on map={MAP} emit={emit}><Harness live={b} mode="play" /></SoundFixture>);
    });
    expect(emit).toHaveBeenCalledTimes(1);
    expect((emit.mock.calls[0][0] as { event: string }[])[0].event).toBe("event:/sfx/defeat");
  });

  // Mutation caught: dropping the `gameOverCuedRef` latch — a second `live`
  // update that's still at game_over (e.g. a re-fetch of the same step)
  // would otherwise replay the cue.
  it("only fires once across repeated game_over updates", () => {
    const emit = vi.fn();
    const a = step("combat_play", {});
    const b = step("game_over", { victory: false } as Partial<CombatState>);
    const c = step("game_over", { victory: false } as Partial<CombatState>);
    const { rerender } = render(
      <SoundFixture on map={MAP} emit={emit}><Harness live={a} mode="play" /></SoundFixture>,
    );
    act(() => {
      rerender(<SoundFixture on map={MAP} emit={emit}><Harness live={b} mode="play" /></SoundFixture>);
    });
    act(() => {
      rerender(<SoundFixture on map={MAP} emit={emit}><Harness live={c} mode="play" /></SoundFixture>);
    });
    expect(emit).toHaveBeenCalledTimes(1);
  });
});

/** The events emitted across every `emit` call, flattened in order. */
function emitted(emit: { mock: { calls: unknown[][] } }): string[] {
  return emit.mock.calls.flatMap((c) => (c[0] as { event: string }[]).map((q) => q.event));
}

/** Renders `a`, then `b`, and returns everything emitted on the transition. */
function transition(a: Step, b: Step, mode: "replay" | "play" | "mcts" = "play"): string[] {
  const emit = vi.fn();
  const { rerender } = render(
    <SoundFixture on map={MAP} emit={emit}><Harness live={a} mode={mode} /></SoundFixture>,
  );
  emit.mockClear();
  act(() => {
    rerender(<SoundFixture on map={MAP} emit={emit}><Harness live={b} mode={mode} /></SoundFixture>);
  });
  return emitted(emit);
}

function relics(n: number): Partial<CombatState> {
  return {
    player: {
      name: "Ironclad", hp: 80, max_hp: 80, block: 0,
      relics: Array.from({ length: n }, (_, i) => ({ index: i, id: `r${i}`, name: `Relic ${i}`, description: "", image: null })),
    },
  } as Partial<CombatState>;
}

describe("SoundBridge — combat victory cue", () => {
  it("emits on arrival at the rewards screen", () => {
    expect(transition(step("combat_play", {}), step("card_reward", {})))
      .toContain("res://debug_audio/victory.mp3");
  });

  // Mutation caught: dropping the previous-decision ref, which makes every
  // background poll of the same rewards step re-announce the win.
  it("does not re-emit while the rewards screen stays open", () => {
    expect(transition(step("card_reward", {}), step("card_reward", {})))
      .not.toContain("res://debug_audio/victory.mp3");
  });

  it("stays silent while replaying", () => {
    expect(transition(step("combat_play", {}), step("card_reward", {}), "replay")).toEqual([]);
  });
});

describe("SoundBridge — relic-obtained cue", () => {
  it("emits when a relic joins the inventory", () => {
    expect(transition(step("event_choice", relics(0)), step("map_select", relics(1))))
      .toContain("res://debug_audio/relic_get.mp3");
  });

  // Mutation caught: diffing against `undefined` as if it were zero. `relics`
  // is optional on the wire, so a step that omits it would otherwise read as
  // "lost them all" and then the next step as "gained them all back".
  it("treats a step with no relics field as no baseline, not as zero", () => {
    const withRelics = step("map_select", relics(2));
    const without = step("combat_play", {});
    const emit = vi.fn();
    const { rerender } = render(
      <SoundFixture on map={MAP} emit={emit}><Harness live={withRelics} mode="play" /></SoundFixture>,
    );
    act(() => {
      rerender(<SoundFixture on map={MAP} emit={emit}><Harness live={without} mode="play" /></SoundFixture>);
    });
    emit.mockClear();
    act(() => {
      rerender(<SoundFixture on map={MAP} emit={emit}><Harness live={withRelics} mode="play" /></SoundFixture>);
    });
    expect(emitted(emit)).not.toContain("res://debug_audio/relic_get.mp3");
  });

  it("does not emit when the inventory holds steady", () => {
    expect(transition(step("map_select", relics(2)), step("shop", relics(2))))
      .not.toContain("res://debug_audio/relic_get.mp3");
  });
});

describe("SoundBridge — Neow greeting", () => {
  it("emits on arrival in Neow's room", () => {
    expect(transition(step("map_select", {}), step("event_choice", { event_name: "neow" })))
      .toContain("event:/sfx/npcs/neow/neow_welcome");
  });

  // Mutation caught: gating on the decision alone — `event_choice` is every
  // event room, and Neow's voice at the Dead Adventurer is worse than silence.
  it("stays silent in another event room", () => {
    expect(transition(step("map_select", {}), step("event_choice", { event_name: "Golden Idol" })))
      .not.toContain("event:/sfx/npcs/neow/neow_welcome");
  });

  it("does not re-emit while her screen stays open", () => {
    expect(transition(step("event_choice", { event_name: "neow" }), step("event_choice", { event_name: "neow" })))
      .not.toContain("event:/sfx/npcs/neow/neow_welcome");
  });
});
