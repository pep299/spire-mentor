import { describe, it, expect } from "vitest";
import { resolveMode } from "../useEffectsMode";

describe("resolveMode", () => {
  it("off toggle always wins", () => {
    expect(resolveMode({ toggle: "off", autoplaying: false, reducedMotion: false })).toBe("off");
    expect(resolveMode({ toggle: "off", autoplaying: true, reducedMotion: true })).toBe("off");
  });
  it("autoplay downgrades to reduced so attack/hurt cues still play (motion calmed)", () => {
    expect(resolveMode({ toggle: "full", autoplaying: true, reducedMotion: false })).toBe("reduced");
    expect(resolveMode({ toggle: "reduced", autoplaying: true, reducedMotion: false })).toBe("reduced");
  });
  it("fx:off still fully suppresses during autoplay (opt-in silent fast-step)", () => {
    expect(resolveMode({ toggle: "off", autoplaying: true, reducedMotion: false })).toBe("off");
  });
  it("prefers-reduced-motion downgrades to reduced (still shows calm cues, not off)", () => {
    expect(resolveMode({ toggle: "full", autoplaying: false, reducedMotion: true })).toBe("reduced");
  });
  it("full when nothing suppresses it", () => {
    expect(resolveMode({ toggle: "full", autoplaying: false, reducedMotion: false })).toBe("full");
  });
});
