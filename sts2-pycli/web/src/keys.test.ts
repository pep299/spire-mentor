import { describe, it, expect } from "vitest";
import { isBrowserChord, isTextInput } from "./keys";

describe("isTextInput", () => {
  it("is true for input/textarea/select elements", () => {
    expect(isTextInput(document.createElement("input"))).toBe(true);
    expect(isTextInput(document.createElement("textarea"))).toBe(true);
    expect(isTextInput(document.createElement("select"))).toBe(true);
  });
  it("is false for other elements and null", () => {
    expect(isTextInput(document.createElement("div"))).toBe(false);
    expect(isTextInput(null)).toBe(false);
  });
});

describe("isBrowserChord", () => {
  const key = (init: KeyboardEventInit) => new KeyboardEvent("keydown", { key: "a", ...init });
  it("is true for Ctrl / Meta / Alt chords", () => {
    expect(isBrowserChord(key({ ctrlKey: true }))).toBe(true);
    expect(isBrowserChord(key({ metaKey: true }))).toBe(true);
    expect(isBrowserChord(key({ altKey: true }))).toBe(true);
  });
  it("is false for a bare or Shift-ed key, and for AltGr", () => {
    expect(isBrowserChord(key({}))).toBe(false);
    expect(isBrowserChord(key({ key: "?", shiftKey: true }))).toBe(false);
    expect(isBrowserChord(key({ ctrlKey: true, altKey: true, modifierAltGraph: true } as KeyboardEventInit))).toBe(false);
  });
});
