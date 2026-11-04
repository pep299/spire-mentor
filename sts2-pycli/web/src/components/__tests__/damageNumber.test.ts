import { describe, it, expect } from "vitest";
import { damageNumberJitter } from "../combatantMetrics";

describe("damageNumberJitter", () => {
  it("throws consecutive hits along visibly different paths", () => {
    // A multi-hit attack lands several numbers on one creature. Identical
    // throws stack them into one illegible blob.
    const a = damageNumberJitter(1) as Record<string, string>;
    const b = damageNumberJitter(2) as Record<string, string>;
    expect(a["--dmg-dx"]).not.toBe(b["--dmg-dx"]);
    expect(a["--dmg-jx"]).not.toBe(b["--dmg-jx"]);
    expect(a["--dmg-rot"]).not.toBe(b["--dmg-rot"]);
  });

  it("is stable for one hit, so a re-render can't teleport it mid-flight", () => {
    expect(damageNumberJitter(7)).toEqual(damageNumberJitter(7));
  });

  it("keeps every throw inside the game's ranges", () => {
    for (let n = 0; n < 200; n++) {
      const j = damageNumberJitter(n) as Record<string, string>;
      expect(Math.abs(parseFloat(j["--dmg-jx"]))).toBeLessThanOrEqual(6);
      expect(parseFloat(j["--dmg-jy"])).toBeGreaterThanOrEqual(-58);
      expect(parseFloat(j["--dmg-jy"])).toBeLessThanOrEqual(-52);
      expect(Math.abs(parseFloat(j["--dmg-dx"]))).toBeLessThanOrEqual(66);
      expect(parseFloat(j["--dmg-vy"])).toBeGreaterThanOrEqual(0.93);
      expect(parseFloat(j["--dmg-vy"])).toBeLessThanOrEqual(1.07);
      expect(Math.abs(parseFloat(j["--dmg-rot"]))).toBeLessThanOrEqual(5);
    }
  });

  it("spreads throws across the range rather than clustering", () => {
    // A weak hash of consecutive nonces would drift by a hair each time and
    // every number would fly the same way.
    const left: number[] = [], right: number[] = [];
    for (let n = 0; n < 60; n++) {
      const dx = parseFloat((damageNumberJitter(n) as Record<string, string>)["--dmg-dx"]);
      (dx < 0 ? left : right).push(dx);
    }
    expect(left.length).toBeGreaterThan(15);
    expect(right.length).toBeGreaterThan(15);
  });
});
