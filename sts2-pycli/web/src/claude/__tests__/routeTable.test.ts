// Reads api.ts and the Rust route table off disk with node:fs, so it lives in
// the tsconfig.node-tests project (see that project's comment).
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { MUTATING, READ_ONLY_LIVE } from "../link";

describe("Claude link route classification", () => {
  // The two sets are hand-kept; this ties them to the route tables, so a new
  // POST route has to be classified rather than silently wiping the scores.
  const classify = (routes: string[], where: string) => {
    expect(routes.length, `no POST /api/live routes parsed from ${where}`).toBeGreaterThan(10);
    for (const r of routes) {
      const ro = READ_ONLY_LIVE.has(r);
      const mu = MUTATING.has(r);
      expect(ro !== mu, `${where}: POST ${r} must be in exactly one of READ_ONLY_LIVE / MUTATING`).toBe(true);
    }
  };
  it("every POST /api/live route in api.ts is classified", () => {
    const src = readFileSync(resolve(__dirname, "../../api.ts"), "utf-8");
    const routes = [...src.matchAll(/apiFetch\("(\/api\/live\/[^"?]+)",\s*\{[^}]*?method:\s*"POST"/g)].map((m) => m[1]);
    classify([...new Set(routes)], "api.ts");
  });
  const rs = resolve(__dirname, "../../../../../sts2-rs/crates/sts2-workbench/src/api.rs");
  it.skipIf(!existsSync(rs))("every POST /api/live route in the Rust route table is classified", () => {
    const src = readFileSync(rs, "utf-8");
    const routes: string[] = [];
    for (const m of src.matchAll(/\("POST",\s*((?:"[^"]+"\s*\|?\s*)+)\)/g)) {
      for (const q of m[1].matchAll(/"([^"]+)"/g)) if (q[1].startsWith("/api/live/")) routes.push(q[1]);
    }
    classify([...new Set(routes)], "api.rs");
  });
});
