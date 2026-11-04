import { describe, expect, it, vi } from "vitest";
import { BENCH_SIMS, runBench } from "../benchRun";

describe("runBench", () => {
  it("times every preset through the API shim and reports ms per sim", async () => {
    const calls: number[] = [];
    let t = 0;
    const fetchFn = vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as { sims: number };
      calls.push(body.sims);
      t += body.sims; // 1 ms per sim on this fake clock
      return new Response("{}", { status: 200 });
    });
    const report = await runBench(fetchFn as never, () => t);
    expect(calls).toEqual([...BENCH_SIMS]);
    expect(report).toContain("1000 sims: 1000 ms (1.000 ms/sim)");
    expect(fetchFn.mock.calls[0][0]).toBe("/api/bench");
  });

  it("keeps going and names the status when a preset is refused", async () => {
    const fetchFn = vi.fn(async () => new Response("no combat", { status: 400 }));
    const report = await runBench(fetchFn as never, () => 0);
    expect(report).toContain("200 sims: HTTP 400");
    expect(fetchFn).toHaveBeenCalledTimes(BENCH_SIMS.length);
  });
});
