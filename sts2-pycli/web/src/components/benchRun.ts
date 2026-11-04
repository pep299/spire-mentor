// The bench run behind BenchButton: time the hint search at each advisor
// preset.
import { apiFetch } from "../transport";

/** The sims budgets timed, in order: today's Quick and Deep advisor presets. */
export const BENCH_SIMS = [200, 800, 1000, 4000] as const;

/**
 * Time `POST /api/bench` (the hint endpoint's single-tree search over the
 * live combat, `bench.rs` in sts2-workbench) at each preset and report
 * ms / sim. The result is shown in an alert and copied to the clipboard so
 * it can be pasted into `sts2-rs/docs/perf/` from a phone, where devtools
 * are out of reach. Static build only: the server-backed build has no
 * `/api/bench` route.
 */
export async function runBench(
  fetchFn: typeof apiFetch = apiFetch,
  now: () => number = () => performance.now(),
): Promise<string> {
  const rows: string[] = [];
  for (const sims of BENCH_SIMS) {
    const t0 = now();
    const r = await fetchFn("/api/bench", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sims, seed: 7 }),
    });
    const ms = now() - t0;
    if (!r.ok) {
      rows.push(`${sims} sims: HTTP ${r.status}`);
      continue;
    }
    rows.push(`${sims} sims: ${Math.round(ms)} ms (${(ms / sims).toFixed(3)} ms/sim)`);
  }
  const ua = typeof navigator === "undefined" ? "" : navigator.userAgent;
  return [`workbench bench — ${new Date().toISOString()}`, ua, ...rows].join("\n");
}
