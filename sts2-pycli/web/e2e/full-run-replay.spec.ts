import { test, expect, type Page } from "@playwright/test";
import { menuItem } from "./topbarMenu";

/**
 * End-to-end coverage that the unit + parity tests can't reach: drive a WHOLE
 * run to game_over in the real browser, then exercise the replay path the
 * finished run feeds into. Complements `play-to-end.spec.ts` (which only nudges
 * a run a few steps and never completes a run or touches replay).
 *
 * The old "Fast mode" is gone — every Play-to-end run goes through the same
 * startFaithful + run-mode stepper. A minimal budget (1 monster/boss sim) is
 * the cheapest configuration and plays a full run to its end in a few
 * minutes.
 */

test.describe("full cheap-budget run + replay", () => {
  test.setTimeout(300_000);

  /** Open the app on the play-mode start form. Play is the default view on a
   *  fresh session, so no mode toggle needs clicking. */
  async function openStartForm(page: Page): Promise<void> {
    await page.goto("/");
    await expect(page.getByLabel("seed", { exact: true })).toBeVisible();
  }

  /** Live branch-tree history rows (play mode). */
  function historyRows(page: Page) {
    return page.locator(
      '[role="button"][title="rewind the run to this step"], [role="button"][title="current position"]',
    );
  }

  /** The current live decision, read from the backend through the page's own
   *  session cookie (a separate client gets its own empty session). Returns
   *  null when there is no live run (the endpoint 400s). */
  async function liveDecision(page: Page): Promise<string | null> {
    const r = await page.request.get("/api/live/state");
    if (r.status() === 400) return null;
    const j = await r.json();
    return j?.decision ?? j?.step?.decision ?? null;
  }

  /** Number of recorded steps in the loaded replay run (session.store). */
  async function replayStepCount(page: Page): Promise<number> {
    const r = await page.request.get("/api/timeline");
    if (!r.ok()) return 0;
    const j = await r.json();
    return (j?.steps ?? []).length;
  }

  test("plays to game_over, seeks the finished run, and replays it", async ({ page }) => {
    await openStartForm(page);
    await page.getByLabel("seed", { exact: true }).fill("e2e_fullrun");
    // Cheapest possible budget so a complete run fits in the test timeout.
    await page.getByLabel("monster combat sims").fill("1");
    await page.getByLabel("boss combat sims").fill("1");
    await page.getByRole("button", { name: "Watch the AI play" }).click();

    // --- play the whole run to its end ---
    await expect
      .poll(() => liveDecision(page), { timeout: 270_000, intervals: [1000] })
      .toBe("game_over");

    // game-over screen: outcome banner + a fresh-start affordance.
    await expect(page.getByText(/Defeat|Victory/).first()).toBeVisible();
    await expect(page.getByRole("button", { name: "New run" })).toBeVisible();

    // The run stayed live, so the branch tree is fully populated.
    const rowCount = await historyRows(page).count();
    expect(rowCount).toBeGreaterThan(10);

    // --- bug-5 regression: clicking a history node after the run finished
    //     rewinds the LIVE session and shows that step's view (it used to be
    //     inert because the benchmark path left the UI in replay mode). ---
    const mid = historyRows(page).nth(Math.floor(rowCount / 2));
    await mid.click();
    await expect(mid).toHaveAttribute("aria-current", "step");

    // --- replay: End run lands on the run-ended summary, whose "Review
    //     replay" reopens the finished run in the replay viewer ---
    page.once("dialog", (d) => d.accept());
    await (await menuItem(page, "Run", "End run")).click();
    await page.getByRole("button", { name: "Review replay" }).click();

    await expect.poll(() => replayStepCount(page), { timeout: 20_000 }).toBeGreaterThan(0);
    const steps = await replayStepCount(page);

    // Scrub forward through the replay until a combat decision view renders
    // (the Board shows "End Turn"); every recorded step is reachable.
    // loadRun leaves us at step 0, so "⏮ first" is disabled — just walk forward.
    const fwd = page.getByRole("button", { name: /fwd/ });
    let sawCombat = false;
    for (let i = 0; i < Math.min(steps - 1, 60) && !sawCombat; i++) {
      await fwd.click();
      sawCombat = await page
        .getByText("End Turn", { exact: false })
        .first()
        .isVisible()
        .catch(() => false);
    }
    expect(sawCombat).toBe(true);

    // Replay-mode regret review opens and produces its dock (diff vs MCTS line).
    await (await menuItem(page, "Analyze", /regret review/)).click();
    await expect(page.getByText(/blunder|decision|drop|regret/i).first()).toBeVisible({
      timeout: 20_000,
    });
  });
});
