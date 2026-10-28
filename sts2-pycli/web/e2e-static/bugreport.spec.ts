import { test, expect, type Page } from "@playwright/test";
import { openMenu } from "../e2e/topbarMenu";

/**
 * Bug reports in the browser-only build: the PAGE builds the dump from the
 * engine in its worker and posts it to the host's `/api/bugreport` Function;
 * `?report=<id>` shows it and "Load into this workbench" takes the
 * reporter's run into this browser's engine.
 *
 * `vite preview` has no Function, so the endpoint is played by `page.route`
 * — what this owns is the page side: the probe turning the button on, the
 * dump's content, the round trip through the view, and the import putting
 * the live run back where the report was taken.
 */

const ID = "wb-e2e0000000000001";

// The service worker passes /api/* through untouched, but keep it out of
// the way of the route interception entirely.
test.use({ serviceWorkers: "block" });

function historyRows(page: Page) {
  return page.locator('button[title="rewind the run to this step"], button[title="current position"]');
}

async function startRun(page: Page, seed: string): Promise<void> {
  await page.goto("/");
  await expect(page.getByLabel("seed", { exact: true })).toBeVisible();
  await page.getByLabel("seed", { exact: true }).fill(seed);
  await page.getByRole("button", { name: "Play yourself" }).click();
  await expect(page.locator(".gs-event-option").first()).toBeVisible({ timeout: 60_000 });
}

test("capture posts the engine's dump, the report view shows it, and loading it restores the run", async ({ page, baseURL }) => {
  const posted: Record<string, unknown>[] = [];
  let stored: Record<string, unknown> | null = null;
  await page.route("**/api/bugreport", async (route) => {
    const req = route.request();
    if (req.method() === "GET") {
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ok: true, store: "kv" }) });
      return;
    }
    const dump = req.postDataJSON() as Record<string, unknown>;
    posted.push(dump);
    stored = { ...dump, meta: { ...(dump.meta as object), id: ID, captured_at: "2026-09-03T12:00:00Z" } };
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify({ id: ID, url: `${baseURL}/?report=${ID}` }),
    });
  });
  await page.route(`**/api/bugreport/${ID}`, async (route) => {
    if (!stored) {
      await route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: "unknown report" }) });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(stored) });
  });

  await startRun(page, "BUGREP0RT");
  // The probe answered yes: the button is there (a hosted row of the Tools menu).
  await openMenu(page, "Tools");
  const report = page.getByRole("button", { name: /🐞 Report/ });
  await expect(report).toBeVisible();
  page.once("dialog", (d) => d.accept("e2e: neow looks off"));
  await report.click();
  // The capture toast (`BugReportToast`) shows the stored report's id.
  await expect(page.getByRole("dialog", { name: "Bug report" })).toContainText(ID);

  expect(posted).toHaveLength(1);
  const dump = posted[0];
  expect(dump.schema_version).toBe(1);
  const meta = dump.meta as Record<string, unknown>;
  expect(meta.mode).toBe("live");
  expect(meta.backend).toBe("wasm");
  expect(meta.seed).toBe("BUGREP0RT");
  expect(typeof meta.character).toBe("string");
  expect(meta.host).toBe(new URL(baseURL!).origin);
  expect(dump.note).toBe("e2e: neow looks off");
  expect((dump.ui as { screen: string }).screen).toBe("event_choice");
  const repro = dump.repro as { persist: { nodes: unknown[]; seed: string } };
  expect(repro.persist.seed).toBe("BUGREP0RT");
  expect(repro.persist.nodes.length).toBeGreaterThanOrEqual(1);
  const state = dump.state as { game_state: Record<string, unknown>; map: unknown };
  expect(state.game_state).toBeTruthy();
  expect(state.map).toBeTruthy();

  // Move the run on, so loading the report has something to undo.
  await page.locator(".gs-event-option:not([disabled])").first().click();
  await expect.poll(() => historyRows(page).count(), { timeout: 60_000 }).toBeGreaterThan(1);

  // The read-only view, then the import.
  await page.goto(`/?report=${ID}`);
  await expect(page.getByRole("heading", { name: `Bug report ${ID}` })).toBeVisible();
  await expect(page.getByText("Note: e2e: neow looks off")).toBeVisible();
  await page.getByTestId("bugreport-load").click();
  await expect(page.getByTestId("bugreport-loaded")).toContainText("now the live run", { timeout: 60_000 });
  await page.getByRole("link", { name: "Open the workbench" }).click();
  // Back at Neow — the moment the report was taken — with one history row.
  await expect(page.locator(".gs-event-option").first()).toBeVisible({ timeout: 60_000 });
  await expect.poll(() => historyRows(page).count(), { timeout: 30_000 }).toBe(1);
});

test("without the Function the button stays hidden", async ({ page }) => {
  // `vite preview` answers 404 for /api/bugreport; the probe says no.
  await startRun(page, "NOENDPOINT");
  await openMenu(page, "Tools");
  await expect(page.getByRole("menuitemcheckbox", { name: /🐞 debug/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /🐞 Report/ })).toHaveCount(0);
});
