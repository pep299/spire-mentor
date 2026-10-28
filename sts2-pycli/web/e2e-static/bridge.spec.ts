import http from "node:http";
import type { AddressInfo } from "node:net";
import { test, expect } from "@playwright/test";
import { menuItem, closeMenu } from "../e2e/topbarMenu";
import { chromiumPath } from "../playwright.static.config";

/**
 * The live-game bridge in the browser-only build (GH #709): the page calls the
 * advisor's control plane on loopback DIRECTLY — no server in between — once
 * the user enters its URL in Tools → Live game…, and the control plane has the
 * page's origin allow-listed (its CORS + Private Network Access opt-in).
 *
 * A fake control plane stands in for the advisor here: it answers `/status`
 * and `/tree` the way `realgame_control.py` does with the origin allow-listed,
 * and records what the page sent. The tree node carries RAW combat-log rows,
 * which the wasm renders (`POST /api/bridge/render-tree`) — that is the one
 * transform the old proxy did.
 *
 * Chrome's Local Network Access permission (a prompt in a headed browser,
 * auto-denied headless) is switched off for this file — the equivalent of
 * the user allowing the prompt, which the measurements in
 * docs/browser-workbench.md § "Live-game bridge" cover.
 */
//
// `test.use({ launchOptions })` REPLACES the project's `launchOptions`, so the
// config's `executablePath` (from PLAYWRIGHT_CHROMIUM_PATH) is carried over
// here, or a host without bundled browsers can't launch this spec (GH #879).
test.use({
  launchOptions: {
    ...(chromiumPath ? { executablePath: chromiumPath } : {}),
    args: ["--disable-features=LocalNetworkAccessChecks"],
  },
});

const PAGE_ORIGIN = "http://localhost:4173";

interface Seen {
  method: string;
  path: string;
  origin: string | undefined;
}

let server: http.Server;
let base: string;
const seen: Seen[] = [];

const TREE = {
  nodes: [
    { id: 0, parent_id: null, idx: 0, decision: "root", label: "run start", details: [], error: false, child_ids: [1], floor: 0, act: 1, seekable: true, snapshot_id: "111-0" },
    {
      id: 1, parent_id: 0, idx: 1, decision: "combat_action", label: "f1 · Strike -> Nibbit", details: [], error: false, child_ids: [], floor: 1, act: 1, seekable: false,
      player_id: "IRONCLAD",
      combat_log_raw: [{
        kind: "damage", round: 1, side: "player", text: "IRONCLAD dealt 6 damage to NIBBIT",
        source: { kind: "card", id: "STRIKE_IRONCLAD", upgraded: false },
        dealer: { kind: "player", id: "IRONCLAD", uid: null },
        receiver: { kind: "monster", id: "Nibbit", uid: null },
        unblocked: 6, blocked: 0,
      }],
    },
  ],
  active_path: [0, 1],
  active_leaf: 1,
};

test.beforeAll(async () => {
  server = http.createServer((req, res) => {
    const path = (req.url ?? "/").split("?")[0];
    seen.push({ method: req.method ?? "", path, origin: req.headers.origin });
    // The control plane's opt-in: CORS + PNA headers for the allow-listed
    // origin only (realgame_control.py `_cors_headers`).
    if (req.headers.origin === PAGE_ORIGIN) {
      res.setHeader("Access-Control-Allow-Origin", PAGE_ORIGIN);
      res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "content-type");
      res.setHeader("Access-Control-Allow-Private-Network", "true");
      res.setHeader("Vary", "Origin");
    }
    if (req.method === "OPTIONS") {
      res.writeHead(req.headers.origin === PAGE_ORIGIN ? 204 : 403);
      res.end();
      return;
    }
    const send = (code: number, body: unknown) => {
      res.writeHead(code, { "content-type": "application/json" });
      res.end(JSON.stringify(body));
    };
    if (path === "/status") send(200, { connected: true, character: "Ironclad", seed: "S1", floor: 1, act: 1, log_path: null });
    else if (path === "/tree") send(200, TREE);
    else send(404, { error: `unknown path ${path}` });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

test.afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

test("Tools → Live game connects the page to the control plane and the Game tree opens over it", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("seed", { exact: true })).toBeVisible({ timeout: 60_000 });

  // Off by default: the Tools entry offers to connect, Analyze has no live views.
  const entry = await menuItem(page, "Tools", /Live game/);
  await expect(entry).toHaveText(/Live game…/);
  await entry.click();
  const dialog = page.getByRole("dialog", { name: "Live game" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText(/Browser workbench origin/)).toBeVisible();
  await dialog.getByLabel("control plane URL").fill(base);
  await dialog.getByRole("button", { name: "connect" }).click();
  await dialog.getByRole("button", { name: "close" }).click();
  await expect(dialog).toBeHidden();

  // The probe reached the fake advisor directly, with this page's origin.
  await expect.poll(() => seen.some((s) => s.path === "/status" && s.origin === PAGE_ORIGIN), { timeout: 20_000 }).toBe(true);
  const connected = await menuItem(page, "Tools", /Live game/);
  await expect(connected).toHaveText(/connected/, { timeout: 20_000 });
  await closeMenu(page);

  // The live views appear, and the Game tree comes from the control plane —
  // its raw combat log rendered by the wasm on the way.
  const tree = await menuItem(page, "Analyze", /Game tree/);
  await expect(tree).toBeVisible();
  await tree.click();
  const treeDialog = page.getByRole("dialog", { name: "Branch tree" });
  await expect(treeDialog).toBeVisible({ timeout: 20_000 });
  await expect(treeDialog.getByText("f1 · Strike -> Nibbit")).toBeVisible();
  expect(seen.some((s) => s.path === "/tree" && s.origin === PAGE_ORIGIN)).toBe(true);

  // The setting survives a reload; forgetting it drops the views again.
  await page.reload();
  await expect(page.getByLabel("seed", { exact: true })).toBeVisible({ timeout: 60_000 });
  const again = await menuItem(page, "Tools", /Live game/);
  await expect(again).toHaveText(/connected/, { timeout: 20_000 });
  await again.click();
  await page.getByRole("dialog", { name: "Live game" }).getByRole("button", { name: "forget" }).click();
  await page.getByRole("dialog", { name: "Live game" }).getByRole("button", { name: "close" }).click();
  const off = await menuItem(page, "Tools", /Live game/);
  await expect(off).toHaveText(/Live game…/);
  await closeMenu(page);
  await expect(page.getByRole("button", { name: /^Analyze\b/ })).toBeVisible();
  const analyze = await menuItem(page, "Analyze", /Game tree/);
  await expect(analyze).toHaveCount(0);
});
