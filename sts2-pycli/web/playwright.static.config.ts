import { existsSync } from "node:fs";
import { defineConfig, devices } from "@playwright/test";

// A host that ships its own Chromium (the remote dev containers put one at
// /opt/pw-browsers/chromium, or PLAYWRIGHT_CHROMIUM_PATH names one) may not
// have the revision this Playwright version pins; use it when present.
export const chromiumPath = [process.env.PLAYWRIGHT_CHROMIUM_PATH, "/opt/pw-browsers/chromium"].find(
  (p): p is string => !!p && existsSync(p),
);

// End-to-end tests for the BROWSER-ONLY workbench: no Python backend at all.
// The SPA is built in static mode (`npm run build:static`, which needs the
// wasm bundle from `just wasm-build`) and served by `vite preview`; every
// `/api/*` call is answered by the Rust engine running in the Web Worker.
//
// Run: `npx playwright test -c playwright.static.config.ts` (or `just
// e2e-static` from sts2-pycli/). The specs live in e2e-static/ so the default
// config (which boots the Python backend) never picks them up.
export default defineConfig({
  testDir: "./e2e-static",
  fullyParallel: false,
  workers: 1,
  timeout: 180_000,
  expect: { timeout: 20_000 },
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:4173",
    headless: true,
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        ...(chromiumPath ? { launchOptions: { executablePath: chromiumPath } } : {}),
      },
    },
  ],
  webServer: {
    command: "npm run preview:static -- --port 4173 --strictPort",
    url: "http://localhost:4173",
    timeout: 60_000,
    reuseExistingServer: true,
  },
});
