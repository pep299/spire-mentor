import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import BugReportView from "./BugReportView";
import { api } from "../api";

afterEach(() => vi.restoreAllMocks());

test("renders a fetched dump read-only", async () => {
  vi.spyOn(api, "fetchBugReport").mockResolvedValue({
    schema_version: 1,
    meta: { id: "wb-abc123", mode: "live", character: "Ironclad", captured_at: "2026-06-26T12:00:00Z" },
    repro: { timeline: { active_leaf: 0 } },
    state: { game_state: { floor: 3 }, map: null },
    ui: { screen: "combat", overlay: null },
    note: "enemy hp wrong",
  });
  render(<BugReportView id="wb-abc123" />);
  await waitFor(() => expect(screen.getByText(/wb-abc123/)).toBeInTheDocument());
  expect(screen.getByText(/enemy hp wrong/)).toBeInTheDocument();
  expect(screen.getByText(/combat/)).toBeInTheDocument();
});

test("shows not-found on 404", async () => {
  vi.spyOn(api, "fetchBugReport").mockRejectedValue(new Error("404"));
  render(<BugReportView id="wb-missing" />);
  await waitFor(() => expect(screen.getByText(/not found/i)).toBeInTheDocument());
});

test("static mode: a report that carries a run can be loaded into this workbench", async () => {
  const transport = await import("../transport");
  vi.spyOn(transport, "isStaticMode").mockReturnValue(true);
  const bugreport = await import("../bugreport");
  const load = vi.spyOn(bugreport, "loadReportIntoWorkbench").mockResolvedValue({ kind: "live" });
  vi.spyOn(api, "fetchBugReport").mockResolvedValue({
    schema_version: 1,
    meta: { id: "wb-0123456789abcdef", mode: "live", character: "Silent" },
    repro: { persist: { character: "Silent", nodes: [{ id: 0 }, { id: 1 }] } },
    state: { game_state: { floor: 1 }, map: null },
    ui: { screen: "event", overlay: null },
    note: null,
  });
  render(<BugReportView id="wb-0123456789abcdef" />);
  const button = await screen.findByTestId("bugreport-load");
  expect(button).toHaveTextContent(/Load into this workbench/);
  // The tree's body is summarised, not dumped.
  expect(screen.getByText(/persisted tree, 2 nodes/)).toBeInTheDocument();
  button.click();
  await waitFor(() => expect(screen.getByTestId("bugreport-loaded")).toBeInTheDocument());
  expect(load).toHaveBeenCalledTimes(1);
  expect(screen.getByRole("link", { name: /Open the workbench/ })).toHaveAttribute("href", "/");
});

test("a mod report shows the game side and imports the game position", async () => {
  const bugreport = await import("../bugreport");
  const imp = vi.spyOn(bugreport, "importReportPosition").mockResolvedValue({ lane: "combat" });
  const position = { type: "advice_request", advice_for: "combat_action", game_version: "v0.111.0" };
  vi.spyOn(api, "fetchBugReport").mockResolvedValue({
    schema_version: 1,
    meta: {
      id: "wb-00000000000000ff", source: "mod", mode: "game", character: "Defect",
      game_version: "v0.111.0", mod_version: "0.8.2", seed: "SEED1", ascension: 2, act: 2, floor: 20,
      coach: { mode: "Advisor", style: "Coach", advisor_connected: true }, os: "Windows 10",
    },
    repro: { run_save: { snapshot: { id: "1-2" }, save: { big: "x".repeat(50) } } },
    state: { game_state: position, map: null },
    ui: { screen: "combat_action", overlay: null },
    logs: { mod: ["mod line 1", "mod line 2"], game: "godot tail", advisor: null },
    settings: { AdvisorMode: true },
    note: "orb evoked twice",
  });
  render(<BugReportView id="wb-00000000000000ff" />);
  await screen.findByText(/Spire Mentor 0\.8\.2/);
  expect(screen.getByText(/seed SEED1 · A2 · act 2 floor 20/)).toBeInTheDocument();
  expect(screen.getByText(/advisor connected/)).toBeInTheDocument();
  expect(screen.getByText(/action log \(2 lines\)/)).toBeInTheDocument();
  expect(screen.getByText(/godot\.log \(tail/)).toBeInTheDocument();
  expect(screen.queryByText(/advisor stdout/)).toBeNull();
  // The run save is summarised, not dumped.
  expect(screen.getByText(/<run save, \d+ chars>/)).toBeInTheDocument();
  screen.getByTestId("bugreport-import-position").click();
  await waitFor(() => expect(screen.getByTestId("bugreport-imported")).toBeInTheDocument());
  expect(imp).toHaveBeenCalledTimes(1);
});

test("a workbench report without a game position offers no import", async () => {
  vi.spyOn(api, "fetchBugReport").mockResolvedValue({
    schema_version: 1,
    meta: { id: "wb-0000000000000001", mode: "live" },
    repro: {},
    state: { game_state: { floor: 1 }, map: null },
    ui: { screen: "map", overlay: null },
    note: null,
  });
  render(<BugReportView id="wb-0000000000000001" />);
  await screen.findByText(/wb-0000000000000001/);
  expect(screen.queryByTestId("bugreport-import-position")).toBeNull();
});
