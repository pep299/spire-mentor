import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import BugReportButton from "./BugReportButton";
import BugReportToast, { type BugReportResult } from "./BugReportToast";
import { api } from "../api";

afterEach(() => vi.restoreAllMocks());

const RESULT = {
  id: "wb-abc123",
  url: "https://storage.googleapis.com/b/bugreports/wb-abc123.json",
};

test("captures a report and hands the result up", async () => {
  vi.spyOn(api, "captureBugReport").mockResolvedValue(RESULT);
  // window.prompt is how the button collects the optional note.
  vi.spyOn(window, "prompt").mockReturnValue("enemy hp looks wrong");
  const captured: BugReportResult[] = [];

  render(<BugReportButton mode="play" step={null} screen="combat" runPath={null}
    onCaptured={(r) => captured.push(r)} onFailed={() => {}} />);
  fireEvent.click(screen.getByRole("button", { name: /report/i }));

  await waitFor(() => expect(captured).toEqual([RESULT]));
  expect(api.captureBugReport).toHaveBeenCalledWith({
    mode: "live",
    step: null,
    screen: "combat",
    note: "enemy hp looks wrong",
    run_path: null,
  });
  // ...and nothing is shown next to the button itself: this row lives in a
  // dropdown that unmounts as soon as it closes, so the id rendered here was
  // gone before it could be read (user report 2026-09-12).
  expect(screen.queryByText(/wb-abc123/)).toBeNull();
});

test("the toast shows the link, copies it, and dismisses", async () => {
  const writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
  const onDismiss = vi.fn();
  render(<BugReportToast result={RESULT} error={null} onDismiss={onDismiss} />);

  // The URL is on screen (and selectable), not just in the clipboard.
  expect(screen.getByLabelText("report URL")).toHaveValue(RESULT.url);
  // Copied once on mount, without a second click.
  await waitFor(() => expect(writeText).toHaveBeenCalledWith(RESULT.url));
  fireEvent.click(screen.getByTitle("Copy report URL"));
  await waitFor(() => expect(writeText).toHaveBeenCalledTimes(2));
  fireEvent.click(screen.getByLabelText("dismiss"));
  expect(onDismiss).toHaveBeenCalled();
  vi.unstubAllGlobals();
});

test("the toast keeps the link readable when the clipboard refuses", async () => {
  const writeText = vi.fn().mockRejectedValue(new Error("denied"));
  vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
  render(<BugReportToast result={RESULT} error={null} onDismiss={() => {}} />);
  await waitFor(() => expect(screen.getByText(/clipboard refused/)).toBeInTheDocument());
  expect(screen.getByLabelText("report URL")).toHaveValue(RESULT.url);
  vi.unstubAllGlobals();
});

test("a failed capture is reported, not swallowed", async () => {
  render(<BugReportToast result={null} error="capture failed" onDismiss={() => {}} />);
  expect(screen.getByText("capture failed")).toBeInTheDocument();
});

test("a report saved locally (no store to reach) says so and offers the path", async () => {
  const writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
  const local = {
    id: "wb-00000000000000aa",
    url: "C:\\Users\\<user>\\AppData\\Local\\SpireMentor\\logs\\bugreports\\wb-00000000000000aa.json",
    detail: "Not sent (could not reach the report store). Saved to … -- attach that file.",
  };
  render(<BugReportToast result={local} error={null} onDismiss={() => {}} />);
  expect(screen.getByTestId("bugreport-detail")).toHaveTextContent(/Not sent/);
  expect(screen.getByTitle("Copy report URL")).toHaveTextContent(/copy path/);
  await waitFor(() => expect(writeText).toHaveBeenCalledWith(local.url));
  vi.unstubAllGlobals();
});
