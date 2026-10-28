import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { TimelineEntry } from "../api";
import Timeline from "./Timeline";

const entries = [0, 1, 2].map(
  (idx) => ({ idx, decision: "combat_play", label: `step ${idx}` }) as unknown as TimelineEntry,
);

describe("Timeline coach ticks", () => {
  it("ticks only the marked steps, and a tick seeks", () => {
    const onSeek = vi.fn();
    render(<Timeline idx={0} count={3} entries={entries} onSeek={onSeek} marks={new Map([[2, "blunder"]])} />);
    expect(screen.queryByTestId("coach-tick-0")).toBeNull();
    fireEvent.click(screen.getByTestId("coach-tick-2"));
    expect(onSeek).toHaveBeenCalledWith(2);
  });
  it("draws no tick row without marks", () => {
    render(<Timeline idx={0} count={3} entries={entries} onSeek={() => {}} />);
    expect(screen.queryByLabelText("coach marks")).toBeNull();
  });
});
