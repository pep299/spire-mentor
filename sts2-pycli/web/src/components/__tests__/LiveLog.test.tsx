import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import LiveLog from "../LiveLog";
import type { LiveTimeline } from "../../api";

function node(over: Partial<LiveTimeline["nodes"][number]> & { id: number }) {
  return {
    parent_id: null, idx: 0, decision: "combat_play", label: `n${over.id}`,
    details: [], error: false, child_ids: [], ...over,
  };
}

describe("LiveLog", () => {
  it("renders nothing without a timeline", () => {
    const { container } = render(<LiveLog timeline={null} onSeek={() => {}} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the active path and seeks on click", () => {
    const onSeek = vi.fn();
    const timeline: LiveTimeline = {
      nodes: [
        node({ id: 0, label: "Started Ironclad", child_ids: [1] }),
        node({ id: 1, parent_id: 0, idx: 1, label: "Map → (0,0)", child_ids: [] }),
      ],
      active_path: [0, 1],
      active_leaf: 1,
    };
    render(<LiveLog timeline={timeline} onSeek={onSeek} />);
    fireEvent.click(screen.getByRole("button", { name: /Started Ironclad/ }));
    expect(onSeek).toHaveBeenCalledWith(0);
  });

  it("does not seek when the click ends a text selection in the row", () => {
    // Rows used to be <button>s, whose text a browser will not select; a drag
    // inside one also fired a click and rewound the run.
    const onSeek = vi.fn();
    const timeline: LiveTimeline = {
      nodes: [
        node({ id: 0, label: "Started Ironclad", child_ids: [1] }),
        node({ id: 1, parent_id: 0, idx: 1, label: "Map → (0,0)", child_ids: [] }),
      ],
      active_path: [0, 1],
      active_leaf: 1,
    };
    render(<LiveLog timeline={timeline} onSeek={onSeek} />);
    const row = screen.getByRole("button", { name: /Started Ironclad/ });
    expect(row.tagName).not.toBe("BUTTON");
    getSelection()!.selectAllChildren(screen.getByText("Started Ironclad"));
    fireEvent.click(row);
    expect(onSeek).not.toHaveBeenCalled();
    getSelection()!.removeAllRanges();
    fireEvent.click(row);
    expect(onSeek).toHaveBeenCalledWith(0);
  });

  it("seeks from the keyboard, and not while busy", () => {
    const onSeek = vi.fn();
    const timeline: LiveTimeline = {
      nodes: [
        node({ id: 0, label: "root", child_ids: [1] }),
        node({ id: 1, parent_id: 0, idx: 1, label: "latest move", child_ids: [] }),
      ],
      active_path: [0, 1],
      active_leaf: 1,
    };
    const { rerender } = render(<LiveLog timeline={timeline} onSeek={onSeek} />);
    const row = screen.getByRole("button", { name: /root/ });
    fireEvent.keyDown(row, { key: "Enter" });
    fireEvent.keyDown(row, { key: " " });
    expect(onSeek.mock.calls).toEqual([[0], [0]]);
    rerender(<LiveLog timeline={timeline} onSeek={onSeek} busy />);
    expect(row).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(row);
    fireEvent.keyDown(row, { key: "Enter" });
    expect(onSeek).toHaveBeenCalledTimes(2);
  });

  it("marks the active leaf as the current position", () => {
    const timeline: LiveTimeline = {
      nodes: [
        node({ id: 0, label: "root", child_ids: [1] }),
        node({ id: 1, parent_id: 0, idx: 1, label: "latest move", child_ids: [] }),
      ],
      active_path: [0, 1],
      active_leaf: 1,
    };
    render(<LiveLog timeline={timeline} onSeek={() => {}} />);
    const leafBtn = screen.getByRole("button", { name: /latest move/ });
    expect(leafBtn).toHaveAttribute("aria-current", "step");
    expect(leafBtn).toHaveTextContent("now");
    expect(screen.getByRole("button", { name: /^.*root.*$/ })).not.toHaveAttribute("aria-current");
  });

  it("shows a branch switcher for sibling nodes and seeks to a sibling", () => {
    const onSeek = vi.fn();
    // Node 1 has two children (2 = active, 3 = the other branch).
    const timeline: LiveTimeline = {
      nodes: [
        node({ id: 0, label: "root", child_ids: [1] }),
        node({ id: 1, parent_id: 0, idx: 1, label: "fork point", child_ids: [2, 3] }),
        node({ id: 2, parent_id: 1, idx: 2, label: "took A", child_ids: [] }),
        node({ id: 3, parent_id: 1, idx: 2, label: "took B", child_ids: [] }),
      ],
      active_path: [0, 1, 2],
      active_leaf: 2,
    };
    render(<LiveLog timeline={timeline} onSeek={onSeek} />);
    // The active path shows "took A"; the alternative "took B" is offered.
    const branchBtn = screen.getByRole("button", { name: /took B/ });
    fireEvent.click(branchBtn);
    expect(onSeek).toHaveBeenCalledWith(3);
  });
});
