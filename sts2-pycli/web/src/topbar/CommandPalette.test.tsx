import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import CommandPalette from "./CommandPalette";
import { filterEntries, isPaletteChord, type PaletteEntry } from "./paletteEntries";

const entries = (run = vi.fn()): PaletteEntry[] => [
  { id: "save", label: "Save", group: "Run", run },
  { id: "end-run", label: "End run", group: "Run", run: () => {} },
  { id: "regret", label: "📉 regret review", group: "Analyze", title: "diff every move", run: () => {} },
  { id: "search", label: "🔬 search tree", group: "Analyze", disabledReason: "only during a combat", run: () => {} },
];

describe("filterEntries", () => {
  it("matches every word against group, label and title, case-insensitively", () => {
    expect(filterEntries(entries(), "").map((e) => e.id)).toEqual(["save", "end-run", "regret", "search"]);
    expect(filterEntries(entries(), "RUN").map((e) => e.id)).toEqual(["save", "end-run"]);
    expect(filterEntries(entries(), "every move").map((e) => e.id)).toEqual(["regret"]);
    expect(filterEntries(entries(), "zzz")).toEqual([]);
  });
});

describe("isPaletteChord", () => {
  it("is Ctrl+K or Cmd+K, not Ctrl+Shift+K", () => {
    const ev = (o: Partial<KeyboardEvent>) => ({ ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, key: "k", ...o }) as KeyboardEvent;
    expect(isPaletteChord(ev({ ctrlKey: true }))).toBe(true);
    expect(isPaletteChord(ev({ metaKey: true, key: "K" }))).toBe(true);
    expect(isPaletteChord(ev({}))).toBe(false);
    expect(isPaletteChord(ev({ ctrlKey: true, shiftKey: true }))).toBe(false);
  });
});

describe("CommandPalette", () => {
  it("renders nothing while closed", () => {
    render(<CommandPalette open={false} entries={entries()} onClose={() => {}} />);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("filters as you type and Enter runs the highlighted command, then closes", () => {
    const run = vi.fn();
    const onClose = vi.fn();
    render(<CommandPalette open entries={entries(run)} onClose={onClose} />);
    const box = screen.getByRole("combobox");
    fireEvent.change(box, { target: { value: "sav" } });
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual(["Save"]);
    fireEvent.keyDown(box, { key: "Enter" });
    expect(run).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("will not run a disabled command and shows its reason", () => {
    const onClose = vi.fn();
    render(<CommandPalette open entries={entries()} onClose={onClose} />);
    const box = screen.getByRole("combobox");
    fireEvent.change(box, { target: { value: "search" } });
    const opt = screen.getByRole("option");
    expect(opt).toHaveAttribute("aria-disabled", "true");
    expect(opt.textContent).toContain("only during a combat");
    fireEvent.keyDown(box, { key: "Enter" });
    expect(onClose).not.toHaveBeenCalled();
  });

  it("arrow keys move the cursor and Escape closes", () => {
    const onClose = vi.fn();
    render(<CommandPalette open entries={entries()} onClose={onClose} />);
    const box = screen.getByRole("combobox");
    fireEvent.keyDown(box, { key: "ArrowDown" });
    expect(screen.getAllByRole("option")[1]).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(box, { key: "Escape" });
    expect(onClose).toHaveBeenCalledOnce();
  });
});
