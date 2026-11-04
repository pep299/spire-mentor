import { afterEach, describe, expect, it } from "vitest";
import { endedSelection } from "./textSelection";

function row(text: string): HTMLElement {
  const el = document.createElement("div");
  el.textContent = text;
  document.body.appendChild(el);
  return el;
}

describe("endedSelection", () => {
  afterEach(() => {
    getSelection()?.removeAllRanges();
    document.body.innerHTML = "";
  });

  it("is false with no selection or a collapsed one (a plain click)", () => {
    const el = row("Play Strike");
    expect(endedSelection(el)).toBe(false);
    getSelection()!.collapse(el.firstChild!, 2);
    expect(endedSelection(el)).toBe(false);
  });

  it("is true when selected text lies in the element", () => {
    const el = row("Play Strike");
    getSelection()!.selectAllChildren(el);
    expect(endedSelection(el)).toBe(true);
  });

  it("is false when the selection is in another element", () => {
    const el = row("Play Strike");
    getSelection()!.selectAllChildren(row("End turn"));
    expect(endedSelection(el)).toBe(false);
  });
});
