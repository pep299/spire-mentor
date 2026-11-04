import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Reasoning from "../Reasoning";
import type { Enemy, Score } from "../../api";

function score(over: Partial<Score> = {}): Score {
  return {
    stable_idx: 0, action: "Strike", win_prob: 0.5, q: 1.22, visits: 400,
    is_best: false, q_saturated: false, low_confidence: false, target_index: null,
    wins: 417, losses: 0, ties: 0, ...over,
  };
}

function enemy(over: Partial<Enemy> & { index: number; name: string }): Enemy {
  return { hp: 30, max_hp: 30, block: 0, image: null, ...over };
}

describe("Reasoning panel", () => {
  it("names the engine's tie band and the reason for its pick (GH #1022)", () => {
    render(<Reasoning pv={null} scores={[
      score({ stable_idx: 2, action: "Inflame+", q: 1.20, is_best: true, in_band: true, pick_rank: 0,
        pick_reason: "the most searched of the equal moves" }),
      score({ stable_idx: 0, action: "Strike", q: 1.21, in_band: true, pick_rank: 1 }),
      score({ stable_idx: 1, action: "Defend", q: 0.90, pick_rank: 2 }),
    ]} />);
    expect(screen.getByTestId("tie-band")).toHaveTextContent(
      "≈ about equal: Inflame+ ≈ Strike · ★ the most searched of the equal moves");
  });

  it("says nothing about a band that holds only the pick", () => {
    render(<Reasoning pv={null} scores={[
      score({ stable_idx: 0, action: "Bash", q: 1.2, is_best: true, in_band: true, pick_rank: 0 }),
      score({ stable_idx: 1, action: "Defend", q: 0.9, pick_rank: 1 }),
    ]} />);
    expect(screen.queryByTestId("tie-band")).toBeNull();
  });

  it("says the call is too close when the top two Q values are inside search noise", () => {
    // Bug report wb-e34ef8b6039bc0f3 showed the panel printing a confident #1
    // on a margin barely above its own seed-to-seed spread. Under the bar, the
    // panel says so and drops the bold that reads as "this is the move".
    render(<Reasoning pv={null} scores={[
      score({ stable_idx: 0, action: "Strike", q: 1.2121, is_best: true }),
      score({ stable_idx: 1, action: "Tremble", q: 1.2100 }),
    ]} />);
    expect(screen.getByText(/too close to call/)).toBeInTheDocument();
    expect(screen.getByText("Strike")).toHaveStyle({ fontWeight: "400" });
  });

  it("stays quiet, and keeps the bold, when the top two are separated", () => {
    render(<Reasoning pv={null} scores={[
      score({ stable_idx: 0, action: "Strike", q: 1.30, is_best: true }),
      score({ stable_idx: 1, action: "Tremble", q: 1.20 }),
    ]} />);
    expect(screen.queryByText(/too close to call/)).toBeNull();
    expect(screen.getByText("Strike")).toHaveStyle({ fontWeight: "700" });
  });

  it("ranks by Q, not row order — win_prob is 1.0 for every action in an easy fight", () => {
    // The rows arrive sorted by win_prob, so in a fight the player always wins
    // the incoming order says nothing. Reading gap off row order would miss a
    // near-tie whenever the runner-up is not row 2.
    render(<Reasoning pv={null} scores={[
      score({ stable_idx: 0, action: "Strike", q: 1.2121, is_best: true }),
      score({ stable_idx: 1, action: "end turn", q: 1.1000 }),
      score({ stable_idx: 2, action: "Tremble", q: 1.2110 }),
    ]} />);
    expect(screen.getByText(/too close to call/)).toBeInTheDocument();
  });

  it("names the target instead of printing its slot index", () => {
    render(<Reasoning pv={null} enemies={[enemy({ index: 0, name: "Cultist" })]}
      scores={[score({ target_index: 0 })]} />);
    expect(screen.getByText("Strike → Cultist")).toBeInTheDocument();
  });

  it("disambiguates duplicate enemy names by their position among the duplicates", () => {
    render(<Reasoning pv={null}
      enemies={[enemy({ index: 0, name: "Louse" }), enemy({ index: 1, name: "Louse" }),
                enemy({ index: 2, name: "Fungi Beast" })]}
      scores={[score({ stable_idx: 0, target_index: 1 }), score({ stable_idx: 1, target_index: 2 })]} />);
    expect(screen.getByText("Strike → Louse #2")).toBeInTheDocument();
    expect(screen.getByText("Strike → Fungi Beast")).toBeInTheDocument();
  });

  it("falls back to the index when no enemy list is supplied", () => {
    render(<Reasoning pv={null} scores={[score({ target_index: 1 })]} />);
    expect(screen.getByText("Strike → enemy 1")).toBeInTheDocument();
  });

  it("carries the whole line as a tooltip, for the line that still clamps", () => {
    // The label wraps to a second line before it ellipsizes, but a long move
    // with a target ("drink Potion of Binding → Gremlin Nob #2") can still run
    // past two lines in a 320px panel.
    render(<Reasoning pv={null} scores={[score({ action: "drink Potion of Binding", visits: 397 })]} />);
    const label = screen.getByText("drink Potion of Binding");
    expect(label).toHaveAttribute("title", expect.stringContaining("drink Potion of Binding"));
    expect(label.getAttribute("title")).toContain("Q +1.22");
    expect(label.getAttribute("title")).toContain("397 visits");
    expect(label.getAttribute("title")).toContain("417W/0L/0T");
  });

  it("marks the best line in the tooltip too", () => {
    render(<Reasoning pv={null} scores={[score({ is_best: true })]} />);
    expect(screen.getByText("Strike").getAttribute("title")).toContain("best");
  });

  it("lays the rows out as one grid so the numbers take only what they need", () => {
    // The numeric columns used to be fixed widths (44 + 54 + 30 + 72 px) that
    // ate 228 of the panel's 288, leaving the label nine characters — every
    // card read as "Sweeping…" beside half-empty number columns. As grid
    // tracks they size to their content and the label keeps the rest, wrapping
    // to a second line before it truncates.
    render(<Reasoning pv={null} scores={[score({ action: "Sweeping Beam" })]} />);
    const label = screen.getByText("Sweeping Beam");
    expect(label.parentElement).toHaveStyle({ display: "grid" });
    expect(label.style.getPropertyValue("-webkit-line-clamp")).toBe("2");
    for (const cell of Array.from(label.parentElement!.children)) {
      // Only the visit bar keeps a width; every text column is content-sized.
      if (cell !== label && (cell as HTMLElement).textContent) {
        expect((cell as HTMLElement).style.width).toBe("");
      }
    }
  });

  it("shows a forced move as not searched, not as an even fight", () => {
    // The run-AI skips the search when only one action is legal, and records
    // that move alone with zero visits. Its zero Q and 0/0/0 are "no data",
    // so the row must not print them.
    render(<Reasoning pv={null} scores={[
      score({ stable_idx: -1, action: "end turn", q: 0, win_prob: 0, visits: 0,
        wins: 0, losses: 0, ties: 0, is_best: true }),
    ]} />);
    expect(screen.getByText("forced · not searched")).toBeInTheDocument();
    expect(screen.queryByText("+0.00")).toBeNull();
    expect(screen.queryByText("0/0/0")).toBeNull();
  });

  it("gives the principal variation a tooltip so a long line stays readable", () => {
    render(<Reasoning pv={["Strike", "Defend", "end turn"]} scores={[]} />);
    expect(screen.getByText(/^pv:/)).toHaveAttribute("title", "Strike → Defend → end turn");
  });
});
