import { render, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { useShortcuts, type ShortcutContext } from "./useShortcuts";
import type { Step } from "./api";

function Harness(props: Partial<ShortcutContext>) {
  const ctx: ShortcutContext = {
    mode: "replay", live: null, step: null, count: 0, idx: 0,
    onScrub: () => {}, onRecommend: () => {}, onChoose: () => {},
    onPlayBest: () => {}, onPlayToEnd: () => {}, autoRunning: false,
    onStart: () => {}, onToggleHelp: () => {}, ...props,
  };
  useShortcuts(ctx);
  return <input data-testid="seed" />;
}

function choiceStep(): Step {
  return {
    idx: 0, decision: "event_choice", scores: [], pv: null, chosen_action: null,
    source: "live", meta: { turn: null, step_no: 0 },
    state: {
      decision: "event_choice", round: 0, energy: 0, max_energy: 0,
      draw_pile_count: 0, discard_pile_count: 0, exhaust_pile_count: 0,
      player: { name: "Ironclad", hp: 70, max_hp: 80, block: 0 }, enemies: [], hand: [],
      options: [
        { label: "Take gold", action: "choose_option", args: { option_index: 0 }, enabled: true },
        { label: "Locked", action: "choose_option", args: { option_index: 1 }, enabled: false },
        { label: "Leave", action: "choose_option", args: { option_index: 2 }, enabled: true },
      ],
    },
  };
}

describe("useShortcuts", () => {
  it("'?' toggles help in any mode", () => {
    const onToggleHelp = vi.fn();
    render(<Harness onToggleHelp={onToggleHelp} />);
    fireEvent.keyDown(document.body, { key: "?" });
    expect(onToggleHelp).toHaveBeenCalled();
  });

  it("replay: arrows and Home/End scrub", () => {
    const onScrub = vi.fn();
    render(<Harness mode="replay" count={5} idx={2} onScrub={onScrub} />);
    fireEvent.keyDown(document.body, { key: "ArrowRight" });
    fireEvent.keyDown(document.body, { key: "ArrowLeft" });
    fireEvent.keyDown(document.body, { key: "Home" });
    fireEvent.keyDown(document.body, { key: "End" });
    expect(onScrub.mock.calls).toEqual([[1], [-1], ["home"], ["end"]]);
  });

  it("play choice screen: number triggers the Nth enabled option", () => {
    const onChoose = vi.fn();
    render(<Harness mode="play" live={choiceStep()} onChoose={onChoose} />);
    fireEvent.keyDown(document.body, { key: "2" }); // 2nd ENABLED = "Leave" (option_index 2)
    expect(onChoose).toHaveBeenCalledWith("choose_option", { option_index: 2 });
  });

  it("play choice screen: g / a take the recommended best option", () => {
    const onPlayBest = vi.fn();
    render(<Harness mode="play" live={choiceStep()} onPlayBest={onPlayBest} />);
    fireEvent.keyDown(document.body, { key: "g" });
    fireEvent.keyDown(document.body, { key: "a" });
    expect(onPlayBest).toHaveBeenCalledTimes(2);
  });

  it("play choice screen: p peeks", () => {
    const onPeek = vi.fn();
    render(<Harness mode="play" live={choiceStep()} onPeek={onPeek} />);
    fireEvent.keyDown(document.body, { key: "p" });
    expect(onPeek).toHaveBeenCalledTimes(1);
  });

  it("play choice screen: f plays the run to the end", () => {
    const onPlayToEnd = vi.fn();
    render(<Harness mode="play" live={choiceStep()} onPlayToEnd={onPlayToEnd} />);
    fireEvent.keyDown(document.body, { key: "f" });
    expect(onPlayToEnd).toHaveBeenCalled();
  });

  it("play: keys are swallowed while an autoplay loop is running", () => {
    const onPlayBest = vi.fn();
    const onPlayToEnd = vi.fn();
    const onChoose = vi.fn();
    render(<Harness mode="play" live={choiceStep()} autoRunning
      onPlayBest={onPlayBest} onPlayToEnd={onPlayToEnd} onChoose={onChoose} />);
    fireEvent.keyDown(document.body, { key: "g" });
    fireEvent.keyDown(document.body, { key: "f" });
    fireEvent.keyDown(document.body, { key: "1" });
    expect(onPlayBest).not.toHaveBeenCalled();
    expect(onPlayToEnd).not.toHaveBeenCalled();
    expect(onChoose).not.toHaveBeenCalled();
  });

  it("play combat: hook is inert (Board owns combat keys)", () => {
    const onChoose = vi.fn();
    const combat = { ...choiceStep(), decision: "combat_play" };
    combat.state = { ...combat.state, decision: "combat_play" };
    render(<Harness mode="play" live={combat} onChoose={onChoose} />);
    fireEvent.keyDown(document.body, { key: "1" });
    expect(onChoose).not.toHaveBeenCalled();
  });

  it("ignores shortcuts while typing, but Enter starts a run from the form", () => {
    const onChoose = vi.fn();
    const onStart = vi.fn();
    const { getByTestId } = render(<Harness mode="play" live={null} onChoose={onChoose} onStart={onStart} />);
    const input = getByTestId("seed");
    fireEvent.keyDown(input, { key: "1" });     // typing -> ignored
    expect(onChoose).not.toHaveBeenCalled();
    fireEvent.keyDown(input, { key: "Enter" });  // Enter -> start
    expect(onStart).toHaveBeenCalled();
  });
});
