import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import Board from "../Board";
import type { Step } from "../../api";

/** Two enemies, so a single-target card has a real choice to make. */
function combatStep(enemyCount = 2): Step {
  return {
    idx: 0, decision: "combat_play", scores: [], pv: null, chosen_action: null,
    source: "live", meta: { turn: 1, step_no: 0 },
    state: {
      decision: "combat_play", round: 1, energy: 3, max_energy: 3,
      draw_pile_count: 5, discard_pile_count: 0, exhaust_pile_count: 0,
      player: { name: "Ironclad", hp: 70, max_hp: 80, block: 0 },
      enemies: Array.from({ length: enemyCount }, (_, i) => ({
        index: i, name: `Cultist ${i}`, hp: 50, max_hp: 50, block: 0, image: null,
      })),
      hand: [
        { index: 0, id: "STRIKE", name: "Strike", cost: 1, type: "Attack",
          description: "", can_play: true, target_type: "AnyEnemy", image: null },
      ],
    },
  } as unknown as Step;
}

beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
afterEach(() => vi.useRealTimers());

describe("the targeting arrow", () => {
  it("is not drawn until a card is waiting for a target", () => {
    render(<Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}} />);
    expect(screen.queryByTestId("aim-arrow")).toBeNull();
  });

  it("follows the cursor once a card is picked", () => {
    // The game hangs a fat arrow off the chosen card and lets you aim it. Ours
    // only ever appeared AFTER the play, so live targeting had no aim at all.
    const { container } = render(
      <Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}} />,
    );
    fireEvent.click(screen.getByText("Strike"));
    expect(screen.getByText(/select a target/i)).toBeInTheDocument();
    // The arrow needs a cursor position before it can point anywhere.
    expect(screen.queryByTestId("aim-arrow")).toBeNull();
    fireEvent.mouseMove(container.firstElementChild!, { clientX: 300, clientY: 200 });
    expect(screen.getByTestId("aim-arrow")).toBeInTheDocument();
  });

  it("stops aiming when the target is chosen", () => {
    const onPlayCard = vi.fn();
    const { container } = render(
      <Board step={combatStep()} onPlayCard={onPlayCard} onEndTurn={() => {}} />,
    );
    fireEvent.click(screen.getByText("Strike"));
    fireEvent.mouseMove(container.firstElementChild!, { clientX: 300, clientY: 200 });
    expect(screen.getByTestId("aim-arrow")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "1" });     // pick enemy 1 by hotkey
    expect(onPlayCard).toHaveBeenCalledWith(0, 0);
    expect(screen.queryByTestId("aim-arrow")).toBeNull();
  });

  it("Ctrl / Cmd chords are not combat hotkeys", () => {
    // Ctrl+A used to start autoplay instead of selecting the log to copy it.
    const onAutoplay = vi.fn();
    const onPlayCard = vi.fn();
    render(<Board step={combatStep()} onPlayCard={onPlayCard} onEndTurn={() => {}}
      onAutoplay={onAutoplay} />);
    fireEvent.keyDown(window, { key: "a", ctrlKey: true });
    fireEvent.keyDown(window, { key: "a", metaKey: true });
    fireEvent.keyDown(window, { key: "1", ctrlKey: true });
    expect(onAutoplay).not.toHaveBeenCalled();
    expect(screen.queryByText(/select a target/i)).toBeNull();
    fireEvent.keyDown(window, { key: "a" });
    expect(onAutoplay).toHaveBeenCalledTimes(1);
  });

  it("Escape cancels the aim", () => {
    const { container } = render(
      <Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}} />,
    );
    fireEvent.click(screen.getByText("Strike"));
    fireEvent.mouseMove(container.firstElementChild!, { clientX: 300, clientY: 200 });
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByTestId("aim-arrow")).toBeNull();
  });

  it("leaves nothing behind once the card resolves", () => {
    // The game calls StopDrawing() the instant a card is played. We used to
    // hold a card->enemy arrow for ~0.9s afterwards, which collided with the
    // character->enemy attack arrow and put two arrows on the board at once.
    // What got hit is now told by the damage number and the hurt animation.
    render(<Board step={combatStep(1)} onPlayCard={() => {}} onEndTurn={() => {}} />);
    fireEvent.click(screen.getByText("Strike"));
    expect(screen.queryByTestId("play-arrow")).toBeNull();
    expect(screen.queryByTestId("aim-arrow")).toBeNull();
  });

  it("draws at most one arrow at a time", () => {
    const { container } = render(
      <Board step={combatStep()} onPlayCard={() => {}} onEndTurn={() => {}}
        attack={{ target: 0, nonce: 1 }} />,
    );
    fireEvent.click(screen.getByText("Strike"));
    fireEvent.mouseMove(container.firstElementChild!, { clientX: 300, clientY: 200 });
    expect(container.querySelectorAll("svg > g[data-testid]")).toHaveLength(1);
  });

  it("draws nothing in replay, where there is no card to aim", () => {
    render(<Board step={combatStep()} />);
    expect(screen.queryByTestId("aim-arrow")).toBeNull();
    expect(screen.queryByTestId("play-arrow")).toBeNull();
  });
});
