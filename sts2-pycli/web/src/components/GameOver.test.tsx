import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Step } from "../api";
import GameOver from "./GameOver";

function mkStep(victory: boolean): Step {
  return {
    idx: 0,
    decision: "game_over",
    action_label: "",
    state: { decision: "game_over", victory, floor: 51, player: { hp: 0, max_hp: 80 } },
  } as unknown as Step;
}

describe("GameOver", () => {
  it("offers Review replay in Play so a finished run can be re-opened", () => {
    // Regression: the live game-over screen only had "New run", which quit the
    // session and threw the log path away — so a player who reached a natural
    // run end had no visible route back into their own run.
    const onReviewReplay = vi.fn();
    render(<GameOver step={mkStep(true)} onNewRun={() => {}} onReviewReplay={onReviewReplay} />);
    fireEvent.click(screen.getByRole("button", { name: /review replay/i }));
    expect(onReviewReplay).toHaveBeenCalledTimes(1);
  });

  it("offers the Coach review on a death", () => {
    const onCoachReview = vi.fn();
    render(<GameOver step={mkStep(false)} onNewRun={() => {}} onCoachReview={onCoachReview} />);
    fireEvent.click(screen.getByRole("button", { name: /coach review/i }));
    expect(onCoachReview).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("button", { name: /review replay/i })).toBeNull();
  });

  it("hides Review replay in Replay mode (no handler)", () => {
    render(<GameOver step={mkStep(false)} />);
    expect(screen.queryByRole("button", { name: /review replay/i })).toBeNull();
  });
});
