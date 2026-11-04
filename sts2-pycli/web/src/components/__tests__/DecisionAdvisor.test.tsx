import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { DecisionAnalyzeButton, DecisionWinChip } from "../DecisionAdvisor";
import { useDecisionAdvisor, type DecisionAdvisorState } from "../useDecisionAdvisor";
import { HintsFixture } from "../../hints/HintsContext";
import { type DecisionAdvisorCandidate } from "../../api";

function cand(over: Partial<DecisionAdvisorCandidate> = {}): DecisionAdvisorCandidate {
  return {
    key: "option:0",
    label: "Opt",
    win_rate: 0,
    wins: 0,
    n: 12,
    avg_final_hp: 0,
    avg_floors_advanced: 4.2,
    avg_hp_at_deepest: 30,
    value: 0.05,
    delta: 0.01,
    is_best: true,
    inconclusive: 0,
    ...over,
  };
}

function adv(anyWin: boolean): DecisionAdvisorState {
  return {
    enabled: true,
    advising: false,
    advice: null,
    prog: null,
    elapsed: 0,
    byKey: new Map(),
    anyWin,
    run: () => {},
  };
}

describe("DecisionWinChip", () => {
  it("leads with the act win-rate when at least one option cleared the act", () => {
    render(<DecisionWinChip c={cand({ win_rate: 0.25 })} adv={adv(true)} />);
    expect(screen.getByText(/25% act win/)).toBeInTheDocument();
    // win-rate mode keeps the depth/HP detail on the sub-line
    expect(screen.getByText(/~4\.2f · 30 hp/)).toBeInTheDocument();
  });

  it("falls back to depth + HP when no option cleared the act (the all-zero case)", () => {
    render(<DecisionWinChip c={cand({ win_rate: 0 })} adv={adv(false)} />);
    // no misleading "0% act win" headline — show how far the option got instead
    expect(screen.queryByText(/% act win/)).not.toBeInTheDocument();
    expect(screen.getByText(/~4\.2 floors deeper/)).toBeInTheDocument();
    expect(screen.getByText(/30 hp at the wall/)).toBeInTheDocument();
  });
});

/** Mounts the hook the way Shop / EventScreen do, so the gate is tested once
 *  for every screen that uses it. */
function AdvisorHarness() {
  const advisor = useDecisionAdvisor("screen-1", true);
  return <DecisionAnalyzeButton adv={advisor} />;
}

describe("useDecisionAdvisor hints gate", () => {
  it("offers no EV analysis while hints are off", () => {
    render(
      <HintsFixture on={false}>
        <AdvisorHarness />
      </HintsFixture>,
    );
    expect(screen.queryByRole("button", { name: /analyze options/i })).toBeNull();
  });

  it("offers it once hints are on", () => {
    render(
      <HintsFixture on>
        <AdvisorHarness />
      </HintsFixture>,
    );
    expect(screen.getByRole("button", { name: /analyze options/i })).toBeInTheDocument();
  });
});
