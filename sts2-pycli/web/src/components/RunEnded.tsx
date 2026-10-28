import type { Step } from "../api";
import { RunEndScene } from "./GameOver";
import "../skins/runend.css";

/** Summary shown right after a run is ended (the red "End run" button) — an
 *  abandoned mid-run, or a run closed after game over. Shares GameOver's
 *  run-end scene but leads with clear next actions so the user isn't left
 *  fumbling with the Live/Replay toggle. `onReviewReplay` is only offered
 *  when a recorded run file exists (`logPath`). */
export default function RunEnded({ step, logPath, onNewRun, onReviewReplay, onCoachReview }: {
  step: Step;
  logPath: string | null;
  onNewRun: () => void;
  onReviewReplay: () => void;
  /** Open the Coach review (the autopsy, on a death). */
  onCoachReview?: () => void;
}) {
  const s = step.state;
  const over = s.decision === "game_over";
  const victory = s.victory;
  const heading = over ? (victory ? "Victory!" : "Defeat") : "Run ended";
  const tone = over ? (victory ? "victory" : "defeat") : "ended";
  return (
    <RunEndScene
      step={step}
      heading={heading}
      tone={tone}
      primaryAction={
        <button className="gs-runend-btn gs-runend-btn--primary" onClick={onNewRun}>
          <span className="gs-runend-btnface">New run</span>
        </button>
      }
      secondaryActions={(logPath || onCoachReview) && (
        <>
          {logPath && (
            <button className="gs-runend-btn gs-runend-btn--secondary" onClick={onReviewReplay}>
              <span className="gs-runend-btnface">Review replay</span>
            </button>
          )}
      {onCoachReview && (
        <button className="gs-runend-btn gs-runend-btn--secondary" onClick={onCoachReview}>
          <span className="gs-runend-btnface">Coach review</span>
        </button>
      )}
        </>
      )}
    />
  );
}
