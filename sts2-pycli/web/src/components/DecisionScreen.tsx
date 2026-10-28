import type { Step, ChooseHandler } from "../api";
import CardReward from "./CardReward";
import Shop from "./Shop";
import EventScreen from "./EventScreen";
import RestSite from "./RestSite";
import CardSelect from "./CardSelect";
import BundleSelect from "./BundleSelect";
import GameOver from "./GameOver";
import ChoicePanel from "./ChoicePanel";

/**
 * Routes a non-combat decision step to its bespoke screen, in both modes.
 * ``onChoose`` is provided in Play (interactive) and omitted in Replay
 * (read-only). ``combat_play`` and ``map_select`` are handled by the caller
 * (the interactive Board, and the connected MapView/MapChoices respectively).
 */
export default function DecisionScreen({
  step,
  onChoose,
  onNewRun,
  onReviewReplay,
  onCoachReview,
}: {
  step: Step;
  onChoose?: ChooseHandler;
  onNewRun?: () => void;
  onReviewReplay?: () => void;
  onCoachReview?: () => void;
}) {
  switch (step.decision) {
    case "card_reward":
      return <CardReward step={step} onChoose={onChoose} />;
    case "shop":
      return <Shop step={step} onChoose={onChoose} />;
    case "event_choice":
      return <EventScreen step={step} onChoose={onChoose} />;
    case "rest_site":
      return <RestSite step={step} onChoose={onChoose} />;
    case "card_select":
      return <CardSelect step={step} onChoose={onChoose} />;
    case "bundle_select":
      return <BundleSelect step={step} onChoose={onChoose} />;
    case "game_over":
      return <GameOver step={step} onNewRun={onNewRun} onReviewReplay={onReviewReplay} onCoachReview={onCoachReview} />;
    default:
      // Unknown decision: the generic action list keeps the run unstuck (Play),
      // or a muted note (Replay, where there are no actions to take).
      return onChoose ? (
        <ChoicePanel title={step.action_label || step.decision} options={step.state.options ?? []} onChoose={onChoose} />
      ) : (
        <div className="mono" style={{ padding: 20, color: "var(--muted)" }}>
          {step.decision} — {step.action_label || "(no dedicated view)"}
        </div>
      );
  }
}
