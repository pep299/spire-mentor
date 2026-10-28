import { test, expect, type Page } from "@playwright/test";
import { menuItem } from "../e2e/topbarMenu";

/**
 * Follow mode shows what the real-game advisor graded (coach plan 5): the
 * turn / choice records it wrote into the run log land on their steps
 * (`log_idx`), tick the timeline, open in the Coach review, and a `withdraw`
 * the advisor appends later takes its record away on the next poll.
 *
 * The mod-bundled build mirrors the advisor's log into the page's run store
 * (`liveLog.ts`, covered by its unit test); this build has no control plane,
 * so the spec puts the log there itself — the same `POST /api/runs/put` — and
 * opens the mod's follow link. Everything after the put is the real path:
 * the wasm parses the log, `GET /api/coach` serves its records, and the
 * follow poll re-reads it when the stored log changes.
 */

const RUN = "coach-e2e.jsonl";

function eventState(step: number, floor: number) {
  return {
    step, ts: "t", type: "state",
    data: {
      type: "decision", decision: "event_choice", floor, act: 1, act_name: "UNDERDOCKS",
      event_name: "NEOW", character_id: "IRONCLAD",
      player: { name: "Ironclad", hp: 64, max_hp: 80, block: 0, gold: 99, relics: [], potions: [], deck: [] },
      options: [
        { index: 0, action: "event_choice", label: "Small Capsule", description: "Obtain a random Relic.", enabled: true, event_id: "NEOW" },
        { index: 1, action: "event_choice", label: "Arcane Scroll", description: "Obtain a random Rare Card.", enabled: true, event_id: "NEOW" },
      ],
    },
  };
}

const CHOICE = {
  kind: "choice", id: "choice:n1", node_id: "n1", decision: "event_choice", label_kind: "Event",
  verdict: "flag", strength: "strong preference",
  played: { label: "Arcane Scroll", prob: 0.08, rank: 2 }, top: { label: "Small Capsule", prob: 0.81 },
  alternatives: [], peeked: false, log_name: RUN, log_idx: 0,
};
const TURN = {
  kind: "turn", id: "turn:n2", node_id: "n2", turn: 2, floor: 2, verdict: "blunder",
  you: ["Strike (Cultist)", "end turn"], ai: ["Bash (Cultist)", "Defend", "end turn"],
  hp_drop: 9.2, win_before: 0.96, win_after: 0.94, peeked: false, ungraded_actions: 0,
  log_name: RUN, log_idx: 1, severity: 9.2,
  context: { hand: ["Strike", "Bash", "Defend"], energy: 3, max_energy: 3, hp: 60, max_hp: 80, block: 0,
    enemies: [{ name: "Cultist", hp: 48, intent: "attack 6×2 + debuff" }] },
};

const LINES = [
  { step: 0, ts: "t", type: "meta", data: { character: "Ironclad", seed: "E2ECOACH", ascension: 0 } },
  eventState(1, 1),
  eventState(2, 2),
  // Written late, after a later state, as the advisor does: attached by log_idx.
  { step: 2, ts: "t", type: "coach", data: CHOICE },
  eventState(3, 3),
  { step: 3, ts: "t", type: "coach", data: TURN },
];

async function putLog(page: Page, lines: unknown[]): Promise<void> {
  const status = await page.evaluate(async ({ name, text }) => {
    const f = (window as unknown as { __sts2ApiFetch: typeof fetch }).__sts2ApiFetch;
    const r = await f("/api/runs/put", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ path: name, text }),
    });
    return r.status;
  }, { name: RUN, text: lines.map((l) => JSON.stringify(l)).join("\n") + "\n" });
  expect(status).toBe(200);
}

test("follow mode renders, ticks, reviews and withdraws the advisor's coach records", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByLabel("seed", { exact: true })).toBeVisible({ timeout: 60_000 });
  await putLog(page, LINES);

  // The mod's coach link: the run by name, at the turn's step.
  await page.goto(`/?follow=1&run=${RUN}&step=1`);
  const turnChip = page.getByTestId("turn-chip");
  await expect(turnChip).toBeVisible({ timeout: 60_000 });
  await expect(turnChip).toContainText("Turn 2 · you: Strike (Cultist) → end turn");
  await expect(turnChip).toContainText("≈ 9 HP more lost · win 96% → 94%");
  // Both graded steps are ticked on the timeline.
  await expect(page.getByTestId("coach-tick-0")).toBeVisible();
  await expect(page.getByTestId("coach-tick-1")).toBeVisible();

  // Step 0 carries the flagged choice, in agreement wording.
  await page.getByTestId("coach-tick-0").click();
  const choiceChip = page.getByTestId("choice-record-chip");
  await expect(choiceChip).toContainText("Event · you: Arcane Scroll");
  await expect(choiceChip).toContainText("AI would take: Small Capsule (strong preference) · Arcane Scroll was its 2nd choice");
  await expect(turnChip).toHaveCount(0);

  // The advisor withdraws the choice (an undo in the game): the next follow
  // poll drops it from the step and the timeline, without moving the view.
  await putLog(page, [...LINES, { step: 3, ts: "t", type: "coach", data: { kind: "withdraw", ids: ["choice:n1"] } }]);
  await expect(choiceChip).toHaveCount(0, { timeout: 20_000 });
  await expect(page.getByTestId("coach-tick-0")).toHaveCount(0);
  await expect(page.getByTestId("coach-tick-1")).toBeVisible();

  // The Coach review lists the turn with its position; its row seeks there.
  await (await menuItem(page, "Analyze", /coach review/)).click();
  const dock = page.getByRole("region", { name: "Coach Review" });
  await expect(dock).toBeVisible();
  await expect(dock.getByText("Cultist intends: attack 6×2 + debuff")).toBeVisible();
  await dock.getByRole("button", { name: /blunder/ }).click();
  await expect(turnChip).toBeVisible();
});
