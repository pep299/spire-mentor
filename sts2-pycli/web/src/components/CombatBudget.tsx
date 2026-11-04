import NumberInput from "./NumberInput";

/** Combat search preset: a named pair of (monster sims, boss sims). */
export type CombatPreset = "quick" | "benchmark";

/** The sim budget the two presets stand for. `benchmark` is the offline
 *  win-rate sweep's budget, so a run at it is comparable to a benchmarked
 *  number; `quick` trades that away for iteration speed. */
const PRESET_SIMS: Record<CombatPreset, { monster: number; boss: number }> = {
  quick: { monster: 200, boss: 800 },
  benchmark: { monster: 1000, boss: 4000 },
};

/**
 * The combat search budget controls, rendered both before a run starts and
 * while one is live.
 *
 * Sharing one component between the two is the point: the live copy is what
 * makes "this run is going badly — stop, search harder, resume" possible, and
 * a second hand-built copy of the same four inputs is how the two drift into
 * disagreeing about what a preset means.
 *
 * The live copy passes `pending` + `onApply`. Budget changes are NOT posted as
 * you type: `NumberInput` commits on every keystroke, so typing `1000` over
 * `200` would post 1, 10, 100 and 1000 in turn, and a step landing mid-word
 * would search at whichever prefix arrived. An explicit apply also answers
 * "did that take?" — the button is the diff.
 */
export default function CombatBudget({
  preset, monsterSims, bossSims, threads,
  onPreset, onMonsterSims, onBossSims,
  disabled = false, pending = null, onApply,
}: {
  preset: CombatPreset;
  monsterSims: number;
  bossSims: number;
  threads: number;
  onPreset: (preset: CombatPreset) => void;
  onMonsterSims: (sims: number) => void;
  onBossSims: (sims: number) => void;
  disabled?: boolean;
  /** Live runs only: the budget in the boxes when it differs from the one the
   *  server is actually searching at, else null. Drives the apply button. */
  pending?: { monster: number; boss: number } | null;
  onApply?: () => void;
}) {
  return (
    <>
      <label title="Combat search budget. Benchmark = the offline win-rate sweep's budget (1000 monster / 4000 boss) — the default, so a run here is comparable to a benchmarked number. Quick = cheap sims for fast iteration (200 monster / 800 boss).">
        combat{" "}
        <select
          value={preset}
          disabled={disabled}
          onChange={(e) => {
            const v = e.target.value as CombatPreset;
            onPreset(v);
            onMonsterSims(PRESET_SIMS[v].monster);
            onBossSims(PRESET_SIMS[v].boss);
          }}
        >
          <option value="quick">Quick</option>
          <option value="benchmark">Benchmark</option>
        </select>
      </label>
      <label title="MCTS iterations per move in regular (non-boss) combat — the benchmark's --sims">
        monster sims{" "}
        <NumberInput min={1} max={20000} step={250} value={monsterSims} onChange={onMonsterSims}
          width={72} disabled={disabled} ariaLabel="monster combat sims" />
      </label>
      <label title="MCTS iterations per move in boss combat — the benchmark's --boss-sims (boss fights search deeper)">
        boss sims{" "}
        <NumberInput min={1} max={50000} step={1000} value={bossSims} onChange={onBossSims}
          width={72} disabled={disabled} ariaLabel="boss combat sims" />
      </label>
      {/* Shown, disabled, and labelled as inert rather than removed. The server
          hands this to `sts2_rs.HeuristicRunAgent`, whose constructor discards
          it — play-to-end combat search is single-threaded whatever this says.
          Deleting the box would leave older screenshots and notes claiming a
          setting with nothing to contradict them. */}
      <label title="NOT WIRED on this path — HeuristicRunAgent ignores it, so play-to-end combat search is always single-threaded. (The recommend panel's threads knob is a different code path and does work.) Root-parallel search also SPLITS the sim budget across workers, so turning it on is not free: ~5.8pp of deep-oracle agreement at 1000 sims / 8 workers. See docs/experiments/2026-08-10-root-parallel-decision-quality.md.">
        <span style={{ opacity: 0.5 }}>
          threads{" "}
          <NumberInput min={1} max={16} step={1} value={threads} onChange={() => {}}
            width={48} disabled ariaLabel="root-parallel MCTS workers (not wired)" />
          {" "}<span style={{ fontSize: 10 }}>(n/a)</span>
        </span>
      </label>
      {pending && onApply && (
        <button type="button" onClick={onApply}
          title="apply this budget to the running search — takes effect from the next decision"
          style={{ border: "1px solid var(--accent)", borderRadius: 4, background: "var(--bg)",
            color: "var(--accent)", cursor: "pointer", font: "inherit", padding: "1px 8px" }}>
          ↻ apply {pending.monster}/{pending.boss}
        </button>
      )}
    </>
  );
}
