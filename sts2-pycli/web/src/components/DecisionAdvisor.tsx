import type { DecisionAdvisorCandidate } from "../api";
import type { DecisionAdvisorState } from "./useDecisionAdvisor";

/**
 * Shared "Analyze options (EV)" advisor for the live non-combat-play decision
 * screens — in-combat card selects (Knowledge Daemon's pick, Headbutt…), shops,
 * and events. The deep, slow sibling of the card-reward advisor: it forks the
 * live run and Monte-Carlos the rest of the act out for every option (MCTS in
 * each fight), so it runs only on demand. Each screen renders the button +
 * progress block where it likes and joins {@link DecisionWinChip} onto its own
 * options by the candidate `key` (`"choice:0"` / `"card:1"` / `"option:2"` …).
 *
 * Play mode only — the analysis needs the live in-process run to fork. Replay
 * passes ``enabled = false`` and these render nothing.
 */

/** The "🎲 analyze options (EV)" trigger. Renders nothing in replay. */
export function DecisionAnalyzeButton({ adv, style }: { adv: DecisionAdvisorState; style?: React.CSSProperties }) {
  if (!adv.enabled) return null;
  return (
    <button
      className="mono"
      onClick={adv.run}
      disabled={adv.advising}
      title="Deep EV analysis: fork the run and play the rest of the act out many times for each option (MCTS in every fight). Slow — a few seconds."
      style={{ fontSize: 11, ...style }}
    >
      {adv.advising ? "analyzing…" : "🎲 analyze options (EV)"}
    </button>
  );
}

/** Progress bar + per-option counts while running, then the error / summary line. */
export function DecisionAnalyzeStatus({ adv }: { adv: DecisionAdvisorState }) {
  if (!adv.enabled) return null;
  const { advising, prog, elapsed, advice } = adv;
  return (
    <>
      {advising && (
        <div className="mono" style={{ marginTop: 10, fontSize: 11, color: "var(--muted)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span>analyzing options…{prog ? ` ${prog.done}/${prog.total} sims` : ""}</span>
            <span>{elapsed.toFixed(1)}s</span>
          </div>
          <div style={{ height: 6, background: "var(--field-bg)", borderRadius: 3, overflow: "hidden" }}>
            <div
              style={{
                height: "100%",
                width: prog && prog.total ? `${(prog.done / prog.total) * 100}%` : "0%",
                background: "var(--accent-strong)",
                transition: "width 0.2s ease",
              }}
            />
          </div>
          {prog && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 14, marginTop: 6 }}>
              {prog.cands.map((c) => {
                const n = prog.perCand.get(c.key) ?? 0;
                const full = n >= prog.rollouts;
                return (
                  <span key={c.key} style={{ color: full ? "var(--accent-strong)" : "var(--muted)" }}>
                    {c.label} {n}/{prog.rollouts}
                  </span>
                );
              })}
            </div>
          )}
        </div>
      )}
      {advice?.error && (
        <div className="mono" style={{ color: "var(--hp)", marginTop: 10, fontSize: 11 }}>
          advisor unavailable: {advice.error}
        </div>
      )}
      {advice && !advice.error && (
        <div className="mono" style={{ color: "var(--muted)", marginTop: 10, fontSize: 11 }}>
          {adv.anyWin
            ? `act win-rate over ${advice.candidates[0]?.n ?? "?"} rollouts of the rest of act ${advice.act}`
            : `depth reached + HP at the boss over ${advice.candidates[0]?.n ?? "?"} rollouts of the rest of act ${advice.act} (no option cleared the act)`}
          {" "}· ★ = recommended
        </div>
      )}
    </>
  );
}

/** Per-option advice chip, keyed onto a rendered option by its candidate key.
 *  Green + ★ on the recommended option. Mirrors CardReward's AdviceChip.
 *
 *  Leads with the act win-rate only when at least one option actually cleared
 *  the act ({@link DecisionAdvisorState.anyWin}). Otherwise — the common case
 *  early in a run, where the cheap rollout policy rarely beats the act boss and
 *  every option reads 0% — it falls back to the depth/HP signal the ranking is
 *  really built on (`avg_floors_advanced` + `avg_hp_at_deepest`), so a screen of
 *  all-zero win-rates still shows a meaningful, discriminating number. */
export function DecisionWinChip({ c, adv }: { c: DecisionAdvisorCandidate; adv: DecisionAdvisorState }) {
  const star = c.is_best ? "★ " : "";
  const col = c.is_best ? "oklch(68% 0.13 145)" : "var(--muted)";
  const floors = c.avg_floors_advanced.toFixed(1);
  const hp = c.avg_hp_at_deepest.toFixed(0);
  const headline = adv.anyWin
    ? `${star}${(c.win_rate * 100).toFixed(0)}% act win`
    : `${star}~${floors} floors deeper`;
  const sub = adv.anyWin
    ? `Δ ${c.delta >= 0 ? "+" : ""}${(c.delta * 100).toFixed(0)} · ~${floors}f · ${hp} hp`
    : `${hp} hp at the wall`;
  return (
    <div
      className="mono"
      title={
        (adv.anyWin
          ? `act win-rate over ${c.n} rollouts of the rest of the act (MCTS in combat). `
          : `no option cleared the act in ${c.n} rollouts (the act boss is a wall for the ` +
            `cheap rollout policy), so this shows how far the option got instead. `) +
        `~${c.avg_floors_advanced.toFixed(1)} floors advanced, ${hp} HP at the deepest room. ` +
        `Δ is this option's composite value (act-win + depth reached + HP carried) ` +
        `minus the best other option; ★ = recommended.`
      }
      style={{ fontSize: 11, marginTop: 5, lineHeight: 1.45, color: col, fontWeight: c.is_best ? 600 : 400 }}
    >
      <div>{headline}</div>
      <div style={{ color: "var(--muted)", fontWeight: 400 }}>{sub}</div>
    </div>
  );
}
