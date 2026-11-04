//! Fork-at-the-fatal-fight loss diagnosis (spec
//! `docs/superpowers/specs/2026-07-17-boss-loss-diagnosis-design.md`).
//!
//! Replays a losing corpus seed deterministically, keeps a rolling clone of
//! the [`RunManager`](sts2_core::run_manager::RunManager) at each combat
//! entry, and — once the run ends lost — re-plays the last (fatal) fight K
//! times per arm with a perturbed MCTS seed. Arms differ in sims budget and
//! (via the process-global `STS2_RUNAI_HONEST` env, set by the `fork-fatal`
//! CLI before any threads spawn) clairvoyance.

use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// One diagnosis arm. Honest/clairvoyant is NOT applied here — the
/// `fork-fatal` CLI sets `STS2_RUNAI_HONEST` process-wide from
/// [`ForkArm::clairvoyant`] before spawning rayon workers, because the
/// combat policy reads the env per decision. One arm per process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForkArm {
    /// Deployed config: honest BeliefDraw, eval-cheap (1000/4000).
    Deployed,
    /// Honest, 10x sims (10000/40000) — the search-budget axis.
    Honest10x,
    /// Clairvoyant (`STS2_RUNAI_HONEST=0`), eval-cheap — the information axis.
    Clair,
    /// Clairvoyant + 10x — the winnable-at-entry ceiling.
    Clair10x,
}

impl ForkArm {
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            ForkArm::Deployed => "deployed",
            ForkArm::Honest10x => "honest10x",
            ForkArm::Clair => "clair",
            ForkArm::Clair10x => "clair10x",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "deployed" => Ok(ForkArm::Deployed),
            "honest10x" => Ok(ForkArm::Honest10x),
            "clair" => Ok(ForkArm::Clair),
            "clair10x" => Ok(ForkArm::Clair10x),
            other => Err(format!(
                "unknown arm {other:?} (expected deployed|honest10x|clair|clair10x)"
            )),
        }
    }

    /// `(sims, boss_sims)` overrides vs the frozen eval-cheap budget.
    #[must_use]
    pub fn sims_override(self) -> (Option<u32>, Option<u32>) {
        match self {
            ForkArm::Deployed | ForkArm::Clair => (None, None),
            ForkArm::Honest10x | ForkArm::Clair10x => (Some(10_000), Some(40_000)),
        }
    }

    #[must_use]
    pub fn clairvoyant(self) -> bool {
        matches!(self, ForkArm::Clair | ForkArm::Clair10x)
    }
}

/// One fork replay rep (a JSONL line in the campaign output).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForkRepRow {
    pub seed: i32,
    pub arm: String,
    pub rep: u32,
    pub fatal_act: u8,
    pub fatal_floor: i32,
    pub encounter: String,
    pub entry_hp: i32,
    pub survived: bool,
    pub end_hp: i32,
    pub steps: usize,
    pub wall_ms: u64,
    /// Per-decision trace (opt-in via `fork-fatal --trace`); `None` rows
    /// serialize without the field, so untraced outputs are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<Vec<TraceStep>>,
}

/// One combat decision in a traced fork rep — the player-visible state at
/// decision time. Both arms see everything here (intents are public);
/// the honest/clairvoyant split is draw-pile ORDER only, which is exactly
/// why action-level traces are a fair cross-arm contrast.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceStep {
    pub round: u16,
    pub hp: i32,
    pub block: i32,
    pub energy: i32,
    pub draw_n: usize,
    pub discard_n: usize,
    /// Card tokens currently in hand (`CARD.X` / `CARD.X+`).
    pub hand: Vec<String>,
    /// `"<MONSTER.X>:<hp>+<block>b:<intent>"` per living enemy.
    pub enemies: Vec<String>,
    /// Chosen action, e.g. `Play:CARD.Bash+@MONSTER.Crusher`,
    /// `Potion:POTION.FirePotion`, `EndTurn`, `Resolve:1`.
    pub action: String,
}

/// Replay-fidelity check row (deployed-config prefix vs the corpus record).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FidelityRow {
    pub seed: i32,
    pub ok: bool,
    pub corpus_won: bool,
    pub replay_won: bool,
    pub corpus_killed_by: String,
    pub replay_killed_by: String,
    pub corpus_steps: usize,
    pub replay_steps: usize,
}

/// Minimal per-run summary read from a corpus `<seed>.run.gz`.
#[derive(Clone, Debug)]
pub struct CorpusRun {
    pub seed: i32,
    pub won: bool,
    pub killed_by: String,
    pub steps: usize,
    /// Ascension the run was recorded at (`sts2sim.meta.ascension`). The
    /// prefix replay MUST reconstruct the run at this level: `new_run`
    /// was A0-hardwired until 2026-08-19, which made every seed of an
    /// A8/A10 corpus "fidelity drift" (the replay played a different
    /// game and compared step counts against the record).
    pub ascension: u8,
}

/// Read `seed/won/killed_by_encounter/sts2sim.meta.steps` from every
/// `<seed>.run.gz` in `dir`. Non-run files are skipped by name shape.
pub fn load_corpus_summary(dir: &Path) -> std::io::Result<Vec<CorpusRun>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let p = entry?.path();
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(stem) = name.strip_suffix(".run.gz") else {
            continue;
        };
        let Ok(seed) = stem.parse::<i32>() else {
            continue;
        };
        let mut buf = String::new();
        flate2::read::GzDecoder::new(std::fs::File::open(&p)?).read_to_string(&mut buf)?;
        let v: serde_json::Value = serde_json::from_str(&buf)
            .map_err(|e| std::io::Error::other(format!("{name}: {e}")))?;
        #[allow(clippy::cast_possible_truncation)]
        let ascension = v["sts2sim"]["meta"]["ascension"]
            .as_u64()
            .or_else(|| v["ascension"].as_u64())
            .unwrap_or(0) as u8;
        out.push(CorpusRun {
            seed,
            won: v["win"].as_bool().unwrap_or(false),
            killed_by: v["killed_by_encounter"]
                .as_str()
                .unwrap_or("NONE.NONE")
                .to_string(),
            steps: v["sts2sim"]["meta"]["steps"].as_u64().unwrap_or(0) as usize,
            ascension,
        });
    }
    out.sort_by_key(|r| r.seed);
    Ok(out)
}

/// Deterministic per-rep search-seed perturbation. The combat MCTS derives
/// its per-decision seed from the policy's `run_seed` (`policy/mod.rs`
/// `mcts_pick`), so a distinct `run_seed` per rep gives an independent
/// search stream while the cloned game state (piles, enemy rolls at entry)
/// stays identical.
#[must_use]
pub fn fork_seed(base: i32, rep: u32) -> i32 {
    // splitmix-style scramble, truncated; +1 so rep 0 differs from `base`.
    let x = i64::from(base) ^ ((i64::from(rep) + 1).wrapping_mul(0x9E37_79B9));
    (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 17) as i32
}

use sts2_core::full_run::RunPolicy;
use sts2_core::run_manager::{ActionResult, Phase, RunAction, RunManager};
use sts2_core::CombatState;

use crate::arena::{
    build_policy, parse_character, CellSpec, CombatBudgetKind, PolicyKind,
    DEFAULT_LIVELOCK_STALL_LIMIT, DEFAULT_MAX_STEPS,
};
use crate::trajectory::recorder::{card_token, monster_token, potion_token};

/// Compact intent label: damage-carrying variants keep their payload
/// (`ATTACK:12`, `MULTI_ATTACK:5x3`, `DEATH_BLOW:40`, `DEFEND:8`); the
/// rest are the bare parity tag (`BUFF`, `STUN`, …).
fn intent_label(intent: sts2_core::monsters::Intent) -> String {
    use sts2_core::monsters::Intent;
    let name = intent.python_name();
    match intent {
        Intent::MultiAttack { damage, hits } => format!("{name}:{damage}x{hits}"),
        Intent::Attack { damage } | Intent::DeathBlow { damage } => format!("{name}:{damage}"),
        Intent::Defend { block } => format!("{name}:{block}"),
        _ => name.to_string(),
    }
}

fn creature_token(combat: &CombatState, uid: u32) -> String {
    combat
        .creatures
        .iter()
        .find(|c| c.id.get() == uid)
        .map_or_else(|| format!("uid{uid}"), |c| monster_token(c.monster_id.0))
}

/// Chosen-action label for a trace step (see [`TraceStep::action`]).
fn trace_action_label(combat: &CombatState, action: &RunAction) -> String {
    match action {
        RunAction::PlayCard { hand_idx, target } => {
            let card = combat
                .player
                .hand
                .get(*hand_idx)
                .and_then(|uid| combat.cards.iter().find(|c| c.uid == *uid))
                .map_or_else(|| "?".to_string(), |c| card_token(c.card_id, c.upgraded));
            match target {
                Some(t) => format!("Play:{card}@{}", creature_token(combat, *t)),
                None => format!("Play:{card}"),
            }
        }
        RunAction::EndTurn => "EndTurn".to_string(),
        RunAction::DrinkPotion {
            potion_slot,
            target,
        } => {
            let potion = combat
                .player
                .potions
                .get(*potion_slot)
                .map_or_else(|| "?".to_string(), |k| potion_token(*k));
            match target {
                Some(t) => format!("Potion:{potion}@{}", creature_token(combat, *t)),
                None => format!("Potion:{potion}"),
            }
        }
        RunAction::CombatResolveChoice { option } => match option {
            Some(i) => format!("Resolve:{i}"),
            None => "Resolve:skip".to_string(),
        },
        RunAction::CombatResolveChoiceTakeFirst { count } => format!("ResolveFirst:{count}"),
        other => format!("{other:?}"),
    }
}

/// Snapshot the player-visible combat state + the chosen action.
fn trace_step(combat: &CombatState, action: &RunAction) -> TraceStep {
    let hand = combat
        .player
        .hand
        .iter()
        .filter_map(|uid| combat.cards.iter().find(|c| c.uid == *uid))
        .map(|c| card_token(c.card_id, c.upgraded))
        .collect();
    let enemies = combat
        .enemy_uids
        .iter()
        .filter_map(|uid| combat.creatures.iter().find(|c| c.id == uid.0))
        .filter(|c| c.current_hp > 0)
        .map(|c| {
            let intent = c.intent.map_or_else(|| "None".to_string(), intent_label);
            format!(
                "{}:{}+{}b:{intent}",
                monster_token(c.monster_id.0),
                c.current_hp,
                c.block
            )
        })
        .collect();
    TraceStep {
        round: combat.round,
        hp: combat.player.current_hp,
        block: combat.player.block,
        energy: combat.player.energy,
        draw_n: combat.player.draw.len(),
        discard_n: combat.player.discard.len(),
        hand,
        enemies,
        action: trace_action_label(combat, action),
    }
}

/// Cloned engine state at entry to the run's fatal fight.
#[derive(Debug)]
pub struct FatalSnapshot {
    pub rm: RunManager,
    pub fatal_act: u8,
    pub fatal_floor: i32,
    pub encounter: String,
    pub entry_hp: i32,
}

/// What the deployed-config replay produced (compared to the corpus record).
#[derive(Debug)]
pub struct FidelitySummary {
    pub won: bool,
    pub killed_by: String,
    pub steps: usize,
}

fn base_spec(seed: i32, arm: ForkArm) -> CellSpec {
    let (sims, boss_sims) = arm.sims_override();
    CellSpec {
        seed,
        character: parse_character("ironclad").expect("ironclad parses"),
        ascension: 0,
        policy: PolicyKind::Ppo,
        combat: CombatBudgetKind::EvalCheap,
        sims_override: sims,
        boss_sims_override: boss_sims,
        max_steps: DEFAULT_MAX_STEPS,
        livelock_stall_limit: DEFAULT_LIVELOCK_STALL_LIMIT,
        act1: FORK_ACT1,
        audit: false,
        ablate: None,
    }
}

/// The Act-1 model the fork corpora were recorded on (GH #544).
///
/// `base_spec` and `new_run` BOTH read this constant rather than each naming
/// a variant, so the replay path cannot drift from the spec path — the exact
/// failure mode that broke deckmut replay fidelity in 2026-08-19, when a
/// hand-rolled copy of the arena constructor fell behind.
///
/// Every existing corpus predates the Underdocks port, hence Overgrowth.
/// A corpus recorded on another act needs this threaded through
/// `CorpusRun` instead of pinned here.
const FORK_ACT1: crate::arena::Act1Choice = crate::arena::Act1Choice::Overgrowth;

fn new_run(seed: i32, ascension: u8) -> RunManager {
    // Mirror arena::run_cell EXACTLY by calling the same constructor it
    // does — a hand-rolled copy here silently fell behind when the
    // interactive deck-mutation flags were added to the arena path, which
    // broke replay fidelity on every seed of a deckmut-champion corpus
    // (2026-08-19). Also thread the corpus run's ascension
    // (see `CorpusRun::ascension`): this was A0-hardwired before.
    crate::rollout::ensure_content_installed();
    crate::arena::new_arena_run_manager(
        seed,
        parse_character("ironclad").expect("ironclad parses"),
        ascension,
        FORK_ACT1,
    )
}

fn combat_encounter_lead(rm: &RunManager) -> String {
    rm.combat
        .as_ref()
        .and_then(|combat| {
            combat
                .enemy_uids
                .iter()
                .filter_map(|uid| combat.creatures.iter().find(|c| c.id == uid.0))
                .map(|c| monster_token(c.monster_id.0))
                .next()
        })
        .unwrap_or_else(|| "NONE.NONE".to_string())
}

/// One combat-entry snapshot from a corpus replay (the label factory's
/// unit of work — `fork_label` plays reps from these).
#[derive(Debug)]
pub struct FightSnapshot {
    pub rm: RunManager,
    pub act: u8,
    pub floor: i32,
    /// 0-based index of this fight within the run's combat sequence.
    pub fight_idx: usize,
    pub encounter: String,
    pub entry_hp: i32,
}

/// Deterministically replay `seed` under the DEPLOYED config, cloning a
/// snapshot at EVERY combat entry (wins allowed — the caller compares
/// [`FidelitySummary`] against the corpus record). Shared driver behind
/// [`replay_to_fatal`] (diagnosis) and `fork_label` (label factory).
pub fn replay_all_fights(
    seed: i32,
    ascension: u8,
) -> Result<(Vec<FightSnapshot>, FidelitySummary), String> {
    let mut policy = build_policy(&base_spec(seed, ForkArm::Deployed));
    let mut rm = new_run(seed, ascension);

    let mut steps = 0usize;
    let mut fights: Vec<FightSnapshot> = Vec::new();
    let mut in_combat = false;
    let mut last_killed_by = "NONE.NONE".to_string();
    // Stall watchdog, same contract as drive_full_run: without it a
    // surfaced-no-op loop (e.g. Sozu ShopBuyPotion, GH #270) grinds
    // toward DEFAULT_MAX_STEPS (~hours) instead of failing in seconds.
    let mut last_hash = sts2_core::hashing::run_hash(&rm);
    let mut stall = 0usize;

    loop {
        if rm.is_over() || steps >= DEFAULT_MAX_STEPS {
            break;
        }
        policy.prepare(&mut rm);
        let actions = rm.get_available_actions();
        if actions.is_empty() {
            return Err(format!("seed {seed}: empty actions at step {steps}"));
        }
        let Some(idx) = policy.choose(&rm, &actions) else {
            return Err(format!("seed {seed}: policy halt at step {steps}"));
        };
        match rm.take_action(actions[idx]) {
            ActionResult::Advanced => {}
            // NOT counted in `steps` — drive_full_run breaks before its
            // increment, and fidelity compares against the recorder's count.
            ActionResult::RunOver => break,
            other => return Err(format!("seed {seed}: action rejected: {other:?}")),
        }
        steps += 1;

        let h = sts2_core::hashing::run_hash(&rm);
        if h == last_hash {
            stall += 1;
            if stall >= DEFAULT_LIVELOCK_STALL_LIMIT {
                return Err(format!(
                    "seed {seed}: livelock at step {steps} (GH #270 class)"
                ));
            }
        } else {
            stall = 0;
            last_hash = h;
        }

        let now_in_combat =
            rm.phase() == Phase::Combat && rm.combat.as_ref().is_some_and(|c| !c.is_over);
        if now_in_combat && !in_combat {
            let encounter = combat_encounter_lead(&rm);
            last_killed_by.clone_from(&encounter);
            fights.push(FightSnapshot {
                rm: rm.clone(),
                act: rm.run_state.act_index,
                floor: rm.run_state.total_floor,
                fight_idx: fights.len(),
                encounter,
                entry_hp: rm.run_state.current_hp,
            });
        }
        in_combat = now_in_combat;
    }

    let won = rm.run_state.player_won;
    let fidelity = FidelitySummary {
        won,
        killed_by: if won {
            "NONE.NONE".to_string()
        } else {
            last_killed_by
        },
        steps,
    };
    Ok((fights, fidelity))
}

/// Deterministically replay `seed` under the DEPLOYED config and return
/// the snapshot at entry to the FATAL fight (the last combat before
/// death). Errors if the replay wins (no fatal fight to fork —
/// corpus/replay drift) or never enters combat.
pub fn replay_to_fatal(
    seed: i32,
    ascension: u8,
) -> Result<(FatalSnapshot, FidelitySummary), String> {
    let (mut fights, fidelity) = replay_all_fights(seed, ascension)?;
    if fidelity.won {
        return Err(format!("seed {seed}: replay WON — corpus/replay drift"));
    }
    let last = fights
        .pop()
        .ok_or_else(|| format!("seed {seed}: lost without combat"))?;
    Ok((
        FatalSnapshot {
            rm: last.rm,
            fatal_act: last.act,
            fatal_floor: last.floor,
            encounter: last.encounter,
            entry_hp: last.entry_hp,
        },
        fidelity,
    ))
}

/// Controlled fight-entry mutation for the arrival-state counterfactual
/// probe (spec `2026-07-18-arrival-state-counterfactual-probe-design.md`).
/// Each arm isolates one arrival axis; `All` bounds the combined
/// arrival-state headroom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryMutation {
    /// Player healed to max HP (via `creature_heal`, so HP-threshold
    /// hooks — RedSkull etc. — fire faithfully).
    Heal,
    /// Empty belt slots filled with FirePotion (strong, universally
    /// usable targeted damage).
    Potions,
    /// Every Curse-type card removed from all combat piles.
    Decurse,
    /// Heal + Potions + Decurse together.
    All,
}

impl EntryMutation {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "heal" => Ok(Self::Heal),
            "potions" => Ok(Self::Potions),
            "decurse" => Ok(Self::Decurse),
            "all" => Ok(Self::All),
            other => Err(format!(
                "unknown mutation {other:?} (expected heal|potions|decurse|all)"
            )),
        }
    }

    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            Self::Heal => "heal",
            Self::Potions => "potions",
            Self::Decurse => "decurse",
            Self::All => "all",
        }
    }
}

/// Apply `m` to a fight-entry snapshot IN PLACE (once per snapshot,
/// before the fork reps — mutations are deterministic, so every rep
/// forks the same mutated state). Fight-scoped: run-level mirrors sync
/// at `exit_combat` as usual.
pub fn apply_entry_mutation(snap: &mut FatalSnapshot, m: EntryMutation) {
    use sts2_core::cards::CardType;
    let combat = snap
        .rm
        .combat
        .as_mut()
        .expect("fork snapshot is taken at fight entry");
    if matches!(m, EntryMutation::Heal | EntryMutation::All) {
        if let Some(uid) = combat.player.creature_uid {
            let deficit = combat.player.max_hp;
            let _ = combat.creature_heal(uid, deficit); // clamps at max
        }
        snap.entry_hp = combat.player.current_hp;
    }
    if matches!(m, EntryMutation::Potions | EntryMutation::All) {
        let cap = if combat.player.potion_slots == 0 {
            usize::try_from(snap.rm.run_state.max_potion_slots.max(0)).unwrap_or(0)
        } else {
            combat.player.potion_slots as usize
        };
        while combat.player.potions.len() < cap {
            combat
                .player
                .potions
                .push(sts2_core::potions::PotionKind::FirePotion);
        }
    }
    if matches!(m, EntryMutation::Decurse | EntryMutation::All) {
        let curse_uids: Vec<_> = combat
            .cards
            .iter()
            .filter(|c| c.card_type == CardType::Curse)
            .map(|c| c.uid)
            .collect();
        for uid in curse_uids {
            let _ = combat.remove_card_from_piles(uid);
        }
    }
}

/// Play the snapshotted fatal fight once under `arm` with a rep-perturbed
/// search seed, stopping as soon as the fight resolves (combat over or run
/// over). The cloned game state is identical across reps; only the MCTS
/// search stream varies.
pub fn fork_fight(
    snap: &FatalSnapshot,
    arm: ForkArm,
    rep: u32,
    base_seed: i32,
    trace: bool,
) -> ForkRepRow {
    /// A single combat never legitimately needs this many decisions.
    const FORK_STEP_CAP: usize = 3000;
    let started = std::time::Instant::now();
    let mut rm = snap.rm.clone();
    let mut policy = build_policy(&base_spec(fork_seed(base_seed, rep), arm));

    let mut steps = 0usize;
    let mut trace_steps: Option<Vec<TraceStep>> = trace.then(Vec::new);
    loop {
        let combat_live = rm.combat.as_ref().is_some_and(|c| !c.is_over);
        if rm.is_over() || !combat_live || steps >= FORK_STEP_CAP {
            break;
        }
        policy.prepare(&mut rm);
        let actions = rm.get_available_actions();
        if actions.is_empty() {
            break;
        }
        let Some(idx) = policy.choose(&rm, &actions) else {
            break;
        };
        if let (Some(ts), Some(combat)) = (trace_steps.as_mut(), rm.combat.as_ref()) {
            ts.push(trace_step(combat, &actions[idx]));
        }
        if !matches!(
            rm.take_action(actions[idx]),
            ActionResult::Advanced | ActionResult::RunOver
        ) {
            break;
        }
        steps += 1;
    }

    ForkRepRow {
        seed: base_seed,
        arm: arm.token().to_string(),
        rep,
        fatal_act: snap.fatal_act,
        fatal_floor: snap.fatal_floor,
        encounter: snap.encounter.clone(),
        entry_hp: snap.entry_hp,
        survived: rm.run_state.current_hp > 0,
        end_hp: rm.run_state.current_hp,
        steps,
        wall_ms: started.elapsed().as_millis() as u64,
        trace: trace_steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_parse_roundtrip() {
        for a in [
            ForkArm::Deployed,
            ForkArm::Honest10x,
            ForkArm::Clair,
            ForkArm::Clair10x,
        ] {
            assert_eq!(ForkArm::parse(a.token()).unwrap(), a);
        }
        assert!(ForkArm::parse("bogus").is_err());
    }

    #[test]
    fn arm_sims_overrides() {
        assert_eq!(ForkArm::Deployed.sims_override(), (None, None));
        assert_eq!(
            ForkArm::Honest10x.sims_override(),
            (Some(10_000), Some(40_000))
        );
        assert_eq!(ForkArm::Clair.sims_override(), (None, None));
        assert_eq!(
            ForkArm::Clair10x.sims_override(),
            (Some(10_000), Some(40_000))
        );
        assert!(ForkArm::Clair.clairvoyant() && ForkArm::Clair10x.clairvoyant());
        assert!(!ForkArm::Deployed.clairvoyant() && !ForkArm::Honest10x.clairvoyant());
    }

    #[test]
    fn fork_seeds_distinct_and_stable() {
        let s: Vec<i32> = (0..16).map(|r| fork_seed(1234, r)).collect();
        let uniq: std::collections::HashSet<_> = s.iter().collect();
        assert_eq!(uniq.len(), 16);
        assert_eq!(fork_seed(1234, 3), fork_seed(1234, 3));
        assert_ne!(fork_seed(1234, 0), 1234); // never collides with the corpus stream
    }

    #[test]
    fn intent_labels_keep_damage_payloads() {
        use sts2_core::monsters::Intent;
        assert_eq!(intent_label(Intent::Attack { damage: 12 }), "ATTACK:12");
        assert_eq!(
            intent_label(Intent::MultiAttack { damage: 5, hits: 3 }),
            "MULTI_ATTACK:5x3"
        );
        assert_eq!(intent_label(Intent::Defend { block: 8 }), "DEFEND:8");
        assert_eq!(intent_label(Intent::Stun), "STUN");
    }

    #[test]
    fn rep_row_trace_field_is_optional_on_the_wire() {
        let row = ForkRepRow {
            seed: 1,
            arm: "deployed".into(),
            rep: 0,
            fatal_act: 3,
            fatal_floor: 40,
            encounter: "MONSTER.Aeonglass".into(),
            entry_hp: 60,
            survived: false,
            end_hp: 0,
            steps: 10,
            wall_ms: 5,
            trace: None,
        };
        // Untraced rows keep the pre-trace schema byte-for-byte…
        let js = serde_json::to_string(&row).unwrap();
        assert!(!js.contains("trace"));
        // …and pre-trace rows (no field at all) still deserialize.
        let back: ForkRepRow = serde_json::from_str(&js).unwrap();
        assert!(back.trace.is_none());

        let traced = ForkRepRow {
            trace: Some(vec![TraceStep {
                round: 1,
                hp: 60,
                block: 0,
                energy: 3,
                draw_n: 5,
                discard_n: 0,
                hand: vec!["CARD.Strike".into()],
                enemies: vec!["MONSTER.Aeonglass:200+0b:ATTACK:12".into()],
                action: "EndTurn".into(),
            }]),
            ..row
        };
        let js = serde_json::to_string(&traced).unwrap();
        let back: ForkRepRow = serde_json::from_str(&js).unwrap();
        assert_eq!(back.trace.unwrap().len(), 1);
    }

    #[test]
    fn corpus_summary_reads_real_file() {
        // In-repo corpus data; skip gracefully if a checkout lacks it. The
        // data (`*.run.gz`) is not committed, so a fresh checkout has no dir
        // and skips. Guard on parsed-row COUNT, not just dir existence: a
        // stray empty/partial dir (e.g. a leftover from local experiments)
        // otherwise slips past the existence check and hard-fails the
        // invariant asserts below with `got 0` on machines that lack the
        // actual corpus.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/run-trajectories/ppo-ironclad-a0");
        if !dir.exists() {
            eprintln!("corpus dir missing; skipping");
            return;
        }
        let rows = load_corpus_summary(&dir).unwrap();
        if rows.is_empty() {
            eprintln!("corpus dir present but holds no *.run.gz rows; skipping");
            return;
        }
        assert!(rows.len() >= 900, "expected ~1000 runs, got {}", rows.len());
        let r1 = rows.iter().find(|r| r.seed == 1).unwrap();
        assert!(!r1.won);
        assert_eq!(r1.steps, 198);
    }

    // ── synthetic-corpus fixtures ──────────────────────────────────────

    /// Per-test scratch dir. Nextest runs one process per test, so the
    /// pid alone keeps parallel tests from colliding.
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("sts2-fork-diagnosis-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("scratch dir");
        d
    }

    fn write_gz(dir: &Path, name: &str, body: &str) {
        use std::io::Write;
        let f = std::fs::File::create(dir.join(name)).expect("create scratch file");
        let mut enc = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
        enc.write_all(body.as_bytes()).expect("gz write");
        enc.finish().expect("gz finish");
    }

    fn run_json(win: bool, killed_by: &str, steps: u64) -> String {
        serde_json::json!({
            "win": win,
            "killed_by_encounter": killed_by,
            "sts2sim": { "meta": { "steps": steps } },
        })
        .to_string()
    }

    #[test]
    fn corpus_summary_skips_non_run_files_and_sorts_by_seed() {
        let dir = scratch_dir("summary");
        write_gz(
            &dir,
            "10.run.gz",
            &run_json(false, "MONSTER.Aeonglass", 198),
        );
        write_gz(&dir, "2.run.gz", &run_json(true, "NONE.NONE", 300));
        write_gz(&dir, "-5.run.gz", &run_json(false, "MONSTER.Nibbit", 7));
        // Missing keys fall back to the documented defaults.
        write_gz(&dir, "7.run.gz", "{}");
        // Skipped by name shape, not by content.
        write_gz(&dir, "report.jsonl", "definitely not a run file");
        write_gz(&dir, "champion.run.gz", "also not a run file");
        std::fs::create_dir_all(dir.join("shards")).expect("subdir");

        let rows = load_corpus_summary(&dir).expect("summary");
        let seeds: Vec<i32> = rows.iter().map(|r| r.seed).collect();
        assert_eq!(seeds, vec![-5, 2, 7, 10], "rows must come back seed-sorted");

        let ten = &rows[3];
        assert!(!ten.won);
        assert_eq!(ten.killed_by, "MONSTER.Aeonglass");
        assert_eq!(ten.steps, 198);
        assert!(rows[1].won && rows[1].steps == 300);
        // `{}` → won=false, killed_by="NONE.NONE", steps=0.
        assert!(!rows[2].won);
        assert_eq!(rows[2].killed_by, "NONE.NONE");
        assert_eq!(rows[2].steps, 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corpus_summary_names_the_offending_file_on_bad_json() {
        let dir = scratch_dir("badjson");
        write_gz(&dir, "3.run.gz", "{not json at all");
        let err = load_corpus_summary(&dir).expect_err("malformed run must error");
        assert!(
            err.to_string().contains("3.run.gz"),
            "error must name the file: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn entry_mutation_parse_roundtrip() {
        for m in [
            EntryMutation::Heal,
            EntryMutation::Potions,
            EntryMutation::Decurse,
            EntryMutation::All,
        ] {
            assert_eq!(EntryMutation::parse(m.token()).unwrap(), m);
        }
        let err = EntryMutation::parse("healz").expect_err("unknown arm");
        assert!(err.contains("heal|potions|decurse|all"), "{err}");
    }

    #[test]
    fn base_spec_carries_the_arm_budget_and_the_frozen_deployed_config() {
        let deployed = base_spec(77, ForkArm::Deployed);
        assert_eq!(deployed.seed, 77);
        assert_eq!(deployed.ascension, 0);
        assert_eq!(deployed.policy, PolicyKind::Ppo);
        assert_eq!(deployed.combat, CombatBudgetKind::EvalCheap);
        assert_eq!(deployed.sims_override, None);
        assert_eq!(deployed.boss_sims_override, None);
        assert!(!deployed.audit && deployed.ablate.is_none());

        // The 10x arms differ ONLY in the sims overrides — same policy,
        // same budget kind, same livelock/step caps.
        let ten = base_spec(77, ForkArm::Clair10x);
        assert_eq!(ten.sims_override, Some(10_000));
        assert_eq!(ten.boss_sims_override, Some(40_000));
        assert_eq!(ten.combat, deployed.combat);
        assert_eq!(ten.max_steps, deployed.max_steps);
        assert_eq!(ten.livelock_stall_limit, deployed.livelock_stall_limit);
    }

    // ── live combat-entry fixtures ─────────────────────────────────────

    /// Drive a fresh run with a trivial first-legal-action driver up to
    /// its FIRST combat entry — the same state shape `replay_all_fights`
    /// snapshots, without the PPO/MCTS cost of the real replay.
    fn fight_entry(seed: i32) -> RunManager {
        let mut rm = new_run(seed, 0);
        for step in 0..400 {
            if rm.phase() == Phase::Combat && rm.combat.as_ref().is_some_and(|c| !c.is_over) {
                return rm;
            }
            let actions = rm.get_available_actions();
            assert!(!actions.is_empty(), "no legal actions at step {step}");
            assert!(matches!(rm.take_action(actions[0]), ActionResult::Advanced));
        }
        panic!("seed {seed} never entered combat");
    }

    fn snapshot_at_entry(seed: i32) -> FatalSnapshot {
        let rm = fight_entry(seed);
        FatalSnapshot {
            fatal_act: rm.run_state.act_index,
            fatal_floor: rm.run_state.total_floor,
            encounter: combat_encounter_lead(&rm),
            entry_hp: rm.run_state.current_hp,
            rm,
        }
    }

    fn living_enemies(combat: &CombatState) -> Vec<&sts2_core::Creature> {
        combat
            .enemy_uids
            .iter()
            .filter_map(|uid| combat.creatures.iter().find(|c| c.id == uid.0))
            .filter(|c| c.current_hp > 0)
            .collect()
    }

    #[test]
    fn encounter_lead_is_the_first_enemy_and_falls_back_outside_combat() {
        let fresh = new_run(4242, 0);
        assert!(fresh.combat.is_none());
        assert_eq!(combat_encounter_lead(&fresh), "NONE.NONE");

        let rm = fight_entry(4242);
        let combat = rm.combat.as_ref().expect("in combat");
        let first = combat
            .creatures
            .iter()
            .find(|c| c.id == combat.enemy_uids[0].0)
            .expect("lead enemy");
        let lead = combat_encounter_lead(&rm);
        assert_eq!(lead, monster_token(first.monster_id.0));
        assert!(lead.starts_with("MONSTER."), "{lead}");
    }

    #[test]
    fn trace_action_labels_name_the_card_potion_and_target() {
        let mut rm = fight_entry(4242);
        let enemy_uid = {
            let combat = rm.combat.as_ref().expect("in combat");
            living_enemies(combat)[0].id.get()
        };
        rm.combat
            .as_mut()
            .expect("in combat")
            .player
            .potions
            .insert(0, sts2_core::potions::PotionKind::FirePotion);

        let combat = rm.combat.as_ref().expect("in combat");
        let enemy_tok = creature_token(combat, enemy_uid);
        let hand0 = combat
            .cards
            .iter()
            .find(|c| c.uid == combat.player.hand[0])
            .expect("hand card");
        let card_tok = card_token(hand0.card_id, hand0.upgraded);
        let potion_tok = potion_token(sts2_core::potions::PotionKind::FirePotion);

        let label = |a: RunAction| trace_action_label(combat, &a);
        assert_eq!(
            label(RunAction::PlayCard {
                hand_idx: 0,
                target: Some(enemy_uid)
            }),
            format!("Play:{card_tok}@{enemy_tok}")
        );
        assert_eq!(
            label(RunAction::PlayCard {
                hand_idx: 0,
                target: None
            }),
            format!("Play:{card_tok}")
        );
        // Out-of-range indices degrade to "?" instead of panicking — a
        // trace must never take down a fork rep.
        assert_eq!(
            label(RunAction::PlayCard {
                hand_idx: 99,
                target: None
            }),
            "Play:?"
        );
        assert_eq!(label(RunAction::EndTurn), "EndTurn");
        assert_eq!(
            label(RunAction::DrinkPotion {
                potion_slot: 0,
                target: Some(enemy_uid)
            }),
            format!("Potion:{potion_tok}@{enemy_tok}")
        );
        assert_eq!(
            label(RunAction::DrinkPotion {
                potion_slot: 0,
                target: None
            }),
            format!("Potion:{potion_tok}")
        );
        assert_eq!(
            label(RunAction::DrinkPotion {
                potion_slot: 99,
                target: None
            }),
            "Potion:?"
        );
        assert_eq!(
            label(RunAction::CombatResolveChoice { option: Some(1) }),
            "Resolve:1"
        );
        assert_eq!(
            label(RunAction::CombatResolveChoice { option: None }),
            "Resolve:skip"
        );
        assert_eq!(
            label(RunAction::CombatResolveChoiceTakeFirst { count: 2 }),
            "ResolveFirst:2"
        );
        // Non-combat variants fall through to the Debug form.
        assert_eq!(
            label(RunAction::Move { col: 3 }),
            format!("{:?}", RunAction::Move { col: 3 })
        );
        // A uid with no creature keeps the raw number rather than panicking.
        assert_eq!(creature_token(combat, 987_654), "uid987654");
    }

    #[test]
    fn trace_step_snapshots_visible_state_and_drops_dead_enemies() {
        let mut rm = fight_entry(4242);
        let combat = rm.combat.as_ref().expect("in combat");
        let ts = trace_step(combat, &RunAction::EndTurn);

        assert_eq!(ts.round, combat.round);
        assert_eq!(ts.hp, combat.player.current_hp);
        assert_eq!(ts.block, combat.player.block);
        assert_eq!(ts.energy, combat.player.energy);
        assert_eq!(ts.draw_n, combat.player.draw.len());
        assert_eq!(ts.discard_n, combat.player.discard.len());
        assert_eq!(ts.action, "EndTurn");

        assert_eq!(ts.hand.len(), combat.player.hand.len());
        assert!(!ts.hand.is_empty());
        assert!(
            ts.hand.iter().all(|c| c.starts_with("CARD.")),
            "{:?}",
            ts.hand
        );

        let alive = living_enemies(combat);
        let expected: Vec<String> = alive
            .iter()
            .map(|c| {
                let intent = c.intent.map_or_else(|| "None".to_string(), intent_label);
                format!(
                    "{}:{}+{}b:{intent}",
                    monster_token(c.monster_id.0),
                    c.current_hp,
                    c.block
                )
            })
            .collect();
        assert_eq!(ts.enemies, expected);
        assert!(!ts.enemies.is_empty());

        // A dead enemy drops out of the snapshot entirely.
        let victim = alive[0].id;
        let n_before = ts.enemies.len();
        let combat = rm.combat.as_mut().expect("in combat");
        let idx = combat
            .creatures
            .iter()
            .position(|c| c.id == victim)
            .expect("victim");
        combat.creatures[idx].current_hp = 0;
        let after = trace_step(rm.combat.as_ref().expect("in combat"), &RunAction::EndTurn);
        assert_eq!(after.enemies.len(), n_before - 1);
    }

    // ── arrival-state entry mutations ──────────────────────────────────

    /// Wound the player in a snapshot (creature + the mirrored legacy
    /// field, which `creature_heal` keeps in sync).
    fn wound_player(snap: &mut FatalSnapshot, hp: i32) {
        let combat = snap.rm.combat.as_mut().expect("in combat");
        let uid = combat.player.creature_uid.expect("player creature");
        let idx = combat
            .creatures
            .iter()
            .position(|c| c.id == uid.0)
            .expect("player creature in arena");
        combat.creatures[idx].current_hp = hp;
        combat.player.current_hp = hp;
        snap.entry_hp = hp;
    }

    /// Retype one draw-pile card as a Curse — cheaper than routing a real
    /// curse into the deck, and `apply_entry_mutation` filters on
    /// `card_type` alone.
    fn plant_curse(snap: &mut FatalSnapshot) -> sts2_core::CardUid {
        let combat = snap.rm.combat.as_mut().expect("in combat");
        let uid = combat.player.draw[0];
        let idx = combat
            .cards
            .iter()
            .position(|c| c.uid == uid)
            .expect("draw card");
        combat.cards[idx].card_type = sts2_core::cards::CardType::Curse;
        uid
    }

    #[test]
    fn heal_mutation_tops_up_hp_and_restamps_entry_hp() {
        let mut snap = snapshot_at_entry(4242);
        let max_hp = snap.rm.combat.as_ref().expect("in combat").player.max_hp;
        wound_player(&mut snap, 12);

        apply_entry_mutation(&mut snap, EntryMutation::Heal);

        let combat = snap.rm.combat.as_ref().expect("in combat");
        assert_eq!(combat.player.current_hp, max_hp, "heal clamps at max");
        assert_eq!(snap.entry_hp, max_hp, "entry_hp is restamped for the row");
        // Heal-only leaves the other two axes untouched.
        assert!(combat.player.potions.len() < combat.player.effective_potion_slots());
    }

    #[test]
    fn potions_mutation_fills_the_belt_to_cap_and_keeps_held_potions() {
        let mut snap = snapshot_at_entry(4242);
        let (cap, held) = {
            let combat = snap.rm.combat.as_mut().expect("in combat");
            combat.player.potions.clear();
            combat
                .player
                .potions
                .push(sts2_core::potions::PotionKind::BlockPotion);
            (
                combat.player.potion_slots as usize,
                combat.player.potions.to_vec(),
            )
        };
        assert!(cap > 1, "fixture needs a multi-slot belt, got {cap}");

        apply_entry_mutation(&mut snap, EntryMutation::Potions);

        let potions = &snap.rm.combat.as_ref().expect("in combat").player.potions;
        assert_eq!(potions.len(), cap);
        assert_eq!(potions[0], held[0], "held potions are kept, not replaced");
        assert!(potions[1..]
            .iter()
            .all(|p| *p == sts2_core::potions::PotionKind::FirePotion));

        // Idempotent: a second application cannot overfill the belt.
        apply_entry_mutation(&mut snap, EntryMutation::Potions);
        assert_eq!(
            snap.rm
                .combat
                .as_ref()
                .expect("in combat")
                .player
                .potions
                .len(),
            cap
        );
    }

    #[test]
    fn decurse_mutation_pulls_curses_out_of_the_piles() {
        use sts2_core::cards::CardType;
        let mut snap = snapshot_at_entry(4242);
        let cursed = plant_curse(&mut snap);
        let draw_before = snap
            .rm
            .combat
            .as_ref()
            .expect("in combat")
            .player
            .draw
            .len();

        apply_entry_mutation(&mut snap, EntryMutation::Decurse);

        let combat = snap.rm.combat.as_ref().expect("in combat");
        assert_eq!(combat.player.draw.len(), draw_before - 1);
        assert!(!combat.player.draw.contains(&cursed));
        // Fight-scoped: no Curse survives in ANY pile.
        let in_piles = |uid: &sts2_core::CardUid| {
            combat.player.hand.contains(uid)
                || combat.player.draw.contains(uid)
                || combat.player.discard.contains(uid)
                || combat.player.exhaust.contains(uid)
                || combat.player.play.contains(uid)
        };
        assert!(!combat
            .cards
            .iter()
            .any(|c| c.card_type == CardType::Curse && in_piles(&c.uid)));
    }

    #[test]
    fn all_mutation_applies_every_axis_at_once() {
        use sts2_core::cards::CardType;
        let mut snap = snapshot_at_entry(4242);
        let max_hp = snap.rm.combat.as_ref().expect("in combat").player.max_hp;
        wound_player(&mut snap, 9);
        let cursed = plant_curse(&mut snap);
        let cap = snap
            .rm
            .combat
            .as_ref()
            .expect("in combat")
            .player
            .potion_slots as usize;

        apply_entry_mutation(&mut snap, EntryMutation::All);

        let combat = snap.rm.combat.as_ref().expect("in combat");
        assert_eq!(combat.player.current_hp, max_hp);
        assert_eq!(snap.entry_hp, max_hp);
        assert_eq!(combat.player.potions.len(), cap);
        assert!(!combat.player.draw.contains(&cursed));
        assert_eq!(
            combat
                .cards
                .iter()
                .find(|c| c.uid == cursed)
                .map(|c| c.card_type),
            Some(CardType::Curse),
            "the card object survives; only the PILES are cleaned"
        );
    }
}
