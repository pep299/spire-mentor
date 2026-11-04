//! `ConsoleSimulator` — top-level REPL driver. Mirrors
//! `sts_lightspeed/include/sim/ConsoleSimulator.h` shape: per-screen
//! dispatch + scripted-replay context. v1 implements only the combat
//! screen (the only screen `sts2-core` exposes today); other screens
//! print "screen not implemented" and exit 0 cleanly.

// Cliext Phase 4 (`sts2-rs-90n5`) marked `EncounterKind` deprecated.
// This module is the legacy holder + `build_starter_state` shim that
// keeps the old single-monster API alive for tests and the surviving
// `from_legacy_args` callers; suppress the in-file usage warnings so
// the deprecation only fires for NEW external callers.
#![allow(deprecated)]

use std::io::{self, Write};

use sts2_core::{apply, combat_loop, Action, CombatPhase, CombatState, MonsterKind, StepResult};

use crate::agent::{Agent, AgentChoice};
use crate::sim::print::print_combat_state;

pub mod combat;
pub mod jsonl;
pub mod parse;
pub mod print;
pub mod scenario;

pub use scenario::Scenario;

/// Simulator-wide knobs + scripted-mode result flags. Mirrors
/// `sts_lightspeed::SimulatorContext`.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools, dead_code)] // mirrors lightspeed SimulatorContext layout; some flags reserved for later beads
pub struct SimulatorContext {
    pub print_prompts: bool,
    pub print_input: bool,
    pub print_log_actions: bool,
    /// Render the combined state + indexed action menu before every
    /// `agent.next_action` call (lightspeed-parity auto-prompt). The
    /// human REPL path defaults this to `true`; the random-agent and
    /// `--jsonl`-only paths override to `false` so machine-readable
    /// output stays terse. Mirrors `sts_lightspeed::SimulatorContext`'s
    /// `printActions` flag (renamed for clarity — `show_action_menu`
    /// describes what the flag toggles, not the print verb).
    pub show_action_menu: bool,
    pub quit_on_test_failed: bool,
    pub took_action: bool,
    pub failed_test: bool,
}

impl Default for SimulatorContext {
    fn default() -> Self {
        Self {
            print_prompts: true,
            print_input: false,
            print_log_actions: true,
            show_action_menu: true,
            quit_on_test_failed: true,
            took_action: false,
            failed_test: false,
        }
    }
}

/// Player class — selects starter HP / starter deck.
///
/// Ironclad and Silent are the fully-wired classes (Phase-6 P6-1/P6-2).
/// Defect/Regent/Necrobinder are accepted at the scenario-loader layer
/// (bead `sts2-rs-5m6k`) so .run-corpus fixtures stop bouncing off the
/// loader with `unknown player.class` and instead surface the real
/// per-class content gaps (unknown card / unknown relic) one layer
/// deeper. Their starter HP / deck are placeholders — see
/// `scenario::starter::build_state` and `scenario::deck::class_hint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerClass {
    Ironclad,
    Silent,
    Defect,
    Regent,
    Necrobinder,
}

/// Encounter — selects which monster(s) spawn at combat start.
///
/// As of bead `sts2-rs-2o0k` (cliext B2) the CLI surface no longer
/// references this enum directly — `--encounter` is now a free-form
/// alias string resolved via
/// [`scenario::resolve_encounter`]. The enum is preserved as the
/// internal one-monster type for legacy `Scenario::from_legacy_args`
/// callers (tests + the surviving `build_starter_state` helper).
#[allow(dead_code)]
#[deprecated(
    since = "0.0.2",
    note = "Cliext Phase 4 (sts2-rs-90n5): EncounterKind is the legacy \
            single-monster scenario shape. New code should route through \
            the multi-monster path: Scenario::from_monsters + MonsterSpec \
            with explicit MonsterSpawnArgs (see crates/sts2-content/src/monster_dispatch.rs)."
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncounterKind {
    /// Nibbit (BUTT/SLICE/HISS rotation) — DSL-driven via
    /// `sts2-content::monsters::act1_weak::create_nibbit_ai`.
    Nibbit,
    /// `ShrinkerBeetle` — DSL-driven via
    /// `sts2-content::monsters::act1_weak::create_shrinker_beetle_ai`.
    ShrinkerBeetle,
}

impl EncounterKind {
    pub(crate) const fn monster_kind(self) -> MonsterKind {
        match self {
            Self::Nibbit => MonsterKind::Nibbit,
            Self::ShrinkerBeetle => MonsterKind::ShrinkerBeetle,
        }
    }
}

/// Build a [`CombatState`] ready for `combat_loop::start_combat`. Hand is
/// empty (`start_combat` handles the first draw); deck = 5 Strike + 4
/// Defend + 1 Bash for Ironclad. Monster HP rolled via the engine's
/// `MonsterKind::starting_hp`.
///
/// Thin shim — delegates to
/// [`scenario::starter::build_state`] via
/// [`Scenario::from_legacy_args`]. Preserved so existing call sites
/// (and `tests/{determinism,scripted_replay}.rs` once they exercise
/// it) keep compiling unchanged after the A1 refactor seam
/// (bead `sts2-rs-cu83`).
#[must_use]
#[allow(dead_code)] // legacy single-monster path kept for tests + Scenario::from_legacy_args.
pub fn build_starter_state(
    seed: u64,
    encounter: EncounterKind,
    player: PlayerClass,
    hp_override: Option<u32>,
) -> CombatState {
    let scenario = Scenario::from_legacy_args(encounter);
    scenario::starter::build_state(&scenario, seed, player, hp_override, None)
}

/// Build a [`CombatState`] from a resolved `Vec<MonsterSpec>` rather
/// than the legacy single-monster `EncounterKind`. Routes through
/// [`Scenario::from_monsters`] so position-dependent AI args
/// (`is_alone` / `is_front`) are fixed up automatically. Bead
/// `sts2-rs-2o0k` (cliext B2) — the entry point for multi-monster
/// CLI aliases.
///
/// Delegates to [`build_starter_state_from_monsters_with_ascension`]
/// at ascension 0 — kept byte-stable for existing callers.
#[must_use]
pub fn build_starter_state_from_monsters(
    seed: u64,
    monsters: Vec<scenario::MonsterSpec>,
    player: PlayerClass,
    hp_override: Option<u32>,
    max_hp_override: Option<u32>,
    deck: Option<Vec<scenario::deck::DeckCardSpec>>,
) -> CombatState {
    build_starter_state_from_monsters_with_ascension(
        seed,
        monsters,
        player,
        hp_override,
        max_hp_override,
        deck,
        0,
    )
}

/// Ascension-aware variant of [`build_starter_state_from_monsters`].
/// `ascension` is passed EXPLICITLY by the caller (see
/// [`scenario::starter::build_state_with_ascension`]).
#[must_use]
pub fn build_starter_state_from_monsters_with_ascension(
    seed: u64,
    monsters: Vec<scenario::MonsterSpec>,
    player: PlayerClass,
    hp_override: Option<u32>,
    max_hp_override: Option<u32>,
    deck: Option<Vec<scenario::deck::DeckCardSpec>>,
    ascension: u8,
) -> CombatState {
    let scenario = Scenario::from_monsters(monsters);
    let mut state = scenario::starter::build_state_with_ascension(
        &scenario,
        seed,
        player,
        hp_override,
        max_hp_override,
        ascension,
    );
    if let Some(deck) = deck {
        scenario::deck::install_deck(&mut state, &deck);
    }
    state
}

/// The top-level REPL. Drives `combat_loop::start_combat`, then loops
/// `agent.next_action() -> apply()` until combat ends or the agent quits.
pub struct ConsoleSimulator<W: Write, A: Agent> {
    pub state: CombatState,
    pub out: W,
    pub agent: A,
    pub ctx: SimulatorContext,
    pub jsonl: Option<jsonl::JsonlWriter>,
}

impl<W: Write, A: Agent> ConsoleSimulator<W, A> {
    pub const fn new(state: CombatState, out: W, agent: A, ctx: SimulatorContext) -> Self {
        Self {
            state,
            out,
            agent,
            ctx,
            jsonl: None,
        }
    }

    #[must_use]
    pub fn with_jsonl(mut self, writer: jsonl::JsonlWriter) -> Self {
        self.jsonl = Some(writer);
        self
    }

    /// Run combat until the player wins, loses, or the agent returns
    /// `QuitRepl`. Returns `Ok(())` on clean exit; bumps `ctx.failed_test`
    /// on a scripted-assertion mismatch.
    pub fn play(&mut self) -> io::Result<()> {
        // Setup -> PlayerTurnStart (rolls intents, etc).
        let _ = combat_loop::start_combat(&mut self.state);
        // PlayerTurnStart -> PlayerPlayPhase (deals hand + refills energy).
        let _ = combat_loop::start_player_turn(&mut self.state);
        writeln!(self.out, "── starting combat ──")?;
        print_combat_state(&mut self.out, &self.state)?;

        // Bead `sts2-rs-2zkw`: per-round action cap as a no-progress
        // cycle guard. The MCTS agent can get stuck in a fixed-point
        // loop when the engine accepts an action whose net effect is a
        // game-state-equivalent no-op (e.g. BattleTrance with a copy of
        // itself in discard + empty draw: each play swaps two UIDs
        // between hand and discard but the player-visible state is
        // identical, so MCTS keeps re-picking it). 200 plays/round is
        // well above any legitimate Ironclad combo (deck cycling tops
        // out around 50-60); hitting it indicates a cycle. We inject a
        // synthetic EndTurn so combat reaches a verdict rather than
        // hanging the harness.
        const PER_ROUND_ACTION_CAP: u32 = 200;
        // Bead `sts2-rs-16p0` follow-up: total-rounds cap as a stall
        // guard. STS2 combats terminate within a small handful of
        // rounds in any legitimate play (bosses fall in 5-15 turns;
        // slow / block-heavy decks ~30; nothing real exceeds 50).
        // Decks that exhaust all their damage cards and survive on
        // block alone can stall for 800+ rounds against a boss
        // without an escalation power (e.g. seed=11 random Ironclad
        // vs CeremonialBeast). 100 rounds is well above any legitimate
        // combat; hitting it forces a CombatLost so the sweep harness
        // doesn't burn a 60s timeout on a genuinely-stuck fight.
        const MAX_COMBAT_ROUNDS: u16 = 100;
        let mut round_action_count: u32 = 0;
        let mut last_round = self.state.round;

        loop {
            if self.state.is_over {
                let outcome = if self.state.player_won { "WON" } else { "LOST" };
                writeln!(self.out, "── combat over: {outcome} ──")?;
                if let Some(w) = self.jsonl.as_mut() {
                    w.flush()?;
                }
                return Ok(());
            }

            // Sanity: we should be in PlayerPlayPhase, OR in
            // PlayerTurnStart with a pending_choice live (bead
            // sts2-rs-l2nz: a `before_side_turn_start` listener
            // opened a choice and `start_player_turn` deferred
            // turn-setup until ResolveChoice fires). The latter is
            // a legal resting state — `legal_actions` returns
            // `ResolveChoice` variants and the agent will pick one,
            // which `apply` then routes through
            // `continue_player_turn_setup` to land us in
            // PlayerPlayPhase.
            if self.state.phase != CombatPhase::PlayerPlayPhase
                && !(self.state.phase == CombatPhase::PlayerTurnStart
                    && self.state.pending_choice.is_some())
            {
                writeln!(
                    self.out,
                    "  (engine left in unexpected phase {:?} — exiting)",
                    self.state.phase
                )?;
                return Ok(());
            }

            if self.state.round != last_round {
                last_round = self.state.round;
                round_action_count = 0;
            }
            if self.state.round > MAX_COMBAT_ROUNDS {
                writeln!(
                    self.out,
                    "  (combat exceeded {MAX_COMBAT_ROUNDS} rounds without resolution \
                     — forcing CombatLost; stall guard, bead sts2-rs-2zkw)"
                )?;
                self.state.is_over = true;
                self.state.player_won = false;
                continue;
            }
            if round_action_count >= PER_ROUND_ACTION_CAP {
                writeln!(
                    self.out,
                    "  (per-round action cap {PER_ROUND_ACTION_CAP} hit \
                     — injecting EndTurn; cycle guard, bead sts2-rs-2zkw)"
                )?;
                let before = self.jsonl.as_ref().map(|_| self.state.clone());
                let result = apply(&mut self.state, Action::EndTurn);
                if let (Some(writer), Some(before)) = (self.jsonl.as_mut(), before.as_ref()) {
                    writer.write_step(before, &Action::EndTurn, &self.state, result)?;
                }
                round_action_count = 0;
                continue;
            }
            let choice = self.agent.next_action(&self.state, &mut self.out)?;
            match choice {
                AgentChoice::Action(action) => {
                    let before = self.jsonl.as_ref().map(|_| self.state.clone());
                    let result = apply(&mut self.state, action.clone());
                    self.ctx.took_action = true;
                    round_action_count += 1;
                    if let (Some(writer), Some(before)) = (self.jsonl.as_mut(), before.as_ref()) {
                        writer.write_step(before, &action, &self.state, result)?;
                    }
                    if self.ctx.print_log_actions {
                        writeln!(self.out, "  -> {result:?}")?;
                    }
                    match result {
                        StepResult::CombatWon => {
                            self.state.is_over = true;
                            self.state.player_won = true;
                        }
                        StepResult::CombatLost => {
                            self.state.is_over = true;
                            self.state.player_won = false;
                        }
                        StepResult::PendingChoice => {
                            writeln!(
                                self.out,
                                "  (pending choice — choice resolution not yet supported in v1)"
                            )?;
                        }
                        StepResult::InvalidTarget => {
                            // Bead sts2-rs-vaq: surface OOB / dead-target
                            // plays as a typed reject so the REPL re-prompts
                            // instead of silently advancing.
                            let alive_enemies = self
                                .state
                                .enemy_uids
                                .iter()
                                .filter(|uid| {
                                    self.state
                                        .creatures
                                        .iter()
                                        .any(|c| c.id == uid.0 && c.current_hp > 0)
                                })
                                .count();
                            writeln!(
                                self.out,
                                "  invalid target slot (legal slots: 0..{alive_enemies}); play not applied"
                            )?;
                        }
                        StepResult::IllegalPlay => {
                            // GH #1122: unaffordable / vetoed / wrong-phase
                            // play — re-prompt instead of silently advancing.
                            writeln!(self.out, "  illegal play; play not applied")?;
                        }
                        StepResult::Advanced => {}
                    }
                }
                AgentChoice::Continue => {}
                AgentChoice::QuitRepl => {
                    writeln!(self.out, "── quit ──")?;
                    return Ok(());
                }
                AgentChoice::AssertFailed(msg) => {
                    writeln!(self.out, "  ASSERT FAILED: {msg}")?;
                    self.ctx.failed_test = true;
                    if self.ctx.quit_on_test_failed {
                        return Ok(());
                    }
                }
            }
        }
    }
}
