//! `MctsAgent` — headless MCTS auto-player. At each decision the agent
//! runs `--mcts-hint-default` simulations and picks via the shared
//! `Mcts::pick_root_action` rule (override → proven-win guard →
//! tie-band → root guards), the same rule the deployed run agent uses.
//! Before the 2026-08-09 A8 audit repair this module kept its own
//! strict `Q desc, visits desc` argmax plus a local
//! `lost_position_override` call — but `search_n` already runs (and
//! discarded) the full shared pick, so the local re-derivation was an
//! old/new hybrid that still shifted with every shared-rule change
//! while duplicating the override probe. The hint REPL
//! (`agent::human::print_hint`) still SORTS its display by Q desc /
//! visits desc; that is a presentation order, not this agent's pick.
//!
//! Used by `--agent mcts` (`cli::combat::AgentKind::Mcts`) and by the
//! in-process `mcts-sweep` subcommand. Quiet by default: writes a one-
//! line `mcts> <action>` per decision so JSONL / scripted-replay output
//! stays parseable.

use std::io::{self, Write};

use sts2_core::{Action, CombatState};
use sts2_mcts::{Mcts, MctsConfig, RolloutPolicyKind};

use crate::agent::{Agent, AgentChoice};
use crate::sim::combat::legal_actions;

#[derive(Clone, Copy, Debug)]
pub struct MctsAgentConfig {
    pub sims: u32,
    pub c_puct: f32,
    pub rollout_depth: u32,
    pub seed: u64,
    pub verbose: bool,
    /// When `true`, preserve the search tree across decisions by
    /// re-rooting at the chosen action's subtree (see [`Mcts::reroot`]).
    /// Default is `false` — every decision starts a fresh root. Tree
    /// reuse can amortize search cost across a combat but its statistics
    /// are only approximately valid under PIMC determinization (see
    /// the PIMC caveat in `Mcts::reroot`'s doc-comment).
    pub tree_reuse: bool,
    /// Use the pre-`dfv1` uniform-random rollout instead of the default
    /// expert-heuristic softmax. Wired by `--mcts-uniform-rollout` for
    /// A/B comparisons.
    pub uniform_rollout: bool,
}

pub struct MctsAgent {
    cfg: MctsAgentConfig,
    /// Persistent search tree, kept across decisions when
    /// `cfg.tree_reuse` is set. `None` on first call OR when the previous
    /// re-root failed (e.g. the chosen action was never expanded under
    /// the previous root).
    persistent: Option<Mcts>,
    /// Action we returned from the most recent `next_action`. On the
    /// next call we attempt to reroot the persistent tree along this
    /// edge before searching.
    pending_action: Option<Action>,
}

impl MctsAgent {
    #[must_use]
    pub const fn new(cfg: MctsAgentConfig) -> Self {
        Self {
            cfg,
            persistent: None,
            pending_action: None,
        }
    }

    fn build_config(&self) -> MctsConfig {
        MctsConfig {
            c_puct: self.cfg.c_puct,
            max_simulations: self.cfg.sims,
            max_rollout_depth: self.cfg.rollout_depth,
            seed: self.cfg.seed,
            rollout_policy: if self.cfg.uniform_rollout {
                RolloutPolicyKind::Uniform
            } else {
                RolloutPolicyKind::Heuristic
            },
            ..MctsConfig::default()
        }
    }

    /// Pick the best action for `state` using the agent's configured
    /// MCTS budget. Returns `None` if the state has no legal actions
    /// (terminal). Drives tree reuse when `cfg.tree_reuse` is set.
    pub fn pick(&mut self, state: &CombatState) -> Option<Action> {
        let actions = legal_actions(state);
        if actions.is_empty() {
            return None;
        }
        let config = self.build_config();

        // Reuse-or-rebuild: if we have a persistent tree and an action
        // we returned from the previous call, try to walk to that
        // child. If the action wasn't expanded yet, drop the tree.
        let mut mcts: Mcts = if self.cfg.tree_reuse {
            let prev = self.persistent.take();
            let action = self.pending_action.take();
            match (prev, action) {
                (Some(mut prev), Some(action)) => {
                    if prev.reroot(&action, state.clone()) {
                        prev
                    } else {
                        Mcts::new(state.clone(), config)
                    }
                }
                _ => Mcts::new(state.clone(), config),
            }
        } else {
            Mcts::new(state.clone(), config)
        };

        // One shared rule (`Mcts::pick_root_action`, via `try_search_n`)
        // covers override / proven-win guard / tie-band / root guards —
        // no local re-derivation (see the module doc for the history).
        let chosen = mcts.try_search_n(self.cfg.sims).ok();

        if self.cfg.tree_reuse {
            self.persistent = Some(mcts);
        }
        chosen
    }
}

impl Agent for MctsAgent {
    fn next_action<W: Write>(
        &mut self,
        state: &CombatState,
        out: &mut W,
    ) -> io::Result<AgentChoice> {
        let Some(action) = self.pick(state) else {
            return Ok(AgentChoice::QuitRepl);
        };
        if self.cfg.verbose {
            writeln!(out, "  mcts> {}", fmt_action(&action))?;
        }
        // Record the action so the next `pick` call can reroot the
        // persistent tree at this edge (consumed lazily by `pick`).
        // No-op when `tree_reuse` is disabled (pick won't read it).
        self.pending_action = Some(action.clone());
        Ok(AgentChoice::Action(action))
    }
}

fn fmt_action(a: &Action) -> String {
    match a {
        Action::PlayCard { hand_index, target } => target.as_ref().map_or_else(
            || format!("play {hand_index}"),
            |t| format!("play {hand_index} {t}"),
        ),
        Action::PlayCardAtAlly { hand_index, ally } => format!("play {hand_index} ally {ally}"),
        Action::DrinkPotionAtAlly { potion_slot, ally } => {
            format!("drink {potion_slot} ally {ally}")
        }
        Action::EndTurn => "end".into(),
        Action::UsePotion { slot, target } => target.as_ref().map_or_else(
            || format!("potion {slot}"),
            |t| format!("potion {slot} {t}"),
        ),
        Action::DrinkPotion {
            potion_slot,
            target,
        } => target.as_ref().map_or_else(
            || format!("drink {potion_slot}"),
            |t| format!("drink {potion_slot} {t}"),
        ),
        Action::Channel { orb, target } => target.as_ref().map_or_else(
            || format!("channel {orb:?}"),
            |t| format!("channel {orb:?} {t}"),
        ),
        Action::Evoke { times } => format!("evoke {times}"),
        Action::ResolveChoice { selection } => format!("resolve {selection:?}"),
    }
}
