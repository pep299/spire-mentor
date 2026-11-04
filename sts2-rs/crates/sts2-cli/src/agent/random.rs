//! `RandomAgent` — picks uniformly at random from legal actions. The
//! enumeration matches what the REPL's `actions` command shows, so
//! human-vs-random behavior diverges only on the choice (not on what's
//! considered a move).
//!
//! Determinism: seeded with a `u64`; same seed reproduces the same
//! action trajectory across runs (modulo engine determinism, which is
//! itself seeded).

use std::io::{self, Write};

use sts2_core::{Action, CombatState};

use crate::agent::{Agent, AgentChoice};
use crate::sim::combat::legal_actions;

pub struct RandomAgent {
    rng: sts2_core::dotnet_rng::Rng,
    /// Optional cap on number of decisions returned. `None` = unlimited.
    /// Set by `--steps N` on the CLI; once exhausted the agent returns
    /// `QuitRepl` to terminate the loop.
    pub remaining_steps: Option<u32>,
}

impl RandomAgent {
    #[must_use]
    pub fn with_step_cap(seed: u64, steps: u32) -> Self {
        Self {
            rng: sts2_core::dotnet_rng::Rng::new((seed) as i32),
            remaining_steps: if steps == 0 { None } else { Some(steps) },
        }
    }
}

impl Agent for RandomAgent {
    fn next_action<W: Write>(
        &mut self,
        state: &CombatState,
        out: &mut W,
    ) -> io::Result<AgentChoice> {
        if self.remaining_steps == Some(0) {
            writeln!(out, "  (random agent: step cap reached)")?;
            return Ok(AgentChoice::QuitRepl);
        }
        let actions = legal_actions(state);
        if actions.is_empty() {
            // No legal moves — shouldn't happen in PlayerPlayPhase since
            // EndTurn is always legal, but be defensive.
            return Ok(AgentChoice::QuitRepl);
        }
        let idx = self.rng.next_int_exclusive(0, actions.len() as i32) as usize;
        let choice = actions[idx].clone();
        if let Some(n) = self.remaining_steps.as_mut() {
            *n -= 1;
        }
        writeln!(out, "  random> {}", fmt_action(&choice))?;
        Ok(AgentChoice::Action(choice))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::scenario::{MonsterSpec, Scenario};
    use crate::sim::PlayerClass;
    use sts2_core::orbs::Orb;
    use sts2_core::MonsterKind;

    /// A Nibbit combat driven through `start_combat` into
    /// `PlayerPlayPhase`, where `legal_actions` always offers at least
    /// `EndTurn`.
    fn started_combat() -> CombatState {
        let scenario = Scenario::from_monsters(vec![MonsterSpec::new(MonsterKind::Nibbit)]);
        let mut state = crate::sim::scenario::starter::build_state(
            &scenario,
            1,
            PlayerClass::Ironclad,
            None,
            None,
        );
        let _ = sts2_core::combat_loop::start_combat(&mut state);
        state
    }

    #[test]
    fn with_step_cap_zero_is_unlimited_and_nonzero_is_bounded() {
        assert_eq!(RandomAgent::with_step_cap(1, 0).remaining_steps, None);
        assert_eq!(RandomAgent::with_step_cap(1, 3).remaining_steps, Some(3));
    }

    #[test]
    fn fmt_action_renders_each_variant() {
        assert_eq!(
            fmt_action(&Action::PlayCard {
                hand_index: 1,
                target: None
            }),
            "play 1"
        );
        assert_eq!(
            fmt_action(&Action::PlayCard {
                hand_index: 1,
                target: Some(2)
            }),
            "play 1 2"
        );
        assert_eq!(fmt_action(&Action::EndTurn), "end");
        assert_eq!(
            fmt_action(&Action::UsePotion {
                slot: 0,
                target: None
            }),
            "potion 0"
        );
        assert_eq!(
            fmt_action(&Action::DrinkPotion {
                potion_slot: 2,
                target: Some(1)
            }),
            "drink 2 1"
        );
        assert_eq!(fmt_action(&Action::Evoke { times: 3 }), "evoke 3");
        assert!(fmt_action(&Action::Channel {
            orb: Orb::Frost,
            target: None
        })
        .starts_with("channel"));
        assert!(fmt_action(&Action::ResolveChoice {
            selection: smallvec::smallvec![0]
        })
        .starts_with("resolve"));
    }

    /// Same seed + same state => same first decision (the agent's RNG is
    /// the sole source of nondeterminism).
    #[test]
    fn same_seed_reproduces_first_action() {
        let state = started_combat();
        let mut out = Vec::new();
        let pick = |seed: u64| -> String {
            let mut a = RandomAgent::with_step_cap(seed, 0);
            let mut sink = Vec::new();
            match a.next_action(&state, &mut sink).expect("decision") {
                AgentChoice::Action(act) => fmt_action(&act),
                other => panic!("expected an action, got {other:?}"),
            }
        };
        assert_eq!(pick(99), pick(99), "same seed must reproduce the action");
        // The progress line is written to `out`.
        let mut a = RandomAgent::with_step_cap(99, 0);
        let _ = a.next_action(&state, &mut out).expect("decision");
        assert!(String::from_utf8_lossy(&out).contains("random>"));
    }

    /// A 1-step cap yields one real action, then `QuitRepl` once the cap
    /// is exhausted (exercises the `remaining_steps == Some(0)` guard).
    #[test]
    fn step_cap_exhaustion_quits_repl() {
        let state = started_combat();
        let mut agent = RandomAgent::with_step_cap(5, 1);
        let mut sink = Vec::new();
        assert!(matches!(
            agent.next_action(&state, &mut sink).expect("first"),
            AgentChoice::Action(_)
        ));
        assert_eq!(agent.remaining_steps, Some(0));
        assert!(matches!(
            agent.next_action(&state, &mut sink).expect("second"),
            AgentChoice::QuitRepl
        ));
        assert!(String::from_utf8_lossy(&sink).contains("step cap reached"));
    }
}
