//! Rarely-fired transition branches of [`sts2_core::monster_ai`].
//!
//! The inline `#[cfg(test)] mod tests` in `monster_ai.rs` already covers the
//! happy paths (linear chains, weighted picks, repeat/cooldown weighting,
//! `force_next_state` / `force_current_state`, the all-zero RNG tick). What is
//! left uncovered are the *degraded* edges, all of which the C# state machine
//! has a definite opinion about:
//!
//! `MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine/`
//!
//! * `MoveState.CanTransitionAway` — `MustPerformOnceBeforeTransitioning`
//!   gates on `_performedAtLeastOnce`, and
//!   `MonsterMoveStateMachine.FindNextMoveState` **early-returns** on
//!   `!_currentState.CanTransitionAway`, i.e. *before* it reaches any
//!   `GetNextState` call. So a held move costs zero RNG — the Rust mirror of
//!   that early return (`roll_move` / `roll_move_with_combat`) must consume
//!   zero `MonsterAi` ticks too.
//! * `MoveState.OnExitState()` sets `_performedAtLeastOnce = false`, which is
//!   why a freshly-entered `must_perform_once` move is held again.
//! * `ConditionalBranchState.GetNextState` throws
//!   `InvalidOperationException("No valid next state found.")` when no branch
//!   evaluates `> 0` — the Rust panic is the faithful analogue, not a bug.
//!
//! Everything else pinned here (`peek_intents` fallbacks,
//! `transplant_instance_state_from`, `current_move_opt`) is Rust-side
//! machinery with no C# counterpart; those tests pin the documented contract
//! in `monster_ai.rs` itself.

use hegel::generators as gs;
use hegel::TestCase;
use smallvec::smallvec;
use std::collections::BTreeMap;
use std::sync::Arc;
use sts2_core::monster_ai::{
    ConditionFn, ConditionalBranch, ConditionalBranchState, MonsterAi, MonsterState,
    MoveRepeatType, MoveState, RandomBranchState, StateId, WeightedBranch,
};
use sts2_core::{CombatState, CreatureUid, DotNetRng, Intent};

fn noop_effect(_: &mut CombatState, _: CreatureUid) {}

fn move_state(
    id: StateId,
    damage: i32,
    follow_up: StateId,
    must_perform_once: bool,
) -> MonsterState {
    MonsterState::Move(MoveState {
        id,
        effect: noop_effect,
        intents: smallvec![Intent::Attack { damage }],
        follow_up: Some(follow_up),
        must_perform_once,
    })
}

// ── must_perform_once holds the move (C# `CanTransitionAway`) ──────────

/// `X` (free) → `Y` (`must_perform_once`) → `X`. Once the machine *enters*
/// `Y`, `OnExitState` has cleared `_performedAtLeastOnce`, so the next
/// `RollMove` must return `Y` again and, per the C# early return in
/// `FindNextMoveState`, must not touch the RNG.
#[test]
fn must_perform_once_move_is_held_and_costs_no_rng() {
    let mut states = BTreeMap::new();
    states.insert("X", move_state("X", 1, "Y", false));
    states.insert("Y", move_state("Y", 2, "X", true));
    let mut ai = MonsterAi::new(states, "X");
    let mut rng = DotNetRng::new(7);

    ai.on_move_performed();
    assert_eq!(ai.roll_move(&mut rng).id, "Y", "X transitions into Y");
    let after_entering_y = rng.counter();

    // No `on_move_performed` in between: Y has not performed yet.
    assert_eq!(ai.roll_move(&mut rng).id, "Y", "Y is held until performed");
    assert_eq!(ai.roll_move(&mut rng).id, "Y", "…and stays held");
    assert_eq!(
        rng.counter(),
        after_entering_y,
        "a held move early-returns before GetNextState — zero RNG",
    );

    ai.on_move_performed();
    assert_eq!(
        ai.roll_move(&mut rng).id,
        "X",
        "performing releases the hold"
    );
}

/// Same contract on the combat-aware roll path.
#[test]
fn must_perform_once_move_is_held_on_the_combat_path() {
    let mut states = BTreeMap::new();
    states.insert("X", move_state("X", 1, "Y", false));
    states.insert("Y", move_state("Y", 2, "X", true));
    let mut ai = MonsterAi::new(states, "X");
    let combat = CombatState::default();
    let mut rng = DotNetRng::new(7);

    ai.on_move_performed();
    assert_eq!(ai.roll_move_with_combat(&mut rng, &combat).id, "Y");
    let after_entering_y = rng.counter();
    assert_eq!(ai.roll_move_with_combat(&mut rng, &combat).id, "Y");
    assert_eq!(rng.counter(), after_entering_y);
    ai.on_move_performed();
    assert_eq!(ai.roll_move_with_combat(&mut rng, &combat).id, "X");
}

/// The performed flag belongs to the state that was current when the move
/// STARTED. C# `MoveState.PerformMove` sets `_performedAtLeastOnce` before the
/// body (`MoveState.cs:55-60`); a `SetMoveImmediate` from inside the body
/// (the Waterfall Giant killed by thorns during STOMP is forced onto
/// ABOUT_TO_BLOW) leaves that state through `OnExitState` (`:62-65`) and enters
/// a state that has not performed; `OnMovePerformed` afterwards only sets
/// `_performedFirstMove` (`MonsterMoveStateMachine.cs:51-54`). So the forced
/// `must_perform_once` state is held for the next turn.
///
/// The post-move half is `on_move_performed_state(<the move that ran>)` (#894),
/// which the combat walk calls right after each move; it flags only a move
/// that is still current.
#[test]
fn a_must_perform_once_state_forced_mid_move_is_held() {
    let mut states = BTreeMap::new();
    states.insert("X", move_state("X", 1, "X", false));
    states.insert("Y", move_state("Y", 2, "Z", true));
    states.insert("Z", move_state("Z", 3, "Z", false));
    let mut ai = MonsterAi::new(states, "X");
    let mut rng = DotNetRng::new(7);

    ai.begin_move(); // X's body starts...
    ai.force_current_state("Y"); // ...and forces Y,
    ai.on_move_performed_state("X"); // then X's move completes.
    assert_eq!(ai.roll_move(&mut rng).id, "Y", "Y has not performed: held");

    ai.begin_move();
    ai.on_move_performed_state("Y");
    assert_eq!(ai.roll_move(&mut rng).id, "Z", "performing Y releases it");
}

/// The STUNNED analogue: a `SetMoveImmediate(stun)` (`MonsterModel.cs:562-575`)
/// whose `CanTransitionAway` guard (`:564`) reads the move that is still
/// running.
fn stun_self(state: &mut CombatState, owner: CreatureUid) {
    sts2_core::hooks::apply_effects(
        state,
        &[sts2_core::hooks::Effect::StunMonster {
            target: owner,
            state_id: "S",
            next_move: None,
        }],
    );
}

/// #830 (#862) x #806 (#894): the case neither side covered alone. A monster
/// whose FIRST must-perform move stuns itself inside its own body (the shape
/// of Shriek tripping on Thorns mid-Thrash, or a Ravenous devour inside the
/// slug's own turn). C# has already marked that move performed when the body
/// runs (`_performedAtLeastOnce = true`, `MoveState.cs:57`, then `await
/// _onPerform`, `:59`), so `SetMoveImmediate`'s `CanTransitionAway` guard
/// (`MonsterModel.cs:564`) lets the stun in; the stun is a fresh, unperformed
/// must-perform state, so the boundary roll holds it and the monster serves
/// it next turn, then resumes. #894 alone refused the stun (no pre-body mark:
/// the guard saw an unperformed must-perform move); #862 alone had no stun
/// guard at all. Driven through the real enemy walk.
#[test]
fn a_self_stun_inside_a_first_must_perform_move_lands_and_is_held() {
    let mut states = BTreeMap::new();
    states.insert(
        "A",
        MonsterState::Move(MoveState {
            id: "A",
            effect: stun_self,
            intents: smallvec![Intent::Buff {
                kind: sts2_core::PowerKind::Strength,
                amount: 0
            }],
            follow_up: Some("B"),
            must_perform_once: true,
        }),
    );
    states.insert(
        "S",
        MonsterState::Move(MoveState {
            id: "S",
            effect: noop_effect,
            intents: smallvec![Intent::Stun],
            follow_up: Some("B"),
            must_perform_once: true,
        }),
    );
    states.insert("B", move_state("B", 0, "B", false));
    let mut ai = MonsterAi::new(states, "A");

    let mut state = CombatState::default();
    state.player.current_hp = 50;
    state.player.max_hp = 50;
    let _ = state.init_player_creature();
    let uid = state.spawn_enemy(sts2_core::MonsterKind::Nibbit);
    ai.set_owner(uid);
    let intent = ai.current_move().intents.first().copied();
    state.set_creature_ai(uid, ai);
    let c = state
        .creatures
        .iter_mut()
        .find(|c| c.id == uid.0)
        .expect("enemy");
    c.current_hp = 30;
    c.max_hp = 30;
    c.intent = intent;
    state.round = 1;
    let current = |s: &CombatState| {
        s.creatures
            .iter()
            .find(|c| c.id == uid.0)
            .and_then(|c| c.ai.as_ref())
            .map(MonsterAi::current_state_id)
    };
    let telegraph = |s: &CombatState| {
        s.creatures
            .iter()
            .find(|c| c.id == uid.0)
            .and_then(|c| c.intent)
    };

    state.phase = sts2_core::CombatPhase::EnemyTurnExec;
    let _ = sts2_core::combat_loop::enemy_turn_exec(&mut state);
    assert_eq!(
        current(&state),
        Some("S"),
        "the stun landed inside A's body"
    );
    let _ = sts2_core::combat_loop::enemy_turn_end(&mut state);
    assert_eq!(
        current(&state),
        Some("S"),
        "the fresh stun is unserved, so the boundary roll holds it"
    );
    assert_eq!(telegraph(&state), Some(Intent::Stun));

    // Next enemy turn: the stun is served, then the machine resumes at the
    // move it was stunned out of (`StateLog.Last()`, Creature.cs:532-535).
    state.phase = sts2_core::CombatPhase::EnemyTurnExec;
    let _ = sts2_core::combat_loop::enemy_turn_exec(&mut state);
    let _ = sts2_core::combat_loop::enemy_turn_end(&mut state);
    assert_eq!(current(&state), Some("A"));
}

/// The same force landing between two turns (on the player's turn, after the
/// enemy's move completed) is held too, and `on_move_performed` (the atomic
/// begin + complete) still releases it — the pre-existing contract.
#[test]
fn a_must_perform_once_state_forced_between_turns_is_held_until_performed() {
    let mut states = BTreeMap::new();
    states.insert("X", move_state("X", 1, "X", false));
    states.insert("Y", move_state("Y", 2, "Z", true));
    states.insert("Z", move_state("Z", 3, "Z", false));
    let mut ai = MonsterAi::new(states, "X");
    let mut rng = DotNetRng::new(7);

    ai.on_move_performed();
    assert_eq!(ai.roll_move(&mut rng).id, "X");
    ai.force_current_state("Y");
    assert_eq!(ai.roll_move(&mut rng).id, "Y");
    ai.on_move_performed();
    assert_eq!(ai.roll_move(&mut rng).id, "Z");
}

// ── ConditionalBranch dead ends ────────────────────────────────────────

/// C# `ConditionalBranchState.GetNextState` throws when every branch
/// evaluates to 0. A monster graph that can reach such a state is a content
/// bug, and the engine must surface it loudly rather than silently pick a
/// branch.
#[test]
#[should_panic(expected = "no condition matched")]
fn conditional_branch_with_no_matching_condition_panics() {
    let mut states = BTreeMap::new();
    states.insert("atk", move_state("atk", 5, "check", false));
    states.insert(
        "check",
        MonsterState::ConditionalBranch(ConditionalBranchState {
            id: "check",
            branches: smallvec![ConditionalBranch {
                condition: ConditionFn::Stateless(|_| false),
                state_id: "alt",
            }],
        }),
    );
    states.insert("alt", move_state("alt", 9, "atk", false));
    let mut ai = MonsterAi::new(states, "atk");
    ai.on_move_performed();
    let combat = CombatState::default();
    let _ = ai.roll_move_with_combat_seeded(0, &combat);
}

/// The no-combat roll path deliberately *skips* owner/counter-aware
/// predicates (they have no context to resolve against) and relies on a
/// stateless always-true fallthrough being present. A graph whose only
/// branches are owner-aware has no such fallthrough, and the degraded path
/// says so instead of guessing — see the `get_next_state_no_combat` comment
/// block (bead `sts2-rs-msi1`).
#[test]
#[should_panic(expected = "no stateless condition matched")]
fn no_combat_path_panics_when_every_branch_is_owner_aware() {
    let mut states = BTreeMap::new();
    states.insert("atk", move_state("atk", 5, "check", false));
    states.insert(
        "check",
        MonsterState::ConditionalBranch(ConditionalBranchState {
            id: "check",
            branches: smallvec![ConditionalBranch {
                condition: ConditionFn::OwnerAware(|_, _| true),
                state_id: "alt",
            }],
        }),
    );
    states.insert("alt", move_state("alt", 9, "atk", false));
    let mut ai = MonsterAi::new(states, "atk");
    ai.on_move_performed();
    let _ = ai.roll_move_seeded(0);
}

// ── current_move / current_move_opt when parked off a Move ─────────────

/// `current_move_opt` is the non-panicking accessor for callers that consult
/// the move table opportunistically (wire reconstruction). Reaching a
/// non-Move `current` needs the move table to be rewritten under the AI —
/// the `Arc::make_mut` seam real spawn-time tweaks use — which is exactly the
/// "parked on an unresolved branch" situation the accessor documents.
#[test]
fn current_move_opt_is_none_when_parked_on_a_branch_state() {
    let mut states = BTreeMap::new();
    states.insert("A", move_state("A", 3, "A", false));
    let mut ai = MonsterAi::new(states, "A");
    assert!(
        ai.current_move_opt().is_some(),
        "resolved to a Move at construction"
    );

    Arc::make_mut(&mut ai.states).insert(
        "A",
        MonsterState::ConditionalBranch(ConditionalBranchState {
            id: "A",
            branches: smallvec![ConditionalBranch {
                condition: ConditionFn::Stateless(|_| true),
                state_id: "A",
            }],
        }),
    );
    assert!(
        ai.current_move_opt().is_none(),
        "a branch state has no telegraph to report",
    );
}

#[test]
#[should_panic(expected = "is not a MoveState")]
fn current_move_panics_when_parked_on_a_branch_state() {
    let mut states = BTreeMap::new();
    states.insert("A", move_state("A", 3, "A", false));
    let mut ai = MonsterAi::new(states, "A");
    Arc::make_mut(&mut ai.states).insert(
        "A",
        MonsterState::ConditionalBranch(ConditionalBranchState {
            id: "A",
            branches: smallvec![ConditionalBranch {
                condition: ConditionFn::Stateless(|_| true),
                state_id: "A",
            }],
        }),
    );
    let _ = ai.current_move();
}

/// `force_current_state_by_name` resolves against move states only: a wire
/// dict naming a *branch* id must miss rather than park the AI somewhere
/// `current_move()` would panic on.
#[test]
fn force_current_state_by_name_rejects_a_known_branch_state_id() {
    let mut states = BTreeMap::new();
    states.insert("A", move_state("A", 3, "ROOT", false));
    states.insert(
        "ROOT",
        MonsterState::RandomBranch(RandomBranchState {
            id: "ROOT",
            branches: smallvec![WeightedBranch {
                state_id: "A",
                repeat_type: MoveRepeatType::CanRepeatForever,
                max_times: 1,
                weight: 1.0,
                cooldown: 0,
            }],
        }),
    );
    let mut ai = MonsterAi::new(states, "A");
    assert!(
        !ai.force_current_state_by_name("ROOT"),
        "branch ids are not restorable move states",
    );
    assert_eq!(
        ai.current_state_id(),
        "A",
        "the miss leaves the AI untouched"
    );
    assert!(
        !ai.force_current_state_by_name_relaxed("ROOT"),
        "the relaxed variant filters non-moves too",
    );
    assert_eq!(ai.current_state_id(), "A");
}

// ── transplant_instance_state_from ─────────────────────────────────────

fn transplant_graph() -> BTreeMap<StateId, MonsterState> {
    let mut states = BTreeMap::new();
    states.insert("X", move_state("X", 1, "Y", false));
    states.insert("Y", move_state("Y", 2, "X", false));
    states.insert("Z", move_state("Z", 3, "X", false));
    states
}

/// The wire carries only `current`; everything else that feeds the next
/// transition (state log, performed flags, counter, a pending
/// `SetMoveImmediate`) has to ride across from the mirror. `current` itself
/// is explicitly *not* transplanted — reality wins on the one observable
/// that is on the wire.
#[test]
fn transplant_carries_hidden_instance_state_but_not_current() {
    let mut mirror = MonsterAi::new(transplant_graph(), "X");
    mirror.on_move_performed();
    assert_eq!(mirror.roll_move_seeded(0).id, "Y");
    mirror.set_counter(5);
    mirror.force_next_state("Z");

    let mut rebuilt = MonsterAi::new(transplant_graph(), "X");
    assert_eq!(rebuilt.counter(), 0);
    assert!(rebuilt.move_history().is_empty());

    rebuilt.transplant_instance_state_from(&mirror);

    assert_eq!(rebuilt.current_state_id(), "X", "`current` is left alone");
    assert_eq!(rebuilt.counter(), 5);
    assert_eq!(rebuilt.move_history(), mirror.move_history());
    assert_eq!(rebuilt.forced_next_state(), Some("Z"));

    // performed_first_move rode across: a fresh AI would hold "X" for its
    // first roll, the transplanted one honours the queued override instead.
    let mut control = MonsterAi::new(transplant_graph(), "X");
    assert_eq!(control.roll_move_seeded(0).id, "X", "fresh AI holds move 1");
    assert_eq!(
        rebuilt.roll_move_seeded(0).id,
        "Z",
        "transplanted AI is past its first move and consumes the override",
    );
}

/// The slot is the one field where the *fresh* build wins if it has one:
/// dynamically-slotted spawns (Ovicopter eggs, Fabricator bots) carry no
/// slot on the wire, so an unset slot inherits the mirror's — but an already
/// assigned slot is not overwritten.
#[test]
fn transplant_fills_slot_only_when_unset() {
    let mut mirror = MonsterAi::new(transplant_graph(), "X");
    mirror.set_slot(3);

    let mut unslotted = MonsterAi::new(transplant_graph(), "X");
    assert_eq!(unslotted.slot(), None);
    unslotted.transplant_instance_state_from(&mirror);
    assert_eq!(
        unslotted.slot(),
        Some(3),
        "unset slot inherits the mirror's"
    );

    let mut slotted = MonsterAi::new(transplant_graph(), "X");
    slotted.set_slot(7);
    slotted.transplant_instance_state_from(&mirror);
    assert_eq!(slotted.slot(), Some(7), "an assigned slot survives");
}

// ── peek_intents degraded branches ─────────────────────────────────────

/// A `must_perform_once` move that has been entered but not performed cannot
/// be transitioned away from, so every lookahead step re-reports the same
/// telegraph instead of walking the graph.
#[test]
fn peek_repeats_a_held_must_perform_once_move() {
    let mut states = BTreeMap::new();
    states.insert("X", move_state("X", 1, "Y", false));
    states.insert("Y", move_state("Y", 2, "X", true));
    let mut ai = MonsterAi::new(states, "X");
    ai.on_move_performed();
    assert_eq!(ai.roll_move_seeded(0).id, "Y");

    let combat = CombatState::default();
    let peeks = ai.peek_intents(3, &combat);
    assert_eq!(peeks.len(), 3);
    for (i, p) in peeks.iter().enumerate() {
        assert_eq!(
            p.intent,
            Intent::Attack { damage: 2 },
            "step {} should still be Y's telegraph",
            i + 1
        );
        assert!((p.certainty - 1.0).abs() < 1e-6, "a held move is certain");
        assert_eq!(p.step, u8::try_from(i + 1).unwrap());
    }
}

/// Lookahead must stay total: a `ConditionalBranch` whose predicates all
/// read false against the *current* snapshot falls back to the first branch
/// rather than aborting the walk (the live roll would panic, but peek is a
/// read-only feature feeder and has to return something).
#[test]
fn peek_falls_back_to_first_branch_when_no_condition_matches() {
    let mut states = BTreeMap::new();
    states.insert("atk", move_state("atk", 5, "check", false));
    states.insert(
        "check",
        MonsterState::ConditionalBranch(ConditionalBranchState {
            id: "check",
            branches: smallvec![
                ConditionalBranch {
                    condition: ConditionFn::Stateless(|_| false),
                    state_id: "first",
                },
                ConditionalBranch {
                    condition: ConditionFn::Stateless(|_| false),
                    state_id: "second",
                },
            ],
        }),
    );
    states.insert("first", move_state("first", 11, "atk", false));
    states.insert("second", move_state("second", 22, "atk", false));
    let mut ai = MonsterAi::new(states, "atk");
    ai.on_move_performed();

    let combat = CombatState::default();
    let peeks = ai.peek_intents(1, &combat);
    assert_eq!(peeks.len(), 1);
    assert_eq!(peeks[0].intent, Intent::Attack { damage: 11 });
    assert!((peeks[0].certainty - 1.0).abs() < 1e-6);
}

/// When every weighted branch zeroes out, the live resolver burns one RNG
/// tick and returns branch 0 (bead `sts2-rs-zdjjx`). Peek has no RNG, so it
/// mirrors the *pick* and reports zero certainty for it.
#[test]
fn peek_reports_zero_certainty_when_all_branches_zero_out() {
    let mut states = BTreeMap::new();
    states.insert("A", move_state("A", 4, "ROOT", false));
    states.insert(
        "ROOT",
        MonsterState::RandomBranch(RandomBranchState {
            id: "ROOT",
            // Single branch back to A, CannotRepeat: with A as the last
            // logged move its weight is 0, so the whole branch sums to 0.
            branches: smallvec![WeightedBranch {
                state_id: "A",
                repeat_type: MoveRepeatType::CannotRepeat,
                max_times: 1,
                weight: 1.0,
                cooldown: 0,
            }],
        }),
    );
    let mut ai = MonsterAi::new(states, "A");
    ai.on_move_performed();

    let combat = CombatState::default();
    let peeks = ai.peek_intents(1, &combat);
    assert_eq!(peeks.len(), 1);
    assert_eq!(peeks[0].intent, Intent::Attack { damage: 4 });
    assert!(
        peeks[0].certainty.abs() < 1e-6,
        "an all-zero branch pick carries no confidence, got {}",
        peeks[0].certainty,
    );
}

// ── property: the roll contract over arbitrary weighted graphs ─────────

const MOVE_IDS: [StateId; 4] = ["M0", "M1", "M2", "M3"];

fn draw_repeat_type(tc: &TestCase) -> MoveRepeatType {
    match tc.draw(gs::integers::<u8>().min_value(0).max_value(3)) {
        0 => MoveRepeatType::CanRepeatForever,
        1 => MoveRepeatType::CannotRepeat,
        2 => MoveRepeatType::CanRepeatXTimes,
        _ => MoveRepeatType::UseOnlyOnce,
    }
}

/// Over arbitrary `RandomBranch` graphs (random weights — including all-zero
/// sums — repeat types, cooldowns and `must_perform_once` flags), a roll:
///
/// * always lands on a `Move` state that exists in the table (never panics,
///   never parks on a branch), and
/// * consumes **exactly one** `MonsterAi` tick — C#
///   `RandomBranchState.GetNextState` always calls `rng.NextFloat(max)` once
///   and `MoveState.GetNextState` never touches the RNG, so a
///   move→branch→move hop is one tick regardless of how the weights land.
///
/// `peek_intents` is asserted pure alongside: same answer twice, no visible
/// mutation of the AI.
#[hegel::test(test_cases = 200)]
fn roll_lands_on_a_move_and_costs_one_tick_per_hop(tc: TestCase) {
    let n_moves = tc.draw(gs::integers::<usize>().min_value(1).max_value(4));
    let mut states = BTreeMap::new();
    let mut branches = smallvec![];
    for &id in MOVE_IDS.iter().take(n_moves) {
        let must_perform_once = tc.draw(gs::booleans());
        states.insert(id, move_state(id, 1, "ROOT", must_perform_once));
        branches.push(WeightedBranch {
            state_id: id,
            repeat_type: draw_repeat_type(&tc),
            max_times: tc.draw(gs::integers::<u8>().min_value(1).max_value(3)),
            // Integer tenths keep the weights exactly representable while
            // still spanning zero-weight branches.
            weight: f64::from(tc.draw(gs::integers::<u8>().min_value(0).max_value(50))) / 10.0,
            cooldown: tc.draw(gs::integers::<u8>().min_value(0).max_value(3)),
        });
    }
    states.insert(
        "ROOT",
        MonsterState::RandomBranch(RandomBranchState {
            id: "ROOT",
            branches,
        }),
    );

    let seed = tc.draw(gs::integers::<i32>().min_value(-10_000).max_value(10_000));
    let mut ai = MonsterAi::new(states, "ROOT");
    let mut rng = DotNetRng::new(seed);
    let combat = CombatState::default();

    for step in 0..20 {
        // peek is a pure read: identical answers, no state disturbance.
        let before_current = ai.current_state_id();
        let before_log = ai.move_history().to_vec();
        let first = ai.peek_intents(3, &combat);
        let second = ai.peek_intents(3, &combat);
        assert_eq!(first, second, "peek is not deterministic at step {step}");
        assert_eq!(ai.current_state_id(), before_current);
        assert_eq!(ai.move_history(), before_log.as_slice());

        ai.on_move_performed();
        let before_ticks = rng.counter();
        let landed = ai.roll_move(&mut rng).id;
        assert!(
            MOVE_IDS[..n_moves].contains(&landed),
            "step {step} landed on unknown state {landed}",
        );
        assert_eq!(ai.current_state_id(), landed, "current tracks the landing");
        assert_eq!(
            rng.counter() - before_ticks,
            1,
            "step {step}: one branch hop must cost exactly one tick",
        );
    }
}
