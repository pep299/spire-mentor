//! `BeforeHandDraw` power arms fire in the owner's power-list order.
//!
//! C# `Hook.BeforeHandDraw` (v0.111.0 `Hook.cs:744-760`) awaits each
//! listener of `IterateCombatHookListeners` in turn, and that snapshot lists
//! the creature's powers in insertion order (`Creature.cs:621`
//! `_powers.Add`). So a CallOfTheVoid applied BEFORE SentryMode puts its
//! Ethereal pool card in hand ahead of Sweeping Gaze, and the reverse order
//! puts Sweeping Gaze first. Necrobinder seed 1008 (v0.111.0) desynced here:
//! the dispatcher ran a fixed SentryMode-then-CotV sequence.

use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{CombatState, Power, PowerKind, PowerUid};

use sts2_content::cards::dispatch as content_dispatch;

fn state_with_powers(kinds: &[PowerKind]) -> CombatState {
    let mut state = CombatState {
        run_rng: sts2_core::run_rng_set::RunRngSet::new(12345_i32),
        ..Default::default()
    };
    state.player.character = Some(CharacterId::Necrobinder);
    for (i, &kind) in kinds.iter().enumerate() {
        let pid = PowerUid::new(i as u32 + 1).expect("nz");
        state.powers.push(Power {
            id: pid.0,
            kind,
            amount: 1,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(pid);
    }
    state
}

fn hand_ids(state: &CombatState) -> Vec<CardId> {
    state
        .player
        .hand
        .iter()
        .map(|uid| {
            state
                .cards
                .iter()
                .find(|c| c.uid == *uid)
                .expect("hand card in arena")
                .card_id
        })
        .collect()
}

#[test]
fn call_of_the_void_before_sentry_mode_puts_pool_card_first() {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    let mut state = state_with_powers(&[PowerKind::CallOfTheVoid, PowerKind::SentryMode]);
    sts2_core::hooks::dispatch_before_hand_draw_powers(&mut state);
    let ids = hand_ids(&state);
    assert_eq!(ids.len(), 2, "one CotV card + one Sweeping Gaze: {ids:?}");
    assert_ne!(ids[0], CardId::SweepingGaze, "CotV card first: {ids:?}");
    assert_eq!(
        ids[1],
        CardId::SweepingGaze,
        "Sweeping Gaze second: {ids:?}"
    );
}

#[test]
fn sentry_mode_before_call_of_the_void_puts_sweeping_gaze_first() {
    content_dispatch::install_default_registry();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    let mut state = state_with_powers(&[PowerKind::SentryMode, PowerKind::CallOfTheVoid]);
    sts2_core::hooks::dispatch_before_hand_draw_powers(&mut state);
    let ids = hand_ids(&state);
    assert_eq!(ids.len(), 2, "one Sweeping Gaze + one CotV card: {ids:?}");
    assert_eq!(ids[0], CardId::SweepingGaze, "Sweeping Gaze first: {ids:?}");
    assert_ne!(ids[1], CardId::SweepingGaze, "CotV card second: {ids:?}");
}
