//! Enemy on-death "second phase" dispatch — the Rust analogue of a C#
//! power's `AfterDeath` that spawns reinforcements (bead `sts2-rs-h9u`
//! Phase-7 sliver; first consumer: `InfestedPower`).
//!
//! Mirrors the [`crate::potion_effects`] / [`crate::relic_effects`]
//! registry shape: per-[`PowerKind`] handlers, keyed off the power the
//! dying creature carries, with the handler bodies living in
//! `sts2-content` (so the spawned reinforcements get their real
//! `MonsterAi` factories) and registered into this core-side storage at
//! run construction.
//!
//! **C# parity.** `InfestedPower.AfterDeath`
//! (`../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Powers/InfestedPower.cs`)
//! spawns 4 stunned `Wriggler`s when its owner (the PhrogParasite elite)
//! dies, and `ShouldStopCombatFromEnding()` returns `true` so the combat
//! does NOT resolve as a victory the instant the phrog — the only live
//! enemy — drops. [`fire_pending`] models both halves: it fires each
//! pending handler and reports whether any fired, so the
//! all-enemies-dead branch in [`crate::combat_loop::check_combat_end`]
//! can keep the combat open when reinforcements were summoned.

use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::sync::{OnceLock, RwLock};

use crate::powers::PowerKind;
use crate::{CombatState, CreatureUid};

/// Handler signature for a power's on-death "second phase". `owner` is
/// the uid of the just-died creature that carried the triggering power.
pub type EnemyDeathFn = fn(&mut CombatState, owner: CreatureUid);

// `BTreeMap` (not `HashMap`) to match the sibling dispatch registries
// (`relic_effects` / `potion_effects`) and the workspace `disallowed_methods`
// lint; lookups here are point queries so iteration order is irrelevant, but
// the house style keeps the registry deterministic.
static REGISTRY: OnceLock<RwLock<BTreeMap<PowerKind, EnemyDeathFn>>> = OnceLock::new();

fn registry() -> &'static RwLock<BTreeMap<PowerKind, EnemyDeathFn>> {
    REGISTRY.get_or_init(|| RwLock::new(BTreeMap::new()))
}

/// Register a single handler. Overwrites any previous entry.
pub fn register(kind: PowerKind, handler: EnemyDeathFn) {
    registry()
        .write()
        .expect("enemy-death dispatch registry poisoned")
        .insert(kind, handler);
}

/// Register many handlers in one go.
pub fn register_many(entries: &[(PowerKind, EnemyDeathFn)]) {
    let mut guard = registry()
        .write()
        .expect("enemy-death dispatch registry poisoned");
    for &(kind, f) in entries {
        guard.insert(kind, f);
    }
}

/// Look up the registered handler for `kind`, or `None` if absent.
#[must_use]
pub fn lookup(kind: PowerKind) -> Option<EnemyDeathFn> {
    registry()
        .read()
        .expect("enemy-death dispatch registry poisoned")
        .get(&kind)
        .copied()
}

/// Number of currently-registered handlers.
#[must_use]
pub fn registered_count() -> usize {
    registry()
        .read()
        .expect("enemy-death dispatch registry poisoned")
        .len()
}

// The tail of the kill: handlers that must run AFTER everything the killing
// hit still does once its `AfterDeath` handler has fired. C#
// `CreatureCmd.Damage` runs `Hook.AfterDamageGiven` for every result
// (`CreatureCmd.cs:388-391`) and only then `Kill(killedCreatures)` (`:409`),
// whose `KillWithoutCheckingWinCondition` runs `Hook.AfterDeath` (`:519`) and
// then `creature.RemoveAllPowersAfterDeath()` (`:533`). The engine fires the
// `AfterDeath` handler early, at the damage chokepoint's just-died witness,
// so a step that C# orders after the killing hit's `AfterDamageGiven` has to
// be split off and run by [`finish_kill`] once those hooks are done.
static AFTER_KILL_REGISTRY: OnceLock<RwLock<BTreeMap<PowerKind, EnemyDeathFn>>> = OnceLock::new();

fn after_kill_registry() -> &'static RwLock<BTreeMap<PowerKind, EnemyDeathFn>> {
    AFTER_KILL_REGISTRY.get_or_init(|| RwLock::new(BTreeMap::new()))
}

/// Register the kill-tail handler for a death power whose [`EnemyDeathFn`]
/// fired. Overwrites any previous entry. See [`finish_kill`].
pub fn register_after_kill(kind: PowerKind, handler: EnemyDeathFn) {
    after_kill_registry()
        .write()
        .expect("enemy-death after-kill registry poisoned")
        .insert(kind, handler);
}

/// Run the kill-tail handler registered for `kind`, if any, on `owner`.
///
/// `kind` is the death power whose handler just fired for `owner`
/// ([`fire_for_owner`]'s return value). The damage chokepoint calls this
/// after the killing hit's `AfterDamageGiven` hooks and the relic `AfterDeath`
/// bus; [`fire_pending`], which has no hit around it, calls it straight after
/// the handler.
pub fn finish_kill(state: &mut CombatState, owner: CreatureUid, kind: PowerKind) {
    let handler = after_kill_registry()
        .read()
        .expect("enemy-death after-kill registry poisoned")
        .get(&kind)
        .copied();
    if let Some(handler) = handler {
        handler(state, owner);
    }
}

/// Fire any pending on-death second-phase effects and report whether any
/// fired.
///
/// A pending effect is a dead enemy (`current_hp <= 0`) carrying a power
/// that (a) has a registered [`EnemyDeathFn`] and (b) still has a
/// non-zero `amount` (the un-fired marker). After a handler runs, the
/// triggering power's `amount` is zeroed **unless** the handler revived
/// its owner (i.e. `owner.current_hp > 0` after the handler returns) —
/// in that case the power must remain armed so it fires again if the
/// creature dies a second time (C# `AdaptablePower.ShouldStopCombatFromEnding`
/// multi-revive semantics; `TestSubject.cs:218-278`). For one-shot handlers
/// (e.g. `InfestedPower.AfterDeath`) the owner stays dead and the amount
/// is still zeroed as before.
///
/// Returns `true` iff at least one handler fired — the caller
/// ([`crate::combat_loop::check_combat_end`]) uses this to decide whether
/// to keep the combat open (C# `ShouldStopCombatFromEnding`).
pub fn fire_pending(state: &mut CombatState) -> bool {
    // Snapshot the (owner, power-id, kind) triples to fire, releasing the
    // immutable borrow on `creatures`/`powers` before the mutable handler
    // loop. Iterate `enemy_uids` in canonical order for determinism.
    let pending: Vec<(CreatureUid, NonZeroU32, PowerKind)> = state
        .enemy_uids
        .iter()
        .filter_map(|uid| {
            let c = state.creatures.iter().find(|c| c.id == uid.0)?;
            if c.current_hp > 0 {
                return None;
            }
            // First registered, un-fired death power on this corpse.
            c.powers.iter().find_map(|puid| {
                let p = state.powers.get_by_id(puid.0)?;
                // `Adaptable` (TestSubject) is revived at the corpse's NEXT
                // enemy turn, NOT at the death chokepoint (C# defers the
                // RESPAWN_MOVE). Its handler stays registered for the
                // enemy-turn revive pass in `combat_loop::enemy_turn_exec`
                // to `lookup`, but must NOT fire here. Bead `sts2-rs-jnf4a`.
                if p.kind != PowerKind::Adaptable && p.amount != 0 && lookup(p.kind).is_some() {
                    Some((*uid, p.id, p.kind))
                } else {
                    None
                }
            })
        })
        .collect();

    let fired = !pending.is_empty();
    for (owner, power_id, kind) in pending {
        if let Some(handler) = lookup(kind) {
            handler(state, owner);
        }
        // No killing hit is in flight here, so its `AfterDamageGiven` hooks
        // are long done: the kill tail runs straight away.
        finish_kill(state, owner, kind);
        // Mark fired: zero the triggering power so a later combat-end
        // check (after the reinforcements die) does not re-summon.
        //
        // Exception: if the handler REVIVED the owner (current_hp > 0 now),
        // skip the zero so the power can fire again when the creature dies
        // a second time (C# multi-revive pattern — AdaptablePower on
        // TestSubject). One-shot handlers (InfestedPower — owner stays dead)
        // are unaffected: the owner has current_hp <= 0, so the zero applies.
        let owner_revived = state
            .creatures
            .iter()
            .any(|c| c.id == owner.0 && c.current_hp > 0);
        if !owner_revived {
            if let Some(p) = state.powers.get_mut_by_id(power_id) {
                p.amount = 0;
            }
        }
    }
    fired
}

/// Would [`fire_pending`] fire anything right now? The same corpse /
/// armed-power / registered-handler filter, with no side effects.
///
/// C# runs a dying creature's `AfterDeath` (Infested's Wriggler spawn) inside
/// the killing `CreatureCmd.Damage`, so by the time a caller asks
/// `CombatManager.IsOverOrEnding` the reinforcements already stand — combat is
/// not ending. Rust defers that spawn to [`fire_pending`] at the combat-end
/// chokepoint, so a side-effect-free "is combat ending?" read has to treat a
/// still-armed death power as "the fight goes on".
#[must_use]
pub fn has_pending(state: &CombatState) -> bool {
    state.enemy_uids.iter().any(|uid| {
        state
            .creatures
            .iter()
            .find(|c| c.id == uid.0)
            .filter(|c| c.current_hp <= 0)
            .is_some_and(|c| {
                c.powers.iter().any(|puid| {
                    state.powers.get_by_id(puid.0).is_some_and(|p| {
                        p.kind != PowerKind::Adaptable && p.amount != 0 && lookup(p.kind).is_some()
                    })
                })
            })
    })
}

/// Fire the on-death second-phase effect for a SINGLE just-died creature,
/// at the moment of its death.
///
/// This is the per-death analogue of [`fire_pending`] (which scans every
/// corpse at the combat-end chokepoint). Some C# `AfterDeath` powers must
/// react the instant their owner is killed — even while other enemies are
/// still alive — because they spawn a replacement creature in the same
/// combat (e.g. `StockPower.AfterDeath` respawning a fresh `Axebot` at the
/// dead bot's slot; C#
/// `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Powers/StockPower.cs`).
/// Without this, a mid-combat kill of one of two Axebots would drop the
/// dead bot and never respawn it, diverging from C# (which keeps two live
/// enemies via the respawn). Called from the `just_died` chokepoint in
/// [`crate::damage`].
///
/// Fires the first registered, un-fired (`amount != 0`) death power on
/// `owner` if `owner` is a corpse (`current_hp <= 0`). The triggering
/// power's `amount` is zeroed afterwards (the corpse stays dead — a
/// respawn creates a *new* creature, it does not revive the owner), so
/// the later combat-end [`fire_pending`] sweep never re-fires it. Returns
/// the death power whose handler fired, `None` if none did; the caller hands
/// it to [`finish_kill`] once the killing hit's `AfterDamageGiven` hooks have
/// run.
pub fn fire_for_owner(state: &mut CombatState, owner: CreatureUid) -> Option<PowerKind> {
    // Only fire for an actual corpse.
    let triple: Option<(NonZeroU32, PowerKind)> = state
        .creatures
        .iter()
        .find(|c| c.id == owner.0)
        .filter(|c| c.current_hp <= 0)
        .and_then(|c| {
            c.powers.iter().find_map(|puid| {
                let p = state.powers.get_by_id(puid.0)?;
                // See `fire_pending`: `Adaptable` is revived at the next
                // enemy turn, not at death. Bead `sts2-rs-jnf4a`.
                if p.kind != PowerKind::Adaptable && p.amount != 0 && lookup(p.kind).is_some() {
                    Some((p.id, p.kind))
                } else {
                    None
                }
            })
        });
    let (power_id, kind) = triple?;
    if let Some(handler) = lookup(kind) {
        handler(state, owner);
    }
    // Mark fired so the combat-end sweep does not re-fire. A respawn
    // handler leaves the owner dead (it spawns a separate creature), so —
    // unlike the revive case in `fire_pending` — the zero always applies.
    let owner_revived = state
        .creatures
        .iter()
        .any(|c| c.id == owner.0 && c.current_hp > 0);
    if !owner_revived {
        if let Some(p) = state.powers.get_mut_by_id(power_id) {
            p.amount = 0;
        }
    }
    Some(kind)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use smallvec::SmallVec;

    use super::*;
    use crate::{Creature, IntentHistory, MonsterId, MonsterKind, Power, PowerUid};

    /// A handler that records it fired by stamping `block = 99` on the
    /// corpse — proves `fire_for_owner` dispatched the registered fn.
    fn marker_handler(state: &mut CombatState, owner: CreatureUid) {
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == owner.0) {
            c.block = 99;
        }
    }

    fn corpse_with_power(kind: PowerKind, amount: i32) -> (CombatState, CreatureUid) {
        let mut state = CombatState::default();
        let uid = CreatureUid(NonZeroU32::new(1).expect("nz"));
        let puid = PowerUid(NonZeroU32::new(1).expect("nz"));
        state.powers.push(Power::new(puid.0, kind, amount));
        state.creatures.push(Creature {
            id: uid.0,
            current_hp: 0, // already a corpse
            max_hp: 40,
            block: 0,
            powers: SmallVec::from_slice(&[puid]),
            species: Some(MonsterKind::Axebot),
            monster_id: MonsterId("Axebot"),
            intent: None,
            intent_history: IntentHistory::default(),
            ai: None,
        });
        state.enemy_uids.push(uid);
        (state, uid)
    }

    #[test]
    fn fire_for_owner_dispatches_and_zeroes_the_power() {
        register(PowerKind::Stock, marker_handler);
        let (mut state, uid) = corpse_with_power(PowerKind::Stock, 2);
        assert_eq!(
            fire_for_owner(&mut state, uid),
            Some(PowerKind::Stock),
            "handler should fire"
        );
        let c = state.creatures.iter().find(|c| c.id == uid.0).unwrap();
        assert_eq!(c.block, 99, "registered handler ran");
        // The triggering power is zeroed so the combat-end sweep won't re-fire.
        let amt = state
            .powers
            .iter()
            .find(|p| p.id == uid.0)
            .map(|p| p.amount);
        assert_eq!(amt, Some(0), "triggering power zeroed after firing");
    }

    #[test]
    fn fire_for_owner_skips_live_creature() {
        register(PowerKind::Stock, marker_handler);
        let (mut state, uid) = corpse_with_power(PowerKind::Stock, 2);
        // Resurrect it: a live creature is not a death trigger.
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == uid.0) {
            c.current_hp = 10;
        }
        assert_eq!(
            fire_for_owner(&mut state, uid),
            None,
            "no fire for a live creature"
        );
    }

    /// `has_pending` is `fire_pending`'s filter without the side effects: true
    /// for an armed corpse, false once the handler fired (amount zeroed) and
    /// for an `Adaptable` corpse (revived at the next enemy turn instead).
    #[test]
    fn has_pending_mirrors_fire_pending_filter() {
        register(PowerKind::Stock, marker_handler);
        let (mut state, _) = corpse_with_power(PowerKind::Stock, 2);
        assert!(has_pending(&state));
        assert!(fire_pending(&mut state));
        assert!(!has_pending(&state), "fired power is spent");

        register(PowerKind::Adaptable, marker_handler);
        let (state, _) = corpse_with_power(PowerKind::Adaptable, 1);
        assert!(
            !has_pending(&state),
            "Adaptable is not a death-chokepoint fire"
        );
    }

    #[test]
    fn fire_for_owner_skips_zero_amount_power() {
        register(PowerKind::Stock, marker_handler);
        let (mut state, uid) = corpse_with_power(PowerKind::Stock, 0);
        assert_eq!(
            fire_for_owner(&mut state, uid),
            None,
            "amount-0 power is already spent"
        );
    }
}
