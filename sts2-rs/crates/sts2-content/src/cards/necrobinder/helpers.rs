//! Necrobinder per-card helpers. Mirrors the regent/defect helper shapes;
//! uses the newer-pool `current_card_*` naming convention. Rebuilt from
//! scratch for the C#-first port.
//!
//! `dead_code` is allowed while the pool is being ported in waves — these are
//! the standard per-card helper surface; not every one is referenced until all
//! 88 cards land. Drop the allow once the pool is complete.
#![allow(dead_code)]
use std::num::NonZeroU32;

use sts2_core::card_effects::{self, deal_damage_player_to_enemy};
use sts2_core::cards::CardId;
use sts2_core::{CombatState, CreatureUid, Power, PowerKind, PowerUid};

/// Arena index of the card currently resolving, or `None` when no card is
/// (GH #422 — an effect can move the resolving card off the Play pile while
/// its remaining play iterations are still owed).
pub(crate) fn current_card_idx(state: &CombatState) -> Option<usize> {
    card_effects::resolving_card_uid(state).map(|u| u.0.get() as usize - 1)
}

/// Whether the card currently resolving is upgraded.
pub(crate) fn current_card_upgraded(state: &CombatState) -> bool {
    card_effects::resolving_card_upgraded(state)
}

/// Energy spent on the current card (X-cost). The engine drains
/// `player.energy` and stamps `energy_spent` before dispatch.
/// Folds the relic `modify_x_value` chain (ChemicalX +2) via
/// [`sts2_core::card_effects::resolving_card_x_value`].
pub(crate) fn current_card_energy_spent(state: &CombatState) -> i32 {
    card_effects::resolving_card_x_value(state)
}

/// `base_damage` of the card currently resolving.
pub(crate) fn current_damage(state: &CombatState) -> i32 {
    card_effects::resolving_card_base_damage(state)
}

/// `base_block` of the card currently resolving.
pub(crate) fn current_block(state: &CombatState) -> i32 {
    card_effects::resolving_card_base_block(state)
}

/// Apply `amount` stacks of `kind` to the player, routing through the
/// canonical creature surface when the d03 mirror is set (so self-powers
/// stack onto the same instance as relic combat-start grants).
pub(crate) fn apply_power_to_player(state: &mut CombatState, kind: PowerKind, amount: i32) {
    if amount <= 0 {
        return;
    }
    if let Some(uid) = state.player.creature_uid {
        card_effects::apply_power_to_creature(state, uid, kind, amount);
        return;
    }
    let existing_uid = state.player.powers.iter().copied().find(|uid| {
        state
            .powers
            .get_by_id(uid.0)
            .is_some_and(|p| p.kind == kind)
    });
    if let Some(uid) = existing_uid {
        if let Some(pow) = state.powers.get_mut_by_id(uid.0) {
            pow.amount = pow.amount.saturating_add(amount);
        }
        return;
    }
    let next_id_raw = state
        .powers
        .iter()
        .map(|p| p.id.get())
        .max()
        .map_or(1, |m| m.saturating_add(1));
    let new_id = NonZeroU32::new(next_id_raw).expect("power id non-zero");
    state.powers.push(Power {
        id: new_id,
        kind,
        amount,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.player.powers.push(PowerUid(new_id));
}

/// Necrobinder starting deck (C# `Necrobinder.StartingDeck`): 4× Strike,
/// 4× Defend, 1× Bodyguard, 1× Unleash.
pub fn necrobinder_starter_deck() -> Vec<crate::cards::types::CardSpec> {
    use super::{bodyguard, defend_necrobinder, strike_necrobinder, unleash};
    let mut deck = Vec::with_capacity(10);
    for _ in 0..4 {
        deck.push(strike_necrobinder::spec());
    }
    for _ in 0..4 {
        deck.push(defend_necrobinder::spec());
    }
    deck.push(bodyguard::spec());
    deck.push(unleash::spec());
    deck
}

/// Sum `kind` stacks on a specific creature (mirror of C#
/// `Creature.GetPowerAmount<P>()`). Returns 0 when the creature or power is
/// absent. Reads `creature.powers` joined against `state.powers`.
pub(crate) fn creature_power_amount(
    state: &CombatState,
    creature: CreatureUid,
    kind: PowerKind,
) -> i32 {
    let Some(c) = state.creatures.iter().find(|c| c.id == creature.0) else {
        return 0;
    };
    c.powers
        .iter()
        .filter_map(|uid| state.powers.get_by_id(uid.0))
        .filter(|p| p.kind == kind)
        .map(|p| p.amount)
        .fold(0_i32, i32::saturating_add)
}

/// Apply a **signed** `amount` of `kind` to `target_uid`, creating a fresh
/// (possibly negative) instance when none exists. Mirrors the sibling pools'
/// `apply_power_signed_to_creature` (silent/ironclad/colorless) — the
/// canonical `apply_power_to_enemy` / `apply_power_to_creature` paths
/// early-return on `amount <= 0`, and `card_effects::apply_signed_attr_power`
/// is `pub(crate)` (not reachable from this crate). Needed for the
/// `TemporaryStrength`-family debuffs (EnfeeblingTouch) that fold a negative
/// Strength delta onto the target on apply.
pub(crate) fn apply_power_signed_to_creature(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
) {
    // A landed player drain fires SleightOfFlesh & co. (C#
    // `Hook.AfterPowerAmountChanged` runs for silent applies too).
    sts2_core::card_effects::signed_power_apply_with_reactors(
        state,
        target_uid,
        kind,
        amount,
        |s| apply_power_signed_to_creature_raw(s, target_uid, kind, amount),
    );
}

fn apply_power_signed_to_creature_raw(
    state: &mut CombatState,
    target_uid: CreatureUid,
    kind: PowerKind,
    amount: i32,
) {
    if amount == 0 {
        return;
    }
    // C# `ArtifactPower` negates a Debuff-direction application (a negative
    // attribute drain) and consumes a stack; the canonical affliction gate
    // misses signed drains, so consult the signed-debuff gate (no-op for
    // positive restores).
    if amount < 0
        && sts2_core::card_effects::try_consume_artifact_for_signed_debuff(
            state, target_uid, amount,
        )
    {
        return;
    }
    let Some(creature_idx) = state
        .creatures
        .iter()
        .position(|c| c.id == target_uid.0 && c.current_hp > 0)
    else {
        return;
    };
    // Find an existing instance of `kind` to stack into, consulting BOTH the
    // mirror creature's power list AND — for the player — the legacy
    // `state.player.powers` list (matching the core-canonical
    // `existing_power_uid_unioned` union). Without the legacy fallback, a
    // negative Strength delta (Friendship's −2) applied while the player's
    // existing Strength lives only on the legacy list would create a DUPLICATE
    // Strength instead of netting into it (parity fuzz: player.powers.len cs=2
    // rs=3). Bead `sts2-rs-friendship-strength-union`.
    let has_kind = |uid: PowerUid| {
        state
            .powers
            .get_by_id(uid.0)
            .is_some_and(|p| p.kind == kind)
    };
    let existing_uid = state.creatures[creature_idx]
        .powers
        .iter()
        .copied()
        .find(|&uid| has_kind(uid))
        .or_else(|| {
            if state.player.creature_uid == Some(target_uid) {
                state
                    .player
                    .powers
                    .iter()
                    .copied()
                    .find(|&uid| has_kind(uid))
            } else {
                None
            }
        });
    if let Some(uid) = existing_uid {
        let mut remove_pid = None;
        if let Some(pow) = state.powers.get_mut_by_id(uid.0) {
            pow.amount = pow.amount.saturating_add(amount);
            // C# `PowerCmd.ModifyAmount` runs `ShouldRemoveDueToAmount()` after
            // the change; for an `AllowNegative` kind like Strength
            // (`PowerModel.cs:413-424`) that removes the instance at EXACTLY 0.
            // A signed drain that nets an existing +N Strength to 0 (SharedFate's
            // −2 onto a +2 enemy, Friendship's −2 onto the player) must DROP the
            // instance, not leave an inert 0-amount husk (parity fuzz: SharedFate
            // seeds 139/422 — Rust kept `STRENGTH(0)`, C# `[]`). A non-zero
            // (negative) result persists — Strength can legitimately go down.
            // Mirrors the positive-stack arm in `apply_power_to_creature`.
            if pow.amount == 0 {
                remove_pid = Some(uid.0);
            }
        }
        if let Some(pid) = remove_pid {
            state.powers.retain(|p| p.id != pid);
            state.player.powers.retain(|u| u.0 != pid);
            for c in &mut state.creatures {
                c.powers.retain(|u| u.0 != pid);
            }
        }
        return;
    }
    let next_id_raw = state
        .powers
        .iter()
        .map(|p| p.id.get())
        .max()
        .map_or(1, |m| m.saturating_add(1));
    let new_id = NonZeroU32::new(next_id_raw).expect("power id non-zero");
    state.powers.push(Power {
        id: new_id,
        kind,
        amount,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    state.creatures[creature_idx].powers.push(PowerUid(new_id));
}

/// Deal `dmg` to every alive enemy via the canonical core fan-out, which
/// resolves all targets' `ModifyDamage` against the pre-death dealer state
/// before any kill (C# `CreatureCmd.Damage` loop-then-`Kill`). Bead
/// `sts2-rs-8gzvn`.
pub(crate) fn deal_damage_all_enemies(state: &mut CombatState, dmg: i32) {
    let card = sts2_core::card_effects::originating_card_id(state);
    let _ = sts2_core::card_effects::deal_damage_to_all_alive_enemies(
        state,
        dmg,
        sts2_core::hooks::ValueProps::MOVE,
        card,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression (`sts2-rs-friendship-strength-union`): a negative Strength
    /// delta (Friendship's −2) applied to the player must NET into an existing
    /// Strength instance that lives ONLY on the legacy `player.powers` list, not
    /// fork a duplicate. Without the legacy fallback the signed apply found no
    /// match on the mirror creature and created a second Strength
    /// (parity fuzz: player.powers.len cs=2 rs=3).
    #[test]
    fn friendship_negative_strength_nets_into_legacy_player_instance() {
        let mut state = CombatState::default();
        state.player.current_hp = 60;
        state.player.max_hp = 60;
        let player_uid = state.init_player_creature();

        // Existing Strength(3) on the LEGACY player power list only — the mirror
        // creature's `powers` list does NOT carry it (the union the fix joins).
        let pid = PowerUid::new(state.powers.len() as u32 + 1).expect("nonzero uid");
        state.powers.push(Power {
            id: pid.0,
            kind: PowerKind::Strength,
            amount: 3,
            counter: 0,
            applier: None,
            skip_next_tick: false,
        });
        state.player.powers.push(pid);

        apply_power_signed_to_creature(&mut state, player_uid, PowerKind::Strength, -2);

        let strengths: Vec<_> = state
            .powers
            .iter()
            .filter(|p| p.kind == PowerKind::Strength)
            .collect();
        assert_eq!(
            strengths.len(),
            1,
            "the −2 delta nets into the legacy instance — no duplicate Strength"
        );
        assert_eq!(
            strengths[0].amount, 1,
            "3 + (−2) = 1 folded onto the existing Strength"
        );
    }

    /// Regression (SharedFate husk): a signed drain that nets an existing
    /// Strength to EXACTLY 0 drops the instance (C# `ShouldRemoveDueToAmount`
    /// removes `AllowNegative` powers at 0) rather than leaving a `STRENGTH(0)`
    /// husk. SharedFate's −2 onto a +2 enemy: parity fuzz seeds 139/422 ended
    /// with `STRENGTH(0)` in Rust vs `[]` in C#.
    #[test]
    fn signed_drain_to_exactly_zero_removes_the_strength_instance() {
        use sts2_core::MonsterKind;
        let mut state = CombatState::default();
        let enemy = state.spawn_enemy(MonsterKind::Nibbit);
        // Enemy already carries Strength(+2).
        sts2_core::card_effects::apply_power_to_enemy(&mut state, enemy, PowerKind::Strength, 2);
        assert_eq!(creature_power_amount(&state, enemy, PowerKind::Strength), 2);

        // SharedFate-style −2 drain nets to 0.
        apply_power_signed_to_creature(&mut state, enemy, PowerKind::Strength, -2);

        assert_eq!(
            state
                .powers
                .iter()
                .filter(|p| p.kind == PowerKind::Strength)
                .count(),
            0,
            "the 0-amount Strength instance is removed, not left as a husk"
        );
        let c = state.creatures.iter().find(|c| c.id == enemy.0).unwrap();
        assert!(
            !c.powers
                .iter()
                .any(|u| state.powers.iter().any(|p| p.id == u.0)),
            "the enemy's power link to the removed Strength is cleaned up"
        );
    }

    /// A signed drain that lands at a NON-zero (negative) Strength persists —
    /// Strength is `AllowNegative`, so only an exact 0 is pruned.
    #[test]
    fn signed_drain_to_negative_strength_persists() {
        use sts2_core::MonsterKind;
        let mut state = CombatState::default();
        let enemy = state.spawn_enemy(MonsterKind::Nibbit);
        sts2_core::card_effects::apply_power_to_enemy(&mut state, enemy, PowerKind::Strength, 2);

        apply_power_signed_to_creature(&mut state, enemy, PowerKind::Strength, -5);

        assert_eq!(
            creature_power_amount(&state, enemy, PowerKind::Strength),
            -3,
            "2 + (−5) = −3 persists (Strength can go negative)"
        );
    }

    /// A player Strength drain on an enemy triggers SleightOfFlesh: C#
    /// `SleightOfFleshPower.AfterPowerAmountChanged` sees `GetTypeForAmount(-2)
    /// == Debuff` (SharedFate — Necrobinder full seed 1875, quick seed 201).
    /// The same drain on the player does not (`power.Owner.IsEnemy`).
    #[test]
    fn signed_enemy_drain_triggers_sleight_of_flesh() {
        use sts2_core::MonsterKind;
        let mut state = CombatState::default();
        state.player.current_hp = 50;
        state.player.max_hp = 50;
        let player_uid = state.init_player_creature();
        let enemy = state.spawn_enemy(MonsterKind::Nibbit);
        let hp = |s: &CombatState, u: CreatureUid| {
            s.creatures.iter().find(|c| c.id == u.0).unwrap().current_hp
        };
        if let Some(c) = state.creatures.iter_mut().find(|c| c.id == enemy.0) {
            c.current_hp = 40;
            c.max_hp = 40;
        }
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            player_uid,
            PowerKind::SleightOfFlesh,
            13,
        );

        apply_power_signed_to_creature(&mut state, player_uid, PowerKind::Strength, -2);
        assert_eq!(hp(&state, enemy), 40, "a self-drain is not an enemy debuff");

        apply_power_signed_to_creature(&mut state, enemy, PowerKind::Strength, -2);
        assert_eq!(hp(&state, enemy), 27, "13 Unpowered SleightOfFlesh damage");
    }
}
