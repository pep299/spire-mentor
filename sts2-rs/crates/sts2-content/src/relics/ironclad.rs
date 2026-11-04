//! Ironclad starter relic: `BurningBlood`.
//!
//! Bead P4-2 (sts2-rs-u4u) ships pure data — no relic-effect
//! interpreter, no after-combat-victory dispatch. The Phase 7
//! umbrella bead (sts2-rs-h9u) wires `RelicSpec.effect` into a
//! runtime executor that fires on `dispatch_after_combat_victory`.

use sts2_core::RelicKind;

use crate::relics::{RelicEffect, RelicRarity, RelicSpec};

// ---------------------------------------------------------------------------
// Factories — one per distinct relic
// ---------------------------------------------------------------------------

/// Burning Blood. Ironclad's starter relic. Heals the owner for 6 HP
/// at end of combat (after victory). Mirrors Python class
/// `BurningBlood` at `sts2_env/relics/starter.py:25-33`:
///
/// ```python
/// class BurningBlood(RelicInstance):
///     relic_id = RelicId.BURNING_BLOOD
///     rarity = RelicRarity.STARTER
///     pool = RelicPool.IRONCLAD
///     HEAL = 6
///     def after_combat_victory(self, owner, combat):
///         if owner.current_hp > 0:
///             owner.heal(self.HEAL)
/// ```
///
/// The `pool = RelicPool.IRONCLAD` field is not yet modelled here;
/// it is run-builder bookkeeping that Phase 7 will surface when
/// pool-restricted random rolls land. The `if owner.current_hp > 0`
/// guard is implicit at the interpreter level (the
/// `after_combat_victory` hook only fires when combat ends in a
/// player victory, which by definition means the player survived).
#[must_use]
pub const fn burning_blood() -> RelicSpec {
    RelicSpec {
        id: RelicKind::BurningBlood,
        name: "BURNING_BLOOD",
        rarity: RelicRarity::Starter,
        effect: RelicEffect::HealAtCombatEnd { amount: 6 },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Burning Blood spec pins: id, name, rarity, effect amount.
    /// These values are parity-locked against
    /// `sts2_env/relics/starter.py:25-33` (`HEAL = 6`,
    /// `relic_id = RelicId.BURNING_BLOOD`,
    /// `rarity = RelicRarity.STARTER`).
    #[test]
    fn burning_blood_spec_matches_python_pin() {
        let r = burning_blood();
        assert_eq!(r.id, RelicKind::BurningBlood);
        assert_eq!(r.name, "BURNING_BLOOD");
        assert_eq!(r.rarity, RelicRarity::Starter);
        assert_eq!(r.effect, RelicEffect::HealAtCombatEnd { amount: 6 });
    }

    /// Burning Blood is a starter relic (Python:
    /// `rarity = RelicRarity.STARTER`).
    #[test]
    fn burning_blood_is_starter_rarity() {
        assert_eq!(burning_blood().rarity, RelicRarity::Starter);
    }

    /// `RelicKind::BurningBlood::python_name()` agrees with the
    /// `name` field of [`burning_blood()`]. This is the parity-output
    /// invariant: the (future Phase-7) parity exporter must emit the
    /// same string regardless of whether it asks the runtime
    /// `RelicKind` or the static `RelicSpec`.
    #[test]
    fn burning_blood_id_round_trips_through_relickind_python_name() {
        assert_eq!(burning_blood().name, RelicKind::BurningBlood.python_name(),);
    }

    /// Factory is pure: returns the same data on every call. Important
    /// because the engine clones `RelicSpec`s into runtime `Relic`
    /// POD by value at relic-pickup time.
    #[test]
    fn factory_is_pure() {
        assert_eq!(burning_blood(), burning_blood());
    }
}
