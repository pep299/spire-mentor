//! Enfeebling Touch (Necrobinder). C#-first port.
//! C# source:
//! ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/EnfeeblingTouch.cs
use super::prelude::*;

/// Enfeebling Touch (Uncommon, 1 energy, Skill, Enemy, ETHEREAL). Target loses
/// 8 Strength until end of turn (11 when upgraded).
///
/// C# `EnfeeblingTouch`: ctor `base(1, Skill, Uncommon, AnyEnemy)` → cost 1,
/// Skill, Uncommon, Enemy (multiplayer `AnyEnemy` collapses to `Enemy`).
/// `CanonicalVars`: `DynamicVar("StrengthLoss", 8)`. `CanonicalKeywords`:
/// `Ethereal`. `OnPlay`: `PowerCmd.Apply<EnfeeblingTouchPower>(target,
/// StrengthLoss, owner, this)` (the `TriggerAnim("Cast")` is cosmetic and
/// ignored). `OnUpgrade`: `DynamicVars["StrengthLoss"].UpgradeValueBy(3)`
/// (8 → 11) — a PowerVar magnitude bump, so it lives in the handler's
/// `current_card_upgraded` branch and `upgrade()` returns None.
///
/// `EnfeeblingTouchPower : TemporaryStrengthPower` with `IsPositive=false`
/// (EnfeeblingTouchPower.cs): on apply it folds `-amount` Strength onto the
/// target (C# `BeforeApplied`, `Sign=-1`), and at the owner-side turn end it
/// removes itself and reverts `+amount` Strength (C#
/// `TemporaryStrengthPower.AfterTurnEnd`). The turn-end revert is already
/// wired in `sts2-core::hooks` for `PowerKind::EnfeeblingTouch` (bead
/// `sts2-rs-g4ekc`); this handler lands the apply-side `-amount` Strength fold
/// (via the signed helper) plus the `EnfeeblingTouch` stack itself — mirroring
/// the sibling `TemporaryStrength`-family ports (DarkShackles / PiercingWail /
/// Malaise).
pub(crate) fn play_enfeebling_touch(
    state: &mut CombatState,
    target: Option<CreatureUid>,
) -> EffectResult {
    let t = target.expect("EnfeeblingTouch: enemy target required");
    let strength_loss = if current_card_upgraded(state) { 11 } else { 8 };
    // C# `PowerCmd.Apply<EnfeeblingTouchPower>` is a SINGLE debuff application:
    // Artifact (or any other `ModifyPowerAmountReceived` listener) negates it
    // as a whole and consumes ONE stack, and only if it actually LANDS does
    // `TemporaryStrengthPower.BeforeApplied` fold the `-strength_loss` Strength.
    // Apply the EnfeeblingTouch power FIRST, then drain Strength only when its
    // stack increased. Splitting it into two unconditional applies (the prior
    // port: Strength drain THEN the power) let the Strength drain consume the
    // Artifact, after which the EnfeeblingTouch debuff sailed through unblocked
    // (parity fuzz seed 85: an Artifact-warded CHOMPER ended with
    // EnfeeblingTouch(8) in Rust vs `[]` in C#). Mirrors the sibling
    // DarkShackles port. Bead `sts2-rs-enfeebling-artifact`.
    let before = creature_power_amount(state, t, PowerKind::EnfeeblingTouch);
    apply_power_to_enemy(state, t, PowerKind::EnfeeblingTouch, strength_loss);
    let after = creature_power_amount(state, t, PowerKind::EnfeeblingTouch);
    if after > before {
        // TemporaryStrengthPower.BeforeApplied: `PowerCmd.Apply<StrengthPower>(
        // target, Sign * amount)` with Sign = -1 (IsPositive=false).
        // `amount` is the modified (landed) amount (PowerCmd.cs:129-136), e.g.
        // doubled by Unsettling Lamp.
        apply_power_signed_to_creature(state, t, PowerKind::Strength, -(after - before));
    }
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::EnfeeblingTouch,
        name: "ENFEEBLING_TOUCH",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::ETHEREAL,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // C# OnUpgrade: `DynamicVars["StrengthLoss"].UpgradeValueBy(3)` (8 → 11) —
    // a PowerVar magnitude bump, not a damage/block/cost stat. Per conventions
    // that bump is applied in the handler via the `current_card_upgraded`
    // branch, so the spec-level upgrade delta is None.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mint EnfeeblingTouch onto the Play pile (so `current_card_upgraded`
    /// resolves the resolving card).
    fn mint_enfeebling_into_play(state: &mut CombatState) {
        let mut card = crate::cards::spec_index::card_pod_from_id(CardId::EnfeeblingTouch, false)
            .expect("spec");
        let raw = u32::try_from(state.cards.len() + 1).expect("fits");
        card.uid = sts2_core::CardUid::new(raw).expect("nz");
        state.player.play.push(card.uid);
        state.cards.push(card);
    }

    /// On a normal (un-warded) enemy, EnfeeblingTouch lands its power and folds
    /// the −8 Strength.
    #[test]
    fn enfeebling_touch_lands_power_and_strength_on_unwarded_enemy() {
        let mut state = CombatState::default();
        let enemy = state.spawn_enemy(sts2_core::MonsterKind::Nibbit);
        mint_enfeebling_into_play(&mut state);

        play_enfeebling_touch(&mut state, Some(enemy));

        assert_eq!(
            creature_power_amount(&state, enemy, PowerKind::EnfeeblingTouch),
            8,
            "EnfeeblingTouch(8) applied"
        );
        assert_eq!(
            creature_power_amount(&state, enemy, PowerKind::Strength),
            -8,
            "BeforeApplied folds −8 Strength"
        );
    }

    /// Regression (`sts2-rs-enfeebling-artifact`): Artifact negates the SINGLE
    /// EnfeeblingTouch debuff application as a whole and consumes one stack — no
    /// EnfeeblingTouch power AND no Strength fold (C# `BeforeApplied` never
    /// runs). The prior two-apply port let the Strength drain consume the
    /// Artifact and the EnfeeblingTouch debuff then sailed through (fuzz seed
    /// 85: Rust ended EnfeeblingTouch(8), C# `[]`).
    #[test]
    fn enfeebling_touch_is_fully_negated_by_artifact() {
        let mut state = CombatState::default();
        let enemy = state.spawn_enemy(sts2_core::MonsterKind::Nibbit);
        apply_power_to_enemy(&mut state, enemy, PowerKind::Artifact, 1);
        mint_enfeebling_into_play(&mut state);

        play_enfeebling_touch(&mut state, Some(enemy));

        assert_eq!(
            creature_power_amount(&state, enemy, PowerKind::EnfeeblingTouch),
            0,
            "Artifact negates the EnfeeblingTouch power entirely"
        );
        assert_eq!(
            creature_power_amount(&state, enemy, PowerKind::Strength),
            0,
            "no Strength fold when the parent debuff is negated"
        );
        assert_eq!(
            creature_power_amount(&state, enemy, PowerKind::Artifact),
            0,
            "exactly one Artifact stack consumed"
        );
    }
}
