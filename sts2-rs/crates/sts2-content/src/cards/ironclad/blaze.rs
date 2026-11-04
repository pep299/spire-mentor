//! Ironclad Blaze — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Blaze (Uncommon, 2 energy, Skill, AnyAlly) — v0.111.0 `Cards/Blaze.cs`,
/// `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `AnyAlly` — unplayable in single-player (`card_effects::card_is_any_ally`);
/// in co-op the effect lands on the teammate (`any_ally_card_target`).
pub(crate) fn play_blaze(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    // C# `PowerCmd.Apply<StrengthPower>(cardPlay.Target, 5 (upg 7))`: the
    // teammate's Strength.
    let amount = if card_effects::resolving_card_upgraded(state) {
        7
    } else {
        5
    };
    sts2_core::coop::apply_power_to_card_target(state, PowerKind::Strength, amount);
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Blaze,
        name: "BLAZE",
        cost: 2,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    None
}
