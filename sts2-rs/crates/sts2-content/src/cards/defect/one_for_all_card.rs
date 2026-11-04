//! Defect One For All — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// One For All (Rare, 1 energy, Power, AllAllies) — v0.111.0
/// `Cards/OneForAll.cs`, `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `Players` includes the owner, so in single-player the owner gets it.
pub(crate) fn play_one_for_all(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    // v0.111.0: OneForAllPower 3 (upg 4) on every player (decomp
    // OneForAll.cs:24-30). The power's behaviour is S8b's (#794).
    let amt = if current_card_upgraded(state) { 4 } else { 3 };
    sts2_core::coop::every_player(state, |state| {
        apply_power_to_player(state, PowerKind::OneForAll, amt);
    });
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::OneForAllCard,
        name: "ONE_FOR_ALL_CARD",
        cost: 1,
        card_type: CardType::Power,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
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
