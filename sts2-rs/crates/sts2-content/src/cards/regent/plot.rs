//! Regent Plot — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Plot (Uncommon, 1 energy, Skill, AllAllies) — v0.111.0 `Cards/Plot.cs`,
/// `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `GetTeammatesOf` includes the owner (`CombatState.cs:656-659`), so in
/// single-player the owner draws the extra cards next turn.
pub(crate) fn play_plot(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    // v0.111.0: DrawCardsNextTurn 2 (upg 3) on every living player on the
    // side (decomp Plot.cs:22-29).
    let n = if current_card_upgraded(state) { 3 } else { 2 };
    sts2_core::coop::every_player(state, |state| {
        apply_power_to_player(state, PowerKind::DrawCardsNextTurn, n);
    });
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Plot,
        name: "PLOT",
        cost: 1,
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
