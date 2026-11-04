//! Silent Blade Symphony — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Blade Symphony (Uncommon, 2 energy, Skill, AllAllies) — v0.111.0
/// `Cards/BladeSymphony.cs`, `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `CombatState.GetTeammatesOf` is `GetCreaturesOnSide(creature.Side)`
/// (`CombatState.cs:656-659`) — it INCLUDES the owner, so in single-player the
/// owner is the one living player teammate and gets the 2 Shivs.
pub(crate) fn play_blade_symphony(
    state: &mut CombatState,
    _t: Option<CreatureUid>,
) -> EffectResult {
    // v0.111.0: 2 Shivs into the hand of every living player on the side
    // (decomp BladeSymphony.cs:29-41).
    sts2_core::coop::every_player(state, |state| {
        for _ in 0..2 {
            let _ = spawn_shiv(state, PileKind::Hand);
        }
    });
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::BladeSymphony,
        name: "BLADE_SYMPHONY",
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
    // `EnergyCost.UpgradeBy(-1)` — the row lives in `card_upgrades_generated`.
    None
}
