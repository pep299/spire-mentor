//! Banshee's Cry (Necrobinder). Hand-authored C#-first port via character-port skill.
//! C# source: ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/BansheesCry.cs
use super::prelude::*;

/// Banshee's Cry (Rare, 9 energy, Attack — AllEnemies). Deal 33 damage to ALL
/// enemies. Cost is reduced by 2 (the `EnergyVar` magnitude) for every Ethereal
/// card the owner plays this combat (and retroactively, on combat entry, by 2×
/// the count of Ethereal cards already played). Upgraded: base cost 9 → 7
/// (`EnergyCost.UpgradeBy(-2)`, in card_upgrades_generated.rs).
///
/// The `AfterCardPlayed` discount runs in the engine's play pipeline
/// (`CombatState::discount_banshees_cry_on_ethereal_play`); the
/// `AfterCardEnteredCombat` retroactive discount for a Banshee's Cry
/// generated mid-combat after Ethereal plays runs in
/// `card_effects::banshees_cry_entered_combat` (a deck copy enters
/// combat before any play, where it is 0).
pub(crate) fn play_banshees_cry(
    state: &mut CombatState,
    _target: Option<CreatureUid>,
) -> EffectResult {
    // C# BansheesCry.OnPlay:
    //   DamageCmd.Attack(Damage.BaseValue).FromCard(this).TargetingAllOpponents(...)
    // Damage is a plain stat (33), unaffected by OnUpgrade.
    deal_damage_all_enemies(state, current_damage(state));
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::BansheesCry,
        name: "BANSHEES_CRY",
        cost: 9,
        card_type: CardType::Attack,
        target_type: TargetType::AllEnemies,
        rarity: CardRarity::Rare,
        base_damage: 33,
        base_block: 0,
        // C# `ExtraHoverTips` surfaces the Ethereal keyword only as a hover tip
        // for the cost-discount mechanic; the card itself is NOT Ethereal.
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // C# OnUpgrade: base.EnergyCost.UpgradeBy(-2) → cost 9 → 7. This pool routes
    // the upgrade delta through the central table; see
    // crates/sts2-content/src/cards/card_upgrades_generated.rs (CardId::BansheesCry).
    None
}
