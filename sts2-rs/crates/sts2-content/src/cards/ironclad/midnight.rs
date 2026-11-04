//! Ironclad Midnight — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Midnight (Rare, 12 energy, Attack, AnyEnemy) — v0.111.0 `Cards/Midnight.cs`,
/// `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// The per-exhaust cost reduction is a card-side listener, fired from the
/// core exhaust chokepoint (`hooks::reduce_midnight_costs_v0111`). The
/// `AfterCardEnteredCombat` catch-up (every player's exhausts so far) runs for
/// a Midnight generated or transformed in mid-combat, which only a co-op pool
/// can produce (`hooks::midnight_entered_combat_v0111`).
pub(crate) fn play_midnight(state: &mut CombatState, target: Option<CreatureUid>) -> EffectResult {
    let t = target.expect("Midnight: Enemy-targeted");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Midnight,
        name: "MIDNIGHT",
        cost: 12,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Rare,
        base_damage: 60,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // Damage 60 -> 72 — the row lives in `card_upgrades_generated`.
    None
}
