//! Ironclad Outrage — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Outrage (Uncommon, 0 energy, Attack, AnyEnemy) — v0.111.0
/// `Cards/Outrage.cs`, `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `GetTeammatesOf` includes the owner (`CombatState.cs:656-659`): one clone
/// into the discard pile of every living player, the owner's included.
pub(crate) fn play_outrage(state: &mut CombatState, target: Option<CreatureUid>) -> EffectResult {
    let t = target.expect("Outrage: Enemy-targeted");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    if let Some(template) = sts2_core::card_effects::resolving_card_inst(state).copied() {
        sts2_core::coop::every_player(state, |state| {
            let _ = spawn_card_into_pile(state, template, PileKind::Discard);
        });
    }
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Outrage,
        name: "OUTRAGE",
        cost: 0,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 9,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // Damage 9 -> 13 — the row lives in `card_upgrades_generated`.
    None
}
