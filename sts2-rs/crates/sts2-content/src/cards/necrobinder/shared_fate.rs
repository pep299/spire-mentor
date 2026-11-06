//! Shared Fate (Necrobinder). C#-first port.
//! C# source:
//! ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/SharedFate.cs
use super::prelude::*;

/// Shared Fate (Rare, 0 energy, Skill, Enemy). Exhaust.
///
/// C# `SharedFate.OnPlay`:
///   * `PowerCmd.Apply<StrengthPower>(owner.Creature,
///     -DynamicVars["PlayerStrengthLoss"].BaseValue, …)` → the player loses 2
///     Strength (base; never upgraded).
///   * `PowerCmd.Apply<StrengthPower>(cardPlay.Target,
///     -DynamicVars["EnemyStrengthLoss"].BaseValue, …)` → the targeted enemy
///     loses 2 Strength (base), 3 when upgraded.
///
/// Both deltas are NEGATIVE Strength, applied via the pool's signed helper (the
/// canonical `apply_power_to_*` paths early-return on `amount <= 0`). The player
/// delta routes through the mirror creature so the damage path's mirror+legacy
/// union reads it.
///
/// `CanonicalVars`: `DynamicVar("EnemyStrengthLoss", 2)`, `DynamicVar("PlayerStrengthLoss", 2)`.
/// `CanonicalKeywords`: `Exhaust`.
pub(crate) fn play_shared_fate(
    state: &mut CombatState,
    target: Option<CreatureUid>,
) -> EffectResult {
    // Player loses PlayerStrengthLoss = 2 Strength. OnUpgrade does NOT touch
    // this var, so it is always 2.
    let player_loss = 2;
    if let Some(uid) = state.player.creature_uid {
        apply_power_signed_to_creature(state, uid, PowerKind::Strength, -player_loss);
    }

    // Targeted enemy loses EnemyStrengthLoss = 2 (base), 3 (upgraded:
    // OnUpgrade `DynamicVars["EnemyStrengthLoss"].UpgradeValueBy(1m)` raises the
    // magnitude, applied as a more-negative delta).
    let enemy_loss = if current_card_upgraded(state) { 3 } else { 2 };
    let t = target.expect("SharedFate: Enemy-targeted");
    // `PowerCmd.Apply` runs this drain through Unsettling Lamp like any card
    // debuff (negative Strength reads as a Debuff), so the Lamp can capture and
    // double it. Loadout Necrobinder 1101 v0.111.0.
    let drain = sts2_core::card_effects::fold_player_card_signed_apply(
        state,
        t,
        PowerKind::Strength,
        -enemy_loss,
    );
    apply_power_signed_to_creature(state, t, PowerKind::Strength, drain);

    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::SharedFate,
        name: "SHARED_FATE",
        cost: 0,
        card_type: CardType::Skill,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // C# OnUpgrade: `DynamicVars["EnemyStrengthLoss"].UpgradeValueBy(1m)` — bumps
    // a PowerVar amount (the enemy Strength magnitude), not a damage/block/cost
    // stat. Per conventions, that bump is applied in the handler via the
    // `current_card_upgraded` branch, so the spec-level upgrade delta is None.
    None
}
