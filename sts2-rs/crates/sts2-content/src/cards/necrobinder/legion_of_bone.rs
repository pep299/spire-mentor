//! Hand-authored (C#-first port). Keep the prelude import.
use super::prelude::*;

/// LegionOfBone (Uncommon, 2 energy, Skill, AllAllies→Self_). Summon 6 (Osty),
/// Exhaust.
// C# source: ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/LegionOfBone.cs
//
// C# `OnPlay`: iterates `CombatState.PlayerCreatures.Where(c.IsAlive)` and calls
// `OstyCmd.Summon(ctx, item.Player, Summon.BaseValue, this)` once per alive
// player creature. In single-player there is exactly one player creature, so
// this `AllAllies`/`MultiplayerOnly` fan-out collapses to a single
// `summon_osty` call (the intentional multiplayer-collapse port simplification;
// `MultiplayerConstraint.MultiplayerOnly` is dropped). The "Cast" anim is
// cosmetic and ignored. `OstyCmd.Summon(amount)` grows the live Osty's max HP
// by `amount`, or spawns/revives the Osty at `amount` max HP when dead/absent —
// `CombatState::summon_osty` implements exactly that. `CanonicalVars` is a
// single `SummonVar(6m)` → base 6; `OnUpgrade` does `Summon.UpgradeValueBy(2m)`
// → 8 upgraded. The summon amount is a magnitude (not damage/block/cost), so the
// upgrade bump is gated here on `current_card_upgraded` and `upgrade()` returns
// `None`. `CanonicalKeywords` is `Exhaust`.
pub(crate) fn play_legion_of_bone(
    state: &mut CombatState,
    _target: Option<CreatureUid>,
) -> EffectResult {
    let amount = if current_card_upgraded(state) { 8 } else { 6 };
    // C# `LegionOfBone.cs`: every living player summons.
    sts2_core::coop::every_player(state, |state| {
        let _ = state.summon_osty(amount);
    });
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::LegionOfBone,
        name: "LEGION_OF_BONE",
        cost: 2,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Uncommon,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // OnUpgrade bumps the SummonVar (6 → 8); applied in the handler via the
    // `current_card_upgraded` branch, so no stat-field delta here.
    None
}
