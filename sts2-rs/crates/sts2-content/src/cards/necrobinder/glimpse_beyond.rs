//! Hand-authored (C#-first port via character-port skill).
use super::prelude::*;

/// Glimpse Beyond (Rare, 1 energy, Skill). Generate 3 Souls into the draw pile
/// at random positions (4 when upgraded). Exhaust.
// C# source: ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/GlimpseBeyond.cs
//
// C# OnPlay loops over each living player teammate and, for each, runs
//   Soul.Create(creature.Player, DynamicVars.Cards.IntValue, CombatState)
//   CardPileCmd.AddGeneratedCardsToCombat(cards, PileType.Draw, addedByPlayer: true,
//                                         CardPilePosition.Random)
// CardsVar(3); OnUpgrade `Cards.UpgradeValueBy(1m)` → 4. Exhaust keyword.
// MultiplayerConstraint.MultiplayerOnly + TargetType.AllAllies collapse to a
// single-player self-targeted skill: there is exactly one player teammate (self),
// so the per-teammate loop reduces to generating `cards` Souls for the player.
// CardPilePosition.Random → spawn_card_into_draw_pile_random_position, which
// consumes the Shuffle-RNG read per card (matches C# AddInternal Random index).
pub(crate) fn play_glimpse_beyond(
    state: &mut CombatState,
    target: Option<CreatureUid>,
) -> EffectResult {
    let _ = target;
    let count = if current_card_upgraded(state) { 4 } else { 3 };
    // C# `GlimpseBeyond.cs`: Souls into the draw pile of every living player
    // on the side.
    sts2_core::coop::every_player(state, |state| {
        for _ in 0..count {
            let soul = card_from_spec(&status_specs::soul());
            spawn_card_into_draw_pile_random_position(state, soul);
        }
    });
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::GlimpseBeyond,
        name: "GLIMPSE_BEYOND",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        rarity: CardRarity::Rare,
        base_damage: 0,
        base_block: 0,
        keywords: CardKeywords::EXHAUST,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // C# OnUpgrade only bumps CardsVar (3 → 4). That magnitude is applied in the
    // handler via the current_card_upgraded branch, so no stat delta here.
    None
}
