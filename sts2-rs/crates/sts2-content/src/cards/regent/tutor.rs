//! Regent Tutor — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Tutor (Rare, 1 energy, Skill, AnyAlly) — v0.111.0 `Cards/Tutor.cs`,
/// `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `AnyAlly` — C# `CardModel.CanPlay` (`CardModel.cs:1742`) rejects it
/// without a living ally, which single-player never has
/// (`card_effects::card_is_any_ally`); in co-op the teammate picks.
pub(crate) fn play_tutor(state: &mut CombatState, _t: Option<CreatureUid>) -> EffectResult {
    // C#: the teammate picks a card from their own draw pile into their hand
    // (Tutor.cs:20-29), answered on the spot (`coop::seated_player_picks`).
    sts2_core::coop::with_card_target(state, |state| {
        let options: Vec<sts2_core::ChoiceOption> = state
            .player
            .draw
            .iter()
            .filter_map(|&uid| {
                let card = state.cards.get(uid.0.get() as usize - 1)?;
                Some(sts2_core::ChoiceOption {
                    card_id: card.card_id,
                    upgraded: card.upgraded,
                    card_uid: Some(uid),
                    source_pile: Some(sts2_core::cards::PileKind::Draw),
                })
            })
            .collect();
        let picked =
            sts2_core::coop::seated_player_picks(state, &options).and_then(|i| options[i].card_uid);
        if let Some(uid) = picked {
            state.move_card_to_hand(uid);
        }
    });
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Tutor,
        name: "TUTOR",
        cost: 1,
        card_type: CardType::Skill,
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
    // `EnergyCost.UpgradeBy(-1)` — the row lives in `card_upgrades_generated`.
    None
}
