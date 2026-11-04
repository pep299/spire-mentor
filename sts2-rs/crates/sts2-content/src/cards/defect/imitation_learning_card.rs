//! Defect Imitation Learning — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// Imitation Learning (Rare, 1 energy, Skill, Exhaust, AnyAlly) — v0.111.0
/// `Cards/ImitationLearning.cs`, `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `AnyAlly` — unplayable in single-player (`card_effects::card_is_any_ally`).
pub(crate) fn play_imitation_learning(
    state: &mut CombatState,
    _t: Option<CreatureUid>,
) -> EffectResult {
    // The power sits on the PLAYER, bound to the target teammate's seat
    // (`CoopState::power_seat`); `coop::imitation_learning_copies` plays a
    // clone of each Power card that teammate plays (GH #992). A second play
    // at the same teammate raises that instance (C# `PowerCmd.ModifyAmount`)
    // instead of adding another.
    let Some(ally) = state.any_ally_card_target() else {
        return EffectResult::Resolved;
    };
    let amt = if current_card_upgraded(state) { 3 } else { 2 };
    let owner = state.coop.seated;
    // Without seats (teammates sent as HP and block only) nothing is bound.
    let target = state.coop.stash.get(usize::from(ally)).map(|s| s.seat);
    let bound = |state: &CombatState, id| {
        target.is_none_or(|t| sts2_core::coop::imitation_learning_target(state, id, owner) == t)
    };
    let existing = state.player_power_uids().into_iter().find(|p| {
        state
            .powers
            .get_by_id(p.0)
            .is_some_and(|p| p.kind == PowerKind::ImitationLearning)
            && bound(state, p.0)
    });
    if let Some(uid) = existing {
        if let Some(p) = state.powers.get_mut_by_id(uid.0) {
            p.amount += amt;
        }
        return EffectResult::Resolved;
    }
    match target {
        Some(seat) if state.player.creature_uid.is_some() => {
            let _ = sts2_core::coop::apply_power_bound_to_seat(
                state,
                PowerKind::ImitationLearning,
                amt,
                seat,
            );
        }
        _ => apply_power_to_player(state, PowerKind::ImitationLearning, amt),
    }
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::ImitationLearningCard,
        name: "IMITATION_LEARNING_CARD",
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
    None
}
