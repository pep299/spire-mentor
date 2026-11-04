//! Colorless The Ball — v0.111.0-only, MP-only (GH #793).
use super::prelude::*;

/// The Ball (Uncommon, 1 energy, Attack, AnyEnemy) — v0.111.0
/// `Cards/TheBall.cs`, `MultiplayerConstraint => MultiplayerOnly`:
///
/// [decompiled C# omitted — see the cited path]
///
/// `GetResultLocationForCardPlay` sends the played card to a random living
/// TEAMMATE (`Rng.CombatTargets`), into their draw pile at a random place when
/// the result was Discard; with no living teammate the default result location
/// stands. That half is the engine's, not this handler's: the roll is
/// `sts2_core::coop::roll_result_player` at play start and the move
/// `sts2_core::coop::hand_over_played_card` when it leaves Play (GH #1017). The
/// per-play growth mutates this instance's damage base, which moves with the
/// card and survives a downgrade exactly like C#'s `AfterDowngraded` re-add.
pub(crate) fn play_the_ball(state: &mut CombatState, target: Option<CreatureUid>) -> EffectResult {
    let t = target.expect("TheBall: Enemy-targeted");
    let _ = deal_damage_player_to_enemy(state, t, current_damage(state));
    // v0.111.0: Damage += Increase (10, upg 15) after the hit (decomp TheBall.cs:44-45).
    let increase = if current_upgraded(state) { 15 } else { 10 };
    if let Some(uid) = sts2_core::card_effects::resolving_card_uid(state) {
        if let Some(card) = state.cards.get_mut(uid.0.get() as usize - 1) {
            card.base_damage = card.base_damage.saturating_add(increase);
        }
    }
    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::TheBall,
        name: "THE_BALL",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        rarity: CardRarity::Uncommon,
        base_damage: 10,
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // Only the `Increase` var moves — read in the handler.
    None
}
