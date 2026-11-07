//! LizardTail — one-shot death-prevention revive.
use super::prelude::*;

/// `LizardTail.should_die_late` + `after_preventing_death` — when the
/// player would die and the relic is still armed, consume it and
/// revive at 50% max HP. Per C# canon
/// `MegaCrit.Sts2.Core.Models.Relics/LizardTail.cs`:
///
/// ```text
/// ShouldDieLate(creature):
///     return creature != Owner.Creature || WasUsed   // true = die
/// AfterPreventingDeath(creature):
///     WasUsed = true
///     amount = Math.Max(1m, MaxHp * (Heal(50) / 100m))
///     CreatureCmd.Heal(creature, amount)
/// ```
///
/// The Rust channel merges the veto + heal into one listener (the
/// dispatcher treats `Resolved` as the prevent vote and re-checks
/// `current_hp > 0` afterwards — see
/// [`crate::relic_effects::dispatch_should_die_late`]). C# clamps HP
/// at 0 on damage (`Creature.cs:450`) and `HealInternal` truncates the
/// decimal sum (`Creature.cs:488-490`), so the post-revive HP is
/// `max(1, floor(MaxHp / 2))` — same integer shape as the
/// FairyInABottle revive in
/// `combat_loop::try_fairy_in_a_bottle_prevent_death` (30% → 50%).
///
/// One-shot state lives on `Relic.counter` (0 = armed, 1 = consumed —
/// C# `WasUsed` is a run-persistent `[SavedProperty]`, mirrored
/// run-side through the `enter_combat`/`exit_combat` counter bridge,
/// bead `sts2-rs-pfiwm`). Unlike RuinedHelmet there is NO
/// `after_combat_end` reset: the relic stays spent for the rest of
/// the run.
pub(crate) fn on_should_die_late_lizard_tail(state: &mut CombatState) -> EffectResult {
    // Ownership + armed gate. The channel is only consulted for the
    // PLAYER death check (combat_loop's two `current_hp <= 0` sites),
    // which matches C#'s `creature == Owner.Creature` gate.
    let Some(relic_idx) = state
        .relics
        .iter()
        .position(|r| !r.is_melted && r.kind == crate::RelicKind::LizardTail && r.counter == 0)
    else {
        return EffectResult::NotImplemented;
    };
    let Some(uid) = state.player.creature_uid else {
        return EffectResult::NotImplemented;
    };
    let Some(creature_idx) = state.creatures.iter().position(|c| c.id == uid.0) else {
        return EffectResult::NotImplemented;
    };
    state.relics[relic_idx].counter = 1;
    // The lethal hit's 0-HP `AfterCurrentHpChanged` ran before `ShouldDie`
    // (see `red_skull_probe_at_death`).
    red_skull_probe_at_death(state);
    let max_hp = state.creatures[creature_idx].max_hp;
    // `max(1, MaxHp * 50/100)` healed from the 0-HP death floor ==
    // `max(1, floor(MaxHp / 2))`, capped at MaxHp (MaxHp=1 edge).
    let revive = (max_hp.saturating_mul(50) / 100).max(1).min(max_hp);
    state.creatures[creature_idx].current_hp = revive;
    state.player.current_hp = revive;
    // C# heals via `CreatureCmd.Heal`, which raises the player HP-changed
    // hooks — RedSkull re-probes its 50% threshold on every player HP
    // change (same re-probe `CombatState::creature_heal` performs).
    if player_owns_relic(state, crate::RelicKind::RedSkull) {
        red_skull_check_hp(state);
    }
    EffectResult::Resolved
}

pub(crate) fn install() {
    register_owned_should_die_late(
        RelicId::LizardTail,
        crate::RelicKind::LizardTail,
        on_should_die_late_lizard_tail as ShouldDieLateFn,
    );
}
