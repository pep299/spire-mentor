//! Rattle (Necrobinder). Hand-authored C#-first port via the character-port skill.
//! C# source: ../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Cards/Rattle.cs
use super::prelude::*;

/// Rattle (Uncommon, 1 energy, Attack, AnyEnemy). The Osty attacks the target
/// for 7 (9 upgraded), **once per hit count**, where the hit count is
/// `1 + (number of times the Osty has attacked this turn)`. OstyAttack card.
//
// C# OnPlay:
//   ArgumentNullException.ThrowIfNull(cardPlay.Target);
//   if (!Osty.CheckMissingWithAnim(Owner))
//       DamageCmd.Attack(OstyDamage.BaseValue)   // 7, +2 on upgrade → 9
//           .FromOsty(Owner.Osty, this)          // attacker is the OSTY
//           .Targeting(cardPlay.Target)
//           .WithHitCount((int)((CalculatedVar)DynamicVars["CalculatedHits"])
//                              .Calculate(cardPlay.Target))
//           .WithHitFx(...)                       // cosmetic, ignored
//           .Execute(...);
//
// CanonicalVars:
//   OstyDamageVar(7m, Move)         → base_damage 7 (cosmetic metadata; the hit
//                                     is dealt by the Osty, not the player).
//                                     OnUpgrade +2 → 9.
//   CalculationBaseVar(0m)          → CalculatedVar Base term = 0.
//   CalculationExtraVar(1m)         → CalculatedVar Extra term = 1.
//   CalculatedVar("CalculatedHits").WithMultiplier(card, _ =>
//       1 + History.Entries.OfType<CreatureAttackedEntry>()
//             .Count(e => e.Actor == card.Owner.Osty && e.HappenedThisTurn(...)))
//     → hit count = Base + Extra × multiplier? NO — `WithMultiplier` here returns
//       the full hit count directly (`1 + ostyAttacksThisTurn`). The Base(0)/
//       Extra(1) vars are the CalculatedVar scaffolding; the effective hit count
//       is `1 + ostyAttacksThisTurn`. So Rattle hits ONCE when the Osty has not
//       yet attacked this turn, and one additional time per prior Osty attack.
// CanonicalTags: OstyAttack (cosmetic glow tag; not carried in CardSpec here).
// ShouldGlowRedInternal => Owner.IsOstyMissing — pure UI, ignored.
//
// MULTI-HIT SCALING: the hit count is `1 + (Osty attacks this turn)`, mirroring
// C# `CalculatedHits` (1 + `CreatureAttackedEntry` count where `Actor == Osty
// && HappenedThisTurn`). The per-turn Osty-attack tally now lives on
// `CombatLog::osty_attacks_this_turn` (incremented at the damage
// chokepoint for an Osty dealer, reset at the owner's turn start), so the hit
// count is exact. Bead `sts2-rs-osty-attacks-this-turn`.
//
// `deal_damage_from_osty_to_enemy` is a no-op when the Osty is absent
// (`osty_dealer_ref` → None), reproducing `CheckMissingWithAnim` (minus the
// cosmetic summon-prompt animation we don't model).
/// `CalculatedHits`: one hit, plus one per Osty attack already made this turn.
/// Shared with the card face — see `crate::cards::preview`.
pub(crate) fn preview_rattle(
    state: &CombatState,
    _card: &Card,
    _target: Option<CreatureUid>,
) -> i32 {
    1 + state.combat_log.osty_attacks_this_turn.max(0)
}

pub(crate) fn play_rattle(state: &mut CombatState, target: Option<CreatureUid>) -> EffectResult {
    let t = target.expect("Rattle: enemy target required");

    // OstyDamage: 7 base, +2 on upgrade. Computed here (the OstyDamage is not a
    // plain player-attack stat), gated on current_card_upgraded like the sibling
    // OstyAttack cards (poke/flatten/sic_em/unleash/bone_shards). No-op if the
    // Osty is missing.
    let dmg = if current_card_upgraded(state) { 9 } else { 7 };

    // Hit count = 1 + (Osty attacks already made this turn). C# computes the
    // count ONCE (`WithHitCount`) from history, so the extra hits this card
    // deals must NOT feed back into their own count — snapshot before looping.
    let hits = resolving_preview(state, preview_rattle, target);
    if osty_dealer_ref(state).is_some() {
        // All the hits are ONE `AttackCommand`, so they add one entry to the
        // tally, not one per hit (AttackCommand.cs:672).
        sts2_core::card_effects::osty_attack_command(state, |state| {
            for _ in 0..hits {
                deal_damage_from_osty_to_enemy(state, t, dmg);
            }
        });
    }

    EffectResult::Resolved
}

#[must_use]
pub(crate) const fn spec() -> CardSpec {
    CardSpec {
        id: CardId::Rattle,
        name: "RATTLE",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy, // C# TargetType.AnyEnemy → single-player Enemy
        rarity: CardRarity::Uncommon,
        base_damage: 7, // OstyDamageVar(7m) — cosmetic; dealt via the Osty
        base_block: 0,
        keywords: CardKeywords::NONE,
        ..CardSpec::base()
    }
}

#[allow(clippy::unnecessary_wraps, dead_code)]
pub(crate) fn upgrade() -> Option<CardUpgradeDelta> {
    // C# OnUpgrade: OstyDamage.UpgradeValueBy(2) (7 → 9). The OstyDamage is not a
    // plain attack stat (the hit is dealt by the Osty), so the bump is computed in
    // the handler via current_card_upgraded rather than a static spec delta.
    // Returns None to keep one source of truth, matching the sibling OstyAttack
    // cards (sic_em/flatten).
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mint card `id` and push it onto the Play pile (so `current_card_*`
    /// helpers read it as the resolving card).
    fn mint_into_play(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
        let mut card = crate::cards::spec_index::card_pod_from_id(id, false).expect("spec exists");
        let raw = u32::try_from(state.cards.len() + 1).expect("fits");
        let uid = sts2_core::CardUid::new(raw).expect("nz");
        card.uid = uid;
        state.cards.push(card);
        state.player.play.push(uid);
        uid
    }

    /// Current HP of the enemy creature `uid`.
    fn enemy_hp(state: &CombatState, uid: CreatureUid) -> i32 {
        state
            .creatures
            .iter()
            .find(|c| c.id == uid.0)
            .expect("enemy present")
            .current_hp
    }

    /// Regression (`sts2-rs-osty-attacks-this-turn`): Rattle's hit count is
    /// `1 + (Osty attacks already made this turn)`. With no prior Osty attack
    /// it lands ONE 7-damage hit; the hit itself then bumps the per-turn tally
    /// (so the next OstyAttack card scales).
    #[test]
    fn rattle_lands_one_hit_when_osty_has_not_attacked_yet() {
        let mut state = CombatState::default();
        state.summon_osty(20).expect("Osty summoned");
        let enemy = state.spawn_enemy(sts2_core::MonsterKind::Nibbit);
        let hp0 = enemy_hp(&state, enemy);
        mint_into_play(&mut state, CardId::Rattle);
        assert_eq!(state.combat_log.osty_attacks_this_turn, 0);

        let r = play_rattle(&mut state, Some(enemy));
        assert_eq!(r, EffectResult::Resolved);
        assert_eq!(
            enemy_hp(&state, enemy),
            hp0 - 7,
            "1 + 0 prior Osty attacks → a single 7-damage Osty hit"
        );
        assert_eq!(
            state.combat_log.osty_attacks_this_turn, 1,
            "the Osty hit bumps the per-turn tally"
        );
    }

    /// One prior Osty attack this turn ⇒ Rattle lands `1 + 1 = 2` hits (14
    /// damage). The hit count is snapshotted BEFORE looping, so the card's own
    /// hits do not feed back into their own count (no runaway).
    #[test]
    fn rattle_scales_with_prior_osty_attacks_this_turn() {
        let mut state = CombatState::default();
        state.summon_osty(20).expect("Osty summoned");
        let enemy = state.spawn_enemy(sts2_core::MonsterKind::Nibbit);
        let hp0 = enemy_hp(&state, enemy);
        mint_into_play(&mut state, CardId::Rattle);
        // Pretend the Osty already attacked once this turn.
        state.combat_log.osty_attacks_this_turn = 1;

        play_rattle(&mut state, Some(enemy));
        assert_eq!(
            enemy_hp(&state, enemy),
            hp0 - 14,
            "1 + 1 prior Osty attack → two 7-damage hits"
        );
        assert_eq!(
            state.combat_log.osty_attacks_this_turn, 2,
            "two hits are one AttackCommand: one entry added to the existing 1"
        );
    }

    /// An Osty attack on every enemy (Bone Shards / High Five) is ONE
    /// `AttackCommand`, so it adds one `CreatureAttackedEntry`, not one per
    /// target (AttackCommand.cs:672). Necrobinder 1016 (GH #1054): the per-hit
    /// tally gave a later Rattle 6 hits where C# gave 5.
    #[test]
    fn osty_attack_on_all_enemies_counts_once() {
        let mut state = CombatState::default();
        state.summon_osty(20).expect("Osty summoned");
        let a = state.spawn_enemy(sts2_core::MonsterKind::Nibbit);
        let b = state.spawn_enemy(sts2_core::MonsterKind::Nibbit);
        let (hp_a, hp_b) = (enemy_hp(&state, a), enemy_hp(&state, b));

        sts2_core::card_effects::deal_damage_from_osty_to_all_alive_enemies(&mut state, 3);
        assert_eq!(enemy_hp(&state, a), hp_a - 3);
        assert_eq!(enemy_hp(&state, b), hp_b - 3);
        assert_eq!(state.combat_log.osty_attacks_this_turn, 1);
    }
}
