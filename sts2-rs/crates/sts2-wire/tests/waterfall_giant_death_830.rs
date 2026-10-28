//! Waterfall Giant death-interrupt fidelity (#830 boss audit, findings D1, D2,
//! D4). Every expectation cites the v0.107.1 C# it reproduces.

mod common;

use common::{
    boss, boss_mut, end_turn, fight, move_id, play, player, power_amount, power_kinds,
    pressurized_giant,
};
use sts2_core::monsters::Intent;
use sts2_core::{apply, Action, PowerKind, StepResult};

/// D1. C# `CreatureCmd.KillWithoutCheckingWinCondition` runs `Hook.AfterDeath`
/// (`CreatureCmd.cs:519`, SteamEruption revives the giant) and then
/// `RemoveAllPowersAfterDeath` (`CreatureCmd.cs:533`), which keeps only the
/// powers overriding `ShouldPowerBeRemovedAfterOwnerDeath => false`
/// (`SteamEruptionPower.cs:37-40`). A Weak put on the giant before the kill is
/// therefore gone, and EXPLODE (`WaterfallGiant.cs:291-299`, a powered attack)
/// hits for the full SteamEruption amount. Rust used to revive the giant with
/// every power, so the Weak cut the blast from 20 to 15.
#[test]
fn waterfall_giant_revives_with_only_steam_eruption() {
    use sts2_core::cards::CardId;
    let mut state = pressurized_giant(
        &["STRIKE_IRONCLAD"; 12],
        &[],
        &["WEAK_POTION", "VULNERABLE_POTION"],
    );
    // Drink both potions at the giant, whichever slots the belt reports.
    while let Some(drink) = sts2_core::legal_actions(&state)
        .into_iter()
        .find(|a| matches!(a, Action::DrinkPotion { .. }))
    {
        let Action::DrinkPotion { potion_slot, .. } = drink else {
            unreachable!()
        };
        let r = apply(
            &mut state,
            Action::DrinkPotion {
                potion_slot,
                target: Some(0),
            },
        );
        assert_eq!(r, StepResult::Advanced);
    }
    let giant = boss(&state);
    assert!(power_amount(&state, giant, PowerKind::Weak) > 0);
    assert!(power_amount(&state, giant, PowerKind::Vulnerable) > 0);

    // Kill it (one Strike on a 5-HP giant).
    boss_mut(&mut state).current_hp = 5;
    assert_eq!(
        play(&mut state, CardId::StrikeIronclad),
        StepResult::Advanced
    );
    let giant = boss(&state);
    assert_eq!(giant.current_hp, 999_999_999, "revived by SteamEruption");
    assert_eq!(move_id(giant), "ABOUT_TO_BLOW");
    assert_eq!(
        power_kinds(&state, giant),
        vec![PowerKind::SteamEruption],
        "RemoveAllPowersAfterDeath keeps only SteamEruption"
    );
    assert_eq!(power_amount(&state, giant, PowerKind::SteamEruption), 20);

    // Enemy turn 2: ABOUT_TO_BLOW captures the 20. Player turn 3 sees it.
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    assert_eq!(boss(&state).intent, Some(Intent::DeathBlow { damage: 20 }));

    // Enemy turn 3: EXPLODE for the full 20 (no block), then the giant dies.
    let hp_before = player(&state).current_hp;
    assert_eq!(end_turn(&mut state), StepResult::CombatWon);
    assert_eq!(
        hp_before - player(&state).current_hp,
        20,
        "EXPLODE is not weakened by the pre-kill Weak"
    );
}

/// D2. A giant killed by thorns during its own STOMP. C# `MoveState.PerformMove`
/// marks STOMP performed before the move body (`MoveState.cs:55-60`); the
/// revive's `SetMoveImmediate(ABOUT_TO_BLOW, forceTransition: true)` leaves
/// STOMP (`OnExitState` clears its flag, `:62-65`) and enters an unperformed
/// ABOUT_TO_BLOW, whose `MustPerformOnceBeforeTransitioning`
/// (`WaterfallGiant.cs:200-203`) holds it for the next turn: round 3
/// ABOUT_TO_BLOW, round 4 EXPLODE. Rust used to keep one AI-wide flag, set
/// after the move, so the roll skipped ABOUT_TO_BLOW, EXPLODE dealt its 0
/// placeholder, and the real blast came a round late.
#[test]
fn waterfall_giant_killed_during_its_own_move_still_telegraphs_about_to_blow() {
    // Bronze Scales: Thorns 3 (`BronzeScales.cs`), enough for a 2-HP giant.
    let mut state = pressurized_giant(&["DEFEND_IRONCLAD"; 12], &["BRONZE_SCALES"], &[]);
    boss_mut(&mut state).current_hp = 2;

    // Enemy turn 2: STOMP hits, thorns kill the giant mid-move, it revives.
    // STOMP's body then finishes: Weak on the player and SteamEruption +3.
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    let giant = boss(&state);
    assert_eq!(giant.current_hp, 999_999_999, "revived");
    assert_eq!(move_id(giant), "ABOUT_TO_BLOW", "held, not rolled past");
    assert_eq!(giant.intent, Some(Intent::Stun));
    assert_eq!(power_amount(&state, giant, PowerKind::SteamEruption), 23);

    // Enemy turn 3: ABOUT_TO_BLOW captures 23 and strips the power.
    let hp_before = player(&state).current_hp;
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    assert_eq!(
        hp_before,
        player(&state).current_hp,
        "ABOUT_TO_BLOW deals nothing"
    );
    let giant = boss(&state);
    assert_eq!(move_id(giant), "EXPLODE");
    assert_eq!(giant.intent, Some(Intent::DeathBlow { damage: 23 }));
    assert_eq!(power_amount(&state, giant, PowerKind::SteamEruption), 0);

    // Enemy turn 4: the blast lands and ends the fight.
    let hp_before = player(&state).current_hp;
    assert_eq!(end_turn(&mut state), StepResult::CombatWon);
    assert_eq!(hp_before - player(&state).current_hp, 23);
}

/// D4. Feed's Fatal check is C# `DamageResult.WasTargetKilled` (`Feed.cs:42`),
/// set by `Creature.LoseHpInternal` when THIS hit takes a living target to 0
/// (`Creature.cs:447`). The pressurized giant is revived to 999,999,999 inside
/// the hit, so "is it dead afterwards" said no and Feed's +3 max HP was lost.
#[test]
fn feed_on_the_pressurized_giant_grants_max_hp() {
    use sts2_core::cards::CardId;
    let mut state = pressurized_giant(&["FEED"; 12], &[], &[]);
    boss_mut(&mut state).current_hp = 5;
    let max_before = player(&state).max_hp;
    assert_eq!(play(&mut state, CardId::Feed), StepResult::Advanced);
    assert_eq!(
        boss(&state).current_hp,
        999_999_999,
        "revived inside the hit"
    );
    assert_eq!(player(&state).max_hp, max_before + 3, "Feed's Fatal fired");
}

/// D4, the other `WasTargetKilled` readers: Sunder's on-kill energy
/// (`Sunder.cs:33-35`) and Echoing Slash's extra wave per kill
/// (`EchoingSlash.cs:34`) fire on the pressurized giant too.
#[test]
fn on_kill_cards_count_the_pressurized_giant_as_killed() {
    use sts2_core::cards::CardId;
    // Sunder: +3 energy on a kill.
    let mut state = fight(
        "Defect",
        &["SUNDER"; 12],
        90,
        &[],
        &[],
        "WATERFALL_GIANT_BOSS",
        10,
    );
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    boss_mut(&mut state).current_hp = 5;
    state.player.energy = 3;
    assert_eq!(play(&mut state, CardId::Sunder), StepResult::Advanced);
    assert_eq!(boss(&state).current_hp, 999_999_999);
    assert_eq!(state.player.energy, 3, "paid 3, refunded 3 for the kill");

    // Echoing Slash: the kill buys a second wave, which hits the revived giant.
    let mut state = fight(
        "Silent",
        &["ECHOING_SLASH"; 12],
        90,
        &[],
        &[],
        "WATERFALL_GIANT_BOSS",
        10,
    );
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    boss_mut(&mut state).current_hp = 5;
    assert_eq!(play(&mut state, CardId::EchoingSlash), StepResult::Advanced);
    assert_eq!(
        boss(&state).current_hp,
        999_999_999 - 10,
        "the second wave (10 damage) lands on the revived giant"
    );
}

/// D1 ordering (#830 review). C# `CreatureCmd.Damage` runs the killing
/// result's `Hook.AfterDamageGiven` (`CreatureCmd.cs:388-391`) before
/// `Kill(killedCreatures)` (`:409`), and `KillWithoutCheckingWinCondition`
/// strips the revived giant with `RemoveAllPowersAfterDeath` (`:533`) after
/// `Hook.AfterDeath` (`:519`). So the Poison that Envenom
/// (`EnvenomPower.cs:20-26`) puts on the giant with the killing Strike is
/// stripped too. Rust used to strip inside the death interrupt, before the
/// killing hit's AfterDamageGiven, and that Poison survived the revive.
#[test]
fn a_power_the_killing_hit_applies_is_stripped_by_the_revive() {
    use sts2_core::cards::CardId;
    let mut state = fight(
        "Silent",
        &[
            "ENVENOM_CARD",
            "STRIKE_SILENT",
            "ENVENOM_CARD",
            "STRIKE_SILENT",
            "ENVENOM_CARD",
            "STRIKE_SILENT",
            "ENVENOM_CARD",
            "STRIKE_SILENT",
            "ENVENOM_CARD",
            "STRIKE_SILENT",
            "ENVENOM_CARD",
            "STRIKE_SILENT",
        ],
        90,
        &[],
        &[],
        "WATERFALL_GIANT_BOSS",
        10,
    );
    // Enemy turn 1: PRESSURIZE (SteamEruption 20 at A9+).
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    assert_eq!(
        power_amount(&state, boss(&state), PowerKind::SteamEruption),
        20
    );

    // Envenom is a self-targeted power card.
    let envenom = state
        .player
        .hand
        .iter()
        .position(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::EnvenomCard)
        .expect("Envenom in hand");
    assert_eq!(
        apply(
            &mut state,
            Action::PlayCard {
                hand_index: u8::try_from(envenom).unwrap(),
                target: None,
            },
        ),
        StepResult::Advanced
    );
    assert!(power_amount(&state, player(&state), PowerKind::Envenom) > 0);

    // Control: a Strike that does NOT kill leaves Envenom's Poison on the
    // giant, so the stripped Poison below is not an absent one.
    let mut control = state.clone();
    boss_mut(&mut control).current_hp = 100;
    assert_eq!(
        play(&mut control, CardId::StrikeSilent),
        StepResult::Advanced
    );
    assert_eq!(
        power_amount(&control, boss(&control), PowerKind::Poison),
        1,
        "Envenom poisons the giant on unblocked Strike damage"
    );

    // The killing Strike: the giant revives with only SteamEruption.
    boss_mut(&mut state).current_hp = 5;
    assert_eq!(play(&mut state, CardId::StrikeSilent), StepResult::Advanced);
    let giant = boss(&state);
    assert_eq!(giant.current_hp, 999_999_999, "revived by SteamEruption");
    assert_eq!(move_id(giant), "ABOUT_TO_BLOW");
    assert_eq!(
        power_kinds(&state, giant),
        vec![PowerKind::SteamEruption],
        "the killing hit's Poison is stripped with the rest"
    );
}
