//! The two GH #830 evaluation levers on real encounters, built the way the
//! combat bench builds them (`scenario::build_scenario_state`, A10):
//!
//! - `sts2_core::combat_loop::effective_intent_damage` must be the number
//!   the game displays: the wire's `effective_amount` (the workbench's
//!   intent label) and the C# `AttackIntent.GetSingleDamage` arithmetic,
//!   with Strength / Weak on the dealer, Vulnerable / Intangible on the
//!   player and Kaiser Crab's back attack.
//! - `sts2_mcts::enemy_eval::enemy_hp_remaining` must count the forms a
//!   multi-form or reviving enemy still has, per family.
//!
//! Lives in `sts2-wire` because this crate has the scenario builder and
//! depends on `sts2-mcts`.

use sts2_core::combat_loop::effective_intent_damage;
use sts2_core::hooks::{CreatureRef, ValueProps};
use sts2_core::{apply, creature_by_id, Action, CombatState, CreatureUid, Intent, PowerKind};
use sts2_mcts::enemy_eval::{enemy_hp_remaining, IncomingDamage};
use sts2_mcts::rollout_net::{export_features_spec, FEATURES_V1, FEATURES_V3};
use sts2_mcts::{loss_quality_with, Mcts, MctsConfig};
use sts2_wire::emit::combat::intent_to_value;
use sts2_wire::scenario::build_scenario_state;

fn deck() -> Vec<String> {
    let mut d = vec!["STRIKE_IRONCLAD".to_owned(); 5];
    d.extend(vec!["DEFEND_IRONCLAD".to_owned(); 4]);
    d.push("BASH".to_owned());
    d
}

/// `encounter` at A10, 70 / 80 HP, starter deck, at the first decision.
fn fight(encounter: &str) -> CombatState {
    build_scenario_state(
        7,
        "Ironclad",
        &deck(),
        70,
        80,
        &[],
        &[],
        encounter,
        10,
        None,
        None,
    )
    .unwrap_or_else(|e| panic!("{encounter}: {e}"))
}

fn creature(s: &CombatState, uid: CreatureUid) -> &sts2_core::Creature {
    creature_by_id(&s.creatures, uid.0).expect("enemy exists")
}

/// The power of `kind` on `uid`, mutably.
fn power_mut(s: &mut CombatState, uid: CreatureUid, kind: PowerKind) -> &mut sts2_core::Power {
    let pid = creature(s, uid)
        .powers
        .iter()
        .copied()
        .find(|pu| s.powers.get_by_id(pu.0).is_some_and(|p| p.kind == kind))
        .unwrap_or_else(|| panic!("no {kind:?} on {uid:?}"));
    s.powers.get_mut_by_id(pid.0).expect("power exists")
}

fn player_uid(s: &CombatState) -> CreatureUid {
    s.player.creature_uid.expect("player creature")
}

/// C# `Hook.ModifyDamage` for one hit: `(base + additive) × Π multipliers`,
/// truncated once (`(int)num`), floored at 0.
fn csharp_hit(base: i32, additive: i32, muls: &[(i32, i32)]) -> i32 {
    let num: i32 = muls.iter().map(|m| m.0).product();
    let den: i32 = muls.iter().map(|m| m.1).product();
    ((base + additive) * num).div_euclid(den).max(0)
}

const BACK: (i32, i32) = (3, 2);
const WEAK: (i32, i32) = (3, 4);
const VULN: (i32, i32) = (3, 2);

/// Helper, wire and C# arithmetic agree on `uid`'s telegraph, and the
/// helper is read-only.
fn check(s: &CombatState, uid: CreatureUid, csharp: i32, hits: u8, what: &str) {
    let before = s.clone();
    let got = effective_intent_damage(s, uid).expect("an attack intent");
    assert!(*s == before, "{what}: the helper changed the state");
    let intent = creature(s, uid).intent.expect("intent");
    let wire = intent_to_value(intent, s, uid);
    assert_eq!(
        wire["effective_amount"].as_i64(),
        Some(i64::from(got.0)),
        "{what}: wire {wire}"
    );
    assert_eq!(got, (csharp, hits), "{what}: base {intent:?}");
    let c = creature(s, uid);
    assert_eq!(
        IncomingDamage::Effective.total(s, c),
        csharp * i32::from(hits),
        "{what}"
    );
}

#[test]
fn effective_intent_damage_is_the_displayed_number() {
    let mut s = fight("KAISER_CRAB_BOSS");
    let (crusher, rocket) = (s.enemy_uids[0], s.enemy_uids[1]);
    let crusher_base = creature(&s, crusher).intent.expect("telegraph").damage();
    let rocket_base = creature(&s, rocket).intent.expect("telegraph").damage();
    assert_eq!(crusher_base, 14, "A9+ Thrash");
    assert!(rocket_base > 0, "the Rocket opens with an attack");
    let me = player_uid(&s);

    // Turn 1: facing Right, so the Crusher (BackAttackLeft) hits from behind.
    check(
        &s,
        crusher,
        csharp_hit(14, 0, &[BACK]),
        1,
        "Thrash from behind",
    );
    check(&s, rocket, rocket_base, 1, "Rocket from the front");
    assert_eq!(
        IncomingDamage::Base.total(&s, creature(&s, crusher)),
        14,
        "the base reading is the move table's number"
    );

    // Face the Crusher: now the Rocket stands behind.
    power_mut(&mut s, me, PowerKind::Surrounded).counter = sts2_core::hooks::SURROUNDED_FACING_LEFT;
    check(&s, crusher, 14, 1, "Thrash from the front");
    check(
        &s,
        rocket,
        csharp_hit(rocket_base, 0, &[BACK]),
        1,
        "Rocket from behind",
    );

    // Strength on the dealer is additive, before the multipliers.
    sts2_core::card_effects::apply_power_to_creature(&mut s, rocket, PowerKind::Strength, 3);
    check(
        &s,
        rocket,
        csharp_hit(rocket_base, 3, &[BACK]),
        1,
        "Rocket +3 Str from behind",
    );

    // Weak on the dealer.
    sts2_core::card_effects::apply_power_to_creature(&mut s, crusher, PowerKind::Weak, 1);
    check(&s, crusher, csharp_hit(14, 0, &[WEAK]), 1, "Weak Thrash");

    // Vulnerable on the player, composed with the rest before one floor.
    sts2_core::card_effects::apply_power_to_creature(&mut s, me, PowerKind::Vulnerable, 1);
    check(
        &s,
        crusher,
        csharp_hit(14, 0, &[WEAK, VULN]),
        1,
        "Weak Thrash, Vulnerable",
    );
    check(
        &s,
        rocket,
        csharp_hit(rocket_base, 3, &[BACK, VULN]),
        1,
        "Rocket +3 Str from behind, Vulnerable",
    );

    // A multi-hit telegraph reports per hit and the hit count.
    s.creatures
        .iter_mut()
        .find(|c| c.id == crusher.0)
        .expect("crusher")
        .intent = Some(Intent::MultiAttack { damage: 7, hits: 2 });
    check(
        &s,
        crusher,
        csharp_hit(7, 0, &[WEAK, VULN]),
        2,
        "Bug Sting 7x2",
    );

    // Intangible on the player caps every displayed hit at 1
    // (`IntangiblePower.ModifyDamageCap`).
    sts2_core::card_effects::apply_power_to_creature(&mut s, me, PowerKind::Intangible, 1);
    check(&s, crusher, 1, 2, "Intangible, multi-hit");
    check(&s, rocket, 1, 1, "Intangible");

    // No attack, no number.
    s.creatures
        .iter_mut()
        .find(|c| c.id == rocket.0)
        .expect("rocket")
        .intent = Some(Intent::Sleep);
    assert_eq!(effective_intent_damage(&s, rocket), None);
    assert_eq!(IncomingDamage::Effective.total(&s, creature(&s, rocket)), 0);
}

/// The Waterfall Giant's death blow is a powered attack too: the display
/// carries the giant's Weak and the player's Vulnerable.
#[test]
fn a_death_blow_reads_like_an_attack() {
    let mut s = fight("WATERFALL_GIANT_BOSS");
    let giant = s.enemy_uids[0];
    let me = player_uid(&s);
    s.creatures
        .iter_mut()
        .find(|c| c.id == giant.0)
        .expect("giant")
        .intent = Some(Intent::DeathBlow { damage: 44 });
    check(&s, giant, 44, 1, "explode");
    sts2_core::card_effects::apply_power_to_creature(&mut s, me, PowerKind::Vulnerable, 1);
    check(
        &s,
        giant,
        csharp_hit(44, 0, &[VULN]),
        1,
        "explode, Vulnerable",
    );
}

fn kill(s: &mut CombatState, uid: CreatureUid) {
    let _ = sts2_core::damage::apply_damage(
        s,
        CreatureRef::Player,
        uid,
        9_999,
        ValueProps::UNPOWERED,
        None,
    );
}

#[allow(clippy::cast_precision_loss)]
fn removed(s: &CombatState, forms: bool) -> f32 {
    let (hp, max) = enemy_hp_remaining(s, forms);
    1.0 - hp as f32 / max as f32
}

/// Test Subject's three forms (111 / 212 / 313 at A10): the form-aware
/// reading keeps one denominator and never falls when a form dies and the
/// next arrives; the base reading resets at every revive.
#[test]
fn test_subject_counts_its_remaining_forms() {
    let mut s = fight("TEST_SUBJECT_BOSS");
    let ts = s.enemy_uids[0];
    let total = 111 + 212 + 313;
    assert_eq!(enemy_hp_remaining(&s, false), (111, 111));
    assert_eq!(enemy_hp_remaining(&s, true), (111 + 212 + 313, total));
    let mut trail = vec![removed(&s, true)];

    s.creatures
        .iter_mut()
        .find(|c| c.id == ts.0)
        .expect("ts")
        .current_hp = 40;
    assert_eq!(enemy_hp_remaining(&s, true), (40 + 212 + 313, total));
    trail.push(removed(&s, true));

    // Form 1 dies: the corpse waits for its RESPAWN turn.
    kill(&mut s, ts);
    assert!(!s.is_over, "Adaptable keeps the fight open");
    assert_eq!(enemy_hp_remaining(&s, false), (0, 111), "base: all removed");
    assert_eq!(enemy_hp_remaining(&s, true), (212 + 313, total));
    trail.push(removed(&s, true));
    let corpse = s.clone();

    // The enemy turn revives form 2 (the engine's bare 200, see
    // `test_subject::second_form_hp`).
    let _ = apply(&mut s, Action::EndTurn);
    let c = creature(&s, ts);
    assert!(c.current_hp > 0, "form 2 revived");
    let form2 = c.current_hp;
    assert_eq!(enemy_hp_remaining(&s, false), (form2, form2), "base: reset");
    assert_eq!(enemy_hp_remaining(&s, true), (form2 + 313, total));
    trail.push(removed(&s, true));

    kill(&mut s, ts);
    assert!(!s.is_over);
    assert_eq!(enemy_hp_remaining(&s, true), (313, total));
    trail.push(removed(&s, true));

    let _ = apply(&mut s, Action::EndTurn);
    let c = creature(&s, ts);
    assert!(c.current_hp > 0, "form 3 revived");
    let form3 = c.current_hp;
    assert!(
        c.powers
            .iter()
            .all(|pu| s.powers.get_by_id(pu.0).map(|p| p.kind) != Some(PowerKind::Adaptable)),
        "form 3 has no Adaptable"
    );
    assert_eq!(enemy_hp_remaining(&s, true), (form3, total));
    trail.push(removed(&s, true));
    assert!(
        trail.windows(2).all(|w| w[1] >= w[0]),
        "form-aware progress never falls: {trail:?}"
    );

    // The loss band: with the knob off, killing form 1 scores as all enemy
    // HP removed, and the revive takes it all back.
    let (off, on) = (
        MctsConfig::default(),
        MctsConfig {
            removed_counts_forms: true,
            ..MctsConfig::default()
        },
    );
    let mut revived = corpse.clone();
    let _ = apply(&mut revived, Action::EndTurn);
    assert!(loss_quality_with(&corpse, &off) > loss_quality_with(&revived, &off) + 0.5);
    assert!(loss_quality_with(&corpse, &on) <= loss_quality_with(&revived, &on));

    // The v3 playout feature reads the same way; v1 keeps the base one.
    let (v1, v3) = (FEATURES_V1, FEATURES_V3);
    assert!((export_features_spec(&corpse, &v1).state[15] - 1.0).abs() < 1e-6);
    assert!((export_features_spec(&corpse, &v3).state[15] - removed(&corpse, true)).abs() < 1e-6);
}

/// The Waterfall Giant has one form; its death leaves it at the
/// 999,999,999 sentinel only to explode, so it reads as removed both ways.
#[test]
fn a_beaten_waterfall_giant_counts_as_removed() {
    let mut s = fight("WATERFALL_GIANT_BOSS");
    let giant = s.enemy_uids[0];
    assert_eq!(enemy_hp_remaining(&s, true), (250, 250));
    // PRESSURIZE (its opening move) arms the death interrupt.
    sts2_core::card_effects::apply_power_to_creature(&mut s, giant, PowerKind::SteamEruption, 20);
    kill(&mut s, giant);
    assert!(creature(&s, giant).is_beaten_sentinel());
    assert_eq!(enemy_hp_remaining(&s, false), (0, 250));
    assert_eq!(enemy_hp_remaining(&s, true), (0, 250));
}

/// A dead Decimillipede segment with a living sibling will reattach at 25.
#[test]
fn a_dead_segment_that_will_reattach_counts_its_reattach_hp() {
    let mut s = fight("DECIMILLIPEDE_ELITE");
    assert!(s.enemy_uids.len() >= 2);
    let (base_hp, base_max) = enemy_hp_remaining(&s, false);
    assert_eq!(enemy_hp_remaining(&s, true), (base_hp, base_max));
    let seg = s.enemy_uids[0];
    let seg_hp = creature(&s, seg).current_hp;
    kill(&mut s, seg);
    assert!(!s.is_over);
    assert_eq!(enemy_hp_remaining(&s, false), (base_hp - seg_hp, base_max));
    assert_eq!(
        enemy_hp_remaining(&s, true),
        (base_hp - seg_hp + 25, base_max)
    );
}

/// Illusions revive to full while a primary enemy lives and die with the
/// last one, so their HP is never progress.
#[test]
fn an_illusion_counts_nothing() {
    let mut s = fight("KAISER_CRAB_BOSS");
    let (crusher, rocket) = (s.enemy_uids[0], s.enemy_uids[1]);
    sts2_core::card_effects::apply_power_to_creature(&mut s, rocket, PowerKind::Illusion, 1);
    let c = creature(&s, crusher);
    assert_eq!(enemy_hp_remaining(&s, true), (c.current_hp, c.max_hp));
}

/// A death-spawn carrier adds the spawns' mean HP to both sums, so the
/// denominator moves only by the rolled-minus-mean HP when they arrive.
#[test]
fn death_spawns_count_as_forms() {
    let mean = |k: sts2_core::MonsterKind| k.mean_initial_hp(10);

    // Axebots: Stock 2 each, so two more bots per slot.
    let mut s = fight("AXEBOTS_NORMAL");
    let bots = s.enemy_uids.len();
    let (hp, max) = enemy_hp_remaining(&s, false);
    let extra = i32::try_from(bots).expect("few") * 2 * mean(sts2_core::MonsterKind::Axebot);
    assert_eq!(enemy_hp_remaining(&s, true), (hp + extra, max + extra));
    let (_, before) = enemy_hp_remaining(&s, true);
    let first = s.enemy_uids[0];
    kill(&mut s, first);
    // The respawn is inserted at the dead bot's slot, ahead of its corpse.
    assert_eq!(s.enemy_uids.len(), bots + 1, "a fresh bot took the slot");
    let fresh = creature(&s, s.enemy_uids[0]);
    assert!(fresh.current_hp > 0 && creature(&s, s.enemy_uids[1]).current_hp <= 0);
    let fresh = fresh.max_hp;
    let (_, after) = enemy_hp_remaining(&s, true);
    assert_eq!(after - before, fresh - mean(sts2_core::MonsterKind::Axebot));

    // Phrog Parasite: 4 Wrigglers on death.
    let mut s = fight("PHROG_PARASITE_ELITE");
    let (hp, max) = enemy_hp_remaining(&s, false);
    let extra = 4 * mean(sts2_core::MonsterKind::Wriggler);
    assert_eq!(enemy_hp_remaining(&s, true), (hp + extra, max + extra));
    let before = enemy_hp_remaining(&s, true).1;
    let phrog = s.enemy_uids[0];
    kill(&mut s, phrog);
    assert_eq!(s.enemy_uids.len(), 5, "4 Wrigglers joined");
    let rolled: i32 = s.enemy_uids[1..]
        .iter()
        .map(|u| creature(&s, *u).max_hp)
        .sum();
    assert_eq!(enemy_hp_remaining(&s, true).1 - before, rolled - extra);

    // Gremlin Merc: a Sneaky and a Fat Gremlin on death.
    let s = fight("GREMLIN_MERC_NORMAL");
    let (hp, max) = enemy_hp_remaining(&s, false);
    let extra =
        mean(sts2_core::MonsterKind::SneakyGremlin) + mean(sts2_core::MonsterKind::FatGremlin);
    assert_eq!(enemy_hp_remaining(&s, true), (hp + extra, max + extra));
}

/// On a fight with no form enemy the two readings agree.
#[test]
fn the_readings_agree_without_a_form_enemy() {
    let s = fight("KAISER_CRAB_BOSS");
    assert_eq!(enemy_hp_remaining(&s, true), enemy_hp_remaining(&s, false));
}

/// The v3 playout features read the Kaiser Crab's back attack; v1's don't.
#[test]
fn v3_features_read_the_displayed_incoming() {
    let s = fight("KAISER_CRAB_BOSS");
    let sum = |read: IncomingDamage| -> i32 {
        s.enemy_uids
            .iter()
            .map(|u| read.total(&s, creature(&s, *u)))
            .sum()
    };
    let (base, eff) = (sum(IncomingDamage::Base), sum(IncomingDamage::Effective));
    assert!(eff > base, "{eff} vs {base}");
    #[allow(clippy::cast_precision_loss)]
    let frac = |v: i32| v as f32 / s.player.max_hp as f32;
    let v1 = export_features_spec(&s, &FEATURES_V1);
    let v3 = export_features_spec(&s, &FEATURES_V3);
    assert!((v1.state[10] - frac(base)).abs() < 1e-6);
    assert!((v3.state[10] - frac(eff)).abs() < 1e-6);
    // Unblocked (no block yet) follows.
    assert!((v3.state[11] - frac(eff)).abs() < 1e-6);
    assert_eq!(v1.actions.len(), v3.actions.len());
}

/// `effective_incoming` on: a seeded search stays deterministic, and on a
/// board whose telegraphs the base reading misreads it changes the
/// heuristic playout (the HP-aware Defend bonus reads incoming).
#[test]
fn effective_incoming_moves_the_heuristic_search_deterministically() {
    let s = fight("KAISER_CRAB_BOSS");
    let cfg = |on: bool| MctsConfig {
        max_simulations: 200,
        max_rollout_depth: 60,
        seed: 5,
        effective_incoming: on,
        ..MctsConfig::default()
    };
    let run = |on: bool| {
        let mut m = Mcts::new(s.clone(), cfg(on));
        let pick = m.search();
        let visits: Vec<u32> = m.tree.arena.iter().map(|n| n.visits).collect();
        let values: Vec<u32> = m.tree.arena.iter().map(|n| n.value_sum.to_bits()).collect();
        (pick, visits, values)
    };
    let (a, b, off) = (run(true), run(true), run(false));
    assert_eq!(a, b, "deterministic for a fixed seed");
    assert_ne!(a.2, off.2, "the knob changed no playout value");
}
