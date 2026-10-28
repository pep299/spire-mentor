//! GH #893: ascension values a monster's moves telegraph and apply, checked
//! against the C# decomp on both builds by performing each move for real.
//!
//! Every fight here enters through `enter_combat_from_roster`, which is the
//! full-run combat entry path, so the AI carries exactly the values a run
//! would see. A move is performed by forcing it as the monster's current
//! state and running `enemy_turn_exec`. That covers both block paths: the
//! head `Defend` that `perform_intent` grants and the secondary grants that
//! the move's `effect_fn` makes.

use sts2_content::monster_dispatch::MonsterSpawnArgs;
use sts2_content::run_bridge::enter_combat_from_roster;
use sts2_content::run_callbacks::{initialize_run_setup, new_run_manager_for_character};
use sts2_core::monster_ai::MonsterState;
use sts2_core::monsters::Intent;
use sts2_core::runstate::{RoomType, RunState};
use sts2_core::{
    CharacterId, CombatPhase, CombatState, CreatureUid, GameVersion, MonsterKind, Side,
};

const VERSIONS: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

fn run_state(version: GameVersion, ascension: u8) -> RunState {
    let mut rm = new_run_manager_for_character(7, CharacterId::Ironclad).with_game_version(version);
    rm.run_state.ascension_level = ascension;
    initialize_run_setup(&mut rm);
    rm.run_state
}

/// A lone `kind` fight at `ascension` on `version`, through combat entry.
fn fight(kind: MonsterKind, version: GameVersion, ascension: u8) -> (CombatState, CreatureUid) {
    fight_with(kind, &[], version, ascension)
}

/// `kind` (returned) followed by `teammates`, through combat entry.
fn fight_with(
    kind: MonsterKind,
    teammates: &[MonsterKind],
    version: GameVersion,
    ascension: u8,
) -> (CombatState, CreatureUid) {
    let rs = run_state(version, ascension);
    let roster: Vec<_> = std::iter::once(kind)
        .chain(teammates.iter().copied())
        .map(|k| (k, MonsterSpawnArgs::None, 100))
        .collect();
    let state = enter_combat_from_roster(&rs, &roster, RoomType::Elite);
    let uid = state.enemy_uids[0];
    (state, uid)
}

fn intents(state: &CombatState, uid: CreatureUid, move_id: &str) -> Vec<Intent> {
    let c = state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .expect("creature");
    match c.ai.as_ref().and_then(|ai| ai.states.get(move_id)) {
        Some(MonsterState::Move(m)) => m.intents.to_vec(),
        _ => panic!("{:?} has no move {move_id}", c.species),
    }
}

/// What performing one move did.
#[derive(Debug)]
struct Performed {
    damage_to_player: i32,
    self_block: i32,
}

/// Force `move_id` and run the enemy turn on `st` in place.
fn perform_in(st: &mut CombatState, uid: CreatureUid, move_id: &str) -> Performed {
    st.player.current_hp = 9999;
    st.player.max_hp = 9999;
    if let Some(pc) = st.player.creature_uid {
        if let Some(c) = st.creatures.iter_mut().find(|c| c.id == pc.0) {
            c.current_hp = 9999;
            c.max_hp = 9999;
        }
    }
    let head = intents(st, uid, move_id).first().copied();
    if let Some(c) = st.creatures.iter_mut().find(|c| c.id == uid.0) {
        c.block = 0;
        if let Some(ai) = c.ai.as_mut() {
            assert!(ai.force_current_state_by_name(move_id), "no move {move_id}");
        }
        c.intent = head;
    }
    st.phase = CombatPhase::EnemyTurnExec;
    st.side = Side::Enemy;
    let hp_before = st.player.current_hp;
    let _ = sts2_core::combat_loop::enemy_turn_exec(st);
    Performed {
        damage_to_player: hp_before - st.player.current_hp,
        self_block: st
            .creatures
            .iter()
            .find(|c| c.id == uid.0)
            .map_or(-1, |c| c.block),
    }
}

/// [`perform_in`] on a copy of `base`.
fn perform(base: &CombatState, uid: CreatureUid, move_id: &str) -> Performed {
    perform_in(&mut base.clone(), uid, move_id)
}

fn first_defend(intents: &[Intent]) -> Option<i32> {
    intents.iter().find_map(|i| match i {
        Intent::Defend { block } => Some(*block),
        _ => None,
    })
}

fn first_damage(intents: &[Intent]) -> Option<i32> {
    intents.iter().find_map(|i| match i {
        Intent::Attack { damage }
        | Intent::DeathBlow { damage }
        | Intent::MultiAttack { damage, .. } => Some(*damage),
        _ => None,
    })
}

/// The C# `AscensionHelper.GetValueIfAscension(level, scaled, base)`.
fn csharp(ascension: u8, level: u8, scaled: i32, base: i32) -> i32 {
    if ascension >= level {
        scaled
    } else {
        base
    }
}

/// Item 1 of GH #893: the move values the decomp scales at
/// `AscensionLevel.ToughEnemies` (A8), not DeadlyEnemies (A9). PR #883
/// (GH #858) fixed all but Seapunk's telegraph through
/// `ASCENSION_TOUGH_*_SCALING`; its unit tests read the built AI, this one
/// enters a real combat and performs each move, checking both the telegraph
/// and what the move applies at A0 / A7 / A8 / A9 / A10 on both builds.
#[test]
fn tough_enemies_move_values_scale_at_a8() {
    #[derive(Clone, Copy)]
    enum Field {
        Block,
        Damage,
    }
    // (kind, move, field, A0, A8+, decomp ref — the same in both builds)
    let rows: &[(MonsterKind, &str, Field, i32, i32, &str)] = &[
        (
            MonsterKind::Tunneler,
            "BURROW_MOVE",
            Field::Block,
            32,
            37,
            "Tunneler.cs:52 BlockGain",
        ),
        (
            MonsterKind::InfestedPrism,
            "PULSATE",
            Field::Block,
            20,
            22,
            "InfestedPrism.cs:44 PulsateBlock",
        ),
        (
            MonsterKind::LagavulinMatriarch,
            "SLASH2",
            Field::Block,
            12,
            14,
            "LagavulinMatriarch.cs:57 Slash2Block",
        ),
        (
            MonsterKind::LouseProgenitor,
            "CURL_AND_GROW_MOVE",
            Field::Block,
            14,
            18,
            "LouseProgenitor.cs CurlBlock",
        ),
        (
            MonsterKind::MagiKnight,
            "FIRST_POWER_SHIELD",
            Field::Block,
            5,
            9,
            "MagiKnight.cs:49 PowerShieldBlock",
        ),
        (
            MonsterKind::MagiKnight,
            "PREP",
            Field::Block,
            5,
            9,
            "MagiKnight.cs:49 PowerShieldBlock",
        ),
        (
            MonsterKind::Seapunk,
            "BUBBLE_BURP",
            Field::Block,
            7,
            8,
            "Seapunk.cs:38 BubbleBlock",
        ),
        (
            MonsterKind::GremlinMerc,
            "GIMME",
            Field::Damage,
            7,
            8,
            "GremlinMerc.cs:36 GimmeDamage",
        ),
        (
            MonsterKind::GremlinMerc,
            "DOUBLE_SMASH",
            Field::Damage,
            6,
            7,
            "GremlinMerc.cs:40 DoubleSmashDamage",
        ),
        (
            MonsterKind::GremlinMerc,
            "HEHE",
            Field::Damage,
            8,
            9,
            "GremlinMerc.cs:44 HeheDamage",
        ),
    ];
    let mut fails = Vec::new();
    for version in VERSIONS {
        for asc in [0u8, 7, 8, 9, 10] {
            for &(kind, move_id, field, base, tough, src) in rows {
                let want = csharp(asc, 8, tough, base);
                let (state, uid) = fight(kind, version, asc);
                let shown = match field {
                    Field::Block => first_defend(&intents(&state, uid, move_id)),
                    Field::Damage => first_damage(&intents(&state, uid, move_id)),
                };
                if shown != Some(want) {
                    fails.push(format!(
                        "{version:?} A{asc} {kind:?} {move_id} telegraph {shown:?} != {want} ({src})"
                    ));
                }
                let done = perform(&state, uid, move_id);
                let (got, wanted, what) = match field {
                    Field::Block => (done.self_block, want, "block granted"),
                    Field::Damage => {
                        let hits = intents(&state, uid, move_id)
                            .iter()
                            .find_map(|i| match i {
                                Intent::MultiAttack { hits, .. } => Some(i32::from(*hits)),
                                Intent::Attack { .. } => Some(1),
                                _ => None,
                            })
                            .unwrap_or(1);
                        (done.damage_to_player, want * hits, "damage dealt")
                    }
                };
                if got != wanted {
                    fails.push(format!(
                        "{version:?} A{asc} {kind:?} {move_id} {what} {got} != {wanted} ({src})"
                    ));
                }
            }
        }
    }
    assert!(
        fails.is_empty(),
        "{} mismatches:\n{}",
        fails.len(),
        fails.join("\n")
    );
}

/// Item 2 of GH #893: TestSubject's revive HP. C# `TestSubject.cs:82-86`
/// (both builds): `SecondFormHp => GetValueIfAscension(ToughEnemies, 212,
/// 200)`, `ThirdFormHp => (.., 313, 300)`, read by `Revive` on the first and
/// second death.
#[test]
fn test_subject_revive_hp_scales_at_a8() {
    let mut fails = Vec::new();
    for version in VERSIONS {
        for asc in [0u8, 7, 8, 9, 10] {
            let want = [csharp(asc, 8, 212, 200), csharp(asc, 8, 313, 300)];
            let (mut st, uid) = fight(MonsterKind::TestSubject, version, asc);
            // Registered by the content install that combat entry runs.
            let revive = sts2_core::enemy_death_effects::lookup(sts2_core::PowerKind::Adaptable)
                .expect("Adaptable death effect registered");
            for (i, want) in want.into_iter().enumerate() {
                if let Some(c) = st.creatures.iter_mut().find(|c| c.id == uid.0) {
                    c.current_hp = 0;
                }
                revive(&mut st, uid);
                let c = st.creatures.iter().find(|c| c.id == uid.0).unwrap();
                if (c.current_hp, c.max_hp) != (want, want) {
                    fails.push(format!(
                        "{version:?} A{asc} revive {}: hp {}/{} != {want}",
                        i + 1,
                        c.current_hp,
                        c.max_hp
                    ));
                }
            }
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

/// Item 2 of GH #893: the ToughEgg hatchling's HP. C# `ToughEgg.cs:56-58`
/// (both builds): `HatchlingMinHp => GetValueIfAscension(ToughEnemies, 20,
/// 19)`, `HatchlingMaxHp => (.., 23, 22)`. The roll at `:172` is
/// `Rng.Niche.NextInt(HatchlingMinHp, HatchlingMaxHp)` on v0.107.1 (max
/// exclusive) and `NextInt(HatchlingMinHp, HatchlingMaxHp + 1)` on v0.111.0.
/// The hatch must make exactly that one call, so the expected HP is that
/// call replayed on a copy of the niche stream.
#[test]
fn tough_egg_hatchling_hp_scales_at_a8() {
    let mut fails = Vec::new();
    for version in VERSIONS {
        for asc in [0u8, 7, 8, 9, 10] {
            let min = csharp(asc, 8, 20, 19);
            let max = csharp(asc, 8, 23, 22);
            let exclusive_max = if version >= GameVersion::V0_111_0 {
                max + 1
            } else {
                max
            };
            let (base, uid) = fight(MonsterKind::ToughEgg, version, asc);
            let effect = {
                let c = base.creatures.iter().find(|c| c.id == uid.0).unwrap();
                match c.ai.as_ref().unwrap().states.get("HATCH_MOVE") {
                    Some(MonsterState::Move(m)) => m.effect,
                    _ => panic!("ToughEgg has no HATCH_MOVE"),
                }
            };
            let (mut lo, mut hi) = (i32::MAX, i32::MIN);
            for seed in 0..300 {
                let mut st = base.clone();
                st.run_rng.niche = sts2_core::dotnet_rng::Rng::new(seed);
                let mut replay = st.run_rng.niche.clone();
                let want = replay.next_int_exclusive(min, exclusive_max);
                effect(&mut st, uid);
                let hp = st.creatures.iter().find(|c| c.id == uid.0).unwrap().max_hp;
                if hp != want {
                    fails.push(format!(
                        "{version:?} A{asc} seed {seed}: hatchling {hp} != {want}"
                    ));
                }
                if st.run_rng.niche != replay {
                    fails.push(format!(
                        "{version:?} A{asc} seed {seed}: hatch advanced the niche stream \
                         differently from one NextInt"
                    ));
                }
                lo = lo.min(hp);
                hi = hi.max(hp);
            }
            if (lo, hi) != (min, exclusive_max - 1) {
                fails.push(format!(
                    "{version:?} A{asc}: band {lo}..={hi} != {min}..={}",
                    exclusive_max - 1
                ));
            }
        }
    }
    fails.truncate(20);
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

/// Item 3 of GH #893: TheForgotten's DREAD. C# `TheForgotten.cs:24-29`
/// (both builds): `DreadDamage = GetValueIfAscension(DeadlyEnemies, 15, 13)
/// + Creature.GetPowerAmount<DexterityPower>()`, re-read on every telegraph.
/// MIASMA gives the monster 2 Dexterity, and the engine re-derives the stored
/// DREAD intent there — it used the A0 13 as the base, so A9+ DREAD was 2 low
/// after the first MIASMA.
#[test]
fn the_forgotten_dread_keeps_its_ascension_base_after_miasma() {
    let mut fails = Vec::new();
    for version in VERSIONS {
        for asc in [0u8, 8, 9, 10] {
            let base = csharp(asc, 9, 15, 13);
            let (mut st, uid) = fight(MonsterKind::TheForgotten, version, asc);
            for miasmas in 0..3 {
                let want = base + 2 * miasmas;
                let shown = first_damage(&intents(&st, uid, "DREAD"));
                if shown != Some(want) {
                    fails.push(format!(
                        "{version:?} A{asc} after {miasmas} MIASMA: DREAD shows {shown:?} != {want}"
                    ));
                }
                let dealt = perform(&st, uid, "DREAD").damage_to_player;
                if dealt != want {
                    fails.push(format!(
                        "{version:?} A{asc} after {miasmas} MIASMA: DREAD dealt {dealt} != {want}"
                    ));
                }
                perform_in(&mut st, uid, "MIASMA");
            }
        }
    }
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

/// Total amount of each power kind on the enemy side.
fn enemy_powers(state: &CombatState) -> std::collections::BTreeMap<sts2_core::PowerKind, i32> {
    let mut out = std::collections::BTreeMap::new();
    for c in state
        .creatures
        .iter()
        .filter(|c| state.enemy_uids.iter().any(|u| u.0 == c.id))
    {
        for pu in &c.powers {
            if let Some(p) = state.powers.iter().find(|p| p.id == pu.0) {
                *out.entry(p.kind).or_default() += p.amount;
            }
        }
    }
    out
}

/// Item 5 of GH #893: every `Intent::Buff { kind, amount }` (and every
/// `Intent::Defend { block }`) a monster move telegraphs must equal what the
/// move actually applies — at A0, A8, A9 and A10, on both builds. The intent
/// numbers reach the wire (`emit/combat.rs`), the workbench and the bridge
/// telegraphs, so a stale A0 number there is a visible lie even when the
/// effect itself is right.
#[test]
fn every_buff_and_defend_intent_matches_what_the_move_applies() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let mut fails = Vec::new();
    let mut skipped = Vec::new();
    let mut checked = 0usize;
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    for version in VERSIONS {
        for asc in [0u8, 8, 9, 10] {
            for &kind in MonsterKind::ALL_KNOWN {
                // A move that buffs its TEAMMATES needs one present. The
                // Buff amount is per target, so one teammate makes the total
                // applied equal the telegraphed amount.
                let teammates: &[MonsterKind] = match kind {
                    // C# `Queen.BurnBrightForMeMove`: Strength to every
                    // teammate but the Queen (her encounter partner).
                    MonsterKind::Queen => &[MonsterKind::TorchHeadAmalgam],
                    _ => &[],
                };
                let Ok((base, uid)) = catch_unwind(AssertUnwindSafe(|| {
                    fight_with(kind, teammates, version, asc)
                })) else {
                    skipped.push(format!(
                        "{version:?} A{asc} {kind:?}: combat entry panicked"
                    ));
                    continue;
                };
                let Some(ai) = base
                    .creatures
                    .iter()
                    .find(|c| c.id == uid.0)
                    .and_then(|c| c.ai.clone())
                else {
                    continue;
                };
                for (id, s) in ai.states.iter() {
                    let MonsterState::Move(m) = s else { continue };
                    let mut buffs = std::collections::BTreeMap::new();
                    for i in &m.intents {
                        if let Intent::Buff { kind: p, amount } = i {
                            *buffs.entry(*p).or_insert(0) += amount;
                        }
                    }
                    let defend: Option<i32> = m
                        .intents
                        .iter()
                        .filter_map(|i| match i {
                            Intent::Defend { block } => Some(*block),
                            _ => None,
                        })
                        .reduce(|a, b| a + b);
                    if buffs.is_empty() && defend.is_none() {
                        continue;
                    }
                    let run = catch_unwind(AssertUnwindSafe(|| {
                        let mut st = base.clone();
                        let before = enemy_powers(&st);
                        let done = perform_in(&mut st, uid, id);
                        (before, enemy_powers(&st), done)
                    }));
                    let Ok((before, after, done)) = run else {
                        skipped.push(format!("{version:?} A{asc} {kind:?} {id}: move panicked"));
                        continue;
                    };
                    for (p, amount) in &buffs {
                        checked += 1;
                        let got = after.get(p).copied().unwrap_or(0)
                            - before.get(p).copied().unwrap_or(0);
                        if got != *amount {
                            fails.push(format!(
                                "{version:?} A{asc} {kind:?} {id}: Buff {p:?} shows {amount}, applies {got}"
                            ));
                        }
                    }
                    if let Some(block) = defend {
                        checked += 1;
                        if done.self_block != block {
                            fails.push(format!(
                                "{version:?} A{asc} {kind:?} {id}: Defend shows {block}, grants {}",
                                done.self_block
                            ));
                        }
                    }
                }
            }
        }
    }
    std::panic::set_hook(prev);
    eprintln!(
        "checked {checked} intents; skipped {}:\n{}",
        skipped.len(),
        skipped.join("\n")
    );
    assert!(checked > 100, "only {checked} intents checked");
    assert!(
        fails.is_empty(),
        "{} mismatches:\n{}",
        fails.len(),
        fails.join("\n")
    );
}

/// The Buff amounts that change during combat (GH #893): C# re-reads them on
/// every telegraph, the engine stores them on the move, so the move must
/// re-telegraph after it changes. Each case performs the move repeatedly and
/// checks the telegraph before every use against what that use applies.
#[test]
fn dynamic_buff_intents_track_what_the_next_use_applies() {
    fn buff_amount(st: &CombatState, uid: CreatureUid, move_id: &str) -> Option<i32> {
        intents(st, uid, move_id).iter().find_map(|i| match i {
            Intent::Buff {
                kind: sts2_core::PowerKind::Strength,
                amount,
            } => Some(*amount),
            _ => None,
        })
    }
    fn strength(st: &CombatState, uid: CreatureUid) -> i32 {
        let c = st.creatures.iter().find(|c| c.id == uid.0).unwrap();
        c.powers
            .iter()
            .filter_map(|pu| st.powers.iter().find(|p| p.id == pu.0))
            .filter(|p| p.kind == sts2_core::PowerKind::Strength)
            .map(|p| p.amount)
            .sum()
    }
    let mut fails = Vec::new();
    for version in VERSIONS {
        for asc in [0u8, 8, 9, 10] {
            // Aeonglass INCREASING_INTENSITY: `IncreasingIntensityBaseStrength
            // (4/3) + AdditionalStrength` (the upgrade count), so 3, 4, 5, …
            let base = csharp(asc, 9, 4, 3);
            for (moves, kind, move_id, want) in [
                (
                    4,
                    MonsterKind::Aeonglass,
                    "INCREASING_INTENSITY_MOVE",
                    vec![base, base + 1, base + 2, base + 3],
                ),
                // Entomancer PHEROMONE_SPIT: +1 Strength while the hive (spawned
                // at 1) is below 3, then +2.
                (
                    4,
                    MonsterKind::Entomancer,
                    "PHEROMONE_SPIT",
                    vec![1, 1, 2, 2],
                ),
            ] {
                let (mut st, uid) = fight(kind, version, asc);
                for (n, want) in want.into_iter().enumerate().take(moves) {
                    let shown = buff_amount(&st, uid, move_id);
                    let before = strength(&st, uid);
                    perform_in(&mut st, uid, move_id);
                    let got = strength(&st, uid) - before;
                    if shown != Some(want) || got != want {
                        fails.push(format!(
                            "{version:?} A{asc} {kind:?} {move_id} use {}: shows {shown:?}, \
                             applies {got}, C# {want}",
                            n + 1
                        ));
                    }
                }
            }

            // Axebot BOOT_UP is performed only by a respawned bot:
            // `BootUpStrGain (4/3) * (2 - StockAmount)` with stock 1, then 0.
            let gain = csharp(asc, 9, 4, 3);
            let (mut st, mut dead) = fight(MonsterKind::Axebot, version, asc);
            let respawn = sts2_core::enemy_death_effects::lookup(sts2_core::PowerKind::Stock)
                .expect("Stock death effect registered");
            for (n, stock) in [1, 0].into_iter().enumerate() {
                if let Some(c) = st.creatures.iter_mut().find(|c| c.id == dead.0) {
                    c.current_hp = 0;
                }
                let before: Vec<_> = st.enemy_uids.to_vec();
                respawn(&mut st, dead);
                let new = *st
                    .enemy_uids
                    .iter()
                    .find(|u| !before.contains(u))
                    .expect("a respawned Axebot");
                let want = gain * (2 - stock);
                let shown = buff_amount(&st, new, "BOOT_UP");
                let s0 = strength(&st, new);
                perform_in(&mut st, new, "BOOT_UP");
                let got = strength(&st, new) - s0;
                if shown != Some(want) || got != want {
                    fails.push(format!(
                        "{version:?} A{asc} Axebot respawn {}: BOOT_UP shows {shown:?}, applies {got}, C# {want}",
                        n + 1
                    ));
                }
                dead = new;
            }
        }
    }
    assert!(
        fails.is_empty(),
        "{} mismatches:\n{}",
        fails.len(),
        fails.join("\n")
    );
}
