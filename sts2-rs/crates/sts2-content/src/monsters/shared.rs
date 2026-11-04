// Doc comments use upper-snake-case DSL state names (e.g. `RAND_ATTACK`,
// `FLAME_BEAM`) that clippy flags as missing-backticks identifiers.
// They are state-id strings, not Rust paths.
#![allow(clippy::doc_markdown)]

//! Shared / event / boss-encounter monster ports (bead sts2-rs-uzn).
//!
//! Ports the 18 monsters from
//! `sts2-rl-agent@01c094d:sts2_env/monsters/shared.py` (506 LOC) to
//! data-driven `MonsterAi` instances. Conventions match [`super::act1_overgrowth`]
//! through [`super::act1_underdocks`]:
//!
//! - Effect bodies are no-ops; intent metadata is preserved.
//! - Constructors return `MonsterAi` only.
//! - Pets / dummies come in two shapes, and the difference is on the wire:
//!   the ones C# gives a `HiddenIntent` telegraph one `Intent::Unknown`
//!   (`hidden_intent_ai_with_state`), the ones C# gives no intents at all
//!   telegraph nothing (`silent_ai_with_state`). Python collapsed both onto
//!   `IntentType.UNKNOWN`.
//!
//! # TheAdversary 3-phase encounter contract
//!
//! Mk1 / Mk2 / Mk3 are three separate creature constructors. The
//! Python encounter spawns Mk1 first, and the encounter-loop bead
//! (sts2-rs-bf8 or a future combat-loop hook) is responsible for
//! observing Mk1's death and spawning Mk2, then Mk3. This module
//! exposes each Mk independently — the phase-advancement glue lives in
//! `combat_loop.rs` / encounter-scripts when that bead lands. Test
//! coverage here locks each Mk's state machine in isolation.
//!
//! # `MysteriousKnight` canonicalization
//!
//! Python's `shared.py:create_mysterious_knight` is the canonical
//! definition (FlailKnight base + Strength(6) + Plating(6)). The local
//! [`super::act3_glory::create_mysterious_knight_ai`] is a duplicate ported
//! for act3 self-containment; when the dispatcher bead wires power
//! application, the act3 wrapper can drop and re-export this module's
//! `create_mysterious_knight_ai` instead.
//!
//! # `BattlewornDummyTimeLimit` timer
//!
//! BattleFriend V1/V2/V3 all carry `PowerKind::BattlewornDummyTimeLimit`
//! (amount=3, registered via the dispatcher bead's
//! after-added-to-room hook). The DSL state-machine itself doesn't
//! tick the timer — that's a power-listener responsibility on round
//! end. This bead exposes the ctors with `NOTHING_MOVE` self-loops;
//! the timer landing-bead (P6 power listeners) wires the
//! `round_end → decrement` listener body.

use std::collections::BTreeMap;

use smallvec::smallvec;
use sts2_core::monster_ai::{
    MonsterAi, MonsterState, MoveRepeatType, MoveState, RandomBranchState, WeightedBranch,
};
use sts2_core::monsters::Intent;
use sts2_core::powers::PowerKind;
use sts2_core::CombatState;

use super::apply_power_to_player_via_legacy_surface;
use super::dsl::{mv, proto_ai, proto_ais, ProtoAis};

/// `FakeMerchantMonster` ENRAGE_MOVE: self-buff `Strength(2)`. Mirrors C#
/// `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Monsters/FakeMerchantMonster.cs:104-109`
/// `EnrageMove`: `await PowerCmd.Apply<StrengthPower>(base.Creature, 2m, base.Creature, null)`.
/// The `Intent::Buff { Strength, 2 }` on ENRAGE is telegraph-only
/// (`perform_intent` runs the effect_fn for non-attack intents); the power
/// apply lives here. Previously `noop_effect`.
fn fake_merchant_enrage_effect(state: &mut CombatState, owner: sts2_core::CreatureUid) {
    sts2_core::card_effects::apply_power_to_creature(state, owner, PowerKind::Strength, 2);
}

/// `FakeMerchantMonster` THROW_RELIC_MOVE: apply `Frail(1)` to the player.
/// Mirrors C#
/// `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Monsters/FakeMerchantMonster.cs:94-102`
/// `ThrowRelicMove`: attack then
/// `await PowerCmd.Apply<FrailPower>(targets, 1m, base.Creature, null)`.
/// The attack damage is delivered by the `Intent::Attack` half; the trailing
/// `Intent::Debuff { Frail, 1 }` is telegraph-only, so the Frail apply lives
/// here. Applies via the legacy `state.player.powers` surface (same rationale
/// as the Flyconid/Fogmog player-debuff helpers in `act1_overgrowth`).
fn fake_merchant_throw_relic_effect(state: &mut CombatState, _owner: sts2_core::CreatureUid) {
    apply_power_to_player_via_legacy_surface(state, PowerKind::Frail, 1);
}

/// `FlailKnight`/`MysteriousKnight` WAR_CHANT: self-buff `Strength(3)`.
/// Mirrors C#
/// `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Monsters/FlailKnight.cs:60-65`
/// `WarChantMove`: `await PowerCmd.Apply<StrengthPower>(base.Creature, 3m, base.Creature, null)`.
/// The `Intent::Buff { Strength, 3 }` is telegraph-only; the power apply lives
/// here. Previously `noop_effect`.
fn war_chant_effect(state: &mut CombatState, owner: sts2_core::CreatureUid) {
    sts2_core::card_effects::apply_power_to_creature(state, owner, PowerKind::Strength, 3);
}

/// `DenseVegetationWriggler` (C# `Wriggler`) WRIGGLE_MOVE: self-buff
/// `Strength(2)` and add one `Infection` status card to the player's discard.
/// Mirrors C#
/// `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Monsters/Wriggler.cs:86-100`
/// `WriggleMove`:
/// `await CardPileCmd.AddToCombatAndPreview<Infection>(targets, PileType.Discard, 1, addedByPlayer: false)`
/// then `await PowerCmd.Apply<StrengthPower>(base.Creature, 2m, base.Creature, null)`.
/// The `Intent::Buff { Strength, 2 }` and `Intent::StatusCard` on WRIGGLE are
/// telegraph-only; both side-effects live here. Previously `noop_effect`.
fn wriggle_effect(state: &mut CombatState, owner: sts2_core::CreatureUid) {
    use sts2_core::card_effects::PileKind;
    // Co-op: every player the move targets gets one (GH #989), as in the
    // act-2 Wriggler's copy of this move.
    sts2_core::coop::each_player(state, |state| {
        let _ = crate::cards::status::spawn_infection(state, PileKind::Discard);
    });
    sts2_core::card_effects::apply_power_to_creature(state, owner, PowerKind::Strength, 2);
}

/// `TheAdversaryMkOne` BARRAGE_MOVE: self-buff `Strength(2)` after the
/// multi-attack. Mirrors C#
/// `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Monsters/TheAdversaryMkOne.cs:65-72`
/// `BarrageMove`: `await PowerCmd.Apply<StrengthPower>(base.Creature, 2m, base.Creature, null)`.
fn adversary_mk_one_barrage_effect(state: &mut CombatState, owner: sts2_core::CreatureUid) {
    sts2_core::card_effects::apply_power_to_creature(state, owner, PowerKind::Strength, 2);
}

/// `TheAdversaryMkTwo` BARRAGE_MOVE: self-buff `Strength(3)`. Mirrors C#
/// `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Monsters/TheAdversaryMkTwo.cs:69-76`
/// `BarrageMove`: `await PowerCmd.Apply<StrengthPower>(base.Creature, 3m, base.Creature, null)`.
fn adversary_mk_two_barrage_effect(state: &mut CombatState, owner: sts2_core::CreatureUid) {
    sts2_core::card_effects::apply_power_to_creature(state, owner, PowerKind::Strength, 3);
}

/// `TheAdversaryMkThree` BARRAGE_MOVE: self-buff `Strength(4)`. Mirrors C#
/// `../sts2-wiki/decompiled/v0.103.2/MegaCrit.Sts2.Core.Models.Monsters/TheAdversaryMkThree.cs:67-74`
/// `BarrageMove`: `await PowerCmd.Apply<StrengthPower>(base.Creature, 4m, base.Creature, null)`.
fn adversary_mk_three_barrage_effect(state: &mut CombatState, owner: sts2_core::CreatureUid) {
    sts2_core::card_effects::apply_power_to_creature(state, owner, PowerKind::Strength, 4);
}

fn br_w(
    state_id: &'static str,
    repeat: MoveRepeatType,
    max_times: u8,
    weight: f64,
    cooldown: u8,
) -> WeightedBranch {
    WeightedBranch {
        state_id,
        repeat_type: repeat,
        max_times,
        weight,
        cooldown,
    }
}

/// A no-op self-loop whose move telegraphs a `HiddenIntent` — C#
/// `new MoveState(id, NothingMove, new HiddenIntent())`. The wire emits
/// `"Hidden"` for it and `parse_intent` maps that to [`Intent::Unknown`],
/// so `Unknown` here is the round-trip target, not a placeholder.
fn hidden_intent_ai_with_state(monster_state_id: &'static str) -> MonsterAi {
    let mut s = BTreeMap::new();
    s.insert(
        monster_state_id,
        mv(
            monster_state_id,
            smallvec![Intent::Unknown],
            monster_state_id,
        ),
    );
    MonsterAi::new(s, monster_state_id)
}

/// A no-op self-loop whose move telegraphs **nothing at all** — C#
/// `new MoveState(id, _ => Task.CompletedTask)`, i.e. `Intents` is the
/// empty `params` array.
///
/// This is a different shape from [`hidden_intent_ai_with_state`] and the
/// difference is observable: the DLL wire serializes a `HiddenIntent` as one
/// `"Hidden"` entry but an intent-less move as none, so a fabricated
/// `Intent::Unknown` here shows up as a mirror-vs-world divergence on every
/// board the creature is on (bead: 55 such records on the v16 sweep, 4 of
/// them BattleFriendV2). The old Python port collapsed both shapes onto
/// `IntentType.UNKNOWN`; `sts2-wiki/decompiled/v0.107.1` keeps them apart.
fn silent_ai_with_state(monster_state_id: &'static str) -> MonsterAi {
    let mut s = BTreeMap::new();
    s.insert(
        monster_state_id,
        mv(monster_state_id, smallvec![], monster_state_id),
    );
    MonsterAi::new(s, monster_state_id)
}

// ────────────────────────────────────────────────────────────────────────────
// Pets / no-op monsters
// ────────────────────────────────────────────────────────────────────────────

/// `Architect` — NOTHING self-loop (Unknown intent).
pub fn create_architect_ai() -> MonsterAi {
    hidden_intent_ai_with_state("NOTHING")
}

/// `Byrdpip` — NOTHING_MOVE self-loop.
pub fn create_byrdpip_ai() -> MonsterAi {
    silent_ai_with_state("NOTHING_MOVE")
}

/// `PaelsLegion` — NOTHING_MOVE self-loop.
pub fn create_paels_legion_ai() -> MonsterAi {
    silent_ai_with_state("NOTHING_MOVE")
}

// ────────────────────────────────────────────────────────────────────────────
// Event combat
// ────────────────────────────────────────────────────────────────────────────

/// `FakeMerchantMonster` — SWIPE → RAND(all 4, CannotRepeat; ENRAGE on
/// cooldown 3). THROW_RELIC routes to `RAND_ATTACK` (attack-only
/// subgraph). Initial state: SWIPE.
pub fn create_fake_merchant_monster_ai() -> MonsterAi {
    let mut s = BTreeMap::new();
    s.insert(
        "RAND",
        MonsterState::RandomBranch(RandomBranchState {
            id: "RAND",
            branches: smallvec![
                br_w("SWIPE", MoveRepeatType::CannotRepeat, 1, 1.0, 0),
                br_w("SPEW_COINS", MoveRepeatType::CannotRepeat, 1, 1.0, 0),
                br_w("THROW_RELIC", MoveRepeatType::CannotRepeat, 1, 1.0, 0),
                br_w("ENRAGE", MoveRepeatType::CannotRepeat, 1, 1.0, 3),
            ],
        }),
    );
    s.insert(
        "RAND_ATTACK",
        MonsterState::RandomBranch(RandomBranchState {
            id: "RAND_ATTACK",
            branches: smallvec![
                br_w("SWIPE", MoveRepeatType::CannotRepeat, 1, 1.0, 0),
                br_w("SPEW_COINS", MoveRepeatType::CannotRepeat, 1, 1.0, 0),
                br_w("THROW_RELIC", MoveRepeatType::CannotRepeat, 1, 1.0, 0),
            ],
        }),
    );
    s.insert(
        "SWIPE",
        mv("SWIPE", smallvec![Intent::Attack { damage: 13 }], "RAND"),
    );
    s.insert(
        "SPEW_COINS",
        mv(
            "SPEW_COINS",
            smallvec![Intent::MultiAttack { damage: 2, hits: 8 }],
            "RAND",
        ),
    );
    s.insert(
        "THROW_RELIC",
        MonsterState::Move(MoveState {
            id: "THROW_RELIC",
            effect: fake_merchant_throw_relic_effect,
            intents: smallvec![
                // C# `FakeMerchantMonster.cs:38` ThrowRelicDamage base = 9
                // (asc DeadlyEnemies = 10). Was 13 — copy-paste of SWIPE's
                // SwipeDamage. Engine deals the intent damage; effect only
                // applies Frail (sts2-rs monster-audit divergent tail).
                Intent::Attack { damage: 9 },
                Intent::Debuff {
                    kind: PowerKind::Frail,
                    amount: 1
                }
            ],
            follow_up: Some("RAND_ATTACK"),
            must_perform_once: false,
        }),
    );
    s.insert(
        "ENRAGE",
        MonsterState::Move(MoveState {
            id: "ENRAGE",
            effect: fake_merchant_enrage_effect,
            intents: smallvec![Intent::Buff {
                kind: PowerKind::Strength,
                amount: 2
            }],
            follow_up: Some("RAND"),
            must_perform_once: false,
        }),
    );
    MonsterAi::new(s, "SWIPE")
}

/// `MysteriousKnight` (event combat, canonical version) — RAND(WAR_CHANT
/// CannotRepeat | FLAIL w=2 | RAM w=2); initial=RAM. Strength(6) +
/// Plating(6) applied at the dispatcher bead's after-added-to-room
/// hook.
pub fn create_mysterious_knight_ai() -> MonsterAi {
    let mut s = BTreeMap::new();
    s.insert(
        "RAND",
        MonsterState::RandomBranch(RandomBranchState {
            id: "RAND",
            branches: smallvec![
                br_w("WAR_CHANT", MoveRepeatType::CannotRepeat, 1, 1.0, 0),
                br_w("FLAIL", MoveRepeatType::CanRepeatForever, 0, 2.0, 0),
                br_w("RAM", MoveRepeatType::CanRepeatForever, 0, 2.0, 0),
            ],
        }),
    );
    s.insert(
        "WAR_CHANT",
        MonsterState::Move(MoveState {
            id: "WAR_CHANT",
            effect: war_chant_effect,
            intents: smallvec![Intent::Buff {
                kind: PowerKind::Strength,
                amount: 3
            }],
            follow_up: Some("RAND"),
            must_perform_once: false,
        }),
    );
    s.insert(
        "FLAIL",
        mv(
            "FLAIL",
            smallvec![Intent::MultiAttack { damage: 9, hits: 2 }],
            "RAND",
        ),
    );
    s.insert(
        "RAM",
        mv("RAM", smallvec![Intent::Attack { damage: 15 }], "RAND"),
    );
    MonsterAi::new(s, "RAM")
}

/// `DenseVegetationWriggler` — NASTY_BITE ↔ WRIGGLE with slot-dep
/// initial. Slots 1/3 start on BITE; slots 2/4 start on WRIGGLE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseVegetationWrigglerSlot {
    First,
    Second,
    Third,
    Fourth,
}

pub fn create_dense_vegetation_wriggler_ai(slot: DenseVegetationWrigglerSlot) -> MonsterAi {
    let mut s = BTreeMap::new();
    s.insert(
        "NASTY_BITE",
        mv(
            "NASTY_BITE",
            smallvec![Intent::Attack { damage: 6 }],
            "WRIGGLE",
        ),
    );
    s.insert(
        "WRIGGLE",
        MonsterState::Move(MoveState {
            id: "WRIGGLE",
            effect: wriggle_effect,
            intents: smallvec![
                Intent::Buff {
                    kind: PowerKind::Strength,
                    amount: 2
                },
                Intent::StatusCard
            ],
            follow_up: Some("NASTY_BITE"),
            must_perform_once: false,
        }),
    );
    let initial = match slot {
        DenseVegetationWrigglerSlot::Second | DenseVegetationWrigglerSlot::Fourth => "WRIGGLE",
        _ => "NASTY_BITE",
    };
    MonsterAi::new(s, initial)
}

/// `Wriggler` spawned with `StartStunned = true` (the PhrogParasite elite's
/// `InfestedPower.AfterDeath` second phase, C# `InfestedPower.cs:28-33`).
///
/// Per C# `Wriggler.cs::GenerateMoveStateMachine`, when `StartStunned` is set
/// the initial state is `SPAWNED_MOVE` (a `StunIntent` no-op), whose
/// `FollowUpState` is the slot-conditional branch: `wriggler1`/`wriggler3`
/// branch to `NASTY_BITE`, `wriggler2`/`wriggler4` to `WRIGGLE`. After the
/// spawn-turn stun the wrigglers alternate NASTY_BITE ↔ WRIGGLE.
///
/// This differs from [`create_dense_vegetation_wriggler_ai`], where the
/// encounter sets `StartStunned = false` (C# `DenseVegetationEventEncounter`)
/// so there is no spawn-turn stun and the slot-conditional branch is the
/// initial state directly. Bead `sts2-rs-2u9w1`: the InfestedPower spawn
/// previously used the slot-LESS `act2_hive::create_wriggler_ai()`, which
/// always followed the stun with NASTY_BITE regardless of slot — the
/// MONSTER_STATE / PLAYER_HP parity divergence.
///
/// Four of these spawn on every PhrogParasite death inside the search, so
/// each is a clone of a per-thread prototype (`proto_ai`).
pub fn create_infested_wriggler_ai(slot: DenseVegetationWrigglerSlot) -> MonsterAi {
    thread_local! {
        static PROTO: ProtoAis<4> = const { proto_ais() };
    }
    proto_ai(&PROTO, slot as usize, || build_infested_wriggler_ai(slot))
}

pub(crate) fn build_infested_wriggler_ai(slot: DenseVegetationWrigglerSlot) -> MonsterAi {
    let mut s = BTreeMap::new();
    // SPAWNED_MOVE → the slot-conditional first real move (StartStunned=true).
    let post_stun = match slot {
        DenseVegetationWrigglerSlot::Second | DenseVegetationWrigglerSlot::Fourth => "WRIGGLE",
        _ => "NASTY_BITE",
    };
    s.insert(
        "SPAWNED_MOVE",
        mv("SPAWNED_MOVE", smallvec![Intent::Stun], post_stun),
    );
    s.insert(
        "NASTY_BITE",
        mv(
            "NASTY_BITE",
            smallvec![Intent::Attack { damage: 6 }],
            "WRIGGLE",
        ),
    );
    s.insert(
        "WRIGGLE",
        MonsterState::Move(MoveState {
            id: "WRIGGLE",
            effect: wriggle_effect,
            intents: smallvec![
                Intent::Buff {
                    kind: PowerKind::Strength,
                    amount: 2
                },
                Intent::StatusCard
            ],
            follow_up: Some("NASTY_BITE"),
            must_perform_once: false,
        }),
    );
    MonsterAi::new(s, "SPAWNED_MOVE")
}

// ────────────────────────────────────────────────────────────────────────────
// Boss minions
// ────────────────────────────────────────────────────────────────────────────

/// `TorchHeadAmalgam` — TACKLE_1 → TACKLE_2 → BEAM → TACKLE_3 →
/// TACKLE_4 → BEAM → TACKLE_3 …
pub fn create_torch_head_amalgam_ai() -> MonsterAi {
    let mut s = BTreeMap::new();
    s.insert(
        "TACKLE_1",
        mv(
            "TACKLE_1",
            smallvec![Intent::Attack { damage: 18 }],
            "TACKLE_2",
        ),
    );
    s.insert(
        "TACKLE_2",
        mv("TACKLE_2", smallvec![Intent::Attack { damage: 18 }], "BEAM"),
    );
    s.insert(
        "BEAM",
        mv(
            "BEAM",
            smallvec![Intent::MultiAttack { damage: 8, hits: 3 }],
            "TACKLE_3",
        ),
    );
    s.insert(
        "TACKLE_3",
        mv(
            "TACKLE_3",
            smallvec![Intent::Attack { damage: 14 }],
            "TACKLE_4",
        ),
    );
    s.insert(
        "TACKLE_4",
        mv("TACKLE_4", smallvec![Intent::Attack { damage: 14 }], "BEAM"),
    );
    MonsterAi::new(s, "TACKLE_1")
}

/// v0.111.0 `TorchHeadAmalgam`: STRONG_TACKLE → TACKLE_2 → BEAM → TACKLE_3 →
/// TACKLE_4 → BEAM → … (GH #796). The opener is now its own move
/// (`STRONG_TACKLE_MOVE` → `StrongTackleMove`, `SingleAttackIntent(
/// StrongTackleDamage)`, 26 / A9 32) instead of a first TACKLE; the rest of
/// the ring is unchanged. Base damages here are A0; the v0.111.0 Tackle /
/// Weak Tackle A9 values (22 / 16) come from
/// [`crate::ascension_scaling::V0111_ATTACK_VALUES`].
///
/// [decompiled C# omitted — see the cited path]
pub fn create_torch_head_amalgam_v0111_ai() -> MonsterAi {
    let mut s = BTreeMap::new();
    // v0.111.0: STRONG_TACKLE_MOVE 26 (A9 32) opener (decomp TorchHeadAmalgam.cs:51)
    s.insert(
        "STRONG_TACKLE",
        mv(
            "STRONG_TACKLE",
            smallvec![Intent::Attack { damage: 26 }],
            "TACKLE_2",
        ),
    );
    s.insert(
        "TACKLE_2",
        mv("TACKLE_2", smallvec![Intent::Attack { damage: 18 }], "BEAM"),
    );
    s.insert(
        "BEAM",
        mv(
            "BEAM",
            smallvec![Intent::MultiAttack { damage: 8, hits: 3 }],
            "TACKLE_3",
        ),
    );
    s.insert(
        "TACKLE_3",
        mv(
            "TACKLE_3",
            smallvec![Intent::Attack { damage: 14 }],
            "TACKLE_4",
        ),
    );
    s.insert(
        "TACKLE_4",
        mv("TACKLE_4", smallvec![Intent::Attack { damage: 14 }], "BEAM"),
    );
    MonsterAi::new(s, "STRONG_TACKLE")
}

// ────────────────────────────────────────────────────────────────────────────
// TheAdversary 3-phase boss encounter
// ────────────────────────────────────────────────────────────────────────────

/// `TheAdversaryMkOne` — SMASH (12) → BEAM (15) → BARRAGE (8×2 + buff)
/// → SMASH. Encounter ctors spawn Mk2 on Mk1's death.
pub fn create_the_adversary_mk_one_ai() -> MonsterAi {
    let mut s = BTreeMap::new();
    s.insert(
        "SMASH",
        mv("SMASH", smallvec![Intent::Attack { damage: 12 }], "BEAM"),
    );
    s.insert(
        "BEAM",
        mv("BEAM", smallvec![Intent::Attack { damage: 15 }], "BARRAGE"),
    );
    s.insert(
        "BARRAGE",
        MonsterState::Move(MoveState {
            id: "BARRAGE",
            effect: adversary_mk_one_barrage_effect,
            intents: smallvec![
                Intent::MultiAttack { damage: 8, hits: 2 },
                Intent::Buff {
                    kind: PowerKind::Strength,
                    amount: 2
                }
            ],
            follow_up: Some("SMASH"),
            must_perform_once: false,
        }),
    );
    MonsterAi::new(s, "SMASH")
}

/// `TheAdversaryMkTwo` — BASH (13) → FLAME_BEAM (16) → BARRAGE (9×2 +
/// buff) → BASH. Spawned on Mk1's death.
pub fn create_the_adversary_mk_two_ai() -> MonsterAi {
    let mut s = BTreeMap::new();
    s.insert(
        "BASH",
        mv(
            "BASH",
            smallvec![Intent::Attack { damage: 13 }],
            "FLAME_BEAM",
        ),
    );
    s.insert(
        "FLAME_BEAM",
        mv(
            "FLAME_BEAM",
            smallvec![Intent::Attack { damage: 16 }],
            "BARRAGE",
        ),
    );
    s.insert(
        "BARRAGE",
        MonsterState::Move(MoveState {
            id: "BARRAGE",
            effect: adversary_mk_two_barrage_effect,
            intents: smallvec![
                Intent::MultiAttack { damage: 9, hits: 2 },
                Intent::Buff {
                    kind: PowerKind::Strength,
                    amount: 3
                }
            ],
            follow_up: Some("BASH"),
            must_perform_once: false,
        }),
    );
    MonsterAi::new(s, "BASH")
}

/// `TheAdversaryMkThree` — CRASH (15) → FLAME_BEAM (18) → BARRAGE (10×2
/// + buff) → CRASH. Spawned on Mk2's death.
pub fn create_the_adversary_mk_three_ai() -> MonsterAi {
    let mut s = BTreeMap::new();
    s.insert(
        "CRASH",
        mv(
            "CRASH",
            smallvec![Intent::Attack { damage: 15 }],
            "FLAME_BEAM",
        ),
    );
    s.insert(
        "FLAME_BEAM",
        mv(
            "FLAME_BEAM",
            smallvec![Intent::Attack { damage: 18 }],
            "BARRAGE",
        ),
    );
    s.insert(
        "BARRAGE",
        MonsterState::Move(MoveState {
            id: "BARRAGE",
            effect: adversary_mk_three_barrage_effect,
            intents: smallvec![
                Intent::MultiAttack {
                    damage: 10,
                    hits: 2
                },
                Intent::Buff {
                    kind: PowerKind::Strength,
                    amount: 4
                }
            ],
            follow_up: Some("CRASH"),
            must_perform_once: false,
        }),
    );
    MonsterAi::new(s, "CRASH")
}

// ────────────────────────────────────────────────────────────────────────────
// Battleworn dummies
// ────────────────────────────────────────────────────────────────────────────

/// `BattleFriendV1` — NOTHING_MOVE self-loop. Carries
/// `BattlewornDummyTimeLimit(3)` (applied via the after-added-to-room
/// dispatcher hook). HP 75 in Python.
pub fn create_battle_friend_v1_ai() -> MonsterAi {
    silent_ai_with_state("NOTHING_MOVE")
}

/// `BattleFriendV2` — same shape as V1, HP 150.
pub fn create_battle_friend_v2_ai() -> MonsterAi {
    silent_ai_with_state("NOTHING_MOVE")
}

/// `BattleFriendV3` — same shape as V1, HP 300.
pub fn create_battle_friend_v3_ai() -> MonsterAi {
    silent_ai_with_state("NOTHING_MOVE")
}

// ────────────────────────────────────────────────────────────────────────────
// Test monsters
// ────────────────────────────────────────────────────────────────────────────

/// `BigDummy` — NOTHING self-loop. HP 9999.
pub fn create_big_dummy_ai() -> MonsterAi {
    hidden_intent_ai_with_state("NOTHING")
}

/// `OneHpMonster` — NOTHING self-loop. HP 1.
pub fn create_one_hp_monster_ai() -> MonsterAi {
    hidden_intent_ai_with_state("NOTHING")
}

/// `TenHpMonster` — NOTHING self-loop. HP 10.
pub fn create_ten_hp_monster_ai() -> MonsterAi {
    hidden_intent_ai_with_state("NOTHING")
}

/// `SingleAttackMoveMonster` — POKE (1 dmg) self-loop.
pub fn create_single_attack_move_monster_ai() -> MonsterAi {
    let mut s = BTreeMap::new();
    s.insert(
        "POKE",
        mv("POKE", smallvec![Intent::Attack { damage: 1 }], "POKE"),
    );
    MonsterAi::new(s, "POKE")
}

/// `MultiAttackMoveMonster` — POKE (1×5) self-loop.
pub fn create_multi_attack_move_monster_ai() -> MonsterAi {
    let mut s = BTreeMap::new();
    s.insert(
        "POKE",
        mv(
            "POKE",
            smallvec![Intent::MultiAttack { damage: 1, hits: 5 }],
            "POKE",
        ),
    );
    MonsterAi::new(s, "POKE")
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_ctor_has_resolvable_initial_move() {
        let smokes: Vec<MonsterAi> = vec![
            create_architect_ai(),
            create_byrdpip_ai(),
            create_paels_legion_ai(),
            create_fake_merchant_monster_ai(),
            create_mysterious_knight_ai(),
            create_dense_vegetation_wriggler_ai(DenseVegetationWrigglerSlot::First),
            create_dense_vegetation_wriggler_ai(DenseVegetationWrigglerSlot::Second),
            create_dense_vegetation_wriggler_ai(DenseVegetationWrigglerSlot::Third),
            create_dense_vegetation_wriggler_ai(DenseVegetationWrigglerSlot::Fourth),
            create_torch_head_amalgam_ai(),
            create_the_adversary_mk_one_ai(),
            create_the_adversary_mk_two_ai(),
            create_the_adversary_mk_three_ai(),
            create_battle_friend_v1_ai(),
            create_battle_friend_v2_ai(),
            create_battle_friend_v3_ai(),
            create_big_dummy_ai(),
            create_one_hp_monster_ai(),
            create_ten_hp_monster_ai(),
            create_single_attack_move_monster_ai(),
            create_multi_attack_move_monster_ai(),
        ];
        for ai in smokes {
            let _ = ai.current_move();
        }
    }

    fn run_three_moves(ai: &mut MonsterAi) -> Vec<&'static str> {
        let mut out = Vec::new();
        out.push(ai.current_move().id);
        for seed in 0..2 {
            ai.on_move_performed();
            out.push(ai.roll_move_seeded(seed).id);
        }
        out
    }

    #[test]
    fn architect_self_loops_on_nothing() {
        let mut ai = create_architect_ai();
        for _ in 0..5 {
            assert_eq!(ai.current_move().id, "NOTHING");
            ai.on_move_performed();
            ai.roll_move_seeded(0);
        }
    }

    #[test]
    fn fake_merchant_first_move_is_swipe() {
        assert_eq!(create_fake_merchant_monster_ai().current_move().id, "SWIPE");
    }

    #[test]
    fn fake_merchant_enrage_has_cooldown_three() {
        // After ENRAGE fires, it must not reappear within the next 3
        // resolved RAND rolls (cooldown=3 in the WeightedBranch).
        let mut ai = create_fake_merchant_monster_ai();
        let mut last_enrage = None::<usize>;
        let mut log = Vec::new();
        for seed in 0..200u64 {
            let id = ai.roll_move_seeded(seed).id;
            log.push(id);
            if id == "ENRAGE" {
                if let Some(prev) = last_enrage {
                    let gap = log.len() - 1 - prev;
                    assert!(
                        gap > 3,
                        "ENRAGE within cooldown window: gap={gap} at seed {seed}"
                    );
                }
                last_enrage = Some(log.len() - 1);
            }
            ai.on_move_performed();
        }
    }

    #[test]
    fn mysterious_knight_starts_on_ram() {
        assert_eq!(create_mysterious_knight_ai().current_move().id, "RAM");
    }

    #[test]
    fn dense_vegetation_wriggler_slot_dispatch() {
        assert_eq!(
            create_dense_vegetation_wriggler_ai(DenseVegetationWrigglerSlot::First)
                .current_move()
                .id,
            "NASTY_BITE"
        );
        assert_eq!(
            create_dense_vegetation_wriggler_ai(DenseVegetationWrigglerSlot::Second)
                .current_move()
                .id,
            "WRIGGLE"
        );
        assert_eq!(
            create_dense_vegetation_wriggler_ai(DenseVegetationWrigglerSlot::Third)
                .current_move()
                .id,
            "NASTY_BITE"
        );
        assert_eq!(
            create_dense_vegetation_wriggler_ai(DenseVegetationWrigglerSlot::Fourth)
                .current_move()
                .id,
            "WRIGGLE"
        );
    }

    #[test]
    fn torch_head_amalgam_six_step_cycle() {
        let mut ai = create_torch_head_amalgam_ai();
        assert_eq!(
            run_three_moves(&mut ai),
            vec!["TACKLE_1", "TACKLE_2", "BEAM"]
        );
        ai.on_move_performed();
        assert_eq!(ai.roll_move_seeded(0).id, "TACKLE_3");
        ai.on_move_performed();
        assert_eq!(ai.roll_move_seeded(0).id, "TACKLE_4");
        ai.on_move_performed();
        assert_eq!(ai.roll_move_seeded(0).id, "BEAM");
        ai.on_move_performed();
        assert_eq!(ai.roll_move_seeded(0).id, "TACKLE_3");
    }

    /// Bead acceptance: TheAdversary 3-phase encounter — exercise each
    /// Mk's state machine in isolation. Phase advancement (Mk1 dies →
    /// Mk2 spawns → Mk3 spawns) lives in the encounter / combat-loop
    /// bead; this test locks the individual cycles so encounter glue
    /// can rely on them.
    #[test]
    fn the_adversary_three_phase_cycles() {
        let mut mk1 = create_the_adversary_mk_one_ai();
        assert_eq!(run_three_moves(&mut mk1), vec!["SMASH", "BEAM", "BARRAGE"]);
        mk1.on_move_performed();
        assert_eq!(mk1.roll_move_seeded(0).id, "SMASH");

        let mut mk2 = create_the_adversary_mk_two_ai();
        assert_eq!(
            run_three_moves(&mut mk2),
            vec!["BASH", "FLAME_BEAM", "BARRAGE"]
        );
        mk2.on_move_performed();
        assert_eq!(mk2.roll_move_seeded(0).id, "BASH");

        let mut mk3 = create_the_adversary_mk_three_ai();
        assert_eq!(
            run_three_moves(&mut mk3),
            vec!["CRASH", "FLAME_BEAM", "BARRAGE"]
        );
        mk3.on_move_performed();
        assert_eq!(mk3.roll_move_seeded(0).id, "CRASH");
    }

    #[test]
    fn battle_friends_all_self_loop_on_nothing_move() {
        for mut ai in [
            create_battle_friend_v1_ai(),
            create_battle_friend_v2_ai(),
            create_battle_friend_v3_ai(),
        ] {
            for _ in 0..3 {
                assert_eq!(ai.current_move().id, "NOTHING_MOVE");
                ai.on_move_performed();
                ai.roll_move_seeded(0);
            }
        }
    }

    /// Bead `sts2-rs-pdsdl`: telegraph-only Buff/Debuff power applies.
    /// Helper: sum a creature's Strength.
    fn creature_strength(state: &CombatState, uid: sts2_core::CreatureUid) -> i32 {
        let creature = state
            .creatures
            .iter()
            .find(|c| c.id == uid.0)
            .expect("spawned creature");
        creature
            .powers
            .iter()
            .filter_map(|puid| {
                state
                    .powers
                    .iter()
                    .find(|p| p.id == puid.0 && p.kind == PowerKind::Strength)
                    .map(|p| p.amount)
            })
            .sum()
    }

    #[test]
    fn fake_merchant_enrage_applies_two_strength_to_self() {
        use sts2_core::MonsterKind;
        let mut state = CombatState::default();
        let uid = state.spawn_enemy(MonsterKind::FakeMerchantMonster);
        fake_merchant_enrage_effect(&mut state, uid);
        assert_eq!(creature_strength(&state, uid), 2);
    }

    #[test]
    fn fake_merchant_throw_relic_applies_one_frail_to_player() {
        use sts2_core::MonsterKind;
        let mut state = CombatState::default();
        let owner = state.spawn_enemy(MonsterKind::FakeMerchantMonster);
        fake_merchant_throw_relic_effect(&mut state, owner);
        let frail: i32 = state
            .player
            .powers
            .iter()
            .filter_map(|puid| {
                state
                    .powers
                    .iter()
                    .find(|p| p.id == puid.0 && p.kind == PowerKind::Frail)
                    .map(|p| p.amount)
            })
            .sum();
        assert_eq!(frail, 1, "THROW_RELIC must apply Frail(1) to the player");
    }

    #[test]
    fn war_chant_applies_three_strength_to_self() {
        use sts2_core::MonsterKind;
        let mut state = CombatState::default();
        let uid = state.spawn_enemy(MonsterKind::MysteriousKnight);
        // `spawn_enemy` now applies MysteriousKnight's AfterAddedToRoom
        // self-buff (Strength 6, bead `sts2-rs-pdsdl`), so assert the
        // WAR_CHANT *delta* rather than an absolute that assumed a
        // clean 0-Strength spawn. Bead `sts2-rs-gq1z6`. (Whether
        // WAR_CHANT's own Strength is 3 vs the C#-faithful value is a
        // separate moveset question — C# MysteriousKnight.cs shows only
        // the AfterAddedToRoom StrengthPower(6), no explicit WarChant
        // Strength — flagged for the shared.rs/pdsdl moveset owner.)
        let before = creature_strength(&state, uid);
        war_chant_effect(&mut state, uid);
        assert_eq!(
            creature_strength(&state, uid) - before,
            3,
            "WAR_CHANT adds 3 Strength on top of the spawn buff",
        );
    }

    #[test]
    fn wriggle_applies_two_strength_and_one_infection() {
        use sts2_core::cards::CardId;
        use sts2_core::MonsterKind;
        let mut state = CombatState::default();
        let uid = state.spawn_enemy(MonsterKind::Wriggler);
        let pre = state.player.discard.len();
        wriggle_effect(&mut state, uid);
        assert_eq!(creature_strength(&state, uid), 2);
        assert_eq!(state.player.discard.len(), pre + 1);
        let cuid = *state.player.discard.last().expect("one card");
        let idx = cuid.0.get() as usize - 1;
        assert_eq!(state.cards[idx].card_id, CardId::Infection);
    }

    #[test]
    fn adversary_barrage_applies_scaling_strength_to_self() {
        use sts2_core::MonsterKind;
        for (kind, effect, amount) in [
            (
                MonsterKind::TheAdversaryMkOne,
                adversary_mk_one_barrage_effect as fn(&mut CombatState, sts2_core::CreatureUid),
                2,
            ),
            (
                MonsterKind::TheAdversaryMkTwo,
                adversary_mk_two_barrage_effect,
                3,
            ),
            (
                MonsterKind::TheAdversaryMkThree,
                adversary_mk_three_barrage_effect,
                4,
            ),
        ] {
            let mut state = CombatState::default();
            let uid = state.spawn_enemy(kind);
            effect(&mut state, uid);
            assert_eq!(creature_strength(&state, uid), amount);
        }
    }

    #[test]
    fn test_monsters_emit_expected_intents() {
        let poke = create_single_attack_move_monster_ai().current_move().id;
        assert_eq!(poke, "POKE");
        let multi = create_multi_attack_move_monster_ai().current_move().id;
        assert_eq!(multi, "POKE");
    }

    /// Mutation coverage (issue #65): `apply_power_to_player_via_legacy_surface`
    /// (shared.rs:146) must stack into the SAME-kind player instance, ignoring
    /// a decoy power of a different kind. Kills the `==`/`&&`/`==` mutants on
    /// the existing-power lookup `p.id == puid.0 && p.kind == kind`.
    #[test]
    fn legacy_surface_stacks_same_kind_past_decoy() {
        let mut state = CombatState::default();

        let player_power_amount = |state: &CombatState, kind: PowerKind| -> i32 {
            state
                .player
                .powers
                .iter()
                .find_map(|pu| {
                    state
                        .powers
                        .iter()
                        .find(|p| p.id == pu.0 && p.kind == kind)
                        .map(|p| p.amount)
                })
                .unwrap_or(0)
        };

        // Decoy of a DIFFERENT kind first.
        apply_power_to_player_via_legacy_surface(&mut state, PowerKind::Vulnerable, 2);
        // First Frail creates a new instance; second must find + stack it.
        apply_power_to_player_via_legacy_surface(&mut state, PowerKind::Frail, 1);
        apply_power_to_player_via_legacy_surface(&mut state, PowerKind::Frail, 3);

        assert_eq!(
            player_power_amount(&state, PowerKind::Frail),
            4,
            "second Frail stacks into the existing instance (1+3)"
        );
        let frail_instances = state
            .player
            .powers
            .iter()
            .filter(|pu| {
                state
                    .powers
                    .iter()
                    .any(|p| p.id == pu.0 && p.kind == PowerKind::Frail)
            })
            .count();
        assert_eq!(frail_instances, 1, "no duplicate Frail instance created");
        assert_eq!(player_power_amount(&state, PowerKind::Vulnerable), 2);
    }

    /// Mutation coverage (issue #65): `apply_power_to_player_via_legacy_surface`
    /// id-match (shared.rs:146:32). With NO decoy power present, stacking the
    /// SAME kind twice must reuse the one instance — under `id != puid.0` the
    /// lookup finds nothing and pushes a duplicate. Kills `==`→`!=` (146:32).
    #[test]
    fn legacy_surface_stacks_into_single_instance_no_decoy() {
        let mut state = CombatState::default();
        apply_power_to_player_via_legacy_surface(&mut state, PowerKind::Frail, 1);
        apply_power_to_player_via_legacy_surface(&mut state, PowerKind::Frail, 3);
        let frail_instances = state
            .player
            .powers
            .iter()
            .filter(|pu| {
                state
                    .powers
                    .iter()
                    .any(|p| p.id == pu.0 && p.kind == PowerKind::Frail)
            })
            .count();
        assert_eq!(
            frail_instances, 1,
            "stacking reuses the single Frail instance (id match)"
        );
        assert_eq!(
            state.player.powers.len(),
            1,
            "exactly one player power total"
        );
        let amount = state
            .player
            .powers
            .iter()
            .find_map(|pu| {
                state
                    .powers
                    .iter()
                    .find(|p| p.id == pu.0 && p.kind == PowerKind::Frail)
                    .map(|p| p.amount)
            })
            .unwrap_or(0);
        assert_eq!(amount, 4, "1 + 3 stacked");
    }
}
