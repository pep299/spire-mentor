//! Encounter pools.
//!
//! Static, hand-coded encounter specs for the P9 vertical-slice run
//! loop. An [`EncounterSpec`] is the run-side identity of a single
//! fight: an ordered list of [`EncounterMember`]s (monster kind +
//! starting HP). The bridge layer (P9.0.c `enter_combat`) is
//! responsible for materializing each member into a `(Creature,
//! MonsterAi)` pair by dispatching on [`MonsterKind`] — the AI
//! factories live in `sts2-content::monsters::*` so the dep direction
//! stays `content → core` (encounters.rs in core only carries the
//! kind + HP data).
//!
//! Scope:
//! - **Act 1** — 8 normal (Nibbit/Mawler/Fogmog/Cubex/VineShambler/
//!   Flyconid/SnappingJaxfruit/SlitheringStrangler) + 1 boss (`vantom`); bead
//!   `sts2-rs-5mfyd` (P9.0.b thin slice).
//! - **Act 2** — 11 normal + 3 elite + 3 boss, the full per-act
//!   roster from `sts2_env/encounters/act2.py`; bead `sts2-rs-9cz6b`
//!   (P9.2.o).
//! - **Act 3** — 9 normal + 3 elite + 3 boss, the full per-act
//!   roster from `sts2_env/encounters/act3.py`; bead `sts2-rs-9cz6b`
//!   (P9.2.o).
//! - **No RNG selection** — every accessor is paired with
//!   [`select_first`]. The RNG-driven sampler lands in P9.1.f.
//! - **No weak-encounter pools** — weak encounters are act-1 only
//!   gameplay-wise (acts 2/3 use the same pool for the first floor as
//!   for subsequent normals at the pin's map driver), so this module
//!   carries `normal`/`elite`/`boss` accessors only and leaves the
//!   weak Python lists out until a downstream sampler asks for them.
//!
//! HP values track each Python pin constant (`*_BASE_HP` /
//! `*_BASE_MIN_HP..*_BASE_MAX_HP` median). The `min..max` ranges
//! collapse to a single deterministic value for the thin slice — the
//! RNG roll lands alongside the sampler. Ascension scaling
//! (`*_TOUGH_HP`) is out of scope at the A0 thin slice.
//!
//! Mirrors a slice of `sts2-rl-agent@01c094d:sts2_env/run/run_manager.py`
//! `_get_encounter_pools(act_index)` and the per-act pool definitions
//! in `sts2_env/encounters/act{1,2,3}.py`.

use crate::monsters::MonsterKind;

/// A single monster in an [`EncounterSpec`]: identity + starting HP.
///
/// The HP value is the post-roll concrete value used by the thin
/// slice (P9.1.f replaces this with a `(min, max)` band + RNG roll).
/// AI is resolved from [`MonsterKind`] by the bridge layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EncounterMember {
    /// Monster discriminant — dispatch key for the AI factory.
    pub kind: MonsterKind,
    /// Starting HP (and max HP) for this member at combat init.
    pub hp: i32,
}

/// One encounter — the monsters in a single fight.
///
/// `name` is a stable identifier (snake_case, matches the Python
/// `setup_*` function name minus the leading `setup_`) used by parity
/// fixtures and bead/audit cross-refs. `members` is ordered: index 0
/// is the front-most (player-facing) monster.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EncounterSpec {
    /// Stable snake_case identifier (e.g. `"nibbit_solo"`).
    pub name: &'static str,
    /// Ordered roster; index 0 is the front-most monster.
    pub members: &'static [EncounterMember],
}

// ---------------------------------------------------------------------------
// Act 1 — normal pool
// ---------------------------------------------------------------------------

const ACT1_NORMAL_NIBBIT_SOLO_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::Nibbit,
    // Python pin `monsters/act1_weak.py::create_nibbit` rolls HP in
    // [44, 46]; the thin slice pins to 45 (median) for determinism.
    hp: 45,
}];

// Act-1 normal pool, expanded past the original Nibbit-only thin slice
// (sts2-rs-?? / workbench bug: every Act-1 monster room spawned Nibbit
// because the pool had a single entry). HP per member is pinned to the
// median of the monster's `MonsterKind::hp_roll` range — matching the
// pinned-HP convention the rest of this table already uses — so picks
// stay deterministic. Only encounters whose monsters resolve a working
// AI under `MonsterSpawnArgs::None` (the arg the run bridge passes) are
// included; arg-dependent rosters (Inklets / slimes / ruby-raider
// randomizer) are deferred to the richer content `EncounterDef` table.
const ACT1_NORMAL_MAWLER_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::Mawler,
    hp: 72,
}];
const ACT1_NORMAL_FOGMOG_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::Fogmog,
    hp: 74,
}];
const ACT1_NORMAL_CUBEX_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::CubexConstruct,
    hp: 65,
}];
const ACT1_NORMAL_VINE_SHAMBLER_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::VineShambler,
    hp: 61,
}];
const ACT1_NORMAL_FLYCONID_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::LeafSlimeM,
        hp: 33,
    },
    EncounterMember {
        kind: MonsterKind::Flyconid,
        hp: 48,
    },
];
const ACT1_NORMAL_SNAPPING_JAXFRUIT_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::SnappingJaxfruit,
        hp: 32,
    },
    EncounterMember {
        kind: MonsterKind::Flyconid,
        hp: 48,
    },
];
const ACT1_NORMAL_SLITHERING_STRANGLER_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::SnappingJaxfruit,
        hp: 32,
    },
    EncounterMember {
        kind: MonsterKind::SlitheringStrangler,
        hp: 54,
    },
];

// C# `TwoTailedRatsNormal.GenerateMonsters`: three TwoTailedRats with a
// rotated StarterMoveIndex so each opens on a distinct move. `build_ai`
// maps the member's `position.index` (0/1/2) onto SCRATCH / DISEASE_BITE /
// SCREECH, reproducing that distinct-starter rotation. HP pinned to the
// median of `MonsterKind::hp_roll` 17..=21 like the rest of this table.
// Bead `sts2-rs-dvude`.
const ACT1_NORMAL_TWO_TAILED_RATS_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::TwoTailedRat,
        hp: 19,
    },
    EncounterMember {
        kind: MonsterKind::TwoTailedRat,
        hp: 19,
    },
    EncounterMember {
        kind: MonsterKind::TwoTailedRat,
        hp: 19,
    },
];

// The pools are `static`, not `const`: callers identify an entry by address
// (`std::ptr::eq(spec, &pool[i])`), and a `const` has no guaranteed single
// address. Built with `opt-level = 1` (the dev profile below), 9 such
// assertions in sts2-content failed against a `const` pool.
static ACT1_NORMAL: &[EncounterSpec] = &[
    // `nibbit_solo` MUST stay at index 0: `run_bridge` tests and
    // `select_first(act1_normal_pool())` callers pin the canonical
    // first entry to Nibbit@45.
    EncounterSpec {
        name: "nibbit_solo",
        members: ACT1_NORMAL_NIBBIT_SOLO_MEMBERS,
    },
    EncounterSpec {
        name: "mawler_normal",
        members: ACT1_NORMAL_MAWLER_MEMBERS,
    },
    EncounterSpec {
        name: "fogmog_normal",
        members: ACT1_NORMAL_FOGMOG_MEMBERS,
    },
    EncounterSpec {
        name: "cubex_construct_normal",
        members: ACT1_NORMAL_CUBEX_MEMBERS,
    },
    EncounterSpec {
        name: "vine_shambler_normal",
        members: ACT1_NORMAL_VINE_SHAMBLER_MEMBERS,
    },
    EncounterSpec {
        name: "flyconid_normal",
        members: ACT1_NORMAL_FLYCONID_MEMBERS,
    },
    EncounterSpec {
        name: "snapping_jaxfruit_normal",
        members: ACT1_NORMAL_SNAPPING_JAXFRUIT_MEMBERS,
    },
    EncounterSpec {
        name: "slithering_strangler_normal",
        members: ACT1_NORMAL_SLITHERING_STRANGLER_MEMBERS,
    },
    EncounterSpec {
        name: "two_tailed_rats_normal",
        members: ACT1_NORMAL_TWO_TAILED_RATS_MEMBERS,
    },
];

// ---------------------------------------------------------------------------
// Act 1 — boss pool
// ---------------------------------------------------------------------------

const ACT1_BOSS_VANTOM_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::Vantom,
    // Python pin `monsters/act1.py::VANTOM_BASE_HP = 173`. The
    // Tough-ascension variant (183) is gated A7+ and is out of scope
    // for the A0 thin slice.
    hp: 173,
}];

static ACT1_BOSS: &[EncounterSpec] = &[EncounterSpec {
    name: "vantom",
    members: ACT1_BOSS_VANTOM_MEMBERS,
}];

// ---------------------------------------------------------------------------
// Act 2 — normal pool (Python: encounters/act2.py::NORMAL_ENCOUNTERS,
// 11 entries). HP values are the median of each Python pin's
// `*_BASE_*_HP` constants in `monsters/act2.py`.
// ---------------------------------------------------------------------------

const ACT2_NORMAL_BOWLBUGS_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::BowlbugEgg,
        hp: 22,
    },
    EncounterMember {
        kind: MonsterKind::BowlbugRock,
        hp: 47,
    },
    EncounterMember {
        kind: MonsterKind::BowlbugSilk,
        hp: 42,
    },
];
const ACT2_NORMAL_CHOMPERS_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::Chomper,
        hp: 62,
    },
    EncounterMember {
        kind: MonsterKind::Chomper,
        hp: 62,
    },
];
const ACT2_NORMAL_EXOSKELETONS_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::Exoskeleton,
        hp: 26,
    },
    EncounterMember {
        kind: MonsterKind::Exoskeleton,
        hp: 26,
    },
    EncounterMember {
        kind: MonsterKind::Exoskeleton,
        hp: 26,
    },
];
const ACT2_NORMAL_HUNTER_KILLER_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::HunterKiller,
    hp: 121,
}];
const ACT2_NORMAL_LOUSE_PROGENITOR_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::LouseProgenitor,
    hp: 135,
}];
const ACT2_NORMAL_MYTES_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::Myte,
        hp: 64,
    },
    EncounterMember {
        kind: MonsterKind::Myte,
        hp: 64,
    },
];
const ACT2_NORMAL_OVICOPTER_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::Ovicopter,
    hp: 127,
}];
const ACT2_NORMAL_SLUMBERING_BEETLE_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::SlumberingBeetle,
    hp: 86,
}];
const ACT2_NORMAL_SPINY_TOAD_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::SpinyToad,
    hp: 118,
}];
const ACT2_NORMAL_THE_OBSCURA_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::TheObscura,
    hp: 123,
}];
// `setup_tunneler_normal` rolls the bug companion between BowlbugEgg
// and BowlbugSilk; the thin slice pins BowlbugEgg per Python list
// order (RNG sampler lands in P9.1.f).
const ACT2_NORMAL_TUNNELER_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::BowlbugEgg,
        hp: 22,
    },
    EncounterMember {
        kind: MonsterKind::Tunneler,
        hp: 87,
    },
];

static ACT2_NORMAL: &[EncounterSpec] = &[
    EncounterSpec {
        name: "bowlbugs_normal",
        members: ACT2_NORMAL_BOWLBUGS_MEMBERS,
    },
    EncounterSpec {
        name: "chompers_normal",
        members: ACT2_NORMAL_CHOMPERS_MEMBERS,
    },
    EncounterSpec {
        name: "exoskeletons_normal",
        members: ACT2_NORMAL_EXOSKELETONS_MEMBERS,
    },
    EncounterSpec {
        name: "hunter_killer_normal",
        members: ACT2_NORMAL_HUNTER_KILLER_MEMBERS,
    },
    EncounterSpec {
        name: "louse_progenitor_normal",
        members: ACT2_NORMAL_LOUSE_PROGENITOR_MEMBERS,
    },
    EncounterSpec {
        name: "mytes_normal",
        members: ACT2_NORMAL_MYTES_MEMBERS,
    },
    EncounterSpec {
        name: "ovicopter_normal",
        members: ACT2_NORMAL_OVICOPTER_MEMBERS,
    },
    EncounterSpec {
        name: "slumbering_beetle_normal",
        members: ACT2_NORMAL_SLUMBERING_BEETLE_MEMBERS,
    },
    EncounterSpec {
        name: "spiny_toad_normal",
        members: ACT2_NORMAL_SPINY_TOAD_MEMBERS,
    },
    EncounterSpec {
        name: "the_obscura_normal",
        members: ACT2_NORMAL_THE_OBSCURA_MEMBERS,
    },
    EncounterSpec {
        name: "tunneler_normal",
        members: ACT2_NORMAL_TUNNELER_MEMBERS,
    },
];

// ---------------------------------------------------------------------------
// Act 2 — elite pool (Python: encounters/act2.py::ELITE_ENCOUNTERS,
// 3 entries).
// ---------------------------------------------------------------------------

const ACT2_ELITE_DECIMILLIPEDE_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::DecimillipedeSegment,
        hp: 45,
    },
    EncounterMember {
        kind: MonsterKind::DecimillipedeSegment,
        hp: 45,
    },
    EncounterMember {
        kind: MonsterKind::DecimillipedeSegment,
        hp: 45,
    },
];
const ACT2_ELITE_ENTOMANCER_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::Entomancer,
    hp: 145,
}];
const ACT2_ELITE_INFESTED_PRISMS_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::InfestedPrism,
    hp: 200,
}];

static ACT2_ELITE: &[EncounterSpec] = &[
    EncounterSpec {
        name: "decimillipede_elite",
        members: ACT2_ELITE_DECIMILLIPEDE_MEMBERS,
    },
    EncounterSpec {
        name: "entomancer_elite",
        members: ACT2_ELITE_ENTOMANCER_MEMBERS,
    },
    EncounterSpec {
        name: "infested_prisms_elite",
        members: ACT2_ELITE_INFESTED_PRISMS_MEMBERS,
    },
];

// ---------------------------------------------------------------------------
// Act 2 — boss pool (Python: encounters/act2.py::BOSS_ENCOUNTERS, 3
// entries: TheInsatiable / KnowledgeDemon / KaiserCrab = Crusher +
// Rocket combo).
// ---------------------------------------------------------------------------

const ACT2_BOSS_THE_INSATIABLE_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::TheInsatiable,
    hp: 321,
}];
const ACT2_BOSS_KNOWLEDGE_DEMON_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::KnowledgeDemon,
    hp: 379,
}];
// Kaiser Crab: Crusher (front) + Rocket (back). Rocket's
// `SURROUNDED` power is applied by the Python setup function; the
// EncounterSpec carries only kind + HP so the room-setup wiring is
// the bridge layer's job (mirrors how Act-1 Cubex room-setup wires
// from `enter_combat`).
const ACT2_BOSS_KAISER_CRAB_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::Crusher,
        hp: 199,
    },
    EncounterMember {
        // C# `Rocket.cs:40` MinInitialHp = MaxInitialHp = 199 (asc-0).
        // The `monsters.rs` HpRoll table was corrected 189→199 but this
        // encounter spec — what `run_bridge` actually spawns from — was
        // left stale (sts2-rs monster-audit divergent tail).
        kind: MonsterKind::Rocket,
        hp: 199,
    },
];

static ACT2_BOSS: &[EncounterSpec] = &[
    EncounterSpec {
        name: "the_insatiable_boss",
        members: ACT2_BOSS_THE_INSATIABLE_MEMBERS,
    },
    EncounterSpec {
        name: "knowledge_demon_boss",
        members: ACT2_BOSS_KNOWLEDGE_DEMON_MEMBERS,
    },
    EncounterSpec {
        name: "kaiser_crab_boss",
        members: ACT2_BOSS_KAISER_CRAB_MEMBERS,
    },
];

// ---------------------------------------------------------------------------
// Act 3 — normal pool (Python: encounters/act3.py::NORMAL_ENCOUNTERS,
// 9 entries).
// ---------------------------------------------------------------------------

// `…wb8lk.11`: v0.107.1 AxebotsNormal is a SINGLE Axebot (HP 70 at A0).
const ACT3_NORMAL_AXEBOTS_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::Axebot,
    hp: 70,
}];
const ACT3_NORMAL_CONSTRUCT_MENAGERIE_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::PunchConstruct,
        hp: 55,
    },
    EncounterMember {
        kind: MonsterKind::CubexConstruct,
        hp: 65,
    },
    EncounterMember {
        kind: MonsterKind::CubexConstruct,
        hp: 65,
    },
];
const ACT3_NORMAL_FABRICATOR_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::Fabricator,
    hp: 150,
}];
const ACT3_NORMAL_FROG_KNIGHT_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::FrogKnight,
    hp: 191,
}];
const ACT3_NORMAL_GLOBE_HEAD_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::GlobeHead,
    hp: 148,
}];
const ACT3_NORMAL_OWL_MAGISTRATE_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::OwlMagistrate,
    hp: 234,
}];
// Python `setup_scrolls_of_biting_normal` spawns three RNG-rotated
// starter-move scrolls + a fourth fixed-move scroll. The thin slice
// pins 4 scrolls at the median 35 HP each.
const ACT3_NORMAL_SCROLLS_OF_BITING_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::ScrollOfBiting,
        hp: 35,
    },
    EncounterMember {
        kind: MonsterKind::ScrollOfBiting,
        hp: 35,
    },
    EncounterMember {
        kind: MonsterKind::ScrollOfBiting,
        hp: 35,
    },
    EncounterMember {
        kind: MonsterKind::ScrollOfBiting,
        hp: 35,
    },
];
const ACT3_NORMAL_SLIMED_BERSERKER_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::SlimedBerserker,
    hp: 266,
}];
const ACT3_NORMAL_LOST_AND_FORGOTTEN_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::TheLost,
        hp: 93,
    },
    EncounterMember {
        kind: MonsterKind::TheForgotten,
        hp: 106,
    },
];

static ACT3_NORMAL: &[EncounterSpec] = &[
    EncounterSpec {
        name: "axebots_normal",
        members: ACT3_NORMAL_AXEBOTS_MEMBERS,
    },
    EncounterSpec {
        name: "construct_menagerie_normal",
        members: ACT3_NORMAL_CONSTRUCT_MENAGERIE_MEMBERS,
    },
    EncounterSpec {
        name: "fabricator_normal",
        members: ACT3_NORMAL_FABRICATOR_MEMBERS,
    },
    EncounterSpec {
        name: "frog_knight_normal",
        members: ACT3_NORMAL_FROG_KNIGHT_MEMBERS,
    },
    EncounterSpec {
        name: "globe_head_normal",
        members: ACT3_NORMAL_GLOBE_HEAD_MEMBERS,
    },
    EncounterSpec {
        name: "owl_magistrate_normal",
        members: ACT3_NORMAL_OWL_MAGISTRATE_MEMBERS,
    },
    EncounterSpec {
        name: "scrolls_of_biting_normal",
        members: ACT3_NORMAL_SCROLLS_OF_BITING_MEMBERS,
    },
    EncounterSpec {
        name: "slimed_berserker_normal",
        members: ACT3_NORMAL_SLIMED_BERSERKER_MEMBERS,
    },
    EncounterSpec {
        name: "the_lost_and_forgotten_normal",
        members: ACT3_NORMAL_LOST_AND_FORGOTTEN_MEMBERS,
    },
];

// ---------------------------------------------------------------------------
// Act 3 — elite pool (Python: encounters/act3.py::ELITE_ENCOUNTERS,
// 3 entries).
// ---------------------------------------------------------------------------

const ACT3_ELITE_KNIGHTS_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::FlailKnight,
        hp: 101,
    },
    EncounterMember {
        kind: MonsterKind::SpectralKnight,
        hp: 93,
    },
    EncounterMember {
        kind: MonsterKind::MagiKnight,
        hp: 82,
    },
];
const ACT3_ELITE_MECHA_KNIGHT_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::MechaKnight,
    hp: 300,
}];
const ACT3_ELITE_SOUL_NEXUS_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::SoulNexus,
    hp: 234,
}];

static ACT3_ELITE: &[EncounterSpec] = &[
    EncounterSpec {
        name: "knights_elite",
        members: ACT3_ELITE_KNIGHTS_MEMBERS,
    },
    EncounterSpec {
        name: "mecha_knight_elite",
        members: ACT3_ELITE_MECHA_KNIGHT_MEMBERS,
    },
    EncounterSpec {
        name: "soul_nexus_elite",
        members: ACT3_ELITE_SOUL_NEXUS_MEMBERS,
    },
];

// ---------------------------------------------------------------------------
// Act 3 — boss pool (C# `Glory.BossDiscoveryOrder`, v0.107.1: 3
// entries Queen / TestSubject / Aeonglass; list order is Queen,
// TestSubject, Aeonglass — Aeonglass replaced Doormaker).
// ---------------------------------------------------------------------------

// Queen boss pairs a TorchHeadAmalgam side spawn with the Queen
// herself; Python `setup_queen_boss` spawns the amalgam first then
// the Queen.
const ACT3_BOSS_QUEEN_MEMBERS: &[EncounterMember] = &[
    EncounterMember {
        kind: MonsterKind::TorchHeadAmalgam,
        hp: 199,
    },
    EncounterMember {
        kind: MonsterKind::Queen,
        hp: 400,
    },
];
// TestSubject has three sequential forms (100 → 200 → 300 HP); the
// EncounterSpec pins the first-form HP since the form transitions
// are AI-driven state changes, not separate encounter members.
const ACT3_BOSS_TEST_SUBJECT_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::TestSubject,
    hp: 100,
}];
// Aeonglass boss (v0.107.1, replaces Doormaker): a SINGLE Aeonglass, HP 512
// (A0) / 535 (ToughEnemies). C# `AeonglassBoss.GenerateMonsters` is a
// one-element list of `Aeonglass`. Bead `sts2-rs-v0107-content-port-wb8lk.2`.
const ACT3_BOSS_AEONGLASS_MEMBERS: &[EncounterMember] = &[EncounterMember {
    kind: MonsterKind::Aeonglass,
    hp: 512,
}];

static ACT3_BOSS: &[EncounterSpec] = &[
    EncounterSpec {
        name: "queen_boss",
        members: ACT3_BOSS_QUEEN_MEMBERS,
    },
    EncounterSpec {
        name: "test_subject_boss",
        members: ACT3_BOSS_TEST_SUBJECT_MEMBERS,
    },
    EncounterSpec {
        name: "aeonglass_boss",
        members: ACT3_BOSS_AEONGLASS_MEMBERS,
    },
];

// ---------------------------------------------------------------------------
// Public accessors
// ---------------------------------------------------------------------------

/// Act-1 normal-encounter pool. Length 1 at this pin.
#[must_use]
pub fn act1_normal_pool() -> &'static [EncounterSpec] {
    ACT1_NORMAL
}

/// Act-1 boss-encounter pool. Length 1 at this pin.
#[must_use]
pub fn act1_boss_pool() -> &'static [EncounterSpec] {
    ACT1_BOSS
}

/// Act-2 normal-encounter pool. Length 11 — full roster from
/// `sts2_env/encounters/act2.py::NORMAL_ENCOUNTERS`.
#[must_use]
pub fn act2_normal_pool() -> &'static [EncounterSpec] {
    ACT2_NORMAL
}

/// Act-2 elite-encounter pool. Length 3 — full roster from
/// `sts2_env/encounters/act2.py::ELITE_ENCOUNTERS`.
#[must_use]
pub fn act2_elite_pool() -> &'static [EncounterSpec] {
    ACT2_ELITE
}

/// Act-2 boss-encounter pool. Length 3 — full roster from
/// `sts2_env/encounters/act2.py::BOSS_ENCOUNTERS`.
#[must_use]
pub fn act2_boss_pool() -> &'static [EncounterSpec] {
    ACT2_BOSS
}

/// Act-3 normal-encounter pool. Length 9 — full roster from
/// `sts2_env/encounters/act3.py::NORMAL_ENCOUNTERS`.
#[must_use]
pub fn act3_normal_pool() -> &'static [EncounterSpec] {
    ACT3_NORMAL
}

/// Act-3 elite-encounter pool. Length 3 — full roster from
/// `sts2_env/encounters/act3.py::ELITE_ENCOUNTERS`.
#[must_use]
pub fn act3_elite_pool() -> &'static [EncounterSpec] {
    ACT3_ELITE
}

/// Act-3 boss-encounter pool. Length 3 — full roster from
/// `sts2_env/encounters/act3.py::BOSS_ENCOUNTERS`.
#[must_use]
pub fn act3_boss_pool() -> &'static [EncounterSpec] {
    ACT3_BOSS
}

/// Deterministic pool sampler used by P9.0 — always returns the
/// first entry. Replaced by an RNG-driven sampler in P9.1.f.
///
/// # Panics
/// Panics if `pool` is empty; the static pools above are never empty.
#[must_use]
pub fn select_first(pool: &'static [EncounterSpec]) -> &'static EncounterSpec {
    &pool[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn act1_pools_are_non_empty() {
        assert!(!act1_normal_pool().is_empty());
        assert!(!act1_boss_pool().is_empty());
    }

    #[test]
    fn act1_normal_canonical_first_is_nibbit_solo() {
        // `select_first` (index 0) must stay Nibbit@45 — run_bridge tests
        // and other select_first callers pin this canonical entry.
        let spec = select_first(act1_normal_pool());
        assert_eq!(spec.name, "nibbit_solo");
        assert_eq!(spec.members.len(), 1);
        assert_eq!(spec.members[0].kind, MonsterKind::Nibbit);
        assert_eq!(spec.members[0].hp, 45);
    }

    #[test]
    fn act1_normal_pool_has_variety() {
        // Regression: the pool used to hold a single encounter, so every
        // Act-1 monster room spawned Nibbit. It must now offer several
        // distinct encounters drawing on more than one monster kind.
        let pool = act1_normal_pool();
        assert!(pool.len() > 1, "act-1 normal pool must offer >1 encounter");
        let kinds: std::collections::HashSet<MonsterKind> = pool
            .iter()
            .flat_map(|e| e.members.iter().map(|m| m.kind))
            .collect();
        assert!(
            kinds.len() > 1,
            "pool must span >1 monster kind, got {kinds:?}"
        );
        // Every pinned HP must be positive (guards a typo'd 0 HP).
        for e in pool {
            for m in e.members {
                assert!(
                    m.hp > 0,
                    "{}: member {:?} has non-positive hp",
                    e.name,
                    m.kind
                );
            }
        }
    }

    #[test]
    fn act1_boss_thin_slice_is_vantom() {
        let spec = select_first(act1_boss_pool());
        assert_eq!(spec.name, "vantom");
        assert_eq!(spec.members.len(), 1);
        assert_eq!(spec.members[0].kind, MonsterKind::Vantom);
        assert_eq!(spec.members[0].hp, 173);
    }

    #[test]
    fn select_first_is_deterministic() {
        let pool = act1_normal_pool();
        assert!(std::ptr::eq(select_first(pool), select_first(pool)));
    }

    // ── Act 2 / Act 3 (bead `sts2-rs-9cz6b`) ─────────────────────────

    /// Act-2 pool sizes match the Python pin counts exactly. Pinning
    /// length here keeps the per-act sampler honest: an accidentally
    /// half-ported pool fails this check instead of silently shrinking
    /// the per-floor roster the run loop draws from.
    #[test]
    fn act2_pool_lengths_match_python_pin() {
        assert_eq!(act2_normal_pool().len(), 11, "act2 normal");
        assert_eq!(act2_elite_pool().len(), 3, "act2 elite");
        assert_eq!(act2_boss_pool().len(), 3, "act2 boss");
    }

    /// Act-3 pool sizes match the Python pin counts exactly.
    #[test]
    fn act3_pool_lengths_match_python_pin() {
        assert_eq!(act3_normal_pool().len(), 9, "act3 normal");
        assert_eq!(act3_elite_pool().len(), 3, "act3 elite");
        assert_eq!(act3_boss_pool().len(), 3, "act3 boss");
    }

    /// Every encounter spec has at least one member. Acts as a
    /// regression check against a future const accidentally pointing
    /// at an empty member slice.
    #[test]
    fn every_pool_entry_has_at_least_one_member() {
        for pool in [
            act1_normal_pool(),
            act1_boss_pool(),
            act2_normal_pool(),
            act2_elite_pool(),
            act2_boss_pool(),
            act3_normal_pool(),
            act3_elite_pool(),
            act3_boss_pool(),
        ] {
            for spec in pool {
                assert!(!spec.members.is_empty(), "{} has no members", spec.name);
            }
        }
    }

    /// Encounter names are unique within each pool — guards against
    /// copy-paste duplicates that would silently bias the eventual
    /// uniform RNG sampler.
    #[test]
    fn names_are_unique_within_each_pool() {
        for pool in [
            act2_normal_pool(),
            act2_elite_pool(),
            act2_boss_pool(),
            act3_normal_pool(),
            act3_elite_pool(),
            act3_boss_pool(),
        ] {
            let mut names: Vec<&str> = pool.iter().map(|s| s.name).collect();
            names.sort_unstable();
            let len_before_dedup = names.len();
            names.dedup();
            assert_eq!(len_before_dedup, names.len(), "duplicate names in pool");
        }
    }

    /// Act-2 boss roster matches the canonical Python order:
    /// TheInsatiable / KnowledgeDemon / KaiserCrab.
    #[test]
    fn act2_boss_order_matches_python_pin() {
        let pool = act2_boss_pool();
        assert_eq!(pool[0].name, "the_insatiable_boss");
        assert_eq!(pool[1].name, "knowledge_demon_boss");
        assert_eq!(pool[2].name, "kaiser_crab_boss");
        assert_eq!(pool[2].members.len(), 2);
        assert_eq!(pool[2].members[0].kind, MonsterKind::Crusher);
        assert_eq!(pool[2].members[1].kind, MonsterKind::Rocket);
    }

    /// Act-3 boss roster matches the C# `Glory.BossDiscoveryOrder` (v0.107.1):
    /// Queen / TestSubject / Aeonglass.
    #[test]
    fn act3_boss_order_matches_python_pin() {
        let pool = act3_boss_pool();
        assert_eq!(pool[0].name, "queen_boss");
        assert_eq!(pool[1].name, "test_subject_boss");
        assert_eq!(pool[2].name, "aeonglass_boss");
        assert_eq!(pool[2].members[0].kind, MonsterKind::Aeonglass);
        assert_eq!(pool[0].members[0].kind, MonsterKind::TorchHeadAmalgam);
        assert_eq!(pool[0].members[1].kind, MonsterKind::Queen);
    }

    /// HP values are positive across every pool — guards against a
    /// zero-HP slip-up while transcribing from the Python pin.
    #[test]
    fn every_member_hp_is_positive() {
        for pool in [
            act1_normal_pool(),
            act1_boss_pool(),
            act2_normal_pool(),
            act2_elite_pool(),
            act2_boss_pool(),
            act3_normal_pool(),
            act3_elite_pool(),
            act3_boss_pool(),
        ] {
            for spec in pool {
                for m in spec.members {
                    assert!(
                        m.hp > 0,
                        "{} member {:?} has non-positive hp",
                        spec.name,
                        m.kind
                    );
                }
            }
        }
    }
}
