//! Encounter-group registry (`docs/plans/2026-05-14-…-cli.md` decision D3).
//!
//! Static tables mapping a canonical Python encounter id to its
//! `Vec<(MonsterKind, MonsterSpawnArgs)>` composition. Phase 3 wires
//! Act 1 (`sts2-rs-k923`); future Acts 2/3/4 add sibling modules.
//!
//! `EncounterDef` is `&'static`-friendly so the CLI (`sts2-rs-90n5`)
//! can iterate the table without an engine round-trip. Three Python
//! encounters (`slimes_weak`, `ruby_raiders_normal`, `knights_elite`)
//! draw a random sub-selection at spawn time — they expose an optional
//! [`EncounterRandomizer`] (bead `sts2-rs-6inu`) that callers route an
//! RNG through via [`EncounterDef::roster`]. The static `monsters`
//! slice on those rows is the canonical fixed permutation, kept as the
//! deterministic fallback for callers that don't have an RNG handy
//! (parity fixtures, doc listings).

// Zone-named per C# canonical act classes. Act 1 ships TWO models at the
// same index — Overgrowth (`IsDefault`) and Underdocks — and a run walks
// exactly one of them; see `sts2_core::act_selection` (GH #544).
use sts2_core::act_selection::ActVariant;
pub mod act1_overgrowth;
pub mod act1_underdocks;
pub mod act2_hive;
pub mod act3_glory;

use crate::monster_dispatch::MonsterSpawnArgs;
use smallvec::SmallVec;
use sts2_core::MonsterKind;

/// Inline capacity for an encounter's monster composition. Tuned to the
/// largest in-table encounter (Act 1 Slimes = 3) plus headroom for any
/// future randomized roster that draws more.
pub const ENCOUNTER_ROSTER_INLINE: usize = 8;

/// Concrete return type for [`EncounterRandomizer`] / [`EncounterDef::roster`].
pub type EncounterRoster = SmallVec<[(MonsterKind, MonsterSpawnArgs); ENCOUNTER_ROSTER_INLINE]>;

/// C# `Rng.NextItem(list)` = `list[NextInt(0, list.Count)]`. The single
/// `next_int_exclusive(0, len)` read happens **even for a 1-element list**
/// — C# `Rng.NextInt(0, 1)` still advances the counter (see
/// `feedback_csharp_reads_rng_for_single_element_range`), so a faithful
/// port MUST keep the read to stay cursor-aligned with the surrounding
/// stream. The encounter randomizers below all draw their roster via this
/// helper, mirroring `EncounterModel.GenerateMonsters`'s `base.Rng.NextItem`.
pub(crate) fn next_item<T: Copy>(rng: &mut sts2_core::dotnet_rng::Rng, list: &[T]) -> T {
    debug_assert!(!list.is_empty(), "NextItem on an empty list");
    list[rng.next_int_exclusive(0, list.len() as i32) as usize]
}

/// Spawn-time RNG draw for encounters whose Python `setup_*` function
/// samples a sub-selection of monsters (`slimes_weak`,
/// `ruby_raiders_normal`, `knights_elite`). Bead `sts2-rs-6inu`.
///
/// The randomizer accepts a `&mut dyn rand::RngCore` so callers can pass
/// any concrete RNG (the engine uses `Xoshiro256PlusPlus`; tests use the
/// same; future map/run-state layers may use a derived RNG) without
/// `sts2-content` taking on a hard dep on a specific concrete type. The
/// returned roster mirrors the static `EncounterDef::monsters` slot in
/// `(MonsterKind, MonsterSpawnArgs)` shape so downstream consumers
/// (`specs_from_encounter_def`, parity bridge state) can ignore whether
/// the roster came from the static slice or the randomizer.
pub type EncounterRandomizer = fn(&mut sts2_core::dotnet_rng::Rng) -> EncounterRoster;

/// Resolve an encounter id across ALL act registries (Act 1 → 2 → 3),
/// returning the first match. Encounter ids are globally unique
/// (`UPPER_SNAKE` per the C# class name), so chaining the per-act `find`s
/// is unambiguous. Callers that know the act can still hit the per-act
/// `find` directly; this is for act-agnostic consumers — the PyO3
/// scenario builder (`build_scenario_state`) and the parity harness, which
/// pass an encounter id without tracking which act it lives in. Bead
/// `sts2-rs-zdjjx` (all-acts coverage sweep) — previously the scenario
/// builder consulted only `act1_overgrowth::find`, so every Act-2/Act-3
/// encounter id raised `unknown encounter id`.
#[must_use]
pub fn find(id: &str) -> Option<&'static EncounterDef> {
    act1_overgrowth::find(id)
        .or_else(|| act1_underdocks::find(id))
        .or_else(|| act2_hive::find(id))
        .or_else(|| act3_glory::find(id))
}

/// The Act-1 encounter table for `act1` (GH #544).
///
/// STS2 ships two acts at index 0, so "the act-1 encounters" is not a
/// constant — anything resolving an act-1 encounter, boss pin or display name
/// during a live run must go through here rather than naming
/// [`act1_overgrowth::ACT1_ENCOUNTERS`] directly. Reaching for the Overgrowth
/// table on an Underdocks run does not error; it silently resolves a boss the
/// run never fights (that was the bug: the map showed `Vantom` for a run whose
/// boss was `LAGAVULIN_MATRIARCH_BOSS`).
///
/// Act-2/3 variants exist in `ActVariant` but are not act-1 models; they fall
/// back to Overgrowth rather than panicking, matching `run_room_gen::acts_for`.
#[must_use]
pub fn act1_encounters(act1: ActVariant) -> &'static [EncounterDef] {
    match act1 {
        ActVariant::Underdocks => act1_underdocks::ACT1_UNDERDOCKS_ENCOUNTERS,
        _ => act1_overgrowth::ACT1_ENCOUNTERS,
    }
}

/// Size of the stable encounter-index space — the concatenation of
/// `ACT1_ENCOUNTERS ++ ACT2_ENCOUNTERS ++ ACT3_ENCOUNTERS` in table
/// order. This is the index space [`encounter_index`] maps into, and
/// the dimension the run-AI combat-surrogate's encounter embedding
/// (`enc_emb`) is sized to (AZ Milestone 1 Task 2: the realistic
/// combat-surrogate label harvester needs a stable, engine-independent
/// index so a real self-play pre-combat state can be tagged with which
/// encounter it's about to face).
#[must_use]
pub fn n_encounters() -> usize {
    act1_overgrowth::ACT1_ENCOUNTERS.len()
        + act2_hive::ACT2_ENCOUNTERS.len()
        + act3_glory::ACT3_ENCOUNTERS.len()
        + act1_underdocks::ACT1_UNDERDOCKS_ENCOUNTERS.len()
}

/// Stable index of `def` within the [`n_encounters`] index space:
/// act-offset (Act 1 starts at 0, Act 2 at `ACT1_ENCOUNTERS.len()`, Act 3
/// at `ACT1_ENCOUNTERS.len() + ACT2_ENCOUNTERS.len()`) plus the def's
/// position within its own act's table. Matches by `id` (globally unique
/// per [`find`]) rather than raw pointer identity so it works for any
/// `&EncounterDef` handle the caller holds, not only the exact `&'static`
/// reference stored in the table.
///
/// # Panics
///
/// Panics if `def.id` isn't found in any of the three act tables — every
/// live `EncounterDef` handed to this function is expected to have come
/// from one of those tables (via [`find`] or direct iteration).
#[must_use]
pub fn encounter_index(def: &EncounterDef) -> usize {
    if let Some(i) = act1_overgrowth::ACT1_ENCOUNTERS
        .iter()
        .position(|e| e.id == def.id)
    {
        return i;
    }
    let off1 = act1_overgrowth::ACT1_ENCOUNTERS.len();
    if let Some(i) = act2_hive::ACT2_ENCOUNTERS
        .iter()
        .position(|e| e.id == def.id)
    {
        return off1 + i;
    }
    let off2 = off1 + act2_hive::ACT2_ENCOUNTERS.len();
    if let Some(i) = act3_glory::ACT3_ENCOUNTERS
        .iter()
        .position(|e| e.id == def.id)
    {
        return off2 + i;
    }
    // Underdocks is appended AFTER act 3 rather than beside Overgrowth on
    // purpose: this index is baked into the combat surrogate's encounter
    // embedding, so inserting the new act in act order would renumber every
    // Act-2/Act-3 row and silently repoint learned weights (GH #544 — the
    // same failure the PPO vocab pin exists to prevent).
    let off3 = off2 + act3_glory::ACT3_ENCOUNTERS.len();
    if let Some(i) = act1_underdocks::ACT1_UNDERDOCKS_ENCOUNTERS
        .iter()
        .position(|e| e.id == def.id)
    {
        return off3 + i;
    }
    panic!("encounter_index: unknown encounter id {:?}", def.id);
}

/// The encounter at stable index `idx` of the [`n_encounters`] index space:
/// the inverse of [`encounter_index`]. `None` past the end. Lets a reader of
/// `RunState::current_encounter_idx` name the encounter (the e2e campaign's
/// `CombatState.encounter_id`, GH #1097).
#[must_use]
pub fn by_index(idx: usize) -> Option<&'static EncounterDef> {
    let tables: [&'static [EncounterDef]; 4] = [
        act1_overgrowth::ACT1_ENCOUNTERS,
        act2_hive::ACT2_ENCOUNTERS,
        act3_glory::ACT3_ENCOUNTERS,
        act1_underdocks::ACT1_UNDERDOCKS_ENCOUNTERS,
    ];
    let mut rest = idx;
    for table in tables {
        if rest < table.len() {
            return Some(&table[rest]);
        }
        rest -= table.len();
    }
    None
}

/// Encounter table row. Keyed by the Python pin's canonical encounter
/// id (`UPPER_SNAKE`). `display` is the human-readable label rendered by
/// `sts2-cli list encounters`.
#[derive(Debug, Clone, Copy)]
pub struct EncounterDef {
    /// Stable `UPPER_SNAKE` id sourced from the Python pin
    /// (`sts2_env/encounters/act1.py` setup-fn name minus the
    /// `setup_` prefix). Used by `--encounter <id>` resolution and as
    /// the parity-bridge encounter tag.
    pub id: &'static str,
    /// Human-readable display label for `sts2-cli list encounters`.
    pub display: &'static str,
    /// Encounter composition: monster kind plus spawn args. Inklet /
    /// `KinFollower` rows carry an explicit `MonsterSpawnArgs::Inklet`
    /// / `MonsterSpawnArgs::KinFollower` so the dispatch table picks
    /// the correct starting move without relying on positional
    /// fallback.
    pub monsters: &'static [(MonsterKind, MonsterSpawnArgs)],
    /// Encounter tier — used by random pickers
    /// (`random-act1-weak` / `random-act1-normal` / `…`) and by
    /// future map-progression beads.
    pub tier: EncounterTier,
    /// Optional spawn-time randomizer. When `Some`, callers SHOULD
    /// invoke [`EncounterDef::roster`] (or the randomizer directly) to
    /// pick the actual roster at combat start. When `None`,
    /// `monsters` is the canonical static roster. Bead
    /// `sts2-rs-6inu`. The rows that set this — `SLIMES_WEAK`,
    /// `RUBY_RAIDERS_NORMAL` — keep their canonical fixed permutation in
    /// `monsters` so callers without an RNG (parity recorders, `sts2-cli list
    /// encounters`) still see a deterministic composition. (`KNIGHTS_ELITE`
    /// was originally in this set but is a deterministic fixed-3 roster in C#;
    /// it now uses `randomize: None` — bead `sts2-rs-ybtmu`.)
    pub randomize: Option<EncounterRandomizer>,
}

impl EncounterDef {
    /// Return the encounter's monster roster. When [`Self::randomize`]
    /// is set, the randomizer is invoked with `rng`; otherwise the
    /// static [`Self::monsters`] slice is returned as-is. Bead
    /// `sts2-rs-6inu`.
    #[must_use]
    pub fn roster(&self, rng: &mut sts2_core::dotnet_rng::Rng) -> EncounterRoster {
        if let Some(randomizer) = self.randomize {
            randomizer(rng)
        } else {
            self.monsters.iter().copied().collect()
        }
    }

    /// Static / deterministic roster — the canonical permutation
    /// recorded in [`Self::monsters`], ignoring any randomizer. Used by
    /// no-RNG call sites (`sts2-cli list encounters`, parity fixture
    /// builders). Bead `sts2-rs-6inu`.
    #[must_use]
    pub fn canonical_roster(&self) -> EncounterRoster {
        self.monsters.iter().copied().collect()
    }

    /// The encounter's C# `EncounterModel.Slots` length, or `0` for the
    /// majority that declare none.
    ///
    /// Only the 19 scene-bearing encounters have slots. Slots decide two
    /// things: the enemy list stays sorted by slot index
    /// (`CombatState.SortEnemiesBySlotName`, re-run on every add), and a
    /// summon picks a FREE slot — so a mid-combat add lands in a specific
    /// place in turn order, not on the end. `enemy_uids` order is also the
    /// positional `target_index`, which is why getting it wrong is observable
    /// (GH #552).
    ///
    /// Id-keyed rather than a per-row field for the same reason
    /// [`Self::pressure_threshold`] is: it keeps the ~80 `EncounterDef`
    /// literals across the four act tables untouched.
    #[must_use]
    pub const fn slot_count(&self) -> u8 {
        match self.id.as_bytes() {
            b"AXEBOTS_NORMAL" => 1,
            b"FOGMOG_NORMAL"
            | b"KAISER_CRAB_BOSS"
            | b"MYTES_NORMAL"
            | b"NIBBITS_NORMAL"
            | b"QUEEN_BOSS"
            | b"THE_OBSCURA_NORMAL" => 2,
            b"DECIMILLIPEDE_ELITE" | b"EXOSKELETONS_WEAK" | b"THE_KIN_BOSS" => 3,
            b"EXOSKELETONS_NORMAL" | b"PHANTASMAL_GARDENERS_ELITE" => 4,
            b"FABRICATOR_NORMAL" | b"PHROG_PARASITE_ELITE" | b"TWO_TAILED_RATS_NORMAL" => 5,
            b"LIVING_FOG_NORMAL" | b"OVICOPTER_NORMAL" => 6,
            _ => 0,
        }
    }

    /// Which slot each spawned member occupies, positionally against the
    /// roster — C# `GenerateMonsters` returns `(model, slotName)` pairs and
    /// this is those names resolved to `Slots` indices.
    ///
    /// Every shipped roster is already slot-ASCENDING, so stamping these
    /// reorders nothing at combat start. They matter because a later summon's
    /// free-slot search reads them: the Two-Tailed Rats seed slots 2/3/4 and
    /// leave 0/1 free, and the Ovicopter/LivingFog sit in the LAST slot with
    /// their spawn slots below.
    #[must_use]
    pub const fn member_slots(&self) -> &'static [u8] {
        match self.id.as_bytes() {
            // The summoner occupies the tail slot; its spawns fill below it.
            b"LIVING_FOG_NORMAL" | b"OVICOPTER_NORMAL" => &[5],
            // `[bot1, bot2, fabricator, bot3, bot4]` — the bots land either
            // side of it, cheapest-index first.
            b"FABRICATOR_NORMAL" => &[2],
            // `[phrog, wriggler1..4]` — the summoner leads here.
            b"PHROG_PARASITE_ELITE" => &[0],
            // `[illusion, fogmog]` / `[illusion, obscura]` — the illusion is
            // summoned into slot 0, AHEAD of its summoner.
            b"FOGMOG_NORMAL" | b"THE_OBSCURA_NORMAL" => &[1],
            // `[first .. fifth]`, seeded at 2/3/4; CALL_FOR_BACKUP takes the
            // LAST free slot, so summons land at 1 then 0 — ahead of the
            // originals (GH #552).
            b"TWO_TAILED_RATS_NORMAL" => &[2, 3, 4],
            b"AXEBOTS_NORMAL" => &[0],
            // Rosters that simply fill their slots from 0 upwards.
            b"KAISER_CRAB_BOSS" | b"MYTES_NORMAL" | b"NIBBITS_NORMAL" | b"QUEEN_BOSS" => &[0, 1],
            b"DECIMILLIPEDE_ELITE" | b"EXOSKELETONS_WEAK" | b"THE_KIN_BOSS" => &[0, 1, 2],
            b"EXOSKELETONS_NORMAL" | b"PHANTASMAL_GARDENERS_ELITE" => &[0, 1, 2, 3],
            _ => &[],
        }
    }

    /// Round threshold past which the leaf evaluator's `time_pressure`
    /// term goes negative — see `sts2_core::eval_features` column
    /// `time_pressure`. Default `99` (= effectively no pressure); a
    /// small handful of phase-change / escalating bosses set lower
    /// thresholds so the search learns to prioritize finishes before
    /// the boss's bad phases trigger. Encoded as an id-keyed match
    /// rather than a per-row field so the ~60 `EncounterDef` literals
    /// across `act{1,2,3}.rs` don't all need to be touched when a
    /// new threshold is tuned.
    ///
    /// Eval consumes [`sts2_core::monster_pressure_threshold`] rather
    /// than this method (the leaf evaluator has only `CombatState`,
    /// not the encounter id); the two MUST stay in sync for any
    /// encounter listed here — a regression test in this crate asserts
    /// so for `VANTOM_BOSS`. Bead `sts2-rs-ot78`.
    #[must_use]
    pub const fn pressure_threshold(&self) -> u16 {
        match self.id.as_bytes() {
            b"VANTOM_BOSS" => 5,
            _ => 99,
        }
    }
}

/// Act 1 encounter tier per the Python pin's WEAK / NORMAL / ELITE /
/// BOSS partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncounterTier {
    Weak,
    Normal,
    Elite,
    Boss,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The unified [`find`] resolves ids from every act registry, not just
    /// Act 1. Bead `sts2-rs-zdjjx` (all-acts coverage sweep). Before this
    /// the PyO3 scenario builder only saw `act1_overgrowth::find`, so every
    /// Act-2/Act-3 encounter id raised `unknown encounter id`.
    #[test]
    fn find_resolves_across_all_acts() {
        assert_eq!(find("VANTOM_BOSS").map(|e| e.id), Some("VANTOM_BOSS")); // act1
        assert_eq!(
            find("CHOMPERS_NORMAL").map(|e| e.id),
            Some("CHOMPERS_NORMAL")
        ); // act2
        assert_eq!(find("QUEEN_BOSS").map(|e| e.id), Some("QUEEN_BOSS")); // act3
        assert!(find("DEFINITELY_NOT_AN_ENCOUNTER").is_none());
    }

    /// [`encounter_index`] agrees with an explicit act-offset walk of the
    /// three tables, and every index it produces is `< n_encounters()`.
    /// AZ Milestone 1 Task 2.
    #[test]
    fn encounter_index_matches_act_offset_walk() {
        let off1 = act1_overgrowth::ACT1_ENCOUNTERS.len();
        let off2 = off1 + act2_hive::ACT2_ENCOUNTERS.len();
        let off3 = off2 + act3_glory::ACT3_ENCOUNTERS.len();
        assert_eq!(
            n_encounters(),
            off3 + act1_underdocks::ACT1_UNDERDOCKS_ENCOUNTERS.len()
        );

        for (i, e) in act1_overgrowth::ACT1_ENCOUNTERS.iter().enumerate() {
            assert_eq!(encounter_index(e), i);
        }
        for (i, e) in act2_hive::ACT2_ENCOUNTERS.iter().enumerate() {
            assert_eq!(encounter_index(e), off1 + i);
        }
        for (i, e) in act3_glory::ACT3_ENCOUNTERS.iter().enumerate() {
            assert_eq!(encounter_index(e), off2 + i);
        }
        // Underdocks occupies the TAIL of the index space, not the slot its
        // act number would suggest. That is deliberate: this index is the
        // combat surrogate's encounter-embedding row, so inserting the new
        // act in act order would renumber every Act-2/Act-3 row (GH #544).
        for (i, e) in act1_underdocks::ACT1_UNDERDOCKS_ENCOUNTERS
            .iter()
            .enumerate()
        {
            assert_eq!(encounter_index(e), off3 + i);
        }
        for e in act1_overgrowth::ACT1_ENCOUNTERS
            .iter()
            .chain(act2_hive::ACT2_ENCOUNTERS)
            .chain(act3_glory::ACT3_ENCOUNTERS)
            .chain(act1_underdocks::ACT1_UNDERDOCKS_ENCOUNTERS)
        {
            assert!(encounter_index(e) < n_encounters());
        }
    }

    /// [`by_index`] inverts [`encounter_index`] over the whole index space and
    /// stops at its end.
    #[test]
    fn by_index_inverts_encounter_index() {
        for idx in 0..n_encounters() {
            let def = by_index(idx).expect("in range");
            assert_eq!(encounter_index(def), idx, "{}", def.id);
        }
        assert!(by_index(n_encounters()).is_none());
        assert_eq!(
            by_index(encounter_index(find("QUEEN_BOSS").unwrap())).map(|e| e.id),
            Some("QUEEN_BOSS")
        );
    }
}
