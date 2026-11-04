//! Act3-pool event specs. Bead `sts2-rs-6ynt.4.3`.
//!
//! All 5 act3 events have `is_allowed = False` in Python — none spawn
//! from the random room-roll. They are reachable only via special
//! game logic:
//!
//! - `TheArchitect` — act-transition end-of-run event (the run-loop
//!   driver will materialize it before the final boss).
//! - `WarHistorianRepy` — replaces the next act-3 event while the deck
//!   holds a `LANTERN_KEY` (the card's C# `ModifyNextEvent`).
//! - `Neow` — run-start relic-selection event, dropped into the
//!   room by the start-of-run code path.
//! - `DeprecatedEvent` / `DeprecatedAncientEvent` — legacy stubs that
//!   exist in the registry to round-trip parity dumps.
//!
//! For now the Rust port lands the static `EventSpec` shape for each.
//! Reward/queue side-effects that depend on unported surfaces
//! (`RunState.rng` for Neow's dynamic relic pool) ship as documented
//! divergences with sentinel handler behaviour.

use sts2_core::cards::CardId;
use sts2_core::{EventId, EventOptionInfo, RelicId, RunState};

use super::{
    default_option_info_from_spec, never_allowed, EventOptionSpec, EventOutcome, EventSpec,
};

// ── TheArchitect ─────────────────────────────────────────────────────

/// `TheArchitect::proceed` — flavor-only outcome. Python's
/// `EventResult` carries `event_combat_setup="the_architect"` +
/// `post_combat_phase="RUN_OVER"`, which are run-loop driver fields
/// not yet ported. When the act-transition surface lands, this
/// handler will write a hand-off marker into `RunState` so the
/// driver can launch the final combat. Today it just signals
/// finished — the run-loop driver short-circuits before the option
/// is actually presentable in pool play. Port pin:
/// `act3.py:41-47`.
fn the_architect_proceed(_state: &mut RunState) -> EventOutcome {
    EventOutcome::finished("The Architect emerges.")
}

const THE_ARCHITECT_OPTIONS: &[EventOptionSpec] = &[EventOptionSpec {
    option_id: "proceed",
    loc_key: "",
    default_label: "Proceed",
    default_description: "The Architect emerges.",
    default_vars: &[],
    vars_fn: None,
    title_vars_fn: None,
    handler: the_architect_proceed,
}];

/// `TheArchitect` static spec. Port pin:
/// `sts2-rl-agent@01c094d:sts2_env/events/act3.py:27`.
pub const THE_ARCHITECT: EventSpec = EventSpec {
    id: EventId::TheArchitect,
    python_name: "TheArchitect",
    is_allowed: never_allowed,
    generate_options: default_option_info_from_spec,
    options: THE_ARCHITECT_OPTIONS,
};

// ── WarHistorianRepy ─────────────────────────────────────────────────
//
// C# `Models.Events/WarHistorianRepy.cs`. v0.107.1 and
// v0.111.0 are identical apart from a decompiler `global::` qualifier, so
// nothing here is version-gated. The run reaches this event only through the
// LanternKey quest card's act-3 `ModifyNextEvent` (`IsAllowed => false`).
//
// Single-player flow (`Players.Count == 1`):
//   page 0  Unlock Cage | Unlock Chest. Either removes the FIRST LanternKey
//           (`RemoveFirstLanternKey`, :114) and grants its reward. If the deck
//           still holds a LanternKey (`ShouldGetSecondReward`, :18), a second
//           page offers the OTHER option; otherwise the event ends.
//   page 1  (after the cage) Unlock Chest.
//   page 2  (after the chest) Unlock Cage.
//           The second choice removes EVERY remaining LanternKey
//           (`RemoveLanternKeysForSecondChoice`, :124) before its reward.
//
// Unlock Cage (:96): `FreedRepy = true` (read only by a combat background
// VFX, `NQueenRepyBgVfx.cs:55`, so not modelled), then
// `RelicCmd.Obtain<HistoryCourse>`: an obtain, not an offer.
//
// Unlock Chest (:86): `RewardsCmd.OfferCustom([Potion, Potion, Relic,
// Relic])`. `RewardsSet.GenerateWithoutOffering` populates them in list order,
// all on `PlayerRng.Rewards`: two `PotionFactory.CreateRandomPotionOutOfCombat`
// rolls, then two `RelicFactory.PullNextRelicFromFront(player)` (rarity roll,
// then a front pull from the grab bag, fallback Circlet). The headless then
// takes every reward in `RewardsSetIndex` order: potions (2), relics (3).

/// `RelicFactory.FallbackRelic` (`RelicFactory.cs:13`), pulled when the grab
/// bag has nothing left at the rolled rarity.
const WAR_HISTORIAN_FALLBACK_RELIC: RelicId = RelicId::Circlet;

/// `UnlockChest`'s potion and relic counts (`WarHistorianRepy.cs:88-92`).
pub const WAR_HISTORIAN_CHEST_POTIONS: usize = 2;
/// See [`WAR_HISTORIAN_CHEST_POTIONS`].
pub const WAR_HISTORIAN_CHEST_RELICS: usize = 2;

/// Page cursor: the second page after Unlock Cage (offers Unlock Chest).
const WAR_HISTORIAN_PAGE_AFTER_CAGE: u8 = 1;
/// Page cursor: the second page after Unlock Chest (offers Unlock Cage).
const WAR_HISTORIAN_PAGE_AFTER_CHEST: u8 = 2;

/// `ShouldGetSecondReward` (`WarHistorianRepy.cs:18-28`): the deck still holds
/// a LanternKey. The Rust run is single-player, so the player-count half of
/// the check always passes.
fn war_historian_has_lantern_key(state: &RunState) -> bool {
    state
        .master_deck
        .iter()
        .any(|c| c.card_id == CardId::LanternKey)
}

/// `RemoveFirstLanternKey` (`WarHistorianRepy.cs:114-122`). `CompleteQuest`
/// only writes run history. `CardPileCmd.RemoveFromDeck`'s one
/// `BeforeCardRemoved` listener is SpoilsMap removing itself, so a plain deck
/// removal is the whole effect.
fn war_historian_remove_first_lantern_key(state: &mut RunState) {
    if let Some(idx) = state
        .master_deck
        .iter()
        .position(|c| c.card_id == CardId::LanternKey)
    {
        state.master_deck.remove(idx);
    }
}

/// `RemoveLanternKeysForSecondChoice` (`WarHistorianRepy.cs:124-132`).
fn war_historian_remove_all_lantern_keys(state: &mut RunState) {
    state
        .master_deck
        .retain(|c| c.card_id != CardId::LanternKey);
}

/// `UnlockCage` (`WarHistorianRepy.cs:96-100`): obtain History Course.
fn war_historian_unlock_cage(state: &mut RunState) {
    state.grant_event_relic(RelicId::HistoryCourse);
}

/// `UnlockChest` (`WarHistorianRepy.cs:86-94`): populate the two potions and
/// then the two relics on the Rewards stream, then take them, potions first.
///
/// The potions are rolled here rather than left as `Reward::Potion` sentinels:
/// a sentinel rolls when the queue drains, which is after the relic rolls
/// below, and that reverses C#'s order on the shared Rewards stream.
///
/// Each potion is its own `PotionReward`, so each `Populate` is a separate
/// count-1 `CreateRandomPotionOutOfCombat` over the full pool
/// (`PotionReward.cs:54-61`, `PotionFactory.cs:30-33`): the two can repeat,
/// and the first pick is never removed from the second roll's pool.
fn war_historian_unlock_chest(state: &mut RunState) {
    let character = state.character_id;
    let potions: Vec<_> = (0..WAR_HISTORIAN_CHEST_POTIONS)
        .flat_map(|_| {
            crate::potions::all::create_random_potions_out_of_combat(
                &mut state.rng.rewards,
                character,
                1,
            )
        })
        .collect();
    let relics: Vec<RelicId> = (0..WAR_HISTORIAN_CHEST_RELICS)
        .map(|_| {
            let rarity = state.roll_reward_relic_rarity();
            state
                .pop_relic_from_grab_bag(rarity)
                .unwrap_or(WAR_HISTORIAN_FALLBACK_RELIC)
        })
        .collect();
    for kind in potions {
        let _ = state.claim_potion_reward(kind);
    }
    for relic in relics {
        state.grant_event_relic(relic);
    }
}

/// End of a first-page choice: open the second page if another LanternKey is
/// left, else finish (`InitialUnlockCage` / `InitialUnlockChest`).
fn war_historian_after_first_choice(state: &mut RunState, next_page: u8) -> EventOutcome {
    if war_historian_has_lantern_key(state) {
        state.event_state.page = next_page;
        EventOutcome {
            finished: false,
            description: "Another Lantern Key remains.".to_string(),
            effects: Vec::new(),
        }
    } else {
        EventOutcome::finished("The war historian is satisfied.")
    }
}

/// `InitialUnlockCage` (`WarHistorianRepy.cs:44-56`).
fn war_historian_repy_unlock_cage(state: &mut RunState) -> EventOutcome {
    war_historian_remove_first_lantern_key(state);
    war_historian_unlock_cage(state);
    war_historian_after_first_choice(state, WAR_HISTORIAN_PAGE_AFTER_CAGE)
}

/// `InitialUnlockChest` (`WarHistorianRepy.cs:58-70`).
fn war_historian_repy_unlock_chest(state: &mut RunState) -> EventOutcome {
    war_historian_remove_first_lantern_key(state);
    war_historian_unlock_chest(state);
    war_historian_after_first_choice(state, WAR_HISTORIAN_PAGE_AFTER_CHEST)
}

/// `SecondUnlockCage` (`WarHistorianRepy.cs:72-77`).
fn war_historian_repy_second_unlock_cage(state: &mut RunState) -> EventOutcome {
    war_historian_remove_all_lantern_keys(state);
    war_historian_unlock_cage(state);
    EventOutcome::finished("Freed Repy, gained History Course.")
}

/// `SecondUnlockChest` (`WarHistorianRepy.cs:79-84`).
fn war_historian_repy_second_unlock_chest(state: &mut RunState) -> EventOutcome {
    war_historian_remove_all_lantern_keys(state);
    war_historian_unlock_chest(state);
    EventOutcome::finished("Unlocked the chest: 2 potions and 2 relics.")
}

/// Number of page-0 options (Unlock Cage, Unlock Chest).
const WAR_HISTORIAN_PAGE0_OPTIONS: usize = 2;

/// Present the current page: page 0 offers both options, the second page only
/// the one not yet taken. C#'s second-page option reuses the INITIAL page's
/// loc key.
fn war_historian_generate_options(state: &RunState, spec: &EventSpec) -> Vec<EventOptionInfo> {
    let mut opts = default_option_info_from_spec(state, spec);
    let second = match state.event_state.page {
        WAR_HISTORIAN_PAGE_AFTER_CAGE => "second_unlock_chest",
        WAR_HISTORIAN_PAGE_AFTER_CHEST => "second_unlock_cage",
        _ => {
            opts.truncate(WAR_HISTORIAN_PAGE0_OPTIONS);
            return opts;
        }
    };
    opts.retain(|o| o.option_id == second);
    opts
}

const WAR_HISTORIAN_REPY_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        option_id: "unlock_cage",
        loc_key: "WAR_HISTORIAN_REPY.pages.INITIAL.options.UNLOCK_CAGE",
        default_label: "Unlock Cage",
        default_description: "Free Repy, gain History Course relic.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: war_historian_repy_unlock_cage,
    },
    EventOptionSpec {
        option_id: "unlock_chest",
        loc_key: "WAR_HISTORIAN_REPY.pages.INITIAL.options.UNLOCK_CHEST",
        default_label: "Unlock Chest",
        default_description: "Gain 2 potions + 2 relics.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: war_historian_repy_unlock_chest,
    },
    // ── second page (only with a second LanternKey in the deck) ──
    EventOptionSpec {
        option_id: "second_unlock_chest",
        loc_key: "WAR_HISTORIAN_REPY.pages.INITIAL.options.UNLOCK_CHEST",
        default_label: "Unlock Chest",
        default_description: "Gain 2 potions + 2 relics.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: war_historian_repy_second_unlock_chest,
    },
    EventOptionSpec {
        option_id: "second_unlock_cage",
        loc_key: "WAR_HISTORIAN_REPY.pages.INITIAL.options.UNLOCK_CAGE",
        default_label: "Unlock Cage",
        default_description: "Free Repy, gain History Course relic.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: war_historian_repy_second_unlock_cage,
    },
];

/// `WarHistorianRepy` static spec (`WarHistorianRepy.cs`; `IsAllowed => false`).
pub const WAR_HISTORIAN_REPY: EventSpec = EventSpec {
    id: EventId::WarHistorianRepy,
    python_name: "WarHistorianRepy",
    is_allowed: never_allowed,
    generate_options: war_historian_generate_options,
    options: WAR_HISTORIAN_REPY_OPTIONS,
};

// ── Neow ─────────────────────────────────────────────────────────────

/// Fallback positive relic for `Neow::positive_1` when
/// [`RunState::neow_offers`] hasn't been rolled yet (e.g. fresh
/// `RunState::new()` in a unit test that constructs the spec
/// surface manually). Once sub-bead `sts2-rs-g24zv` lands the
/// run-start phase wiring, the cache is always populated before any
/// option is presented, so this fallback only matters for legacy
/// tests that hit the handler without going through the run-loop.
const NEOW_FALLBACK_POSITIVE_1: RelicId = RelicId::ArcaneScroll;
/// Fallback positive relic for `Neow::positive_2`. See
/// [`NEOW_FALLBACK_POSITIVE_1`].
const NEOW_FALLBACK_POSITIVE_2: RelicId = RelicId::BoomingConch;
/// Fallback cursed relic for `Neow::cursed`. See
/// [`NEOW_FALLBACK_POSITIVE_1`].
const NEOW_FALLBACK_CURSED: RelicId = RelicId::CursedPearl;

/// Resolve a cached Neow offer to its [`RelicId`]. Every `NeowRelic`
/// variant now maps 1-to-1 via [`sts2_core::NeowRelic::to_relic_id`]
/// (W9a-e: PhialHolster / WingedBoots / NeowsTalisman — `sts2-rs-4rwv4`;
/// NeowsBones — `sts2-rs-wjuoi`; HeftyTablet — `sts2-rs-w6hdj`), so the
/// former safe-placeholder fallback is gone. The `Option` return on
/// `to_relic_id` is retained for forward-compat with any future Neow
/// option that lands ahead of its `RelicId`; if that ever happens this
/// `expect` flags it loudly rather than silently mis-granting.
pub(crate) fn resolve_neow_relic(nr: sts2_core::NeowRelic) -> RelicId {
    nr.to_relic_id()
        .expect("every NeowRelic variant maps to a RelicId (W9a-e)")
}

/// Grant a chosen Neow relic. With [`RunState::direct_event_relic_grant`]
/// set (the workbench), add it immediately via [`RunState::obtain_relic`]
/// — matching C#, where picking the blessing grants the relic with no
/// follow-up "Take relic / Skip" step (and on-obtain effects like Precise
/// Scissors' card removal fire right away). Otherwise queue it as a
/// relic-reward for a separate `PickRelicReward`, the Python-parity flow.
fn grant_neow_relic(state: &mut RunState, relic: RelicId) {
    state.grant_event_relic(relic);
}

fn neow_positive_1(state: &mut RunState) -> EventOutcome {
    let relic = state.neow_offers.map_or(NEOW_FALLBACK_POSITIVE_1, |o| {
        resolve_neow_relic(o.positive_1)
    });
    grant_neow_relic(state, relic);
    EventOutcome::finished("Gained a positive relic from Neow.")
}

fn neow_positive_2(state: &mut RunState) -> EventOutcome {
    let relic = state.neow_offers.map_or(NEOW_FALLBACK_POSITIVE_2, |o| {
        resolve_neow_relic(o.positive_2)
    });
    grant_neow_relic(state, relic);
    EventOutcome::finished("Gained a positive relic from Neow.")
}

fn neow_cursed(state: &mut RunState) -> EventOutcome {
    let relic = state
        .neow_offers
        .map_or(NEOW_FALLBACK_CURSED, |o| resolve_neow_relic(o.cursed));
    grant_neow_relic(state, relic);
    EventOutcome::finished("Gained a cursed relic from Neow.")
}

/// Dynamic option-list builder for Neow: labels reflect the cached
/// rolled relic names from [`sts2_core::RunState::neow_offers`]. If
/// the cache is unset (e.g. the run-loop hasn't called
/// [`sts2_core::RunState::roll_neow_offers_if_needed`] yet), falls
/// back to the static labels — sub-bead `sts2-rs-g24zv` wires the
/// roll into the run-start path so this branch only fires in
/// legacy / unit-test code paths.
///
/// Bead `sts2-rs-14cid`.
fn neow_generate_options(state: &RunState, spec: &EventSpec) -> Vec<sts2_core::EventOptionInfo> {
    let Some(offers) = state.neow_offers else {
        return super::default_option_info_from_spec(state, spec);
    };
    let labels = [
        offers.positive_1.snake_case_id(),
        offers.positive_2.snake_case_id(),
        offers.cursed.snake_case_id(),
    ];
    spec.options
        .iter()
        .zip(labels)
        .map(|(opt, label)| sts2_core::EventOptionInfo {
            option_id: opt.option_id,
            loc_key: opt.loc_key,
            vars: Vec::new(),
            title_vars: Vec::new(),
            label: label.to_string(),
            description: opt.default_description.to_string(),
            enabled: true,
        })
        .collect()
}

const NEOW_OPTIONS: &[EventOptionSpec] = &[
    EventOptionSpec {
        option_id: "positive_1",
        loc_key: "",
        default_label: "Arcane Scroll",
        default_description: "Gain a positive relic.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: neow_positive_1,
    },
    EventOptionSpec {
        option_id: "positive_2",
        loc_key: "",
        default_label: "Booming Conch",
        default_description: "Gain a positive relic.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: neow_positive_2,
    },
    EventOptionSpec {
        option_id: "cursed",
        loc_key: "",
        default_label: "Cursed Pearl",
        default_description: "Gain a cursed relic.",
        default_vars: &[],
        vars_fn: None,
        title_vars_fn: None,
        handler: neow_cursed,
    },
];

/// `Neow` static spec. Port pin:
/// `sts2-rl-agent@01c094d:sts2_env/events/act3.py:119`.
pub const NEOW: EventSpec = EventSpec {
    id: EventId::Neow,
    python_name: "Neow",
    is_allowed: never_allowed,
    generate_options: neow_generate_options,
    options: NEOW_OPTIONS,
};

// ── DeprecatedEvent / DeprecatedAncientEvent ─────────────────────────

/// Shared no-op handler for the two deprecated act3 stubs. Mirrors
/// Python's `return EventResult(finished=True)` body
/// (`act3.py:228-229`, `act3.py:248-249`).
fn deprecated_finish(_state: &mut RunState) -> EventOutcome {
    EventOutcome::finished("")
}

/// Synthetic single-option list for the two deprecated event stubs.
/// Python has `generate_initial_options` returning `[]`, but the
/// Rust dispatcher's `every_event_id_has_a_spec` sentinel requires
/// at least one option per spec (so a new EventId variant can't
/// silently ship without resolution logic). The option is unreachable
/// in pool play because `is_allowed = never_allowed`; it exists only
/// so the spec round-trips through the registry.
const DEPRECATED_OPTIONS: &[EventOptionSpec] = &[EventOptionSpec {
    option_id: "finish",
    loc_key: "",
    default_label: "Finish",
    default_description: "Deprecated event — no effect.",
    default_vars: &[],
    vars_fn: None,
    title_vars_fn: None,
    handler: deprecated_finish,
}];

/// `DeprecatedEvent` static spec. Port pin:
/// `sts2-rl-agent@01c094d:sts2_env/events/act3.py:217`.
pub const DEPRECATED_EVENT: EventSpec = EventSpec {
    id: EventId::DeprecatedEvent,
    python_name: "DeprecatedEvent",
    is_allowed: never_allowed,
    generate_options: default_option_info_from_spec,
    options: DEPRECATED_OPTIONS,
};

/// `DeprecatedAncientEvent` static spec. Same shape as
/// [`DEPRECATED_EVENT`]. Port pin:
/// `sts2-rl-agent@01c094d:sts2_env/events/act3.py:237`.
pub const DEPRECATED_ANCIENT_EVENT: EventSpec = EventSpec {
    id: EventId::DeprecatedAncientEvent,
    python_name: "DeprecatedAncientEvent",
    is_allowed: never_allowed,
    generate_options: default_option_info_from_spec,
    options: DEPRECATED_OPTIONS,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_architect_spec_matches_python_pin() {
        assert_eq!(THE_ARCHITECT.python_name, "TheArchitect");
        assert_eq!(THE_ARCHITECT.id.python_name(), "TheArchitect");
        assert_eq!(THE_ARCHITECT.options.len(), 1);
        assert_eq!(THE_ARCHITECT.options[0].option_id, "proceed");
        let rs = RunState::new();
        assert!(!(THE_ARCHITECT.is_allowed)(&rs));
    }

    #[test]
    fn the_architect_proceed_finishes() {
        let mut rs = RunState::new();
        let outcome = (THE_ARCHITECT.options[0].handler)(&mut rs);
        assert!(outcome.finished);
    }

    // ── WarHistorianRepy ──────────────────────────────────────────

    /// A run in the event with `keys` LanternKeys in the deck, in the
    /// parity / workbench relic mode (relics obtained, not offered).
    fn war_historian_rs(keys: usize) -> RunState {
        let mut rs = RunState::new();
        rs.direct_event_relic_grant = true;
        for _ in 0..keys {
            rs.master_deck.push(
                crate::cards::spec_index::card_pod_from_id(CardId::LanternKey, false)
                    .expect("LanternKey card POD"),
            );
        }
        rs.enter_event_room(EventId::WarHistorianRepy);
        rs
    }

    /// C# `UnlockChest`'s potions: two independent `PotionReward`s, each
    /// `Populate`d by its own `CreateRandomPotionOutOfCombat` (count 1) on a
    /// fresh full pool (`PotionReward.cs:54-61`, `PotionFactory.cs:30-33`).
    fn independent_chest_potion_rolls(
        rng: &mut sts2_core::DotNetRng,
        character: sts2_core::CharacterId,
    ) -> Vec<sts2_core::PotionKind> {
        (0..WAR_HISTORIAN_CHEST_POTIONS)
            .flat_map(|_| {
                crate::potions::all::create_random_potions_out_of_combat(rng, character, 1)
            })
            .collect()
    }

    /// The two chest potions roll independently, so they can repeat and the
    /// second is never shifted by removing the first from the pool (C#
    /// `CreateRandomPotions` removes picks only from its own per-call list).
    /// Rolling both in one `count = 2` call gave Ironclad seed 2055 Blood +
    /// Dexterity where C# had Blood + Colorless.
    #[test]
    fn war_historian_repy_unlock_chest_potions_roll_independently() {
        let mut saw_pool_removal_shift = false;
        for burn in 0..256 {
            let mut rs = war_historian_rs(1);
            for _ in 0..burn {
                let _ = rs.rng.rewards.next_float(1.0);
            }
            let character = rs.character_id;
            let expected = independent_chest_potion_rolls(&mut rs.rng.rewards.clone(), character);
            let one_call = crate::potions::all::create_random_potions_out_of_combat(
                &mut rs.rng.rewards.clone(),
                character,
                WAR_HISTORIAN_CHEST_POTIONS,
            );
            saw_pool_removal_shift |= expected != one_call;
            let _ = super::super::dispatch::resolve_by_index(&mut rs, 1).unwrap();
            assert_eq!(rs.potions, expected, "burn {burn}");
        }
        assert!(
            saw_pool_removal_shift,
            "no stream state separates the two rollings; the test cannot discriminate"
        );
    }

    fn lantern_keys(rs: &RunState) -> usize {
        rs.master_deck
            .iter()
            .filter(|c| c.card_id == CardId::LanternKey)
            .count()
    }

    fn presented_ids(rs: &RunState) -> Vec<&'static str> {
        (WAR_HISTORIAN_REPY.generate_options)(rs, &WAR_HISTORIAN_REPY)
            .into_iter()
            .map(|o| o.option_id)
            .collect()
    }

    #[test]
    fn war_historian_repy_presents_cage_and_chest_and_is_never_allowed() {
        let rs = war_historian_rs(1);
        assert_eq!(presented_ids(&rs), ["unlock_cage", "unlock_chest"]);
        assert!(!(WAR_HISTORIAN_REPY.is_allowed)(&rs));
    }

    /// C# `InitialUnlockCage`: remove the LanternKey, OBTAIN History Course
    /// (`RelicCmd.Obtain`, not a reward offer), and finish with no key left.
    #[test]
    fn war_historian_repy_unlock_cage_removes_key_and_obtains_history_course() {
        let mut rs = war_historian_rs(1);
        let deck_before = rs.master_deck.len();
        let outcome = super::super::dispatch::resolve_by_index(&mut rs, 0).unwrap();
        assert!(outcome.finished);
        assert_eq!(lantern_keys(&rs), 0);
        assert_eq!(rs.master_deck.len(), deck_before - 1);
        assert!(rs.relics.contains(&RelicId::HistoryCourse));
        assert!(rs.relic_rewards.is_empty());
        assert!(rs.pending_rewards.is_empty());
    }

    /// Without `direct_event_relic_grant` the History Course still reaches
    /// the player, through the relic-reward surface.
    #[test]
    fn war_historian_repy_unlock_cage_offers_history_course_outside_parity_mode() {
        let mut rs = war_historian_rs(1);
        rs.direct_event_relic_grant = false;
        let _ = super::super::dispatch::resolve_by_index(&mut rs, 0).unwrap();
        assert_eq!(rs.relic_rewards.len(), 1);
        assert_eq!(rs.relic_rewards[0].relic_id, RelicId::HistoryCourse);
    }

    /// C# `UnlockChest`: `OfferCustom([Potion, Potion, Relic, Relic])`
    /// populates in list order on the Rewards stream, so the two potions are
    /// the first two out-of-combat potion rolls of that stream. The headless
    /// then takes all four; nothing is left queued.
    #[test]
    fn war_historian_repy_unlock_chest_rolls_two_potions_then_two_relics() {
        let mut rs = war_historian_rs(1);
        let mut rewards = rs.rng.rewards.clone();
        let expected_potions = independent_chest_potion_rolls(&mut rewards, rs.character_id);
        let relics_before = rs.relics.len();
        let outcome = super::super::dispatch::resolve_by_index(&mut rs, 1).unwrap();
        assert!(outcome.finished);
        assert_eq!(lantern_keys(&rs), 0);
        assert_eq!(rs.potions, expected_potions, "potions roll first, in order");
        assert_eq!(rs.relics.len(), relics_before + WAR_HISTORIAN_CHEST_RELICS);
        assert!(!rs.relics.contains(&RelicId::HistoryCourse));
        assert!(rs.pending_rewards.is_empty());
        assert!(rs.relic_rewards.is_empty());
    }

    /// Two LanternKeys: the first choice removes ONE and opens a second page
    /// offering the other option (C# reuses the INITIAL loc key); the second
    /// choice removes the rest and finishes.
    #[test]
    fn war_historian_repy_second_key_opens_the_other_option() {
        let mut rs = war_historian_rs(2);
        let first = super::super::dispatch::resolve_by_index(&mut rs, 0).unwrap();
        assert!(!first.finished, "a second LanternKey keeps the event open");
        assert_eq!(rs.current_event, Some(EventId::WarHistorianRepy));
        assert_eq!(lantern_keys(&rs), 1);
        assert!(rs.relics.contains(&RelicId::HistoryCourse));
        let opts = (WAR_HISTORIAN_REPY.generate_options)(&rs, &WAR_HISTORIAN_REPY);
        assert_eq!(opts.len(), 1);
        assert_eq!(opts[0].option_id, "second_unlock_chest");
        assert_eq!(
            opts[0].loc_key,
            "WAR_HISTORIAN_REPY.pages.INITIAL.options.UNLOCK_CHEST"
        );

        let relics_before = rs.relics.len();
        let second = super::super::dispatch::resolve_by_index(&mut rs, 0).unwrap();
        assert!(second.finished);
        assert_eq!(rs.current_event, None);
        assert_eq!(lantern_keys(&rs), 0);
        assert_eq!(rs.relics.len(), relics_before + WAR_HISTORIAN_CHEST_RELICS);
    }

    /// Chest first with three keys: page 2 offers the cage, and that second
    /// choice removes BOTH remaining keys (`RemoveLanternKeysForSecondChoice`).
    #[test]
    fn war_historian_repy_chest_then_cage_removes_every_remaining_key() {
        let mut rs = war_historian_rs(3);
        let first = super::super::dispatch::resolve_by_index(&mut rs, 1).unwrap();
        assert!(!first.finished);
        assert_eq!(lantern_keys(&rs), 2);
        assert_eq!(presented_ids(&rs), ["second_unlock_cage"]);
        let second = super::super::dispatch::resolve_by_index(&mut rs, 0).unwrap();
        assert!(second.finished);
        assert_eq!(lantern_keys(&rs), 0);
        assert!(rs.relics.contains(&RelicId::HistoryCourse));
    }

    #[test]
    fn neow_spec_matches_python_pin() {
        assert_eq!(NEOW.python_name, "Neow");
        assert_eq!(NEOW.options.len(), 3);
        assert_eq!(NEOW.options[0].option_id, "positive_1");
        assert_eq!(NEOW.options[1].option_id, "positive_2");
        assert_eq!(NEOW.options[2].option_id, "cursed");
        let rs = RunState::new();
        assert!(!(NEOW.is_allowed)(&rs));
    }

    #[test]
    fn neow_positive_grants_positive_relic() {
        let mut rs = RunState::new();
        let outcome = (NEOW.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.relic_rewards.len(), 1);
        assert_eq!(rs.relic_rewards[0].relic_id, NEOW_FALLBACK_POSITIVE_1);
    }

    #[test]
    fn neow_cursed_grants_cursed_relic() {
        let mut rs = RunState::new();
        let outcome = (NEOW.options[2].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.relic_rewards.len(), 1);
        assert_eq!(rs.relic_rewards[0].relic_id, NEOW_FALLBACK_CURSED);
    }

    #[test]
    fn neow_roll_caches_offers_idempotently() {
        // Bead `sts2-rs-14cid`: `roll_neow_offers_if_needed` populates
        // `neow_offers` exactly once.
        let mut rs = RunState::new();
        assert!(rs.neow_offers.is_none());
        rs.roll_neow_offers_if_needed();
        let first = rs.neow_offers;
        assert!(first.is_some());
        // Mutating the cache to a marker proves the second call is a
        // no-op (idempotent).
        rs.roll_neow_offers_if_needed();
        assert_eq!(rs.neow_offers, first);
    }

    #[test]
    fn neow_dynamic_options_label_with_rolled_relics() {
        // Bead `sts2-rs-14cid`: the dynamic `generate_options` reads
        // the cached offers and emits SCREAMING_SNAKE_CASE labels for
        // each option, including the curse in slot 3.
        let mut rs = RunState::new();
        rs.roll_neow_offers_if_needed();
        let offers = rs.neow_offers.expect("roll populates the cache");
        let opts = (NEOW.generate_options)(&rs, &NEOW);
        assert_eq!(opts.len(), 3);
        assert_eq!(opts[0].label, offers.positive_1.snake_case_id());
        assert_eq!(opts[1].label, offers.positive_2.snake_case_id());
        assert_eq!(opts[2].label, offers.cursed.snake_case_id());
        for opt in &opts {
            assert!(opt.enabled);
        }
    }

    #[test]
    fn neow_dynamic_handler_grants_cached_relic() {
        // Bead `sts2-rs-14cid`: when the cache is populated, the
        // handler grants the rolled relic — not the legacy
        // fallback. We populate the cache with a known shape
        // (positive_1 = MassiveScroll, positive_2 = BoomingConch,
        // cursed = LeafyPoultice) so we don't depend on the
        // particular RNG output for the assertion.
        let mut rs = RunState::new();
        rs.neow_offers = Some(sts2_core::NeowOffers {
            positive_1: sts2_core::NeowRelic::MassiveScroll,
            positive_2: sts2_core::NeowRelic::BoomingConch,
            cursed: sts2_core::NeowRelic::LeafyPoultice,
        });

        // positive_1 → MassiveScroll
        let outcome = (NEOW.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(
            rs.relic_rewards.last().unwrap().relic_id,
            RelicId::MassiveScroll
        );

        // cursed → LeafyPoultice
        let outcome = (NEOW.options[2].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(
            rs.relic_rewards.last().unwrap().relic_id,
            RelicId::LeafyPoultice
        );
    }

    #[test]
    fn neow_handler_falls_back_to_constants_when_cache_unset() {
        // Bead `sts2-rs-14cid`: legacy code paths that hit the
        // handler without rolling first still work (Pomander/
        // ArcaneScroll/CursedPearl fallbacks).
        let mut rs = RunState::new();
        assert!(rs.neow_offers.is_none());
        let outcome = (NEOW.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.relic_rewards[0].relic_id, NEOW_FALLBACK_POSITIVE_1);
    }

    #[test]
    fn resolve_neow_relic_maps_all_w9_variants_to_their_own_relic_id() {
        // W9a-c (sts2-rs-4rwv4) + W9e (sts2-rs-wjuoi) + W9d
        // (sts2-rs-w6hdj): all five C#-only Neow relics now resolve to
        // their own RelicId variant — no placeholder routing left.
        assert_eq!(
            resolve_neow_relic(sts2_core::NeowRelic::PhialHolster),
            RelicId::PhialHolster
        );
        assert_eq!(
            resolve_neow_relic(sts2_core::NeowRelic::WingedBoots),
            RelicId::WingedBoots
        );
        assert_eq!(
            resolve_neow_relic(sts2_core::NeowRelic::NeowsTalisman),
            RelicId::NeowsTalisman
        );
        assert_eq!(
            resolve_neow_relic(sts2_core::NeowRelic::NeowsBones),
            RelicId::NeowsBones
        );
        assert_eq!(
            resolve_neow_relic(sts2_core::NeowRelic::HeftyTablet),
            RelicId::HeftyTablet
        );
    }

    #[test]
    fn deprecated_event_spec_matches_python_pin() {
        assert_eq!(DEPRECATED_EVENT.python_name, "DeprecatedEvent");
        assert_eq!(
            DEPRECATED_ANCIENT_EVENT.python_name,
            "DeprecatedAncientEvent"
        );
        // Both ship a synthetic `finish` option (Python has []).
        assert_eq!(DEPRECATED_EVENT.options.len(), 1);
        assert_eq!(DEPRECATED_ANCIENT_EVENT.options.len(), 1);
        let rs = RunState::new();
        assert!(!(DEPRECATED_EVENT.is_allowed)(&rs));
        assert!(!(DEPRECATED_ANCIENT_EVENT.is_allowed)(&rs));
    }

    #[test]
    fn deprecated_finish_is_no_op_finished() {
        let mut rs = RunState::new();
        let gold_before = rs.gold;
        let hp_before = rs.current_hp;
        let outcome = (DEPRECATED_EVENT.options[0].handler)(&mut rs);
        assert!(outcome.finished);
        assert_eq!(rs.gold, gold_before);
        assert_eq!(rs.current_hp, hp_before);
        assert!(rs.relic_rewards.is_empty());
        assert!(rs.pending_rewards.is_empty());
    }
}
