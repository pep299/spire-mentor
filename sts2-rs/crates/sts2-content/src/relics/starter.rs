//! Canonical non-Ironclad starter relics — bead `sts2-rs-evs` (P7-2).
//!
//! Ports the four canonical non-Ironclad starter-relic factories from
//! `sts2-rl-agent@01c094d:sts2_env/relics/starter.py`:
//!
//! - [`cracked_core`] — Defect starter (Channel 1 Lightning orb on
//!   round 1).
//! - [`ring_of_the_snake`] — Silent starter (+2 cards drawn on
//!   round 1).
//! - [`divine_right`] — Regent starter (Gain 3 Stars on round 1). The
//!   modded fork's third starter; replaces vanilla STS's `PureWater`
//!   (which does **not** exist in `sts2-rl-agent@01c094d`).
//! - [`bound_phylactery`] — Necrobinder starter (summon 1 Osty at
//!   combat start + every post-round-1 energy reset). Plumbing-only:
//!   the `RelicSpec` lands with `sts2-rs-qeo` (umbrella `sts2-rs-bay`);
//!   the dispatcher handler arrives with the Necrobinder character
//!   sub-bead.
//!
//! Burning Blood (Ironclad starter) is shipped separately by bead
//! `sts2-rs-u4u` in [`crate::relics::ironclad`].
//!
//! Dispatch handlers for these three live in
//! `sts2-core::relic_effects::install_default_registry` (already
//! shipped by P7-1 / `sts2-rs-l7l`). This module therefore provides
//! only the *static* `RelicSpec` data and a no-op `install_pool` that
//! delegates to the core installer.

use sts2_core::relic_effects;
use sts2_core::RelicKind;

use crate::relics::{ChannelOrbKind, RelicEffect, RelicRarity, RelicSpec, SideTurnPhase};

/// Cracked Core — Defect starter. At the start of round 1, channel one
/// Lightning orb. Mirrors `sts2_env/relics/starter.py::CrackedCore`
/// (`LIGHTNING_COUNT = 1`). The orb plumbing lands with bead
/// `sts2-rs-spy` (P7-7); the on-trigger handler currently exists as a
/// `Resolved` marker in `sts2-core::relic_effects`.
#[must_use]
pub const fn cracked_core() -> RelicSpec {
    RelicSpec {
        id: RelicKind::CrackedCore,
        name: "CRACKED_CORE",
        rarity: RelicRarity::Starter,
        effect: RelicEffect::ChannelOrbAtSideTurnStart {
            orb_kind: ChannelOrbKind::Lightning,
            count: CRACKED_CORE_LIGHTNING,
            phase: SideTurnPhase::Before,
            max_round: 1,
        },
    }
}

/// Ring of the Snake — Silent starter. On round 1, draw 2 additional
/// cards. Mirrors `sts2_env/relics/starter.py::RingOfTheSnake`
/// (`EXTRA_CARDS = 2`). The hand-draw modifier lands when the P7-8
/// integration bead wires `modify_hand_draw` against the relic
/// registry.
#[must_use]
pub const fn ring_of_the_snake() -> RelicSpec {
    RelicSpec {
        id: RelicKind::RingOfTheSnake,
        name: "RING_OF_THE_SNAKE",
        rarity: RelicRarity::Starter,
        effect: RelicEffect::ExtraHandDrawForRounds {
            extra_cards: RING_OF_THE_SNAKE_EXTRA_CARDS,
            max_round: 1,
        },
    }
}

/// Divine Right — Regent starter (modded fork's third canonical
/// starter; vanilla STS's `PureWater` is not present in
/// `sts2-rl-agent@01c094d`). On round 1, gain 3 Stars. Mirrors
/// `sts2_env/relics/starter.py::DivineRight` (STARS = 3). The
/// `gain_stars` runtime plumbing lands with the Regent character bead.
#[must_use]
pub const fn divine_right() -> RelicSpec {
    RelicSpec {
        id: RelicKind::DivineRight,
        name: "DIVINE_RIGHT",
        rarity: RelicRarity::Starter,
        effect: RelicEffect::GainStarsAtSideTurnStart {
            amount: DIVINE_RIGHT_STARS,
            max_round: 1,
        },
    }
}

/// `CrackedCore.LIGHTNING_COUNT = 1` (`starter.py`).
pub const CRACKED_CORE_LIGHTNING: i32 = 1;
/// `RingOfTheSnake.EXTRA_CARDS = 2` (`starter.py`).
pub const RING_OF_THE_SNAKE_EXTRA_CARDS: i32 = 2;
/// `DivineRight.STARS = 3` (`starter.py`).
pub const DIVINE_RIGHT_STARS: i32 = 3;

/// Bound Phylactery — Necrobinder starter. Summons 1 Osty at combat
/// start and on every energy-reset after round 1. Mirrors
/// `sts2_env/relics/starter.py::BoundPhylactery` (`SUMMON_COUNT = 1`).
/// Plumbing bead `sts2-rs-qeo` (umbrella `sts2-rs-bay`) lays the
/// `RelicSpec` here; the `summon_osty` runtime + the `Resolved`
/// marker in `sts2-core::relic_effects` land with the Necrobinder
/// character sub-bead. Today the dispatcher returns
/// `EffectResult::NotImplemented` for `RelicId::BoundPhylactery`
/// — exactly the pre-handler contract used by Burning Blood prior to
/// P7-2.
#[must_use]
pub const fn bound_phylactery() -> RelicSpec {
    // The `RelicSpec.effect` slot carries the relic's *primary*
    // on-trigger shape (`before_combat_start`). BoundPhylactery's
    // secondary listener (`after_energy_reset` when `round_number > 1`)
    // is registered directly against the `after_energy_reset` channel
    // in `sts2_core::relic_effects` so the AST stays single-effect-
    // per-spec for now. The full split lands when the Necrobinder
    // character bead ports the `summon_osty` runtime.
    RelicSpec {
        id: RelicKind::BoundPhylactery,
        name: "BOUND_PHYLACTERY",
        rarity: RelicRarity::Starter,
        effect: RelicEffect::SummonAtCombatStart {
            count: BOUND_PHYLACTERY_SUMMON_COUNT,
        },
    }
}

/// `BoundPhylactery.SUMMON_COUNT = 1` (`starter.py`).
pub const BOUND_PHYLACTERY_SUMMON_COUNT: i32 = 1;

// ---------------------------------------------------------------------------
// Event-pool starter upgrades — bead `sts2-rs-6ynt.5.2`
// ---------------------------------------------------------------------------
//
// The five EVENT-pool relics in `sts2_env/relics/starter.py` are
// post-event upgrades of the five canonical starters. Each carries
// the same hook shape as its base but with a bumped numeric pin:
//
//   BurningBlood  (HEAL=6)  → BlackBlood        (HEAL=12)
//   RingOfTheSnake (CARDS=2, T=1) → RingOfTheDrake (CARDS=2, T=3)
//   CrackedCore  (LIGHTNING=1) → InfusedCore  (LIGHTNING=3)
//   DivineRight  (STARS=3) → DivineDestiny  (STARS=6)
//   BoundPhylactery (SUMMON=1) → PhylacteryUnbound (START_COMBAT=5,
//                                                   START_TURN=2)
//
// The dispatch handlers for BlackBlood (`sts2-rs-dp7o`) and
// RingOfTheDrake (`sts2-rs-mvrj`) are already wired in
// `sts2-core::relic_effects`; this bead lands the matching `RelicSpec`
// factories so the run-builder can materialise them. The other three
// (InfusedCore / PhylacteryUnbound / DivineDestiny) inherit their
// dispatch handlers from the umbrella `install_default_registry`
// (currently `Resolved` markers); the listener bodies land when the
// matching base-starter handler (CrackedCore / BoundPhylactery /
// DivineRight) is wired alongside the upgrade arm. Until then the
// RelicSpec carries the correct AST variant so the parity exporter
// + run-builder are unblocked.

/// `BlackBlood.HEAL = 12` (`starter.py:42`).
pub const BLACK_BLOOD_HEAL: i32 = 12;
/// `RingOfTheDrake.EXTRA_CARDS = 2`, `TURNS = 3` (`starter.py:71-72`).
pub const RING_OF_THE_DRAKE_EXTRA_CARDS: i32 = 2;
/// `RingOfTheDrake.TURNS = 3` (`starter.py:72`).
pub const RING_OF_THE_DRAKE_TURNS: u16 = 3;
/// `InfusedCore.LIGHTNING_COUNT = 3` (`starter.py:102`).
pub const INFUSED_CORE_LIGHTNING: i32 = 3;
/// `PhylacteryUnbound.START_COMBAT_COUNT = 5` (`starter.py:134`).
pub const PHYLACTERY_UNBOUND_START_COMBAT_COUNT: i32 = 5;
/// `PhylacteryUnbound.START_TURN_COUNT = 2` (`starter.py:135`).
pub const PHYLACTERY_UNBOUND_START_TURN_COUNT: i32 = 2;
/// `DivineDestiny.STARS = 6` (`starter.py:166`).
pub const DIVINE_DESTINY_STARS: i32 = 6;

/// Black Blood — EVENT-pool upgrade of Burning Blood. Heals 12 HP at
/// `after_combat_victory` (vs Burning Blood's 6). Mirrors
/// `starter.py::BlackBlood`. Dispatch handler shipped in bead
/// `sts2-rs-dp7o`.
#[must_use]
pub const fn black_blood() -> RelicSpec {
    RelicSpec {
        id: RelicKind::BlackBlood,
        name: "BLACK_BLOOD",
        rarity: RelicRarity::Starter,
        effect: RelicEffect::HealAtCombatEnd {
            amount: BLACK_BLOOD_HEAL,
        },
    }
}

/// Ring of the Drake — EVENT-pool upgrade of Ring of the Snake. On
/// rounds 1–3, draw `EXTRA_CARDS = 2` additional cards (vs Snake's
/// round-1 only). Mirrors `starter.py::RingOfTheDrake`. Dispatch
/// handler shipped in bead `sts2-rs-mvrj`.
#[must_use]
pub const fn ring_of_the_drake() -> RelicSpec {
    RelicSpec {
        id: RelicKind::RingOfTheDrake,
        name: "RING_OF_THE_DRAKE",
        rarity: RelicRarity::Starter,
        effect: RelicEffect::ExtraHandDrawForRounds {
            extra_cards: RING_OF_THE_DRAKE_EXTRA_CARDS,
            max_round: RING_OF_THE_DRAKE_TURNS,
        },
    }
}

/// Infused Core — EVENT-pool upgrade of Cracked Core. At the start of
/// round 1, channel 3 Lightning orbs (vs Cracked Core's 1). Python
/// fires on `after_side_turn_start` (not `before_`) — the [`SideTurnPhase`]
/// discriminant captures the difference at registration time. Mirrors
/// `starter.py::InfusedCore`. Dispatch handler:
/// `on_after_side_turn_start_infused_core` in `sts2-core::relic_effects`
/// (bead `sts2-rs-eyowc`).
#[must_use]
pub const fn infused_core() -> RelicSpec {
    RelicSpec {
        id: RelicKind::InfusedCore,
        name: "INFUSED_CORE",
        rarity: RelicRarity::Starter,
        effect: RelicEffect::ChannelOrbAtSideTurnStart {
            orb_kind: ChannelOrbKind::Lightning,
            count: INFUSED_CORE_LIGHTNING,
            phase: SideTurnPhase::After,
            max_round: 1,
        },
    }
}

/// Phylactery Unbound — EVENT-pool upgrade of Bound Phylactery.
/// Summons 5 Osty at combat start and 2 more on every player
/// `after_side_turn_start` (no round gate, per
/// `starter.py:140-142`). The [`RelicSpec.effect`] slot carries the
/// primary `before_combat_start` shape; the secondary turn-start
/// listener is registered directly on the after-side-turn-start
/// channel in `sts2-core::relic_effects` once the Necrobinder
/// `summon_osty` runtime lands — same split as the base
/// `bound_phylactery` factory above.
#[must_use]
pub const fn phylactery_unbound() -> RelicSpec {
    RelicSpec {
        id: RelicKind::PhylacteryUnbound,
        name: "PHYLACTERY_UNBOUND",
        rarity: RelicRarity::Starter,
        effect: RelicEffect::SummonAtCombatStart {
            count: PHYLACTERY_UNBOUND_START_COMBAT_COUNT,
        },
    }
}

/// Divine Destiny — EVENT-pool upgrade of Divine Right. Round 1
/// only: gain 6 Stars (vs Divine Right's 3). Mirrors
/// `starter.py::DivineDestiny`. Dispatch handler:
/// `on_after_side_turn_start_divine_destiny` in
/// `sts2-core::relic_effects` (bead `sts2-rs-eyowc`).
#[must_use]
pub const fn divine_destiny() -> RelicSpec {
    RelicSpec {
        id: RelicKind::DivineDestiny,
        name: "DIVINE_DESTINY",
        rarity: RelicRarity::Starter,
        effect: RelicEffect::GainStarsAtSideTurnStart {
            amount: DIVINE_DESTINY_STARS,
            max_round: 1,
        },
    }
}

/// Install starter-relic dispatch handlers. The actual handler
/// functions live in `sts2_core::relic_effects`
/// (`install_default_registry`); this is a thin wrapper so the P7-2
/// pool boots through a uniform `install_pool` API.
pub fn install_pool() {
    relic_effects::install_default_registry();
}

#[cfg(test)]
mod tests {
    use super::*;
    use sts2_core::relics::RelicId;

    #[test]
    fn cracked_core_spec_pin() {
        let r = cracked_core();
        assert_eq!(r.id, RelicKind::CrackedCore);
        assert_eq!(r.name, "CRACKED_CORE");
        assert_eq!(r.rarity, RelicRarity::Starter);
        assert_eq!(
            r.effect,
            RelicEffect::ChannelOrbAtSideTurnStart {
                orb_kind: ChannelOrbKind::Lightning,
                count: CRACKED_CORE_LIGHTNING,
                phase: SideTurnPhase::Before,
                max_round: 1,
            }
        );
        assert_eq!(r.name, RelicId::CrackedCore.to_python_name());
        assert_eq!(CRACKED_CORE_LIGHTNING, 1);
    }

    #[test]
    fn ring_of_the_snake_spec_pin() {
        let r = ring_of_the_snake();
        assert_eq!(r.id, RelicKind::RingOfTheSnake);
        assert_eq!(r.name, "RING_OF_THE_SNAKE");
        assert_eq!(r.rarity, RelicRarity::Starter);
        assert_eq!(
            r.effect,
            RelicEffect::ExtraHandDrawForRounds {
                extra_cards: RING_OF_THE_SNAKE_EXTRA_CARDS,
                max_round: 1,
            }
        );
        assert_eq!(r.name, RelicId::RingOfTheSnake.to_python_name());
        assert_eq!(RING_OF_THE_SNAKE_EXTRA_CARDS, 2);
    }

    #[test]
    fn divine_right_spec_pin() {
        let r = divine_right();
        assert_eq!(r.id, RelicKind::DivineRight);
        assert_eq!(r.name, "DIVINE_RIGHT");
        assert_eq!(r.rarity, RelicRarity::Starter);
        assert_eq!(
            r.effect,
            RelicEffect::GainStarsAtSideTurnStart {
                amount: DIVINE_RIGHT_STARS,
                max_round: 1,
            }
        );
        assert_eq!(r.name, RelicId::DivineRight.to_python_name());
        assert_eq!(DIVINE_RIGHT_STARS, 3);
    }

    #[test]
    fn bound_phylactery_spec_pin() {
        let r = bound_phylactery();
        assert_eq!(r.id, RelicKind::BoundPhylactery);
        assert_eq!(r.name, "BOUND_PHYLACTERY");
        assert_eq!(r.rarity, RelicRarity::Starter);
        assert_eq!(
            r.effect,
            RelicEffect::SummonAtCombatStart {
                count: BOUND_PHYLACTERY_SUMMON_COUNT,
            }
        );
        assert_eq!(r.name, RelicId::BoundPhylactery.to_python_name());
        assert_eq!(BOUND_PHYLACTERY_SUMMON_COUNT, 1);
    }

    #[test]
    fn install_pool_registers_three_starter_listeners() {
        use sts2_core::relic_effects;
        install_pool();
        // Each starter relic registers a real typed hook listener
        // (lightning channel / +2 round-1 draw / per-turn stars).
        assert!(relic_effects::lookup_before_side_turn_start(RelicId::CrackedCore).is_some());
        assert!(relic_effects::lookup_modify_hand_draw(RelicId::RingOfTheSnake).is_some());
        assert!(relic_effects::lookup_after_side_turn_start(RelicId::DivineRight).is_some());
    }

    #[test]
    fn factories_are_pure_and_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<RelicSpec>();
        assert_eq!(cracked_core(), cracked_core());
        assert_eq!(ring_of_the_snake(), ring_of_the_snake());
        assert_eq!(divine_right(), divine_right());
        assert_eq!(bound_phylactery(), bound_phylactery());
        assert_eq!(black_blood(), black_blood());
        assert_eq!(ring_of_the_drake(), ring_of_the_drake());
        assert_eq!(infused_core(), infused_core());
        assert_eq!(phylactery_unbound(), phylactery_unbound());
        assert_eq!(divine_destiny(), divine_destiny());
    }

    // -----------------------------------------------------------------
    // Bead `sts2-rs-6ynt.5.2` — EVENT-pool starter-upgrade pins
    //
    // Each upgrade is the post-event variant of the matching canonical
    // starter with a bumped numeric pin. The spec pin asserts (a) the
    // RelicEffect variant matches the base-starter shape with the
    // upgraded constants, and (b) the load-bearing Python constants.
    // -----------------------------------------------------------------

    #[test]
    fn black_blood_spec_pin() {
        let r = black_blood();
        assert_eq!(r.id, RelicKind::BlackBlood);
        assert_eq!(r.name, "BLACK_BLOOD");
        assert_eq!(r.rarity, RelicRarity::Starter);
        assert_eq!(
            r.effect,
            RelicEffect::HealAtCombatEnd {
                amount: BLACK_BLOOD_HEAL,
            }
        );
        assert_eq!(r.name, RelicId::BlackBlood.to_python_name());
        assert_eq!(BLACK_BLOOD_HEAL, 12);
        // Upgrade ordering invariant — BlackBlood heals 2× BurningBlood.
        assert_eq!(BLACK_BLOOD_HEAL, 2 * 6);
    }

    #[test]
    fn ring_of_the_drake_spec_pin() {
        let r = ring_of_the_drake();
        assert_eq!(r.id, RelicKind::RingOfTheDrake);
        assert_eq!(r.name, "RING_OF_THE_DRAKE");
        assert_eq!(r.rarity, RelicRarity::Starter);
        assert_eq!(
            r.effect,
            RelicEffect::ExtraHandDrawForRounds {
                extra_cards: RING_OF_THE_DRAKE_EXTRA_CARDS,
                max_round: RING_OF_THE_DRAKE_TURNS,
            }
        );
        assert_eq!(r.name, RelicId::RingOfTheDrake.to_python_name());
        assert_eq!(RING_OF_THE_DRAKE_EXTRA_CARDS, 2);
        assert_eq!(RING_OF_THE_DRAKE_TURNS, 3);
        // Upgrade invariant — same per-round draw as Snake, but lasts
        // 3 rounds instead of 1.
        assert_eq!(RING_OF_THE_DRAKE_EXTRA_CARDS, RING_OF_THE_SNAKE_EXTRA_CARDS);
    }

    #[test]
    fn infused_core_spec_pin() {
        let r = infused_core();
        assert_eq!(r.id, RelicKind::InfusedCore);
        assert_eq!(r.name, "INFUSED_CORE");
        assert_eq!(r.rarity, RelicRarity::Starter);
        assert_eq!(
            r.effect,
            RelicEffect::ChannelOrbAtSideTurnStart {
                orb_kind: ChannelOrbKind::Lightning,
                count: INFUSED_CORE_LIGHTNING,
                // Python `InfusedCore` fires on `after_side_turn_start`
                // (not `before_` like CrackedCore) — this discriminant
                // is the load-bearing difference besides the count.
                phase: SideTurnPhase::After,
                max_round: 1,
            }
        );
        assert_eq!(r.name, RelicId::InfusedCore.to_python_name());
        assert_eq!(INFUSED_CORE_LIGHTNING, 3);
        assert_eq!(INFUSED_CORE_LIGHTNING, 3 * CRACKED_CORE_LIGHTNING);
    }

    #[test]
    fn phylactery_unbound_spec_pin() {
        let r = phylactery_unbound();
        assert_eq!(r.id, RelicKind::PhylacteryUnbound);
        assert_eq!(r.name, "PHYLACTERY_UNBOUND");
        assert_eq!(r.rarity, RelicRarity::Starter);
        assert_eq!(
            r.effect,
            RelicEffect::SummonAtCombatStart {
                count: PHYLACTERY_UNBOUND_START_COMBAT_COUNT,
            }
        );
        assert_eq!(r.name, RelicId::PhylacteryUnbound.to_python_name());
        assert_eq!(PHYLACTERY_UNBOUND_START_COMBAT_COUNT, 5);
        assert_eq!(PHYLACTERY_UNBOUND_START_TURN_COUNT, 2);
    }

    #[test]
    fn divine_destiny_spec_pin() {
        let r = divine_destiny();
        assert_eq!(r.id, RelicKind::DivineDestiny);
        assert_eq!(r.name, "DIVINE_DESTINY");
        assert_eq!(r.rarity, RelicRarity::Starter);
        assert_eq!(
            r.effect,
            RelicEffect::GainStarsAtSideTurnStart {
                amount: DIVINE_DESTINY_STARS,
                max_round: 1,
            }
        );
        assert_eq!(r.name, RelicId::DivineDestiny.to_python_name());
        assert_eq!(DIVINE_DESTINY_STARS, 6);
        assert_eq!(DIVINE_DESTINY_STARS, 2 * DIVINE_RIGHT_STARS);
    }
}
