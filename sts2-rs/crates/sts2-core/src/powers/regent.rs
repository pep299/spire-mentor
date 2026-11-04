#![allow(clippy::doc_markdown)]
//! Regent character power listeners — bead `sts2-rs-jxz` (umbrella).
//!
//! Regent-exclusive PowerKinds. The Regent character revolves around a
//! "stars" resource: card plays generate/spend stars, and many powers in
//! this module react to the stars pipeline (`on_stars_gained`,
//! `on_stars_spent`) or the closely-coupled draw/energy/card-generation
//! surfaces.
//!
//! ## Sub-bead structure
//!
//! Per the audit at `docs/reality-checks/2026-05-12-necro-regent-power-audit.md`,
//! the 26 Regent-exclusive PowerKinds split across three sub-beads, all
//! editing this single file:
//!
//! * `sts2-rs-jxz.1` — 7 stars-resource + draw-side powers:
//!   `BlackHole`, `ChildOfTheStars`, `StarNextTurn`, `DyingStar`,
//!   `PaleBlueDot`, `PillarOfCreation`, `Orbit`.
//! * `sts2-rs-jxz.2` (jxz-turn-cycle) — 10 turn-cycle /
//!   energy-block-cycle powers: `Conqueror`, `CrushUnder`,
//!   `ForegoneConclusion`, `Furnace`, `Genesis`, `MonarchsGaze`,
//!   `Royalties`, `SeekingEdge`, `Tyranny`, `VoidForm`.
//! * `sts2-rs-jxz.3` (jxz-cards-damage) — 9 card-played /
//!   damage-modifier / forge powers: `Arsenal`, `HammerTime`,
//!   `Monologue`, `Parry`, `Reflect`, `SpectrumShift`, `SwordSage`,
//!   `TheSealedThrone`, `Vigor`. Closes the jxz umbrella.
//!
//! The `regent_owns(PowerKind) -> bool` umbrella predicate covers
//! all 26 Regent-exclusive variants once jxz.3 lands.
//!
//! ## Architecture note
//!
//! As with `ironclad.rs` / `silent.rs`, the hooks engine in
//! `crates/sts2-core/src/hooks.rs` resolves listener behaviour through
//! inline `match` arms on [`PowerKind`] inside each hook dispatcher.
//! This module is the **per-power semantic surface**: each
//! `pub fn <power>_*` function encodes the numeric semantics ported
//! from the Python pin (not a `0` no-op stub). The hook dispatcher
//! calls these from its match arms when the stars / draw / energy
//! plumbing in `hooks.rs` is wired. Until then, these functions are
//! exercised by per-power unit tests below to pin the ported
//! semantics against drift.

use crate::PowerKind;

/// True iff `kind` is a Regent-exclusive PowerKind discriminant. The
/// hook dispatcher uses this to route listener invocations to the
/// bodies below; tests use it to confirm the variant set has not
/// drifted from the Python pin.
///
/// Covers the full 26-variant Regent-exclusive set: 7 (`jxz.1`) +
/// 10 (`jxz.2`) + 9 (`jxz.3`).
#[must_use]
pub const fn regent_owns(kind: PowerKind) -> bool {
    matches!(
        kind,
        // sts2-rs-jxz.1 — stars-resource + draw-side cluster (7)
        PowerKind::BlackHole
            | PowerKind::ChildOfTheStars
            | PowerKind::StarNextTurn
            | PowerKind::DyingStar
            | PowerKind::PaleBlueDot
            | PowerKind::PillarOfCreation
            | PowerKind::Orbit
            // sts2-rs-jxz.2 — turn-cycle + energy/block-cycle cluster (10)
            | PowerKind::Conqueror
            | PowerKind::CrushUnder
            | PowerKind::ForegoneConclusion
            | PowerKind::Furnace
            | PowerKind::Genesis
            | PowerKind::MonarchsGaze
            | PowerKind::Royalties
            | PowerKind::SeekingEdge
            | PowerKind::Tyranny
            | PowerKind::VoidForm
            // sts2-rs-jxz.3 — card-played + damage-modifier + forge cluster (9)
            | PowerKind::Arsenal
            | PowerKind::HammerTime
            | PowerKind::Monologue
            | PowerKind::Parry
            | PowerKind::Reflect
            | PowerKind::SpectrumShift
            | PowerKind::SwordSage
            | PowerKind::TheSealedThrone
            | PowerKind::Vigor
    )
}

// ---------------------------------------------------------------------------
// jxz.1 — stars-resource + draw-side listener bodies (7 powers).
// ---------------------------------------------------------------------------

/// `BlackHole` — when the owner gains or spends any stars, deal
/// `amount` (unpowered) damage to every living enemy.
///
/// Returns the per-enemy damage value the dispatcher should apply for
/// the triggering stars-{gained,spent} event (0 if the event delta is
/// non-positive, i.e. no damage burst).
#[must_use]
pub const fn black_hole_damage_on_stars_event(power_amount: i32, stars_delta: i32) -> i32 {
    if stars_delta > 0 {
        power_amount
    } else {
        0
    }
}

/// `ChildOfTheStars` — when the owner spends stars, gain
/// `amount * stars` block.
///
/// Returns the block delta the dispatcher should grant for the
/// triggering `on_stars_spent` event. The Python pin in
/// `remaining_a.py:413` calls `owner.gain_block(self.amount * stars)`
/// only when `stars > 0`.
#[must_use]
pub const fn child_of_the_stars_block_on_spend(power_amount: i32, stars_spent: i32) -> i32 {
    if stars_spent > 0 {
        power_amount.saturating_mul(stars_spent)
    } else {
        0
    }
}

/// `StarNextTurn` — at owner's `after_side_turn_start`, gain
/// `amount` stars (via `combat.gain_stars(owner, amount)`) and zero
/// the power's stack to mark itself for removal.
///
/// Returns the number of stars to grant; the dispatcher is
/// responsible for clearing the power instance afterward.
#[must_use]
pub const fn star_next_turn_stars_granted(power_amount: i32, is_owner_side: bool) -> i32 {
    if is_owner_side {
        power_amount
    } else {
        0
    }
}

/// `DyingStar` — at the end of the owner's turn, apply `-amount`
/// Strength to the owner (Python pin applies `+self.amount` where
/// `amount` is already negative for this debuff variant) and clear
/// the stack to signal removal.
///
/// Returns the Strength delta the dispatcher should apply. The
/// Python pin passes `self.amount` directly into
/// `owner.apply_power(STRENGTH, self.amount)` — for `DyingStar` this
/// is the stored (negative) amount.
#[must_use]
pub const fn dying_star_strength_delta_on_turn_end(power_amount: i32, is_owner_side: bool) -> i32 {
    if is_owner_side {
        power_amount
    } else {
        0
    }
}

/// `PaleBlueDot` — at hand-draw time, if the owner played at least
/// `CARD_PLAY_THRESHOLD` (5) cards last turn, add `amount` to the
/// draw count.
///
/// Returns the additive draw bonus (0 if the threshold isn't met).
#[must_use]
pub const fn pale_blue_dot_draw_bonus(power_amount: i32, cards_played_last_turn: i32) -> i32 {
    const CARD_PLAY_THRESHOLD: i32 = 5;
    if cards_played_last_turn >= CARD_PLAY_THRESHOLD {
        power_amount
    } else {
        0
    }
}

/// `PillarOfCreation` — when a card is generated and added to combat
/// for the owner's player, gain `amount` block (unpowered).
///
/// Returns the block delta the dispatcher should grant per
/// `on_card_generated` event. The Python pin grants
/// `owner.gain_block(self.amount)` unconditionally per event.
#[must_use]
pub const fn pillar_of_creation_block_per_generation(power_amount: i32) -> i32 {
    power_amount
}

/// `Orbit` — tracks total energy spent by the owner; every 4 energy
/// spent grants `amount` energy. Given the running totals
/// (energy_spent_so_far before this event, energy_spent_this_event,
/// and the stored trigger_count), returns the number of new
/// `amount`-sized energy grants that fire.
///
/// Returns the *count* of triggers (caller multiplies by
/// `power_amount` and calls `combat.gain_energy`). Encapsulates the
/// `triggers = energy_spent // 4 - trigger_count` arithmetic from
/// the pin verbatim, clamped to `>= 0`.
#[must_use]
pub const fn orbit_triggers_on_energy_spent(
    energy_spent_total_after: i32,
    trigger_count_prev: i32,
) -> i32 {
    const ENERGY_INCREMENT: i32 = 4;
    let raw = energy_spent_total_after / ENERGY_INCREMENT - trigger_count_prev;
    if raw > 0 {
        raw
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// jxz.2 — turn-cycle + energy/block-cycle listener bodies (10 powers).
// ---------------------------------------------------------------------------

/// `Conqueror` — Sovereign-Blade-tagged debuff: a `Sovereign Blade`
/// attack dealt against this Creature is doubled while powered. Ticks
/// down one stack at the end of the owner's side turn.
///
/// `modify_damage_multiplicative` semantics: returns a Q16.16-style
/// multiplier numerator/denominator pair (caller multiplies). Here we
/// return the integer scale factor `2` when the gate fires, else `1`.
///
/// Gate (per Python pin): the active `card_source` is
/// `CardId::SovereignBlade`, the damage `target` is the owner of this
/// power, and the damage props are `powered`.
#[must_use]
pub const fn conqueror_damage_scale(
    is_sovereign_blade_source: bool,
    target_is_owner: bool,
    is_powered: bool,
) -> i32 {
    if is_sovereign_blade_source && target_is_owner && is_powered {
        2
    } else {
        1
    }
}

/// `Conqueror` end-of-turn tick: decrements stack by 1 on the owner's
/// side, otherwise leaves the stack untouched. Returns the new stack.
#[must_use]
pub const fn conqueror_tick_on_turn_end(stack: i32, is_owner_side: bool) -> i32 {
    if is_owner_side {
        stack - 1
    } else {
        stack
    }
}

/// `CrushUnder` — temporary-Strength-style debuff. The initial
/// application reduces the owner's Strength by `amount`; at the end of
/// the owner's turn the power restores `+amount` Strength and signals
/// removal (stack -> 0).
///
/// Returns the Strength delta to apply to the owner on
/// `after_turn_end`, gated to the owner's side.
#[must_use]
pub const fn crush_under_strength_restore_on_turn_end(
    power_amount: i32,
    is_owner_side: bool,
) -> i32 {
    if is_owner_side {
        power_amount
    } else {
        0
    }
}

/// `ForegoneConclusion` — at the start of the owner's side turn,
/// invite the (player) owner to select `amount` cards from the draw
/// pile into their hand, then mark this power for removal.
///
/// Returns the number of cards to select. Gated to owner-side
/// AND the owner being a player; off-side or non-player returns 0
/// (no-op).
#[must_use]
pub const fn foregone_conclusion_cards_to_pick(
    power_amount: i32,
    is_owner_side: bool,
    owner_is_player: bool,
) -> i32 {
    if is_owner_side && owner_is_player {
        power_amount
    } else {
        0
    }
}

/// `Furnace` — at the start of the owner's side turn, forge a card
/// `amount` times (Regent forge mechanic). Returns the forge-call
/// count; 0 off-side.
#[must_use]
pub const fn furnace_forge_count(power_amount: i32, is_owner_side: bool) -> i32 {
    if is_owner_side {
        power_amount
    } else {
        0
    }
}

/// `Genesis` — at the start of the owner's side turn (player only),
/// the owner gains `amount` stars.
///
/// Returns the number of stars to grant; 0 if off-side or owner is
/// not a player.
#[must_use]
pub const fn genesis_stars_granted(
    power_amount: i32,
    is_owner_side: bool,
    owner_is_player: bool,
) -> i32 {
    if is_owner_side && owner_is_player {
        power_amount
    } else {
        0
    }
}

/// `MonarchsGaze` — when the owner deals powered-attack damage, apply
/// `amount` `MonarchsGazeStrengthDown` to the target. Returns the
/// `MonarchsGazeStrengthDown` delta to stack on the target; 0 if the
/// dealer is not the owner or the damage props are not powered.
#[must_use]
pub const fn monarchs_gaze_strength_down_amount(
    power_amount: i32,
    dealer_is_owner: bool,
    is_powered: bool,
) -> i32 {
    if dealer_is_owner && is_powered {
        power_amount
    } else {
        0
    }
}

/// `Royalties` — on combat victory, the owner gains `amount` gold as
/// an extra reward. Returns the gold delta to grant; unconditional
/// per the Python pin (the `after_combat_victory` channel only fires
/// on victory, so no extra gating is needed here).
#[must_use]
pub const fn royalties_gold_on_victory(power_amount: i32) -> i32 {
    power_amount
}

/// `SeekingEdge` — flag power for the Regent Forge mechanic. The
/// Python pin defines **no listener hooks** (StackType.Single,
/// ExtraHoverTips: Forge). Card-side / forge-side code reads the
/// power's presence; there is no per-turn/per-event body to port.
///
/// Returns `true` if the flag would be active for `power_amount`
/// (any positive stack). Surfaced as a typed predicate so callers
/// route through `regent_owns` + this helper instead of touching the
/// raw stack.
#[must_use]
pub const fn seeking_edge_is_active(power_amount: i32) -> bool {
    power_amount > 0
}

/// `Tyranny` — additive hand-draw bonus. Returns the draw delta the
/// `modify_hand_draw` dispatcher should add for the owner's player.
#[must_use]
pub const fn tyranny_draw_bonus(power_amount: i32) -> i32 {
    power_amount
}

/// `Tyranny` — at the start of the player owner's side turn, request
/// the player to exhaust `amount` cards from hand. Returns the
/// exhaust-request count; 0 off-side or for non-player owners.
#[must_use]
pub const fn tyranny_exhaust_request(
    power_amount: i32,
    is_owner_side: bool,
    owner_is_player: bool,
) -> i32 {
    if is_owner_side && owner_is_player {
        power_amount
    } else {
        0
    }
}

/// `VoidForm` — the first `amount` cards played per turn cost 0
/// energy (and 0 stars). Returns `Some(0)` when the gate fires
/// (override the card's energy cost to 0), else `None` (leave cost
/// untouched). Caller passes the running `cards_played_this_turn`
/// counter and the owner-equality flag.
#[must_use]
pub const fn void_form_cost_override(
    power_amount: i32,
    cards_played_this_turn: i32,
    card_owned_by_owner: bool,
) -> Option<i32> {
    if !card_owned_by_owner {
        return None;
    }
    if cards_played_this_turn >= power_amount {
        return None;
    }
    Some(0)
}

/// `VoidForm` — increments the cards-played counter when the owner
/// plays a card. Returns the new counter value (`prev_counter + 1` if
/// owned by the owner, else unchanged).
#[must_use]
pub const fn void_form_increment_on_play(prev_counter: i32, card_owned_by_owner: bool) -> i32 {
    if card_owned_by_owner {
        prev_counter + 1
    } else {
        prev_counter
    }
}

/// `VoidForm` — at the start of the owner's side turn, reset the
/// cards-played counter to 0. Returns the new counter value (0 on
/// owner side, otherwise the unchanged `prev_counter`).
#[must_use]
pub const fn void_form_reset_on_turn_start(prev_counter: i32, is_owner_side: bool) -> i32 {
    if is_owner_side {
        0
    } else {
        prev_counter
    }
}

// ---------------------------------------------------------------------------
// jxz.3 — card-played + damage-modifier + forge listener bodies (9 powers).
// ---------------------------------------------------------------------------

/// `Arsenal` — after a Colorless card is played by (or for) the
/// owner's player, the owner gains `amount` Strength. Returns the
/// Strength delta the dispatcher should apply on the
/// `after_card_played` event.
///
/// Gate (per Python pin): the card is colorless AND its `owner` is
/// either the owner or unset (None == implicit owner-side, matches
/// the Python `card_owner is owner or card_owner is None` branch).
#[must_use]
pub const fn arsenal_strength_on_card_played(
    power_amount: i32,
    card_is_colorless: bool,
    card_owner_matches: bool,
) -> i32 {
    if card_is_colorless && card_owner_matches {
        power_amount
    } else {
        0
    }
}

/// `HammerTime` — when the owner forges, every other living player
/// teammate also forges the same `amount`. Returns the per-teammate
/// forge amount to dispatch; 0 if the gate fails.
///
/// Gate (per Python pin): the forge `source` is not this power itself
/// (to break recursion), the forger is the owner, and `amount > 0`.
/// The dispatcher loops over `combat.get_teammates_of(owner)` and
/// calls `combat.forge(teammate, amount, source=self)`.
///
/// The `after_forge` dispatcher
/// already exists in `hooks.rs` (see `dispatch_after_forge`); wiring
/// listeners is left to a follow-up plumbing pass.
#[must_use]
pub const fn hammer_time_teammate_forge_amount(
    forge_amount: i32,
    forger_is_owner: bool,
    source_is_self: bool,
) -> i32 {
    if source_is_self || !forger_is_owner || forge_amount <= 0 {
        0
    } else {
        forge_amount
    }
}

/// `Monologue` — boss-flavor power: per card played by the owner,
/// gain `strength_per_card` Strength (default 1); at the end of the
/// owner's turn, undo the total Strength accumulated this turn.
///
/// Returns the Strength delta the `after_card_played` dispatcher
/// should apply (gated to owner-owned card plays). The accumulator
/// (`_strength_applied` in Python) is tracked by the dispatcher.
#[must_use]
pub const fn monologue_strength_per_card_played(
    strength_per_card: i32,
    card_owned_by_owner: bool,
) -> i32 {
    if card_owned_by_owner {
        strength_per_card
    } else {
        0
    }
}

/// `Monologue` — at end of owner's turn, restore `-strength_applied`
/// Strength (undo the accumulated buff) and signal removal. Returns
/// the Strength delta the dispatcher should apply; 0 off-side.
#[must_use]
pub const fn monologue_strength_undo_on_turn_end(
    strength_applied: i32,
    is_owner_side: bool,
) -> i32 {
    if is_owner_side {
        -strength_applied
    } else {
        0
    }
}

/// `Parry` — after the owner plays a Sovereign Blade card, gain
/// `amount` block (unpowered). Returns the block delta the
/// `after_card_played` dispatcher should grant; 0 if gate fails.
///
/// Gate (per Python pin): the card is Sovereign Blade AND its
/// `owner` matches the power owner (or is None == implicit
/// owner-side, mirroring Arsenal's branch).
#[must_use]
pub const fn parry_block_on_sovereign_blade(
    power_amount: i32,
    card_is_sovereign_blade: bool,
    card_owner_matches: bool,
) -> i32 {
    if card_is_sovereign_blade && card_owner_matches {
        power_amount
    } else {
        0
    }
}

/// `Reflect` — when the owner blocks some part of a powered attack,
/// deal the blocked amount back at the dealer as unpowered damage.
/// Returns the reflection damage to deal; 0 if the gate fails.
///
/// Gate (per Python pin): `blocked_amount > 0`, a dealer exists, and
/// the incoming damage is `powered`. The dispatcher is responsible
/// for issuing `combat.deal_damage(owner -> dealer, …, UNPOWERED)`.
#[must_use]
pub const fn reflect_damage_to_dealer(
    blocked_amount: i32,
    dealer_present: bool,
    is_powered: bool,
) -> i32 {
    if blocked_amount > 0 && dealer_present && is_powered {
        blocked_amount
    } else {
        0
    }
}

/// `Reflect` — at the start of the owner's side, decrement the
/// power's stack by 1; the dispatcher prunes the power when the
/// resulting stack drops to 0 or below. Returns the new stack.
#[must_use]
pub const fn reflect_tick_on_turn_start(stack: i32, is_owner_side: bool) -> i32 {
    if is_owner_side {
        stack - 1
    } else {
        stack
    }
}

/// `SpectrumShift` — at the start of the owner's side, if the owner
/// is a player, generate `amount` Colorless cards directly into hand.
/// Returns the number of cards to generate; 0 off-side or for
/// non-player owners.
#[must_use]
pub const fn spectrum_shift_cards_to_generate(
    power_amount: i32,
    is_owner_side: bool,
    owner_is_player: bool,
) -> i32 {
    if is_owner_side && owner_is_player {
        power_amount
    } else {
        0
    }
}

/// `SwordSage` — repeat-count modifier for Sovereign Blade card plays.
/// Returns the number of times a Sovereign Blade should repeat
/// (`amount + 1`, mirroring the Python `get_sovereign_blade_repeats`).
#[must_use]
pub const fn sword_sage_sovereign_blade_repeats(power_amount: i32) -> i32 {
    power_amount + 1
}

/// `SwordSage` — additive cost increase for Sovereign Blade card
/// plays. Returns the extra energy cost the
/// `modify_card_play_count` / cost-mod dispatcher should add.
#[must_use]
pub const fn sword_sage_sovereign_blade_cost_increase(power_amount: i32) -> i32 {
    power_amount
}

/// `TheSealedThrone` — before the owner plays any of their cards,
/// gain `amount` star(s). Returns the stars-delta the
/// `before_card_played` dispatcher should grant; 0 if the played
/// card is not owner-owned.
#[must_use]
pub const fn the_sealed_throne_stars_on_before_play(
    power_amount: i32,
    card_owned_by_owner: bool,
) -> i32 {
    if card_owned_by_owner {
        power_amount
    } else {
        0
    }
}

/// `Vigor` — the next powered card-attack dealt by the owner gets
/// `+amount` additive damage, after which the buff is consumed.
/// Returns the additive damage bonus to apply for the
/// `modify_damage_additive` event; 0 if the gate fails.
///
/// Gate (per Python pin): the dealer is the owner, the damage props
/// are `powered`, and the tracked attack matches the active
/// card-source / attack (the dispatcher tracks `_command_to_modify`
/// from `before_attack`).
#[must_use]
pub const fn vigor_damage_additive(
    power_amount: i32,
    dealer_is_owner: bool,
    is_powered: bool,
    tracks_current_attack: bool,
) -> i32 {
    if dealer_is_owner && is_powered && tracks_current_attack {
        power_amount
    } else {
        0
    }
}

/// `Vigor` — `before_attack` registration predicate. Returns `true`
/// if the dispatcher should mark this attack as the consumed-on-hit
/// attack for the Vigor buff (i.e. start tracking).
///
/// Gate (per Python pin): the attacker is the owner, the damage
/// props are `powered`, the attack has a card-typed `model_source`,
/// and no other attack is currently being tracked.
#[must_use]
#[allow(clippy::fn_params_excessive_bools)] // mirrors the 4-way Python gate verbatim
pub const fn vigor_should_track_attack(
    attacker_is_owner: bool,
    is_powered: bool,
    model_source_is_card: bool,
    already_tracking: bool,
) -> bool {
    attacker_is_owner && is_powered && model_source_is_card && !already_tracking
}

/// `Vigor` — `after_attack` consumption: when the tracked attack
/// resolves, decrement the power's amount by the snapshot captured
/// in `before_attack`. Returns the new amount; the dispatcher prunes
/// the power when the result drops to 0 (or below, if
/// `allow_negative=false`).
#[must_use]
pub const fn vigor_consume_on_after_attack(
    power_amount: i32,
    amount_when_attack_started: i32,
    attack_matches_tracked: bool,
) -> i32 {
    if attack_matches_tracked {
        power_amount - amount_when_attack_started
    } else {
        power_amount
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `regent_owns` covers the full 26-variant Regent-exclusive set
    /// (7 jxz.1 + 10 jxz.2 + 9 jxz.3) and rejects shared / non-Regent
    /// kinds. Catches drift between this module and the
    /// auto-generated `PowerKind` enum.
    #[test]
    fn regent_owns_all_jxz_variants() {
        let owned: Vec<PowerKind> = PowerKind::all()
            .iter()
            .copied()
            .filter(|&k| regent_owns(k))
            .collect();
        assert_eq!(
            owned.len(),
            26,
            "expected 26 Regent PowerKinds (7 jxz.1 + 10 jxz.2 + 9 jxz.3), got {owned:?}"
        );
        for &k in &[
            // jxz.1
            PowerKind::BlackHole,
            PowerKind::ChildOfTheStars,
            PowerKind::StarNextTurn,
            PowerKind::DyingStar,
            PowerKind::PaleBlueDot,
            PowerKind::PillarOfCreation,
            PowerKind::Orbit,
            // jxz.2
            PowerKind::Conqueror,
            PowerKind::CrushUnder,
            PowerKind::ForegoneConclusion,
            PowerKind::Furnace,
            PowerKind::Genesis,
            PowerKind::MonarchsGaze,
            PowerKind::Royalties,
            PowerKind::SeekingEdge,
            PowerKind::Tyranny,
            PowerKind::VoidForm,
            // jxz.3
            PowerKind::Arsenal,
            PowerKind::HammerTime,
            PowerKind::Monologue,
            PowerKind::Parry,
            PowerKind::Reflect,
            PowerKind::SpectrumShift,
            PowerKind::SwordSage,
            PowerKind::TheSealedThrone,
            PowerKind::Vigor,
        ] {
            assert!(regent_owns(k), "regent_owns must claim {k:?}");
        }
    }

    /// Non-Regent powers are not claimed (shared core pipeline / other
    /// characters).
    #[test]
    fn regent_does_not_claim_shared_or_other_character() {
        assert!(!regent_owns(PowerKind::Strength));
        assert!(!regent_owns(PowerKind::Vulnerable));
        assert!(!regent_owns(PowerKind::Poison));
        assert!(!regent_owns(PowerKind::Barricade)); // Ironclad
        assert!(!regent_owns(PowerKind::Envenom)); // Silent
    }

    /// `BlackHole`: only fires on positive stars deltas; zero/negative
    /// events deal no damage.
    #[test]
    fn black_hole_only_on_positive_stars() {
        assert_eq!(black_hole_damage_on_stars_event(3, 1), 3);
        assert_eq!(black_hole_damage_on_stars_event(7, 5), 7); // amount, not delta-scaled
        assert_eq!(black_hole_damage_on_stars_event(3, 0), 0);
        assert_eq!(black_hole_damage_on_stars_event(3, -1), 0);
    }

    /// `ChildOfTheStars`: block scales linearly with stars_spent when
    /// positive; zero/negative => 0.
    #[test]
    fn child_of_the_stars_block_scaling() {
        assert_eq!(child_of_the_stars_block_on_spend(2, 3), 6);
        assert_eq!(child_of_the_stars_block_on_spend(5, 1), 5);
        assert_eq!(child_of_the_stars_block_on_spend(4, 0), 0);
        assert_eq!(child_of_the_stars_block_on_spend(4, -2), 0);
    }

    /// `StarNextTurn`: grants `amount` stars on owner's side turn
    /// start, nothing otherwise.
    #[test]
    fn star_next_turn_owner_gating() {
        assert_eq!(star_next_turn_stars_granted(2, true), 2);
        assert_eq!(star_next_turn_stars_granted(2, false), 0);
        assert_eq!(star_next_turn_stars_granted(0, true), 0);
    }

    /// `DyingStar`: applies the stored Strength delta at owner-side
    /// turn end; gated to owner side.
    #[test]
    fn dying_star_strength_delta() {
        assert_eq!(dying_star_strength_delta_on_turn_end(-3, true), -3);
        assert_eq!(dying_star_strength_delta_on_turn_end(-3, false), 0);
        assert_eq!(dying_star_strength_delta_on_turn_end(2, true), 2);
    }

    /// `PaleBlueDot`: threshold gated at 5 cards played last turn.
    #[test]
    fn pale_blue_dot_threshold() {
        assert_eq!(pale_blue_dot_draw_bonus(2, 5), 2);
        assert_eq!(pale_blue_dot_draw_bonus(2, 10), 2);
        assert_eq!(pale_blue_dot_draw_bonus(2, 4), 0);
        assert_eq!(pale_blue_dot_draw_bonus(2, 0), 0);
    }

    /// `PillarOfCreation`: grants `amount` block per generation
    /// event, unconditionally.
    #[test]
    fn pillar_of_creation_per_event_block() {
        assert_eq!(pillar_of_creation_block_per_generation(3), 3);
        assert_eq!(pillar_of_creation_block_per_generation(0), 0);
    }

    /// `Orbit`: every 4 energy spent triggers a fresh `amount`-sized
    /// energy grant; matches Python `energy_spent // 4 - trigger_count`.
    #[test]
    fn orbit_energy_threshold_arithmetic() {
        // Spend 4 energy from clean state -> 1 trigger.
        assert_eq!(orbit_triggers_on_energy_spent(4, 0), 1);
        // Spend 3 more (total 7) with 1 prior trigger -> 0 triggers.
        assert_eq!(orbit_triggers_on_energy_spent(7, 1), 0);
        // Spend 1 more (total 8) with 1 prior trigger -> 1 trigger.
        assert_eq!(orbit_triggers_on_energy_spent(8, 1), 1);
        // Big jump: 12 total, no prior triggers -> 3 triggers.
        assert_eq!(orbit_triggers_on_energy_spent(12, 0), 3);
        // Anomalous negative branch (defensive) -> 0.
        assert_eq!(orbit_triggers_on_energy_spent(2, 5), 0);
    }

    // -----------------------------------------------------------------
    // jxz.2 — turn-cycle + energy/block-cycle tests.
    // -----------------------------------------------------------------

    /// `Conqueror`: 2x only when Sovereign-Blade source + target ==
    /// owner + powered; anything else is identity (1x).
    #[test]
    fn conqueror_damage_scale_gates() {
        assert_eq!(conqueror_damage_scale(true, true, true), 2);
        // Non-Sovereign-Blade source -> identity.
        assert_eq!(conqueror_damage_scale(false, true, true), 1);
        // Wrong target -> identity.
        assert_eq!(conqueror_damage_scale(true, false, true), 1);
        // Unpowered -> identity.
        assert_eq!(conqueror_damage_scale(true, true, false), 1);
    }

    /// `Conqueror`: tick down only on owner side.
    #[test]
    fn conqueror_tick_owner_gating() {
        assert_eq!(conqueror_tick_on_turn_end(3, true), 2);
        assert_eq!(conqueror_tick_on_turn_end(3, false), 3);
        // Tick from 1 -> 0 (signals expiry); the dispatcher prunes
        // the power once stack <= 0.
        assert_eq!(conqueror_tick_on_turn_end(1, true), 0);
    }

    /// `CrushUnder`: restores `amount` Strength on owner-side
    /// turn-end; nothing off-side.
    #[test]
    fn crush_under_restore_strength_gating() {
        assert_eq!(crush_under_strength_restore_on_turn_end(2, true), 2);
        assert_eq!(crush_under_strength_restore_on_turn_end(2, false), 0);
        assert_eq!(crush_under_strength_restore_on_turn_end(0, true), 0);
    }

    /// `ForegoneConclusion`: only picks cards for the player owner on
    /// their own side; off-side or non-player -> 0.
    #[test]
    fn foregone_conclusion_pick_gating() {
        assert_eq!(foregone_conclusion_cards_to_pick(2, true, true), 2);
        assert_eq!(foregone_conclusion_cards_to_pick(2, false, true), 0);
        assert_eq!(foregone_conclusion_cards_to_pick(2, true, false), 0);
        assert_eq!(foregone_conclusion_cards_to_pick(2, false, false), 0);
    }

    /// `Furnace`: forges `amount` times on owner side, 0 off-side.
    #[test]
    fn furnace_owner_gating() {
        assert_eq!(furnace_forge_count(3, true), 3);
        assert_eq!(furnace_forge_count(3, false), 0);
    }

    /// `Genesis`: grants `amount` stars on owner-side player turns
    /// only.
    #[test]
    fn genesis_owner_player_gating() {
        assert_eq!(genesis_stars_granted(1, true, true), 1);
        assert_eq!(genesis_stars_granted(1, false, true), 0);
        assert_eq!(genesis_stars_granted(1, true, false), 0);
        assert_eq!(genesis_stars_granted(2, true, true), 2);
    }

    /// `MonarchsGaze`: only fires when the dealer is the owner and
    /// the damage is powered.
    #[test]
    fn monarchs_gaze_gates() {
        assert_eq!(monarchs_gaze_strength_down_amount(2, true, true), 2);
        assert_eq!(monarchs_gaze_strength_down_amount(2, false, true), 0);
        assert_eq!(monarchs_gaze_strength_down_amount(2, true, false), 0);
        assert_eq!(monarchs_gaze_strength_down_amount(2, false, false), 0);
    }

    /// `Royalties`: grants `amount` gold unconditionally on victory.
    #[test]
    fn royalties_gold_unconditional() {
        assert_eq!(royalties_gold_on_victory(5), 5);
        assert_eq!(royalties_gold_on_victory(0), 0);
    }

    /// `SeekingEdge`: flag predicate based on stack > 0.
    #[test]
    fn seeking_edge_flag_predicate() {
        assert!(seeking_edge_is_active(1));
        assert!(seeking_edge_is_active(2));
        assert!(!seeking_edge_is_active(0));
        assert!(!seeking_edge_is_active(-1));
    }

    /// `Tyranny`: draw bonus is straight additive.
    #[test]
    fn tyranny_draw_bonus_additive() {
        assert_eq!(tyranny_draw_bonus(1), 1);
        assert_eq!(tyranny_draw_bonus(3), 3);
        assert_eq!(tyranny_draw_bonus(0), 0);
    }

    /// `Tyranny`: exhaust request only fires for the player owner on
    /// their side.
    #[test]
    fn tyranny_exhaust_request_gating() {
        assert_eq!(tyranny_exhaust_request(2, true, true), 2);
        assert_eq!(tyranny_exhaust_request(2, false, true), 0);
        assert_eq!(tyranny_exhaust_request(2, true, false), 0);
    }

    /// `VoidForm`: the first `amount` owned-by-owner card plays get
    /// their cost overridden to 0.
    #[test]
    fn void_form_cost_override_gating() {
        // Owner's first 2 cards (amount=2) are free.
        assert_eq!(void_form_cost_override(2, 0, true), Some(0));
        assert_eq!(void_form_cost_override(2, 1, true), Some(0));
        // 3rd card -> no override.
        assert_eq!(void_form_cost_override(2, 2, true), None);
        // Non-owner-owned cards never overridden.
        assert_eq!(void_form_cost_override(2, 0, false), None);
        // Zero stack -> no override.
        assert_eq!(void_form_cost_override(0, 0, true), None);
    }

    /// `VoidForm`: increment counter only on owner-owned plays.
    #[test]
    fn void_form_counter_increment() {
        assert_eq!(void_form_increment_on_play(0, true), 1);
        assert_eq!(void_form_increment_on_play(3, true), 4);
        assert_eq!(void_form_increment_on_play(3, false), 3);
    }

    /// `VoidForm`: reset counter at the owner's side turn start.
    #[test]
    fn void_form_reset_on_turn_start() {
        assert_eq!(super::void_form_reset_on_turn_start(7, true), 0);
        assert_eq!(super::void_form_reset_on_turn_start(7, false), 7);
    }

    // -----------------------------------------------------------------
    // jxz.3 — card-played + damage-modifier + forge tests.
    // -----------------------------------------------------------------

    /// `Arsenal`: only fires for colorless cards belonging to (or
    /// implicitly belonging to) the owner.
    #[test]
    fn arsenal_colorless_gates() {
        assert_eq!(arsenal_strength_on_card_played(2, true, true), 2);
        assert_eq!(arsenal_strength_on_card_played(2, false, true), 0);
        assert_eq!(arsenal_strength_on_card_played(2, true, false), 0);
        assert_eq!(arsenal_strength_on_card_played(0, true, true), 0);
    }

    /// `HammerTime`: re-forge for teammates iff source isn't self,
    /// forger is owner, and amount > 0.
    #[test]
    fn hammer_time_teammate_gates() {
        assert_eq!(hammer_time_teammate_forge_amount(3, true, false), 3);
        // source-is-self break-recursion gate.
        assert_eq!(hammer_time_teammate_forge_amount(3, true, true), 0);
        // Wrong forger.
        assert_eq!(hammer_time_teammate_forge_amount(3, false, false), 0);
        // Non-positive amount.
        assert_eq!(hammer_time_teammate_forge_amount(0, true, false), 0);
        assert_eq!(hammer_time_teammate_forge_amount(-1, true, false), 0);
    }

    /// `Monologue`: apply `strength_per_card` per owner-owned play;
    /// undo total on owner-side turn end.
    #[test]
    fn monologue_play_and_undo() {
        assert_eq!(monologue_strength_per_card_played(1, true), 1);
        assert_eq!(monologue_strength_per_card_played(2, true), 2);
        assert_eq!(monologue_strength_per_card_played(1, false), 0);

        assert_eq!(monologue_strength_undo_on_turn_end(5, true), -5);
        assert_eq!(monologue_strength_undo_on_turn_end(5, false), 0);
        assert_eq!(monologue_strength_undo_on_turn_end(0, true), 0);
    }

    /// `Parry`: block only on owner-owned Sovereign Blade plays.
    #[test]
    fn parry_sovereign_blade_gates() {
        assert_eq!(parry_block_on_sovereign_blade(4, true, true), 4);
        assert_eq!(parry_block_on_sovereign_blade(4, false, true), 0);
        assert_eq!(parry_block_on_sovereign_blade(4, true, false), 0);
    }

    /// `Reflect`: reflects blocked damage on powered attacks with a
    /// known dealer; ticks down on owner-side turn start.
    #[test]
    fn reflect_block_and_tick() {
        assert_eq!(reflect_damage_to_dealer(3, true, true), 3);
        // No blocked amount.
        assert_eq!(reflect_damage_to_dealer(0, true, true), 0);
        // No dealer.
        assert_eq!(reflect_damage_to_dealer(3, false, true), 0);
        // Unpowered.
        assert_eq!(reflect_damage_to_dealer(3, true, false), 0);

        assert_eq!(reflect_tick_on_turn_start(2, true), 1);
        assert_eq!(reflect_tick_on_turn_start(2, false), 2);
        assert_eq!(reflect_tick_on_turn_start(1, true), 0);
    }

    /// `SpectrumShift`: generates `amount` Colorless cards iff
    /// owner-side AND owner is a player.
    #[test]
    fn spectrum_shift_owner_player_gates() {
        assert_eq!(spectrum_shift_cards_to_generate(2, true, true), 2);
        assert_eq!(spectrum_shift_cards_to_generate(2, false, true), 0);
        assert_eq!(spectrum_shift_cards_to_generate(2, true, false), 0);
    }

    /// `SwordSage`: repeat count is `amount + 1`; cost-increase is
    /// `amount`.
    #[test]
    fn sword_sage_repeats_and_cost() {
        assert_eq!(sword_sage_sovereign_blade_repeats(0), 1);
        assert_eq!(sword_sage_sovereign_blade_repeats(1), 2);
        assert_eq!(sword_sage_sovereign_blade_repeats(3), 4);

        assert_eq!(sword_sage_sovereign_blade_cost_increase(0), 0);
        assert_eq!(sword_sage_sovereign_blade_cost_increase(2), 2);
    }

    /// `TheSealedThrone`: grants `amount` stars iff the played card
    /// belongs to the owner.
    #[test]
    fn the_sealed_throne_stars_gating() {
        assert_eq!(the_sealed_throne_stars_on_before_play(1, true), 1);
        assert_eq!(the_sealed_throne_stars_on_before_play(2, true), 2);
        assert_eq!(the_sealed_throne_stars_on_before_play(2, false), 0);
    }

    /// `Vigor`: additive damage iff dealer-is-owner, powered, AND
    /// the tracked attack matches.
    #[test]
    fn vigor_damage_additive_gates() {
        assert_eq!(vigor_damage_additive(3, true, true, true), 3);
        // Dealer is not the owner.
        assert_eq!(vigor_damage_additive(3, false, true, true), 0);
        // Unpowered.
        assert_eq!(vigor_damage_additive(3, true, false, true), 0);
        // Tracking mismatch — Vigor is consumed by exactly one attack.
        assert_eq!(vigor_damage_additive(3, true, true, false), 0);
    }

    /// `Vigor`: tracks the attack iff attacker is owner, powered,
    /// model-source is a card, and no other attack is being tracked.
    #[test]
    fn vigor_should_track_attack_gates() {
        assert!(vigor_should_track_attack(true, true, true, false));
        assert!(!vigor_should_track_attack(false, true, true, false));
        assert!(!vigor_should_track_attack(true, false, true, false));
        assert!(!vigor_should_track_attack(true, true, false, false));
        assert!(!vigor_should_track_attack(true, true, true, true));
    }

    /// `Vigor`: consume on tracked-attack resolution — subtract the
    /// snapshot captured at `before_attack`.
    #[test]
    fn vigor_consume_on_after_attack_resolution() {
        assert_eq!(vigor_consume_on_after_attack(5, 5, true), 0);
        assert_eq!(vigor_consume_on_after_attack(5, 3, true), 2);
        // Untracked attacks leave the amount untouched.
        assert_eq!(vigor_consume_on_after_attack(5, 3, false), 5);
    }
}
