//! Ironclad-exclusive power listener bodies — bead `sts2-rs-le3` (P6-3).
//!
//! ## Architecture note
//!
//! The hooks engine in `crates/sts2-core/src/hooks.rs` resolves listener
//! behaviour through inline `match` arms on [`PowerKind`] inside each hook
//! dispatcher (e.g. `modify_damage_additive`, `modify_block_additive`,
//! `after_block_gained`, …). Adding a new owner-side power listener
//! body means landing a new `match` arm in the relevant dispatcher.
//!
//! This module is the **registration / documentation surface** for the
//! ~20 Ironclad-exclusive `PowerKind` variants. Each `register_*` function
//! exists to:
//!
//! 1. Pin the Ironclad-side ownership of the variant (grep target).
//! 2. Provide a deterministic install-time hook so future engine
//!    refactors can move the inline `match` bodies into per-character
//!    dispatch tables without changing the call-site API.
//!
//! ## Powers covered
//!
//! Ironclad-exclusive variants present in the auto-generated `PowerKind`
//! enum (subset of canonical STS Ironclad-only powers; Python pin
//! `sts2-rl-agent@01c094d` carries a reduced set):
//!
//! * `Rage`         — gain block on every Attack played this turn.
//! * `Barricade`    — block carries between turns.
//! * `Corruption`   — Skills cost 0 and exhaust.
//! * `DarkEmbrace`  — draw 1 each time a card is exhausted.
//! * `DemonForm`    — start of turn: +N Strength.
//! * `FeelNoPain`   — gain N block each time a card is exhausted.
//! * `Juggernaut`   — deal N to a random enemy whenever block is gained.
//! * `Metallicize`  — gain N block at end of turn.
//! * `NoBlock`      — block is zero this turn (debuff variant).
//! * `NoDraw`       — cannot draw cards this turn.
//! * `Pyre`         — exhaust-side incinerator power.
//! * `Rupture`      — gain N Strength per HP-loss event.
//! * `FlameBarrier` — already wired in `hooks.rs::after_damage_received`;
//!                    used by the Flame Barrier card.
//!
//! ## Behaviour status
//!
//! Concrete hook bodies for these variants land progressively as the
//! card-pool surface that USES each power is exercised by parity fixtures.
//! Until then, owner-side `PowerKind` records are created by the cards
//! (`apply_power_to_player` calls in `crates/sts2-content/src/cards/ironclad.rs`)
//! and silently accumulate — the dispatchers in `hooks.rs` fall through
//! their `_` arms without modifying state. This is the "registered but
//! inert" coverage state that the parity-nightly known-divergences entry
//! tracks until each listener body lands.
//!
//! **Bead `sts2-rs-zlx.1`**: `FeelNoPain` and `DarkEmbrace` are no
//! longer inert — their `after_card_exhausted` bodies are wired in
//! [`crate::hooks::dispatch_after_card_exhausted`]. `Corruption`
//! remains inert at this hook (the Python pin does not implement
//! `on_card_exhausted` on `CorruptionPower`; its cost-mod +
//! force-exhaust behaviour is gated by `sts2-rs-zlx.5`'s play-time
//! cost-mod surface).
//!
//! **Bead `sts2-rs-zlx.3`**: `Rage` is no longer inert — its
//! `after_card_played` body (filter `card_type == Attack` → gain
//! block) is wired in [`crate::hooks::dispatch_after_card_played`].
//!
//! **Bead `sts2-rs-zlx.4`**: `DemonForm` is no longer inert — its
//! `after_side_turn_start` body (+N Strength on owner when
//! `side == owner.side`) is wired via `Effect::ApplyPower` in
//! [`crate::hooks::after_side_turn_start`].
//!
//! **Bead `sts2-rs-6ynt.6.1`**: `SetupStrike` and `ManglePower` are no
//! longer inert — their `after_turn_end` bodies are wired in
//! [`crate::hooks::after_turn_end`]. `SetupStrike` emits
//! `Effect::ApplyPower(STRENGTH, -amount)` + `Effect::RemovePower` on
//! owner-side turn-end (mirroring Python pin literally — there is no
//! apply-time +STR; the docstring describes unimplemented C# behaviour).
//! `ManglePower` emits `Effect::ApplyPower(STRENGTH, +amount)` +
//! `Effect::DecrementPowerAmount(-amount)` on owner-side turn-end,
//! restoring the strength previously lost.
//!
//! **Bead `sts2-rs-76t5`**: `OneTwoPunch` is no longer deferred. The
//! `CardPlayCountCtx` was extended with `owner` / `card_type` /
//! identity fields, and the `modify_card_play_count` +
//! `after_modifying_card_play_count` listener bodies are wired in
//! [`crate::hooks::modify_card_play_count`] /
//! [`crate::hooks::after_modifying_card_play_count`]. The dispatch +
//! replay loop in `combat_loop::play_card` runs the attack effect one
//! extra time per OneTwoPunch stack and decrements the power instance
//! (popping it on exhaustion).
//!
//! **Bead `sts2-rs-6ynt.6.3`**: `CrimsonMantle` and `Inferno` are no
//! longer inert. Both ride the per-instance `Power::counter` field
//! (`self_damage` accumulator, bead `sts2-rs-6ynt.6.11`).
//!
//! - **`CrimsonMantle.after_side_turn_start`**: on owner's player
//!   turn-start, deal `pow.counter` self-damage
//!   (`UNBLOCKABLE | UNPOWERED`) then gain `pow.amount` block
//!   (`UNPOWERED`). Wired in [`crate::hooks::after_side_turn_start`].
//! - **`Inferno.after_side_turn_start`**: same self-damage shape (no
//!   block-gain side effect). Wired in
//!   [`crate::hooks::after_side_turn_start`].
//! - **`Inferno.after_damage_received`**: when the owner takes
//!   unblocked damage on the owner's own turn, deal `pow.amount`
//!   `UNPOWERED` damage to every alive enemy. Wired inline in
//!   [`crate::hooks::dispatch_after_damage_received`] (Envenom-style
//!   post-walk reaction — same pattern as
//!   `dispatch_after_damage_given` for `EnvenomPower`).
//!
//! `IncrementSelfDamage()` (per-card-play counter bump) is wired in
//! `play_crimson_mantle` / `play_inferno` in
//! `crates/sts2-content/src/cards/ironclad.rs`. See those handler
//! doc-comments for the Python-vs-C# pin rationale.

use crate::PowerKind;

/// The set of Ironclad-exclusive `PowerKind` variants this module owns.
/// Listed alphabetically. The list is exposed for the coverage-census
/// hook-call test to assert ownership without scraping comments.
#[must_use]
pub const fn ironclad_power_kinds() -> &'static [PowerKind] {
    &[
        PowerKind::Aggression,
        PowerKind::Barricade,
        PowerKind::Colossus,
        PowerKind::Corruption,
        PowerKind::CrimsonMantle,
        PowerKind::Cruelty,
        PowerKind::DarkEmbrace,
        PowerKind::DemonForm,
        PowerKind::FeelNoPain,
        PowerKind::Grapple,
        PowerKind::Hellraiser,
        PowerKind::Inferno,
        PowerKind::Juggling,
        PowerKind::Juggernaut,
        PowerKind::ManglePower,
        PowerKind::Metallicize,
        PowerKind::NoBlock,
        PowerKind::NoDraw,
        PowerKind::OneTwoPunch,
        PowerKind::Plating,
        PowerKind::Pyre,
        PowerKind::Rage,
        PowerKind::Rupture,
        PowerKind::SetupStrike,
        PowerKind::Stampede,
        PowerKind::Tank,
        PowerKind::Unmovable,
        PowerKind::Vicious,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin: the Ironclad ownership list is non-empty and every variant
    /// has a non-empty `python_name`. Catches drift between this module
    /// and the auto-generated `PowerKind` enum.
    #[test]
    fn ironclad_power_kinds_have_python_names() {
        let kinds = ironclad_power_kinds();
        assert!(kinds.len() >= 10, "expected at least 10 Ironclad powers");
        for &k in kinds {
            assert!(!k.python_name().is_empty(), "empty name for {k:?}");
        }
    }

    /// Pin: each Ironclad power maps to a known Python `PowerId.<NAME>`.
    /// Stable string lock for parity-bridge tests.
    #[test]
    fn ironclad_power_names_match_python() {
        assert_eq!(PowerKind::Barricade.python_name(), "BARRICADE");
        assert_eq!(PowerKind::Colossus.python_name(), "COLOSSUS");
        assert_eq!(PowerKind::Corruption.python_name(), "CORRUPTION");
        assert_eq!(PowerKind::CrimsonMantle.python_name(), "CRIMSON_MANTLE");
        assert_eq!(PowerKind::Cruelty.python_name(), "CRUELTY");
        assert_eq!(PowerKind::DarkEmbrace.python_name(), "DARK_EMBRACE");
        assert_eq!(PowerKind::Inferno.python_name(), "INFERNO");
        assert_eq!(PowerKind::DemonForm.python_name(), "DEMON_FORM");
        assert_eq!(PowerKind::FeelNoPain.python_name(), "FEEL_NO_PAIN");
        assert_eq!(PowerKind::Juggernaut.python_name(), "JUGGERNAUT");
        assert_eq!(PowerKind::Metallicize.python_name(), "METALLICIZE");
        assert_eq!(PowerKind::NoBlock.python_name(), "NO_BLOCK");
        assert_eq!(PowerKind::NoDraw.python_name(), "NO_DRAW");
        assert_eq!(PowerKind::Pyre.python_name(), "PYRE");
        assert_eq!(PowerKind::Rage.python_name(), "RAGE");
        assert_eq!(PowerKind::Rupture.python_name(), "RUPTURE");
        assert_eq!(PowerKind::Plating.python_name(), "PLATING");
        assert_eq!(PowerKind::SetupStrike.python_name(), "SETUP_STRIKE");
        assert_eq!(PowerKind::ManglePower.python_name(), "MANGLE_POWER");
        assert_eq!(PowerKind::Tank.python_name(), "TANK");
        assert_eq!(PowerKind::Unmovable.python_name(), "UNMOVABLE");
    }
}
