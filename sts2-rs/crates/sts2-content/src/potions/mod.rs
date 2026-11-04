//! Potion content: static data definitions for all 63 potions.
//!
//! Ports `sts2-rl-agent@01c094d:sts2_env/potions/{all,base,effects}.py`.
//!
//! ## Crate boundary
//!
//! [`PotionSpec`] is the *static template* (immutable data shared
//! across all instances of the same potion). It is distinct from
//! `sts2_core::Potion`, which is the *runtime instance* (per-run /
//! per-combat). The split mirrors `CardSpec`/`Card` and
//! `RelicSpec`/`Relic`.
//!
//! [`PotionEffect`] enumerates the on-drink effect shapes shipped so
//! far. Bodies are pure data; the interpreter lives in
//! `sts2-core`'s Phase-7 on-drink dispatcher (umbrella bead
//! `sts2-rs-h9u`). For now every effect is `Noop` because the bead
//! lands the registry plumbing, not the gameplay bodies.

use sts2_core::{CharacterId, PotionKind, PotionRarity, PotionTargetType, PotionUsageType};

pub mod all;
pub mod dispatch;

// ---------------------------------------------------------------------------
// PotionEffect — on-drink effect AST
// ---------------------------------------------------------------------------

/// Pure-data on-drink effect for a potion. The Phase-7 dispatcher
/// (umbrella bead `sts2-rs-h9u`) will consume this enum from the
/// matching `on_drink` hook. For now the AST has a single `Noop`
/// variant — all 63 potions carry it; the registry-side acceptance
/// for this bead is "no longer `NotImplemented` for any `PotionKind`".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PotionEffect {
    /// No on-drink effect — placeholder for potions not yet ported.
    Noop,
}

// ---------------------------------------------------------------------------
// PotionSpec — static template for one potion
// ---------------------------------------------------------------------------

/// Static potion template. One [`PotionSpec`] exists per distinct
/// [`PotionKind`]. Concrete in-run potions (`sts2_core::Potion`) are
/// instantiated from a `PotionSpec` at potion-pickup time.
///
/// Field selection mirrors Python's `PotionModel`
/// (`sts2_env/potions/base.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PotionSpec {
    /// Stable identifier (matches `sts2_core::PotionKind`).
    pub id: PotionKind,
    /// Python `PotionId.<v>.name` — the parity-export identifier
    /// (e.g. `"BLOCK_POTION"`).
    pub name: &'static str,
    pub rarity: PotionRarity,
    pub usage_type: PotionUsageType,
    pub target_type: PotionTargetType,
    /// `Some(c)` iff this potion may only be drunk by character `c`.
    pub character_lock: Option<CharacterId>,
    /// `true` iff this potion is event-only / token-tier or carries
    /// the Python `can_be_generated_in_combat=False` flag. Used to
    /// filter the standard reward drop pool.
    pub is_event_only: bool,
    /// On-drink effect AST. Consumed by the Phase-7 dispatcher.
    pub effect: PotionEffect,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn potion_spec_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<PotionSpec>();
        assert_copy::<PotionEffect>();
    }
}
