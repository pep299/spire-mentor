//! Shared card-spec types (extracted from `cards/mod.rs` to keep the latter a
//! thin module aggregator).
//!
//! Bead `sts2-rs-pmf` (P6-1b) moved `CardEffect`, `CardKeywords`, `CardSpec`,
//! and `CardRarity` out of `mod.rs` to eliminate hand-edit / codegen
//! contention on the module-aggregator file, which was generated from
//! `cards.toml` at the time. The generator has since been retired and
//! `cards/mod.rs` is hand-maintained, but the split still earns its keep: it
//! keeps the aggregator to `pub mod` declarations plus `pub use types::*;`
//! re-exports, so a merge conflict there is never in type definitions.

use sts2_core::cards::{CardId, CardType, TargetType};
use sts2_core::PowerKind;

// ---------------------------------------------------------------------------
// CardEffect — on-play effect AST
// ---------------------------------------------------------------------------

/// Pure-data on-play effect for a card. Phase 6 (bead `sts2-rs-2un`) will
/// implement an interpreter that consumes this enum from the active card
/// during `Action::PlayCard`.
///
/// Variants intentionally cover only the four starter-card shapes plus a
/// `Noop` for cards that do nothing on play (curses like `AscendersBane`,
/// which is unplayable). Phase 6 adds more variants — `Draw`,
/// `GainEnergy`, `ExhaustFromHand`, `ApplyDebuffOnly`, and so on.
/// Closures are deliberately avoided so [`CardSpec`] stays
/// `Copy`-eligible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CardEffect {
    /// No on-play effect (e.g. unplayable curses).
    Noop,
    /// Deal `base` damage to a single targeted enemy. The card's own
    /// `target_type` must be `Enemy` for this variant to be valid.
    DealDamage { base: i32 },
    /// Gain `base` block on the player (self).
    GainBlock { base: i32 },
    /// Deal `damage` to a single targeted enemy, then apply `amount`
    /// stacks of `kind` to that same enemy.
    DealDamageAndApplyPower {
        damage: i32,
        kind: PowerKind,
        amount: i32,
    },
}

// ---------------------------------------------------------------------------
// CardKeywords — bitflags for boolean modifiers
// ---------------------------------------------------------------------------

/// Bit-packed keyword flags. Mirrors Python's
/// `CardInstance.keywords: frozenset[str]` plus the `is_removable` /
/// `upgradeable` properties — both modelled as flags here so [`CardSpec`]
/// stays a flat POD.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CardKeywords(pub u16);

impl CardKeywords {
    /// Card has no flags set.
    pub const NONE: Self = Self(0);
    /// Card cannot be played from hand. Mirrors `"unplayable"`.
    pub const UNPLAYABLE: Self = Self(1 << 0);
    /// Card discards itself at end of turn (regardless of play). Mirrors
    /// `"ethereal"`.
    pub const ETHEREAL: Self = Self(1 << 1);
    /// Card stays in hand at end of turn. Mirrors `"retain"`.
    pub const RETAIN: Self = Self(1 << 2);
    /// Card play exhausts the card. Mirrors `"exhaust"`.
    pub const EXHAUST: Self = Self(1 << 3);
    /// Card cannot be removed from a deck. Mirrors `"eternal"`.
    pub const NON_REMOVABLE: Self = Self(1 << 4);
    /// Card has no upgraded variant. Used for `AscendersBane`.
    pub const NON_UPGRADEABLE: Self = Self(1 << 5);
    /// Card is drawn at the start of combat. Mirrors `"innate"`.
    pub const INNATE: Self = Self(1 << 6);
    /// Card cannot be removed from the deck; it persists forever.
    /// C# `CanonicalKeywords = [Eternal]`, and this is THE removal gate:
    /// `spec_index::card_is_removable_instance` tests this bit and nothing
    /// else, mirroring `CardModel.IsRemovable =>
    /// !Keywords.Contains(CardKeyword.Eternal)` (`CardModel.cs:738`).
    /// `NON_REMOVABLE` above is the older local alias, still set alongside it
    /// but read by nothing that decides removability.
    /// Bit 7 — mirrors `KeywordOverride::ETERNAL` (same bit index).
    pub const ETERNAL: Self = Self(1 << 7);
    /// Card is Sly: it triggers when discarded (full mechanic =
    /// discard-trigger semantics + the single-turn-Sly flag). Mirrors
    /// C# `CardKeyword.Sly`. Representational only — this is the
    /// display-keyword companion that mirrors the C# keyword set exactly;
    /// the Sly *effect* is wired ad-hoc per card / deferred (see bead
    /// `sts2-rs-vq2g3`, scope (a)).
    /// Bit 8 — distinct from every other keyword bit.
    pub const SLY: Self = Self(1 << 8);

    /// Add `other` to this flag set, returning the union. Const-friendly.
    #[inline]
    #[must_use]
    pub const fn with(self, other: Self) -> Self {
        self.union(other)
    }

    /// Remove `other`'s flags from this set. Const-friendly.
    #[inline]
    #[must_use]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Test whether this set contains every flag in `other`.
    #[inline]
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Combine two flag sets. `const`-friendly.
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOr for CardKeywords {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

// ---------------------------------------------------------------------------
// CardRarity / CardSpec
// ---------------------------------------------------------------------------

/// Card rarity bucket. Mirrors the authoritative C# `CardRarity`
/// (`MegaCrit.Sts2.Core.Entities.Cards/CardRarity.cs`): None, Basic,
/// Common, Uncommon, Rare, Ancient, Event, Token, Status, Curse, Quest.
///
/// The variant declaration order here does NOT match the C# enum's integer
/// values (it groups Status/Curse before Event for historical reasons), so
/// any code that needs the C# `orderby c.Rarity` ordinal maps it explicitly
/// (see [`super::spec_index::card_rarity_rank_csharp`] /
/// `droplet_of_precognition::csharp_rarity_ordinal`) rather than casting the
/// discriminant. `Token` is the per-combat token pool's rarity (GiantRock,
/// Shiv, Soul, the Regent minion cards, …): C#
/// `CardFactory.GetDefaultTransformationOptions` forces a Token (like Event /
/// Ancient) original onto the `ColorlessCardPool`, so it must be its own
/// bucket rather than folded into `Status`. Bead `sts2-rs-p52dg`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CardRarity {
    Basic,
    Common,
    Uncommon,
    Rare,
    Ancient,
    Status,
    Curse,
    Event,
    Token,
    Quest,
}

/// Spec-level upgrade delta for a card: the upgraded *absolute* values of
/// the POD fields that change when a card is upgraded. `None` fields are
/// left at their un-upgraded value. Sourced from the Python-pin
/// `X if upgraded else Y` ternaries (generated into
/// `card_upgrades_generated::upgrade_delta`), cross-checked against the C#
/// `OnUpgrade`/`UpgradeValueBy` deltas. `effect_vars`-only upgrades
/// (Vulnerable 2->3, draw 1->2, …) are NOT represented here — those are
/// handled inline by the card effect handlers. Bead `sts2-rs-vd6pf`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CardUpgradeDelta {
    pub cost: Option<i8>,
    pub base_damage: Option<i32>,
    pub base_block: Option<i32>,
    /// Absolute upgraded keyword set, when upgrading changes it (e.g.
    /// Aggression gains `INNATE`, BurningPact drops `EXHAUST`). `None`
    /// leaves the un-upgraded keywords intact.
    pub keywords: Option<CardKeywords>,
}

/// Static card template. One [`CardSpec`] exists per distinct [`CardId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardSpec {
    pub id: CardId,
    pub name: &'static str,
    pub cost: i8,
    pub card_type: CardType,
    pub target_type: TargetType,
    pub rarity: CardRarity,
    pub base_damage: i32,
    pub base_block: i32,
    pub keywords: CardKeywords,
    pub effect: CardEffect,
}

impl CardSpec {
    /// Compile-time-friendly default for fields most cards leave at zero.
    pub(crate) const fn base() -> Self {
        Self {
            id: CardId::StrikeIronclad, // overwritten by every factory
            name: "",
            cost: 0,
            card_type: CardType::Skill,
            target_type: TargetType::None,
            rarity: CardRarity::Basic,
            base_damage: 0,
            base_block: 0,
            keywords: CardKeywords::NONE,
            effect: CardEffect::Noop,
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_removable(&self) -> bool {
        !self.keywords.contains(CardKeywords::NON_REMOVABLE)
    }

    #[inline]
    #[must_use]
    pub const fn is_upgradeable(&self) -> bool {
        !self.keywords.contains(CardKeywords::NON_UPGRADEABLE)
    }

    #[inline]
    #[must_use]
    pub const fn is_unplayable(&self) -> bool {
        self.keywords.contains(CardKeywords::UNPLAYABLE)
    }

    #[inline]
    #[must_use]
    pub const fn is_ethereal(&self) -> bool {
        self.keywords.contains(CardKeywords::ETHEREAL)
    }

    #[inline]
    #[must_use]
    pub const fn exhausts(&self) -> bool {
        self.keywords.contains(CardKeywords::EXHAUST)
    }

    #[inline]
    #[must_use]
    pub const fn retains(&self) -> bool {
        self.keywords.contains(CardKeywords::RETAIN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords_contains_works() {
        let k = CardKeywords::ETHEREAL | CardKeywords::UNPLAYABLE;
        assert!(k.contains(CardKeywords::ETHEREAL));
        assert!(k.contains(CardKeywords::UNPLAYABLE));
        assert!(!k.contains(CardKeywords::EXHAUST));
        assert!(k.contains(CardKeywords::ETHEREAL | CardKeywords::UNPLAYABLE));
        assert!(k.contains(CardKeywords::NONE));
    }

    /// Kills `| -> ^` in `CardKeywords::union`: union must be idempotent /
    /// OR-like, so unioning overlapping sets keeps the shared bit. Under
    /// XOR the shared bit would cancel out.
    #[test]
    fn keywords_union_is_or_not_xor() {
        let a = CardKeywords::EXHAUST | CardKeywords::ETHEREAL;
        let b = CardKeywords::EXHAUST | CardKeywords::RETAIN;
        let u = a.union(b);
        // EXHAUST overlaps; under XOR it would be dropped.
        assert!(
            u.contains(CardKeywords::EXHAUST),
            "shared bit must survive union"
        );
        assert!(u.contains(CardKeywords::ETHEREAL));
        assert!(u.contains(CardKeywords::RETAIN));
        assert_eq!(
            u.0,
            CardKeywords::EXHAUST.0 | CardKeywords::ETHEREAL.0 | CardKeywords::RETAIN.0
        );
    }

    /// Kills `exhausts -> false` and `retains -> {true,false}`: a card whose
    /// keyword set includes EXHAUST/RETAIN reports them, and a bare card does
    /// not. Distinguishes the real flag check from constant returns.
    #[test]
    fn exhausts_and_retains_track_keywords() {
        let mut exhaust_spec = CardSpec::base();
        exhaust_spec.keywords = CardKeywords::EXHAUST;
        assert!(
            exhaust_spec.exhausts(),
            "EXHAUST keyword => exhausts() true"
        );
        assert!(
            !exhaust_spec.retains(),
            "no RETAIN keyword => retains() false"
        );

        let mut retain_spec = CardSpec::base();
        retain_spec.keywords = CardKeywords::RETAIN;
        assert!(retain_spec.retains(), "RETAIN keyword => retains() true");
        assert!(
            !retain_spec.exhausts(),
            "no EXHAUST keyword => exhausts() false"
        );

        let bare = CardSpec::base();
        assert!(!bare.exhausts());
        assert!(!bare.retains());
    }

    #[test]
    fn keywords_flags_are_disjoint() {
        let all = [
            CardKeywords::UNPLAYABLE,
            CardKeywords::ETHEREAL,
            CardKeywords::RETAIN,
            CardKeywords::EXHAUST,
            CardKeywords::NON_REMOVABLE,
            CardKeywords::NON_UPGRADEABLE,
            CardKeywords::INNATE,
            CardKeywords::ETERNAL,
            CardKeywords::SLY,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                if i != j {
                    assert_eq!(a.0 & b.0, 0, "flags at {i} and {j} overlap");
                }
            }
        }
    }
}
