//! Ironclad starter card data.
//!
//! Bead P4-1 (sts2-rs-zcl) ships pure data — no card-play interpreter,
//! no enchantments, no upgrade machinery. The Phase 6 umbrella bead
//! (sts2-rs-2un) wires `CardSpec.effect` into a runtime executor.
//!
//! ## Starter-deck composition (per Python pin)
//!
//! - `ironclad_starter_deck()` returns **10 cards**: 5×Strike, 4×Defend,
//!   1×Bash. `AscendersBane` is NOT part of the base starter deck —
//!   `sts2_env/run/run_state.py:1163-1176` adds it separately at
//!   ascension level >= 5 (A5+).
//! - `ironclad_starter_deck_with_ascenders_bane()` returns the **11-card**
//!   ascender variant: the 10-card base + 1×`AscendersBane`. This is the
//!   shape referenced in the P4-1 bead description.

use sts2_core::cards::{CardId, CardType, TargetType};
use sts2_core::PowerKind;

use crate::cards::{CardEffect, CardKeywords, CardRarity, CardSpec};

// ---------------------------------------------------------------------------
// Factories — one per distinct card
// ---------------------------------------------------------------------------

/// Strike (Ironclad). Cost 1, Attack, deals 6 damage to a single enemy.
/// Mirrors `make_strike_ironclad` in `ironclad_basic.py:48-57` and
/// `ironclad.py:87-97` (the second is the canonical version with an
/// `upgraded` parameter — bead P4-1 only ships base values).
#[must_use]
pub const fn strike() -> CardSpec {
    CardSpec {
        id: CardId::StrikeIronclad,
        name: "STRIKE_IRONCLAD",
        cost: 1,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        base_damage: 6,
        effect: CardEffect::DealDamage { base: 6 },
        ..CardSpec::base()
    }
}

/// Defend (Ironclad). Cost 1, Skill, gain 5 block on self.
/// Mirrors `make_defend_ironclad` in `ironclad_basic.py:60-69` /
/// `ironclad.py:101-112`.
#[must_use]
pub const fn defend() -> CardSpec {
    CardSpec {
        id: CardId::DefendIronclad,
        name: "DEFEND_IRONCLAD",
        cost: 1,
        card_type: CardType::Skill,
        target_type: TargetType::Self_,
        base_block: 5,
        effect: CardEffect::GainBlock { base: 5 },
        ..CardSpec::base()
    }
}

/// Bash. Cost 2, Attack, deal 8 damage and apply 2 Vulnerable to a single
/// enemy. Mirrors `make_bash` in `ironclad_basic.py:72-82`. The
/// vulnerable amount lives in Python's `effect_vars["vulnerable"] = 2`;
/// in Rust it is encoded directly into the [`CardEffect`] AST so the
/// interpreter reads a single field instead of poking a string-keyed map.
#[must_use]
pub const fn bash() -> CardSpec {
    CardSpec {
        id: CardId::Bash,
        name: "BASH",
        cost: 2,
        card_type: CardType::Attack,
        target_type: TargetType::Enemy,
        base_damage: 8,
        effect: CardEffect::DealDamageAndApplyPower {
            damage: 8,
            kind: PowerKind::Vulnerable,
            amount: 2,
        },
        ..CardSpec::base()
    }
}

/// Ascender's Bane. Cost -1 (unplayable), Curse, no on-play effect,
/// ethereal, eternal, non-removable, non-upgradeable. Mirrors
/// `make_ascenders_bane` in `sts2_env/cards/status.py:160-166` and the
/// deck-add path in `run_state.py:1163-1176`.
///
/// Python encodes the keywords as `frozenset({"unplayable", "ethereal"})`;
/// C# `AscendersBane.cs` lists `CanonicalKeywords = [Eternal, Unplayable,
/// Ethereal]`. Rust packs all three into the [`CardKeywords`] bitfield.
/// `ETERNAL` is the LOAD-BEARING one: `card_is_removable_instance` gates on
/// it and nothing else, exactly as C# `CardModel.IsRemovable` does. An
/// earlier revision of this comment called it "representational only" and
/// pointed at a legacy `CardId::is_removable` — that predicate is gone, and
/// believing it was the gate is what let four curses ship without the
/// keyword (bead `sts2-rs-8ixr6`).
#[must_use]
pub const fn ascenders_bane() -> CardSpec {
    CardSpec {
        id: CardId::AscendersBane,
        name: "ASCENDERS_BANE",
        cost: -1,
        card_type: CardType::Curse,
        target_type: TargetType::None,
        rarity: CardRarity::Curse,
        keywords: CardKeywords::UNPLAYABLE
            .union(CardKeywords::ETHEREAL)
            .union(CardKeywords::ETERNAL)
            .union(CardKeywords::NON_REMOVABLE)
            .union(CardKeywords::NON_UPGRADEABLE),
        effect: CardEffect::Noop,
        ..CardSpec::base()
    }
}

// ---------------------------------------------------------------------------
// Starter-deck constructors
// ---------------------------------------------------------------------------

/// Build the **10-card** Ironclad starter deck (no `AscendersBane`).
///
/// Distribution: 5×Strike, 4×Defend, 1×Bash. Mirrors
/// `create_ironclad_starter_deck` in
/// `sts2-rl-agent@01c094d:sts2_env/cards/ironclad_basic.py:85-93` /
/// `ironclad.py:2152-2160`.
///
/// `AscendersBane` is added by the run-builder (Python:
/// `run_state.py:1163-1176`) only at ascension >= 5; pass through
/// [`ironclad_starter_deck_with_ascenders_bane`] for that variant.
#[must_use]
pub fn ironclad_starter_deck() -> Vec<CardSpec> {
    let mut deck = Vec::with_capacity(10);
    for _ in 0..5 {
        deck.push(strike());
    }
    for _ in 0..4 {
        deck.push(defend());
    }
    deck.push(bash());
    deck
}

/// Build the **11-card** ascender Ironclad starter deck: the 10-card base
/// + 1×`AscendersBane` (matches what the run-builder produces at A5+).
///
/// The `AscendersBane` is appended last to mirror Python's
/// `add_card_instance_to_deck(curse)` call ordering at
/// `run_state.py:1176`, after the base deck has been seeded by
/// `_get_starter_deck`.
#[must_use]
pub fn ironclad_starter_deck_with_ascenders_bane() -> Vec<CardSpec> {
    let mut deck = ironclad_starter_deck();
    deck.push(ascenders_bane());
    deck
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin: base starter deck has exactly 10 cards (Python A0 deck).
    #[test]
    fn ironclad_starter_deck_has_10_cards() {
        let deck = ironclad_starter_deck();
        assert_eq!(deck.len(), 10);
    }

    /// Pin: ascender starter deck has exactly 11 cards (A5+).
    #[test]
    fn ironclad_starter_deck_has_11_cards_with_ascenders_bane() {
        let deck = ironclad_starter_deck_with_ascenders_bane();
        assert_eq!(deck.len(), 11);
    }

    /// Distribution lock: 5/4/1 base, 5/4/1/1 ascender.
    #[test]
    fn ironclad_deck_card_distribution_matches_python() {
        let deck = ironclad_starter_deck();
        let n_strike = deck
            .iter()
            .filter(|c| c.id == CardId::StrikeIronclad)
            .count();
        let n_defend = deck
            .iter()
            .filter(|c| c.id == CardId::DefendIronclad)
            .count();
        let n_bash = deck.iter().filter(|c| c.id == CardId::Bash).count();
        let n_curse = deck
            .iter()
            .filter(|c| c.id == CardId::AscendersBane)
            .count();
        assert_eq!(n_strike, 5, "expected 5 Strike, got {n_strike}");
        assert_eq!(n_defend, 4, "expected 4 Defend, got {n_defend}");
        assert_eq!(n_bash, 1, "expected 1 Bash, got {n_bash}");
        assert_eq!(n_curse, 0, "base deck must not contain AscendersBane");

        let ascender = ironclad_starter_deck_with_ascenders_bane();
        let n_ascender_curse = ascender
            .iter()
            .filter(|c| c.id == CardId::AscendersBane)
            .count();
        assert_eq!(n_ascender_curse, 1, "ascender deck has exactly 1 curse");
    }

    /// Strike spec pins: name, cost=1, type=Attack, target=Enemy,
    /// `base_damage=6`, rarity=Basic, effect=`DealDamage{6}`.
    #[test]
    fn strike_specs_match_python_pin() {
        let s = strike();
        assert_eq!(s.id, CardId::StrikeIronclad);
        assert_eq!(s.name, "STRIKE_IRONCLAD");
        assert_eq!(s.cost, 1);
        assert_eq!(s.card_type, CardType::Attack);
        assert_eq!(s.target_type, TargetType::Enemy);
        assert_eq!(s.base_damage, 6);
        assert_eq!(s.base_block, 0);
        assert_eq!(s.rarity, CardRarity::Basic);
        assert!(s.is_upgradeable());
        assert!(s.is_removable());
        assert!(!s.exhausts());
        assert!(!s.is_ethereal());
        assert!(!s.is_unplayable());
        assert_eq!(s.effect, CardEffect::DealDamage { base: 6 });
    }

    /// Defend spec pins: name, cost=1, type=Skill, target=`Self_`,
    /// `base_block=5`, effect=`GainBlock{5}`.
    #[test]
    fn defend_specs_match_python_pin() {
        let d = defend();
        assert_eq!(d.id, CardId::DefendIronclad);
        assert_eq!(d.name, "DEFEND_IRONCLAD");
        assert_eq!(d.cost, 1);
        assert_eq!(d.card_type, CardType::Skill);
        assert_eq!(d.target_type, TargetType::Self_);
        assert_eq!(d.base_damage, 0);
        assert_eq!(d.base_block, 5);
        assert_eq!(d.rarity, CardRarity::Basic);
        assert!(d.is_upgradeable());
        assert!(d.is_removable());
        assert_eq!(d.effect, CardEffect::GainBlock { base: 5 });
    }

    /// Bash spec pins: name, cost=2, type=Attack, target=Enemy,
    /// `base_damage=8`, effect = `DealDamageAndApplyPower` { damage:8,
    /// kind:Vulnerable, amount:2 }. The Vulnerable amount=2 is the
    /// load-bearing parity claim.
    #[test]
    fn bash_specs_match_python_pin() {
        let b = bash();
        assert_eq!(b.id, CardId::Bash);
        assert_eq!(b.name, "BASH");
        assert_eq!(b.cost, 2);
        assert_eq!(b.card_type, CardType::Attack);
        assert_eq!(b.target_type, TargetType::Enemy);
        assert_eq!(b.base_damage, 8);
        assert_eq!(b.rarity, CardRarity::Basic);
        assert!(b.is_upgradeable());
        assert!(b.is_removable());
        assert_eq!(
            b.effect,
            CardEffect::DealDamageAndApplyPower {
                damage: 8,
                kind: PowerKind::Vulnerable,
                amount: 2,
            }
        );
    }

    /// `AscendersBane` is non-removable, non-upgradeable, ethereal, eternal,
    /// unplayable, cost -1, Curse, Noop. These are the parity-locked
    /// invariants that downstream rest-site / event code keys on.
    /// C# `AscendersBane.cs` `CanonicalKeywords = [Eternal, Unplayable,
    /// Ethereal]`; `ETERNAL` is representational only (bead `sts2-rs-8ixr6`).
    #[test]
    fn ascenders_bane_is_non_removable() {
        let a = ascenders_bane();
        assert_eq!(a.id, CardId::AscendersBane);
        assert_eq!(a.name, "ASCENDERS_BANE");
        assert_eq!(a.cost, -1);
        assert_eq!(a.card_type, CardType::Curse);
        assert_eq!(a.target_type, TargetType::None);
        assert_eq!(a.rarity, CardRarity::Curse);
        assert!(!a.is_removable(), "AscendersBane must be non-removable");
        assert!(!a.is_upgradeable(), "AscendersBane must be non-upgradeable");
        assert!(a.is_ethereal(), "AscendersBane is ethereal");
        assert!(a.is_unplayable(), "AscendersBane is unplayable");
        assert!(
            a.keywords.contains(CardKeywords::ETERNAL),
            "AscendersBane must carry the ETERNAL keyword (C# CanonicalKeywords)"
        );
        assert_eq!(a.effect, CardEffect::Noop);
    }

    /// Determinism / value-stability: factories return the same data on
    /// every call. Important because the engine clones `CardSpec`s into
    /// runtime `Card` POD by value.
    #[test]
    fn factories_are_pure() {
        assert_eq!(strike(), strike());
        assert_eq!(defend(), defend());
        assert_eq!(bash(), bash());
        assert_eq!(ascenders_bane(), ascenders_bane());
    }

    /// `CardSpec` is `Copy`-able when no string allocation is involved
    /// (verified at compile time). This is load-bearing for cheap
    /// instance materialisation in Phase 6.
    #[test]
    fn card_spec_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<CardSpec>();
        assert_copy::<CardEffect>();
        assert_copy::<CardRarity>();
        assert_copy::<CardKeywords>();
    }

    /// Names round-trip in the expected upper-snake-case form so the
    /// parity exporter writes the same strings Python's
    /// `CardId.<v>.name` emits.
    #[test]
    fn card_names_match_python_card_id_names() {
        // Locked: any rename forces a parity exporter audit.
        assert_eq!(strike().name, "STRIKE_IRONCLAD");
        assert_eq!(defend().name, "DEFEND_IRONCLAD");
        assert_eq!(bash().name, "BASH");
        assert_eq!(ascenders_bane().name, "ASCENDERS_BANE");
    }
}
