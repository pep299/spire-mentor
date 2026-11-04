//! Pael's Growth's rest-site CLONE option (both builds run the same C#).
//!
//! C# `PaelsGrowth.TryModifyRestSiteOptions` (`Relics/PaelsGrowth.cs:31-39`,
//! v0.107.1 and v0.111.0) appends a `CloneRestSiteOption` for its owner.
//! `CloneRestSiteOption.OnSelect` (`Entities.RestSite/CloneRestSiteOption.cs:41-52`)
//! snapshots every deck card whose enchantment is `Clone`, then for each one
//! `RunState.CloneCard(item)` (`ClonePreservingMutability`, so the copy keeps
//! its upgrade and the Clone enchantment) and `CardPileCmd.Add(card,
//! PileType.Deck)` — one add per copy, with the deck-add hooks.
//!
//! Evidence: deep act23 probe, Ironclad seed 1019 — C# offered
//! `[HEAL, SMITH, LIFT, CLONE]`, Rust rejected option 3.

use sts2_core::cards::{Card, CardId, EnchantmentId};
use sts2_core::runstate::{RestSiteActionOutcome, RestSiteOption, RunState};
use sts2_core::{GameVersion, RelicId};

const VERSIONS: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

fn card(id: CardId) -> Card {
    let mut c = Card::pod_defaults();
    c.card_id = id;
    c
}

fn run_state(version: GameVersion, relics: &[RelicId]) -> RunState {
    sts2_content::relics::dispatch::ensure_installed();
    let mut rs = RunState::new();
    rs.game_version = version;
    rs.relics.extend_from_slice(relics);
    rs
}

#[test]
fn paels_growth_offers_clone_after_earlier_relic_options() {
    for version in VERSIONS {
        let mut rs = run_state(version, &[RelicId::Girya, RelicId::PaelsGrowth]);
        rs.master_deck.push(card(CardId::Bash));
        let ids: Vec<&str> = rs
            .rest_site_options()
            .iter()
            .map(RestSiteOption::option_id)
            .collect();
        assert_eq!(ids, ["HEAL", "SMITH", "LIFT", "CLONE"], "{version:?}");
    }
}

#[test]
fn no_clone_option_without_paels_growth() {
    for version in VERSIONS {
        let rs = run_state(version, &[RelicId::Girya]);
        assert!(
            !rs.rest_site_options()
                .iter()
                .any(|o| matches!(o, RestSiteOption::Clone)),
            "{version:?}"
        );
    }
}

#[test]
fn clone_duplicates_every_clone_enchanted_deck_card() {
    for version in VERSIONS {
        let mut rs = run_state(version, &[RelicId::PaelsGrowth]);
        let mut bash = card(CardId::Bash);
        bash.set_enchantment(EnchantmentId::Clone, 4);
        bash.upgraded = true;
        let mut anger = card(CardId::Anger);
        anger.set_enchantment(EnchantmentId::Clone, 4);
        rs.master_deck
            .extend([card(CardId::StrikeIronclad), bash, anger]);

        let outcome = rs.do_rest_site_action(RestSiteOption::Clone);
        assert_eq!(outcome, RestSiteActionOutcome::Cloned, "{version:?}");

        let ids: Vec<CardId> = rs.master_deck.iter().map(|c| c.card_id).collect();
        assert_eq!(
            ids,
            [
                CardId::StrikeIronclad,
                CardId::Bash,
                CardId::Anger,
                CardId::Bash,
                CardId::Anger
            ],
            "{version:?}"
        );
        // `ClonePreservingMutability`: the copies keep upgrade + enchantment.
        let copy = rs.master_deck[3];
        assert!(copy.upgraded, "{version:?}");
        assert_eq!(copy.enchantment(EnchantmentId::Clone), 4, "{version:?}");
        assert_eq!(
            rs.master_deck[4].enchantment(EnchantmentId::Clone),
            4,
            "{version:?}"
        );
    }
}

#[test]
fn clone_with_no_clone_cards_succeeds_and_adds_nothing() {
    for version in VERSIONS {
        let mut rs = run_state(version, &[RelicId::PaelsGrowth]);
        rs.master_deck.push(card(CardId::Bash));
        let outcome = rs.do_rest_site_action(RestSiteOption::Clone);
        assert_eq!(outcome, RestSiteActionOutcome::Cloned, "{version:?}");
        assert_eq!(rs.master_deck.len(), 1, "{version:?}");
    }
}
