//! Rest-site options added by relics follow the order the relics were
//! obtained in. C# `Hook.ModifyRestSiteOptions` (Hook.cs:2093 v0.107.1,
//! Hook.cs:2102 v0.111.0) walks `RunState.IterateHookListeners`, which
//! visits the player's relics in acquisition order, so a Girya obtained
//! before a Shovel puts Lift ahead of Dig.

use sts2_core::cards::{Card, CardId};
use sts2_core::runstate::{RestSiteOption, RunState};
use sts2_core::{GameVersion, RelicId};

fn relic_options(relics: &[RelicId], version: GameVersion) -> Vec<&'static str> {
    sts2_content::relics::dispatch::ensure_installed();
    let mut rs = RunState::new();
    rs.game_version = version;
    rs.relics.extend_from_slice(relics);
    rs.rest_site_options()
        .into_iter()
        .filter_map(|o| match o {
            RestSiteOption::Dig => Some("dig"),
            RestSiteOption::Lift { .. } => Some("lift"),
            _ => None,
        })
        .collect()
}

#[test]
fn relic_rest_options_follow_acquisition_order() {
    for version in [GameVersion::V0_107_1, GameVersion::V0_111_0] {
        assert_eq!(
            relic_options(&[RelicId::Shovel, RelicId::Girya], version),
            ["dig", "lift"],
            "{version:?}"
        );
        assert_eq!(
            relic_options(&[RelicId::Girya, RelicId::Shovel], version),
            ["lift", "dig"],
            "{version:?}"
        );
    }
}

/// The Byrdonis Egg's Hatch option comes BEFORE every relic-added option.
/// C# `RestSiteOption.Generate` (RestSiteOption.cs:53-74 v0.111.0) builds
/// [Heal, Smith] and then calls `Hook.ModifyRestSiteOptions`, whose
/// `RunState.IterateHookListeners` (RunState.cs:123-156, identical in
/// v0.107.1 and v0.111.0) queues every deck card (and its enchantment)
/// first, then relics, potions and modifiers. `ByrdonisEgg` is a deck card
/// (`ByrdonisEgg.TryModifyRestSiteOptions`, ByrdonisEgg.cs:19-27), so its
/// Hatch lands ahead of Girya's Lift and Shovel's Dig. The option index is
/// what the policy's action mapping selects, so the order is load-bearing.
#[test]
fn hatch_precedes_relic_rest_options() {
    sts2_content::relics::dispatch::ensure_installed();
    for version in [GameVersion::V0_107_1, GameVersion::V0_111_0] {
        let mut rs = RunState::new();
        rs.game_version = version;
        rs.relics
            .extend_from_slice(&[RelicId::Girya, RelicId::Shovel]);
        let mut egg = Card::pod_defaults();
        egg.card_id = CardId::ByrdonisEgg;
        rs.master_deck.push(egg);
        let names: Vec<&str> = rs
            .rest_site_options()
            .into_iter()
            .map(|o| match o {
                RestSiteOption::Rest => "rest",
                RestSiteOption::Smith { .. } => "smith",
                RestSiteOption::Hatch => "hatch",
                RestSiteOption::Lift { .. } => "lift",
                RestSiteOption::Dig => "dig",
                _ => "other",
            })
            .collect();
        assert_eq!(
            names,
            ["rest", "smith", "hatch", "lift", "dig"],
            "{version:?}"
        );
    }
}
