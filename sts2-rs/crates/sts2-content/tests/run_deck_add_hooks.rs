//! GH #806 (run side): out-of-combat deck adds and non-combat card rewards
//! run the same relic hooks as C#, in both builds.
//!
//! - Every C# deck add goes through `CardPileCmd.Add(card, PileType.Deck)`
//!   (`CardPileCmd.cs:497-636`) or, for a transform, `CardCmd.Transform`
//!   (`CardCmd.cs:427-447`). Both run `Hook.ModifyCardBeingAddedToDeck` (the
//!   eggs, FresnelLens) and then `Hook.AfterCardChangedPiles` (LuckyFysh,
//!   DarkstonePeriapt, BookOfFiveRings, BingBong). Rust used to push most
//!   event and relic adds straight onto `master_deck`, so none of them fired.
//! - Lost Coffer / Orrery / event card grants are real `CardReward`s
//!   (`LostCoffer.cs:20-21`), so their slates run
//!   `Hook.TryModifyCardRewardOptions` (`CardFactory.cs:104`) like a combat
//!   reward. Rust's non-combat roller skipped it, so SilkenTress and
//!   SilverCrucible never saw those slates.

use sts2_core::cards::{CardId, EnchantmentId};
use sts2_core::runstate::{Room, RoomType, RunState};
use sts2_core::{GameVersion, RelicId};

const BOTH_BUILDS: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

fn run_state(version: GameVersion, relics: &[RelicId]) -> RunState {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_core::relic_effects::ensure_installed();
    let mut rs = RunState::new();
    rs.game_version = version;
    rs.character_id = sts2_core::CharacterId::Ironclad;
    rs.relics.extend_from_slice(relics);
    rs
}

fn deck_ids(rs: &RunState) -> Vec<CardId> {
    rs.master_deck.iter().map(|c| c.card_id).collect()
}

/// ByrdonisNest Take: `CardPileCmd.Add(CreateCard<ByrdonisEgg>, Deck)`
/// (`ByrdonisNest.cs`). LuckyFysh pays 15 gold per deck add
/// (`LuckyFysh.cs:24-32`).
#[test]
fn event_card_grant_pays_lucky_fysh() {
    for version in BOTH_BUILDS {
        let mut rs = run_state(version, &[RelicId::LuckyFysh]);
        let gold = rs.gold;
        let take = &sts2_content::events::shared::BYRDONIS_NEST.options[1];
        assert_eq!(take.option_id, "take");
        let _ = (take.handler)(&mut rs);
        assert_eq!(deck_ids(&rs), vec![CardId::ByrdonisEgg], "{version}");
        assert_eq!(rs.gold - gold, 15, "{version}");
    }
}

/// BingBong clones every card that lands in the deck (`BingBong.cs:28-38`);
/// the clone is itself a hooked add (LuckyFysh pays twice) that BingBong
/// skips (`clonedBy == this`). It was wired on the combat side only.
#[test]
fn bing_bong_clones_an_event_card_grant() {
    for version in BOTH_BUILDS {
        let mut rs = run_state(version, &[RelicId::BingBong, RelicId::LuckyFysh]);
        let gold = rs.gold;
        let _ = (sts2_content::events::shared::BYRDONIS_NEST.options[1].handler)(&mut rs);
        assert_eq!(
            deck_ids(&rs),
            vec![CardId::ByrdonisEgg, CardId::ByrdonisEgg],
            "{version}"
        );
        assert_eq!(rs.gold - gold, 30, "{version}");
    }
}

/// UnrestSite Rest: `CardPileCmd.AddCursesToDeck(PoorSleep)`. DarkstonePeriapt
/// gains 6 Max HP whenever a Curse lands in the deck
/// (`DarkstonePeriapt.cs:16-24`).
#[test]
fn event_curse_add_fires_darkstone_periapt() {
    for version in BOTH_BUILDS {
        let mut rs = run_state(version, &[RelicId::DarkstonePeriapt]);
        rs.current_hp = rs.max_hp / 2;
        let max_hp = rs.max_hp;
        let rest = &sts2_content::events::shared_wave_b::UNREST_SITE.options[0];
        assert_eq!(rest.option_id, "rest");
        let _ = (rest.handler)(&mut rs);
        assert_eq!(deck_ids(&rs), vec![CardId::PoorSleep], "{version}");
        assert_eq!(rs.max_hp - max_hp, 6, "{version}");
    }
}

/// v0.111.0 NeowsSacrifice adds Guilty with `CardPileCmd.Add(card, Deck)`
/// (`NeowsSacrifice.cs:30-35`), so DarkstonePeriapt fires on it.
#[test]
fn neows_sacrifice_guilty_fires_darkstone_periapt_v0111() {
    let mut rs = run_state(
        GameVersion::V0_111_0,
        &[RelicId::DarkstonePeriapt, RelicId::NeowsSacrifice],
    );
    let max_hp = rs.max_hp;
    rs.last_obtained_relic = Some(RelicId::NeowsSacrifice);
    let _ = sts2_core::run_effects::dispatch_after_obtained(&mut rs);
    assert_eq!(deck_ids(&rs), vec![CardId::Guilty]);
    assert_eq!(rs.max_hp - max_hp, 6);
}

/// A transform replacement runs `ModifyCardBeingAddedToDeck` (MoltenEgg
/// upgrades an Attack, `MoltenEgg.cs`) and `AfterCardChangedPiles`
/// (LuckyFysh) — `CardCmd.cs:427-447`.
#[test]
fn transform_replacement_runs_the_deck_add_hooks() {
    for version in BOTH_BUILDS {
        let mut rs = run_state(version, &[RelicId::MoltenEgg, RelicId::LuckyFysh]);
        let strike =
            sts2_content::cards::spec_index::card_pod_from_id(CardId::StrikeIronclad, false)
                .expect("strike");
        rs.master_deck.push(strike);
        let gold = rs.gold;
        let bash =
            sts2_content::cards::spec_index::card_pod_from_id(CardId::Bash, false).expect("bash");
        assert_eq!(rs.transform_specific_cards(&[0], vec![bash]), 1);
        assert_eq!(deck_ids(&rs), vec![CardId::Bash], "{version}");
        assert!(rs.master_deck[0].upgraded, "{version}: MoltenEgg upgrades");
        assert_eq!(rs.gold - gold, 15, "{version}");
    }
}

/// FresnelLens enchants only what Nimble can enchant
/// (`FresnelLens.cs:41-53`, `Nimble.cs:7-14`: no Curse, must gain block):
/// now that every add runs the hook, an added curse must stay bare.
#[test]
fn fresnel_lens_leaves_an_added_curse_bare() {
    for version in BOTH_BUILDS {
        let mut rs = run_state(version, &[RelicId::FresnelLens]);
        rs.current_hp = rs.max_hp / 2;
        let _ = (sts2_content::events::shared_wave_b::UNREST_SITE.options[0].handler)(&mut rs);
        assert!(rs.master_deck[0].enchantments.is_empty(), "{version}");
    }
}

/// Lost Coffer's slate: SilkenTress Glams the first card reward it sees
/// (`SilkenTress.cs:47-83`, gated on `IsCardReward`, which every
/// `CardReward` carries, `CardReward.cs:115`).
#[test]
fn noncombat_card_reward_runs_silken_tress() {
    for version in BOTH_BUILDS {
        let mut rs = run_state(version, &[RelicId::SilkenTress]);
        let options = sts2_content::cards::spec_index::roll_noncombat_card_reward_options(&mut rs);
        assert_eq!(options.len(), 3, "{version}");
        assert_eq!(rs.card_reward_enchantments.len(), 3, "{version}");
        assert!(
            rs.card_reward_enchantments
                .iter()
                .any(|e| e.get(EnchantmentId::Glam) == 1),
            "{version}: {:?}",
            rs.card_reward_enchantments
        );
        assert_eq!(
            rs.relic_counters.get(&RelicId::SilkenTress),
            Some(&1),
            "{version}"
        );
    }
}

/// SilverCrucible upgrades every upgradable option of its first 3 card
/// rewards (`SilverCrucible.cs:94-115`); act 1 rolls no upgrade on its own.
#[test]
fn noncombat_card_reward_runs_silver_crucible() {
    for version in BOTH_BUILDS {
        let mut rs = run_state(version, &[RelicId::SilverCrucible]);
        let _ = sts2_content::cards::spec_index::roll_noncombat_card_reward_options(&mut rs);
        assert!(
            rs.card_reward_upgraded_flags.iter().any(|&u| u),
            "{version}: {:?}",
            rs.card_reward_upgraded_flags
        );
        let mut bare = run_state(version, &[]);
        let _ = sts2_content::cards::spec_index::roll_noncombat_card_reward_options(&mut bare);
        assert!(
            bare.card_reward_upgraded_flags.iter().all(|&u| !u),
            "{version}"
        );
    }
}

/// LavaLamp upgrades a card reward only in a combat room
/// (`LavaLamp.cs:66`): an Orrery bought in a shop after a no-damage fight
/// stays bare, a combat room's reward is upgraded.
#[test]
fn lava_lamp_upgrades_only_combat_room_slates() {
    for version in BOTH_BUILDS {
        let mut shop = run_state(version, &[RelicId::LavaLamp]);
        shop.current_room = Some(Room::new(RoomType::Shop));
        let _ = sts2_content::cards::spec_index::roll_noncombat_card_reward_options(&mut shop);
        assert!(
            shop.card_reward_upgraded_flags.iter().all(|&u| !u),
            "{version}"
        );

        let mut fight = run_state(version, &[RelicId::LavaLamp]);
        fight.current_room = Some(Room::new(RoomType::Monster));
        let _ = sts2_content::cards::spec_index::roll_noncombat_card_reward_options(&mut fight);
        assert!(
            fight.card_reward_upgraded_flags.iter().any(|&u| u),
            "{version}"
        );
    }
}
