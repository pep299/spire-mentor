//! A relic bought from the merchant runs its `AfterObtained` screens inside the purchase:
//! C# `MerchantRelicEntry.OnTryPurchase` awaits `RelicCmd.Obtain` before
//! `RestockAfterPurchase` (`MerchantRelicEntry.cs:52-63`).

use sts2_core::cards::{CardType, EnchantmentId};
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{ActionResult, Phase, RunAction, RunManager};
use sts2_core::shop::{ShopInventory, ShopRelicEntry, ShopRelicRarity, SHOP_ENTRY_SOLD_OUT_PRICE};
use sts2_core::{GameVersion, RelicId};

const PRICE: i32 = 150;

/// An interactive host standing in a shop that stocks only `relic`.
fn shop_with(relic: RelicId) -> RunManager {
    sts2_content::relics::dispatch::ensure_installed();
    let mut rm = sts2_content::run_callbacks::new_run_manager_for_character_from_string_seed_for(
        GameVersion::V0_111_0,
        "SHOPRELIC",
        CharacterId::Ironclad,
    );
    rm.run_state.interactive_card_removal = true;
    rm.run_state.interactive_card_upgrade = true;
    rm.run_state.interactive_card_transform = true;
    rm.run_state.direct_event_relic_grant = true;
    rm.run_state.gold = 999;
    let mut inventory = ShopInventory::empty();
    inventory.relics.push(ShopRelicEntry {
        relic_rarity: ShopRelicRarity::Shop,
        relic_id: Some(relic),
        price: PRICE,
    });
    rm.debug_open_shop(inventory);
    rm
}

fn buy(rm: &mut RunManager) {
    assert_eq!(
        rm.take_action(RunAction::ShopBuyRelic { idx: 0 }),
        ActionResult::Advanced
    );
}

fn enchanted(rm: &RunManager, ench: EnchantmentId) -> usize {
    rm.run_state
        .master_deck
        .iter()
        .filter(|c| c.enchantment(ench) > 0)
        .count()
}

/// `DollysMirror.AfterObtained` (`DollysMirror.cs:16-24`) is a
/// `FromDeckGeneric(prefs 1, Type != Quest)` pick: the player chooses the card
/// and nothing is rolled.
#[test]
fn dollys_mirror_opens_a_deck_pick_and_duplicates_the_chosen_card() {
    let mut rm = shop_with(RelicId::DollysMirror);
    let deck = rm.run_state.master_deck.clone();
    let rewards_before = rm.run_state.rng.rewards.clone();
    buy(&mut rm);
    let actions = rm.get_available_actions();
    assert_eq!(actions.len(), deck.len(), "one pick per deck card, no skip");
    assert!(actions
        .iter()
        .all(|a| matches!(a, RunAction::GrantDeckCard { .. })));
    // Bash sits at the end of the Ironclad starter deck.
    let pick = deck.len() - 1;
    assert_eq!(
        rm.take_action(RunAction::GrantDeckCard { idx: pick }),
        ActionResult::Advanced
    );
    assert_eq!(rm.run_state.master_deck.len(), deck.len() + 1);
    assert_eq!(
        rm.run_state.master_deck.last().map(|c| c.card_id),
        Some(deck[pick].card_id)
    );
    assert!(rm.run_state.pending_card_grants.is_empty());
    assert_eq!(rm.phase(), Phase::Shop);
    assert_eq!(rm.run_state.rng.rewards, rewards_before);
}

/// Gnarled Hammer's select is `CardSelectorPrefs(prompt, 0, 3)`
/// (`GnarledHammer.cs:28-40`). `FromDeckForEnchantment` auto-takes only when
/// `cards.Count <= MinSelect` (`CardSelectCmd.cs:691`), so a pool of two
/// Attacks still opens the screen, and the player may take none.
#[test]
fn gnarled_hammer_small_pool_still_opens_an_optional_pick() {
    let mut rm = shop_with(RelicId::GnarledHammer);
    let mut attacks = 0;
    rm.run_state.master_deck.retain(|c| {
        if c.card_type != CardType::Attack {
            return true;
        }
        attacks += 1;
        attacks <= 2
    });
    buy(&mut rm);
    assert_eq!(rm.enchant_eligible_deck_indices().len(), 2);
    assert_eq!(rm.run_state.pending_deck_enchants.len(), 2);
    assert!(rm
        .run_state
        .pending_deck_enchants
        .iter()
        .all(|e| e.optional));
    assert_eq!(enchanted(&rm, EnchantmentId::Sharp), 0);
    assert_eq!(rm.decline_optional_deck_enchants(), 2);
    assert_eq!(enchanted(&rm, EnchantmentId::Sharp), 0);
    assert_eq!(rm.phase(), Phase::Shop);
}

/// TheCourier restocks the bought slot only after all five Orrery card rewards
/// have rolled: C# rolls them inside `RelicCmd.Obtain`, then
/// `RestockAfterPurchase` reads `Rewards` in `RelicFactory.RollRarity`. The same
/// purchase without the Courier reads the identical card rolls; only the last
/// pick, which returns to the shop, adds the restock reads.
#[test]
fn courier_restock_follows_the_orrery_card_rewards() {
    let mut plain = shop_with(RelicId::Orrery);
    let mut courier = shop_with(RelicId::Orrery);
    courier.run_state.relics.push(RelicId::TheCourier);
    let sold_out =
        |rm: &RunManager| rm.shop_inventory().unwrap().relics[0].price == SHOP_ENTRY_SOLD_OUT_PRICE;
    buy(&mut plain);
    buy(&mut courier);
    for n in 0..5 {
        assert_eq!(courier.phase(), Phase::CardReward, "card reward {n}");
        assert!(
            sold_out(&courier),
            "no restock while rewards are open ({n})"
        );
        assert_eq!(courier.card_reward_options(), plain.card_reward_options());
        assert_eq!(
            courier.run_state.rng.rewards, plain.run_state.rng.rewards,
            "card reward {n} rolled before any restock read"
        );
        for rm in [&mut plain, &mut courier] {
            assert_eq!(
                rm.take_action(RunAction::PickCardReward { idx: None }),
                ActionResult::Advanced
            );
        }
    }
    assert_eq!(courier.phase(), Phase::Shop);
    assert!(sold_out(&plain));
    assert!(
        !sold_out(&courier),
        "the Courier restocked the slot on return"
    );
    assert_ne!(courier.run_state.rng.rewards, plain.run_state.rng.rewards);
}
