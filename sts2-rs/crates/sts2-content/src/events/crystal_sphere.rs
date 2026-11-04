//! The CrystalSphere divination minigame (GH #817).
//!
//! C# `CrystalSphereMinigame` (`Events.Custom.CrystalSphereEvent/
//! CrystalSphereMinigame.cs`) and its items (`…CrystalSphereItems/*.cs`).
//! The decomp of every file here is identical between v0.107.1 and v0.111.0
//! apart from decompiler noise and an `Array.Empty` overload, so there is no
//! version branch.
//!
//! An 11x11 fogged grid hides fifteen items. Each divination clears either a
//! single cell (Small tool) or a 3x3 block (Big tool); an item is revealed
//! once every cell it covers is clear. When the divinations run out the
//! revealed items become rewards (`OneOffSynchronizer.OfferCrystalSphereRewards`).
//! Everything — placement, the rewards' rolls — runs on the event's own Rng,
//! in the order below.
//!
//! **Choices.** Live, a human picks the tool and the cells. The C# parity
//! oracle has no screen to click, so it plays a fixed policy
//! (`RunSimulator.PlayCrystalSphereHeadless`: Big tool, then the centres in
//! [`HEADLESS_SWEEP_CENTERS`]) and presents no decision. The harness drives
//! the Rust engine off the oracle's choices, so the Rust event plays the same
//! policy and offers no per-cell choice either; [`CrystalSphereMinigame`]
//! itself takes any tool and cell, so an interactive host can drive it.

use sts2_core::cards::{Card, CardId};
use sts2_core::dotnet_rng::Rng;
use sts2_core::{PotionKind, PotionRarity, RelicId, RunState};

use crate::cards::CardRarity;

/// `_defaultWidth` / `_defaultHeight` (`CrystalSphereMinigame.cs:28-30`).
pub const GRID_SIZE: i32 = 11;

/// `PopulateItems` retry cap (`CrystalSphereMinigame.cs:139-145`).
const MAX_POPULATE_ATTEMPTS: u32 = 10;

/// `CrystalSphereCardReward.ToReward`: `new CardReward(options, 3, owner)`.
const CARD_REWARD_OPTIONS: usize = 3;

/// The oracle's headless cell policy (`RunSimulator.cs`
/// `CrystalSphereSweepCenters`): Big tool, two rows of three 3x3 blocks
/// covering x 0..8, y 2..7. Three divinations use the first three, six use
/// all of them.
pub const HEADLESS_SWEEP_CENTERS: [(i32, i32); 6] =
    [(1, 3), (4, 3), (7, 3), (1, 6), (4, 6), (7, 6)];

/// `CrystalSphereToolType` (`CrystalSphereMinigame.cs:22-27`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrystalSphereTool {
    /// Clears the clicked cell only.
    Small,
    /// Clears the clicked cell and its eight neighbours. The default.
    Big,
}

/// The five item classes and their constructor arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrystalSphereItemKind {
    /// `CrystalSphereRelic`: 4x4, a `RelicReward` on the event Rng.
    Relic,
    /// `CrystalSpherePotion(rarity)`: Common 1x3 (1 wide, 3 tall), Rare 2x2.
    Potion(PotionRarity),
    /// `CrystalSphereCardReward(rarity)`: 2x2, a uniform card reward of
    /// that rarity from the owner's pool.
    CardReward(CardRarity),
    /// `CrystalSphereCurse`: 2x2, adds a Doubt the moment it is revealed.
    Curse,
    /// `CrystalSphereGold(isBig)`: small 1x1 = 10 gold, big 2x1 = 30 gold.
    Gold {
        /// `_isBig`.
        big: bool,
    },
}

impl CrystalSphereItemKind {
    /// `Size` as `(X, Y)`: width along x, height along y.
    #[must_use]
    pub const fn size(self) -> (i32, i32) {
        match self {
            Self::Relic => (4, 4),
            Self::Potion(PotionRarity::Rare) | Self::CardReward(_) | Self::Curse => (2, 2),
            Self::Potion(_) => (1, 3),
            Self::Gold { big: true } => (2, 1),
            Self::Gold { big: false } => (1, 1),
        }
    }
}

/// One `CrystalSphereItem` on (or meant for) the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrystalSphereItem {
    /// Which item.
    pub kind: CrystalSphereItemKind,
    /// `Position`: the top-left cell. `(0, 0)` for an item that never got
    /// placed (C#'s default `Vector2I`).
    pub position: (i32, i32),
    /// How many times `PopulateItems` subscribed `OnItemRevealed` to this
    /// item. One per populate attempt it survived, so an item from a failed
    /// first attempt lands in the revealed list once per attempt.
    subscriptions: u8,
}

/// C# `CrystalSphereMinigame`'s game state.
#[derive(Clone, Debug)]
pub struct CrystalSphereMinigame {
    /// `cells[x, y].IsHidden`, indexed `[x][y]`.
    hidden: [[bool; GRID_SIZE as usize]; GRID_SIZE as usize],
    /// `cells[x, y].Item`, as an index into [`Self::items`].
    item_at: [[Option<u8>; GRID_SIZE as usize]; GRID_SIZE as usize],
    /// `_items`, in `PopulateItems` order.
    items: Vec<CrystalSphereItem>,
    /// `_revealed`, in reveal order (with the subscription multiplicity).
    revealed: Vec<CrystalSphereItemKind>,
    /// `DivinationCount`.
    divinations: u32,
    /// `CrystalSphereTool`.
    tool: CrystalSphereTool,
    /// `PlacedAllItems`.
    placed_all_items: bool,
}

impl CrystalSphereMinigame {
    /// The constructor (`CrystalSphereMinigame.cs:95-148`). Clears the
    /// Manhattan-radius-2 diamond at each corner (before any item exists, so
    /// nothing is revealed), places the fifteen items with one `NextItem`
    /// each, retrying up to ten times, and selects the Big tool.
    #[must_use]
    pub fn new(rng: &mut Rng, divination_count: u32) -> Self {
        let mut game = Self {
            hidden: [[true; GRID_SIZE as usize]; GRID_SIZE as usize],
            item_at: [[None; GRID_SIZE as usize]; GRID_SIZE as usize],
            items: Vec::new(),
            revealed: Vec::new(),
            divinations: 0,
            tool: CrystalSphereTool::Big,
            placed_all_items: false,
        };
        let last = GRID_SIZE - 1;
        let mut cleared: Vec<(i32, i32)> = vec![(0, 0), (last, 0), (last, last), (0, last)];
        for _ in 0..2 {
            let horizontal: Vec<(i32, i32)> = cleared
                .iter()
                .flat_map(|&(x, y)| horizontal_cells(x, y))
                .collect();
            let vertical: Vec<(i32, i32)> = cleared
                .iter()
                .flat_map(|&(x, y)| vertical_cells(x, y))
                .collect();
            cleared.extend(horizontal);
            cleared.extend(vertical);
        }
        for (x, y) in cleared {
            game.hidden[x as usize][y as usize] = false;
        }
        let mut attempts = 0;
        loop {
            game.placed_all_items = game.populate_items(rng);
            attempts += 1;
            if game.placed_all_items || attempts >= MAX_POPULATE_ATTEMPTS {
                break;
            }
        }
        game.divinations = divination_count;
        game
    }

    /// `PopulateItems` (`CrystalSphereMinigame.cs:167-216`). Appends to the
    /// item list without clearing it, and `flag && PlaceItem` stops placing
    /// after the first failure while still adding the rest — both exactly as
    /// the C# does on a retry.
    fn populate_items(&mut self, rng: &mut Rng) -> bool {
        use CrystalSphereItemKind as K;
        let mut kinds = vec![
            K::Relic,
            K::Potion(PotionRarity::Common),
            K::Potion(PotionRarity::Common),
            K::Potion(PotionRarity::Rare),
            K::CardReward(CardRarity::Common),
            K::CardReward(CardRarity::Uncommon),
            K::CardReward(CardRarity::Rare),
            K::Curse,
        ];
        kinds.extend([K::Gold { big: false }; 5]);
        kinds.extend([K::Gold { big: true }; 2]);
        let mut flag = true;
        for kind in kinds {
            let mut item = CrystalSphereItem {
                kind,
                position: (0, 0),
                subscriptions: 0,
            };
            let index = self.items.len();
            flag = flag && self.place_item(rng, &mut item, index);
            self.items.push(item);
        }
        for item in &mut self.items {
            item.subscriptions = item.subscriptions.saturating_add(1);
        }
        flag
    }

    /// `CrystalSphereItem.PlaceItem` (`CrystalSphereItem.cs:31-56`): every
    /// legal top-left cell, x-major then y, and one `NextItem` among them.
    fn place_item(&mut self, rng: &mut Rng, item: &mut CrystalSphereItem, index: usize) -> bool {
        let (w, h) = item.kind.size();
        let mut candidates: Vec<(i32, i32)> = Vec::new();
        for x in 0..GRID_SIZE {
            for y in 0..GRID_SIZE {
                if self.can_place_here(w, h, x, y) {
                    candidates.push((x, y));
                }
            }
        }
        if candidates.is_empty() {
            return false;
        }
        let pos = candidates[rng.next_int_exclusive(0, candidates.len() as i32) as usize];
        item.position = pos;
        let slot = u8::try_from(index).expect("item index fits u8");
        for dx in 0..w {
            for dy in 0..h {
                self.item_at[(pos.0 + dx) as usize][(pos.1 + dy) as usize] = Some(slot);
            }
        }
        true
    }

    /// `CanPlaceHere` (`CrystalSphereItem.cs:65-92`): in bounds, still
    /// fogged and empty.
    fn can_place_here(&self, w: i32, h: i32, x: i32, y: i32) -> bool {
        for dx in 0..w {
            for dy in 0..h {
                let (cx, cy) = (x + dx, y + dy);
                if !(0..GRID_SIZE).contains(&cx) || !(0..GRID_SIZE).contains(&cy) {
                    return false;
                }
                if !self.hidden[cx as usize][cy as usize] {
                    return false;
                }
                if self.item_at[cx as usize][cy as usize].is_some() {
                    return false;
                }
            }
        }
        true
    }

    /// `SetTool`.
    pub fn set_tool(&mut self, tool: CrystalSphereTool) {
        self.tool = tool;
    }

    /// `CellClicked` (`CrystalSphereMinigame.cs:268-288`): spend one
    /// divination and clear the cell (Small) or its 3x3 block (Big, in
    /// `GetAdjacentCells` order). Returns the items revealed by this click,
    /// in reveal order, so the caller can apply `RevealItem` side effects
    /// (the Curse's Doubt).
    pub fn cell_clicked(&mut self, x: i32, y: i32) -> Vec<CrystalSphereItemKind> {
        self.divinations = self.divinations.saturating_sub(1);
        let cells = match self.tool {
            CrystalSphereTool::Small => vec![(x, y)],
            CrystalSphereTool::Big => adjacent_cells(x, y),
        };
        let mut newly = Vec::new();
        for (cx, cy) in cells {
            if let Some(kind) = self.clear_cell(cx, cy) {
                newly.push(kind);
            }
        }
        newly
    }

    /// `ClearCell` (`CrystalSphereMinigame.cs:296-317`). Returns the item
    /// this clear revealed, if any.
    fn clear_cell(&mut self, x: i32, y: i32) -> Option<CrystalSphereItemKind> {
        assert!(
            (0..GRID_SIZE).contains(&x) && (0..GRID_SIZE).contains(&y),
            "[{x},{y}] is not a valid cell on this grid"
        );
        let (ux, uy) = (x as usize, y as usize);
        if !self.hidden[ux][uy] {
            return None;
        }
        self.hidden[ux][uy] = false;
        let index = usize::from(self.item_at[ux][uy]?);
        let item = self.items[index];
        let (w, h) = item.kind.size();
        for dx in 0..w {
            for dy in 0..h {
                if self.hidden[(item.position.0 + dx) as usize][(item.position.1 + dy) as usize] {
                    return None;
                }
            }
        }
        for _ in 0..item.subscriptions {
            self.revealed.push(item.kind);
        }
        Some(item.kind)
    }

    /// `DivinationCount`.
    #[must_use]
    pub const fn divinations(&self) -> u32 {
        self.divinations
    }

    /// `IsFinished`.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.divinations == 0
    }

    /// `PlacedAllItems`.
    #[must_use]
    pub const fn placed_all_items(&self) -> bool {
        self.placed_all_items
    }

    /// `Items`.
    #[must_use]
    pub fn items(&self) -> &[CrystalSphereItem] {
        &self.items
    }

    /// `_revealed`: what the rewards are built from.
    #[must_use]
    pub fn revealed(&self) -> &[CrystalSphereItemKind] {
        &self.revealed
    }

    /// `cells[x, y].IsHidden`.
    #[must_use]
    pub fn is_hidden(&self, x: i32, y: i32) -> bool {
        self.hidden[x as usize][y as usize]
    }
}

/// `GetHorizontalCells`: x-1 then x+1, in bounds.
fn horizontal_cells(x: i32, y: i32) -> Vec<(i32, i32)> {
    [-1, 1]
        .into_iter()
        .map(|d| (x + d, y))
        .filter(|&(nx, _)| (0..GRID_SIZE).contains(&nx))
        .collect()
}

/// `GetVerticalCells`: y-1 then y+1, in bounds.
fn vertical_cells(x: i32, y: i32) -> Vec<(i32, i32)> {
    [-1, 1]
        .into_iter()
        .map(|d| (x, y + d))
        .filter(|&(_, ny)| (0..GRID_SIZE).contains(&ny))
        .collect()
}

/// `GetAdjacentCells`: horizontal, vertical, the four diagonals (x-major),
/// then the cell itself.
fn adjacent_cells(x: i32, y: i32) -> Vec<(i32, i32)> {
    let mut out = horizontal_cells(x, y);
    out.extend(vertical_cells(x, y));
    for dx in [-1, 1] {
        for dy in [-1, 1] {
            let (nx, ny) = (x + dx, y + dy);
            if (0..GRID_SIZE).contains(&nx) && (0..GRID_SIZE).contains(&ny) {
                out.push((nx, ny));
            }
        }
    }
    out.push((x, y));
    out
}

/// Add a curse to the deck through `CardPileCmd.Add` (the add hooks:
/// DarkstonePeriapt and the like).
pub(super) fn add_curse_to_deck(state: &mut RunState, curse: CardId) {
    if let Some(card) = crate::cards::spec_index::card_pod_from_id(curse, false) {
        state.add_card_to_deck_with_hooks(card, sts2_core::relic_effects::AddSource::EventGain);
    }
}

/// Play the minigame with the oracle's headless policy, then grant the
/// rewards. `rng` is the event Rng, already past `CalculateVars`.
///
/// `UncoverFuture` / `PaymentPlan` (`CrystalSphere.cs:66-80`) build the
/// minigame after paying, and `PlayMinigame` → `CompleteMinigame` →
/// `OneOffSynchronizer.OfferCrystalSphereRewards` turns `_revealed` into the
/// reward set.
pub(super) fn play_headless_and_offer(state: &mut RunState, rng: &mut Rng, divinations: u32) {
    let mut game = CrystalSphereMinigame::new(rng, divinations);
    game.set_tool(CrystalSphereTool::Big);
    for (cx, cy) in HEADLESS_SWEEP_CENTERS {
        if game.divinations() == 0 {
            break;
        }
        for kind in game.cell_clicked(cx, cy) {
            // `CrystalSphereCurse.RevealItem`: `AddCurseToDeck<Doubt>` on
            // reveal, not as a reward.
            if kind == CrystalSphereItemKind::Curse {
                add_curse_to_deck(state, CardId::Doubt);
            }
        }
    }
    offer_rewards(state, rng, game.revealed());
}

/// A reward between `ToReward` and selection.
enum SphereReward {
    Gold(i32),
    Potion(Option<PotionKind>),
    Relic(Option<RelicId>),
    Cards(Vec<Card>),
}

impl SphereReward {
    /// `Reward.RewardsSetIndex`: Gold 1, Potion 2, Relic 3, Card 5.
    const fn rewards_set_index(&self) -> u8 {
        match self {
            Self::Gold(_) => 1,
            Self::Potion(_) => 2,
            Self::Relic(_) => 3,
            Self::Cards(_) => 5,
        }
    }
}

/// The order `RewardsSet.GenerateWithoutOffering` leaves the set in:
/// `Rewards.Sort` by `RewardsSetIndex` (RewardsSet.cs:148, both builds).
/// `List<T>.Sort` is .NET's introsort, which is not stable: a 3-element list
/// runs a swap network, so two equal-index rewards ahead of a lower one swap
/// places. Returns the indices into `keys` in sorted order.
fn rewards_set_order(keys: &[u8]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    sts2_core::cards::dotnet_list_sort(idx.as_mut_slice(), &|a: usize, b: usize| {
        keys[a].cmp(&keys[b])
    });
    idx
}

/// `OfferCrystalSphereRewards` + `RewardsCmd.OfferCustom`, on the event Rng:
///
/// 1. `ToReward` per revealed item, in reveal order. Only the potion rolls
///    here (one `NextItem` over the rarity-filtered potion pool); the curse
///    gives no reward.
/// 2. `RewardsSet.GenerateWithoutOffering` calls `Populate` on every reward,
///    in the same order: gold draws `NextInt(amount, amount + 1)` (the fixed
///    amount, but one draw), the relic rolls its rarity and pulls from the
///    front of the grab bag, a card reward rolls its three cards. The potion
///    is already populated.
/// 3. The set is sorted by `RewardsSetIndex` with .NET's unstable
///    `List.Sort` ([`rewards_set_order`]) and the headless auto-taker
///    selects in that order: gold, potions, relic, then each card reward as
///    a decision.
fn offer_rewards(state: &mut RunState, rng: &mut Rng, revealed: &[CrystalSphereItemKind]) {
    use CrystalSphereItemKind as K;
    let character = state.character_id;
    let mut rewards: Vec<(SphereReward, Option<CardRarity>)> = Vec::new();
    for &kind in revealed {
        match kind {
            K::Gold { big } => rewards.push((SphereReward::Gold(if big { 30 } else { 10 }), None)),
            K::Potion(rarity) => {
                let potion = crate::potions::all::roll_event_potion_full_pool_of_rarity(
                    rng, character, rarity,
                );
                rewards.push((SphereReward::Potion(potion), None));
            }
            K::Relic => rewards.push((SphereReward::Relic(None), None)),
            K::CardReward(rarity) => rewards.push((SphereReward::Cards(Vec::new()), Some(rarity))),
            K::Curse => {}
        }
    }
    for (reward, card_rarity) in &mut rewards {
        match reward {
            SphereReward::Gold(amount) => {
                *amount = rng.next_int_exclusive(*amount, *amount + 1);
            }
            SphereReward::Potion(_) => {}
            SphereReward::Relic(relic) => {
                let rarity = sts2_core::runstate::roll_relic_rarity(rng);
                *relic = state.pop_relic_from_grab_bag(rarity);
            }
            SphereReward::Cards(cards) => {
                let rarity = card_rarity.expect("card reward carries its rarity");
                let rolled = crate::cards::spec_index::roll_uniform_rarity_reward_cards(
                    state,
                    character,
                    rarity,
                    CARD_REWARD_OPTIONS,
                    true,
                    Some(rng),
                );
                *cards = state.offer_card_slate(rolled, true);
            }
        }
    }
    let keys: Vec<u8> = rewards
        .iter()
        .map(|(reward, _)| reward.rewards_set_index())
        .collect();
    let mut slots: Vec<Option<SphereReward>> = rewards
        .into_iter()
        .map(|(reward, _)| Some(reward))
        .collect();
    for i in rewards_set_order(&keys) {
        let Some(reward) = slots[i].take() else {
            continue;
        };
        match reward {
            SphereReward::Gold(amount) => state.gain_gold(amount),
            SphereReward::Potion(Some(kind)) => {
                let _ = state.claim_potion_reward(kind);
            }
            SphereReward::Relic(Some(id)) => state.obtain_relic(id),
            // An empty slate would read as an unrolled sentinel to the drain;
            // the owner's pool always has cards of each rarity.
            SphereReward::Cards(cards) if !cards.is_empty() => state.pending_rewards.push(
                sts2_core::Reward::Card(sts2_core::CardReward::from_cards(&cards)),
            ),
            SphereReward::Cards(_) | SphereReward::Potion(None) | SphereReward::Relic(None) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// .NET `List.Sort` on 3 items is a swap network (unstable): two
    /// equal-index rewards ahead of a lower one come out swapped. Larger
    /// lists (insertion sort) and distinct keys keep the stable order.
    #[test]
    fn rewards_set_order_replays_the_dotnet_sort() {
        assert_eq!(rewards_set_order(&[5, 5, 1]), vec![2, 1, 0]);
        assert_eq!(rewards_set_order(&[5, 1, 3]), vec![1, 2, 0]);
        assert_eq!(rewards_set_order(&[5, 5, 1, 2]), vec![2, 3, 0, 1]);
    }

    fn rng(seed: u64) -> Rng {
        Rng::from_seed(seed)
    }

    /// Every cell of the grid, x-major.
    fn all_cells() -> impl Iterator<Item = (i32, i32)> {
        (0..GRID_SIZE).flat_map(|x| (0..GRID_SIZE).map(move |y| (x, y)))
    }

    /// The constructor pre-clears the Manhattan-radius-2 diamond at each
    /// corner (6 cells each), then places all fifteen items on fogged,
    /// non-overlapping cells: 16 + 3 + 3 + 4 + 4 + 4 + 4 + 4 + 5 + 4 = 51.
    #[test]
    fn constructor_clears_corner_diamonds_and_places_fifteen_items() {
        for seed in 0..200 {
            let game = CrystalSphereMinigame::new(&mut rng(seed), 3);
            let cleared: Vec<(i32, i32)> = all_cells()
                .filter(|&(x, y)| !game.is_hidden(x, y))
                .collect();
            assert_eq!(cleared.len(), 24, "seed {seed}");
            for (x, y) in cleared {
                let dx = x.min(GRID_SIZE - 1 - x);
                let dy = y.min(GRID_SIZE - 1 - y);
                assert!(dx + dy <= 2, "({x},{y}) is not in a corner diamond");
            }
            assert!(game.placed_all_items(), "seed {seed}");
            assert_eq!(game.items().len(), 15);
            let mut covered = std::collections::BTreeSet::new();
            for item in game.items() {
                let (w, h) = item.kind.size();
                for dx in 0..w {
                    for dy in 0..h {
                        let cell = (item.position.0 + dx, item.position.1 + dy);
                        assert!(game.is_hidden(cell.0, cell.1), "item on a cleared cell");
                        assert!(covered.insert(cell), "items overlap at {cell:?}");
                    }
                }
            }
            assert_eq!(covered.len(), 51);
            assert_eq!(game.divinations(), 3);
            assert!(game.revealed().is_empty());
        }
    }

    /// `PopulateItems` order, and one `NextItem` per item over the legal
    /// top-left cells enumerated x-major then y.
    #[test]
    fn placement_is_one_next_item_per_item_in_populate_order() {
        use CrystalSphereItemKind as K;
        let mut a = rng(42);
        let game = CrystalSphereMinigame::new(&mut a, 6);
        let kinds: Vec<K> = game.items().iter().map(|i| i.kind).collect();
        let mut want = vec![
            K::Relic,
            K::Potion(PotionRarity::Common),
            K::Potion(PotionRarity::Common),
            K::Potion(PotionRarity::Rare),
            K::CardReward(CardRarity::Common),
            K::CardReward(CardRarity::Uncommon),
            K::CardReward(CardRarity::Rare),
            K::Curse,
        ];
        want.extend([K::Gold { big: false }; 5]);
        want.extend([K::Gold { big: true }; 2]);
        assert_eq!(kinds, want);

        // The relic goes first, onto the grid with only the corners cleared.
        let mut empty = CrystalSphereMinigame::new(&mut rng(0), 0);
        empty.item_at = [[None; GRID_SIZE as usize]; GRID_SIZE as usize];
        let candidates: Vec<(i32, i32)> = all_cells()
            .filter(|&(x, y)| empty.can_place_here(4, 4, x, y))
            .collect();
        let mut b = rng(42);
        let first = candidates[b.next_int_exclusive(0, candidates.len() as i32) as usize];
        assert_eq!(game.items()[0].position, first);

        // Fifteen draws in all on a first-try placement.
        let mut c = rng(42);
        for _ in 0..15 {
            let _ = c.next_int(0, 1);
        }
        assert_eq!(a, c, "exactly fifteen placement draws");
    }

    /// The Small tool clears one cell; an item is revealed only once all its
    /// cells are clear, and only once.
    #[test]
    fn items_reveal_only_when_fully_cleared() {
        let mut game = CrystalSphereMinigame::new(&mut rng(7), 99);
        let relic = game.items()[0];
        assert_eq!(relic.kind, CrystalSphereItemKind::Relic);
        let (rx, ry) = relic.position;
        game.set_tool(CrystalSphereTool::Small);
        let mut revealed_at = None;
        for dx in 0..4 {
            for dy in 0..4 {
                let newly = game.cell_clicked(rx + dx, ry + dy);
                if newly.contains(&CrystalSphereItemKind::Relic) {
                    assert!(revealed_at.is_none(), "revealed twice");
                    revealed_at = Some((dx, dy));
                }
            }
        }
        assert_eq!(revealed_at, Some((3, 3)), "the 16th cell reveals the relic");
        assert_eq!(game.divinations(), 99 - 16);
        let relics = game
            .revealed()
            .iter()
            .filter(|k| **k == CrystalSphereItemKind::Relic)
            .count();
        assert_eq!(relics, 1);
    }

    /// The Big tool clears the clicked cell's 3x3 block for one divination.
    #[test]
    fn big_tool_clears_a_three_by_three_block() {
        let mut game = CrystalSphereMinigame::new(&mut rng(7), 1);
        let _ = game.cell_clicked(5, 5);
        for x in 4..=6 {
            for y in 4..=6 {
                assert!(!game.is_hidden(x, y));
            }
        }
        assert!(game.is_hidden(3, 5) && game.is_hidden(5, 7));
        assert!(game.is_finished());
    }

    /// Three Big divinations (the oracle's first three centres) cannot
    /// uncover a 4x4, so Uncover Future never finds the relic.
    #[test]
    fn three_divination_sweep_never_reveals_the_relic() {
        for seed in 0..300 {
            let mut game = CrystalSphereMinigame::new(&mut rng(seed), 3);
            for (x, y) in HEADLESS_SWEEP_CENTERS {
                if game.is_finished() {
                    break;
                }
                let _ = game.cell_clicked(x, y);
            }
            assert!(game.is_finished());
            assert!(!game.revealed().contains(&CrystalSphereItemKind::Relic));
        }
    }

    /// Rewards: `ToReward` rolls each potion in reveal order, `Populate`
    /// then rolls cards, gold and relic in reveal order, all on the event
    /// Rng; selection runs gold, potion, relic, then the card reward.
    #[test]
    fn rewards_roll_in_reveal_order_and_take_in_rewards_set_order() {
        use CrystalSphereItemKind as K;
        let revealed = [
            K::CardReward(CardRarity::Common),
            K::Gold { big: true },
            K::Curse,
            K::Potion(PotionRarity::Rare),
            K::Relic,
        ];
        let mut rs = RunState::new();
        rs.gold = 0;
        let mut by_rarity = std::collections::BTreeMap::new();
        by_rarity.insert(sts2_core::RelicRarity::Common, vec![RelicId::PrayerWheel]);
        by_rarity.insert(sts2_core::RelicRarity::Uncommon, vec![RelicId::Nunchaku]);
        by_rarity.insert(sts2_core::RelicRarity::Rare, vec![RelicId::RoyalPoison]);
        rs.populate_relic_grab_bag_by_rarity(by_rarity, vec![], vec![]);
        let relics_before = rs.relics.len();
        let mut probe = rs.clone();
        let mut event = rng(99);
        let mut replay = event.clone();
        offer_rewards(&mut rs, &mut event, &revealed);

        // The same draws, by hand: ToReward (potion), then Populate in
        // reveal order (cards, gold, relic).
        let potion = crate::potions::all::roll_event_potion_full_pool_of_rarity(
            &mut replay,
            probe.character_id,
            PotionRarity::Rare,
        )
        .expect("a rare potion");
        let character = probe.character_id;
        let cards = crate::cards::spec_index::roll_uniform_rarity_reward_cards(
            &mut probe,
            character,
            CardRarity::Common,
            3,
            true,
            Some(&mut replay),
        );
        assert_eq!(replay.next_int_exclusive(30, 31), 30, "gold populate draw");
        let relic = match sts2_core::runstate::roll_relic_rarity(&mut replay) {
            sts2_core::RelicRarity::Common => RelicId::PrayerWheel,
            sts2_core::RelicRarity::Uncommon => RelicId::Nunchaku,
            _ => RelicId::RoyalPoison,
        };
        assert_eq!(event, replay, "same draws, same order");

        assert_eq!(rs.gold, 30, "big gold");
        assert_eq!(rs.potions, vec![potion]);
        assert_eq!(rs.relics.len(), relics_before + 1, "relic obtained");
        assert_eq!(
            rs.relics.last(),
            Some(&relic),
            "front pull of the rolled rarity"
        );
        assert_eq!(
            rs.pending_rewards.len(),
            1,
            "one card reward; the curse gives none"
        );
        let sts2_core::Reward::Card(reward) = &rs.pending_rewards[0] else {
            panic!("card reward expected");
        };
        let want: Vec<CardId> = cards.iter().map(|c| c.card_id).collect();
        assert_eq!(reward.options, want);
        assert_eq!(want.len(), 3);
        assert!(want.iter().all(|id| {
            crate::cards::spec_index::rarity_for_card_id_in(*id, rs.game_version)
                == Some(CardRarity::Common)
        }));
    }

    /// The Curse adds a Doubt when revealed (not as a reward).
    #[test]
    fn revealing_the_curse_adds_doubt() {
        for seed in 0..500 {
            let mut probe = CrystalSphereMinigame::new(&mut rng(seed), 6);
            let mut hits_curse = false;
            for (x, y) in HEADLESS_SWEEP_CENTERS {
                hits_curse |= probe
                    .cell_clicked(x, y)
                    .contains(&CrystalSphereItemKind::Curse);
            }
            if !hits_curse {
                continue;
            }
            let mut rs = RunState::new();
            let before = rs.master_deck.len();
            play_headless_and_offer(&mut rs, &mut rng(seed), 6);
            let doubts = rs.master_deck[before..]
                .iter()
                .filter(|c| c.card_id == CardId::Doubt)
                .count();
            assert_eq!(doubts, 1, "seed {seed}");
            return;
        }
        panic!("no seed in 0..500 reveals the curse");
    }
}
