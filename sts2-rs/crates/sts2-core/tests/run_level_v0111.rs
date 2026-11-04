//! GH #792: run-level v0.111.0 behaviour that lives in `sts2-core` — rest
//! site Cook, forced potion rewards, the `BeforeCombatRewardOffered` hook.
//! Each `*_v0111` test pins the v0.111.0 rule next to the unchanged v0.107.1
//! one. Runs under nextest (one process per test), so the hook-registry
//! writes below cannot leak into another test.

use std::sync::atomic::{AtomicUsize, Ordering};

use sts2_core::card_effects::EffectResult;
use sts2_core::cards::{Card, CardId};
use sts2_core::run_effects::{
    dispatch_before_combat_reward_offered, register_before_combat_reward_offered,
    register_should_force_potion_reward, CombatRewardsSet,
};
use sts2_core::runstate::{
    CardReward, RestSiteActionOutcome, RestSiteOption, Reward, RoomType, RunState,
};
use sts2_core::{GameVersion, RelicId};

fn run(version: GameVersion) -> RunState {
    let mut rs = RunState::new();
    rs.game_version = version;
    rs.max_hp = 80;
    rs.current_hp = 60;
    rs.master_deck = (0..6)
        .map(|_| {
            let mut c = Card::pod_defaults();
            c.card_id = CardId::StrikeIronclad;
            c
        })
        .collect();
    rs
}

/// v0.111.0 `CookRestSiteOption.OnSelect` (auto path): remove 2 cards, then
/// `GainMaxHp(5)` — 9 in v0.107.1 (`CookRestSiteOption.cs:17`).
#[test]
fn cook_removes_two_cards_and_gains_five_max_hp_v0111() {
    let mut rs = run(GameVersion::V0_111_0);
    let out = rs.do_rest_site_action(RestSiteOption::Cook { enabled: true });
    assert_eq!(out, RestSiteActionOutcome::Cooked);
    assert_eq!(rs.master_deck.len(), 4);
    assert_eq!((rs.max_hp, rs.current_hp), (85, 65));
}

/// The interactive path queues the two removals and parks the gain until
/// they drain (C# suspends on the removal pick, then gains).
#[test]
fn cook_interactive_defers_the_gain_to_the_removal_v0111() {
    let mut rs = run(GameVersion::V0_111_0);
    rs.interactive_card_removal = true;
    let out = rs.do_rest_site_action(RestSiteOption::Cook { enabled: true });
    assert_eq!(out, RestSiteActionOutcome::Cooked);
    assert_eq!(rs.pending_deck_removals, 2);
    assert_eq!(rs.pending_post_removal_max_hp_gain, 5);
    assert_eq!((rs.max_hp, rs.master_deck.len()), (80, 6), "nothing yet");
}

/// v0.107.1 Cook runs the same body with `GainMaxHp(9)`
/// (`CookRestSiteOption.cs:17`; act23 probe Silent 1012).
#[test]
fn cook_removes_two_cards_and_gains_nine_max_hp_v0107() {
    let mut rs = run(GameVersion::V0_107_1);
    let out = rs.do_rest_site_action(RestSiteOption::Cook { enabled: true });
    assert_eq!(out, RestSiteActionOutcome::Cooked);
    assert_eq!(rs.master_deck.len(), 4);
    assert_eq!((rs.max_hp, rs.current_hp), (89, 69));
}

fn force_always(_rs: &mut RunState, _rt: RoomType, force: &mut bool) -> EffectResult {
    *force = true;
    EffectResult::Resolved
}

/// v0.111.0 `PotionRewardOdds.Roll`: a forced reward returns before the
/// roll — no `NextFloat`, no -0.1 pity swing (`PotionRewardOdds.cs:57`).
/// v0.107.1 rolled and swung anyway.
#[test]
fn forced_potion_reward_skips_the_roll_v0111() {
    register_should_force_potion_reward(RelicId::WhiteBeastStatue, force_always);

    let mut new = run(GameVersion::V0_111_0);
    let (odds, rng) = (new.potion_reward_odds, new.rng.rewards.clone());
    assert!(new.roll_potion_reward(RoomType::Monster));
    assert_eq!(new.potion_reward_odds, odds, "no pity swing");
    assert_eq!(new.rng.rewards, rng, "no NextFloat");

    let mut old = run(GameVersion::V0_107_1);
    let (odds, rng) = (old.potion_reward_odds, old.rng.rewards.clone());
    assert!(old.roll_potion_reward(RoomType::Monster));
    assert_ne!(old.potion_reward_odds, odds, "v0.107.1 swings pity");
    assert_ne!(old.rng.rewards, rng, "v0.107.1 draws");
}

static OFFERED: AtomicUsize = AtomicUsize::new(0);

fn count_offer(_rs: &mut RunState, set: &CombatRewardsSet) -> EffectResult {
    assert!(set.has_card_reward());
    OFFERED.fetch_add(1, Ordering::SeqCst);
    EffectResult::Resolved
}

/// `Hook.BeforeCombatRewardOffered` exists only from v0.111.0
/// (`CombatRoom.cs:274`); the dispatch is a no-op in v0.107.1.
#[test]
fn before_combat_reward_offered_fires_only_v0111() {
    register_before_combat_reward_offered(RelicId::LastingCandy, count_offer);
    let set = CombatRewardsSet {
        room_type: RoomType::Monster,
        rewards: vec![Reward::Card(CardReward::from_options(vec![CardId::Bash]))],
        hook_relics: Vec::new(),
    };
    let mut old = run(GameVersion::V0_107_1);
    dispatch_before_combat_reward_offered(&mut old, &set);
    assert_eq!(OFFERED.load(Ordering::SeqCst), 0);
    let mut new = run(GameVersion::V0_111_0);
    dispatch_before_combat_reward_offered(&mut new, &set);
    assert_eq!(OFFERED.load(Ordering::SeqCst), 1);
    // A relic the set itself grants (not among the set's hook relics) does
    // not hear the hook: C# fires it before any reward is taken.
    let elite_drop = CombatRewardsSet {
        hook_relics: vec![RelicId::Anchor],
        ..set
    };
    dispatch_before_combat_reward_offered(&mut new, &elite_drop);
    assert_eq!(OFFERED.load(Ordering::SeqCst), 1);
}
