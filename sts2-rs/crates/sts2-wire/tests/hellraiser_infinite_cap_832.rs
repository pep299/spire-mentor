//! GH #832: the Hellraiser infinite-HP auto-play cap (`HellraiserPower.cs`,
//! v0.107.1), whose absence let Pillage loop against the beaten Waterfall Giant.

mod common;

use std::sync::mpsc;
use std::time::Duration;

use common::{boss, boss_mut, boss_uid, end_turn, fight, play, player};
use sts2_core::card_effects::apply_power_to_creature;
use sts2_core::{CombatState, PowerKind, StepResult};

/// D3 / GH #832, the hang shape: Hellraiser auto-plays every drawn Strike, and
/// Pillage (`Pillage.cs:28-33`) draws until a non-Attack or a full hand. Against
/// the beaten giant (999,999,999 HP, `HpDisplay.InfiniteWithoutNumbers`,
/// `WaterfallGiant.cs:305-306`) C# auto-plays at most 9 Strikes per turn
/// (`HellraiserPower.cs:45-56`); the rest stay in hand, the hand fills and
/// Pillage stops. Without the cap every Strike chipped the sentinel and was
/// reshuffled and redrawn about 10^8 times. The play runs on a worker thread so
/// a regression fails on the timeout instead of hanging the suite.
#[test]
fn hellraiser_pillage_against_the_beaten_giant_stops_at_nine_auto_plays() {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        tx.send(hellraiser_pillage_scenario())
            .expect("main thread waits");
    });
    let report = rx
        .recv_timeout(Duration::from_secs(120))
        .expect("Pillage under Hellraiser against the beaten giant must terminate (GH #832)");
    assert_eq!(report.auto_played_turn_2, 9, "9 auto-plays, then the cap");
    assert_eq!(
        report.hand_after_pillage, 10,
        "the capped Strikes fill the hand"
    );
    assert_eq!(
        report.giant_hp_lost_turn_2,
        6 + 9 * 6,
        "Pillage's own 6 plus nine 6-damage Strikes"
    );
    assert_eq!(
        report.counter_after_pillage, 19,
        "every drawn Strike counts"
    );
    assert_eq!(
        report.auto_played_turn_3, 5,
        "the counter resets at the player's turn end (`HellraiserPower.cs:70-78`), \
         so the next opening hand of Strikes is auto-played again"
    );
}

struct HellraiserReport {
    auto_played_turn_2: usize,
    hand_after_pillage: usize,
    giant_hp_lost_turn_2: i32,
    counter_after_pillage: i32,
    auto_played_turn_3: usize,
}

fn strikes_in_hand(state: &CombatState) -> usize {
    state
        .player
        .hand
        .iter()
        .filter(|u| {
            state.cards[u.0.get() as usize - 1].card_id == sts2_core::cards::CardId::StrikeIronclad
        })
        .count()
}

fn hellraiser_counter(state: &CombatState) -> i32 {
    let p = player(state);
    p.powers
        .iter()
        .filter_map(|pu| state.powers.get_by_id(pu.0))
        .find(|p| p.kind == PowerKind::Hellraiser)
        .map(|p| p.counter)
        .expect("player holds Hellraiser")
}

fn hellraiser_pillage_scenario() -> HellraiserReport {
    use sts2_core::cards::CardId;
    // 5 Defends make the opening hand, so turn 1 auto-plays nothing.
    let mut deck = vec!["DEFEND_IRONCLAD"; 5];
    deck.push("PILLAGE");
    deck.extend(["STRIKE_IRONCLAD"; 12]);
    let mut state = fight("Ironclad", &deck, 90, &[], &[], "WATERFALL_GIANT_BOSS", 10);
    // Pin the piles: Defends in hand, then Pillage, then the 12 Strikes.
    let all: Vec<_> = state
        .player
        .hand
        .iter()
        .chain(state.player.draw.iter())
        .copied()
        .collect();
    let of = |id: CardId| -> Vec<_> {
        all.iter()
            .copied()
            .filter(|u| state.cards[u.0.get() as usize - 1].card_id == id)
            .collect()
    };
    let (defends, pillage, strikes) = (
        of(CardId::DefendIronclad),
        of(CardId::Pillage),
        of(CardId::StrikeIronclad),
    );
    state.player.hand = defends.into_iter().collect();
    state.player.draw = pillage.into_iter().chain(strikes).collect();

    // Turn 1 -> PRESSURIZE. Turn 2's hand draw pulls Pillage + 4 Strikes (no
    // Hellraiser yet). Move those 4 Strikes to the discard pile and the
    // discarded Defends out of the fight, so the only cards Pillage can draw
    // are the 12 Strikes (8 in the draw pile, 4 to be reshuffled).
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    let drawn_strikes: Vec<_> = state
        .player
        .hand
        .iter()
        .copied()
        .filter(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::StrikeIronclad)
        .collect();
    assert_eq!(drawn_strikes.len(), 4);
    state.player.hand.retain(|u| !drawn_strikes.contains(u));
    let discarded_defends = std::mem::take(&mut state.player.discard);
    state.player.exhaust.extend(discarded_defends);
    state.player.discard.extend(drawn_strikes);

    // Beat the giant, then give the player Hellraiser.
    boss_mut(&mut state).current_hp = 1;
    let giant_uid = boss_uid(&state);
    let _ = sts2_core::damage::apply_damage_with_final(
        &mut state,
        sts2_core::hooks::CreatureRef::Player,
        giant_uid,
        1,
        sts2_core::hooks::ValueProps::MOVE,
    );
    assert_eq!(boss(&state).current_hp, 999_999_999);
    assert!(state.all_hittable_enemies_show_infinite_hp());
    let player_uid = state.player.creature_uid.expect("player uid");
    apply_power_to_creature(&mut state, player_uid, PowerKind::Hellraiser, 1);

    let giant_hp_before = boss(&state).current_hp;
    assert_eq!(play(&mut state, CardId::Pillage), StepResult::Advanced);
    // Auto-played Strikes end in the discard pile; the hand keeps the Strikes
    // drawn after the cap.
    let hand_after_pillage = state.player.hand.len();
    let giant_hp_lost_turn_2 = giant_hp_before - boss(&state).current_hp;
    let counter_after_pillage = hellraiser_counter(&state);
    let auto_played_turn_2 =
        usize::try_from(counter_after_pillage).expect("non-negative") - strikes_in_hand(&state);

    // Take Pillage out too, so turn 3 draws only Strikes. The turn end resets
    // the counter, the giant performs ABOUT_TO_BLOW, and turn 3's 5-card draw
    // auto-plays every Strike it pulls.
    let pillage_uid = *state
        .player
        .discard
        .iter()
        .find(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::Pillage)
        .expect("Pillage resolved to the discard pile");
    state.player.discard.retain(|u| *u != pillage_uid);
    state.player.exhaust.push(pillage_uid);
    assert_eq!(end_turn(&mut state), StepResult::Advanced);
    let auto_played_turn_3 = usize::try_from(hellraiser_counter(&state)).expect("non-negative")
        - strikes_in_hand(&state);

    HellraiserReport {
        auto_played_turn_2,
        hand_after_pillage,
        giant_hp_lost_turn_2,
        counter_after_pillage,
        auto_played_turn_3,
    }
}
