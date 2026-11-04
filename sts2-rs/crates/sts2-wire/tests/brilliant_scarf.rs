//! Brilliant Scarf, end to end from a wire state.
//!
//! C# `BrilliantScarf.cs`:
//! `CardsPlayedThisTurn` (the relic's `DisplayAmount`, our `Relic.counter`)
//! counts each non-auto play iteration in `AfterCardPlayed`, resets in
//! `BeforeSideTurnStart`, and while it equals `Cards - 1` = 4 a card in the
//! hand or the play pile costs 0 energy (`TryModifyEnergyCostInCombatLate`)
//! and 0 stars (`TryModifyStarCost`).
//!
//! Ported from manojnuthakki/spire-mentor (ef47c32ec, d31aae6d2, ac3529423).

use sts2_wire::state_from_json_str;

fn install() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
}

fn small_fight(hand: &[&str], relics: serde_json::Value) -> serde_json::Value {
    let card = |id: &str, i: usize| serde_json::json!({"id": id, "stable_idx": i, "kind": "card", "upgraded": false});
    serde_json::json!({
        "wire_version": 1, "phase": "PlayerPlayPhase", "round": 1, "side": "Player",
        "is_elite": false, "turn": 1, "combat_id": 0, "rng": {"seed": 1, "ascension": 0},
        "player": {"class": "ironclad", "hp": 80, "max_hp": 80, "block": 0, "energy": 3,
                   "max_energy": 3, "energy_regen_per_turn": 3, "gold": 0, "relics": relics,
                   "powers": [], "potions": []},
        "deck": {"hand": hand.iter().enumerate().map(|(i, id)| card(id, i + 1)).collect::<Vec<_>>(),
                 "draw_pile": (0..3).map(|i| card("STRIKE_IRONCLAD", 10 + i)).collect::<Vec<_>>(),
                 "discard_pile": [], "exhaust_pile": []},
        "encounter": [{"kind": "Nibbit", "uid": 1, "hp": 100, "max_hp": 100, "block": 0,
                       "powers": [], "intents": [{"kind": "Attack", "amount": 6, "times": 1}],
                       "intent_history": [], "is_dead": false, "is_escaped": false}],
    })
}

fn scarf(counter: i32) -> serde_json::Value {
    serde_json::json!([{"id": "BRILLIANT_SCARF", "counter": counter}])
}

fn play(state: &mut sts2_core::CombatState, hand_index: u8, target: Option<u8>) {
    let _ = sts2_core::apply(state, sts2_core::Action::PlayCard { hand_index, target });
}

fn cost(state: &sts2_core::CombatState, i: usize) -> i32 {
    sts2_core::hooks::effective_card_cost(state, state.player.hand[i])
}

/// The fifth card of the turn costs 0 (whichever card it is), the play counts
/// once, the sixth pays again, and the player's turn start resets the count.
#[test]
fn brilliant_scarf_makes_the_fifth_card_free() {
    install();
    let mut state =
        state_from_json_str(&small_fight(&["BASH", "BASH"], scarf(0)).to_string()).unwrap();
    play(&mut state, 0, Some(0));
    assert_eq!(state.relics[0].counter, 1, "one play counts once");

    let state = state_from_json_str(&small_fight(&["BASH", "BASH"], scarf(3)).to_string()).unwrap();
    assert_eq!(cost(&state, 0), 2, "the 4th play pays");

    let mut state =
        state_from_json_str(&small_fight(&["BASH", "BASH"], scarf(4)).to_string()).unwrap();
    assert_eq!(
        (cost(&state, 0), cost(&state, 1)),
        (0, 0),
        "the 5th is free"
    );
    play(&mut state, 0, Some(0));
    assert_eq!(state.player.energy, 3, "the free Bash cost nothing");
    assert_eq!(state.relics[0].counter, 5);
    assert_eq!(cost(&state, 0), 2, "the 6th pays again");

    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert_eq!(
        state.relics[0].counter, 0,
        "reset at the player's turn start"
    );
}

/// `AfterCardPlayed` returns on `cardPlay.IsAutoPlay`: Havoc counts, the
/// Strike it auto-plays does not, so Bash after it still costs 2.
#[test]
fn brilliant_scarf_does_not_count_an_auto_play() {
    install();
    let mut value = small_fight(&["HAVOC", "BASH"], scarf(2));
    value["deck"]["draw_pile"] = serde_json::json!([
        {"id": "STRIKE_IRONCLAD", "stable_idx": 10, "kind": "card", "upgraded": false}
    ]);
    let mut state = state_from_json_str(&value.to_string()).unwrap();
    play(&mut state, 0, None);
    assert_eq!(
        state.relics[0].counter, 3,
        "Havoc counts, its auto-play doesn't"
    );
    assert_eq!(cost(&state, 0), 2);
}

/// `TryModifyStarCost`: the fifth card costs no stars either, so a star card
/// is playable with no stars at counter 4 but not at 3.
#[test]
fn brilliant_scarf_makes_the_fifth_card_free_in_stars() {
    install();
    for (counter, playable) in [(3, false), (4, true)] {
        let value = small_fight(&["CRESCENT_SPEAR"], scarf(counter));
        let state = state_from_json_str(&value.to_string()).unwrap();
        let card = &state.cards[state.player.hand[0].get() as usize - 1];
        assert_eq!(
            sts2_core::effective_star_cost_in_combat(&state, card) == 0,
            playable,
            "counter {counter}"
        );
        let plays = sts2_core::legal_actions(&state)
            .iter()
            .any(|a| matches!(a, sts2_core::Action::PlayCard { hand_index: 0, .. }));
        assert_eq!(plays, playable, "counter {counter}: playable with no stars");
    }
}

/// The count moves after each play iteration (`AfterCardPlayed` sits inside
/// `OnPlayWrapper`'s `playCount` loop, CardModel.cs:1905-1965): a Helix Drill
/// at cost 1 with one replay, played 4th, has its replay be the free 5th.
/// Helix's hits are the energy spent this turn less its own current cost:
/// 1 - 1 = 0 on the play, 1 - 0 = 1 on the free replay, so 3 damage.
#[test]
fn brilliant_scarf_counts_each_replay_as_it_ends() {
    install();
    let mut value = small_fight(&["HELIX_DRILL"], scarf(3));
    let helix = &mut value["deck"]["hand"][0];
    helix["base_cost"] = serde_json::json!(1);
    helix["cost_override"] = serde_json::json!(1);
    helix["base_replay_count"] = serde_json::json!(1);
    let mut state = state_from_json_str(&value.to_string()).unwrap();
    play(&mut state, 0, Some(0));
    let nibbit = state
        .creatures
        .iter()
        .find(|c| c.id == state.enemy_uids[0].0)
        .unwrap();
    assert_eq!(nibbit.current_hp, 100 - 3, "the free replay hits once");
    assert_eq!(state.relics[0].counter, 5);
}
