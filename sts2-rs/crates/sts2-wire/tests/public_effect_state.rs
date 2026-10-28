use serde_json::{json, Value};
use sts2_core::cards::{CardId, CardType};
use sts2_core::{Action, PowerKind, RelicKind};
use sts2_wire::{emit::combat_state_to_value, parse_json, state_from_json_str};

fn fixture() -> Value {
    let mut value: Value =
        serde_json::from_str(include_str!("fixtures/combat_action_ironclad.json")).unwrap();
    value["game_version"] = json!("0.111.0");
    value
}

fn rebuild(value: &Value) -> sts2_core::CombatState {
    state_from_json_str(&value.to_string()).unwrap()
}

#[test]
fn v111_registry_cards_decode_in_both_name_forms() {
    for id in CardId::V0111_ONLY {
        let name = id.to_python_name();
        let mut value = fixture();
        value["game_version"] = json!("0.111.0");
        value["deck"]["hand"] = json!([{"id":name,"kind":"card","stable_idx":4,"upgraded":false}]);
        let parsed = parse_json(&value).unwrap();
        assert_eq!(parsed.deck.hand[0].card_id, Some(id), "{name}");
        let state = rebuild(&value);
        assert_eq!(
            state.cards[state.player.hand[0].get() as usize - 1].card_id,
            id
        );
        if let Some(short) = name.strip_suffix("_CARD") {
            value["deck"]["hand"][0]["id"] = json!(short);
            assert_eq!(parse_json(&value).unwrap().deck.hand[0].card_id, Some(id));
        }
    }
    let mut unknown = fixture();
    unknown["deck"]["hand"][0]["id"] = json!("UNKNOWN_CARD");
    assert!(parse_json(&unknown).is_err());
}

#[test]
fn generated_card_templates_use_the_same_instance_decoder() {
    let card = json!({
        "id":"WITHER", "kind":"status", "stable_idx":0, "upgraded":false,
        "base_damage":12,
        "local_cost":{"base":0,"costs_x":false,"modifiers":[]}
    });
    let decoded = sts2_wire::state_dict::card_from_wire(
        &&card,
        sts2_core::CardUid::new(99).unwrap(),
        "$.cards[0]",
    )
    .unwrap();
    assert_eq!(decoded.uid.get(), 99);
    assert_eq!(decoded.card_id, CardId::Wither);
    assert_eq!(
        decoded.combat_var(sts2_core::cards::VarKey::WitherFakeUpgradeLevel),
        3
    );
    assert_eq!(decoded.base_damage, 12);
}

#[test]
fn granted_sly_survives_when_the_game_version_removes_intrinsic_sly() {
    for (version, intrinsic) in [("0.107.1", true), ("0.111.0", false)] {
        for granted in [false, true] {
            let mut value = fixture();
            value["game_version"] = json!(version);
            value["deck"]["hand"] = json!([{
                "id":"HAZE", "stable_idx":0, "kind":"card", "upgraded":false,
                "flags": if granted { vec!["sly"] } else { vec![] }
            }]);
            let state = rebuild(&value);
            let uid = state.player.hand[0];
            assert_eq!(
                sts2_core::card_effects::card_is_sly(&state, uid),
                intrinsic || granted
            );
            assert_eq!(
                state.player.sly_this_combat.contains(&uid),
                granted && !intrinsic
            );
            let back = rebuild(&combat_state_to_value(&state).unwrap());
            assert_eq!(
                sts2_core::card_effects::card_is_sly(&back, back.player.hand[0]),
                intrinsic || granted
            );
        }
    }
}

#[test]
fn public_effect_state_roundtrips_identity_and_history() {
    let mut value = fixture();
    value["player"]["cards_started_this_turn"] = json!([2, 3, 1, 0, 0]);
    value["player"]["block_gained_card_plays_this_turn"] = json!([0, 0, 3]);
    value["player"]["current_card_play"] = json!(4);
    value["player"]["next_card_play"] = json!(5);
    value["player"]["pen_nib_attack_to_double"] = json!({"pile":"hand","index":0});
    value["player"]["vambrace_triggering_card"] = json!({"pile":"hand","index":0});
    value["player"]["powers"] = json!([{"name":"JUGGLING","amount":1}]);
    value["player"]["relics"] = json!([{"id":"PEN_NIB","counter":0},{"id":"VAMBRACE","counter":0}]);
    let state = rebuild(&value);
    assert_eq!(state.player.card_plays_started_this_turn, 6);
    assert_eq!(
        state.combat_log.cards_started_this_turn[CardType::Attack.index()],
        2
    );
    assert_eq!(
        state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Juggling)
            .unwrap()
            .counter,
        2
    );
    assert_eq!(state.combat_log.current_card_play, Some(4));
    assert_eq!(
        state.combat_log.pen_nib_attack_to_double,
        Some(state.player.hand[0])
    );
    assert_eq!(
        state
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::Vambrace)
            .unwrap()
            .counter,
        -(state.player.hand[0].get() as i32)
    );
    let emitted = combat_state_to_value(&state).unwrap();
    let roundtrip = rebuild(&emitted);
    assert_eq!(
        roundtrip.combat_log.cards_started_this_turn,
        state.combat_log.cards_started_this_turn
    );
    assert_eq!(
        roundtrip.combat_log.unmovable_card_block_plays,
        state.combat_log.unmovable_card_block_plays
    );
    assert_eq!(roundtrip.combat_log.current_card_play, Some(4));
    assert_eq!(roundtrip.combat_log.next_card_play, 5);
    assert_eq!(
        roundtrip.combat_log.pen_nib_attack_to_double,
        Some(roundtrip.player.hand[0])
    );
    assert!(
        roundtrip
            .relics
            .iter()
            .find(|r| r.kind == RelicKind::Vambrace)
            .unwrap()
            .counter
            < 0
    );
    value["player"]["next_card_play"] = json!(4);
    assert!(state_from_json_str(&value.to_string()).is_err());
    value["player"]["next_card_play"] = json!(5);
    value["player"]["pen_nib_attack_to_double"]["index"] = json!(99);
    assert!(state_from_json_str(&value.to_string()).is_err());
}

#[test]
fn earlier_public_block_gain_prevents_second_card_doubling() {
    let mut value = fixture();
    value["player"]["block"] = json!(0);
    value["player"]["powers"] = json!([{"name":"UNMOVABLE","amount":1}]);
    value["player"]["cards_started_this_turn"] = json!([0, 1, 0, 0, 0]);
    value["player"]["block_gained_card_plays_this_turn"] = json!([0]);
    value["player"]["next_card_play"] = json!(1);
    value["deck"]["hand"] =
        json!([{"id":"DEFEND_IRONCLAD","kind":"card","stable_idx":4,"upgraded":false}]);
    let mut state = rebuild(&value);
    sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(state.player.block, 5);
    assert_eq!(
        state.combat_log.unmovable_card_block_plays.as_slice(),
        &[0, 1]
    );
    assert_eq!(state.combat_log.current_card_play, None);
}

#[test]
fn pen_nib_counts_before_selection_and_keeps_source_until_resume() {
    for (before, after) in [(6, 7), (9, 0)] {
        let mut value = fixture();
        value["game_version"] = json!("0.111.0");
        value["player"]["relics"] = json!([{"id":"PEN_NIB","counter":before}]);
        value["player"]["powers"] = json!([]);
        value["encounter"][0]["hp"] = json!(100);
        value["encounter"][0]["max_hp"] = json!(100);
        value["deck"]["hand"] =
            json!([{"id":"SEEKER_STRIKE","kind":"card","stable_idx":4,"upgraded":false}]);
        let mut state = rebuild(&value);
        let uid = state.player.hand[0];
        sts2_core::apply(
            &mut state,
            Action::PlayCard {
                hand_index: 0,
                target: Some(0),
            },
        );
        assert!(state.pending_choice.is_some());
        assert_eq!(state.relics[0].counter, after);
        assert_eq!(
            state.combat_log.pen_nib_attack_to_double,
            (before == 9).then_some(uid)
        );
        let active = state.combat_log.current_card_play;
        assert!(active.is_some());
        let choice = sts2_core::legal_actions(&state)
            .into_iter()
            .find(|a| matches!(a, Action::ResolveChoice { .. }))
            .unwrap();
        sts2_core::apply(&mut state, choice);
        assert!(state.pending_choice.is_none());
        assert_eq!(
            state.relics[0].counter, after,
            "resume must not count the play twice"
        );
        assert_eq!(state.combat_log.pen_nib_attack_to_double, None);
        assert_eq!(state.combat_log.current_card_play, None);
    }
}

#[test]
fn replay_counts_started_attacks_and_pen_nib_before_each_iteration() {
    let mut value = fixture();
    value["player"]["relics"] = json!([{"id":"PEN_NIB","counter":8}]);
    value["player"]["powers"] = json!([]);
    value["encounter"][0]["hp"] = json!(100);
    value["encounter"][0]["max_hp"] = json!(100);
    value["deck"]["hand"] =
        json!([{"id":"STRIKE_IRONCLAD","kind":"card","stable_idx":4,"upgraded":false}]);
    let mut state = rebuild(&value);
    let uid = state.player.hand[0];
    state.cards[uid.get() as usize - 1].base_replay_count = 1;
    sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    assert_eq!(state.relics[0].counter, 0);
    assert_eq!(state.player.card_plays_started_this_turn, 2);
    assert_eq!(
        state.combat_log.cards_started_this_turn[CardType::Attack.index()],
        2
    );
    assert_eq!(state.combat_log.pen_nib_attack_to_double, None);
    assert_eq!(state.combat_log.current_card_play, None);
    let hp = state
        .creatures
        .iter()
        .find(|c| c.id == state.enemy_uids[0].0)
        .unwrap()
        .current_hp;
    assert_eq!(hp, 82, "ninth Strike deals 6, tenth deals 12");
}

#[test]
fn pen_nib_uses_explicit_damage_source_not_the_parent_attack() {
    use sts2_core::hooks::{CreatureRef, ValueProps};
    let mut value = fixture();
    value["player"]["powers"] = json!([]);
    value["player"]["relics"] = json!([{"id":"PEN_NIB","counter":0}]);
    value["deck"]["hand"] = json!([
        {"id":"STRIKE_IRONCLAD","kind":"card","stable_idx":4,"upgraded":false},
        {"id":"STRIKE_IRONCLAD","kind":"card","stable_idx":5,"upgraded":false}
    ]);
    let mut state = rebuild(&value);
    state.relics[0].counter = 9;
    assert_eq!(
        sts2_wire::emit::card_damage_preview(&state)[0].get(&0),
        Some(&12)
    );
    state.relics[0].counter = 0;
    let parent = state.player.hand.remove(0);
    let nested = state.player.hand.remove(0);
    state.player.play.push(parent);
    state.resolving_card = Some(parent);
    state.combat_log.current_card_play = Some(0);
    state.combat_log.pen_nib_attack_to_double = Some(parent);
    let damage = |state: &sts2_core::CombatState, card| {
        sts2_core::damage::compute_modified_damage(
            state,
            CreatureRef::Player,
            state.enemy_uids[0],
            6,
            ValueProps::MOVE,
            card,
        )
    };
    assert_eq!(damage(&state, Some(CardId::StrikeIronclad)), 12);
    assert_eq!(
        damage(&state, None),
        6,
        "powered card-less damage inside an Attack is not doubled"
    );
    state.player.play.push(nested);
    state.resolving_card = Some(nested);
    assert_eq!(
        damage(&state, Some(CardId::StrikeIronclad)),
        6,
        "a second card of the same type has another source identity"
    );
    state.combat_log.pen_nib_attack_to_double = Some(nested);
    assert_eq!(damage(&state, Some(CardId::StrikeIronclad)), 12);
    state.resolving_card = Some(parent);
    assert_eq!(
        damage(&state, Some(CardId::StrikeIronclad)),
        6,
        "a nested marked Attack does not boost its parent"
    );
    state.resolving_card = Some(nested);
    state.relics.set_melted(0, true);
    assert_eq!(damage(&state, Some(CardId::StrikeIronclad)), 6);
}

#[test]
fn nested_block_cards_keep_distinct_play_tags_across_choices() {
    let mut value = fixture();
    value["player"]["block"] = json!(0);
    value["player"]["powers"] = json!([{"name":"UNMOVABLE","amount":1}]);
    value["player"]["relics"] = json!([]);
    value["deck"]["hand"] = json!([
        {"id":"CATASTROPHE","kind":"card","stable_idx":4,"upgraded":false},
        {"id":"STRIKE_IRONCLAD","kind":"card","stable_idx":5,"upgraded":false},
        {"id":"STRIKE_IRONCLAD","kind":"card","stable_idx":6,"upgraded":false},
        {"id":"STRIKE_IRONCLAD","kind":"card","stable_idx":7,"upgraded":false}
    ]);
    value["deck"]["draw_pile"] = json!([
        {"id":"ARMAMENTS","kind":"card","stable_idx":0,"upgraded":false},
        {"id":"ARMAMENTS","kind":"card","stable_idx":1,"upgraded":false}
    ]);
    let mut state = rebuild(&value);
    sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert!(state.pending_choice.is_some());
    assert_eq!(state.combat_log.current_card_play, Some(1));
    for _ in 0..2 {
        let choice = sts2_core::legal_actions(&state)
            .into_iter()
            .find(|a| matches!(a, Action::ResolveChoice { .. }))
            .unwrap();
        sts2_core::apply(&mut state, choice);
    }
    assert!(state.pending_choice.is_none());
    assert_eq!(state.combat_log.current_card_play, None);
    assert_eq!(
        state.combat_log.unmovable_card_block_plays.as_slice(),
        &[1, 2]
    );
    assert_eq!(
        state.combat_log.cards_started_this_turn[CardType::Skill.index()],
        3
    );
    assert_eq!(
        state.player.block, 15,
        "only the first Armaments play doubles"
    );
}

#[test]
fn effect_history_and_active_source_distinguish_search_states() {
    let mut value = fixture();
    value["game_version"] = json!("0.111.0");
    let state = rebuild(&value);
    let baseline = sts2_core::search_hash(&state);
    let mut changed = state.clone();
    changed.combat_log.cards_started_this_turn[CardType::Attack.index()] = 1;
    assert_ne!(sts2_core::search_hash(&changed), baseline);
    let mut changed = state.clone();
    changed.combat_log.unmovable_card_block_plays.push(0);
    assert_ne!(sts2_core::search_hash(&changed), baseline);
    let mut changed = state.clone();
    changed.combat_log.pen_nib_attack_to_double = Some(state.player.hand[0]);
    assert_ne!(sts2_core::search_hash(&changed), baseline);
    let mut before_draw = state.clone();
    before_draw.pending_relic_hand_draw_resume = Some((5, 1));
    let mut later_listener = before_draw.clone();
    later_listener.pending_relic_hand_draw_resume = Some((5, 2));
    assert_ne!(
        sts2_core::search_hash(&before_draw),
        sts2_core::search_hash(&later_listener)
    );
    later_listener.pending_relic_hand_draw_resume = Some((6, 1));
    assert_ne!(
        sts2_core::search_hash(&before_draw),
        sts2_core::search_hash(&later_listener)
    );
}

#[test]
fn temporary_card_flags_survive_leaving_the_hand() {
    let mut value = fixture();
    value["deck"]["discard_pile"][0]["flags"] = json!(["retain_turn", "sly_turn"]);
    let state = rebuild(&value);
    let uid = state.player.discard[0];
    assert!(state.player.retain_this_turn.contains(&uid));
    assert!(state.player.sly_this_turn.contains(&uid));
    let wire = combat_state_to_value(&state).unwrap();
    let back = rebuild(&wire);
    assert!(back
        .player
        .retain_this_turn
        .contains(&back.player.discard[0]));
    assert!(back.player.sly_this_turn.contains(&back.player.discard[0]));
}

#[test]
fn juggling_applied_mid_turn_counts_started_attacks_including_pending_parents() {
    let mut value = fixture();
    value["game_version"] = json!("0.111.0");
    value["player"]["powers"] = json!([]);
    value["player"]["relics"] = json!([]);
    value["player"]["cards_played_this_turn"] = json!([1, 0, 0, 0, 0]);
    value["player"]["cards_started_this_turn"] = json!([2, 0, 0, 0, 0]);
    value["deck"]["hand"] = json!([
        {"id":"JUGGLING_CARD","kind":"card","stable_idx":4,"upgraded":false},
        {"id":"STRIKE_IRONCLAD","kind":"card","stable_idx":5,"upgraded":false}
    ]);
    let mut state = rebuild(&value);
    sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(
        state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Juggling)
            .unwrap()
            .counter,
        2
    );
    let arena_size = state.cards.len();
    sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    assert_eq!(
        state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Juggling)
            .unwrap()
            .counter,
        3
    );
    assert_eq!(state.cards.len(), arena_size + 1);
    assert_eq!(
        state.player.hand.len(),
        1,
        "the third started Attack leaves a generated copy"
    );
}

#[test]
fn ordered_cost_modifiers_roundtrip_without_losing_expiration_or_order() {
    let mut value = fixture();
    value["deck"]["hand"][0]["combat_vars"] = json!({});
    value["deck"]["hand"][0]["local_cost"] = json!({
        "base":3,"costs_x":false,"modifiers":[
            {"kind":"absolute","amount":1,"reduce_only":true,"expiration":6},
            {"kind":"relative","amount":2,"reduce_only":false,"expiration":0}
        ]
    });
    let state = rebuild(&value);
    let card = &state.cards[state.player.hand[0].get() as usize - 1];
    assert_eq!(card.local_costs_after_expiration(), [3, 5, 5, 5]);
    let emitted = combat_state_to_value(&state).unwrap();
    assert!(emitted["deck"]["hand"][0]["local_cost"]["functions"].is_array());
    let back = rebuild(&emitted);
    let card = &back.cards[back.player.hand[0].get() as usize - 1];
    assert_eq!(card.local_costs_after_expiration(), [3, 5, 5, 5]);
    let mut model = card.local_energy_cost.unwrap();
    model.upgrade_base(-1);
    assert_eq!(
        model.costs(),
        [3, 4, 4, 4],
        "the relative +2 after absolute 1 survives a base upgrade"
    );
    value["deck"]["hand"][0]["local_cost"]["modifiers"][0]["expiration"] = json!(0);
    let permanent = rebuild(&value);
    assert_eq!(
        permanent.cards[permanent.player.hand[0].get() as usize - 1].cost,
        state.cards[state.player.hand[0].get() as usize - 1].cost
    );
    assert_ne!(
        sts2_core::search_hash(&permanent),
        sts2_core::search_hash(&state),
        "matching current cost does not imply matching future expiry"
    );
    value["deck"]["hand"][0]["local_cost"]["modifiers"][0]["expiration"] = json!(1);
    assert!(parse_json(&value).is_err());
    value["deck"]["hand"][0]["local_cost"] = emitted["deck"]["hand"][0]["local_cost"].clone();
    value["deck"]["hand"][0]["local_cost"]["modifiers"] = json!([]);
    assert!(
        parse_json(&value).is_err(),
        "two representations cannot both be authoritative"
    );
    value["deck"]["hand"][0]["local_cost"]
        .as_object_mut()
        .unwrap()
        .remove("modifiers");
    value["deck"]["hand"][0]["local_cost"]["functions"][0]["uses_base"] = json!(false);
    value["deck"]["hand"][0]["local_cost"]["functions"][0]["ceiling"] = Value::Null;
    assert!(
        parse_json(&value).is_err(),
        "malformed constant functions must fail at ingest"
    );
}

#[test]
fn played_cost_effects_survive_expiration_of_an_earlier_free_modifier() {
    for (id, base, expected, target) in [
        ("MODDED", 0, 1, None),
        ("UP_MY_SLEEVE", 2, 1, None),
        ("FRANTIC_ESCAPE", 1, 2, None),
        ("MOMENTUM_STRIKE", 1, 0, Some(0)),
    ] {
        let mut value = fixture();
        value["game_version"] = json!("0.111.0");
        value["player"]["powers"] = json!([]);
        value["player"]["relics"] = json!([]);
        value["encounter"][0]["hp"] = json!(1000);
        value["encounter"][0]["max_hp"] = json!(1000);
        value["deck"]["hand"] = json!([{
            "id":id,"kind":"card","stable_idx":4,"upgraded":false,
            "local_cost":{"base":base,"costs_x":false,"modifiers":[
                {"kind":"absolute","amount":0,"reduce_only":false,"expiration":6}
            ]}
        }]);
        let mut state = rebuild(&value);
        let uid = state.player.hand[0];
        sts2_core::apply(
            &mut state,
            Action::PlayCard {
                hand_index: 0,
                target,
            },
        );
        assert!(
            state.player.hand.iter().all(|u| *u != uid),
            "{id} must actually be played"
        );
        let card = &state.cards[uid.get() as usize - 1];
        assert_eq!(card.local_costs_after_expiration(), [expected; 4], "{id}");
    }
}

#[test]
fn adaptive_strike_clone_keeps_its_own_permanent_free_modifier() {
    let mut value = fixture();
    value["player"]["powers"] = json!([]);
    value["player"]["relics"] = json!([]);
    value["encounter"][0]["hp"] = json!(1000);
    value["encounter"][0]["max_hp"] = json!(1000);
    value["deck"]["hand"] = json!([{
        "id":"ADAPTIVE_STRIKE","kind":"card","stable_idx":4,"upgraded":false,
        "local_cost":{"base":2,"costs_x":false,"modifiers":[]}
    }]);
    let mut state = rebuild(&value);
    let source = state.player.hand[0];
    sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    let clone = state
        .player
        .discard
        .iter()
        .copied()
        .find(|uid| {
            *uid != source && state.cards[uid.get() as usize - 1].card_id == CardId::AdaptiveStrike
        })
        .unwrap();
    assert_eq!(
        state.cards[source.get() as usize - 1].local_costs_after_expiration(),
        [2; 4]
    );
    assert_eq!(
        state.cards[clone.get() as usize - 1].local_costs_after_expiration(),
        [0; 4]
    );
}

#[test]
fn slumbering_discount_expires_on_play_and_stomp_backfill_keeps_full_relative_amount() {
    use sts2_core::cards::{EnchantmentId, LocalCostModifier, LocalCostModifierKind};
    let mut value = fixture();
    value["player"]["powers"] = json!([]);
    value["player"]["relics"] = json!([]);
    value["deck"]["hand"] = json!([{
        "id":"DEFEND_IRONCLAD","kind":"card","stable_idx":4,"upgraded":false,
        "local_cost":{"base":1,"costs_x":false,"modifiers":[]}
    }]);
    let mut state = rebuild(&value);
    let uid = state.player.hand[0];
    state.cards[uid.get() as usize - 1].set_enchantment(EnchantmentId::SlumberingEssence, 1);
    sts2_core::enchantment_hooks::dispatch_before_flush(&mut state);
    assert_eq!(
        state.cards[uid.get() as usize - 1].local_costs_after_expiration(),
        [0, 0, 1, 1]
    );
    sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(
        state.cards[uid.get() as usize - 1].local_costs_after_expiration(),
        [1; 4]
    );

    let card = &mut state.cards[uid.get() as usize - 1];
    card.card_id = CardId::Stomp;
    state.player.cards_played_this_turn[CardType::Attack.index()] = 3;
    sts2_core::card_effects::stamp_entered_combat_cost_backfill(&mut state, uid);
    let card = &mut state.cards[uid.get() as usize - 1];
    card.add_local_cost_modifier(LocalCostModifier {
        kind: LocalCostModifierKind::Relative,
        amount: 1,
        reduce_only: false,
        expiration: 0,
    });
    assert_eq!(
        card.local_costs_after_expiration(),
        [0, 2, 0, 2],
        "the -3 turn modifier must not be truncated to a -1 current-cost shave"
    );
}

#[test]
fn tezcatara_ember_changes_base_without_erasing_later_relative_cost() {
    let mut value = fixture();
    value["deck"]["hand"][0]["combat_vars"] = json!({});
    value["deck"]["hand"][0]["local_cost"] = json!({
        "base":2,"costs_x":false,"modifiers":[
            {"kind":"relative","amount":1,"reduce_only":false,"expiration":0}
        ]
    });
    let mut state = rebuild(&value);
    let card = &mut state.cards[state.player.hand[0].get() as usize - 1];
    sts2_content::enchantments::apply_static_enchantment(
        card,
        sts2_core::cards::EnchantmentId::TezcatarasEmber,
        1,
    );
    assert_eq!(card.local_energy_cost.unwrap().base, 0);
    assert_eq!(card.local_costs_after_expiration(), [1; 4]);
}

#[test]
fn bookmark_discount_survives_turn_end_but_not_the_next_play() {
    let mut value = fixture();
    value["player"]["powers"] = json!([]);
    value["player"]["relics"] = json!([{"id":"BOOKMARK","counter":0}]);
    value["deck"]["hand"] = json!([{
        "id":"DEFEND_IRONCLAD","kind":"card","stable_idx":4,"upgraded":false,
        "flags":["retain"],"local_cost":{"base":1,"costs_x":false,"modifiers":[]}
    }]);
    let mut state = rebuild(&value);
    let uid = state.player.hand[0];
    sts2_core::relic_effects::dispatch_after_turn_end(&mut state, sts2_core::Side::Player);
    assert_eq!(
        state.cards[uid.get() as usize - 1].local_costs_after_expiration(),
        [0, 0, 1, 1]
    );
    state.restore_wire_transient_costs_for_turn_start();
    assert_eq!(state.cards[uid.get() as usize - 1].cost, 0);
    sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(
        state.cards[uid.get() as usize - 1].local_costs_after_expiration(),
        [1; 4]
    );
}

#[test]
fn vexing_generated_card_is_only_free_until_played_or_turn_end() {
    let mut positive_cost_cards = 0;
    for seed in 0..12 {
        let mut value = fixture();
        value["round"] = json!(1);
        value["turn"] = json!(1);
        value["rng"]["run_seed"] = json!(seed);
        value["player"]["class"] = json!("defect");
        value["player"]["powers"] = json!([]);
        value["player"]["relics"] = json!([{"id":"VEXING_PUZZLEBOX","counter":0}]);
        let mut state = rebuild(&value);
        let arena_size = state.cards.len();
        sts2_core::relic_effects::dispatch_after_side_turn_start(
            &mut state,
            sts2_core::Side::Player,
        );
        assert_eq!(state.cards.len(), arena_size + 1);
        let card = state.cards.last().unwrap();
        let model = card
            .local_energy_cost
            .expect("generated factory cards retain their exact local cost");
        if model.costs_x {
            continue;
        }
        assert_eq!(card.cost, 0);
        if model.base > 0 {
            positive_cost_cards += 1;
            let mut after_play = *card;
            after_play.expire_local_energy_cost(4);
            assert_eq!(i32::from(after_play.cost), model.base);
            let mut after_turn = *card;
            after_turn.expire_local_energy_cost(2);
            assert_eq!(i32::from(after_turn.cost), model.base);
        }
    }
    assert!(
        positive_cost_cards > 0,
        "fixture seeds must exercise a positive-cost generated card"
    );
}

#[test]
fn discovery_is_skippable_but_abundance_requires_a_selection() {
    for (id, min, skip) in [("DISCOVERY", 0, true), ("ABUNDANCE", 1, false)] {
        let mut value = fixture();
        value["game_version"] = json!("0.111.0");
        value["player"]["powers"] = json!([]);
        value["player"]["relics"] = json!([]);
        value["deck"]["hand"] = json!([{"id":id,"kind":"card","stable_idx":4,"upgraded":false}]);
        let mut state = rebuild(&value);
        sts2_core::apply(
            &mut state,
            Action::PlayCard {
                hand_index: 0,
                target: None,
            },
        );
        let choice = state.pending_choice.as_ref().unwrap();
        assert_eq!(
            (choice.min_select, choice.max_select, choice.can_skip),
            (min, 1, skip),
            "{id}"
        );
        assert_eq!(choice.options.len(), 3);
        if skip {
            sts2_core::apply(
                &mut state,
                Action::ResolveChoice {
                    selection: smallvec::smallvec![],
                },
            );
            assert!(state.pending_choice.is_none());
        }
    }
}

#[test]
fn turn_cost_expiration_precedes_bookmark_after_flush_selection() {
    let mut value = fixture();
    value["player"]["powers"] = json!([]);
    value["player"]["relics"] = json!([{"id":"BOOKMARK","counter":0}]);
    value["deck"]["hand"] = json!([{
        "id":"DEFEND_IRONCLAD","kind":"card","stable_idx":4,"upgraded":false,"flags":["retain"],
        "local_cost":{"base":2,"costs_x":false,"modifiers":[
            {"kind":"absolute","amount":0,"reduce_only":false,"expiration":2}
        ]}
    }]);
    let mut state = rebuild(&value);
    let uid = state.player.hand[0];
    sts2_core::apply(&mut state, Action::EndTurn);
    assert_eq!(
        state.cards[uid.get() as usize - 1].local_costs_after_expiration(),
        [1, 1, 2, 2],
        "Bookmark sees the restored local cost after the turn modifier expires"
    );
}

fn capture_finished_count(
    state: &mut sts2_core::CombatState,
    _: &sts2_core::relic_effects::CardEventCtx,
) -> sts2_core::EffectResult {
    state.player.block = (state.combat_log.card_plays_finished_this_turn
        + state.player.cards_played_this_turn[CardType::Attack.index()])
        as i32;
    sts2_core::EffectResult::Resolved
}

#[test]
fn replay_after_card_played_observes_its_own_completed_history_entry() {
    use sts2_core::relics::RelicId;
    for echo in [false, true] {
        let mut value = fixture();
        value["game_version"] = json!("0.111.0");
        value["player"]["cards_played_this_turn"] = json!([0, 0, 0, 0, 0]);
        value["player"]["relics"] = json!([{"id":"ANCHOR","counter":0}]);
        value["player"]["powers"] = if echo {
            json!([{"name":"ECHO_FORM","amount":1}])
        } else {
            json!([])
        };
        value["encounter"][0]["hp"] = json!(1000);
        value["encounter"][0]["max_hp"] = json!(1000);
        value["deck"]["hand"] =
            json!([{"id":"STRIKE_IRONCLAD","kind":"card","stable_idx":4,"upgraded":false}]);
        let mut state = rebuild(&value);
        if !echo {
            state.cards[state.player.hand[0].get() as usize - 1].base_replay_count = 1;
        }
        sts2_core::relic_effects::register_after_card_played(
            RelicId::Anchor,
            capture_finished_count,
        );
        sts2_core::apply(
            &mut state,
            Action::PlayCard {
                hand_index: 0,
                target: Some(0),
            },
        );
        assert_eq!(
            state.player.block, 4,
            "after-play sees its replay's finished entry (echo={echo})"
        );
    }
}

fn capture_played_cost(
    state: &mut sts2_core::CombatState,
    ctx: &sts2_core::relic_effects::CardEventCtx,
) -> sts2_core::EffectResult {
    state.player.block = i32::from(state.cards[ctx.card_uid.unwrap().get() as usize - 1].cost);
    sts2_core::EffectResult::Resolved
}

#[test]
fn when_played_cost_modifier_remains_live_through_after_card_played() {
    let mut value = fixture();
    value["player"]["powers"] = json!([]);
    value["player"]["relics"] = json!([{"id":"ANCHOR","counter":0}]);
    value["deck"]["hand"] = json!([{
        "id":"MODDED","kind":"card","stable_idx":4,"upgraded":false,
        "local_cost":{"base":2,"costs_x":false,"modifiers":[
            {"kind":"absolute","amount":0,"reduce_only":false,"expiration":6}
        ]}
    }]);
    let mut state = rebuild(&value);
    let uid = state.player.hand[0];
    sts2_core::relic_effects::register_after_card_played(
        sts2_core::relics::RelicId::Anchor,
        capture_played_cost,
    );
    sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: None,
        },
    );
    assert_eq!(
        state.player.block, 1,
        "AfterCardPlayed still observes the earlier free modifier before cleanup"
    );
    assert_eq!(
        state.cards[uid.get() as usize - 1].cost,
        3,
        "WhenPlayed cleanup runs after the hook"
    );
}

#[test]
fn mummified_hand_uses_base_preference_and_current_cost_after_a_free_modifier() {
    let mut value = fixture();
    value["player"]["powers"] = json!([]);
    value["player"]["relics"] = json!([{"id":"MUMMIFIED_HAND","counter":0}]);
    value["deck"]["hand"] = json!([
        {"id":"STRIKE_IRONCLAD","kind":"card","stable_idx":4,"upgraded":false,"combat_vars":{"free_this_turn":1},
         "local_cost":{"base":1,"costs_x":false,"modifiers":[
            {"kind":"absolute","amount":0,"reduce_only":false,"expiration":6},
            {"kind":"relative","amount":1,"reduce_only":false,"expiration":0}]}},
        {"id":"DEFEND_IRONCLAD","kind":"card","stable_idx":5,"upgraded":false,
         "local_cost":{"base":0,"costs_x":false,"modifiers":[
            {"kind":"relative","amount":1,"reduce_only":false,"expiration":0}]}},
        {"id":"INFLAME","kind":"card","stable_idx":6,"upgraded":false}
    ]);
    let mut state = rebuild(&value);
    let preferred = state.player.hand[0];
    let fallback = state.player.hand[1];
    sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 2,
            target: None,
        },
    );
    assert_eq!(state.cards[preferred.get() as usize - 1].cost, 0);
    assert_eq!(
        state.cards[fallback.get() as usize - 1].cost,
        1,
        "zero-base cards are the later fallback pool, even when both currently cost energy"
    );
}

#[test]
fn touch_of_insanity_can_free_a_positive_local_cost_after_an_earlier_free_modifier() {
    let mut value = fixture();
    value["player"]["powers"] = json!([]);
    value["player"]["relics"] = json!([]);
    value["player"]["potions"] = json!(["TOUCH_OF_INSANITY"]);
    value["deck"]["hand"] = json!([{
        "id":"STRIKE_IRONCLAD","kind":"card","stable_idx":4,"upgraded":false,"combat_vars":{"free_this_turn":1},
        "local_cost":{"base":1,"costs_x":false,"modifiers":[
            {"kind":"absolute","amount":0,"reduce_only":false,"expiration":6},
            {"kind":"relative","amount":1,"reduce_only":false,"expiration":0}
        ]}
    }]);
    let mut state = rebuild(&value);
    let uid = state.player.hand[0];
    assert_eq!(state.cards[uid.get() as usize - 1].cost, 1);
    sts2_core::apply(
        &mut state,
        Action::DrinkPotion {
            potion_slot: 0,
            target: None,
        },
    );
    assert_eq!(
        state.cards[uid.get() as usize - 1].local_costs_after_expiration(),
        [0; 4]
    );
}

#[test]
fn displayed_power_progress_roundtrips_both_wire_names() {
    for (name, amount, display, counter, enemy) in [
        ("PANACHE", 14, 2, 2, false),
        ("TENDER", 1, 3, 3, false),
        ("SLOTH", 3, 1, 1, false),
        ("AUTOMATION", 2, 7, 3, false),
        ("ORBIT", 1, 2, 2, false),
        ("SLOW", 1, 40, 4, true),
        ("WITHERING_PRESENCE", 6, 2, 4, true),
    ] {
        let mut value = fixture();
        value["player"]["powers"] = json!([]);
        value["encounter"][0]["powers"] = json!([]);
        let path = if enemy { "encounter" } else { "player" };
        let powers = json!([{"name":name,"amount":amount,"display_amount":display}]);
        if enemy {
            value["encounter"][0]["powers"] = powers;
        } else {
            value["player"]["powers"] = powers;
        }
        let state = rebuild(&value);
        let power = state
            .powers
            .iter()
            .find(|p| p.kind.python_name() == name)
            .unwrap();
        assert_eq!(power.counter, counter, "{name}");
        let mut emitted = combat_state_to_value(&state).unwrap();
        let encoded = if enemy {
            &mut emitted["encounter"][0]["powers"][0]
        } else {
            &mut emitted["player"]["powers"][0]
        };
        assert_eq!(encoded["display_amount"], json!(display), "{path}/{name}");
        encoded.as_object_mut().unwrap().remove("counter");
        let back = rebuild(&emitted);
        assert_eq!(
            back.powers
                .iter()
                .find(|p| p.kind.python_name() == name)
                .unwrap()
                .counter,
            counter,
            "{name}: the display alias alone retains progress"
        );
    }
    let mut value = fixture();
    value["player"]["powers"] = json!([{"name":"PANACHE","amount":14,"display":5,"counter":-1}]);
    let state = rebuild(&value);
    let emitted = combat_state_to_value(&state).unwrap();
    assert_eq!(emitted["player"]["powers"][0]["display_amount"], json!(5));
    let back = rebuild(&emitted);
    assert_eq!(
        back.powers
            .iter()
            .find(|p| p.kind == PowerKind::Panache)
            .unwrap()
            .counter,
        -1,
        "native sentinel is preserved by the explicit counter"
    );
    value["player"]["powers"][0]["display_amount"] = json!(4);
    assert!(
        parse_json(&value).is_err(),
        "conflicting display aliases must not silently select one"
    );
    value["player"]["powers"][0]["display_amount"] = json!(5);
    assert!(parse_json(&value).is_ok());
}

#[test]
fn flatten_osty_reaction_preserves_modifier_order_and_turn_expiration() {
    use sts2_core::cards::{LocalCostModifier, LocalCostModifierKind};
    let mut value = fixture();
    value["game_version"] = json!("0.111.0");
    value["player"]["powers"] = json!([]);
    value["player"]["relics"] = json!([]);
    value["player"]["osty"] = json!({"alive":true,"hp":10,"max_hp":10,"block":0});
    let flatten = json!({"id":"FLATTEN","kind":"card","stable_idx":0,"upgraded":false,
        "local_cost":{"base":2,"costs_x":false,"modifiers":[]}});
    value["deck"]["pinned_top"] = json!([]);
    value["deck"]["hand"] = json!([
        {"id":"UNLEASH","kind":"card","stable_idx":1,"upgraded":false}, flatten.clone()]);
    value["deck"]["draw_pile"] = json!([flatten.clone()]);
    value["deck"]["discard_pile"] = json!([flatten.clone()]);
    value["deck"]["exhaust_pile"] = json!([flatten]);
    let mut state = rebuild(&value);
    let selected = state.player.hand[1];
    assert_eq!(sts2_core::hooks::effective_card_cost(&state, selected), 2);
    sts2_core::apply(
        &mut state,
        Action::PlayCard {
            hand_index: 0,
            target: Some(0),
        },
    );
    for card in state.cards.iter().filter(|c| c.card_id == CardId::Flatten) {
        assert_eq!(card.local_costs_after_expiration(), [0, 2, 0, 2]);
    }
    state.cards[selected.get() as usize - 1].add_local_cost_modifier(LocalCostModifier {
        kind: LocalCostModifierKind::Relative,
        amount: 1,
        reduce_only: false,
        expiration: 0,
    });
    assert_eq!(
        sts2_core::hooks::effective_card_cost(&state, selected),
        1,
        "a later modifier must not be hidden by a permanent free-cost predicate"
    );
    let generated =
        sts2_core::build_card_for_version(CardId::Flatten, false, state.game_version).unwrap();
    let generated = sts2_core::card_effects::spawn_card_into_hand(&mut state, generated);
    assert_eq!(sts2_core::hooks::effective_card_cost(&state, generated), 0);
    sts2_core::apply(&mut state, Action::EndTurn);
    assert_eq!(sts2_core::hooks::effective_card_cost(&state, selected), 3);
    assert_eq!(sts2_core::hooks::effective_card_cost(&state, generated), 2);
}

#[test]
fn when_played_cleanup_covers_discard_exhaust_power_and_hand_results() {
    for (id, base, route) in [
        (CardId::DefendIronclad, 1, "discard"),
        (CardId::Impervious, 2, "exhaust"),
        (CardId::BarricadeCard, 3, "power"),
        (CardId::StrikeIronclad, 1, "hand"),
    ] {
        let mut value = fixture();
        value["player"]["relics"] = json!([]);
        value["player"]["powers"] = if route == "hand" {
            json!([{"name":"FERAL","amount":1}])
        } else {
            json!([])
        };
        value["deck"]["hand"] = json!([{"id":id.to_python_name(),"kind":"card","stable_idx":0,"upgraded":false,
            "local_cost":{"base":base,"costs_x":false,"modifiers":[
                {"kind":"absolute","amount":0,"reduce_only":false,"expiration":6}]}}]);
        let mut state = rebuild(&value);
        let uid = state.player.hand[0];
        sts2_core::apply(
            &mut state,
            Action::PlayCard {
                hand_index: 0,
                target: if route == "hand" { Some(0) } else { None },
            },
        );
        assert!(state.pending_choice.is_none(), "{route}");
        assert_eq!(
            state.cards[uid.get() as usize - 1].local_costs_after_expiration(),
            [base; 4],
            "{route}"
        );
        match route {
            "discard" => assert!(state.player.discard.contains(&uid)),
            "exhaust" => assert!(state.player.exhaust.contains(&uid)),
            "hand" => assert!(state.player.hand.contains(&uid)),
            _ => assert!(!state.player.play.contains(&uid)),
        }
    }
}
