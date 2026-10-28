//! Pro-trained plan scorer forward pass and the turn-plan root filter
//! (GH #830, root-filter screen).

mod common;

use common::{fixture, push_draw};
use serde_json::{json, Value};
use sts2_core::cards::CardId;
use sts2_core::potions::PotionKind;
use sts2_core::{legal_actions, Action, CombatState};
use sts2_mcts::turn_plan::{enumerate_plans, PlanCap};
use sts2_mcts::turn_plan_features::{anchors, deck_context, plan_cards, plan_features, PLAN_DIM};
use sts2_mcts::turn_plan_scorer::{root_filter, FilterFallback, PlanScorer, FORMAT};
use sts2_mcts::{HiddenInfoMode, Mcts, MctsConfig};

fn lin(w: Vec<Vec<f32>>, b: Vec<f32>) -> Value {
    json!({"w": w, "b": b})
}

fn eye(n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

/// A scorer whose plan embedding is `[damage (f3), block (f0)] + mean
/// card embedding` and whose score is `relu(w_dmg·emb0 + w_blk·emb1 +
/// w_ctx·ctx_n0)`, through identity-ish layers.
fn scorer_json(w_dmg: f32, w_blk: f32, w_ctx: f32, card_emb: Vec<Vec<f32>>) -> Value {
    let mut l2 = vec![vec![0.0; PLAN_DIM]; 2];
    l2[0][3] = 1.0;
    l2[1][0] = 1.0;
    let mut s1 = vec![0.0; 8];
    s1[0] = w_ctx;
    s1[6] = w_dmg;
    s1[7] = w_blk;
    json!({
        "format": FORMAT,
        "ctx_mean": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        "ctx_std": [2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "plan_mean": vec![0.0; PLAN_DIM],
        "plan_std": vec![1.0; PLAN_DIM],
        "plan_l1": lin(eye(PLAN_DIM), vec![0.0; PLAN_DIM]),
        "plan_l2": lin(l2, vec![0.0, 0.0]),
        "card_vocab": {"StrikeIronclad": 1, "DefendIronclad": 2},
        "card_emb": card_emb,
        "score_l1": lin(vec![s1], vec![0.0]),
        "score_l2": lin(vec![vec![1.0]], vec![0.0]),
    })
}

fn damage_scorer() -> PlanScorer {
    let emb = vec![vec![0.0, 0.0]; 3];
    PlanScorer::from_json(&scorer_json(1.0, 0.0, 0.0, emb).to_string()).expect("valid scorer")
}

#[test]
fn score_matches_a_hand_computed_forward_pass() {
    // UNK (row 0) = [10, 0], Strike = [2, 4], Defend = [0, 6].
    let emb = vec![vec![10.0, 0.0], vec![2.0, 4.0], vec![0.0, 6.0]];
    let s = PlanScorer::from_json(&scorer_json(1.0, 0.5, 2.0, emb).to_string()).unwrap();
    let mut f = [0.0f32; PLAN_DIM];
    f[3] = 6.0; // damage
    f[0] = 5.0; // block
    f[1] = -3.0; // negative: zeroed by the first ReLU
    let ctx = [2.5, 0.0, 0.0, 0.0, 0.0, 0.0]; // ctx_n0 = (2.5 - 0.5) / 2 = 1
    let cards = ["StrikeIronclad".to_string(), "Unknown".to_string()];
    // bag = mean([2,4], [10,0]) = [6, 2]; emb = [6+6, 5+2] = [12, 7]
    // score = relu(2·1 + 1·12 + 0.5·7) = 17.5
    assert!((s.score(&ctx, &f, &cards) - 17.5).abs() < 1e-6);
    // No cards: the bag adds nothing. relu(2 + 6 + 2.5) = 10.5
    assert!((s.score(&ctx, &f, &[]) - 10.5).abs() < 1e-6);
}

#[test]
fn from_json_rejects_a_wrong_format_and_bad_shapes() {
    let emb = vec![vec![0.0, 0.0]; 3];
    let mut v = scorer_json(1.0, 0.0, 0.0, emb.clone());
    v["format"] = json!("nope");
    assert!(PlanScorer::from_json(&v.to_string())
        .unwrap_err()
        .contains("format"));
    let mut v = scorer_json(1.0, 0.0, 0.0, emb.clone());
    v["plan_mean"] = json!([0.0, 1.0]);
    assert!(PlanScorer::from_json(&v.to_string()).is_err());
    let mut v = scorer_json(1.0, 0.0, 0.0, emb);
    v["card_vocab"] = json!({"StrikeIronclad": 9});
    assert!(PlanScorer::from_json(&v.to_string()).is_err());
}

/// The exported scorer (`ppo.turnplan.pro_filter`) reproduces the torch
/// scores of recorded positions.
#[test]
fn exported_scorer_matches_torch_on_recorded_positions() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/turnplan_root_filter"
    );
    let scorer = PlanScorer::from_json(
        &std::fs::read_to_string(format!("{dir}/root_filter.json")).expect("weights"),
    )
    .expect("valid export");
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{dir}/parity.json")).expect("fixture"),
    )
    .unwrap();
    let mut n = 0;
    let mut worst = 0.0f64;
    for pos in fixture["positions"].as_array().unwrap() {
        let ctx: [f32; 6] = serde_json::from_value(pos["ctx"].clone()).unwrap();
        for (plan, want) in pos["plans"]
            .as_array()
            .unwrap()
            .iter()
            .zip(pos["scores"].as_array().unwrap())
        {
            let f: [f32; PLAN_DIM] = serde_json::from_value(plan["features"].clone()).unwrap();
            let cards: Vec<String> = serde_json::from_value(plan["cards"].clone()).unwrap();
            let got = f64::from(scorer.score(&ctx, &f, &cards));
            worst = worst.max((got - want.as_f64().unwrap()).abs());
            n += 1;
        }
    }
    eprintln!("parity: max abs diff {worst:e} over {n} plans");
    assert!(n > 100, "only {n} recorded plans");
    assert!(worst <= 1e-5, "max |rust - torch| = {worst} over {n} plans");
}

fn five_cards() -> CombatState {
    fixture(
        &[
            CardId::StrikeIronclad,
            CardId::DefendIronclad,
            CardId::Bash,
            CardId::StrikeIronclad,
            CardId::Inflame,
        ],
        60,
        &[PotionKind::BlockPotion],
        3,
    )
}

#[test]
fn a_forced_root_is_left_alone() {
    let state = fixture(&[], 40, &[], 3);
    let f = root_filter(&state, &damage_scorer(), 1, PlanCap::default());
    assert_eq!(f.fallback, Some(FilterFallback::Forced));
    assert!(f.allowed.is_empty());
}

#[test]
fn a_capped_enumeration_falls_back() {
    let cap = PlanCap {
        max_plans: 2,
        max_nodes: 20_000,
    };
    let f = root_filter(&five_cards(), &damage_scorer(), 1, cap);
    assert_eq!(f.fallback, Some(FilterFallback::Cap));
    assert!(f.allowed.is_empty());
}

#[test]
fn a_large_k_allows_every_legal_action() {
    let state = five_cards();
    let f = root_filter(&state, &damage_scorer(), 10_000, PlanCap::default());
    assert_eq!(f.fallback, None);
    let legal = legal_actions(&state);
    assert_eq!(f.allowed.len(), legal.len());
    assert!(f
        .allowed
        .iter()
        .all(|(a, via_top)| legal.contains(a) && *via_top));
    assert_eq!(f.kept_n, f.plans_n);
}

/// K = 1 keeps the top-scored plan and the anchor plans: every allowed
/// action starts one of them, every hand copy of an allowed card is
/// allowed, and the top plan's first action is marked as a top-K start.
#[test]
fn k1_keeps_the_top_plan_and_the_anchors() {
    let state = five_cards();
    let scorer = damage_scorer();
    let f = root_filter(&state, &scorer, 1, PlanCap::default());
    assert_eq!(f.fallback, None);
    let set = enumerate_plans(&state, HiddenInfoMode::BeliefDraw, PlanCap::default());
    let ctx = deck_context(&state).ctx;
    let scores: Vec<f32> = set
        .plans
        .iter()
        .map(|p| scorer.score(&ctx, &plan_features(&state, p), &plan_cards(&state, p)))
        .collect();
    let top = (0..scores.len())
        .reduce(|b, i| if scores[i] > scores[b] { i } else { b })
        .unwrap();
    let flags = anchors(&state, &set);
    let kept: Vec<usize> = (0..set.plans.len())
        .filter(|&i| {
            let f = flags[i];
            i == top || f.lethal || f.max_block || f.max_damage || f.heuristic || f.rl2
        })
        .collect();
    assert_eq!(f.kept_n, kept.len());
    assert!(
        f.allowed.len() < legal_actions(&state).len(),
        "nothing filtered: {f:?}"
    );
    let first_card = |a: &Action| match a {
        Action::PlayCard { hand_index, target } => {
            let uid = state.player.hand[usize::from(*hand_index)];
            Some((state.cards[uid.0.get() as usize - 1].card_id, *target))
        }
        _ => None,
    };
    for (a, _) in &f.allowed {
        assert!(
            kept.iter().any(|&i| {
                let p = &set.plans[i].actions[0];
                p == a || (first_card(p).is_some() && first_card(p) == first_card(a))
            }),
            "{a:?} starts no kept plan"
        );
    }
    let top_first = &set.plans[top].actions[0];
    assert!(f
        .allowed
        .iter()
        .any(|(a, via)| *via && (a == top_first || first_card(a) == first_card(top_first))));
    // Both Strike copies or neither.
    let strikes: Vec<_> = f
        .allowed
        .iter()
        .filter(|(a, _)| first_card(a).map(|c| c.0) == Some(CardId::StrikeIronclad))
        .collect();
    assert!(strikes.is_empty() || strikes.len() == 2, "{strikes:?}");
}

#[test]
fn the_filter_reads_no_hidden_draw_order() {
    let mut a = fixture(
        &[CardId::PommelStrike, CardId::DefendIronclad, CardId::Bash],
        40,
        &[],
        3,
    );
    let mut b = a.clone();
    push_draw(&mut a, &[CardId::StrikeIronclad, CardId::Bash]);
    push_draw(&mut b, &[CardId::Bash, CardId::StrikeIronclad]);
    let fa = root_filter(&a, &damage_scorer(), 1, PlanCap::default());
    let fb = root_filter(&b, &damage_scorer(), 1, PlanCap::default());
    assert_eq!(fa, fb);
}

#[test]
fn with_root_allowed_restricts_only_the_root() {
    let state = five_cards();
    let legal = legal_actions(&state);
    let allowed = vec![legal[0].clone(), Action::EndTurn];
    let config = MctsConfig {
        max_simulations: 300,
        seed: 3,
        hidden_info: HiddenInfoMode::BeliefDraw,
        ..MctsConfig::default()
    };
    let mut m = Mcts::new(state.clone(), config).with_root_allowed(&allowed);
    let pick = m.search_n(300);
    assert!(allowed.contains(&pick), "{pick:?}");
    let outcomes = m.root_outcomes();
    assert!(!outcomes.is_empty());
    assert!(
        outcomes.iter().all(|o| allowed.contains(&o.action)),
        "{outcomes:?}"
    );
}

#[test]
fn an_allowed_set_matching_nothing_leaves_the_root_unrestricted() {
    let state = five_cards();
    let config = MctsConfig {
        max_simulations: 200,
        seed: 3,
        hidden_info: HiddenInfoMode::BeliefDraw,
        ..MctsConfig::default()
    };
    let bogus = [Action::DrinkPotion {
        potion_slot: 7,
        target: None,
    }];
    let mut m = Mcts::new(state.clone(), config).with_root_allowed(&bogus);
    m.search_n(200);
    assert_eq!(m.root_outcomes().len(), {
        let mut n = Mcts::new(state, config);
        n.search_n(200);
        n.root_outcomes().len()
    });
}
