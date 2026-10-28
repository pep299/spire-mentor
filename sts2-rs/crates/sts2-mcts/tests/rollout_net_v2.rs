//! Feature version 2 of the playout net (`sts2_mcts::rollout_net`): the
//! versioned weights file, the v2 features, the encounter embedding, the
//! zero-padded upgrade of a v1 net, and the honest-view / scored playouts the
//! RL trainer runs (`ppo.rollout_learn rl --view belief --reward search`).

mod common;

use common::build_vertical_slice_hp;
use rand::{Rng, SeedableRng};
use sts2_core::card_effects::apply_power_to_creature;
use sts2_core::{
    creature_by_id_mut, legal_actions, Action, CombatState, Intent, MonsterKind, PowerKind,
};
use sts2_mcts::expert::END_TURN_PRIORITY;
use sts2_mcts::rollout_net::{
    export_features_spec, playout_trajectories_with, FeatureSpec, FeatureVersion, RolloutNet,
    StepCtx, TrajOptions, ACTION_DIM, ACTION_DIM_V2, PLAYER_V2_POWERS, STATE_DIM, STATE_DIM_V2,
    V2_ACT, V2_ENEMY_POWERS, V2_INTENTS, V2_IS_BOSS, V2_PLAYER_POWERS, V2_X_TARGET,
};
use sts2_mcts::{LeafEval, MctsConfig, TerminalEvalMode};

fn mat(rng: &mut impl Rng, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| rng.gen_range(-0.5f32..0.5)).collect())
        .collect()
}

/// A random net of `version` (with the encounter embedding when `monsters`).
fn random_net(version: u32, seed: u64, monsters: bool) -> serde_json::Value {
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    let (sd, ad) = if version == 1 {
        (STATE_DIM, ACTION_DIM)
    } else {
        (STATE_DIM_V2, ACTION_DIM_V2)
    };
    let (h, d) = (12usize, 5usize);
    let keys = [
        "<unk>",
        "end_turn",
        "card:STRIKE_IRONCLAD",
        "card:DEFEND_IRONCLAD",
        "card:BASH",
    ];
    let mut net = serde_json::json!({
        "feature_version": version, "state_dim": sd, "action_dim": ad, "hidden": h,
        "emb_dim": d, "temperature": 1.0, "keys": keys,
        "w1": mat(&mut rng, h, sd), "b1": mat(&mut rng, 1, h)[0], "w2": mat(&mut rng, d, h),
        "b2": mat(&mut rng, 1, d)[0], "emb": mat(&mut rng, keys.len(), d),
        "bias": mat(&mut rng, 1, keys.len())[0], "proj": mat(&mut rng, d, ad),
        "lin": mat(&mut rng, 1, ad)[0],
    });
    if version == 2 {
        net["monster_act"] = serde_json::json!({"Nibbit": 1, "NOT_A_MONSTER": 3});
        if monsters {
            net["monster_keys"] = serde_json::json!(["Nibbit", "NOT_A_MONSTER"]);
            net["monster_emb"] = serde_json::json!(mat(&mut rng, 2, h));
        }
    }
    net
}

fn load(v: &serde_json::Value) -> Result<RolloutNet, String> {
    RolloutNet::from_json(&v.to_string())
}

/// A starter fight on round 1 with powers on both sides and a boss flag.
fn powered_state() -> CombatState {
    let mut s = build_vertical_slice_hp(9, 70, 90);
    s.is_boss = true;
    let enemy = s.enemy_uids[0];
    apply_power_to_creature(&mut s, enemy, PowerKind::Thorns, 3);
    apply_power_to_creature(&mut s, enemy, PowerKind::Artifact, 2);
    let player = s.player.creature_uid.expect("player creature");
    apply_power_to_creature(&mut s, player, PowerKind::DemonForm, 2);
    apply_power_to_creature(&mut s, player, PowerKind::Corruption, 1);
    s
}

#[test]
fn the_file_declares_its_version_and_dims() {
    assert_eq!(
        load(&random_net(1, 1, false))
            .expect("v1")
            .feature_spec()
            .version,
        FeatureVersion::V1
    );
    let v2 = load(&random_net(2, 1, true)).expect("v2");
    assert_eq!(v2.feature_spec().version, FeatureVersion::V2);
    assert_eq!(
        (v2.state_dim(), v2.action_dim()),
        (STATE_DIM_V2, ACTION_DIM_V2)
    );

    // A v1-sized file that claims v2, and vice versa.
    let mut wrong = random_net(1, 1, false);
    wrong["feature_version"] = 2.into();
    assert!(load(&wrong).unwrap_err().contains("feature dims"));
    let mut wrong = random_net(2, 1, false);
    wrong["feature_version"] = 1.into();
    assert!(load(&wrong).is_err());
    // No version key = v1 (every file written before v2).
    let mut legacy = random_net(1, 1, false);
    legacy
        .as_object_mut()
        .expect("obj")
        .remove("feature_version");
    assert!(load(&legacy).is_ok());
    let mut bad = random_net(1, 1, false);
    bad["feature_version"] = 7.into();
    assert!(load(&bad).unwrap_err().contains("feature_version"));
    // The per-species tables belong to v2, and v2 cannot do without the act.
    let mut v1_tables = random_net(1, 1, false);
    v1_tables["monster_act"] = serde_json::json!({"Nibbit": 1});
    assert!(load(&v1_tables).unwrap_err().contains("feature_version 2"));
    let mut no_act = random_net(2, 1, false);
    no_act.as_object_mut().expect("obj").remove("monster_act");
    assert!(load(&no_act).unwrap_err().contains("monster_act"));
    let mut upper = random_net(2, 1, false);
    upper["monster_act"] = serde_json::json!({"NIBBIT": 1});
    assert!(
        load(&upper).unwrap_err().contains("python_name"),
        "an act table matching no species is a naming mistake, not data"
    );
    let mut half = random_net(2, 1, true);
    half.as_object_mut().expect("obj").remove("monster_emb");
    assert!(load(&half).is_err());
}

#[test]
fn v2_features_read_boss_act_intent_and_powers() {
    let s = powered_state();
    let acts = legal_actions(&s);
    let spec = FeatureSpec::v2(&[(MonsterKind::Nibbit, 2)], &[(MonsterKind::Nibbit, 1)]);
    let ctx = StepCtx::with_spec(&s, &acts, &spec);
    let v1 = StepCtx::new(&s, &acts);
    assert_eq!(
        ctx.state[..STATE_DIM],
        v1.state[..STATE_DIM],
        "v2 extends v1"
    );
    assert!(v1.state[STATE_DIM..].iter().all(|v| *v == 0.0));
    assert!((ctx.state[V2_IS_BOSS] - 1.0).abs() < 1e-6);
    assert!(
        (ctx.state[V2_ACT] - 2.0 / 3.0).abs() < 1e-6,
        "act from the table"
    );
    let intent: f32 = ctx.state[V2_INTENTS..V2_INTENTS + 7].iter().sum();
    assert!(
        (intent - 1.0).abs() < 1e-6,
        "one intent type for the Nibbit"
    );
    assert!(
        (ctx.state[V2_ENEMY_POWERS] - 2.0 / 3.0).abs() < 1e-6,
        "Artifact 2"
    );
    assert!(
        (ctx.state[V2_ENEMY_POWERS + 1] - 3.0 / 5.0).abs() < 1e-6,
        "Thorns 3"
    );
    let pos = |k| {
        V2_PLAYER_POWERS
            + PLAYER_V2_POWERS
                .iter()
                .position(|(p, _)| *p == k)
                .expect("listed")
    };
    assert!((ctx.state[pos(PowerKind::DemonForm)] - 2.0 / 3.0).abs() < 1e-6);
    assert!((ctx.state[pos(PowerKind::Corruption)] - 1.0).abs() < 1e-6);
    assert_eq!(ctx.monsters[0], 1, "the Nibbit's embedding row");
    assert_eq!(ctx.species[0], Some(MonsterKind::Nibbit));

    // Per action: a targeted attack sees the target's Thorns and Artifact;
    // Corruption makes a Skill exhaust.
    let ex = export_features_spec(&s, &spec);
    assert_eq!(ex.state.len(), STATE_DIM_V2);
    let mut saw_attack = false;
    let mut saw_skill = false;
    for (a, _, x) in &ex.actions {
        assert_eq!(x.len(), ACTION_DIM_V2);
        if let Action::PlayCard {
            target: Some(_), ..
        } = a
        {
            saw_attack = true;
            assert!((x[V2_X_TARGET + 1] - 0.6).abs() < 1e-6, "{x:?}");
            assert!((x[V2_X_TARGET + 3] - 1.0).abs() < 1e-6);
        }
        if let Action::PlayCard { target: None, .. } = a {
            if x[9] > 0.5 {
                saw_skill = true;
                assert!(
                    (x[24] - 1.0).abs() < 1e-6,
                    "Skills exhaust under Corruption"
                );
            }
        }
    }
    assert!(saw_attack && saw_skill);
}

/// The beaten Waterfall Giant sits at a 999,999,999 HP sentinel and still
/// acts: ABOUT_TO_BLOW, then EXPLODE's DeathBlow (`WaterfallGiant.cs`
/// `TriggerAboutToBlowState` / `ExplodeMove`). v2 keeps such an enemy in
/// play, so its deadliest turn does not read as "no enemies".
#[test]
fn a_beaten_sentinel_keeps_its_v2_intent_species_act_and_target_facts() {
    let mut s = powered_state();
    let enemy = s.enemy_uids[0];
    let c = creature_by_id_mut(&mut s.creatures, enemy.0).expect("enemy");
    c.current_hp = 999_999_999;
    c.max_hp = 999_999_999;
    c.intent = Some(Intent::DeathBlow { damage: 47 });
    assert!(c.is_beaten_sentinel());
    let acts = legal_actions(&s);
    let spec = FeatureSpec::v2(&[(MonsterKind::Nibbit, 2)], &[(MonsterKind::Nibbit, 1)]);
    let ctx = StepCtx::with_spec(&s, &acts, &spec);
    let mhp = s.player.max_hp as f32;
    assert!((ctx.state[V2_ACT] - 2.0 / 3.0).abs() < 1e-6, "act");
    assert!(
        (ctx.state[V2_INTENTS] - 1.0).abs() < 1e-6,
        "DeathBlow is an attack"
    );
    assert!(
        (ctx.state[V2_INTENTS + 7] - 47.0 / mhp).abs() < 1e-6,
        "its damage"
    );
    assert_eq!(ctx.monsters[0], 1, "the sentinel's embedding row");
    assert_eq!(ctx.species[0], Some(MonsterKind::Nibbit));
    assert!(
        (ctx.state[V2_ENEMY_POWERS + 1] - 3.0 / 5.0).abs() < 1e-6,
        "its Thorns"
    );
    let ex = export_features_spec(&s, &spec);
    let targeted: Vec<_> = ex
        .actions
        .iter()
        .filter(|(a, _, _)| {
            matches!(
                a,
                Action::PlayCard {
                    target: Some(_),
                    ..
                }
            )
        })
        .collect();
    assert!(!targeted.is_empty(), "the sentinel can be targeted");
    for (_, _, x) in targeted {
        assert!(
            (x[V2_X_TARGET + 4] - 1.0).abs() < 1e-6,
            "the target intends to attack"
        );
    }
}

/// The act reads every enemy the fight has spawned (the dead stay in
/// `enemy_uids`), and an encounter roster lifts it: two act-1 constructs
/// are act 1 until the whole act-3 Construct Menagerie (Punch + Cubex x2)
/// is on the board, and it stays act 3 after one of them dies.
#[test]
fn the_act_reads_spawned_enemies_and_whole_encounter_rosters() {
    use sts2_content::monster_dispatch::MonsterSpawnArgs;
    use sts2_content::run_bridge::spawn_roster;
    use MonsterKind::{CubexConstruct, Nibbit, PunchConstruct};
    let acts_table = [(Nibbit, 1), (PunchConstruct, 1), (CubexConstruct, 1)];
    let rosters = vec![
        (1, vec![PunchConstruct]),
        (1, vec![CubexConstruct]),
        (3, vec![PunchConstruct, CubexConstruct, CubexConstruct]),
    ];
    let spec = FeatureSpec::v2(&acts_table, &[]).with_encounters(&rosters);
    let plain = FeatureSpec::v2(&acts_table, &[]);
    let act = |s: &CombatState, spec: &FeatureSpec| {
        StepCtx::with_spec(s, &legal_actions(s), spec).state[V2_ACT] * 3.0
    };
    let mut s = build_vertical_slice_hp(9, 70, 90);
    spawn_roster(
        &mut s,
        &[
            (PunchConstruct, MonsterSpawnArgs::None, 40),
            (CubexConstruct, MonsterSpawnArgs::None, 30),
        ],
    );
    assert!((act(&s, &spec) - 1.0).abs() < 1e-6, "not the menagerie yet");
    let cubex = spawn_roster(&mut s, &[(CubexConstruct, MonsterSpawnArgs::None, 30)])[0];
    assert!((act(&s, &spec) - 3.0).abs() < 1e-6, "the whole roster");
    assert!((act(&s, &plain) - 1.0).abs() < 1e-6, "species alone say 1");
    creature_by_id_mut(&mut s.creatures, cubex.0)
        .expect("cubex")
        .current_hp = 0;
    assert!((act(&s, &spec) - 3.0).abs() < 1e-6, "a dead member counts");

    // The same through a weights file's `encounter_act`.
    let mut file = random_net(2, 4, false);
    file["monster_act"] =
        serde_json::json!({"Nibbit": 1, "PunchConstruct": 1, "CubexConstruct": 1});
    file["encounter_act"] = serde_json::json!([
        [1, ["PunchConstruct"]],
        [3, ["PunchConstruct", "CubexConstruct", "CubexConstruct"]],
        [2, ["NOT_A_MONSTER"]],
    ]);
    let net = load(&file).expect("v2 with encounter_act");
    assert!((act(&s, net.feature_spec()) - 3.0).abs() < 1e-6);
    let mut v1 = random_net(1, 4, false);
    v1["encounter_act"] = serde_json::json!([[3, ["PunchConstruct"]]]);
    assert!(load(&v1).unwrap_err().contains("feature_version 2"));
}

/// A v1 net loaded as v2 with zero weights on every new input plays the
/// same policy: what `rollout_learn rl --feature-version 2` relies on to
/// start a v2 run from a v1 champion.
#[test]
fn a_zero_padded_v1_net_keeps_its_logits_under_v2() {
    let v1 = random_net(1, 5, false);
    let mut v2 = v1.clone();
    v2["feature_version"] = 2.into();
    v2["state_dim"] = STATE_DIM_V2.into();
    v2["action_dim"] = ACTION_DIM_V2.into();
    let pad = |rows: &serde_json::Value, to: usize| -> serde_json::Value {
        rows.as_array()
            .expect("rows")
            .iter()
            .map(|r| {
                let mut r: Vec<f32> = serde_json::from_value(r.clone()).expect("row");
                r.resize(to, 0.0);
                r
            })
            .collect::<Vec<_>>()
            .into()
    };
    v2["w1"] = pad(&v1["w1"], STATE_DIM_V2);
    v2["proj"] = pad(&v1["proj"], ACTION_DIM_V2);
    let mut lin: Vec<f32> = serde_json::from_value(v1["lin"].clone()).expect("lin");
    lin.resize(ACTION_DIM_V2, 0.0);
    v2["lin"] = lin.into();
    v2["monster_act"] = serde_json::json!({"Nibbit": 1});
    let (a, b) = (load(&v1).expect("v1"), load(&v2).expect("v2"));
    let s = powered_state();
    let acts = legal_actions(&s);
    let (la, lb) = (
        a.logits(&s, &acts, END_TURN_PRIORITY),
        b.logits(&s, &acts, END_TURN_PRIORITY),
    );
    for (x, y) in la.iter().zip(&lb) {
        assert!((x - y).abs() < 1e-5, "{la:?} vs {lb:?}");
    }
}

/// The encounter embedding adds the species' row to the hidden layer, so
/// it moves the logits, and `explain` reports exactly what `logits` returns.
#[test]
fn the_encounter_embedding_moves_the_logits_and_explain_agrees() {
    let with = load(&random_net(2, 9, true)).expect("with");
    let mut without_json = random_net(2, 9, true);
    let obj = without_json.as_object_mut().expect("obj");
    obj.remove("monster_keys");
    obj.remove("monster_emb");
    let without = load(&without_json).expect("without");
    let s = powered_state();
    let acts = legal_actions(&s);
    let (lw, lo) = (
        with.logits(&s, &acts, END_TURN_PRIORITY),
        without.logits(&s, &acts, END_TURN_PRIORITY),
    );
    assert!(lw.iter().zip(&lo).any(|(a, b)| (a - b).abs() > 1e-4));
    let ex = with.explain(&s);
    assert_eq!(ex.monsters[0], 1, "Nibbit is monster row 1");
    assert_eq!(ex.actions.len(), lw.len());
    for ((_, _, _, l), want) in ex.actions.iter().zip(&lw) {
        assert_eq!(l.to_bits(), want.to_bits());
    }
}

/// Honest-view playouts: each playout draws its own deck order, the same
/// seed replays the same playout, and a scored playout carries the search's
/// terminal value (`playout_return`) in the band its outcome implies.
#[test]
fn belief_playouts_are_seeded_and_scored_like_the_search() {
    let net = load(&random_net(2, 3, false)).expect("net");
    let start = build_vertical_slice_hp(4, 40, 70);
    let jobs: Vec<_> = (0..24).map(|i| (start.clone(), 1000 + i)).collect();
    let cfg = MctsConfig {
        terminal_eval_mode: TerminalEvalMode::RichBoth,
        ..MctsConfig::default()
    };
    let leaf = LeafEval::default();
    let opts = TrajOptions {
        temperature: 1.0,
        no_early_end_turn: false,
        max_steps: 600,
        belief: true,
        score: Some((&cfg, &leaf)),
    };
    let a = playout_trajectories_with(&net, &jobs, &opts);
    let b = playout_trajectories_with(&net, &jobs, &opts);
    let sig = |t: &[sts2_mcts::rollout_net::Trajectory]| {
        t.iter()
            .map(|t| (t.steps.len(), t.won, t.search_value.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(sig(&a), sig(&b), "same seeds, same playouts");
    for t in &a {
        assert!(t.search_value.is_finite());
        if t.won {
            assert!((1.0..=1.5).contains(&t.search_value), "{}", t.search_value);
        } else {
            assert!(
                t.search_value < -1.0 && t.search_value >= -1.45,
                "{}",
                t.search_value
            );
        }
    }
    // A raw run (the true draw order and RNG) with the same policy seeds
    // plays differently from the belief run on at least one job.
    let raw = playout_trajectories_with(
        &net,
        &jobs,
        &TrajOptions {
            belief: false,
            ..opts
        },
    );
    assert_ne!(sig(&a), sig(&raw), "the belief view changes the playouts");
}
