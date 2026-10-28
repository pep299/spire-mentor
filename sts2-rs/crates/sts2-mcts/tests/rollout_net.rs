//! The playout net (`sts2_mcts::rollout_net`): the training
//! features line up with `legal_actions`, and the built-in nets are the ones
//! searches use. `rollout_net_installed.rs` installs a process-wide net, so it
//! lives in its own test binary.

mod common;

use common::build_vertical_slice_hp;
use sts2_core::{legal_actions, Action, CharacterId, CombatState};
use sts2_mcts::rollout_net::{
    builtin_net, export_features, heuristic_mirror_net, net_for, playout_trajectories, ActionKey,
};
use sts2_mcts::{Mcts, MctsConfig, RolloutPolicyKind};

/// `STS2_ROLLOUT_NET` replaces every built-in net, so a test about the
/// built-in nets cannot run under it; say so instead of failing obscurely.
fn no_experiment_net() {
    assert!(
        std::env::var_os("STS2_ROLLOUT_NET").is_none(),
        "unset STS2_ROLLOUT_NET to run this test: it replaces every built-in net"
    );
}

/// Root statistics of a 300-sim search of `state` under `policy`.
fn root_stats(state: &CombatState, policy: RolloutPolicyKind) -> String {
    let cfg = MctsConfig {
        max_simulations: 300,
        max_rollout_depth: 200,
        rollout_policy: policy,
        seed: 11,
        ..MctsConfig::default()
    };
    let mut mcts = Mcts::new(state.clone(), cfg);
    mcts.search();
    format!("{:?}", mcts.root_outcomes())
}

/// The deployed dialect selects `Net` for every character. A fight with no
/// character has no built-in net (every character has one since GH #823 and
/// #824), so it must search exactly as under `Heuristic`, with the same random draws and the same
/// root statistics.
#[test]
fn net_policy_without_a_net_is_the_heuristic() {
    let mut state = build_vertical_slice_hp(5, 80, 40);
    state.player.character = None;
    no_experiment_net();
    assert!(net_for(&state).is_none(), "no character, no net");
    assert_eq!(
        root_stats(&state, RolloutPolicyKind::Net),
        root_stats(&state, RolloutPolicyKind::Heuristic)
    );
}

/// A character has a built-in net exactly when its file
/// `data/rollout_nets/<character>.json` is in the checkout, and every
/// compiled-in net parses.
#[test]
fn built_in_nets_load() {
    for (c, name) in [
        (CharacterId::Ironclad, "ironclad"),
        (CharacterId::Silent, "silent"),
        (CharacterId::Defect, "defect"),
        (CharacterId::Regent, "regent"),
        (CharacterId::Necrobinder, "necrobinder"),
    ] {
        let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../data/rollout_nets/{name}.json"));
        assert_eq!(builtin_net(c).is_some(), file.is_file(), "{c:?}");
    }
}

/// A built-in net loads at the temperature its full-run gate measured
/// (T = 1; the trainer once wrote 0.25 by default, the GH #777 gotcha), and
/// that character's searches under `Net` actually use it.
#[allow(dead_code)]
fn searches_use_the_built_in_net(character: CharacterId) {
    let net = builtin_net(character).expect("built-in net");
    assert!(
        (net.temperature - 1.0).abs() < f32::EPSILON,
        "T = {}",
        net.temperature
    );
    assert!(net.rows > 100, "{} embedding rows", net.rows);
    let mut state = build_vertical_slice_hp(5, 80, 40);
    state.player.character = Some(character);
    no_experiment_net();
    assert!(net_for(&state).is_some());
    assert_ne!(
        root_stats(&state, RolloutPolicyKind::Net),
        root_stats(&state, RolloutPolicyKind::Heuristic)
    );
}

#[cfg(rollout_net_ironclad)]
#[test]
fn ironclad_searches_use_the_built_in_net() {
    searches_use_the_built_in_net(CharacterId::Ironclad);
}

#[cfg(rollout_net_silent)]
#[test]
fn silent_searches_use_the_built_in_net() {
    searches_use_the_built_in_net(CharacterId::Silent);
}

#[cfg(rollout_net_defect)]
#[test]
fn defect_searches_use_the_built_in_net() {
    searches_use_the_built_in_net(CharacterId::Defect);
}

#[cfg(rollout_net_necrobinder)]
#[test]
fn necrobinder_searches_use_the_built_in_net() {
    searches_use_the_built_in_net(CharacterId::Necrobinder);
}

/// GH #822: shipped on the owner's call despite a NULL gate (A0 +8.5pp,
/// A9 read +3.5pp; neither significant at p < 0.01).
#[cfg(rollout_net_regent)]
#[test]
fn regent_searches_use_the_built_in_net() {
    searches_use_the_built_in_net(CharacterId::Regent);
}

#[test]
fn features_follow_legal_action_order() {
    let state = build_vertical_slice_hp(3, 80, 40);
    let legal = legal_actions(&state);
    let (s, rows) = export_features(&state);
    assert_eq!(rows.len(), legal.len());
    assert!(s.iter().all(|v| v.is_finite()));
    assert!((s[0] - 1.0).abs() < 1e-6, "full HP");
    for ((a, key, x), l) in rows.iter().zip(&legal) {
        assert_eq!(a, l);
        match a {
            Action::PlayCard { .. } => {
                assert!(matches!(key, ActionKey::Card { .. }));
                assert!((x[0] - 1.0).abs() < 1e-6);
            }
            Action::EndTurn => {
                assert_eq!(*key, ActionKey::EndTurn);
                assert!((x[1] - 1.0).abs() < 1e-6);
            }
            _ => {}
        }
    }
}

#[test]
fn trajectories_record_allowed_choices() {
    let net = heuristic_mirror_net();
    let jobs: Vec<_> = (0..4)
        .map(|i| (build_vertical_slice_hp(7 + i, 80, 40), 100 + i))
        .collect();
    let trajs = playout_trajectories(&net, &jobs, 1.0, true, 600);
    assert_eq!(trajs.len(), jobs.len());
    for t in &trajs {
        assert!(!t.steps.is_empty());
        assert!((0.0..=1.0).contains(&t.removed));
        for step in &t.steps {
            let n = step.rows.len();
            assert_eq!(step.x.len(), n);
            assert_eq!(step.allowed.len(), n);
            assert!(step.allowed.iter().filter(|ok| **ok).count() >= 2);
            assert!(
                step.allowed[usize::from(step.chosen)],
                "sampled a masked action"
            );
        }
    }
    // Same seeds, same trajectories.
    let again = playout_trajectories(&net, &jobs, 1.0, true, 600);
    let lens = |ts: &[sts2_mcts::rollout_net::Trajectory]| {
        ts.iter()
            .map(|t| (t.steps.len(), t.won))
            .collect::<Vec<_>>()
    };
    assert_eq!(lens(&trajs), lens(&again));
}
