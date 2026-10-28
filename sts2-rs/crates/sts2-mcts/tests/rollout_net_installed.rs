//! A search under `RolloutPolicyKind::Net` runs on a net installed with
//! `set_global_net`. The install is process-wide and permanent, so this test
//! has its own binary: sharing one with the built-in-net tests would make
//! them depend on test order under a shared-process runner.

mod common;

use common::build_vertical_slice_hp;
use sts2_core::legal_actions;
use sts2_mcts::rollout_net::{heuristic_mirror_net, set_global_net};
use sts2_mcts::{Mcts, MctsConfig, RolloutPolicyKind};

#[test]
fn search_runs_on_an_installed_net() {
    assert!(set_global_net(heuristic_mirror_net()));
    let state = build_vertical_slice_hp(5, 80, 40);
    let cfg = MctsConfig {
        max_simulations: 300,
        max_rollout_depth: 200,
        rollout_policy: RolloutPolicyKind::Net,
        seed: 11,
        ..MctsConfig::default()
    };
    let a1 = Mcts::new(state.clone(), cfg).search();
    let a2 = Mcts::new(state.clone(), cfg).search();
    assert_eq!(a1, a2, "deterministic for a fixed seed");
    assert!(legal_actions(&state).contains(&a1));
}
