//! The deployed playout net is a version-1 weights file (32 state / 24
//! action features). Later feature versions must leave its logits
//! BIT-identical: this test replays a fixed set of decision states through the
//! built-in Ironclad net and compares every logit's bits with a golden dump
//! taken before feature version 2 existed (`tests/data/rollout_net_v1_golden.json`).
//!
//! Re-bless (only when the v1 features are changed on purpose, which retrains
//! nothing and silently changes the deployed search, so think twice):
//! `STS2_BLESS_ROLLOUT_GOLDEN=1 cargo nextest run -p sts2-mcts --test rollout_net_v1_golden`.

mod common;

use rand::{Rng, SeedableRng};
use sts2_core::{apply, is_terminal, legal_actions, CombatState};
use sts2_mcts::expert::END_TURN_PRIORITY;
use sts2_mcts::rollout_net::RolloutNet;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/rollout_net_v1_golden.json"
);
const NET: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/rollout_nets/ironclad.json"
);

/// Decision states of random walks from starter-deck fights, some with
/// potions, low HP and a tougher enemy, so the potion / lethal / incoming
/// features take more than one value.
fn states() -> Vec<CombatState> {
    let mut out = Vec::new();
    for seed in 1..=16u64 {
        let hp = 20 + 4 * i32::try_from(seed).expect("small");
        let mut s = common::build_vertical_slice_hp(seed, hp, 30 + 3 * hp);
        if seed % 3 == 0 {
            s.player.potions.push(sts2_core::PotionKind::FirePotion);
            s.player.potions.push(sts2_core::PotionKind::BlockPotion);
        }
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed * 31);
        for _ in 0..40 {
            if is_terminal(&s) {
                break;
            }
            let acts = legal_actions(&s);
            if acts.is_empty() {
                break;
            }
            out.push(s.clone());
            let i = rng.gen_range(0..acts.len());
            let _ = apply(&mut s, acts[i].clone());
        }
    }
    out
}

fn logit_bits(net: &RolloutNet, states: &[CombatState]) -> Vec<Vec<u32>> {
    states
        .iter()
        .map(|s| {
            let acts = legal_actions(s);
            net.logits(s, &acts, END_TURN_PRIORITY)
                .iter()
                .map(|l| l.to_bits())
                .collect()
        })
        .collect()
}

#[test]
fn v1_net_logits_are_bit_identical_to_the_golden_dump() {
    let net = RolloutNet::from_json(&std::fs::read_to_string(NET).expect("built-in net file"))
        .expect("v1 net loads");
    let states = states();
    assert!(states.len() > 200, "{} states", states.len());
    let got = logit_bits(&net, &states);
    if std::env::var_os("STS2_BLESS_ROLLOUT_GOLDEN").is_some() {
        std::fs::write(
            GOLDEN,
            serde_json::to_string(&serde_json::json!({ "logit_bits": got })).expect("json"),
        )
        .expect("write golden");
        return;
    }
    let want: Vec<Vec<u32>> = serde_json::from_value(
        serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(GOLDEN).expect(
            "golden dump (bless with STS2_BLESS_ROLLOUT_GOLDEN=1 on the pre-change engine)",
        ))
        .expect("json")["logit_bits"]
            .clone(),
    )
    .expect("logit_bits");
    assert_eq!(got.len(), want.len(), "state count");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "state {i}: logits changed bits");
    }
}
