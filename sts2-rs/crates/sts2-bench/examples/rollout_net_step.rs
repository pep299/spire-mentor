//! `rollout_net_step` — per-step cost of the playout net
//! (`sts2_mcts::rollout_net`), the micro-benchmark behind GH #777's feature
//! versions.
//!
//! Builds Act-1 elite and boss fights for random full-pool Ironclad decks,
//! plays each out once with the net to collect its decision states, then
//! times two things on that fixed state set:
//!
//! - `net`: `RolloutNet::logits` per decision (state features, the heuristic
//!   priorities the net reads, action features and the logits) — what a
//!   playout step pays for the net on top of the engine;
//! - `playout`: whole playouts (`legal_actions` + net pick + `apply`) per step.
//!
//! ```sh
//! cargo build --release --example rollout_net_step -p sts2-bench
//! taskset -c 20 target/release/examples/rollout_net_step [--net PATH] [--decks N] [--reps R]
//! ```
//!
//! `--net` defaults to the built-in Ironclad net. `sink=` hashes the bits of
//! every logit, so two builds that print the same sink computed identical
//! logits on identical states (the bit-identity check for feature-version
//! refactors).

#[path = "common/mod.rs"]
mod common;

use std::time::Instant;

use rand::SeedableRng;
use sts2_content::cards::spec_index::IRONCLAD_SPECS;
use sts2_content::cards::CardSpec;
use sts2_content::encounters::act1_overgrowth::ACT1_ENCOUNTERS;
use sts2_content::encounters::EncounterTier;
use sts2_core::{apply, is_terminal, legal_actions, Action, CombatState};
use sts2_mcts::expert::END_TURN_PRIORITY;
use sts2_mcts::rollout_net::{pick_with, RolloutNet};

fn random_deck(seed: u64, size: usize) -> Vec<CardSpec> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let mut rng = sts2_core::dotnet_rng::Rng::new((seed ^ 0xDEC0_DEC0_DEC0_DEC0) as i32);
    (0..size)
        .map(|_| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                clippy::cast_sign_loss
            )]
            let idx = rng.next_int_exclusive(0, IRONCLAD_SPECS.len() as i32) as usize;
            IRONCLAD_SPECS[idx]()
        })
        .collect()
}

fn fnv1a(hash: u64, bytes: &[u8]) -> u64 {
    let mut h = hash;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// One playout of `state` with `net`; returns the decision states it visited
/// (each with its legal actions) and the number of steps.
fn play(
    net: &RolloutNet,
    state: &CombatState,
    seed: u64,
) -> (Vec<(CombatState, Vec<Action>)>, u32) {
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut s = state.clone();
    let mut out = Vec::new();
    let mut steps = 0u32;
    while !is_terminal(&s) && steps < 600 {
        let acts = legal_actions(&s);
        if acts.is_empty() {
            break;
        }
        let i = pick_with(
            net,
            &mut rng,
            &s,
            &acts,
            END_TURN_PRIORITY,
            net.temperature,
            net.no_early_end_turn,
        );
        out.push((s.clone(), acts.to_vec()));
        let _ = apply(&mut s, acts[i].clone());
        steps += 1;
    }
    (out, steps)
}

fn main() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let default_net = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/rollout_nets/ironclad.json"
    );
    let path: String = arg(&args, "--net", default_net.to_owned());
    let n_decks: u64 = arg(&args, "--decks", 6);
    let reps: u32 = arg(&args, "--reps", 20);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let net = RolloutNet::from_json(&text).unwrap_or_else(|e| panic!("{path}: {e}"));

    let fights: Vec<CombatState> = ACT1_ENCOUNTERS
        .iter()
        .filter(|e| matches!(e.tier, EncounterTier::Elite | EncounterTier::Boss))
        .flat_map(|e| {
            (0..n_decks).map(move |d| {
                let deck = random_deck(d.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x51, 22);
                common::build_ironclad_state_with_deck(e.id, d + 1, &deck)
            })
        })
        .collect();
    let mut states = Vec::new();
    for (i, f) in fights.iter().enumerate() {
        states.extend(play(&net, f, 1000 + i as u64).0);
    }

    // Net cost per decision.
    let mut sink = 0xCBF2_9CE4_8422_2325_u64;
    for (s, a) in &states {
        for l in net.logits(s, a, END_TURN_PRIORITY) {
            sink = fnv1a(sink, &l.to_bits().to_le_bytes());
        }
    }
    let t = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..reps {
        for (s, a) in &states {
            acc += net.logits(s, a, END_TURN_PRIORITY).iter().sum::<f32>();
        }
    }
    let net_ns = t.elapsed().as_secs_f64() * 1e9 / (f64::from(reps) * states.len() as f64);

    // Whole playout steps.
    let t = Instant::now();
    let mut steps = 0u64;
    for r in 0..reps.div_ceil(4) {
        for (i, f) in fights.iter().enumerate() {
            steps += u64::from(play_steps(&net, f, 7_000 + u64::from(r) * 997 + i as u64));
        }
    }
    let step_ns = t.elapsed().as_secs_f64() * 1e9 / steps as f64;
    println!(
        "net={path} fights={} states={} net_us/decision={:.3} playout_us/step={:.3} sink={sink:#018x} (acc {acc:.1})",
        fights.len(),
        states.len(),
        net_ns / 1000.0,
        step_ns / 1000.0,
    );
}

/// [`play`] without recording, for the playout timing.
fn play_steps(net: &RolloutNet, state: &CombatState, seed: u64) -> u32 {
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut s = state.clone();
    let mut steps = 0u32;
    while !is_terminal(&s) && steps < 600 {
        let acts = legal_actions(&s);
        if acts.is_empty() {
            break;
        }
        let i = pick_with(
            net,
            &mut rng,
            &s,
            &acts,
            END_TURN_PRIORITY,
            net.temperature,
            net.no_early_end_turn,
        );
        let _ = apply(&mut s, acts[i].clone());
        steps += 1;
    }
    steps
}
