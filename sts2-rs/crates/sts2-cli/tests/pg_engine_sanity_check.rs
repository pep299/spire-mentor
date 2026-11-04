//! Bead `sts2-rs-kog8`: PG_ELITE engine-vs-human sanity check.
//!
//! Hand-coded aggressive defensive policy on the
//! `0029a7ba-..._floor009_phantasmal_gardeners_elite` fixture
//! (asc 3, 66/80 HP, won by the human in 4 turns per provenance).
//! Skips MCTS entirely — plays a fixed priority-based card order.
//!
//! Acceptance: dump turn-by-turn trace + 10-seed variance sweep
//! to stdout (use `cargo test --test pg_engine_sanity_check
//! -- --nocapture`). Verdict gets pasted into
//! `docs/perf/2026-05-18-pg-engine-sanity-check.md`.

use std::path::{Path, PathBuf};

use sts2_cli::scenario_to_combat_state;
use sts2_cli::sim::scenario::toml_file::load_scenario_file;
use sts2_core::cards::CardId;
use sts2_core::{apply, combat_loop, Action, CombatState, StepResult};

const FIXTURE: &str =
    "/tmp/run-fixtures-combined/ironclad/0029a7ba-88f9-4a40-9428-c17174bf62d0_floor009_phantasmal_gardeners_elite.toml";

const MAX_TURNS: u16 = 12;

struct Outcome {
    won: bool,
    lost: bool,
    turn: u16,
    hp: i32,
    max_hp: i32,
    alive: usize,
    opening_hand: Vec<String>,
}

#[test]
fn kog8_pg_engine_sanity_check_aggressive_defensive() {
    let path = PathBuf::from(FIXTURE);
    if !path.exists() {
        eprintln!("[kog8] fixture missing — skipping ({FIXTURE})");
        return;
    }

    // Single trace at seed=42 (used by the doc) — without pre-shuffle.
    let _ = run_trace(&path, 42, true, false);

    println!("\n=== Variance sweep (no pre-shuffle) — seeds 0..30 ===");
    let mut wins = 0u32;
    for seed in 0..30u64 {
        let o = run_trace(&path, seed, false, false);
        if o.won {
            wins += 1;
        }
        println!(
            "seed {seed:>2}: turn={:>2} hp={}/{} alive={}/4 won={} lost={} hand0={:?}",
            o.turn, o.hp, o.max_hp, o.alive, o.won, o.lost, o.opening_hand,
        );
    }
    println!("=== No pre-shuffle: {wins}/30 wins ===");

    println!("\n=== Variance sweep (WITH manual pre-shuffle) — seeds 0..30 ===");
    let mut wins_shuffled = 0u32;
    for seed in 0..30u64 {
        let o = run_trace(&path, seed, false, true);
        if o.won {
            wins_shuffled += 1;
        }
        println!(
            "seed {seed:>2}: turn={:>2} hp={}/{} alive={}/4 won={} lost={} hand0={:?}",
            o.turn, o.hp, o.max_hp, o.alive, o.won, o.lost, o.opening_hand,
        );
    }
    println!("=== WITH pre-shuffle: {wins_shuffled}/30 wins ===");
}

fn run_trace(path: &Path, seed: u64, verbose: bool, pre_shuffle: bool) -> Outcome {
    let scenario = load_scenario_file(path).expect("load scenario");
    let mut state = scenario_to_combat_state(&scenario, seed).expect("build state");
    if pre_shuffle {
        // Simulate the missing Python-pin combat-init shuffle by
        // shuffling state.player.draw before start_combat. Mirrors
        // sts2-rl-agent@01c094d:sts2_env/core/combat.py:615
        // (self.rng.shuffle(state.draw) inside _setup_combat_state).
        // Split borrow via mem::take, shuffle, restore.
        let mut draw = std::mem::take(&mut state.player.draw);
        state.rng.shuffle(draw.as_mut_slice());
        state.player.draw = draw;
    }
    let _ = combat_loop::start_combat(&mut state);
    let _ = combat_loop::start_player_turn(&mut state);

    let mut log = Vec::<String>::new();
    macro_rules! v {
        ($($tt:tt)*) => {
            if verbose { log.push(format!($($tt)*)); }
        };
    }

    v!(
        "=== kog8 PG sanity check — fixture {} (seed={seed}) ===",
        path.file_name().unwrap().to_string_lossy()
    );
    v!(
        "Initial: HP {}/{}  block {}  energy {}  enemies {}",
        state.player.current_hp,
        state.player.max_hp,
        state.player.block,
        state.player.energy,
        state.enemy_uids.len()
    );

    let opening_hand = describe_hand(&state);
    let mut won = false;
    let mut lost = false;
    let mut final_turn = 0u16;

    'turns: for turn in 1..=MAX_TURNS {
        final_turn = turn;
        v!("");
        v!(
            "── Turn {turn} start ── HP {}/{}  block {}  energy {}/{}  hand {}  draw {}  disc {}",
            state.player.current_hp,
            state.player.max_hp,
            state.player.block,
            state.player.energy,
            state.player.max_energy,
            state.player.hand.len(),
            state.player.draw.len(),
            state.player.discard.len(),
        );
        if verbose {
            for (i, uid) in state.enemy_uids.iter().enumerate() {
                let c = state
                    .creatures
                    .iter()
                    .find(|c| c.id == uid.0)
                    .expect("enemy creature");
                log.push(format!(
                    "    enemy[{i}] hp={}/{} block={} intent={:?}",
                    c.current_hp, c.max_hp, c.block, c.intent
                ));
            }
            log.push(format!("    hand: {}", describe_hand(&state).join(", ")));
        }

        let mut steps = 0;
        loop {
            steps += 1;
            if steps > 50 {
                v!("    !! step-cap inside turn — aborting (would loop)");
                break 'turns;
            }

            let action = pick_action(&state, turn).unwrap_or(Action::EndTurn);
            v!("    play -> {}", describe_action(&state, &action));

            let result = apply(&mut state, action.clone());
            match result {
                StepResult::Advanced => {}
                StepResult::PendingChoice => {
                    v!("    pending choice -> auto-resolve [0]");
                    let r2 = apply(
                        &mut state,
                        Action::ResolveChoice {
                            selection: smallvec::smallvec![0u8],
                        },
                    );
                    match r2 {
                        StepResult::CombatWon => {
                            won = true;
                            break 'turns;
                        }
                        StepResult::CombatLost => {
                            lost = true;
                            break 'turns;
                        }
                        _ => {}
                    }
                }
                StepResult::CombatWon => {
                    won = true;
                    v!("    ** combat WON on turn {turn} step {steps} **");
                    break 'turns;
                }
                StepResult::CombatLost => {
                    lost = true;
                    v!("    ** combat LOST on turn {turn} step {steps} **");
                    break 'turns;
                }
                StepResult::InvalidTarget | StepResult::IllegalPlay => {
                    v!("    !! {result:?}; force EndTurn");
                    match apply(&mut state, Action::EndTurn) {
                        StepResult::CombatWon => {
                            won = true;
                            break 'turns;
                        }
                        StepResult::CombatLost => {
                            lost = true;
                            break 'turns;
                        }
                        _ => {}
                    }
                    break;
                }
            }

            if matches!(action, Action::EndTurn) {
                break;
            }
        }
    }

    let alive = state
        .enemy_uids
        .iter()
        .filter(|uid| {
            state
                .creatures
                .iter()
                .any(|c| c.id == uid.0 && c.current_hp > 0)
        })
        .count();

    if verbose {
        let verdict = if won {
            "engine VICTORY"
        } else if lost {
            "engine TPK"
        } else {
            "indeterminate (turn cap)"
        };
        log.push(String::new());
        log.push(format!("=== Verdict: {verdict} ==="));
        log.push(format!(
            "Final HP {}/{}  enemies alive {} / {}  round {}",
            state.player.current_hp,
            state.player.max_hp,
            alive,
            state.enemy_uids.len(),
            state.round,
        ));
        println!("{}", log.join("\n"));
    }

    Outcome {
        won,
        lost,
        turn: final_turn,
        hp: state.player.current_hp,
        max_hp: state.player.max_hp,
        alive,
        opening_hand,
    }
}

/// Priority-based picker for the bead-specified policy.
///
/// Priority:
///   1. Bloodletting+ if energy=0 AND hp≥25 (gets +3 energy)
///   2. Bash if no enemy has Vulnerable AND energy≥2
///   3. Defend (any) if energy≥1
///   4. FlameBarrier+ if energy≥2
///   5. Damage cards on Vulnerable-or-weakest if energy≥1
///   6. None → EndTurn
fn pick_action(state: &CombatState, _turn: u16) -> Option<Action> {
    let energy = state.player.energy;
    let weakest = weakest_enemy(state);
    let vuln_or_weakest = vulnerable_enemy(state).or(weakest);

    if energy == 0 && state.player.current_hp >= 25 {
        if let Some((idx, _)) = find_in_hand(state, &[CardId::Bloodletting]) {
            return Some(play(idx, None));
        }
    }

    if energy == 0 {
        return None;
    }

    let any_vuln = vulnerable_enemy(state).is_some();
    if !any_vuln && energy >= 2 {
        if let (Some((idx, _)), Some(t)) = (find_in_hand(state, &[CardId::Bash]), weakest) {
            return Some(play(idx, Some(t)));
        }
    }

    if energy >= 1 {
        if let Some((idx, _)) = find_in_hand(state, &[CardId::DefendIronclad]) {
            return Some(play(idx, None));
        }
    }

    if energy >= 2 {
        if let Some((idx, _)) = find_in_hand(state, &[CardId::FlameBarrierCard]) {
            return Some(play(idx, None));
        }
    }

    if energy >= 1 {
        let attack_priority = [
            CardId::PommelStrike,
            CardId::Pillage,
            CardId::ViciousCard,
            CardId::StrikeIronclad,
        ];
        if let (Some((idx, _)), Some(t)) = (find_in_hand(state, &attack_priority), vuln_or_weakest)
        {
            return Some(play(idx, Some(t)));
        }
    }

    None
}

fn play(hand_index: usize, target: Option<u8>) -> Action {
    Action::PlayCard {
        hand_index: u8::try_from(hand_index).expect("hand idx fits u8"),
        target,
    }
}

fn find_in_hand(state: &CombatState, wanted: &[CardId]) -> Option<(usize, CardId)> {
    for want in wanted {
        for (i, uid) in state.player.hand.iter().enumerate() {
            let idx = uid.0.get() as usize - 1;
            let cid = state.cards[idx].card_id;
            if cid == *want {
                return Some((i, cid));
            }
        }
    }
    None
}

fn weakest_enemy(state: &CombatState) -> Option<u8> {
    state
        .enemy_uids
        .iter()
        .enumerate()
        .filter_map(|(i, uid)| {
            state
                .creatures
                .iter()
                .find(|c| c.id == uid.0 && c.current_hp > 0)
                .map(|c| (i, c.current_hp))
        })
        .min_by_key(|(_, hp)| *hp)
        .map(|(i, _)| u8::try_from(i).unwrap_or(0))
}

fn vulnerable_enemy(state: &CombatState) -> Option<u8> {
    use sts2_core::PowerKind;
    state.enemy_uids.iter().enumerate().find_map(|(i, uid)| {
        let alive = state
            .creatures
            .iter()
            .any(|c| c.id == uid.0 && c.current_hp > 0);
        if !alive {
            return None;
        }
        let has_vuln = state
            .powers
            .iter()
            .any(|p| p.id == uid.0 && p.kind == PowerKind::Vulnerable && p.amount > 0);
        has_vuln.then(|| u8::try_from(i).unwrap_or(0))
    })
}

fn describe_hand(state: &CombatState) -> Vec<String> {
    state
        .player
        .hand
        .iter()
        .map(|uid| {
            let idx = uid.0.get() as usize - 1;
            format!("{:?}", state.cards[idx].card_id)
        })
        .collect()
}

fn describe_action(state: &CombatState, action: &Action) -> String {
    match action {
        Action::PlayCard { hand_index, target } => {
            let idx = *hand_index as usize;
            let cid = state.player.hand.get(idx).map_or_else(
                || "<?>".into(),
                |uid| {
                    let i = uid.0.get() as usize - 1;
                    format!("{:?}", state.cards[i].card_id)
                },
            );
            match target {
                Some(t) => format!("PlayCard {cid} -> enemy[{t}]"),
                None => format!("PlayCard {cid}"),
            }
        }
        Action::EndTurn => "EndTurn".into(),
        other => format!("{other:?}"),
    }
}
