//! `tie_rule_probe` — what the in-band root rules do to the deployed pilot's
//! own picks (GH #1022 phase 3).
//!
//! The rules of `sts2_mcts::pick_root` rule 5 only ever choose among moves
//! the search scored within `tie_eps` of each other, so a win-rate panel is
//! the wrong first instrument: it cannot see a change confined to noise-equal
//! moves (#444, #128). This probe measures a rule set where it acts:
//!
//! - **Positions.** Every fixture under the pool roots is realized and the
//!   deployed pilot ([`sts2_runai::policy::pilot_search`], the run agent's
//!   search at the eval-cheap 1000 / 4000 budget) plays the fight from its
//!   first turn. Every searched decision on that trajectory is a position,
//!   so the mix is the pilot's own, not a random warm-up's.
//! - **Arms over one tree.** Each position is searched under `--seeds`
//!   seeds. Each tree is then re-picked under every `--arms` rule set
//!   (`Mcts::explain_root_pick_with`). The search never reads the rules, so
//!   the arms differ by the rule alone: no search noise between them.
//! - **Forced-first-move playouts** (`--playouts N`). Where an arm's pick on
//!   the seed-0 tree differs from the pilot's, both moves are forced and the
//!   pilot finishes the fight `N` times from a fresh re-deal of the hidden
//!   draw pile (paired: playout `t` re-deals the same way for every move).
//!   The final HP and win difference is the rule's regret in real units.
//!
//! The seed-0 pilot pick advances the trajectory; the arms never steer it.
//!
//! Output: JSONL in `--out`, flushed per row so a stuck run shows. A `h`
//! header names the arms, then one `d` row per decision (per seed: each
//! arm's move key and rule, and the root children as `[key, visits, q,
//! se]`) and one `f` row per flipped decision with the playout outcomes.
//! `docs/experiments/2026-10-29-tie-rules/analyze.py` reads it.
//!
//! Move keys merge copies of a card: two Strikes at the same cost on the
//! same target are one move (`sts2_runai::tie_probe::move_key`), so a pick
//! that only swaps copies is neither a flip nor instability.
//!
//! ```text
//! cargo run --release -p sts2-bench --example tie_rule_probe -- \
//!     --out rows.jsonl [--chars ironclad,defect] [--per-char 150] \
//!     [--seeds 8] [--arms none,block,free,junk,block+free+junk] \
//!     [--playouts 32] [--playout-sims 1000] [--max-decisions 80] \
//!     <pool_root> [<pool_root> ...]
//! ```
//!
//! A pool root holds `the-<character>/*_{common,elite,boss}.toml` (the
//! `harvest.sh` layout). `--per-char` takes the first fixtures by name.

#[path = "common/mod.rs"]
mod common;

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;

use rayon::prelude::*;
use serde_json::{json, Value};
use sts2_core::{apply, is_terminal, legal_actions, Action, CombatState};
use sts2_mcts::{RootOutcome, TieRules};
use sts2_runai::policy::{pilot_search, BudgetTier, CombatBudget};
use sts2_runai::tie_probe::move_key;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// Salt for a playout's re-deal, so it is not the search seed.
const DETERMINIZE_SALT: u64 = 0xDEA1_0000_0000_0001;
/// Applied-action ceiling per fight or playout (forced moves included).
const MAX_STEPS: u32 = 600;

struct Opts {
    chars: Vec<String>,
    per_char: usize,
    seeds: u32,
    arms: Vec<(String, TieRules)>,
    playouts: u32,
    playout_sims: Option<u32>,
    max_decisions: u32,
    out: PathBuf,
    pools: Vec<PathBuf>,
}

struct Fixture {
    pool: String,
    character: String,
    tier: BudgetTier,
    path: PathBuf,
    name: String,
}

const fn tier_name(t: BudgetTier) -> &'static str {
    match t {
        BudgetTier::Common => "common",
        BudgetTier::Elite => "elite",
        BudgetTier::Boss => "boss",
    }
}

fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fixtures(opts: &Opts) -> Vec<Fixture> {
    let mut out = Vec::new();
    for root in &opts.pools {
        let pool = root
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        for character in &opts.chars {
            let dir = root.join(format!("the-{character}"));
            let Ok(entries) = std::fs::read_dir(&dir) else {
                eprintln!("[tie_rule_probe] no fixtures in {}", dir.display());
                continue;
            };
            let mut paths: Vec<PathBuf> = entries
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "toml"))
                .collect();
            paths.sort();
            for path in paths.into_iter().take(opts.per_char) {
                let name = path
                    .file_stem()
                    .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                let tier = if name.ends_with("_boss") {
                    BudgetTier::Boss
                } else if name.ends_with("_elite") {
                    BudgetTier::Elite
                } else {
                    BudgetTier::Common
                };
                out.push(Fixture {
                    pool: pool.clone(),
                    character: character.clone(),
                    tier,
                    path,
                    name,
                });
            }
        }
    }
    out
}

/// One seed's search at a position: the pilot's own pick, each arm's pick
/// over the same tree, and the root children.
struct Cell {
    seed: u64,
    played: Action,
    reference: Value,
    arms: Vec<Value>,
    arm_picks: Vec<(Action, String)>,
    children: Vec<RootOutcome>,
}

fn search_cell(
    state: &CombatState,
    tier: BudgetTier,
    sims: u32,
    seed: u64,
    arms: &[(String, TieRules)],
) -> Option<Cell> {
    let mut m = pilot_search(state, tier, sims, seed);
    m.try_search_n(sims).ok()?;
    let pick = m.explain_root_pick().ok()?;
    let mut arm_rows = Vec::with_capacity(arms.len());
    let mut arm_picks = Vec::with_capacity(arms.len());
    for (_, rules) in arms {
        let p = m.explain_root_pick_with(*rules).ok()?;
        let key = move_key(state, &p.action);
        arm_rows.push(json!({"key": key, "rule": p.rule.as_str()}));
        arm_picks.push((p.action, key));
    }
    Some(Cell {
        seed,
        reference: json!({
            "key": move_key(state, &pick.action),
            "rule": pick.rule.as_str(),
            "band": pick.band.len(),
            "gq": pick.gap_q,
            "gz": pick.gap_z,
        }),
        played: pick.action,
        arms: arm_rows,
        arm_picks,
        children: m.root_outcomes(),
    })
}

fn cell_json(state: &CombatState, c: &Cell) -> Value {
    let ch: Vec<Value> = c
        .children
        .iter()
        .map(|o| {
            json!([
                move_key(state, &o.action),
                o.visits,
                o.mean_value,
                o.mean_stderr()
            ])
        })
        .collect();
    json!({"seed": c.seed, "ref": c.reference, "arms": c.arms, "ch": ch})
}

/// Force `first`, then let the pilot finish the fight from a re-deal of the
/// hidden state. `(won, final_hp)`.
fn playout(
    start: &CombatState,
    first: &Action,
    tier: BudgetTier,
    sims: u32,
    seed: u64,
) -> (bool, i32) {
    let mut state = start.clone();
    sts2_core::blindfold::blindfold_root(&mut state, seed ^ DETERMINIZE_SALT);
    let _ = apply(&mut state, first.clone());
    let mut steps: u32 = 1;
    while !is_terminal(&state) && steps < MAX_STEPS {
        let legal = legal_actions(&state);
        let next = match legal.as_slice() {
            [] => Action::EndTurn,
            [only] => only.clone(),
            _ => {
                let step_seed = seed ^ u64::from(steps).wrapping_mul(GOLDEN);
                let mut m = pilot_search(&state, tier, sims, step_seed);
                match m.try_search_n(sims) {
                    Ok(a) => a,
                    Err(_) => Action::EndTurn,
                }
            }
        };
        let _ = apply(&mut state, next);
        steps += 1;
    }
    (
        is_terminal(&state) && state.player_won,
        state.player.current_hp.max(0),
    )
}

struct Writer {
    file: std::io::BufWriter<std::fs::File>,
}

impl Writer {
    fn row(&mut self, v: &Value) {
        let _ = writeln!(self.file, "{v}");
        let _ = self.file.flush();
    }
}

#[derive(Default)]
struct Tally {
    decisions: u64,
    cells: u64,
    banded: u64,
    /// Per arm: cells whose pick differs from the pilot's.
    flips: Vec<u64>,
    flipped_decisions: u64,
}

#[allow(clippy::too_many_lines)]
fn probe_fight(fx: &Fixture, opts: &Opts, out: &Mutex<Writer>) -> Tally {
    let mut tally = Tally {
        flips: vec![0; opts.arms.len()],
        ..Tally::default()
    };
    let fight_seed = fnv1a(&format!("{}/{}/{}", fx.pool, fx.character, fx.name));
    let mut state = match common::realize_fixture(&fx.path, fight_seed) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[tie_rule_probe] skip {}: {e}", fx.name);
            return tally;
        }
    };
    let budget = CombatBudget::eval_cheap();
    let sims = budget.sims_for_with(fx.tier, false);
    let playout_sims = opts.playout_sims.unwrap_or(sims);
    let mut steps: u32 = 0;
    let mut d: u32 = 0;
    while !is_terminal(&state) && steps < MAX_STEPS && d < opts.max_decisions {
        let legal = legal_actions(&state);
        if legal.len() < 2 {
            let only = legal.first().cloned().unwrap_or(Action::EndTurn);
            let _ = apply(&mut state, only);
            steps += 1;
            continue;
        }
        let cells: Vec<Cell> = (0..opts.seeds)
            .into_par_iter()
            .filter_map(|s| {
                let seed = splitmix(fight_seed ^ (u64::from(d) << 32) ^ u64::from(s));
                search_cell(&state, fx.tier, sims, seed, &opts.arms)
            })
            .collect();
        let Some(first) = cells.first() else {
            eprintln!("[tie_rule_probe] {}: no search at decision {d}", fx.name);
            break;
        };
        tally.decisions += 1;
        let ref_key = move_key(&state, &first.played);
        for c in &cells {
            tally.cells += 1;
            if c.reference["band"].as_u64().unwrap_or(0) > 1 {
                tally.banded += 1;
            }
            let k = move_key(&state, &c.played);
            for (i, (_, ak)) in c.arm_picks.iter().enumerate() {
                if *ak != k {
                    tally.flips[i] += 1;
                }
            }
        }
        let row = json!({
            "k": "d",
            "pool": fx.pool,
            "char": fx.character,
            "tier": tier_name(fx.tier),
            "fx": fx.name,
            "d": d,
            "legal": legal.len(),
            "hp": state.player.current_hp,
            "round": state.round,
            "s": cells.iter().map(|c| cell_json(&state, c)).collect::<Vec<_>>(),
        });
        out.lock().expect("writer").row(&row);

        // The seed-0 tree's distinct alternative picks, with the arms that
        // chose each.
        let mut alts: Vec<(Action, String, Vec<String>)> = Vec::new();
        if opts.playouts > 0 {
            for ((name, _), (action, key)) in opts.arms.iter().zip(&first.arm_picks) {
                if *key == ref_key {
                    continue;
                }
                match alts.iter_mut().find(|(_, k, _)| k == key) {
                    Some((_, _, names)) => names.push(name.clone()),
                    None => alts.push((action.clone(), key.clone(), vec![name.clone()])),
                }
            }
        }
        if !alts.is_empty() {
            tally.flipped_decisions += 1;
            let mut moves = vec![first.played.clone()];
            moves.extend(alts.iter().map(|(a, _, _)| a.clone()));
            let n = opts.playouts;
            let results: Vec<(usize, u32, bool, i32)> = (0..moves.len())
                .flat_map(|i| (0..n).map(move |t| (i, t)))
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(i, t)| {
                    let seed = splitmix(fight_seed ^ (u64::from(d) << 40) ^ u64::from(t));
                    let (won, hp) = playout(&state, &moves[i], fx.tier, playout_sims, seed);
                    (i, t, won, hp)
                })
                .collect();
            let mut wins = vec![vec![0u8; n as usize]; moves.len()];
            let mut hps = vec![vec![0i32; n as usize]; moves.len()];
            for (i, t, won, hp) in results {
                wins[i][t as usize] = u8::from(won);
                hps[i][t as usize] = hp;
            }
            let mut keys = vec![ref_key.clone()];
            keys.extend(alts.iter().map(|(_, k, _)| k.clone()));
            let row = json!({
                "k": "f",
                "pool": fx.pool,
                "char": fx.character,
                "tier": tier_name(fx.tier),
                "fx": fx.name,
                "d": d,
                "keys": keys,
                "arms": alts.iter().map(|(_, _, names)| names.clone()).collect::<Vec<_>>(),
                "sims": playout_sims,
                "w": wins,
                "hp": hps,
            });
            out.lock().expect("writer").row(&row);
        }

        let _ = apply(&mut state, first.played.clone());
        steps += 1;
        d += 1;
    }
    tally
}

fn parse_args() -> Result<Opts, String> {
    let mut opts = Opts {
        chars: vec!["ironclad".into(), "defect".into()],
        per_char: usize::MAX,
        seeds: 8,
        arms: Vec::new(),
        playouts: 0,
        playout_sims: None,
        max_decisions: 80,
        out: PathBuf::new(),
        pools: Vec::new(),
    };
    let mut arms = "none,block,free,junk,block+free+junk".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |what: &str| args.next().ok_or_else(|| format!("{what} needs a value"));
        match a.as_str() {
            "--out" => opts.out = PathBuf::from(val("--out")?),
            "--chars" => {
                opts.chars = val("--chars")?.split(',').map(str::to_owned).collect();
            }
            "--per-char" => {
                opts.per_char = val("--per-char")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--seeds" => opts.seeds = val("--seeds")?.parse().map_err(|e| format!("{e}"))?,
            "--arms" => arms = val("--arms")?,
            "--playouts" => {
                opts.playouts = val("--playouts")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--playout-sims" => {
                opts.playout_sims =
                    Some(val("--playout-sims")?.parse().map_err(|e| format!("{e}"))?);
            }
            "--max-decisions" => {
                opts.max_decisions = val("--max-decisions")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}")),
            pool => opts.pools.push(PathBuf::from(pool)),
        }
    }
    for name in arms.split(',').filter(|n| !n.is_empty()) {
        opts.arms.push((name.to_owned(), TieRules::parse(name)?));
    }
    if opts.out.as_os_str().is_empty() || opts.pools.is_empty() || opts.seeds == 0 {
        return Err("usage: tie_rule_probe --out <rows.jsonl> [flags] <pool_root>...".into());
    }
    Ok(opts)
}

fn main() {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let fxs = fixtures(&opts);
    eprintln!(
        "[tie_rule_probe] {} fixtures, {} seeds, arms {:?}, playouts {}",
        fxs.len(),
        opts.seeds,
        opts.arms
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>(),
        opts.playouts
    );
    let file = std::fs::File::create(&opts.out).unwrap_or_else(|e| {
        panic!("create {}: {e}", opts.out.display());
    });
    let out = Mutex::new(Writer {
        file: std::io::BufWriter::new(file),
    });
    out.lock().expect("writer").row(&json!({
        "k": "h",
        "arms": opts.arms.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        "seeds": opts.seeds,
        "playouts": opts.playouts,
        "playout_sims": opts.playout_sims,
        "max_decisions": opts.max_decisions,
        "pools": opts.pools.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "git": common::git_short_sha(),
        "fixtures": fxs.len(),
    }));
    let done = std::sync::atomic::AtomicUsize::new(0);
    let t0 = std::time::Instant::now();
    let tallies: Vec<Tally> = fxs
        .par_iter()
        .map(|fx| {
            let t = probe_fight(fx, &opts, &out);
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            eprintln!(
                "[tie_rule_probe] {n}/{} {} {} {}: {} decisions, {} flipped ({:.0}s)",
                fxs.len(),
                fx.character,
                tier_name(fx.tier),
                fx.name,
                t.decisions,
                t.flipped_decisions,
                t0.elapsed().as_secs_f64()
            );
            t
        })
        .collect();
    let mut total = Tally {
        flips: vec![0; opts.arms.len()],
        ..Tally::default()
    };
    for t in &tallies {
        total.decisions += t.decisions;
        total.cells += t.cells;
        total.banded += t.banded;
        total.flipped_decisions += t.flipped_decisions;
        for (a, b) in total.flips.iter_mut().zip(&t.flips) {
            *a += b;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let pct = |n: u64| 100.0 * n as f64 / total.cells.max(1) as f64;
    println!(
        "decisions {}  cells {}  band>1 {:.1}%",
        total.decisions,
        total.cells,
        pct(total.banded)
    );
    for ((name, _), flips) in opts.arms.iter().zip(&total.flips) {
        println!(
            "  arm {name:<20} differs from the pilot on {:.2}% of cells",
            pct(*flips)
        );
    }
}
