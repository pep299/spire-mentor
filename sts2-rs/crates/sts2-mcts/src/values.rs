//! Recording-free playout values under the search's own playout (GH #830,
//! deep bets build 1).
//!
//! The search scores a leaf by one playout ([`crate::Mcts`]'s simulation
//! step: the private `rollout`, then a clamp to the backup bound). Value
//! labels and the oracle turn planner need that same number for arbitrary
//! states without running a search, and without the per-step recording
//! `rollout_net::playout_trajectories_with` does. [`search_playout`] is that
//! playout, and [`playout_value`] wraps it with the three starting views the
//! deep-bets plan uses:
//!
//! * [`PlayoutMode::Raw`] — the state as given, its own draw order and RNG
//!   cursors. On a search leaf's realized state with the simulation's seed
//!   this IS the search's playout 0 (pinned by a test).
//! * [`PlayoutMode::Belief`] — the honest view first:
//!   [`sts2_core::blindfold::belief_root`]`(seed)`, the conversion the
//!   deployed `BeliefDraw` search applies to its root.
//! * [`PlayoutMode::BeliefEndTurn`] — the honest view, then `EndTurn`, then
//!   the playout: the value of a turn's afterstate. Seed `j` re-seeds every
//!   RNG stream identically, so two plans scored with the same seed are
//!   dealt the same next-turn randomness (common random numbers).

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use sts2_core::{apply as core_apply, is_terminal, legal_actions, Action, CombatState};

use crate::{effective_rollout_depth, rollout, LeafEval, MctsConfig, MAX_LEAF_VALUE_ABS};

/// One playout exactly as the search runs it at a leaf: the search's
/// playout under `cfg` (`cfg.rollout_policy` for at most
/// `cfg.max_rollout_depth` actions, quiescence when `cfg.use_quiescence`,
/// then the terminal or depth-cut value), clamped to the backup bound. Plays
/// `state` forward in place to the playout's end; draws the policy's
/// randomness from `rng`.
///
/// `cfg.max_rollout_depth` is used as given. A search raises it once, at
/// construction, by [`effective_rollout_depth`]; [`playout_value`] does the
/// same for the state it scores.
pub fn search_playout(
    state: &mut CombatState,
    rng: &mut Xoshiro256PlusPlus,
    cfg: &MctsConfig,
    leaf_eval: &LeafEval,
) -> f32 {
    rollout(
        state,
        rng,
        cfg.max_rollout_depth,
        cfg.rollout_policy,
        leaf_eval,
        cfg.use_quiescence,
        cfg,
    )
    .clamp(-MAX_LEAF_VALUE_ABS, MAX_LEAF_VALUE_ABS)
}

/// Where a [`playout_value`] playout starts. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayoutMode {
    /// The state as given.
    Raw,
    /// `belief_root(seed)` first.
    Belief,
    /// `belief_root(seed)`, then `EndTurn`, then the playout.
    BeliefEndTurn,
}

impl PlayoutMode {
    /// `"raw"`, `"belief"` or `"belief_endturn"`; `None` for anything else.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "raw" => Some(Self::Raw),
            "belief" => Some(Self::Belief),
            "belief_endturn" => Some(Self::BeliefEndTurn),
            _ => None,
        }
    }
}

/// One scored playout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlayoutValue {
    /// What the search backs up for this playout ([`search_playout`]'s
    /// value; equal to [`crate::playout_return`] of the end state).
    pub value: f32,
    /// The playout ended in a won fight.
    pub won: bool,
    /// The player's HP at the playout's end (floored at 0).
    pub hp: i32,
    /// The playout ended the fight (`false`: it stopped at the depth cap or
    /// ran out of legal actions, and `value` is the depth-cut leaf value).
    pub terminal: bool,
}

/// Why a [`playout_value`] could not be computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayoutError {
    /// [`PlayoutMode::BeliefEndTurn`] on a live state where `EndTurn` is not
    /// a legal action (a card choice is open).
    EndTurnIllegal,
}

impl std::fmt::Display for PlayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EndTurnIllegal => write!(
                f,
                "belief_endturn: EndTurn is not legal in this state (a choice is open)"
            ),
        }
    }
}

impl std::error::Error for PlayoutError {}

/// Score one playout of `state` under the search config `cfg` and leaf
/// evaluator `leaf_eval`, started from `mode`'s view, with the policy RNG
/// seeded `seed` (`Xoshiro256PlusPlus::seed_from_u64(seed)`, the RNG a
/// search seeded `seed` gives its first simulation). The belief modes call
/// `belief_root(state, seed)`, so `Belief` at seed `s` equals `Raw` at seed
/// `s` on the state's belief view at `s`.
///
/// `cfg.max_rollout_depth` is raised by [`effective_rollout_depth`] for
/// `state`, as a search rooted at `state` would. `cfg.leaf_playouts` is not
/// read: this is one playout, and a caller wanting `k` passes `k` seeds.
/// A terminal state is scored as it stands (in `BeliefEndTurn` too).
///
/// # Errors
/// [`PlayoutError::EndTurnIllegal`] in `BeliefEndTurn` mode when the live
/// state cannot end its turn.
pub fn playout_value(
    state: &CombatState,
    seed: u64,
    mode: PlayoutMode,
    cfg: &MctsConfig,
    leaf_eval: &LeafEval,
) -> Result<PlayoutValue, PlayoutError> {
    let mut cfg = *cfg;
    cfg.max_rollout_depth = effective_rollout_depth(state, &cfg);
    let mut s = state.clone();
    if mode != PlayoutMode::Raw {
        sts2_core::blindfold::belief_root(&mut s, seed);
    }
    if mode == PlayoutMode::BeliefEndTurn && !is_terminal(&s) {
        if !legal_actions(&s).contains(&Action::EndTurn) {
            return Err(PlayoutError::EndTurnIllegal);
        }
        let _ = core_apply(&mut s, Action::EndTurn);
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let value = search_playout(&mut s, &mut rng, &cfg, leaf_eval);
    Ok(PlayoutValue {
        value,
        won: s.player_won,
        hp: s.player.current_hp.max(0),
        terminal: is_terminal(&s),
    })
}

/// [`playout_value`] for many `(state, seed)` jobs, in parallel on the rayon
/// pool (`RAYON_NUM_THREADS` caps it). One result per job, in job order;
/// each depends only on its own job, so the thread count never changes a
/// value.
#[must_use]
pub fn playout_values(
    jobs: &[(CombatState, u64)],
    mode: PlayoutMode,
    cfg: &MctsConfig,
    leaf_eval: &LeafEval,
) -> Vec<Result<PlayoutValue, PlayoutError>> {
    jobs.par_iter()
        .map(|(state, seed)| playout_value(state, *seed, mode, cfg, leaf_eval))
        .collect()
}

/// Length of [`rng_fingerprint`]: every combat RNG stream plus the legacy one.
pub const RNG_FINGERPRINT_LEN: usize = sts2_core::run_rng_set::COMBAT_STREAM_COUNT + 1;

/// The cursors the search's chance detector compares: every combat RNG
/// stream's counter ([`sts2_core::run_rng_set::RunRngSet::combat_counter_fingerprint`])
/// followed by the legacy `state.rng` counter. A transition is a chance
/// edge to the search exactly when it changes this value — the expansion
/// step in `Mcts::one_simulation` tests the same two terms. Two states that
/// differ only in how much randomness they have consumed differ here and
/// nowhere in [`sts2_core::search_hash`].
///
/// Exact for every hidden-information mode except `DrawChance`, whose
/// detector adds a term for a first draw from a heterogeneous unknown
/// segment (a draw that consumes no RNG in that mode). Under the deployed
/// `BeliefDraw` a draw from an unknown segment of two or more cards
/// consumes the shuffle stream and so shows up here.
///
/// **Precondition for reading it as the deployed search's chance detector:
/// [`fingerprint_sees_draws`]** — a belief view (`belief_root`, the Python
/// `belief_view`) or one of its descendants. The deployed search converts
/// its root with `belief_root`, so it draws from the hidden pile at random,
/// which consumes RNG. A raw state keeps a materialized draw order: drawing
/// from it reads the true next card and consumes nothing, so this
/// fingerprint stays put on an action the deployed search makes a chance
/// node, and a planner that compares it there is silently clairvoyant.
#[must_use]
pub fn rng_fingerprint(state: &CombatState) -> [u32; RNG_FINGERPRINT_LEN] {
    let streams = state.run_rng.combat_counter_fingerprint();
    let mut out = [0; RNG_FINGERPRINT_LEN];
    out[..streams.len()].copy_from_slice(&streams);
    out[streams.len()] = state.rng.counter();
    out
}

/// Whether comparing [`rng_fingerprint`] across an action from `state`
/// flags exactly the transitions the deployed (`BeliefDraw`) search makes
/// chance nodes: `true` when the draw order is deferred
/// ([`sts2_core::PlayerCombatState::deferred_draw_order`], set by
/// `belief_root`) or when the unknown middle of the draw pile holds fewer
/// than two cards (no hidden order to draw from). `false` on a raw state
/// with a hidden draw order, where a draw reads the materialized order
/// without touching any RNG stream.
#[must_use]
pub fn fingerprint_sees_draws(state: &CombatState) -> bool {
    state.player.deferred_draw_order
        || sts2_core::cards::unknown_middle_uids(&state.player).len() < 2
}
