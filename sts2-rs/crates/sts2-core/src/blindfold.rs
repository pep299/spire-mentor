//! Root blindfold for honest-search combat MCTS.
//!
//! Bead `sts2-rs-vqckz` (F0, plan §3.1/§3.4 of
//! `docs/plans/2026-07-12-clairvoyance-removal-plan.md`). The deployed
//! chance-node MCTS reads two pieces of hidden information a player
//! cannot observe: the true materialized draw-pile order (draws consume
//! no RNG, so they never trip the stochastic detector) and the true RNG
//! stream cursors (chance outcome 0 realizes from them). [`blindfold_root`]
//! replaces both with a fresh sample that depends **only** on a caller
//! supplied seed — never on the incoming hidden state — so a search that
//! blindfolds its root before searching is, by construction, invariant to
//! information the player never had.
//!
//! ## Canonicalization: why sort-then-shuffle
//!
//! A plain seeded shuffle of the *incoming* pile order is not enough. Two
//! permutations `P1` and `P2` of the same multiset are different inputs to
//! a shuffle — `shuffle(P1, seed)` and `shuffle(P2, seed)` land on
//! different output permutations even though `P1` and `P2` carry the same
//! observable information (a player only tracks which cards are where,
//! not the "hidden" incoming order the search happens to start from — that
//! order is itself a leak). To make the result a pure function of `seed`
//! (and the card multiset / pins, which ARE observable), the unknown
//! portion is first sorted into a canonical order — any deterministic
//! total order on [`crate::CardUid`] works, since the UIDs are unique so
//! the sorted sequence is the same regardless of which permutation of the
//! multiset was the input — and *then* shuffled with an RNG seeded purely
//! from the caller's seed. Two states differing only in unknown-pile order
//! collapse to byte-identical piles after this pipeline.
//!
//! Pinned positions ([`crate::PlayerCombatState::pinned_top`] /
//! `pinned_bottom`) are player-KNOWN information (Headbutt etc.) and are
//! preserved exactly like [`crate::cards::shuffle`] preserves them — the
//! canonicalization sort only touches the same unpinned middle segment
//! that the shuffle permutes.
//!
//! ## What is NOT touched
//!
//! `discard` / `exhaust` / `play` are face-up, publicly observable piles
//! — scrambling them would erase real information the player has and
//! isn't part of the leak. Only the unknown draw-pile segment and the RNG
//! streams are hidden information.

use crate::cards;
use crate::dotnet_rng::Rng;
use crate::run_rng_set::{outcome_stream_seed_u64, splitmix64, LEGACY_RNG_SALT};
use crate::{CardUid, CombatState, PlayerCombatState};

/// Salt distinguishing the blindfold's draw-pile-shuffle seed from every
/// other splitmix64-derived seed in the engine (the [`crate::run_rng_set`]
/// combat-stream salts, `LEGACY_RNG_SALT`, the run-draw salts, and the
/// `clairvoyance_probe` bench's own `PILE_SALT`). Any distinct odd-ish
/// constant works — the splitmix64 finalizer decorrelates them.
const BLINDFOLD_PILE_SALT: u64 = 0xF00D_BEEF_CAFE_0001;

/// Salt distinguishing the blindfold's RNG-stream reseed base from
/// [`BLINDFOLD_PILE_SALT`] and every other salt in the engine.
const BLINDFOLD_STREAM_SALT: u64 = 0xF00D_BEEF_CAFE_0002;

/// Salt distinguishing the per-`(base, n)` chance-outcome pile reshuffle
/// (bead `sts2-rs-i8bgu`, F1 `HiddenInfoMode::DrawChance`, plan §3.2 step
/// 2) from [`BLINDFOLD_PILE_SALT`], [`BLINDFOLD_STREAM_SALT`], every
/// per-stream combat-RNG salt in `crate::run_rng_set`,
/// [`crate::run_rng_set::LEGACY_RNG_SALT`], and the `clairvoyance_probe`
/// bench's own salts. [`shuffle_unknown_pile_for_outcome`] is
/// [`blindfold_root`]'s per-chance-outcome sibling — same
/// canonicalize-then-shuffle pipeline, mixed with `(base, n)` instead of
/// the root's caller-supplied seed, so it draws an independent sample
/// from every other splitmix64-derived stream in the engine.
const CHANCE_OUTCOME_PILE_SALT: u64 = 0xF00D_BEEF_CAFE_0003;

/// Salt distinguishing [`belief_root`]'s RNG-stream reseed base from
/// [`BLINDFOLD_PILE_SALT`], [`BLINDFOLD_STREAM_SALT`],
/// [`CHANCE_OUTCOME_PILE_SALT`], and every other salt in the engine. Bead
/// `sts2-rs-09x5n` (F2 `HiddenInfoMode::BeliefDraw`, lightspeed `CardPile`
/// port).
const BELIEF_ROOT_STREAM_SALT: u64 = 0xF00D_BEEF_CAFE_0004;

/// Replace `state`'s hidden information (unknown draw-pile order + RNG
/// stream cursors) with a fresh sample derived **purely** from `seed`.
///
/// Two calls with the same `seed` on states that differ only in unknown
/// draw-pile order and/or RNG cursor position produce byte-identical
/// results — see the module docs for why the canonicalizing sort is
/// required for that guarantee.
///
/// Intended use: call once on a private clone of the root state before
/// building the search tree ([`HiddenInfoMode::RootScramble`] in
/// `sts2-mcts`), deriving `seed` from the search's own `config.seed` (not
/// from anything in `state`) via the crate's usual splitmix64 pattern.
///
/// [`HiddenInfoMode::RootScramble`]: https://docs.rs/sts2-mcts (see `sts2_mcts::HiddenInfoMode`)
pub fn blindfold_root(state: &mut CombatState, seed: u64) {
    state.for_each_enemy_side_snapshot(|s| blindfold_root(s, seed));
    // 1. Pile: canonicalize the unknown middle segment, then shuffle it
    //    with an RNG seeded purely from `seed`. Pins are untouched by the
    //    canonicalization step and re-preserved by `cards::shuffle`.
    canonicalize_unknown_middle(&mut state.player);
    let mut pile_rng = Rng::from_search_seed(splitmix64(seed ^ BLINDFOLD_PILE_SALT));
    cards::shuffle(&mut state.player, &mut pile_rng);
    // Co-op (GH #1046): a teammate's draw order is as hidden as ours. Their
    // seats shuffle after ours from the same stream, so a solo pile is
    // byte-identical to what it was.
    for ally in &mut state.allies {
        canonicalize_unknown_middle(ally);
        cards::shuffle(ally, &mut pile_rng);
    }

    // 2. RNG streams: reseed every combat stream + the legacy stream from
    //    a base derived purely from `seed`, abandoning the incoming
    //    cursors entirely (mirrors the reseed pattern used by chance-node
    //    outcome realization in `sts2-mcts`).
    reseed_hidden_streams(state, seed, BLINDFOLD_STREAM_SALT);
}

/// Reseed every combat-relevant RNG stream in `state` (every named
/// [`crate::run_rng_set::RunRngSet`] stream plus the legacy `state.rng`)
/// from a base derived **purely** from `seed` and `salt`, abandoning the
/// incoming cursors entirely. Shared by [`blindfold_root`] (F0, salted with
/// [`BLINDFOLD_STREAM_SALT`]) and [`belief_root`] (F2, bead `sts2-rs-09x5n`,
/// salted with [`BELIEF_ROOT_STREAM_SALT`]) — pure extraction, not a
/// behavior change for `blindfold_root`.
fn reseed_hidden_streams(state: &mut CombatState, seed: u64, salt: u64) {
    let base = splitmix64(seed ^ salt);
    state.run_rng.reseed_combat_streams(base, 0);
    state.rng = Rng::from_search_seed(outcome_stream_seed_u64(base, 0, LEGACY_RNG_SALT));
}

/// F2 (bead `sts2-rs-09x5n`, plan §3.2's lightspeed-fork port,
/// `HiddenInfoMode::BeliefDraw`) root conversion: canonicalize `state`'s
/// unknown draw-pile segment (SORT, no shuffle — the random draw ORDER is
/// deferred to draw time from here on, never pre-committed) and reseed the
/// RNG streams from a base derived **purely** from `seed`, then set
/// [`crate::PlayerCombatState::deferred_draw_order`].
///
/// Differs from [`blindfold_root`] in exactly one respect: the pile step
/// stops after the canonicalizing sort instead of also Fisher-Yates
/// shuffling the result — [`sts2_core::cards::draw`] (via
/// [`crate::cards::draw`]/`draw_signaling`) defers the actual random
/// CHOICE of which card comes next to draw time, consuming RNG then, so
/// there is no permutation left to materialize here. The RNG-stream reseed
/// is otherwise identical (shares [`reseed_hidden_streams`], salted
/// distinctly via [`BELIEF_ROOT_STREAM_SALT`]) — same rationale as
/// `blindfold_root`'s step 2: abandon the incoming (unobservable) cursor
/// position so the search's own randomness is a pure function of `seed`.
///
/// Two calls with the same `seed` on states that differ only in unknown
/// draw-pile order and/or RNG cursor position produce byte-identical
/// results, by the same canonicalize-before-touching-anything argument as
/// [`blindfold_root`] (see the module docs) — this is what makes the
/// search's root action invariant to hidden information by construction
/// (G1).
///
/// Intended use: called once on a private clone of the root state before
/// building the search tree (`HiddenInfoMode::BeliefDraw` in
/// `sts2-mcts::Mcts::with_policy`), deriving `seed` from the search's own
/// `config.seed` (never from anything in `state`), exactly like
/// `blindfold_root`'s call site.
pub fn belief_root(state: &mut CombatState, seed: u64) {
    state.for_each_enemy_side_snapshot(|s| belief_root(s, seed));
    canonicalize_unknown_middle(&mut state.player);
    state.player.deferred_draw_order = true;
    // Co-op (GH #1046): each teammate's draws are deferred the same way; the
    // flag travels with their seat through `coop::swap_seat`.
    for ally in &mut state.allies {
        canonicalize_unknown_middle(ally);
        ally.deferred_draw_order = true;
    }
    reseed_hidden_streams(state, seed, BELIEF_ROOT_STREAM_SALT);
}

/// Reshuffle `player`'s unknown draw-pile segment for chance-outcome `n`
/// of a chance node with base `base` (bead `sts2-rs-i8bgu`, F1
/// `HiddenInfoMode::DrawChance`, plan §3.2 step 2). The per-chance-outcome
/// counterpart of [`blindfold_root`]'s pile step: same canonicalize-then-
/// shuffle pipeline (so the result is a pure function of the segment's
/// observable multiset + `(base, n)`, never of the incoming permutation —
/// see the module docs for why the canonicalizing sort is required), but
/// reused once per realized chance outcome instead of once at the root,
/// seeded through [`CHANCE_OUTCOME_PILE_SALT`] so it draws an independent
/// sample from every other stream the engine derives from `(base, n)`
/// (the combat RNG streams via
/// [`crate::run_rng_set::RunRngSet::reseed_combat_streams`], the legacy
/// stream reseed).
///
/// Intended use: called on a private clone of a chance node's PARENT
/// (pre-action) state, before re-applying the stochastic edge, mirroring
/// [`crate::run_rng_set::RunRngSet::reseed_combat_streams`]'s call site in
/// `sts2-mcts`'s chance-outcome realization.
pub fn shuffle_unknown_pile_for_outcome(player: &mut PlayerCombatState, base: u64, n: u32) {
    canonicalize_unknown_middle(player);
    let seed = outcome_stream_seed_u64(base, n, CHANCE_OUTCOME_PILE_SALT);
    let mut pile_rng = Rng::from_search_seed(seed);
    cards::shuffle(player, &mut pile_rng);
}

/// Sort the unpinned middle segment of `player.draw` into a canonical
/// (UID-ascending) order, leaving `pinned_top` / `pinned_bottom` cards at
/// their declared positions untouched. Mirrors the pin-preserving
/// reconstruction in [`crate::cards::shuffle`] exactly, substituting a
/// sort for the RNG shuffle — the two functions must keep agreeing on
/// which segment is "the unknown middle" or the canonicalization argument
/// in the module docs breaks.
pub(crate) fn canonicalize_unknown_middle(player: &mut PlayerCombatState) {
    if player.draw.len() < 2 {
        return;
    }
    if player.pinned_top.is_empty() && player.pinned_bottom.is_empty() {
        player.draw.sort_unstable();
        return;
    }
    // Same stale-pin sweep as `cards::shuffle` — keep the two in lockstep.
    player.pinned_top.retain(|uid| player.draw.contains(uid));
    player.pinned_bottom.retain(|uid| player.draw.contains(uid));
    let top: smallvec::SmallVec<[CardUid; 4]> = player.pinned_top.clone();
    let bot: smallvec::SmallVec<[CardUid; 4]> = player.pinned_bottom.clone();
    let mut middle: smallvec::SmallVec<[CardUid; 32]> = smallvec::SmallVec::new();
    for &uid in &player.draw {
        if !top.contains(&uid) && !bot.contains(&uid) {
            middle.push(uid);
        }
    }
    middle.sort_unstable();
    player.draw.clear();
    player.draw.extend(top.iter().copied());
    player.draw.extend(middle.iter().copied());
    player.draw.extend(bot.iter().copied());
}

/// Is `player.draw`'s unknown middle already in canonical order?
///
/// The read-only twin of [`canonicalize_unknown_middle`], and the only
/// honest way to check the precondition `deferred_draw_order = true`
/// asserts: that the pile is a *belief* (canonically ordered, so every
/// order-dissolving mutation can re-sort instead of permuting) rather than
/// a materialized shuffle.
///
/// Exists because the flag became settable from outside the engine when
/// `sts2-py`'s wire schema started round-tripping it. Inside the engine the
/// flag is only ever set immediately after canonicalizing, so the invariant
/// held by construction; a wire producer can assert it without establishing
/// it, and the failure is silent — [`crate::cards::pop_next_card`] simply
/// consumes RNG differently, so a search runs against a pile whose order
/// means something other than what the flag claims.
///
/// Note for wire callers: this discriminates on *uid* order, which is an
/// engine-internal card-creation order. A state ingested from `sts2-py`'s
/// wire has had its uids minted positionally from the listed pile, so the
/// unpinned case always looks canonical there and only pin/segment
/// incoherence is detectable. Full strength is only available on states the
/// engine itself built.
///
/// Mirrors the segment logic of [`canonicalize_unknown_middle`] exactly; the
/// two must be changed together, which is why they sit adjacent. Note it
/// tolerates stale pins (uids no longer in `draw`) the same way, since the
/// canonicalizer sweeps them rather than rejecting them.
#[must_use]
pub fn unknown_middle_is_canonical(player: &PlayerCombatState) -> bool {
    if player.draw.len() < 2 {
        return true;
    }
    let top: smallvec::SmallVec<[CardUid; 4]> = player
        .pinned_top
        .iter()
        .copied()
        .filter(|uid| player.draw.contains(uid))
        .collect();
    let bot: smallvec::SmallVec<[CardUid; 4]> = player
        .pinned_bottom
        .iter()
        .copied()
        .filter(|uid| player.draw.contains(uid))
        .collect();
    if top.is_empty() && bot.is_empty() {
        return player.draw.windows(2).all(|w| w[0] <= w[1]);
    }
    // Pinned cards must occupy the declared head/tail positions, and the
    // middle must be sorted.
    if player.draw.len() < top.len() + bot.len() {
        return false;
    }
    let split = player.draw.len() - bot.len();
    if player.draw[..top.len()] != top[..] || player.draw[split..] != bot[..] {
        return false;
    }
    player.draw[top.len()..split]
        .windows(2)
        .all(|w| w[0] <= w[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    fn uid(n: u32) -> CardUid {
        CardUid::new(n).expect("non-zero")
    }

    fn player_with_draw(order: &[u32]) -> PlayerCombatState {
        PlayerCombatState {
            draw: order.iter().copied().map(uid).collect::<SmallVec<_>>(),
            ..PlayerCombatState::default()
        }
    }

    #[test]
    fn canonicalize_sorts_regardless_of_input_permutation() {
        let mut a = player_with_draw(&[3, 1, 4, 1_000, 2]);
        let mut b = player_with_draw(&[4, 2, 1_000, 1, 3]);
        canonicalize_unknown_middle(&mut a);
        canonicalize_unknown_middle(&mut b);
        assert_eq!(a.draw, b.draw);
    }

    /// The predicate must accept exactly what the canonicalizer produces —
    /// that equivalence is the whole reason it can be used as a wire gate.
    #[test]
    fn canonical_predicate_agrees_with_the_canonicalizer() {
        for order in [
            &[3u32, 1, 4, 1_000, 2][..],
            &[1, 2, 3][..],
            &[9, 9, 1][..],
            &[7][..],
            &[][..],
        ] {
            let mut p = player_with_draw(order);
            canonicalize_unknown_middle(&mut p);
            assert!(
                unknown_middle_is_canonical(&p),
                "canonicalizer output rejected for input {order:?}"
            );
        }
    }

    #[test]
    fn canonical_predicate_rejects_a_materialized_shuffle() {
        assert!(!unknown_middle_is_canonical(&player_with_draw(&[3, 1, 2])));
        assert!(unknown_middle_is_canonical(&player_with_draw(&[1, 2, 3])));
    }

    #[test]
    fn canonical_predicate_honours_pins() {
        let mut p = player_with_draw(&[10, 1, 2, 3, 20]);
        p.pinned_top = [uid(10)].into_iter().collect();
        p.pinned_bottom = [uid(20)].into_iter().collect();
        assert!(
            unknown_middle_is_canonical(&p),
            "pinned head/tail with a sorted middle is canonical"
        );

        // Middle out of order.
        let mut q = player_with_draw(&[10, 3, 1, 2, 20]);
        q.pinned_top = [uid(10)].into_iter().collect();
        q.pinned_bottom = [uid(20)].into_iter().collect();
        assert!(!unknown_middle_is_canonical(&q));

        // Pin not at its declared position.
        let mut r = player_with_draw(&[1, 10, 2, 3, 20]);
        r.pinned_top = [uid(10)].into_iter().collect();
        r.pinned_bottom = [uid(20)].into_iter().collect();
        assert!(!unknown_middle_is_canonical(&r));
    }

    /// A stale pin (uid no longer in `draw`) is swept by the canonicalizer,
    /// so the predicate must ignore it too rather than failing the state.
    #[test]
    fn canonical_predicate_ignores_stale_pins() {
        let mut p = player_with_draw(&[1, 2, 3]);
        p.pinned_top = [uid(999)].into_iter().collect();
        assert!(unknown_middle_is_canonical(&p));
    }

    #[test]
    fn canonicalize_preserves_pins() {
        let mut p = player_with_draw(&[5, 3, 1, 4, 2]);
        p.pinned_top = SmallVec::from_slice(&[uid(5)]);
        p.pinned_bottom = SmallVec::from_slice(&[uid(2)]);
        canonicalize_unknown_middle(&mut p);
        assert_eq!(p.draw.first().copied(), Some(uid(5)));
        assert_eq!(p.draw.last().copied(), Some(uid(2)));
        // Middle is a sorted permutation of {1,3,4}.
        assert_eq!(&p.draw[1..4], &[uid(1), uid(3), uid(4)]);
    }

    #[test]
    fn blindfold_root_is_pure_function_of_seed() {
        let mut a = CombatState::default();
        a.player.draw = [3u32, 1, 4, 1_000, 2]
            .into_iter()
            .map(uid)
            .collect::<SmallVec<_>>();
        let mut b = a.clone();
        b.player.draw = [4u32, 2, 1_000, 1, 3]
            .into_iter()
            .map(uid)
            .collect::<SmallVec<_>>();
        // Advance b's cursors so the two states also differ there.
        let _ = b.rng.next_int(0, 1000);
        let _ = b.run_rng.shuffle.next_int(0, 1000);

        blindfold_root(&mut a, 0xABC);
        blindfold_root(&mut b, 0xABC);
        assert_eq!(a.player.draw, b.player.draw);
        assert_eq!(a.run_rng.combat_counter_fingerprint(), [0; 9]);
        assert_eq!(b.run_rng.combat_counter_fingerprint(), [0; 9]);
    }

    // ── shuffle_unknown_pile_for_outcome (bead sts2-rs-i8bgu, F1) ──────

    #[test]
    fn chance_outcome_shuffle_is_pure_function_of_base_and_n() {
        let mut a = player_with_draw(&[3, 1, 4, 1_000, 2]);
        let mut b = player_with_draw(&[4, 2, 1_000, 1, 3]);
        shuffle_unknown_pile_for_outcome(&mut a, 0xF00D, 3);
        shuffle_unknown_pile_for_outcome(&mut b, 0xF00D, 3);
        assert_eq!(
            a.draw, b.draw,
            "two incoming permutations of the same multiset must land on \
             the identical shuffled order for the same (base, n)"
        );
    }

    #[test]
    fn chance_outcome_shuffle_varies_by_n() {
        let mut a = player_with_draw(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let mut b = a.clone();
        shuffle_unknown_pile_for_outcome(&mut a, 0xF00D, 0);
        shuffle_unknown_pile_for_outcome(&mut b, 0xF00D, 1);
        assert_ne!(
            a.draw, b.draw,
            "distinct outcome ordinals must (almost surely) draw distinct \
             permutations"
        );
    }

    #[test]
    fn chance_outcome_shuffle_varies_by_base() {
        let mut a = player_with_draw(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let mut b = a.clone();
        shuffle_unknown_pile_for_outcome(&mut a, 0xF00D, 0);
        shuffle_unknown_pile_for_outcome(&mut b, 0xBEEF, 0);
        assert_ne!(
            a.draw, b.draw,
            "distinct chance-node bases must (almost surely) draw distinct \
             permutations"
        );
    }

    #[test]
    fn chance_outcome_shuffle_disagrees_with_blindfold_root_and_reseed() {
        // The three "reseed a stream for outcome n" call sites in this
        // module/crate must be mutually distinct salts — otherwise two of
        // them would draw perfectly correlated (or identical) sequences,
        // silently defeating the "independent sample" contract.
        let base = 0x1234_5678_9ABC_DEF0;
        let root_pile_seed = outcome_stream_seed_u64(base, 0, BLINDFOLD_PILE_SALT);
        let chance_pile_seed = outcome_stream_seed_u64(base, 0, CHANCE_OUTCOME_PILE_SALT);
        let stream_seed = outcome_stream_seed_u64(base, 0, BLINDFOLD_STREAM_SALT);
        assert_ne!(root_pile_seed, chance_pile_seed);
        assert_ne!(root_pile_seed, stream_seed);
        assert_ne!(chance_pile_seed, stream_seed);
    }

    #[test]
    fn chance_outcome_shuffle_preserves_pins() {
        let mut p = player_with_draw(&[5, 3, 1, 4, 2]);
        cards::pin_top(&mut p, uid(5));
        cards::pin_bottom(&mut p, uid(2));
        let mut expected_middle: Vec<CardUid> = vec![uid(1), uid(3), uid(4)];
        expected_middle.sort_unstable();

        shuffle_unknown_pile_for_outcome(&mut p, 0xABCD, 7);

        assert_eq!(p.pinned_top.as_slice(), &[uid(5)]);
        assert_eq!(p.pinned_bottom.as_slice(), &[uid(2)]);
        assert_eq!(p.draw.first().copied(), Some(uid(5)));
        assert_eq!(p.draw.last().copied(), Some(uid(2)));
        let mut middle: Vec<CardUid> = p.draw[1..4].to_vec();
        middle.sort_unstable();
        assert_eq!(middle, expected_middle);
    }

    /// Two co-op boards whose teammate draw piles hold the same cards in
    /// different orders (GH #1046), ours identical.
    fn coop_pair() -> (CombatState, CombatState) {
        let mut a = CombatState {
            player: player_with_draw(&[3, 1, 2]),
            ..CombatState::default()
        };
        a.allies.push(player_with_draw(&[14, 11, 13, 12, 15]));
        let mut b = a.clone();
        b.allies[0].draw = [12u32, 15, 11, 14, 13]
            .into_iter()
            .map(uid)
            .collect::<SmallVec<_>>();
        (a, b)
    }

    #[test]
    fn blindfold_root_hides_a_teammates_draw_order() {
        let (mut a, mut b) = coop_pair();
        let mut solo = CombatState {
            player: a.player.clone(),
            ..CombatState::default()
        };
        blindfold_root(&mut a, 0x5EED);
        blindfold_root(&mut b, 0x5EED);
        blindfold_root(&mut solo, 0x5EED);
        assert_eq!(a.allies[0].draw, b.allies[0].draw);
        assert_eq!(a.player.draw, solo.player.draw, "our pile is the solo one");
    }

    #[test]
    fn belief_root_defers_a_teammates_draws() {
        let (mut a, mut b) = coop_pair();
        belief_root(&mut a, 0x5EED);
        belief_root(&mut b, 0x5EED);
        assert_eq!(a.allies[0].draw, b.allies[0].draw);
        assert!(a.allies[0].deferred_draw_order);
        assert!(unknown_middle_is_canonical(&a.allies[0]));
    }

    // ── belief_root (bead `sts2-rs-09x5n`, F2 `HiddenInfoMode::BeliefDraw`) ──

    #[test]
    fn belief_root_is_pure_function_of_seed() {
        let mut a = CombatState::default();
        a.player.draw = [3u32, 1, 4, 1_000, 2]
            .into_iter()
            .map(uid)
            .collect::<SmallVec<_>>();
        let mut b = a.clone();
        b.player.draw = [4u32, 2, 1_000, 1, 3]
            .into_iter()
            .map(uid)
            .collect::<SmallVec<_>>();
        // Advance b's cursors so the two states also differ there.
        let _ = b.rng.next_int(0, 1000);
        let _ = b.run_rng.shuffle.next_int(0, 1000);

        belief_root(&mut a, 0xABC);
        belief_root(&mut b, 0xABC);
        assert_eq!(
            a.player.draw, b.player.draw,
            "two incoming permutations of the same multiset (+ different RNG \
             cursors) must canonicalize to the identical pile"
        );
        assert_eq!(a.run_rng.combat_counter_fingerprint(), [0; 9]);
        assert_eq!(b.run_rng.combat_counter_fingerprint(), [0; 9]);
    }

    #[test]
    fn belief_root_sorts_without_shuffling() {
        // Unlike `blindfold_root`, the pile step is a pure sort — the
        // result is independent of `seed` entirely (only the RNG streams
        // vary by seed), since the random draw ORDER is deferred to draw
        // time rather than materialized here.
        let mut a = player_with_draw(&[5, 3, 1, 4, 2]);
        let expected: SmallVec<[CardUid; 32]> = [1u32, 2, 3, 4, 5].into_iter().map(uid).collect();
        let mut state_a = CombatState {
            player: a.clone(),
            ..CombatState::default()
        };
        let mut state_b = CombatState {
            player: {
                a.draw.reverse();
                a
            },
            ..CombatState::default()
        };
        belief_root(&mut state_a, 0x1111);
        belief_root(&mut state_b, 0x2222);
        assert_eq!(state_a.player.draw, expected);
        assert_eq!(
            state_b.player.draw, expected,
            "a different seed must not change the canonicalized pile order"
        );
    }

    #[test]
    fn belief_root_sets_deferred_draw_order_flag() {
        let mut state = CombatState {
            player: player_with_draw(&[1, 2, 3]),
            ..CombatState::default()
        };
        assert!(!state.player.deferred_draw_order, "flag defaults false");
        belief_root(&mut state, 0x42);
        assert!(
            state.player.deferred_draw_order,
            "belief_root must set the flag so cards::draw defers randomness"
        );
    }

    #[test]
    fn belief_root_preserves_pins() {
        let mut p = player_with_draw(&[5, 3, 1, 4, 2]);
        cards::pin_top(&mut p, uid(5));
        cards::pin_bottom(&mut p, uid(2));
        let mut state = CombatState {
            player: p,
            ..CombatState::default()
        };

        belief_root(&mut state, 0x5EED);

        assert_eq!(state.player.pinned_top.as_slice(), &[uid(5)]);
        assert_eq!(state.player.pinned_bottom.as_slice(), &[uid(2)]);
        assert_eq!(state.player.draw.first().copied(), Some(uid(5)));
        assert_eq!(state.player.draw.last().copied(), Some(uid(2)));
    }
}
