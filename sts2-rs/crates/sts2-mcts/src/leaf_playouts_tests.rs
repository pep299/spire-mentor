//! GH #830, deep bets build 1: [`MctsConfig::leaf_playouts`], the
//! recording-free playouts of [`crate::values`], the chance fingerprint and
//! the honest root view, checked against the search's own internals.
//!
//! Lives inside the crate so it can read the arena: each test steps the
//! search one simulation at a time, finds the leaf that simulation scored
//! (the one node whose `own_visits` moved), and recomputes that leaf's value
//! from the outside.

use super::*;
use crate::values::{playout_value, rng_fingerprint, search_playout, PlayoutMode};
use rand::RngCore;
use sts2_core::run_manager::{Phase, RunManager};

fn install() {
    sts2_content::cards::dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_content::potions::dispatch::ensure_installed();
    sts2_core::relic_effects::ensure_installed();
}

/// A fresh Ironclad run (seed `seed`) with two draw cards added to the
/// starter deck, driven by first-option picks into its first combat.
fn first_combat(seed: i32) -> CombatState {
    install();
    let mut rm: RunManager =
        sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
            seed,
            sts2_core::CharacterId::Ironclad,
            0,
        )
        .with_neow_start();
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);
    for id in [
        sts2_core::cards::CardId::PommelStrike,
        sts2_core::cards::CardId::ShrugItOff,
    ] {
        rm.run_state
            .master_deck
            .push(sts2_content::cards::spec_index::card_pod_from_id(id, false).expect("card spec"));
    }
    for _ in 0..64 {
        if rm.phase() == Phase::Combat {
            return rm.combat.clone().expect("in combat");
        }
        let actions = rm.get_available_actions();
        assert!(!actions.is_empty(), "run stalled before its first combat");
        let _ = rm.take_action(actions[0]);
    }
    panic!("no combat within 64 run actions");
}

/// Positions along a deterministic (index-hashed) walk through the first
/// combat of three runs, every `stride` steps, at most `per_fight` each.
fn positions(stride: usize, per_fight: usize) -> Vec<CombatState> {
    positions_from(&[2, 9, 13], stride, per_fight)
}

/// [`positions`] over the first combats of the runs seeded `seeds`.
fn positions_from(seeds: &[i32], stride: usize, per_fight: usize) -> Vec<CombatState> {
    let mut out = Vec::new();
    for &seed in seeds {
        let mut s = first_combat(seed);
        let mut k = usize::try_from(seed).expect("positive");
        let mut taken = 0;
        for step in 0..200 {
            if is_terminal(&s) || taken == per_fight {
                break;
            }
            let acts = legal_actions(&s);
            if acts.is_empty() {
                break;
            }
            if step % stride == 0 {
                out.push(s.clone());
                taken += 1;
            }
            k = k.wrapping_mul(2_654_435_761).wrapping_add(17);
            let _ = core_apply(&mut s, acts[k % acts.len()].clone());
        }
    }
    out
}

/// `s` turned into a close fight: the player at 10 HP, every enemy at no
/// more than 22, so playouts of one leaf disagree on who wins.
fn close(s: &CombatState) -> CombatState {
    let mut c = s.clone();
    let player_uid = c.player.creature_uid.map(|u| u.0);
    for cr in &mut c.creatures {
        if c.enemy_uids.iter().any(|u| u.0 == cr.id) {
            cr.current_hp = cr.current_hp.min(22);
        } else if Some(cr.id) == player_uid {
            cr.current_hp = 10;
        }
    }
    c.player.current_hp = 10;
    assert_eq!(
        sts2_core::invariants::check_combat_invariants(&c, "close fixture"),
        Ok(())
    );
    c
}

/// The deployed honest search's config (`mcts_config_for` under
/// `BeliefDraw`, depth 200) at `sims` simulations and seed `seed`.
fn deployed(sims: u32, seed: u64, leaf_playouts: u32) -> MctsConfig {
    let mut cfg = MctsConfig {
        max_simulations: sims,
        seed,
        max_rollout_depth: 200,
        hidden_info: HiddenInfoMode::BeliefDraw,
        ..MctsConfig::default()
    };
    cfg.apply_mode_dialect();
    cfg.leaf_playouts = leaf_playouts;
    cfg
}

fn search(state: &CombatState, cfg: MctsConfig) -> Mcts<UniformPolicy> {
    Mcts::new(state.clone(), cfg).with_leaf_eval(LeafEval::TexelWide)
}

/// The policy-RNG seed of the search's next simulation (`one_simulation`'s
/// `sim_rng`).
fn next_sim_seed(m: &Mcts<UniformPolicy>) -> u64 {
    m.config.seed ^ u64::from(m.sim_index).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// What one simulation did at its leaf. The leaf is usually a node created
/// this simulation, but a transposition merge can hand back an existing
/// one, so sums are kept as before/after pairs: `after == before + value`
/// is exactly the search's own `+=`.
struct LeafStep {
    sim_seed: u64,
    leaf: CombatState,
    own_value: (f32, f32),
    bucket: (u32, u32, u32),
    hp_sum: (f32, f32),
    hp_n: u32,
    rollouts: u64,
    leaves: u64,
}

impl LeafStep {
    /// The search backed up exactly `v` at this leaf.
    fn backed_up(&self, v: f32) -> bool {
        (self.own_value.0 + v).to_bits() == self.own_value.1.to_bits()
    }
}

/// Run one simulation and report its leaf.
fn step(m: &mut Mcts<UniformPolicy>) -> LeafStep {
    let before: Vec<(u32, f32, u32, u32, u32, f32, u32)> = m
        .tree
        .arena
        .iter()
        .map(|n| {
            (
                n.own_visits,
                n.own_value_sum,
                n.wins,
                n.losses,
                n.ties,
                n.hp_sum,
                n.hp_n,
            )
        })
        .collect();
    let (rollouts0, leaves0) = (m.rollouts_total, m.rollout_leaves);
    let sim_seed = next_sim_seed(m);
    m.one_simulation();
    m.sim_index = m.sim_index.wrapping_add(1);
    let moved: Vec<usize> = (0..m.tree.arena.len())
        .filter(|&i| before.get(i).map_or(0, |b| b.0) != m.tree.arena[i].own_visits)
        .collect();
    assert_eq!(moved.len(), 1, "exactly one leaf per simulation");
    let i = moved[0];
    let n = &m.tree.arena[i];
    let b = before.get(i).copied().unwrap_or((0, 0.0, 0, 0, 0, 0.0, 0));
    LeafStep {
        sim_seed,
        leaf: (**n.realized.as_ref().expect("leaf has realized state")).clone(),
        own_value: (b.1, n.own_value_sum),
        bucket: (n.wins - b.2, n.losses - b.3, n.ties - b.4),
        hp_sum: (b.5, n.hp_sum),
        hp_n: n.hp_n - b.6,
        rollouts: m.rollouts_total - rollouts0,
        leaves: m.rollout_leaves - leaves0,
    }
}

fn bucket_triple(v: f32, terminal: bool) -> (u32, u32, u32) {
    match if terminal {
        Bucket::from_value(v)
    } else {
        Bucket::Tie
    } {
        Bucket::Win => (1, 0, 0),
        Bucket::Loss => (0, 1, 0),
        Bucket::Tie => (0, 0, 1),
    }
}

/// `values::playout_value(Raw)` on a leaf's realized state, seeded with the
/// simulation's policy seed, IS the search's playout: bit-identical value
/// for every non-decisive leaf of every simulation.
#[test]
fn raw_playout_value_is_the_in_search_playout() {
    let mut checked = 0;
    for (i, s) in positions(9, 2).iter().enumerate() {
        let mut m = search(s, deployed(48, 0xA11CE + i as u64, 1));
        for _ in 0..48 {
            let st = step(&mut m);
            if st.leaves == 0 {
                continue; // decisive leaf: no playout ran
            }
            assert_eq!(st.rollouts, 1);
            let got = playout_value(
                &st.leaf,
                st.sim_seed,
                PlayoutMode::Raw,
                &m.config,
                &m.leaf_eval,
            )
            .expect("raw never fails");
            assert!(
                st.backed_up(got.value),
                "position {i}: values_many(raw) {} is not what the search backed up ({:?})",
                got.value,
                st.own_value
            );
            assert_eq!(st.bucket, bucket_triple(got.value, got.terminal));
            checked += 1;
        }
    }
    assert!(checked > 200, "only {checked} playout leaves compared");
}

/// At `leaf_playouts = 4` every playout leaf runs exactly four playouts and
/// backs up their mean: playout 0 is the unchanged single playout, playouts
/// 1..4 run on clones reseeded by `reseed_for_outcome(clone,
/// sim_rng.next_u64(), i)` and continue the policy RNG. The bucket and the
/// final-HP statistic come from playout 0. Recomputed from outside, bit for
/// bit.
#[test]
fn four_leaf_playouts_back_up_the_mean_with_playout_zero_statistics() {
    let (mut leaves, mut moved_from_first, mut terminal_firsts) = (0, 0, 0);
    // Leaves where playout 0's bucket is not the mean's: only these can tell
    // "bucket from playout 0" from "bucket from the mean".
    let mut bucket_disagreements = 0;
    let mut states = positions(9, 2);
    states.extend(positions(9, 2).iter().map(close));
    for (i, s) in states.iter().enumerate() {
        let mut m = search(s, deployed(40, 0xB0B + i as u64, 4));
        for _ in 0..40 {
            let st = step(&mut m);
            if st.leaves == 0 {
                assert_eq!(st.rollouts, 0, "a decisive leaf runs no playout");
                continue;
            }
            assert_eq!((st.leaves, st.rollouts), (1, 4), "four playouts per leaf");
            let cfg = m.config;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(st.sim_seed);
            let mut first = st.leaf.clone();
            let v0 = search_playout(&mut first, &mut rng, &cfg, &m.leaf_eval);
            let mut sum = f64::from(v0);
            for j in 1..4 {
                let mut s = st.leaf.clone();
                reseed_for_outcome(&mut s, rng.next_u64(), j);
                sum += f64::from(search_playout(&mut s, &mut rng, &cfg, &m.leaf_eval));
            }
            #[allow(clippy::cast_possible_truncation)]
            let mean = (sum / 4.0) as f32;
            assert!(
                st.backed_up(mean),
                "position {i}: backed up {:?}, not the mean of four {mean}",
                st.own_value
            );
            let hit = is_terminal(&first);
            assert_eq!(st.bucket, bucket_triple(v0, hit), "bucket is playout 0's");
            if hit {
                assert_eq!(st.hp_n, 1);
                assert_eq!(
                    (st.hp_sum.0 + final_hp_of(&first)).to_bits(),
                    st.hp_sum.1.to_bits(),
                    "final HP is playout 0's"
                );
            } else {
                assert_eq!(st.hp_n, 0, "a depth-capped playout 0 records no HP");
            }
            leaves += 1;
            moved_from_first += u32::from(mean.to_bits() != v0.to_bits());
            terminal_firsts += u32::from(hit);
            bucket_disagreements += u32::from(bucket_triple(v0, hit) != bucket_triple(mean, hit));
        }
    }
    assert!(leaves > 200, "only {leaves} playout leaves compared");
    assert!(
        moved_from_first * 2 > leaves,
        "the mean equals playout 0 on {} of {leaves} leaves: extra playouts inert?",
        leaves - moved_from_first
    );
    assert!(terminal_firsts > 0, "no playout 0 reached a terminal");
    assert!(
        bucket_disagreements > 0,
        "no leaf where playout 0's bucket differs from the mean's: the bucket check is vacuous"
    );
}

/// Counters and effect over whole searches: `rollouts_total` is the playout
/// leaf count times k (1 and 4), depth-cap hits never exceed playouts, and
/// four playouts change the root's value estimates and shrink the spread of
/// the returns backed up into it.
#[test]
fn leaf_playouts_four_runs_four_per_leaf_and_moves_root_values() {
    let (mut var1, mut var4, mut differ, mut n) = (0.0_f64, 0.0_f64, 0, 0);
    for (i, s) in positions(13, 1).iter().enumerate() {
        let seed = 0xC0DE + i as u64;
        let mut one = search(s, deployed(200, seed, 1));
        let _ = one.try_search_n(200);
        let mut four = search(s, deployed(200, seed, 4));
        let _ = four.try_search_n(200);
        let (r1, c1) = one.rollout_stats();
        let (r4, c4) = four.rollout_stats();
        assert!(one.rollout_leaves() > 0 && four.rollout_leaves() > 0);
        assert_eq!(r1, one.rollout_leaves(), "k=1: one playout per leaf");
        assert_eq!(r4, 4 * four.rollout_leaves(), "k=4: four playouts per leaf");
        assert!(c1 <= r1 && c4 <= r4);

        let q = |m: &Mcts<UniformPolicy>| -> Vec<u32> {
            m.root_outcomes()
                .iter()
                .map(|o| o.value_sum.to_bits())
                .collect()
        };
        differ += u32::from(q(&one) != q(&four));
        let root_var = |m: &Mcts<UniformPolicy>| {
            let r = m.tree.node(m.tree.root);
            let mean = f64::from(r.value_sum) / f64::from(r.visits);
            f64::from(r.value_sq_sum) / f64::from(r.visits) - mean * mean
        };
        var1 += root_var(&one);
        var4 += root_var(&four);
        n += 1;
    }
    assert!(n >= 3, "{n} positions");
    assert_eq!(
        differ, n,
        "leaf_playouts=4 left some root's values unchanged"
    );
    assert!(
        var4 < var1,
        "averaging four playouts did not shrink the return spread: {var4} vs {var1}"
    );
}

/// Edge counts of [`rng_fingerprint_flags_exactly_the_search_chance_edges`].
#[derive(Default)]
struct EdgeCounts {
    chance: u32,
    chance_plays: u32,
    deterministic: u32,
}

impl EdgeCounts {
    fn add(&mut self, is_chance: bool, action: &Action) {
        if is_chance {
            self.chance += 1;
            self.chance_plays += u32::from(matches!(action, Action::PlayCard { .. }));
        } else {
            self.deterministic += 1;
        }
    }
}

/// `values::rng_fingerprint` is the search's chance detector on every
/// expanded edge of the tree, not only the root's: for each decision node
/// with a stored state, a child edge leads to a chance node exactly when
/// applying the edge's action to that state changes the fingerprint.
///
/// Checked on at least 200 positions (the plan's count), and separately
/// below reseeded chance outcomes (ordinal >= 1, whose RNG streams
/// `reseed_for_outcome` replaced; outcome 0 keeps the parent's cursors).
/// Both kinds occur at the root and below a reseed, including chance edges
/// that are card plays (a draw from the hidden pile), not only `EndTurn`.
#[test]
fn rng_fingerprint_flags_exactly_the_search_chance_edges() {
    let states = positions_from(&(1..=24).collect::<Vec<_>>(), 2, 10);
    assert!(states.len() >= 200, "only {} positions", states.len());
    let (mut root, mut reseeded) = (EdgeCounts::default(), EdgeCounts::default());
    for (i, s) in states.iter().enumerate() {
        let n_acts = u32::try_from(legal_actions(s).len()).expect("small");
        // The first `n_acts` simulations expand every root edge (each
        // expands one untried root edge); the rest descend and expand
        // below chance outcomes. Chance detection does not read the
        // playout, so a short one keeps the 200 positions cheap.
        let sims = n_acts + 40;
        let mut cfg = deployed(sims, 0xF1F0 + i as u64, 1);
        cfg.max_rollout_depth = 4;
        let mut m = search(s, cfg);
        for _ in 0..sims {
            m.one_simulation();
            m.sim_index = m.sim_index.wrapping_add(1);
        }
        let root_node = m.tree.node(m.tree.root);
        assert!(root_node.untried_actions.is_empty() || root_node.is_terminal);
        // (node, depth, below a reseeded outcome)
        let mut stack = vec![(m.tree.root, 0_u32, false)];
        let mut seen = std::collections::HashSet::new();
        while let Some((id, depth, below_reseed)) = stack.pop() {
            if !seen.insert(id.0) {
                continue; // a transposition reached twice
            }
            let node = m.tree.node(id);
            if node.kind == NodeKind::Chance {
                for o in &node.chance_outcomes {
                    stack.push((o.child, depth + 1, below_reseed || o.n >= 1));
                }
                continue;
            }
            // Proven / terminal nodes drop their state; nothing to check.
            let Some(state) = node.realized.as_deref() else {
                continue;
            };
            for (edge, child) in &node.children {
                let action = edge.resolve(state).expect("expanded edge resolves");
                let mut after = state.clone();
                let before = rng_fingerprint(&after);
                let _ = core_apply(&mut after, action.clone());
                let flagged = rng_fingerprint(&after) != before;
                let is_chance = m.tree.node(*child).kind == NodeKind::Chance;
                assert_eq!(
                    flagged, is_chance,
                    "position {i}, depth {depth}, below a reseed {below_reseed}, {action:?}: \
                     fingerprint says chance={flagged}, the search built chance={is_chance}"
                );
                if id == m.tree.root {
                    root.add(is_chance, &action);
                } else if below_reseed {
                    reseeded.add(is_chance, &action);
                }
                stack.push((*child, depth + 1, below_reseed));
            }
        }
    }
    for (name, c) in [("root", &root), ("below a reseed", &reseeded)] {
        assert!(
            c.chance >= 100 && c.deterministic >= 100,
            "{name}: {} chance / {} deterministic edges",
            c.chance,
            c.deterministic
        );
        assert!(c.chance_plays > 0, "{name}: no card play was a chance edge");
    }
}

/// [`values::fingerprint_sees_draws`] is the precondition for reading
/// [`rng_fingerprint`] as the search's chance detector, and it is needed: on
/// a raw position (materialized draw order) a card that draws from the
/// hidden pile moves no RNG, while on the belief view the deployed search
/// actually searches, the same play moves the fingerprint. The helper says
/// `false` on exactly the raw positions with a hidden order and `true` on
/// every belief view.
#[test]
fn rng_fingerprint_needs_a_belief_view_to_see_draws() {
    let mut missed_draws = 0;
    for s in positions(1, 30) {
        assert!(!s.player.deferred_draw_order, "fixture is a raw position");
        let hidden = sts2_core::cards::unknown_middle_uids(&s.player).len() >= 2;
        assert_eq!(values::fingerprint_sees_draws(&s), !hidden);
        let mut view = s.clone();
        sts2_core::blindfold::belief_root(&mut view, 3);
        assert!(values::fingerprint_sees_draws(&view));
        if !hidden {
            continue;
        }
        for a in legal_actions(&s) {
            if !matches!(a, Action::PlayCard { .. }) {
                continue;
            }
            let mut raw = s.clone();
            let raw_before = rng_fingerprint(&raw);
            let _ = core_apply(&mut raw, a.clone());
            if raw.cards_drawn == s.cards_drawn {
                continue; // drew nothing
            }
            let mut seen = view.clone();
            let seen_before = rng_fingerprint(&seen);
            let _ = core_apply(&mut seen, a);
            missed_draws += u32::from(
                rng_fingerprint(&raw) == raw_before && rng_fingerprint(&seen) != seen_before,
            );
        }
    }
    assert!(
        missed_draws > 0,
        "no draw the raw fingerprint missed and the belief view's caught"
    );
}

/// A Vantom fight: the one enemy that recommends a longer playout horizon
/// (50, [`sts2_core::MonsterKind::recommended_rollout_depth`]).
fn vantom_combat(seed: i32) -> CombatState {
    install();
    let mut rm: RunManager =
        sts2_content::run_callbacks::new_run_manager_for_character_with_ascension(
            seed,
            sts2_core::CharacterId::Ironclad,
            0,
        )
        .with_neow_start();
    sts2_content::run_callbacks::initialize_run_setup(&mut rm);
    let kind = sts2_core::MonsterKind::Vantom;
    let rs = &mut rm.run_state;
    let hp = sts2_core::monsters::roll_unique_max_hps(&[kind], &mut rs.rng.niche, 0)[0];
    let roster = [(
        kind,
        sts2_content::monster_dispatch::MonsterSpawnArgs::None,
        hp,
    )];
    sts2_content::run_bridge::enter_combat_from_roster(
        rs,
        &roster,
        sts2_core::runstate::RoomType::Boss,
    )
}

/// `playout_value` raises the playout depth exactly as the search does
/// ([`effective_rollout_depth`]): a Vantom fight searched at a configured
/// depth of 10 runs 50-action playouts, and `playout_value` handed the
/// caller's depth-10 config still returns what the search backed up, bit
/// for bit. Non-vacuous: a depth-10 playout of the same leaf differs.
#[test]
fn playout_value_raises_the_depth_as_the_search_does() {
    let s = vantom_combat(4);
    let mut cfg = deployed(40, 0xDEE7, 1);
    cfg.max_rollout_depth = 10;
    assert_eq!(effective_rollout_depth(&s, &cfg), 50);
    let mut m = search(&s, cfg);
    assert_eq!(
        m.config.max_rollout_depth, 50,
        "the search raised its depth"
    );
    let (mut checked, mut depth_mattered) = (0_u32, 0_u32);
    for _ in 0..40 {
        let st = step(&mut m);
        if st.leaves == 0 {
            continue;
        }
        let got = playout_value(&st.leaf, st.sim_seed, PlayoutMode::Raw, &cfg, &m.leaf_eval)
            .expect("raw never fails");
        assert!(
            st.backed_up(got.value),
            "playout_value {} is not what the search backed up ({:?})",
            got.value,
            st.own_value
        );
        let mut short = st.leaf.clone();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(st.sim_seed);
        let v10 = search_playout(&mut short, &mut rng, &cfg, &m.leaf_eval);
        depth_mattered += u32::from(v10.to_bits() != got.value.to_bits());
        checked += 1;
    }
    assert!(checked >= 20, "only {checked} playout leaves");
    assert!(
        depth_mattered * 2 > checked,
        "depth 10 and 50 agree on {} of {checked} leaves: the depth is not exercised",
        checked - depth_mattered
    );
}

/// `belief_root` (the Python `belief_view`) is idempotent and forgets the
/// view it starts from: applying it to any earlier view, at any seed, then
/// at `seed` gives exactly `belief_root(seed)` of the original. The hidden
/// part it rewrites (the unknown middle of the draw pile, every RNG stream)
/// is a function of the observable part and `seed` alone.
#[test]
fn belief_view_is_idempotent() {
    let view = |s: &CombatState, seed: u64| {
        let mut out = s.clone();
        sts2_core::blindfold::belief_root(&mut out, seed);
        out
    };
    for s in positions(5, 3) {
        let once = view(&s, 77);
        assert!(once == view(&once, 77), "belief_view(belief_view(s)) moved");
        assert!(
            once == view(&view(&s, 5), 77),
            "a prior view leaked through"
        );
        assert!(once != view(&s, 78), "seed is ignored: vacuous");
    }
}

/// Honesty: the belief view of every possible world consistent with what
/// the player sees (`blindfold_root`, the Python `sample_world`) is one
/// state, equal to the true position's belief view. Non-vacuous: the worlds
/// themselves differ (draw order and RNG streams) — and `search_hash`
/// cannot see that difference at all, so it is no honesty detector by
/// itself.
#[test]
fn belief_view_of_any_sampled_world_is_the_same_state() {
    for s in positions(5, 3) {
        let mut truth = s.clone();
        sts2_core::blindfold::belief_root(&mut truth, 4242);
        let mut worlds = Vec::new();
        for r in 0..6_u64 {
            let mut w = s.clone();
            sts2_core::blindfold::blindfold_root(&mut w, 1000 + r);
            let mut v = w.clone();
            sts2_core::blindfold::belief_root(&mut v, 4242);
            assert!(
                v == truth,
                "world {r}: belief view differs from the truth's"
            );
            assert_eq!(sts2_core::search_hash(&v), sts2_core::search_hash(&truth));
            assert_eq!(sts2_core::state_hash(&v), sts2_core::state_hash(&truth));
            worlds.push(w);
        }
        for pair in worlds.windows(2) {
            assert!(pair[0] != pair[1], "two sampled worlds are identical");
            assert_ne!(
                sts2_core::state_hash(&pair[0]),
                sts2_core::state_hash(&pair[1]),
                "worlds differ only below state_hash"
            );
            assert_eq!(
                sts2_core::search_hash(&pair[0]),
                sts2_core::search_hash(&pair[1]),
                "search_hash saw the hidden part"
            );
        }
    }
}

/// `Belief` is `Raw` on the belief view at the same seed; `BeliefEndTurn`
/// is `Raw` on that view after `EndTurn`; a terminal state scores as it
/// stands.
#[test]
fn belief_modes_are_raw_on_the_belief_view() {
    let cfg = deployed(1000, 0, 1);
    let leaf = LeafEval::TexelWide;
    for (i, s) in positions(7, 2).iter().enumerate() {
        let seed = 0x5EED + i as u64;
        let mut view = s.clone();
        sts2_core::blindfold::belief_root(&mut view, seed);
        let raw_view = playout_value(&view, seed, PlayoutMode::Raw, &cfg, &leaf).expect("raw");
        let belief = playout_value(s, seed, PlayoutMode::Belief, &cfg, &leaf).expect("belief");
        assert_eq!(belief, raw_view);
        if legal_actions(&view).contains(&Action::EndTurn) {
            let mut ended = view.clone();
            let _ = core_apply(&mut ended, Action::EndTurn);
            let want = playout_value(&ended, seed, PlayoutMode::Raw, &cfg, &leaf).expect("raw");
            let got =
                playout_value(s, seed, PlayoutMode::BeliefEndTurn, &cfg, &leaf).expect("endturn");
            assert_eq!(got, want);
        }
    }
    let mut dead = positions(7, 1).remove(0);
    dead.is_over = true;
    dead.player_won = true;
    let v = playout_value(&dead, 1, PlayoutMode::BeliefEndTurn, &cfg, &leaf).expect("terminal");
    assert!(v.terminal && v.won);
}
