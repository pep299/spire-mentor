//! Features of a turn plan, the deck context it is played in, baseline
//! anchors, and a plan's log-probability under a playout net (GH #830,
//! turn-plan search G0).
//!
//! A learned scorer ranks the plans [`crate::turn_plan::enumerate_plans`]
//! lists. [`plan_features`] describes what a plan does to the board,
//! [`deck_context`] the position it is played from, [`anchors`] flags the
//! plans simple baselines would pick, and [`plan_logprob`] scores a plan
//! under a playout policy.
//!
//! Every function replays a plan on a clone of the caller's RAW state, not
//! the hidden-info view the enumerator searched under. That is valid only
//! when the two agree on every prefix action. It does NOT hold on the
//! pre-#785 engine for Havoc and Cascade, which pop the top draw card
//! without consuming RNG: the enumerator plans past them with the belief
//! view's top card, the replay plays the true one. G1 must replay on the
//! enumerator's root view (#912). The last action of an
//! `EndTurn` or `Chance` plan is never applied: features describe the board
//! the player commits to, not the enemy turn or a roll.
#![allow(clippy::cast_precision_loss)]

use sts2_core::cards::{Card, CardType};
use sts2_core::{
    apply, creature_by_id, is_terminal, legal_actions, tt_hash_v2, Action, CombatState, Creature,
    PowerKind,
};

use crate::expert::END_TURN_PRIORITY;
use crate::play_identity;
use crate::rollout_net::{heuristic_mirror_net, net_for, RolloutNet};
use crate::turn_plan::TURN_PLAN_ROOT_SEED;
use crate::turn_plan::{PlanEnd, PlanSet, TurnPlan};
use crate::{apply_hidden_info_root, canonicalize_actions, HiddenInfoMode};

/// Width of [`plan_features`]:
///
/// | idx | feature |
/// |---|---|
/// | 0 | player block gained |
/// | 1 | player HP lost |
/// | 2 | energy left |
/// | 3 | enemy HP removed (sum over the enemies present before the plan) |
/// | 4 | enemies killed |
/// | 5 | min HP fraction left over living enemies (0 when none lives) |
/// | 6 | Strength gained |
/// | 7 | Dexterity gained |
/// | 8 | Vulnerable stacks applied (sum over enemies) |
/// | 9 | Weak stacks applied (sum over enemies) |
/// | 10 | cards played |
/// | 11 | cards exhausted |
/// | 12 | powers played |
/// | 13 | potions drunk |
/// | 14 | base telegraphed incoming damage minus block, floored at 0 |
/// | 15–17 | `end` one-hot: `EndTurn`, `Chance`, `Terminal` |
///
/// 0–9, 11 and 14 are read off the state after the replay (so they leave
/// out a final `Chance` action's effect); 10, 12 and 13 count every action
/// of the plan, a final `Chance` card play included.
pub const PLAN_DIM: usize = 18;

/// Upper bound on a greedy line's length in [`anchors`]; a turn this long
/// is outside any enumerated plan set anyway.
const MAX_GREEDY_STEPS: usize = 256;

/// The card an action plays at `state` (`None` for a non-card action).
fn played_card(state: &CombatState, action: &Action) -> Option<Card> {
    match action {
        Action::PlayCard { hand_index, .. } => state
            .player
            .hand
            .get(usize::from(*hand_index))
            .and_then(|uid| state.cards.get(uid.0.get() as usize - 1))
            .copied(),
        _ => None,
    }
}

/// A plan replayed on a clone of `pre`.
struct Replay {
    /// The state after every applied action.
    post: CombatState,
    /// The card each action plays, read before it is applied.
    cards: Vec<Option<Card>>,
    potions: u32,
}

/// Replay `plan` on `pre`, applying every action except the last of an
/// `EndTurn` or `Chance` plan.
fn replay(pre: &CombatState, plan: &TurnPlan) -> Replay {
    let n_apply = match plan.end {
        PlanEnd::Terminal => plan.actions.len(),
        PlanEnd::EndTurn | PlanEnd::Chance => plan.actions.len().saturating_sub(1),
    };
    let mut post = pre.clone();
    let mut cards = Vec::with_capacity(plan.actions.len());
    let mut potions = 0;
    for (i, action) in plan.actions.iter().enumerate() {
        cards.push(played_card(&post, action));
        potions += u32::from(matches!(action, Action::DrinkPotion { .. }));
        if i < n_apply {
            let _ = apply(&mut post, action.clone());
        }
    }
    Replay {
        post,
        cards,
        potions,
    }
}

/// `kind`'s amount on creature `c`, 0 when absent.
fn creature_power(state: &CombatState, c: &Creature, kind: PowerKind) -> i32 {
    c.powers
        .iter()
        .filter_map(|pu| state.powers.iter().find(|p| p.id == pu.0))
        .filter(|p| p.kind == kind)
        .map(|p| p.amount)
        .sum()
}

/// The enemy creatures of `state`, in `enemy_uids` order.
fn enemies(state: &CombatState) -> impl Iterator<Item = &Creature> {
    state
        .enemy_uids
        .iter()
        .filter_map(|uid| creature_by_id(&state.creatures, uid.0))
}

fn is_alive(c: &Creature) -> bool {
    c.current_hp > 0 && !c.is_beaten_sentinel()
}

/// Base telegraphed damage (intent damage × hits, no modifiers) of the
/// enemies the playout net counts: the living ones and beaten sentinels
/// (`rollout_net::StepCtx`).
fn base_incoming(state: &CombatState) -> i32 {
    enemies(state)
        .filter(|c| c.current_hp > 0 || c.is_beaten_sentinel())
        .filter_map(|c| c.intent)
        .map(|i| i.damage().saturating_mul(i32::from(i.hits())))
        .sum()
}

/// [`PLAN_DIM`] features of `plan` played from `pre`; see [`PLAN_DIM`] for
/// the layout.
#[must_use]
pub fn plan_features(pre: &CombatState, plan: &TurnPlan) -> [f32; PLAN_DIM] {
    let r = replay(pre, plan);
    let post = &r.post;
    let (mut hp_removed, mut killed, mut vuln, mut weak) = (0i32, 0i32, 0i32, 0i32);
    for before in enemies(pre) {
        let after = creature_by_id(&post.creatures, before.id);
        let hp_after = after.map_or(0, |c| c.current_hp.max(0));
        hp_removed += (before.current_hp.max(0) - hp_after).max(0);
        killed += i32::from(is_alive(before) && !after.is_some_and(is_alive));
    }
    let mut min_frac = f32::INFINITY;
    for after in enemies(post) {
        let before = creature_by_id(&pre.creatures, after.id);
        let delta = |kind| {
            let was = before.map_or(0, |c| creature_power(pre, c, kind));
            (creature_power(post, after, kind) - was).max(0)
        };
        vuln += delta(PowerKind::Vulnerable);
        weak += delta(PowerKind::Weak);
        if is_alive(after) {
            min_frac = min_frac.min(after.current_hp as f32 / after.max_hp.max(1) as f32);
        }
    }
    if !min_frac.is_finite() {
        min_frac = 0.0;
    }
    let played = || r.cards.iter().flatten();
    let gain = |kind| post.player_power_amount(kind) - pre.player_power_amount(kind);
    let mut f = [0.0f32; PLAN_DIM];
    f[0] = (post.player.block - pre.player.block) as f32;
    f[1] = (pre.player.current_hp - post.player.current_hp) as f32;
    f[2] = f32::from(post.player.energy);
    f[3] = hp_removed as f32;
    f[4] = killed as f32;
    f[5] = min_frac;
    f[6] = gain(PowerKind::Strength) as f32;
    f[7] = gain(PowerKind::Dexterity) as f32;
    f[8] = vuln as f32;
    f[9] = weak as f32;
    f[10] = played().count() as f32;
    f[11] = post
        .player
        .exhaust
        .len()
        .saturating_sub(pre.player.exhaust.len()) as f32;
    f[12] = played().filter(|c| c.card_type == CardType::Power).count() as f32;
    f[13] = r.potions as f32;
    f[14] = (base_incoming(post) - post.player.block).max(0) as f32;
    f[15 + match plan.end {
        PlanEnd::EndTurn => 0,
        PlanEnd::Chance => 1,
        PlanEnd::Terminal => 2,
    }] = 1.0;
    f
}

/// A card's name: its `CardId` debug name, `+` when upgraded.
fn card_name(card: &Card) -> String {
    format!("{:?}{}", card.card_id, if card.upgraded { "+" } else { "" })
}

/// The cards `plan` plays from `pre`, one name per card play in order
/// (a final `Chance` card play included).
#[must_use]
pub fn plan_cards(pre: &CombatState, plan: &TurnPlan) -> Vec<String> {
    replay(pre, plan)
        .cards
        .iter()
        .flatten()
        .map(card_name)
        .collect()
}

/// The position a plan is played from.
#[derive(Clone, Debug, PartialEq)]
pub struct DeckContext {
    /// Every card in hand, draw, discard and exhaust piles ([`plan_cards`]
    /// naming), sorted: a multiset, so the hidden draw order never leaks.
    pub cards: Vec<String>,
    /// The player's relics (`RelicKind` debug names).
    pub relics: Vec<String>,
    /// The player's powers (`PowerKind` debug name, amount).
    pub powers: Vec<(String, i32)>,
    /// `[turn number, player HP fraction, energy, max energy, base
    /// telegraphed incoming damage / max HP, alive enemy count]`.
    pub ctx: [f32; 6],
}

/// The [`DeckContext`] of `state`.
#[must_use]
pub fn deck_context(state: &CombatState) -> DeckContext {
    let p = &state.player;
    let mut cards: Vec<String> = [&p.hand[..], &p.draw[..], &p.discard[..], &p.exhaust[..]]
        .into_iter()
        .flatten()
        .filter_map(|uid| state.cards.get(uid.0.get() as usize - 1))
        .map(card_name)
        .collect();
    cards.sort_unstable();
    let relics = state
        .relics
        .iter()
        .map(|r| format!("{:?}", r.kind))
        .collect();
    let powers = state
        .player_power_uids()
        .iter()
        .filter_map(|pu| state.powers.iter().find(|pw| pw.id == pu.0))
        .map(|pw| (format!("{:?}", pw.kind), pw.amount))
        .collect();
    let max_hp = p.max_hp.max(1) as f32;
    let ctx = [
        f32::from(state.round),
        p.current_hp.max(0) as f32 / max_hp,
        f32::from(p.energy),
        f32::from(p.max_energy),
        base_incoming(state) as f32 / max_hp,
        enemies(state).filter(|c| is_alive(c)).count() as f32,
    ];
    DeckContext {
        cards,
        relics,
        powers,
        ctx,
    }
}

/// Which baselines would pick a plan; one per plan of the set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AnchorFlags {
    /// The plan wins the fight: a `Terminal` plan with the player alive, or
    /// a `Chance` plan whose final action, applied on the enumerator's
    /// belief view of `pre` (`apply_hidden_info_root` under `BeliefDraw`
    /// with its seed), ends the combat with the player alive. The belief
    /// view is one realization of the roll; the real draw order is never
    /// read.
    pub lethal: bool,
    /// The first `EndTurn` plan with the most block gained (feature 0).
    pub max_block: bool,
    /// The first `EndTurn` plan with the most enemy HP removed (feature 3).
    pub max_damage: bool,
    /// The greedy line of [`heuristic_mirror_net`].
    pub heuristic: bool,
    /// The greedy line of the deployed playout net ([`net_for`]); never set
    /// when the character has none.
    pub rl2: bool,
}

/// What ends a plan, keyed the way the enumerator dedups plans: the state
/// `EndTurn` is chosen from, the state and action of a random step, or the
/// state a terminal action reaches. Hashes are of the replay on the
/// caller's state, so plans and greedy lines compare on equal terms.
#[derive(Clone, Debug, PartialEq)]
enum Boundary {
    EndTurn(u64),
    Chance(u64, Action),
    Terminal(u64),
}

/// Each plan's [`Boundary`], and the hash of every state a plan passes
/// through before its last action.
fn boundaries(pre: &CombatState, set: &PlanSet) -> (Vec<Boundary>, Vec<u64>) {
    let mut keys = Vec::with_capacity(set.plans.len());
    let mut inner = Vec::new();
    for plan in &set.plans {
        let mut s = pre.clone();
        let (last, prefix) = plan.actions.split_last().expect("a plan has an action");
        for a in prefix {
            inner.push(tt_hash_v2(&s));
            let _ = apply(&mut s, a.clone());
        }
        let h = tt_hash_v2(&s);
        inner.push(h);
        keys.push(match plan.end {
            PlanEnd::EndTurn => Boundary::EndTurn(h),
            PlanEnd::Chance => Boundary::Chance(h, last.clone()),
            PlanEnd::Terminal => {
                let _ = apply(&mut s, last.clone());
                Boundary::Terminal(tt_hash_v2(&s))
            }
        });
    }
    inner.sort_unstable();
    inner.dedup();
    (keys, inner)
}

/// `action`'s representative among `canonicalize_actions` — the hand copy
/// the enumerator plays for an identical card.
fn canonical(state: &CombatState, legal: &[Action], action: &Action) -> Action {
    let Some(id) = play_identity(state, action) else {
        return action.clone();
    };
    canonicalize_actions(state, legal.iter().cloned().collect())
        .into_iter()
        .find(|c| play_identity(state, c) == Some(id))
        .expect("a legal play has a canonical representative")
}

/// The index of the plan `net`'s greedy (argmax-logit, first on ties) line
/// from `pre` matches, if the set holds it.
///
/// The line is walked step by step against the set instead of re-deriving
/// randomness: at each state, `EndTurn` matches the `EndTurn` plan ending
/// there; any other action matches a `Chance` plan with that state and
/// action if there is one (the enumerator found it random); otherwise it is
/// applied, and a terminal result matches the `Terminal` plan reaching that
/// state. The walk gives up (`None`) once it reaches a state no plan passes
/// through — a random action whose plan the cap cut, for instance.
fn greedy_plan(
    net: &RolloutNet,
    pre: &CombatState,
    keys: &[Boundary],
    inner: &[u64],
) -> Option<usize> {
    let find = |b: &Boundary| keys.iter().position(|k| k == b);
    let mut s = pre.clone();
    for _ in 0..MAX_GREEDY_STEPS {
        let h = tt_hash_v2(&s);
        if inner.binary_search(&h).is_err() {
            return None;
        }
        let legal = legal_actions(&s);
        let logits = net.logits(&s, &legal, END_TURN_PRIORITY);
        let best = (0..legal.len()).reduce(|b, i| if logits[i] > logits[b] { i } else { b })?;
        let action = canonical(&s, &legal, &legal[best]);
        if action == Action::EndTurn {
            return find(&Boundary::EndTurn(h));
        }
        if let Some(i) = find(&Boundary::Chance(h, action.clone())) {
            return Some(i);
        }
        let _ = apply(&mut s, action);
        if is_terminal(&s) {
            return find(&Boundary::Terminal(tt_hash_v2(&s)));
        }
    }
    None
}

/// Index of the first maximum of `score` over the `EndTurn` plans.
fn argmax_end_turn(set: &PlanSet, score: impl Fn(usize) -> f32) -> Option<usize> {
    (0..set.plans.len())
        .filter(|&i| set.plans[i].end == PlanEnd::EndTurn)
        .reduce(|b, i| if score(i) > score(b) { i } else { b })
}

/// [`AnchorFlags`] for every plan of `set`, which was enumerated from
/// `pre`. `max_block`, `max_damage`, `heuristic` and `rl2` each flag at
/// most one plan; `heuristic` / `rl2` flag none when the greedy line is not
/// in the set (a capped enumeration).
#[must_use]
pub fn anchors(pre: &CombatState, set: &PlanSet) -> Vec<AnchorFlags> {
    let mut flags = vec![AnchorFlags::default(); set.plans.len()];
    let feats: Vec<[f32; PLAN_DIM]> = set.plans.iter().map(|p| plan_features(pre, p)).collect();
    let mut belief: Option<CombatState> = None;
    for (i, plan) in set.plans.iter().enumerate() {
        flags[i].lethal = match plan.end {
            PlanEnd::Terminal => replay(pre, plan).post.player.current_hp > 0,
            PlanEnd::Chance => {
                let view = belief.get_or_insert_with(|| {
                    let mut b = pre.clone();
                    apply_hidden_info_root(&mut b, HiddenInfoMode::BeliefDraw, TURN_PLAN_ROOT_SEED);
                    b
                });
                let mut s = view.clone();
                for a in &plan.actions {
                    let _ = apply(&mut s, a.clone());
                }
                is_terminal(&s) && s.player.current_hp > 0
            }
            PlanEnd::EndTurn => false,
        };
    }
    if let Some(i) = argmax_end_turn(set, |i| feats[i][0]) {
        flags[i].max_block = true;
    }
    if let Some(i) = argmax_end_turn(set, |i| feats[i][3]) {
        flags[i].max_damage = true;
    }
    let (keys, inner) = boundaries(pre, set);
    if let Some(i) = greedy_plan(&heuristic_mirror_net(), pre, &keys, &inner) {
        flags[i].heuristic = true;
    }
    if let Some(i) = net_for(pre).and_then(|net| greedy_plan(net, pre, &keys, &inner)) {
        flags[i].rl2 = true;
    }
    flags
}

/// Σ log-softmax (temperature 1) of `net`'s logits at each step of `plan`
/// from `pre`, for the action the plan takes there — the plan's
/// log-probability as the playout would sample it with the playout's
/// `EndTurn` priority, over all legal actions (a hand copy the plan does
/// not play keeps its own share).
///
/// # Panics
/// When an action of `plan` is not legal at its step: the plan was not
/// enumerated from `pre`.
#[must_use]
pub fn plan_logprob(net: &RolloutNet, pre: &CombatState, plan: &TurnPlan) -> f32 {
    let mut s = pre.clone();
    let mut total = 0.0f32;
    for (i, action) in plan.actions.iter().enumerate() {
        let legal = legal_actions(&s);
        let idx = legal
            .iter()
            .position(|a| a == action)
            .unwrap_or_else(|| panic!("plan action {action:?} is illegal at step {i}"));
        let logits = net.logits(&s, &legal, END_TURN_PRIORITY);
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + logits.iter().map(|l| (l - m).exp()).sum::<f32>().ln();
        total += logits[idx] - lse;
        if i + 1 < plan.actions.len() {
            let _ = apply(&mut s, action.clone());
        }
    }
    total
}
