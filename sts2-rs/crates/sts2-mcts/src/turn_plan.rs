//! Turn-plan enumeration (GH #830, turn-plan search G0).
//!
//! A *plan* is a deterministic action sequence of the player's current
//! turn: it runs until `EndTurn`, until the first action that consumes
//! combat randomness (a draw under `BeliefDraw`, a random target, a roll),
//! or until the combat ends. [`enumerate_plans`] lists every such plan,
//! deduped by the state they reach, so a later stage can price each one.
//!
//! Randomness is detected with the same predicate MCTS expansion uses to
//! allocate a chance node (`apply_detect_stochastic`), and the
//! root gets the same hidden-info conversion the search applies
//! (`apply_hidden_info_root`), so a plan boundary here is a
//! chance node there.

use std::collections::HashSet;

use sts2_core::{is_terminal, legal_actions, tt_hash_v2, Action, CombatState};

use crate::{
    apply_detect_stochastic, apply_hidden_info_root, canonicalize_actions, play_identity,
    HiddenInfoMode,
};

/// Fixed seed for the root hidden-info conversion. Plans only depend on
/// which transitions are random, never on how the reseeded streams roll.
pub(crate) const TURN_PLAN_ROOT_SEED: u64 = 0;

/// How a [`TurnPlan`] ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanEnd {
    /// The last action is `EndTurn` (never applied by the enumerator).
    EndTurn,
    /// The last action consumed combat randomness.
    Chance,
    /// The last action ended the combat (a kill or a death).
    Terminal,
}

/// One deterministic action sequence of the current turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnPlan {
    /// Actions in order; hand indices are valid against the state each
    /// action is applied to, starting from the enumerated state.
    pub actions: Vec<Action>,
    pub end: PlanEnd,
}

/// Budget for [`enumerate_plans`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanCap {
    /// Maximum number of plans returned.
    pub max_plans: usize,
    /// Maximum number of non-`EndTurn` actions applied during the search.
    pub max_nodes: usize,
}

impl Default for PlanCap {
    fn default() -> Self {
        Self {
            max_plans: 512,
            max_nodes: 20_000,
        }
    }
}

/// The plans of one turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanSet {
    pub plans: Vec<TurnPlan>,
    /// A cap stopped the enumeration; `plans` is a prefix of the full set.
    pub cap_hit: bool,
}

/// Enumerate the deterministic plans of `state`'s current turn.
///
/// Depth-first over `canonicalize_actions` order (so identical hand
/// copies are one action and the output order is deterministic). A
/// reached non-chance state (terminal or not) whose `tt_hash_v2` was
/// already seen is skipped, so every `EndTurn` plan ends from a distinct
/// state and every `Terminal` plan reaches a distinct state. `Chance`
/// plans are deduped on `(tt_hash_v2(pre-action state), action)`.
#[must_use]
pub fn enumerate_plans(state: &CombatState, hidden_info: HiddenInfoMode, cap: PlanCap) -> PlanSet {
    let mut root = state.clone();
    apply_hidden_info_root(&mut root, hidden_info, TURN_PLAN_ROOT_SEED);
    let mut e = Enumerator {
        hidden_info,
        cap,
        plans: Vec::new(),
        seen: HashSet::new(),
        chance_seen: HashSet::new(),
        nodes: 0,
        cap_hit: false,
    };
    e.seen.insert(tt_hash_v2(&root));
    e.dfs(&root, &mut Vec::new());
    PlanSet {
        plans: e.plans,
        cap_hit: e.cap_hit,
    }
}

struct Enumerator {
    hidden_info: HiddenInfoMode,
    cap: PlanCap,
    plans: Vec<TurnPlan>,
    seen: HashSet<u64>,
    /// `(tt_hash_v2(pre-action state), action)`; `Action` is not `Hash`,
    /// so the action is keyed by its `Debug` rendering.
    chance_seen: HashSet<(u64, String)>,
    nodes: usize,
    cap_hit: bool,
}

impl Enumerator {
    fn emit(&mut self, prefix: &[Action], last: Action, end: PlanEnd) {
        if self.plans.len() >= self.cap.max_plans {
            self.cap_hit = true;
            return;
        }
        let mut actions = Vec::with_capacity(prefix.len() + 1);
        actions.extend_from_slice(prefix);
        actions.push(last);
        self.plans.push(TurnPlan { actions, end });
    }

    fn dfs(&mut self, state: &CombatState, prefix: &mut Vec<Action>) {
        let acts = canonicalize_actions(state, legal_actions(state));
        let mut state_hash = None;
        for action in acts {
            if self.cap_hit {
                return;
            }
            if action == Action::EndTurn {
                self.emit(prefix, action, PlanEnd::EndTurn);
                continue;
            }
            if self.nodes >= self.cap.max_nodes {
                self.cap_hit = true;
                return;
            }
            self.nodes += 1;
            let mut work = state.clone();
            if apply_detect_stochastic(&mut work, action.clone(), self.hidden_info) {
                let pre = *state_hash.get_or_insert_with(|| tt_hash_v2(state));
                if self.chance_seen.insert((pre, format!("{action:?}"))) {
                    self.emit(prefix, action, PlanEnd::Chance);
                }
                continue;
            }
            if !self.seen.insert(tt_hash_v2(&work)) {
                continue;
            }
            if is_terminal(&work) {
                self.emit(prefix, action, PlanEnd::Terminal);
                continue;
            }
            prefix.push(action);
            self.dfs(&work, prefix);
            prefix.pop();
        }
    }
}

/// Where an externally played turn (a human's, say) lands in a [`PlanSet`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanLocation {
    /// Actions of the played turn that form its plan: through the first
    /// `EndTurn`, RNG-consuming or combat-ending action. `0` when the
    /// sequence is illegal or incomplete.
    pub used: usize,
    /// How the played plan ends; `None` when it could not be cut.
    pub end: Option<PlanEnd>,
    /// The index in `set.plans` of the plan reaching the same state (the
    /// same pre-action state and action identity for a `Chance` plan).
    pub index: Option<usize>,
    /// Why the played turn has no plan: `"illegal"` (an action is not
    /// legal at its state), `"incomplete"` (the actions ran out before the
    /// plan ended). `None` when `end` is set.
    pub failure: Option<&'static str>,
}

/// A plan's identity: what [`enumerate_plans`] dedups on.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PlanKey {
    EndTurn(u64),
    Terminal(u64),
    /// Pre-action state, and the action by play identity (hand slots of a
    /// transposed hand differ, the card does not).
    Chance(u64, String),
}

fn action_key(state: &CombatState, a: &Action) -> String {
    play_identity(state, a).map_or_else(|| format!("{a:?}"), |id| format!("{id:?}"))
}

/// The plan key of a [`TurnPlan`]'s actions replayed from `root` (the
/// hidden-info view it was enumerated on); `None` if the replay does not
/// end where the plan says it does.
fn plan_key(root: &CombatState, plan: &TurnPlan, hidden_info: HiddenInfoMode) -> Option<PlanKey> {
    let mut state = root.clone();
    let last = plan.actions.len().checked_sub(1)?;
    for (i, a) in plan.actions.iter().enumerate() {
        let (key, end) = match step_key(&mut state, a, hidden_info) {
            Some(k) => k,
            None if i < last => continue,
            None => return None,
        };
        return (i == last && end == plan.end).then_some(key);
    }
    None
}

/// Apply `a` to `state` as the enumerator would; `Some((key, end))` when
/// `a` ends the plan (`state` is then left unspecified).
fn step_key(
    state: &mut CombatState,
    a: &Action,
    hidden_info: HiddenInfoMode,
) -> Option<(PlanKey, PlanEnd)> {
    if *a == Action::EndTurn {
        return Some((PlanKey::EndTurn(tt_hash_v2(state)), PlanEnd::EndTurn));
    }
    let pre = state.clone();
    if apply_detect_stochastic(state, a.clone(), hidden_info) {
        let key = PlanKey::Chance(tt_hash_v2(&pre), action_key(&pre, a));
        return Some((key, PlanEnd::Chance));
    }
    is_terminal(state).then(|| (PlanKey::Terminal(tt_hash_v2(state)), PlanEnd::Terminal))
}

/// `a` (legal in `played`) as the enumerator would play it in `canon`.
///
/// The enumerator plays one representative per play identity
/// (`canonicalize_actions`); a human may have played another copy, after
/// which `canon` and `played` hold the same cards in different hand slots.
/// So card plays map by play identity, and choice selections by the
/// selected options' `(card, upgraded)`; potions and `EndTurn` carry over.
fn canonical_action(played: &CombatState, canon: &CombatState, a: &Action) -> Option<Action> {
    if !legal_actions(played).contains(a) {
        return None;
    }
    let legal = legal_actions(canon);
    if let Some(id) = play_identity(played, a) {
        return canonicalize_actions(canon, legal)
            .into_iter()
            .find(|c| play_identity(canon, c) == Some(id));
    }
    let mapped = match a {
        Action::ResolveChoice { selection } => {
            let (from, to) = (
                played.pending_choice.as_ref()?,
                canon.pending_choice.as_ref()?,
            );
            let mut used = vec![false; to.options.len()];
            let mut sel: smallvec::SmallVec<[u8; 4]> = smallvec::SmallVec::new();
            for &i in selection {
                let o = from.options.get(usize::from(i))?;
                let j = (0..to.options.len()).find(|&j| {
                    !used[j]
                        && to.options[j].card_id == o.card_id
                        && to.options[j].upgraded == o.upgraded
                })?;
                used[j] = true;
                sel.push(u8::try_from(j).ok()?);
            }
            if !legal.contains(&Action::ResolveChoice {
                selection: sel.clone(),
            }) {
                sel.sort_unstable();
            }
            Action::ResolveChoice { selection: sel }
        }
        other => other.clone(),
    };
    legal.contains(&mapped).then_some(mapped)
}

/// Cut a played turn into its plan, from `root` (the hidden-info view):
/// `(key, end, used)`, or why it cannot be cut.
fn cut_played(
    root: &CombatState,
    actions: &[Action],
    hidden_info: HiddenInfoMode,
) -> Result<(PlanKey, PlanEnd, usize), &'static str> {
    let mut played = root.clone();
    let mut canon = root.clone();
    for (i, a) in actions.iter().enumerate() {
        let c = canonical_action(&played, &canon, a).ok_or("illegal")?;
        if let Some((key, end)) = step_key(&mut canon, &c, hidden_info) {
            return Ok((key, end, i + 1));
        }
        let _ = sts2_core::apply(&mut played, a.clone());
    }
    Err("incomplete")
}

/// Locate a played turn in `set` (the [`enumerate_plans`] output for
/// `state` under `hidden_info`).
///
/// `actions` are the turn's actions as played on `state` (hand indices
/// valid against the state each applies to). They are cut into a plan
/// exactly as the enumerator cuts one, on the same hidden-info root view,
/// with each card play replaced by the enumerator's representative of its
/// play identity. The played plan matches the enumerated plan reaching
/// the same `tt_hash_v2` state (for `Chance`: the same pre-action state and
/// action identity), so a reordering of the same cards that the enumerator
/// deduped away still matches.
#[must_use]
pub fn locate_plan(
    state: &CombatState,
    hidden_info: HiddenInfoMode,
    set: &PlanSet,
    actions: &[Action],
) -> PlanLocation {
    let mut root = state.clone();
    apply_hidden_info_root(&mut root, hidden_info, TURN_PLAN_ROOT_SEED);
    let (key, end, used) = match cut_played(&root, actions, hidden_info) {
        Ok(cut) => cut,
        Err(failure) => {
            return PlanLocation {
                used: 0,
                end: None,
                index: None,
                failure: Some(failure),
            }
        }
    };
    let index = set
        .plans
        .iter()
        .position(|p| p.end == end && plan_key(&root, p, hidden_info).as_ref() == Some(&key));
    PlanLocation {
        used,
        end: Some(end),
        index,
        failure: None,
    }
}
