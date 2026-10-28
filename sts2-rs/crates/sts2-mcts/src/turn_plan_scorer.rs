//! Pro-trained turn-plan scorer and the root filter built on it (GH #830,
//! `docs/superpowers/specs/2026-10-22-turnplan-pro-root-filter-screen-design.md`).
//!
//! [`PlanScorer`] is the forward pass of `ppo.turnplan.train.PlanScorer`
//! (no-deck variant), loaded from the weights `ppo.turnplan.pro_filter`
//! exports (`root_filter.json`, format [`FORMAT`]):
//!
//! ```text
//! ctx_n  = (ctx - ctx_mean) / ctx_std                      [6]
//! plan_n = (features - plan_mean) / plan_std               [18]
//! plan   = l2(relu(l1(plan_n))) + mean(card_emb[cards])    [64]
//! score  = s2(relu(s1([ctx_n, plan])))                     scalar
//! ```
//!
//! An unknown card name maps to id 0 (UNK), and a plan that plays no card
//! adds the zero vector, as `EmbeddingBag(mode="mean")` does.
//!
//! [`root_filter`] keeps the top-K plans of the current turn by score plus
//! every anchor plan, and returns the root actions those plans start with.

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;
use sts2_core::{legal_actions, Action, CombatState};

use crate::turn_plan::{enumerate_plans, PlanCap};
use crate::turn_plan_features::{anchors, deck_context, plan_cards, plan_features, PLAN_DIM};
use crate::{canonicalize_actions, play_identity, HiddenInfoMode};

/// The weight file's `format` tag.
pub const FORMAT: &str = "turnplan.root_filter.v1";
const CTX_DIM: usize = 6;

#[derive(Deserialize)]
struct LinearFile {
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
}

#[derive(Deserialize)]
struct ScorerFile {
    format: String,
    ctx_mean: Vec<f32>,
    ctx_std: Vec<f32>,
    plan_mean: Vec<f32>,
    plan_std: Vec<f32>,
    plan_l1: LinearFile,
    plan_l2: LinearFile,
    card_vocab: HashMap<String, usize>,
    card_emb: Vec<Vec<f32>>,
    score_l1: LinearFile,
    score_l2: LinearFile,
}

/// A dense layer `y = W x + b`, `W` row-major `[out][in]`.
#[derive(Clone, Debug)]
struct Linear {
    w: Vec<f32>,
    b: Vec<f32>,
    d_in: usize,
}

impl Linear {
    fn from_file(f: LinearFile, d_in: usize, what: &str) -> Result<Self, String> {
        if f.w.len() != f.b.len() || f.w.iter().any(|r| r.len() != d_in) {
            return Err(format!("{what}: expected [{}][{d_in}] weights", f.b.len()));
        }
        Ok(Self {
            w: f.w.into_iter().flatten().collect(),
            b: f.b,
            d_in,
        })
    }

    fn d_out(&self) -> usize {
        self.b.len()
    }

    fn apply(&self, x: &[f32], relu: bool) -> Vec<f32> {
        (0..self.d_out())
            .map(|o| {
                let row = &self.w[o * self.d_in..(o + 1) * self.d_in];
                let y = row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>() + self.b[o];
                if relu {
                    y.max(0.0)
                } else {
                    y
                }
            })
            .collect()
    }
}

/// The no-deck plan scorer; see the module docs.
#[derive(Clone, Debug)]
pub struct PlanScorer {
    ctx_mean: [f32; CTX_DIM],
    ctx_std: [f32; CTX_DIM],
    plan_mean: [f32; PLAN_DIM],
    plan_std: [f32; PLAN_DIM],
    plan_l1: Linear,
    plan_l2: Linear,
    card_vocab: HashMap<String, usize>,
    card_emb: Vec<Vec<f32>>,
    score_l1: Linear,
    score_l2: Linear,
}

fn arr<const N: usize>(v: Vec<f32>, what: &str) -> Result<[f32; N], String> {
    v.try_into()
        .map_err(|v: Vec<f32>| format!("{what}: expected {N} values, got {}", v.len()))
}

impl PlanScorer {
    /// Parse the exported weight file.
    ///
    /// # Errors
    /// On malformed JSON, a wrong `format`, or inconsistent shapes.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let f: ScorerFile = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if f.format != FORMAT {
            return Err(format!("format {:?}, expected {FORMAT:?}", f.format));
        }
        let plan_l1 = Linear::from_file(f.plan_l1, PLAN_DIM, "plan_l1")?;
        let plan_l2 = Linear::from_file(f.plan_l2, plan_l1.d_out(), "plan_l2")?;
        let emb_dim = plan_l2.d_out();
        if f.card_emb.is_empty() || f.card_emb.iter().any(|r| r.len() != emb_dim) {
            return Err(format!("card_emb: expected rows of {emb_dim}"));
        }
        if f.card_vocab.values().any(|&i| i >= f.card_emb.len()) {
            return Err("card_vocab: id outside card_emb".into());
        }
        let score_l1 = Linear::from_file(f.score_l1, CTX_DIM + emb_dim, "score_l1")?;
        let score_l2 = Linear::from_file(f.score_l2, score_l1.d_out(), "score_l2")?;
        if score_l2.d_out() != 1 {
            return Err("score_l2: expected one output".into());
        }
        Ok(Self {
            ctx_mean: arr(f.ctx_mean, "ctx_mean")?,
            ctx_std: arr(f.ctx_std, "ctx_std")?,
            plan_mean: arr(f.plan_mean, "plan_mean")?,
            plan_std: arr(f.plan_std, "plan_std")?,
            plan_l1,
            plan_l2,
            card_vocab: f.card_vocab,
            card_emb: f.card_emb,
            score_l1,
            score_l2,
        })
    }

    /// The score of one plan with features `features` playing `cards`, from
    /// a position with context `ctx` ([`deck_context`]`.ctx`).
    #[must_use]
    pub fn score(&self, ctx: &[f32; CTX_DIM], features: &[f32; PLAN_DIM], cards: &[String]) -> f32 {
        let plan_n: Vec<f32> = (0..PLAN_DIM)
            .map(|i| (features[i] - self.plan_mean[i]) / self.plan_std[i])
            .collect();
        let mut plan = self
            .plan_l2
            .apply(&self.plan_l1.apply(&plan_n, true), false);
        if !cards.is_empty() {
            #[allow(clippy::cast_precision_loss)]
            let n = cards.len() as f32;
            let mut bag = vec![0.0f32; plan.len()];
            for c in cards {
                let row = &self.card_emb[self.card_vocab.get(c).copied().unwrap_or(0)];
                for (b, v) in bag.iter_mut().zip(row) {
                    *b += v;
                }
            }
            for (p, b) in plan.iter_mut().zip(bag) {
                *p += b / n;
            }
        }
        let mut x: Vec<f32> = (0..CTX_DIM)
            .map(|i| (ctx[i] - self.ctx_mean[i]) / self.ctx_std[i])
            .collect();
        x.extend(plan);
        self.score_l2.apply(&self.score_l1.apply(&x, true), false)[0]
    }
}

/// Why [`root_filter`] left the root unfiltered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterFallback {
    /// At most one canonical root action: nothing to filter.
    Forced,
    /// The plan enumeration hit its cap; the plan list is partial.
    Cap,
    /// No surviving plan starts with a legal root action.
    Empty,
}

impl FilterFallback {
    /// The bench record spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Forced => "forced",
            Self::Cap => "cap",
            Self::Empty => "empty",
        }
    }
}

/// What [`root_filter`] decided for one root.
#[derive(Clone, Debug, PartialEq)]
pub struct RootFilter {
    /// Canonical legal root actions (identical hand copies counted once).
    pub legal_n: usize,
    /// Enumerated plans.
    pub plans_n: usize,
    /// Plans kept (top-K ∪ anchors); 0 on a fallback.
    pub kept_n: usize,
    /// The allowed root actions (legal actions of `state`, every hand copy
    /// of an allowed card play included), each with whether a top-K plan
    /// starts with it (`false`: only anchor plans do). Empty on a fallback.
    pub allowed: Vec<(Action, bool)>,
    /// Set when the root is left unfiltered.
    pub fallback: Option<FilterFallback>,
}

/// An action's identity across hand copies: the play identity for a card
/// play, the action itself otherwise.
fn key(state: &CombatState, a: &Action) -> String {
    play_identity(state, a).map_or_else(|| format!("{a:?}"), |id| format!("{id:?}"))
}

/// The root filter at `state` (the player's decision): enumerate the turn's
/// plans (BeliefDraw view, `cap`), keep the `k` best by `scorer` (ties: the
/// lower plan index) plus every anchor plan, and allow the root actions the
/// kept plans start with. Plans are enumerated on the belief view and
/// scored on `state` exactly as the scorer's training labels were
/// (`MctsCombatAgent.turn_plan_labels`); no hidden draw order is read.
#[must_use]
pub fn root_filter(state: &CombatState, scorer: &PlanScorer, k: usize, cap: PlanCap) -> RootFilter {
    let legal = legal_actions(state);
    let legal_n = canonicalize_actions(state, legal.clone()).len();
    let unfiltered = |plans_n, fallback| RootFilter {
        legal_n,
        plans_n,
        kept_n: 0,
        allowed: Vec::new(),
        fallback: Some(fallback),
    };
    if legal_n <= 1 {
        return unfiltered(0, FilterFallback::Forced);
    }
    let set = enumerate_plans(state, HiddenInfoMode::BeliefDraw, cap);
    if set.cap_hit {
        return unfiltered(set.plans.len(), FilterFallback::Cap);
    }
    let ctx = deck_context(state).ctx;
    let scores: Vec<f32> = set
        .plans
        .iter()
        .map(|p| scorer.score(&ctx, &plan_features(state, p), &plan_cards(state, p)))
        .collect();
    let mut order: Vec<usize> = (0..set.plans.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let top: Vec<usize> = order.into_iter().take(k).collect();
    let flags = anchors(state, &set);
    // First-action key -> started by a top-K plan?
    let mut first: BTreeMap<String, bool> = BTreeMap::new();
    let mut kept_n = 0;
    for (i, plan) in set.plans.iter().enumerate() {
        let in_top = top.contains(&i);
        let f = flags[i];
        if !(in_top || f.lethal || f.max_block || f.max_damage || f.heuristic || f.rl2) {
            continue;
        }
        kept_n += 1;
        let via = first.entry(key(state, &plan.actions[0])).or_insert(false);
        *via |= in_top;
    }
    let allowed: Vec<(Action, bool)> = legal
        .iter()
        .filter_map(|a| first.get(&key(state, a)).map(|&via| (a.clone(), via)))
        .collect();
    if allowed.is_empty() {
        return unfiltered(set.plans.len(), FilterFallback::Empty);
    }
    RootFilter {
        legal_n,
        plans_n: set.plans.len(),
        kept_n,
        allowed,
        fallback: None,
    }
}
