//! Combat pile-search / multi-pick / Discover-style choice resolution for
//! the **meta** policy (issue #164).
//!
//! The combat MCTS already searches these choices natively: a card's
//! `OnPlay` opens a `PendingChoice`, `legal_actions` fans it out into
//! single-index `ResolveChoice` actions (sequential decomposition, bead
//! `sts2-rs-sopgf`), and the search expands every branch. But when the
//! **meta** policy plays combat without the MCTS bridge — `--combat none`,
//! or the rare untranslatable-action fallback — it had no machinery for
//! these at all and blindly took option 0 (`heuristic.rs` `Phase::Combat
//! => 0`). First-option is actively bad here: for an exhaust/discard
//! choice it would dump whatever card sits first in hand (possibly your
//! best Attack), and for a Discover it would take an arbitrary generated
//! card.
//!
//! This module gives that path a competent greedy resolver. Each decision
//! is a single pick (the engine re-opens the choice after every
//! incremental selection), so greedy-per-pick is the natural fit — the
//! same shape as the shop knapsack (`surfaces::shop`) and exactly what
//! the issue's step 4 calls for. Scoring is direction-by-`ChoiceKind`:
//! "acquire / keep / draw" kinds want the **best** card, "exhaust /
//! discard / make-ethereal" kinds want the **worst**, and the do-nothing
//! FINALIZE option is the zero baseline — so a skippable removal only
//! fires on a genuinely dead card and never sheds a useful one.
//!
//! Honest-by-construction: any `ChoiceKind` whose desirable direction
//! isn't clearly card-value-shaped (orb slots, enemy targets, monster
//! curse picks, the character-specific transforms) returns `None` and the
//! caller falls back to first-option — never a regression vs. today.

use sts2_core::cards::CardId;
use sts2_core::run_manager::RunAction;
use sts2_core::{ChoiceKind, PendingChoice};

use crate::eval::card_stats::card_stats;
use crate::policy::surfaces::picks::argmax;

/// Value of a curse / status card — a dead or harmful deck slot. Far
/// below any real card so removal kinds always target it first and
/// acquire kinds never reach for it.
const CURSE_VALUE: f64 = -10.0;

/// Greedy combat-tactical value of a single card: higher = better to
/// hold / draw / keep, lower = better to exhaust / discard. Built from
/// the same static [`card_stats`] features the heuristic-V deck terms
/// use, so it stays consistent with the rest of the agent and never
/// invents numbers for unknown cards (they score ~neutral).
fn card_value(card_id: CardId, upgraded: bool) -> f64 {
    let s = card_stats(card_id, upgraded);
    if s.curse_like {
        return CURSE_VALUE;
    }
    let mut v = s.damage_per_play + 0.8 * s.block_per_play + 2.0 * s.draw + 3.0 * s.energy;
    if s.scaling {
        v += 4.0;
    }
    v - 0.5 * s.cost
}

/// Whether a `ChoiceKind` wants the highest-value card (`Max`) or the
/// lowest-value one (`Min`). `None` ⇒ no clear card-value signal; the
/// caller keeps the first-option default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dir {
    Max,
    Min,
}

fn kind_direction(kind: ChoiceKind) -> Option<Dir> {
    use ChoiceKind::{
        AddFromDiscardFree, AddFromDiscardToHand, AddFromDrawByType, AddFromDrawToHand,
        AddGeneratedFreeToHand, AddGeneratedRetainedToHand, AddGeneratedToHand,
        ApplyEtherealInHand, ApplyRetainInHand, ApplySlyInHand, AutoPlaySkillFromHand,
        CloneChosenCardNextTurn, DiscardFromHand, DiscardFromHandAndDraw, ExhaustFromHand,
        ExhaustPile, ForegoneConclusionFromDraw, MakeFreeInHand, PutBack, RetainAtTurnEnd,
    };
    match kind {
        // Acquire a card / keep a card / set up a card to draw next:
        // always want the strongest option.
        PutBack
        | AddGeneratedToHand
        | AddGeneratedRetainedToHand
        | AddGeneratedFreeToHand
        | AddFromDrawByType
        | ChoiceKind::AddFromRandomDrawSubset
        | AddFromDiscardFree
        | AddFromDiscardToHand
        | AddFromDrawToHand
        | ForegoneConclusionFromDraw
        | MakeFreeInHand
        | ApplyRetainInHand
        | ApplySlyInHand
        | AutoPlaySkillFromHand
        | CloneChosenCardNextTurn
        | RetainAtTurnEnd => Some(Dir::Max),
        // Shed a card / saddle a card with a downside keyword: target the
        // weakest option.
        ExhaustFromHand
        | ExhaustPile
        | DiscardFromHand
        | DiscardFromHandAndDraw
        | ApplyEtherealInHand => Some(Dir::Min),
        // No clear card-value direction — handled by `option_score`'s
        // explicit arms (UpgradeInHand) or left to first-option.
        _ => None,
    }
}

/// Desirability of *selecting* a given option under `kind` — higher means
/// the policy wants it more. `None` ⇒ this kind has no card-value signal,
/// so the whole choice defers to first-option.
fn option_score(kind: ChoiceKind, card_id: CardId, upgraded: bool) -> Option<f64> {
    match kind {
        // Upgrade the card that gains the most from being upgraded, not
        // merely the best card — a base Strike gains more than an
        // already-strong but upgrade-light card.
        ChoiceKind::UpgradeInHand => Some(card_value(card_id, true) - card_value(card_id, false)),
        _ => match kind_direction(kind)? {
            Dir::Max => Some(card_value(card_id, upgraded)),
            Dir::Min => Some(-card_value(card_id, upgraded)),
        },
    }
}

/// Resolve one greedy pick of an open combat choice.
///
/// Returns `(index into `actions`, per-action scores for the audit log)`,
/// or `None` to defer to the caller's first-option default. The returned
/// index always points at one of the supplied (legal) `actions`, so the
/// engine's selection invariants are upheld. The FINALIZE / skip action
/// (`option: None`) scores `0.0` — the do-nothing baseline — so a
/// skippable removal fires only when shedding the worst option is a net
/// positive, and an acquire skips only when every option is a net
/// negative.
#[must_use]
pub fn choose_combat_choice(
    pc: &PendingChoice,
    actions: &[RunAction],
) -> Option<(usize, Vec<f64>)> {
    let mut scores: Vec<f64> = Vec::with_capacity(actions.len());
    for a in actions {
        let s = match a {
            RunAction::CombatResolveChoice { option: Some(i) } => {
                let opt = pc.options.get(*i)?;
                // A kind with no card-value signal bails the whole choice.
                option_score(pc.kind, opt.card_id, opt.upgraded)?
            }
            // FINALIZE / skip: the do-nothing baseline.
            RunAction::CombatResolveChoice { option: None } => 0.0,
            // Anything else in a choice action list (the defensive
            // TakeFirst fallback, a stray PlayCard): don't risk
            // mis-scoring — defer to first-option.
            _ => return None,
        };
        scores.push(s);
    }
    let best = argmax(&scores)?;
    Some((best, scores))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sts2_core::cards::CardId;
    use sts2_core::ChoiceOption;

    fn opt(card_id: CardId, upgraded: bool) -> ChoiceOption {
        ChoiceOption {
            card_id,
            upgraded,
            card_uid: None,
            source_pile: None,
        }
    }

    fn choice(kind: ChoiceKind, options: Vec<ChoiceOption>, can_skip: bool) -> PendingChoice {
        PendingChoice {
            kind,
            options,
            min_select: u8::from(!can_skip),
            max_select: 1,
            can_skip,
            ..Default::default()
        }
    }

    /// One `CombatResolveChoice{Some(i)}` per option, in order; no skip.
    fn pick_actions(n: usize) -> Vec<RunAction> {
        (0..n)
            .map(|i| RunAction::CombatResolveChoice { option: Some(i) })
            .collect()
    }

    /// Discover (`AddGeneratedFreeToHand`) takes the strongest generated
    /// card, not option 0.
    #[test]
    fn discover_takes_best_card() {
        // option 0 = a curse, option 1 = a strong attack.
        let pc = choice(
            ChoiceKind::AddGeneratedFreeToHand,
            vec![opt(CardId::Decay, false), opt(CardId::Bludgeon, false)],
            false,
        );
        let actions = pick_actions(2);
        let (idx, _) = choose_combat_choice(&pc, &actions).expect("scored");
        assert_eq!(idx, 1, "should take the strong attack over the curse");
    }

    /// A skippable exhaust (Purity / Ashwater) dumps a curse but keeps a
    /// real card — and skips entirely when nothing is worth shedding.
    #[test]
    fn exhaust_targets_curse_and_skips_good_cards() {
        // Hand: [Strike, Decay(curse)], plus the FINALIZE/skip action.
        let pc = choice(
            ChoiceKind::ExhaustFromHand,
            vec![
                opt(CardId::StrikeIronclad, false),
                opt(CardId::Decay, false),
            ],
            true,
        );
        let mut actions = pick_actions(2);
        actions.push(RunAction::CombatResolveChoice { option: None }); // FINALIZE
        let (idx, _) = choose_combat_choice(&pc, &actions).expect("scored");
        assert_eq!(idx, 1, "must exhaust the curse, not the Strike");

        // Hand with only good cards → finalize (skip) wins over shedding.
        let pc_good = choice(
            ChoiceKind::ExhaustFromHand,
            vec![
                opt(CardId::StrikeIronclad, false),
                opt(CardId::DefendIronclad, false),
            ],
            true,
        );
        let mut good_actions = pick_actions(2);
        let finalize = good_actions.len();
        good_actions.push(RunAction::CombatResolveChoice { option: None });
        let (idx, _) = choose_combat_choice(&pc_good, &good_actions).expect("scored");
        assert_eq!(
            idx, finalize,
            "no dead card → skip rather than shed a real one"
        );
    }

    /// A forced exhaust (no skip, e.g. Brand) still sheds the worst card.
    #[test]
    fn forced_exhaust_sheds_worst_when_no_skip() {
        let pc = choice(
            ChoiceKind::ExhaustFromHand,
            vec![opt(CardId::Bludgeon, false), opt(CardId::Decay, false)],
            false,
        );
        let actions = pick_actions(2);
        let (idx, _) = choose_combat_choice(&pc, &actions).expect("scored");
        assert_eq!(idx, 1, "forced to exhaust → drop the curse, keep Bludgeon");
    }

    /// Kinds with no card-value direction defer to first-option (`None`).
    #[test]
    fn neutral_kinds_defer_to_first_option() {
        let pc = choice(
            ChoiceKind::MonsterCurseSelect,
            vec![opt(CardId::Decay, false), opt(CardId::Decay, false)],
            false,
        );
        let actions = pick_actions(2);
        assert!(
            choose_combat_choice(&pc, &actions).is_none(),
            "no card-value signal → caller keeps first-option"
        );
    }

    /// Curses score far below any real card in both directions.
    #[test]
    fn card_value_orders_curse_below_real_cards() {
        assert!(card_value(CardId::Decay, false) < card_value(CardId::StrikeIronclad, false));
        assert!(card_value(CardId::StrikeIronclad, false) < card_value(CardId::Bludgeon, false));
    }
}
