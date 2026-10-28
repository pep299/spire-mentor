//! In-fight potion-play degradation seams
//! (`STS2_RUNAI_POTION_TIMING`), the last untested potion axis.
//!
//! Every potion experiment so far priced potions ACROSS fights (keep vs
//! spend, swap, capacity, volume). None touched play WITHIN a fight:
//! which turn to drink on, and which potion to take off the belt. The
//! only evidence there is indirect — 55–66% of drink plies sit inside
//! the search's tie band (top-gap ≤ 0.02), i.e. the search is nearly
//! indifferent — and indifference is a measurement of the SEARCH, not
//! proof the choice is worthless.
//!
//! Both modes are DEGRADATIONS, not oracles: they answer "how much value
//! does current in-fight play extract?", which bounds from below what a
//! better in-fight policy could be worth. If randomizing a dimension is
//! free, the search is currently extracting nothing from it.
//!
//! - `defer:<p>` — when the search picks a drink, with probability `p`
//!   take the best NON-drink root action instead. The potion stays in
//!   the belt and can be drunk on a later ply, so this perturbs *timing*
//!   while (mostly) preserving volume.
//! - `random-slot` — when the search picks a drink, drink a uniformly
//!   random OTHER legal potion instead. Volume and timing are identical;
//!   only *which potion* changes. Isolates selection exactly.
//!
//! The coin is per (run seed, floor, decision counter) off `splitmix64`,
//! so no engine RNG stream is touched and an arm is reproducible.

use sts2_core::Action;
use sts2_mcts::RootOutcome;

/// Resolved `STS2_RUNAI_POTION_TIMING` mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PotionTiming {
    /// Defer a chosen drink to the best non-drink action with prob `p`.
    Defer(f64),
    /// Swap a chosen drink's slot for a random other legal drink.
    RandomSlot,
}

impl PotionTiming {
    /// Pure resolver. Unset ⇒ `None` ⇒ deployed behavior untouched;
    /// malformed ⇒ panic (measurement seam).
    pub(crate) fn resolve(env: Option<&str>) -> Option<Self> {
        let raw = env?.trim();
        if raw == "random-slot" {
            return Some(Self::RandomSlot);
        }
        if let Some(p) = raw.strip_prefix("defer:") {
            let p = p
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
                .unwrap_or_else(|| {
                    panic!(
                        "STS2_RUNAI_POTION_TIMING: defer probability must be in [0,1], got {p:?}"
                    )
                });
            return Some(Self::Defer(p));
        }
        panic!(
            "STS2_RUNAI_POTION_TIMING: unknown mode {raw:?} \
             (expected \"defer:<p>\" or \"random-slot\")"
        )
    }

    /// Rewrite the search's pick, or leave it alone. `chosen` is what the
    /// MCTS returned; `outcomes` are the legal root children.
    pub(crate) fn perturb(
        self,
        chosen: &Action,
        outcomes: &[RootOutcome],
        coin: u64,
    ) -> Option<Action> {
        if !matches!(chosen, Action::DrinkPotion { .. }) {
            return None;
        }
        match self {
            Self::Defer(p) => {
                #[allow(clippy::cast_precision_loss)] // 53-bit mantissa
                let u = (coin >> 11) as f64 / (1u64 << 53) as f64;
                if u >= p {
                    return None;
                }
                // Best non-drink child by the same scalar the engine
                // ranks on (`mean_value`), so the deferral costs the
                // search's own second choice, not a random blunder.
                outcomes
                    .iter()
                    .filter(|o| !matches!(o.action, Action::DrinkPotion { .. }))
                    .filter(|o| o.visits > 0)
                    .max_by(|a, b| {
                        a.mean_value
                            .partial_cmp(&b.mean_value)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|o| o.action.clone())
            }
            Self::RandomSlot => {
                let alts: Vec<&RootOutcome> = outcomes
                    .iter()
                    .filter(|o| matches!(o.action, Action::DrinkPotion { .. }))
                    .filter(|o| o.action != *chosen)
                    .collect();
                if alts.is_empty() {
                    return None;
                }
                Some(alts[(coin % alts.len() as u64) as usize].action.clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drink(slot: u8) -> Action {
        Action::DrinkPotion {
            potion_slot: slot,
            target: None,
        }
    }

    fn outcome(action: Action, mean_value: f32) -> RootOutcome {
        RootOutcome {
            action,
            visits: 10,
            wins: 0,
            losses: 0,
            ties: 0,
            value_sum: mean_value * 10.0,
            value_sq_sum: mean_value * mean_value * 10.0,
            hp_sum: 0.0,
            hp_sq_sum: 0.0,
            hp_n: 0,
            mean_value,
        }
    }

    #[test]
    fn resolver_parses_both_modes_and_rejects_junk() {
        assert_eq!(PotionTiming::resolve(None), None);
        assert_eq!(
            PotionTiming::resolve(Some("random-slot")),
            Some(PotionTiming::RandomSlot)
        );
        assert_eq!(
            PotionTiming::resolve(Some("defer:0.5")),
            Some(PotionTiming::Defer(0.5))
        );
    }

    #[test]
    #[should_panic(expected = "unknown mode")]
    fn resolver_rejects_typos() {
        PotionTiming::resolve(Some("defer 0.5"));
    }

    #[test]
    #[should_panic(expected = "probability must be in")]
    fn resolver_rejects_out_of_range_probability() {
        PotionTiming::resolve(Some("defer:1.5"));
    }

    #[test]
    fn non_drink_picks_are_never_perturbed() {
        let outs = [outcome(Action::EndTurn, 0.5), outcome(drink(0), 0.4)];
        assert_eq!(
            PotionTiming::RandomSlot.perturb(&Action::EndTurn, &outs, 0),
            None
        );
        assert_eq!(
            PotionTiming::Defer(1.0).perturb(&Action::EndTurn, &outs, 0),
            None
        );
    }

    #[test]
    fn defer_takes_the_best_non_drink_child() {
        let outs = [
            outcome(drink(0), 0.9),
            outcome(Action::EndTurn, 0.1),
            outcome(
                Action::PlayCard {
                    hand_index: 2,
                    target: None,
                },
                0.4,
            ),
        ];
        // coin 0 => u = 0 < p, so the deferral fires.
        let got = PotionTiming::Defer(1.0).perturb(&drink(0), &outs, 0);
        assert!(
            matches!(got, Some(Action::PlayCard { hand_index: 2, .. })),
            "expected the best non-drink child, got {got:?}"
        );
        // p = 0 never fires.
        assert_eq!(PotionTiming::Defer(0.0).perturb(&drink(0), &outs, 0), None);
    }

    #[test]
    fn defer_is_a_noop_when_every_child_is_a_drink() {
        let outs = [outcome(drink(0), 0.9), outcome(drink(1), 0.2)];
        assert_eq!(PotionTiming::Defer(1.0).perturb(&drink(0), &outs, 0), None);
    }

    #[test]
    fn random_slot_picks_another_drink_never_the_same_one() {
        let outs = [
            outcome(drink(0), 0.9),
            outcome(drink(1), 0.2),
            outcome(drink(2), 0.1),
            outcome(Action::EndTurn, 0.5),
        ];
        for coin in 0..8u64 {
            let got = PotionTiming::RandomSlot
                .perturb(&drink(0), &outs, coin)
                .expect("an alternative drink exists");
            assert!(matches!(got, Action::DrinkPotion { .. }));
            assert_ne!(got, drink(0), "must not return the search's own pick");
        }
        // Sole drink ⇒ nothing to swap to.
        let one = [outcome(drink(0), 0.9), outcome(Action::EndTurn, 0.5)];
        assert_eq!(PotionTiming::RandomSlot.perturb(&drink(0), &one, 3), None);
    }
}
