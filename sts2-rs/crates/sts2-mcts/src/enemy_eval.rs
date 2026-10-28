//! Two enemy facts the search's scoring reads, each in a base and a faithful
//! reading (GH #830 boss audit).
//!
//! - [`IncomingDamage`]: what an enemy's telegraphed attack is worth. The
//!   base reading is the move table's number (`Intent::damage × hits`); the
//!   effective one is the number the game paints on the intent
//!   ([`sts2_core::combat_loop::effective_intent_damage`]). A Kaiser Crab
//!   Laser from behind is 61 on screen and 35 in the base reading.
//! - [`enemy_hp_remaining`]: how much enemy HP stands between the player and
//!   the win. The base reading sums each creature's HP and max HP
//!   ([`sts2_core::Creature::scored_hp`]). The form-aware one also counts the
//!   forms an enemy still has to go through, so killing Test Subject's first
//!   form no longer reads as losing all progress when it revives at 212.
//!
//! Both base readings are what the search used before either existed, and
//! every caller selects one explicitly, so a search with the knobs off is
//! byte-identical to the engine before this module.

use sts2_core::{CombatState, Creature, MonsterKind, PowerKind, PowerKindSet};

/// How a scorer reads an enemy's telegraphed attack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum IncomingDamage {
    /// The move table's base number: `Intent::damage × hits`.
    #[default]
    Base,
    /// The number the game displays, per hit, times the hits
    /// ([`sts2_core::combat_loop::effective_intent_damage`]: the dealer's
    /// Strength / Weak, the player's Vulnerable / Intangible, a back
    /// attack's ×1.5, …).
    Effective,
}

impl IncomingDamage {
    /// [`Self::Effective`] when `effective`, else [`Self::Base`].
    #[must_use]
    pub const fn from_effective(effective: bool) -> Self {
        if effective {
            Self::Effective
        } else {
            Self::Base
        }
    }

    /// The total damage `creature`'s intent telegraphs, over all its hits;
    /// 0 for a creature telegraphing no attack. Says nothing about whether
    /// the creature is alive: callers filter as they always have.
    #[must_use]
    pub fn total(self, state: &CombatState, creature: &Creature) -> i32 {
        match self {
            Self::Base => creature
                .intent
                .map_or(0, |i| i.damage().saturating_mul(i32::from(i.hits()))),
            Self::Effective => sts2_core::combat_loop::effective_intent_of(state, creature)
                .map_or(0, |(per_hit, hits)| per_hit.saturating_mul(i32::from(hits))),
        }
    }
}

/// Powers whose carrier has forms still to come (or none that count). When
/// none is live the form-aware reading equals the base one.
const FORM_POWERS: PowerKindSet = PowerKindSet::from_kinds(&[
    PowerKind::Adaptable,
    PowerKind::Nemesis,
    PowerKind::Reattach,
    PowerKind::Illusion,
    PowerKind::Stock,
    PowerKind::Infested,
    PowerKind::Surprise,
]);

/// Enemy HP still to remove and the fight's enemy HP total, `(hp, max)`,
/// summed over every creature in `state.enemy_uids` (minions included,
/// corpses included at 0 HP of their max).
///
/// `forms = false` is the base reading: each creature's
/// [`Creature::scored_hp`] (a beaten Waterfall Giant sentinel reads as 0 of
/// its spawn max HP), floored at 0 — exactly the sums the search scored
/// with before this function.
///
/// `forms = true` counts the forms an enemy family still has to go through,
/// in both `hp` and `max`, so neither jumps when a form dies and the next one
/// arrives. Per family (C# `v0.107.1`):
///
/// - **Test Subject** (`AdaptablePower`, then `NemesisPower`;
///   `TestSubject.cs`): three forms of 111 / 212 / 313 HP at A8+ (100 / 200 /
///   300 below; `GetValueIfAscension(ToughEnemies, …)`). Killing a form keeps
///   the combat open and `RESPAWN_MOVE` revives the next one a turn later.
///   The form is the Adaptable power's respawn counter + 1 (corpse
///   included), and 3 once Adaptable is gone. `max` is the sum of all three
///   forms in every form, and `hp` is the current form's HP plus every later
///   form's: form 1 at 40 HP reads `(40 + 212 + 313, 111 + 212 + 313)`, its
///   corpse `(0 + 525, 636)`. The engine revives form 2 / 3 at the bare 200 /
///   300 (a known A8 gap), so a revive here reads as a 12 / 13 HP gain, never
///   a reset.
/// - **Decimillipede segments** (`ReattachPower`): a dead segment heals the
///   Reattach amount (25) at its REATTACH move while any other segment
///   lives, and killing every segment at once ends the fight. So a dead
///   segment with a living sibling counts `min(25, max)` HP still to remove;
///   its `max` is unchanged.
/// - **Illusions** (`IllusionPower`: Parafright, Eye With Teeth): revive to
///   full HP as long as a primary enemy lives, and die with the last one
///   (`Creature.IsSecondaryEnemy`, `CombatManager` win check). Their HP is
///   never progress toward the win, so they count `(0, 0)`.
/// - **Axebot** (`StockPower`, amount n): each death spawns a fresh Axebot
///   with Stock n − 1, so a carrier adds n spawns of the species' mean
///   starting HP to both sums. The corpse keeps its own `max`, and the
///   spawn carries the rest, so `max` stays put across a respawn up to the
///   rolled-vs-mean difference.
/// - **Phrog Parasite** (`InfestedPower`): its death adds 4 Wrigglers
///   (`InfestedPower.AfterDeath`, a fixed 4), so an armed carrier adds 4
///   mean Wriggler HPs.
/// - **Gremlin Merc** (`SurprisePower`): its death adds a Sneaky Gremlin
///   and a Fat Gremlin, so an armed carrier adds their mean HPs.
/// - **Waterfall Giant** (`SteamEruptionPower`): one form. Its death
///   revives it at the 999,999,999 sentinel only to explode; it cannot be
///   damaged toward the win any further, so the sentinel counts as removed
///   (the base reading already does this).
///
/// Not forms, so counted as the board stands: open-ended summoners
/// (Fabricator, Fogmog, Living Fog, Ovicopter, The Obscura, Two-Tailed Rat;
/// how many more they add depends on the line played), the Tough Egg's
/// hatch (killing the egg first prevents it), and Kaiser Crab's Crab Rage
/// block (it expires at the survivor's next turn).
#[must_use]
pub fn enemy_hp_remaining(state: &CombatState, forms: bool) -> (i32, i32) {
    let in_enemies = |c: &&Creature| state.enemy_uids.iter().any(|u| u.0 == c.id);
    let forms = forms && state.powers.present_kinds().intersects(&FORM_POWERS);
    let living_segments = if forms {
        state
            .creatures
            .iter()
            .filter(in_enemies)
            .filter(|c| c.current_hp > 0 && has_power(state, c, PowerKind::Reattach))
            .count()
    } else {
        0
    };
    let (mut hp, mut max) = (0i32, 0i32);
    for c in state.creatures.iter().filter(in_enemies) {
        let (h, m) = c.scored_hp(state.ascension);
        let (h, m) = if forms {
            with_forms(state, c, h.max(0), m.max(0), living_segments)
        } else {
            (h.max(0), m.max(0))
        };
        hp += h;
        max += m;
    }
    (hp, max)
}

fn has_power(state: &CombatState, c: &Creature, kind: PowerKind) -> bool {
    c.powers
        .iter()
        .any(|pu| state.powers.get_by_id(pu.0).is_some_and(|p| p.kind == kind))
}

/// One creature's `(hp, max)` under [`enemy_hp_remaining`]'s form rules,
/// from its base reading `(hp, max)`.
fn with_forms(
    state: &CombatState,
    c: &Creature,
    hp: i32,
    max: i32,
    living_segments: usize,
) -> (i32, i32) {
    let asc = state.ascension;
    let (mut adaptable, mut reattach, mut illusion) = (None, 0, false);
    let (mut stock, mut infested, mut surprise) = (0, 0, 0);
    for p in c
        .powers
        .iter()
        .filter_map(|pu| state.powers.get_by_id(pu.0))
    {
        match p.kind {
            PowerKind::Adaptable => adaptable = Some(p.counter),
            PowerKind::Reattach => reattach = p.amount,
            PowerKind::Illusion => illusion = true,
            PowerKind::Stock => stock = p.amount,
            PowerKind::Infested => infested = p.amount,
            PowerKind::Surprise => surprise = p.amount,
            _ => {}
        }
    }
    if illusion {
        return (0, 0);
    }
    if c.species == Some(MonsterKind::TestSubject) {
        use sts2_core::monsters::test_subject;
        let form_hp = [
            MonsterKind::TestSubject.min_initial_hp(asc),
            test_subject::second_form_hp(asc),
            test_subject::third_form_hp(asc),
        ];
        let form = adaptable.map_or(3, |respawns| respawns.saturating_add(1).clamp(1, 3));
        let total: i32 = form_hp.iter().sum();
        let later: i32 = form_hp[usize::try_from(form).unwrap_or(3)..].iter().sum();
        return ((hp + later).min(total), total);
    }
    // A dead segment counts as removed only when no sibling lives to let it
    // reattach (`ReattachPower.DoReattach`); `c` is dead, so every living
    // carrier is another segment.
    if reattach > 0 && c.current_hp <= 0 && living_segments > 0 {
        return (reattach.min(max), max);
    }
    let mean = |k: MonsterKind| k.mean_initial_hp(asc);
    let mut later = 0i32;
    if stock > 0 {
        later += stock * mean(MonsterKind::Axebot);
    }
    if infested > 0 {
        later += 4 * mean(MonsterKind::Wriggler);
    }
    if surprise > 0 {
        later += mean(MonsterKind::SneakyGremlin) + mean(MonsterKind::FatGremlin);
    }
    (hp + later, max + later)
}
