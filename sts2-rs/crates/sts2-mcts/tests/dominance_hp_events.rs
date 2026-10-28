//! Dominance pruning must switch off where lower enemy HP is not provably
//! better (GH #846). `dominance::enemy_hp_not_monotone` is the only guard;
//! these tests pin the entries added for CorpseSlug's Ravenous and the
//! ally-count summoners.
//!
//! The fixture is the `strike_bash_ordering` hand: Strike (6, 1e) and Bash
//! (8 + 2 Vulnerable, 2e) against one 40-HP enemy, so neither order kills.
//! Bash first deals 8 + 9 = 17 and Strike first 6 + 8 = 14, which is a
//! dominated pair: after Strike, the follow-up Bash is pruned.
//!
//! `build_handful_ex` builds its enemy as a Nibbit, which is itself listed
//! (`IsAlone` AI), so it would flag every test here before the entry under
//! test was reached. The fixture re-keys it to FuzzyWurmCrawler, which is
//! not listed, and every test first asserts the guard is off, so none of
//! them can pass vacuously.

mod common;

use common::build_handful_ex;
use sts2_core::cards::{CardId, TargetType};
use sts2_core::{
    apply, card_effects, Action, CombatState, CreatureUid, GameVersion, Intent, MonsterId,
    MonsterKind, PowerKind, StepResult,
};
use sts2_mcts::dominance::{enemy_hp_not_monotone, pruned_followups};
use sts2_mcts::EdgeAction;

/// Strike, the first card in the fixture's hand, at the 40-HP enemy.
const STRIKE: Action = Action::PlayCard {
    hand_index: 0,
    target: Some(0),
};

fn fixture(ascension: u8, version: GameVersion) -> CombatState {
    let hand = [
        (CardId::StrikeIronclad, 6, 0, 1, TargetType::Enemy),
        (CardId::Bash, 8, 0, 2, TargetType::Enemy),
    ];
    let mut s = build_handful_ex(
        0,
        &hand,
        /*cultist_hp=*/ 40,
        /*player_hp=*/ 60,
        Intent::Attack { damage: 5 },
        &[],
        /*energy=*/ 3,
    );
    let e = s.enemy_uids[0].0;
    let c = s.creatures.iter_mut().find(|c| c.id == e).unwrap();
    c.monster_id = MonsterId("FuzzyWurmCrawler");
    c.species = Some(MonsterKind::FuzzyWurmCrawler);
    s.ascension = ascension;
    s.game_version = version;
    assert!(
        !enemy_hp_not_monotone(&s),
        "fixture precondition: a lone FuzzyWurmCrawler must not disable pruning"
    );
    s
}

/// The follow-ups `pruned_followups` drops after Strike.
fn pruned_after_strike(p: &CombatState) -> Vec<EdgeAction> {
    let mut child = p.clone();
    assert!(matches!(apply(&mut child, STRIKE), StepResult::Advanced));
    pruned_followups(&[(p, STRIKE)], &child)
}

fn is_bash_at_first_enemy(pruned: &[EdgeAction]) -> bool {
    matches!(
        pruned,
        [EdgeAction::PlayCard {
            card_id: CardId::Bash,
            target: Some(0),
            ..
        }]
    )
}

fn power_amount(s: &CombatState, uid: CreatureUid, kind: PowerKind) -> Option<i32> {
    let c = s.creatures.iter().find(|c| c.id == uid.0)?;
    c.powers
        .iter()
        .filter_map(|p| s.powers.get_by_id(p.0))
        .find(|p| p.kind == kind)
        .map(|p| p.amount)
}

/// Remove every `kind` instance from the arena and from each creature.
fn strip(s: &mut CombatState, kind: PowerKind) {
    let gone: Vec<_> = s
        .powers
        .as_slice()
        .iter()
        .filter(|p| p.kind == kind)
        .map(|p| p.id)
        .collect();
    s.powers.retain(|p| p.kind != kind);
    for c in &mut s.creatures {
        c.powers.retain(|p| !gone.contains(&p.0));
    }
}

/// Control: the fixture really reaches a dominated pair, so an empty pruned
/// set in the other tests comes from the guard and not from the fixture.
#[test]
fn control_bash_moved_first_is_pruned() {
    let p = fixture(0, GameVersion::V0_107_1);
    let pruned = pruned_after_strike(&p);
    assert!(
        is_bash_at_first_enemy(&pruned),
        "Bash after Strike is dominated by Bash before Strike: {pruned:?}"
    );
}

/// Control: an extra enemy that is not listed leaves pruning on, so the
/// summoner tests below measure the entry and not the second enemy.
#[test]
fn unlisted_extra_enemy_keeps_pruning() {
    let mut p = fixture(0, GameVersion::V0_107_1);
    p.spawn_enemy(MonsterKind::FuzzyWurmCrawler);
    assert!(!enemy_hp_not_monotone(&p));
    let pruned = pruned_after_strike(&p);
    assert!(is_bash_at_first_enemy(&pruned), "{pruned:?}");
}

/// `RavenousPower.AfterDeath` (`Powers/RavenousPower.cs:24-36`): when an
/// ally dies, each surviving CorpseSlug is stunned and gains `Amount`
/// Strength for the rest of the fight. A line that leaves a slug lower can
/// kill it sooner, which hands the other slugs that Strength sooner.
/// `CorpseSlug.cs:51,88` applies it on spawn at 4, or 5 from A9
/// (DeadlyEnemies); the A10 rows pin that amount too.
#[test]
fn ravenous_slug_disables_pruning() {
    for (ascension, version, amount) in [
        (0, GameVersion::V0_107_1, 4),
        (10, GameVersion::V0_107_1, 5),
        (0, GameVersion::V0_111_0, 4),
        (10, GameVersion::V0_111_0, 5),
    ] {
        let mut p = fixture(ascension, version);
        let slug = p.spawn_enemy(MonsterKind::CorpseSlug);
        assert_eq!(
            power_amount(&p, slug, PowerKind::Ravenous),
            Some(amount),
            "A{ascension} {version:?}: CorpseSlug spawns with Ravenous {amount}"
        );
        assert!(
            enemy_hp_not_monotone(&p),
            "A{ascension} {version:?}: a Ravenous slug must disable pruning"
        );
        assert!(pruned_after_strike(&p).is_empty());

        // The power is the cause: without it the same two enemies prune.
        strip(&mut p, PowerKind::Ravenous);
        assert!(!enemy_hp_not_monotone(&p));
        assert!(is_bash_at_first_enemy(&pruned_after_strike(&p)));
    }
}

/// The entry is by power kind, not by the slug's monster id.
#[test]
fn ravenous_power_alone_is_flagged() {
    let mut p = fixture(0, GameVersion::V0_107_1);
    let enemy = p.enemy_uids[0];
    card_effects::apply_power_to_creature(&mut p, enemy, PowerKind::Ravenous, 4);
    assert_eq!(power_amount(&p, enemy, PowerKind::Ravenous), Some(4));
    assert!(enemy_hp_not_monotone(&p));
    assert!(pruned_after_strike(&p).is_empty());
}

/// Summoners whose move choice counts living allies or free slots: an ally
/// killed sooner changes what they do next. WaterfallGiant is listed by
/// monster because its SteamEruption comes from its opening move, not from
/// the spawn (see `waterfall_giant_flagged_before_pressurize`).
#[test]
fn ally_count_summoners_disable_pruning() {
    let mut unguarded = Vec::new();
    for (kind, why) in [
        (
            MonsterKind::Fabricator,
            "Fabricator.cs:51 CanFabricate = alive teammates < 4",
        ),
        (
            MonsterKind::Ovicopter,
            "Ovicopter.cs:46 CanLay = alive teammates <= 3",
        ),
        (
            MonsterKind::TwoTailedRat,
            "TwoTailedRat.cs:194 CanSummon = a free slot",
        ),
        (
            MonsterKind::WaterfallGiant,
            "WaterfallGiant.cs:221-228 SteamEruption from PRESSURIZE",
        ),
    ] {
        let mut p = fixture(0, GameVersion::V0_107_1);
        p.spawn_enemy(kind);
        if enemy_hp_not_monotone(&p) {
            assert!(pruned_after_strike(&p).is_empty(), "{kind:?}");
        } else {
            unguarded.push(format!("{kind:?} ({why})"));
        }
    }
    assert!(
        unguarded.is_empty(),
        "these must disable pruning: {unguarded:#?}"
    );
}

/// During player turn 1, before PRESSURIZE, the giant carries no
/// SteamEruption, so the power entry alone leaves that turn unguarded.
#[test]
fn waterfall_giant_flagged_before_pressurize() {
    let mut p = fixture(0, GameVersion::V0_107_1);
    let giant = p.spawn_enemy(MonsterKind::WaterfallGiant);
    assert!(
        !p.powers
            .as_slice()
            .iter()
            .any(|x| x.kind == PowerKind::SteamEruption),
        "WaterfallGiant gains SteamEruption from PRESSURIZE, not on spawn"
    );
    assert_eq!(power_amount(&p, giant, PowerKind::SteamEruption), None);
    assert!(enemy_hp_not_monotone(&p));
}
