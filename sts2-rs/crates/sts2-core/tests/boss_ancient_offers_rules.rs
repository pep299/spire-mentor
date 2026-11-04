//! Boss-relic Ancient-event offer composition — rule pins.
//!
//! Complements the inline unit tests in
//! `crates/sts2-core/src/boss_ancient_offers.rs` (which mostly assert pool
//! *membership*). Everything asserted here is read off the v0.107.1 decomp
//! at `../sts2-wiki/decompiled/v0.107.1/MegaCrit.Sts2.Core.Models.Events/`:
//!
//! * `Tezcatara.cs:188-198` — 3 `NextItem` picks, `NutritiousSoup` appended
//!   to `OptionPool1` iff the deck holds a Basic Strike.
//! * `Orobas.cs:186-213` — other-character `NextItem`, `NextFloat() <
//!   0.3333333` PrismaticGem-vs-SeaGlass, then 3 `NextItem` picks; the
//!   variant relic is **appended last** to `OptionPool1`.
//! * `Pael.cs:150-173` — `list.AddRange(list)` doubles OptionPool2 *before*
//!   `PaelsGrowth` is appended; `>= 3` Goopy / `>= 5` removable thresholds;
//!   `PaelsLegion` iff `!HasEventPet()`.
//! * `Nonupeipe.cs:140-149` / `Tanx.cs:134-142` — one pool, optional 10th
//!   entry (`Swift >= 4` / `Instinct >= 3`), `UnstableShuffle`, take 3.
//! * `Vakuu.cs:150-164` — three pools, shuffle each, take the head.
//! * `Darv.cs:147-203` — one `NextItem` per valid relic set in declaration
//!   order (act-gated + PandorasBox modifier gate), shuffle, then
//!   `NextBool()` picks [take 2 + DustyTome] vs [take 3].
//! * `Rng.cs:256-265` — `NextItem` on an EMPTY set returns `default` and
//!   consumes NO sample; on a 1-element set it still draws `NextInt(0, 1)`.
//!
//! The RNG-consumption counts below are the load-bearing part: the per-event
//! stream is shared across all three picks, so a pool that is one entry too
//! long/short silently shifts every later draw.

use hegel::generators as gs;
use hegel::TestCase;
use sts2_core::boss_ancient_offers::{event_rng, roll_offers, AncientConditions};
use sts2_core::run_room_gen::AncientId;
use sts2_core::RelicId;

const ALL_ANCIENTS: [AncientId; 7] = [
    AncientId::Orobas,
    AncientId::Pael,
    AncientId::Tezcatara,
    AncientId::Nonupeipe,
    AncientId::Tanx,
    AncientId::Vakuu,
    AncientId::Darv,
];

/// Baseline: act 2, no deck conditions met, no run modifiers, headless
/// 5-character unlock state (4 other characters).
fn cond() -> AncientConditions {
    AncientConditions {
        act_index: 1,
        goopy_eligible_count: 0,
        removable_count: 0,
        has_event_pet: false,
        instinct_eligible_count: 0,
        swift_eligible_count: 0,
        num_other_characters: 4,
        orobas_pool3: vec![RelicId::TouchOfOrobas, RelicId::ArchaicTooth],
        pandoras_allowed: true,
        tezcatara_basic_strike: false,
    }
}

/// `AllPossibleOptions` per ancient — the closed universe an offer may come
/// from (Darv's includes `DustyTome`, which is not in any relic set).
fn universe(a: AncientId) -> Vec<RelicId> {
    use RelicId::*;
    match a {
        AncientId::Orobas => vec![
            ElectricShrymp,
            GlassEye,
            SandCastle,
            PrismaticGem,
            SeaGlass,
            AlchemicalCoffer,
            Driftwood,
            RadiantPearl,
            TouchOfOrobas,
            ArchaicTooth,
        ],
        AncientId::Pael => vec![
            PaelsFlesh,
            PaelsHorn,
            PaelsTears,
            PaelsWing,
            PaelsClaw,
            PaelsTooth,
            PaelsGrowth,
            PaelsEye,
            PaelsBlood,
            PaelsLegion,
        ],
        AncientId::Tezcatara => vec![
            VeryHotCocoa,
            YummyCookie,
            NutritiousSoup,
            BiiigHug,
            Storybook,
            ToastyMittens,
            GoldenCompass,
            PumpkinCandle,
            ToyBox,
            SealOfGold,
        ],
        AncientId::Nonupeipe => vec![
            BlessedAntler,
            BrilliantScarf,
            DelicateFrond,
            DiamondDiadem,
            FurCoat,
            Glitter,
            JewelryBox,
            LoomingFruit,
            SignetRing,
            BeautifulBracelet,
        ],
        AncientId::Tanx => vec![
            Claws,
            Crossbow,
            IronClub,
            MeatCleaver,
            Sai,
            SpikedGauntlets,
            TanxsWhistle,
            ThrowingAxe,
            WarHammer,
            TriBoomerang,
        ],
        AncientId::Vakuu => vec![
            BloodSoakedRose,
            WhisperingEarring,
            Fiddle,
            PreservedFog,
            SereTalon,
            DistinguishedCape,
            ChoicesParadox,
            MusicBox,
            LordsParasol,
            JeweledMask,
        ],
        AncientId::Darv => vec![
            Astrolabe,
            BlackStar,
            CallingBell,
            EmptyCage,
            PandorasBox,
            RunicPyramid,
            SneckoEye,
            Ectoplasm,
            Sozu,
            PhilosophersStone,
            VelvetChoker,
            DustyTome,
        ],
        AncientId::Neow => Vec::new(),
    }
}

/// Union of the offers rolled over `seeds` seeds under `c`.
fn seen(a: AncientId, c: &AncientConditions, seeds: i32) -> std::collections::HashSet<RelicId> {
    let mut acc = std::collections::HashSet::new();
    for seed in 0..seeds {
        let mut rng = event_rng(seed, a);
        acc.extend(roll_offers(a, &mut rng, c));
    }
    acc
}

// ── Dispatch ────────────────────────────────────────────────────────────

/// Neow is not a boss-relic ancient: its 2-positives-plus-curse offer is
/// rolled by `neow_offers`, so this surface returns nothing and — critically
/// — must not disturb the event RNG stream.
#[test]
fn neow_yields_no_boss_relic_offers_and_consumes_no_rng() {
    let c = cond();
    let mut rng = event_rng(4242, AncientId::Neow);
    assert!(roll_offers(AncientId::Neow, &mut rng, &c).is_empty());
    assert_eq!(rng.counter(), 0, "Neow dispatch must not draw from the RNG");
}

/// Every ancient's offers come from its own `AllPossibleOptions` universe —
/// no relic leaks across events.
#[test]
fn offers_stay_inside_each_ancients_universe() {
    let mut c = cond();
    c.goopy_eligible_count = 9;
    c.removable_count = 9;
    c.instinct_eligible_count = 9;
    c.swift_eligible_count = 9;
    for a in ALL_ANCIENTS {
        let uni = universe(a);
        for seed in 0..96i32 {
            let mut rng = event_rng(seed, a);
            for r in roll_offers(a, &mut rng, &c) {
                assert!(uni.contains(&r), "{a:?} offered out-of-pool {r:?}");
            }
        }
    }
}

// ── RNG consumption (pool sizes are load-bearing) ───────────────────────

/// Exact per-ancient draw counts from the C# call sequence. Each ancient
/// shares ONE event RNG across all its picks, so an off-by-one pool length
/// changes every subsequent offer — these counts pin the pool sizes even
/// where the membership assertions would still pass.
/// `UnstableShuffle` of `n` draws `n - 1` samples (`ListExtensions.cs:21`).
#[test]
fn rng_draw_counts_match_the_csharp_call_sequence() {
    let base = cond();
    // (ancient, conditions, expected draws)
    let cases: Vec<(AncientId, AncientConditions, u32)> = vec![
        // 3 × NextItem.
        (AncientId::Tezcatara, base.clone(), 3),
        (AncientId::Pael, base.clone(), 3),
        // char NextItem + NextFloat + 3 × NextItem.
        (AncientId::Orobas, base.clone(), 5),
        // …minus the char pick when the other-character pool is empty
        // (`Rng.NextItem` returns default without sampling).
        (
            AncientId::Orobas,
            AncientConditions {
                num_other_characters: 0,
                ..base.clone()
            },
            4,
        ),
        // shuffle(3) + shuffle(3) + shuffle(4) = 2 + 2 + 3.
        (AncientId::Vakuu, base.clone(), 7),
        // shuffle(9); shuffle(10) once TriBoomerang joins.
        (AncientId::Tanx, base.clone(), 8),
        (
            AncientId::Tanx,
            AncientConditions {
                instinct_eligible_count: 3,
                ..base.clone()
            },
            9,
        ),
        // shuffle(9); shuffle(10) once BeautifulBracelet joins.
        (AncientId::Nonupeipe, base.clone(), 8),
        (
            AncientId::Nonupeipe,
            AncientConditions {
                swift_eligible_count: 4,
                ..base.clone()
            },
            9,
        ),
        // 8 sets → 8 × NextItem + shuffle(8)=7 + NextBool.
        (AncientId::Darv, base.clone(), 16),
        // PandorasBox filtered out → 7 sets, shuffle(7)=6, NextBool.
        (
            AncientId::Darv,
            AncientConditions {
                pandoras_allowed: false,
                ..base.clone()
            },
            14,
        ),
        // No act gate matches (act 1 has no Darv, but the filter is
        // `== 1` / `== 2`, so any other index drops both sets).
        (
            AncientId::Darv,
            AncientConditions {
                act_index: 0,
                ..base.clone()
            },
            14,
        ),
    ];
    for (a, c, expected) in cases {
        let mut rng = event_rng(777, a);
        let _ = roll_offers(a, &mut rng, &c);
        assert_eq!(
            rng.counter(),
            expected,
            "{a:?} drew {} samples, expected {expected}",
            rng.counter()
        );
    }
}

// ── Orobas ──────────────────────────────────────────────────────────────

/// Reference re-derivation of `Orobas.GenerateInitialOptions`: the
/// PrismaticGem/SeaGlass variant is chosen by `NextFloat() < 1/3` and
/// **appended last** to `OptionPool1` before the pick, so its index is 3.
#[test]
fn orobas_variant_is_appended_last_and_gated_at_one_third() {
    use RelicId::{ElectricShrymp, GlassEye, PrismaticGem, SandCastle, SeaGlass};
    let c = cond();
    let mut saw_gem = false;
    let mut saw_glass = false;
    for seed in 0..256i32 {
        // Mirror the C# read order on an independent stream.
        let mut r = event_rng(seed, AncientId::Orobas);
        let _char_pick = r.next_int_exclusive(0, 4);
        let variant = if r.next_float(1.0) < 0.333_333_3_f64 {
            PrismaticGem
        } else {
            SeaGlass
        };
        let list = [ElectricShrymp, GlassEye, SandCastle, variant];
        let expected0 = list[r.next_int_exclusive(0, 4) as usize];

        let mut rng = event_rng(seed, AncientId::Orobas);
        let offers = roll_offers(AncientId::Orobas, &mut rng, &c);
        assert_eq!(offers[0], expected0, "seed {seed}");
        saw_gem |= variant == PrismaticGem;
        saw_glass |= variant == SeaGlass;
    }
    assert!(saw_gem && saw_glass, "one side of the 1/3 gate never fired");
}

/// `Rng.NextItem` on an empty set consumes nothing (`Rng.cs:258-262`), so a
/// run with no other unlocked character must skip exactly the one char-pick
/// draw and otherwise produce the identical roll.
#[test]
fn orobas_skips_exactly_one_draw_without_other_characters() {
    let mut c0 = cond();
    c0.num_other_characters = 0;
    let c4 = cond();
    for seed in 0..64i32 {
        let mut with_pre = event_rng(seed, AncientId::Orobas);
        let _ = with_pre.next_int_exclusive(0, 4);
        let a = roll_offers(AncientId::Orobas, &mut with_pre, &c0);

        let mut fresh = event_rng(seed, AncientId::Orobas);
        let b = roll_offers(AncientId::Orobas, &mut fresh, &c4);
        assert_eq!(a, b, "seed {seed}: char pick is not exactly one draw");
    }
}

/// A single-entry `OptionPool3` still draws (`NextInt(0, 1)`), so the third
/// offer is forced but the stream position is unchanged.
#[test]
fn orobas_single_entry_pool3_still_draws() {
    let mut c = cond();
    c.orobas_pool3 = vec![RelicId::TouchOfOrobas];
    let mut rng = event_rng(31337, AncientId::Orobas);
    let offers = roll_offers(AncientId::Orobas, &mut rng, &c);
    assert_eq!(offers[2], RelicId::TouchOfOrobas);
    assert_eq!(rng.counter(), 5, "1-element NextItem must still sample");
}

/// An EMPTY `OptionPool3` (no Starter relic, no transcendence starter card)
/// is a LOCKED placeholder option in C# (`Orobas.cs:88-91`): `NextItem` still
/// draws once over that one-element list, and the slot offers no relic. So
/// the first two offers and the stream position match the single-entry case.
#[test]
fn orobas_empty_pool3_is_locked_slot_but_still_draws() {
    for seed in 0..64i32 {
        let mut locked = cond();
        locked.orobas_pool3 = Vec::new();
        let mut rng = event_rng(seed, AncientId::Orobas);
        let offers = roll_offers(AncientId::Orobas, &mut rng, &locked);

        let mut single = cond();
        single.orobas_pool3 = vec![RelicId::TouchOfOrobas];
        let mut rng1 = event_rng(seed, AncientId::Orobas);
        let reference = roll_offers(AncientId::Orobas, &mut rng1, &single);

        assert_eq!(offers, reference[..2], "seed {seed}: locked slot dropped");
        assert_eq!(rng.counter(), rng1.counter(), "seed {seed}: one draw");
    }
}

// ── Pael ────────────────────────────────────────────────────────────────

/// `list.AddRange(list)` doubles OptionPool2 *before* `PaelsGrowth` is
/// appended, so the base list is `[Wing, Wing, Growth]` — Growth is 1-in-3,
/// not 1-in-2. With both deck conditions met it becomes
/// `[Wing, Claw, Tooth, Wing, Claw, Tooth, Growth]` (7 entries).
#[test]
fn pael_option2_list_is_doubled_before_growth_is_appended() {
    use RelicId::{PaelsClaw, PaelsGrowth, PaelsTooth, PaelsWing};
    for (goopy, removable, expected) in [
        (0usize, 0usize, vec![PaelsWing, PaelsWing, PaelsGrowth]),
        (
            3,
            0,
            vec![PaelsWing, PaelsClaw, PaelsWing, PaelsClaw, PaelsGrowth],
        ),
        (
            0,
            5,
            vec![PaelsWing, PaelsTooth, PaelsWing, PaelsTooth, PaelsGrowth],
        ),
        (
            3,
            5,
            vec![
                PaelsWing,
                PaelsClaw,
                PaelsTooth,
                PaelsWing,
                PaelsClaw,
                PaelsTooth,
                PaelsGrowth,
            ],
        ),
    ] {
        let mut c = cond();
        c.goopy_eligible_count = goopy;
        c.removable_count = removable;
        for seed in 0..96i32 {
            // Mirror the read order: pool1 pick, then the pool2 pick.
            let mut r = event_rng(seed, AncientId::Pael);
            let _pool1 = r.next_int_exclusive(0, 3);
            let idx = r.next_int_exclusive(0, expected.len() as i32) as usize;

            let mut rng = event_rng(seed, AncientId::Pael);
            let offers = roll_offers(AncientId::Pael, &mut rng, &c);
            assert_eq!(
                offers[1], expected[idx],
                "goopy={goopy} removable={removable} seed={seed}"
            );
        }
    }
}

/// The deck thresholds are `>= 3` (Goopy → PaelsClaw) and `>= 5`
/// (removable → PaelsTooth); one card short offers neither.
#[test]
fn pael_deck_thresholds_are_inclusive_at_three_and_five() {
    let mut below = cond();
    below.goopy_eligible_count = 2;
    below.removable_count = 4;
    let s = seen(AncientId::Pael, &below, 256);
    assert!(!s.contains(&RelicId::PaelsClaw), "PaelsClaw at Goopy=2");
    assert!(
        !s.contains(&RelicId::PaelsTooth),
        "PaelsTooth at removable=4"
    );

    let mut at = cond();
    at.goopy_eligible_count = 3;
    at.removable_count = 5;
    let s = seen(AncientId::Pael, &at, 256);
    assert!(s.contains(&RelicId::PaelsClaw), "no PaelsClaw at Goopy=3");
    assert!(
        s.contains(&RelicId::PaelsTooth),
        "no PaelsTooth at removable=5"
    );
}

/// `PaelsLegion` joins OptionPool3 only when the player holds no event pet.
#[test]
fn pael_legion_requires_no_event_pet() {
    let mut with_pet = cond();
    with_pet.has_event_pet = true;
    let s = seen(AncientId::Pael, &with_pet, 256);
    assert!(
        !s.contains(&RelicId::PaelsLegion),
        "PaelsLegion offered while holding an event pet"
    );
    assert!(s.contains(&RelicId::PaelsEye) && s.contains(&RelicId::PaelsBlood));

    let s = seen(AncientId::Pael, &cond(), 256);
    assert!(
        s.contains(&RelicId::PaelsLegion),
        "PaelsLegion never offered"
    );
}

// ── Tezcatara / Nonupeipe ───────────────────────────────────────────────

/// `NutritiousSoup` is offered iff the deck holds a Basic-rarity Strike.
#[test]
fn tezcatara_soup_requires_a_basic_strike() {
    let mut c = cond();
    c.tezcatara_basic_strike = false;
    assert!(!seen(AncientId::Tezcatara, &c, 256).contains(&RelicId::NutritiousSoup));
    c.tezcatara_basic_strike = true;
    assert!(
        seen(AncientId::Tezcatara, &c, 256).contains(&RelicId::NutritiousSoup),
        "NutritiousSoup never offered with a Basic Strike in deck"
    );
}

/// Nonupeipe's `BeautifulBracelet` needs `Swift.CanEnchant` on `>= 4` cards —
/// 3 is one short.
#[test]
fn nonupeipe_bracelet_threshold_is_four() {
    let mut c = cond();
    c.swift_eligible_count = 3;
    assert!(!seen(AncientId::Nonupeipe, &c, 256).contains(&RelicId::BeautifulBracelet));
    c.swift_eligible_count = 4;
    assert!(
        seen(AncientId::Nonupeipe, &c, 256).contains(&RelicId::BeautifulBracelet),
        "BeautifulBracelet never offered at Swift=4"
    );
}

/// Tanx's `TriBoomerang` needs `Instinct.CanEnchant` on `>= 3` cards — 2 is
/// one short (the inline suite only covers 0 and 3).
#[test]
fn tanx_triboomerang_threshold_is_three() {
    let mut c = cond();
    c.instinct_eligible_count = 2;
    assert!(!seen(AncientId::Tanx, &c, 256).contains(&RelicId::TriBoomerang));
}

// ── Vakuu ───────────────────────────────────────────────────────────────

/// Vakuu draws the head of each of its three shuffled pools, in pool order.
#[test]
fn vakuu_takes_one_relic_from_each_pool_in_order() {
    use RelicId::{
        BloodSoakedRose, ChoicesParadox, DistinguishedCape, Fiddle, JeweledMask, LordsParasol,
        MusicBox, PreservedFog, SereTalon, WhisperingEarring,
    };
    let pools: [&[RelicId]; 3] = [
        &[BloodSoakedRose, WhisperingEarring, Fiddle],
        &[PreservedFog, SereTalon, DistinguishedCape],
        &[ChoicesParadox, MusicBox, LordsParasol, JeweledMask],
    ];
    let c = cond();
    let mut hit: [std::collections::HashSet<RelicId>; 3] = Default::default();
    for seed in 0..256i32 {
        let mut rng = event_rng(seed, AncientId::Vakuu);
        let offers = roll_offers(AncientId::Vakuu, &mut rng, &c);
        for (i, pool) in pools.iter().enumerate() {
            assert!(pool.contains(&offers[i]), "seed {seed} slot {i}");
            hit[i].insert(offers[i]);
        }
    }
    // Every pool member is reachable — no entry is stranded by the shuffle.
    for (i, pool) in pools.iter().enumerate() {
        assert_eq!(
            hit[i].len(),
            pool.len(),
            "pool {i} never offered all entries"
        );
    }
}

// ── Darv ────────────────────────────────────────────────────────────────

/// `PandorasBox` is dropped when a run modifier clears the player deck.
#[test]
fn darv_pandoras_box_respects_the_deck_clearing_modifier() {
    let mut c = cond();
    c.pandoras_allowed = false;
    assert!(!seen(AncientId::Darv, &c, 256).contains(&RelicId::PandorasBox));
    assert!(seen(AncientId::Darv, &cond(), 256).contains(&RelicId::PandorasBox));
}

/// `NextBool()` selects between [2 shuffled picks + DustyTome] and
/// [3 shuffled picks]; DustyTome is appended, so it is always LAST when
/// present, is never one of the shuffled picks, and both branches occur.
#[test]
fn darv_dusty_tome_branch_appends_last() {
    let c = cond();
    let mut with_tome = 0;
    let mut without_tome = 0;
    for seed in 0..256i32 {
        let mut rng = event_rng(seed, AncientId::Darv);
        let offers = roll_offers(AncientId::Darv, &mut rng, &c);
        assert_eq!(offers.len(), 3);
        if let Some(pos) = offers.iter().position(|r| *r == RelicId::DustyTome) {
            assert_eq!(pos, 2, "seed {seed}: DustyTome must be appended last");
            assert_eq!(
                offers.iter().filter(|r| **r == RelicId::DustyTome).count(),
                1
            );
            with_tome += 1;
        } else {
            without_tome += 1;
        }
    }
    assert!(
        with_tome > 0 && without_tome > 0,
        "one NextBool branch never fired"
    );
}

/// Each relic set contributes at most one offer: `Ectoplasm` and `Sozu`
/// share a set (act 2), as do `PhilosophersStone` and `VelvetChoker`
/// (act 3), so the paired relics can never both be offered.
#[test]
fn darv_paired_relic_sets_contribute_at_most_one_offer() {
    for (act, pair) in [
        (1, (RelicId::Ectoplasm, RelicId::Sozu)),
        (2, (RelicId::PhilosophersStone, RelicId::VelvetChoker)),
    ] {
        let mut c = cond();
        c.act_index = act;
        let mut saw_a = false;
        let mut saw_b = false;
        for seed in 0..512i32 {
            let mut rng = event_rng(seed, AncientId::Darv);
            let offers = roll_offers(AncientId::Darv, &mut rng, &c);
            let a = offers.contains(&pair.0);
            let b = offers.contains(&pair.1);
            assert!(!(a && b), "act {act} seed {seed} offered both {pair:?}");
            saw_a |= a;
            saw_b |= b;
        }
        assert!(saw_a && saw_b, "act {act}: {pair:?} not both reachable");
    }
}

// ── Property ────────────────────────────────────────────────────────────

fn arb_conditions(tc: &TestCase) -> AncientConditions {
    let count = |hi: u32| gs::integers::<u32>().min_value(0).max_value(hi);
    // A non-empty OptionPool3 here; the empty (LOCKED, 2-offer) case is
    // pinned by `orobas_empty_pool3_is_locked_slot_but_still_draws`.
    let pool3 = match tc.draw(count(2)) {
        0 => vec![RelicId::TouchOfOrobas],
        1 => vec![RelicId::ArchaicTooth],
        _ => vec![RelicId::TouchOfOrobas, RelicId::ArchaicTooth],
    };
    AncientConditions {
        act_index: tc.draw(gs::integers::<i32>().min_value(0).max_value(3)),
        goopy_eligible_count: tc.draw(count(12)) as usize,
        removable_count: tc.draw(count(12)) as usize,
        has_event_pet: tc.draw(count(1)) == 1,
        instinct_eligible_count: tc.draw(count(12)) as usize,
        swift_eligible_count: tc.draw(count(12)) as usize,
        num_other_characters: tc.draw(count(5)) as usize,
        orobas_pool3: pool3,
        pandoras_allowed: tc.draw(count(1)) == 1,
        tezcatara_basic_strike: tc.draw(count(1)) == 1,
    }
}

/// Across arbitrary seeds and arbitrary deck/run conditions every non-Neow
/// ancient offers exactly 3 DISTINCT relics from its own universe, and the
/// roll is reproducible from the same seed. Distinctness holds structurally:
/// each event either draws from three disjoint pools or shuffles one pool of
/// distinct relics and takes 3 (Darv's DustyTome is in no relic set).
#[hegel::test(test_cases = 300)]
fn any_conditions_yield_three_distinct_reproducible_offers(tc: TestCase) {
    let seed = tc.draw(gs::integers::<i32>());
    let c = arb_conditions(&tc);
    for a in ALL_ANCIENTS {
        let mut r1 = event_rng(seed, a);
        let offers = roll_offers(a, &mut r1, &c);
        tc.note(&format!("{a:?} seed={seed} offers={offers:?}"));
        assert_eq!(offers.len(), 3, "{a:?}");
        assert!(
            offers[0] != offers[1] && offers[1] != offers[2] && offers[0] != offers[2],
            "{a:?} offered duplicates: {offers:?}"
        );
        let uni = universe(a);
        for r in &offers {
            assert!(uni.contains(r), "{a:?} offered out-of-pool {r:?}");
        }
        let mut r2 = event_rng(seed, a);
        assert_eq!(
            roll_offers(a, &mut r2, &c),
            offers,
            "{a:?} not deterministic"
        );
        assert_eq!(r1.counter(), r2.counter(), "{a:?} draw count not stable");
    }
}

/// Different ancients under the same master seed run independent streams
/// (`with_name` salts the seed with the event's `Id.Entry`), so a run's act-2
/// and act-3 boss-relic rolls are not correlated by construction.
#[hegel::test(test_cases = 200)]
fn event_streams_are_salted_per_ancient(tc: TestCase) {
    let seed = tc.draw(gs::integers::<i32>());
    let mut seeds = std::collections::HashSet::new();
    for a in [
        AncientId::Neow,
        AncientId::Orobas,
        AncientId::Pael,
        AncientId::Tezcatara,
        AncientId::Nonupeipe,
        AncientId::Tanx,
        AncientId::Vakuu,
        AncientId::Darv,
    ] {
        let rng = event_rng(seed, a);
        assert!(
            seeds.insert(rng.seed()),
            "{a:?} collides with another ancient's stream at master {seed}"
        );
    }
}
