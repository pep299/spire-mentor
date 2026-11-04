//! Inverse (canonical-id -> display) maps for the workbench backend.
//!
//! Port of `sts2-pycli/python/sts2_hint/display_maps.py` plus the slice of
//! `sts2_hint/canonical_maps.py` it inverts (`MONSTER_TITLE_TO_ID`,
//! `POWER_TITLE_TO_ID`). The engine hands the workbench canonical ids; the
//! enrichment layer needs them back as the display strings / loc keys.
//!
//! Two id shapes show up from the engine and are normalised here: combat
//! hand cards use `SCREAMING_SNAKE` ids (`STRIKE_IRONCLAD`), the master deck
//! uses `PascalCase` (`StrikeIronclad`), relics carry a `RELIC.` prefix.
//! The localization keys (and card-art slugs) are all `SCREAMING_SNAKE`, so
//! [`loc_id`] collapses every shape to it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use crate::data;
use crate::localization::py_title;

/// Monster display-name -> CamelCase registry id, covering every monster the
/// engine models. Single-sourced from the engine registry × the display
/// names in `localization_eng/monsters.json` and pinned by the
/// `monster_map_sync` tests below; the trailing `Hatchling` entry is the
/// one `MONSTER_TITLE_ALIASES` runtime-only title (Tough Egg's post-hatch
/// name).
///
/// Deliberately EXCLUDED (kept in sync with the sync test's allow-list):
/// synthetic test monsters with no localization entry, and ids whose
/// display name COLLIDES with another id (`DecimillipedeSegment{Back,Front,
/// Middle}`, `DenseVegetationWriggler`).
pub const MONSTER_TITLE_TO_ID: &[(&str, &str)] = &[
    ("Assassin Raider", "AssassinRubyRaider"),
    ("Axe Raider", "AxeRubyRaider"),
    ("Axebot", "Axebot"),
    ("Battle Friend V1.0", "BattleFriendV1"),
    ("Battle Friend V2.0", "BattleFriendV2"),
    ("Battle Friend V3.0", "BattleFriendV3"),
    ("Big Dummy", "BigDummy"),
    ("Bowlbug (Egg)", "BowlbugEgg"),
    ("Bowlbug (Nectar)", "BowlbugNectar"),
    ("Bowlbug (Rock)", "BowlbugRock"),
    ("Bowlbug (Silk)", "BowlbugSilk"),
    ("Brute Raider", "BruteRubyRaider"),
    ("Bygone Effigy", "BygoneEffigy"),
    ("Byrdonis", "Byrdonis"),
    ("Byrdpip", "Byrdpip"),
    ("Calcified Cultist", "CalcifiedCultist"),
    ("Ceremonial Beast", "CeremonialBeast"),
    ("Chomper", "Chomper"),
    ("Corpse Slug", "CorpseSlug"),
    ("Crossbow Raider", "CrossbowRubyRaider"),
    ("Crusher", "Crusher"),
    ("Cubex Construct", "CubexConstruct"),
    ("Damp Cultist", "DampCultist"),
    ("Decimillipede", "DecimillipedeSegment"),
    ("Aeonglass", "Aeonglass"),
    ("Devoted Sculptor", "DevotedSculptor"),
    ("Entomancer", "Entomancer"),
    ("Exoskeleton", "Exoskeleton"),
    ("Eye with Teeth", "EyeWithTeeth"),
    ("Fabricator", "Fabricator"),
    ("Fat Gremlin", "FatGremlin"),
    ("Flail Knight", "FlailKnight"),
    ("Flyconid", "Flyconid"),
    ("Fogmog", "Fogmog"),
    ("Fossil Stalker", "FossilStalker"),
    ("Frog Knight", "FrogKnight"),
    ("Fuzzy Wurm Crawler", "FuzzyWurmCrawler"),
    ("Gas Bomb", "GasBomb"),
    ("Globe Head", "GlobeHead"),
    ("Gremlin Merc", "GremlinMerc"),
    ("Guardbot", "Guardbot"),
    ("Haunted Ship", "HauntedShip"),
    ("Hunter Killer", "HunterKiller"),
    ("Infested Prism", "InfestedPrism"),
    ("Inklet", "Inklet"),
    ("Kin Follower", "KinFollower"),
    ("Kin Priest", "KinPriest"),
    ("Knowledge Demon", "KnowledgeDemon"),
    ("Lagavulin Matriarch", "LagavulinMatriarch"),
    ("Leaf Slime (M)", "LeafSlimeM"),
    ("Leaf Slime (S)", "LeafSlimeS"),
    ("Living Fog", "LivingFog"),
    ("Living Shield", "LivingShield"),
    ("Louse Progenitor", "LouseProgenitor"),
    ("Magi Knight", "MagiKnight"),
    ("Mawler", "Mawler"),
    ("Mecha Knight", "MechaKnight"),
    ("Mysterious Knight", "MysteriousKnight"),
    ("Myte", "Myte"),
    ("Nibbit", "Nibbit"),
    ("Noisebot", "Noisebot"),
    ("Osty", "Osty"),
    ("Ovicopter", "Ovicopter"),
    ("Owl Magistrate", "OwlMagistrate"),
    ("Pael's Legion", "PaelsLegion"),
    ("Parafright", "Parafright"),
    ("Phantasmal Gardener", "PhantasmalGardener"),
    ("Phrog Parasite", "PhrogParasite"),
    ("Punch Construct", "PunchConstruct"),
    ("Queen", "Queen"),
    ("Rocket", "Rocket"),
    ("Scroll of Biting", "ScrollOfBiting"),
    ("Seapunk", "Seapunk"),
    ("Sewer Clam", "SewerClam"),
    ("Shrinker Beetle", "ShrinkerBeetle"),
    ("Skulking Colony", "SkulkingColony"),
    ("Slimed Berserker", "SlimedBerserker"),
    ("Slithering Strangler", "SlitheringStrangler"),
    ("Sludge Spinner", "SludgeSpinner"),
    ("Slumbering Beetle", "SlumberingBeetle"),
    ("Snapping Jaxfruit", "SnappingJaxfruit"),
    ("Sneaky Gremlin", "SneakyGremlin"),
    ("Soul Fysh", "SoulFysh"),
    ("Soul Nexus", "SoulNexus"),
    ("Spectral Knight", "SpectralKnight"),
    ("Spiny Toad", "SpinyToad"),
    ("Stabbot", "Stabbot"),
    ("Terror Eel", "TerrorEel"),
    ("Test Subject #C{Count}", "TestSubject"),
    ("The Adversary Mk 1", "TheAdversaryMkOne"),
    ("The Adversary Mk 2", "TheAdversaryMkTwo"),
    ("The Adversary Mk 3", "TheAdversaryMkThree"),
    ("The Architect", "Architect"),
    ("The Forgotten", "TheForgotten"),
    ("The Insatiable", "TheInsatiable"),
    ("The Lost", "TheLost"),
    ("The Merchant???", "FakeMerchantMonster"),
    ("The Obscura", "TheObscura"),
    ("Thieving Hopper", "ThievingHopper"),
    ("Toadpole", "Toadpole"),
    ("Torch Head Amalgam", "TorchHeadAmalgam"),
    ("Tough Egg", "ToughEgg"),
    ("Tracker Raider", "TrackerRubyRaider"),
    ("Tunneler", "Tunneler"),
    ("Turret Operator", "TurretOperator"),
    ("Twig Slime (M)", "TwigSlimeM"),
    ("Twig Slime (S)", "TwigSlimeS"),
    ("Two-Tailed Rat", "TwoTailedRat"),
    ("Vantom", "Vantom"),
    ("Vine Shambler", "VineShambler"),
    ("Waterfall Giant", "WaterfallGiant"),
    ("Wriggler", "Wriggler"),
    ("Zapbot", "Zapbot"),
    // `MONSTER_TITLE_ALIASES`, merged last: ToughEgg overrides `Title` to
    // `HATCHLING.name` once its HATCH move fires.
    ("Hatchling", "ToughEgg"),
];

/// The runtime-only titles merged on top of the derived map (the Python
/// `MONSTER_TITLE_ALIASES`).
pub const MONSTER_TITLE_ALIASES: &[(&str, &str)] = &[("Hatchling", "ToughEgg")];

/// Ids whose display title collides with another id and so are kept OUT of
/// the title->id map; the id->title direction has no collision, so they are
/// added to [`monster_id_to_title`] explicitly — all three localize to
/// "Decimillipede" (whose slug resolves to `monsters/decimillipede.png`;
/// the per-segment rigs are picked by kind, see `assets::MONSTER_KIND_RIGS`).
const MONSTER_ID_TO_TITLE_EXTRA: &[(&str, &str)] = &[
    ("DecimillipedeSegmentFront", "Decimillipede"),
    ("DecimillipedeSegmentMiddle", "Decimillipede"),
    ("DecimillipedeSegmentBack", "Decimillipede"),
];

/// Registry id -> display title for monsters (the inversion of
/// [`MONSTER_TITLE_TO_ID`], last title wins, plus the segment extras).
pub fn monster_id_to_title() -> &'static BTreeMap<&'static str, &'static str> {
    static MAP: OnceLock<BTreeMap<&'static str, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut out: BTreeMap<&'static str, &'static str> = BTreeMap::new();
        for (title, id) in MONSTER_TITLE_TO_ID {
            out.insert(id, title);
        }
        for (id, title) in MONSTER_ID_TO_TITLE_EXTRA {
            out.insert(id, title);
        }
        out
    })
}

/// `POWER_TITLE_TO_ID` as `canonical_maps.py` builds it, in insertion
/// order: `powers.json`'s `<ID>.title` rows inverted (first title wins on a
/// collision), the `_POWER` suffix stripped from the ids, then augmented
/// with the engine's power registry for the powers that have no loc entry
/// (title-cased fallback; a `_POWER` variant wins over its bare twin), and
/// finally PRUNED of every id the engine registry does not know
/// (`_prune_against_registry`) — the server always has the engine, so a
/// loc-only power (`DECREE_OF_ENTROPY_POWER`, `HEATSINKS_POWER`) has no
/// curated title and [`power_title`] falls back to the title-cased id.
pub(crate) fn power_title_to_id() -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    // `_harvest_title_map`
    for (key, value) in &data::data().loc_powers.0 {
        let Some(reg_id) = key.strip_suffix(".title") else {
            continue;
        };
        // A colliding title keeps its FIRST registration.
        if !out.iter().any(|(t, _)| t == value) {
            out.push((value.clone(), reg_id.to_string()));
        }
    }
    // `_strip_power_suffix` (titles are unique here, so a collision cannot
    // happen; mirrored for fidelity).
    let mut stripped: Vec<(String, String)> = Vec::with_capacity(out.len());
    for (title, pid) in out {
        let bare = pid.strip_suffix("_POWER").unwrap_or(&pid).to_string();
        if !stripped.iter().any(|(t, _)| *t == title) {
            stripped.push((title, bare));
        }
    }
    // `_augment_power_titles_from_registry`
    let loc_titles: BTreeSet<String> = stripped.iter().map(|(t, _)| t.clone()).collect();
    let names = sts2_wire::state_dict::registry_names_for("power").unwrap_or_default();
    for pname in &names {
        let stem = pname.strip_suffix("_POWER").unwrap_or(pname);
        let title = py_title(&stem.replace('_', " "));
        if loc_titles.contains(&title) {
            continue; // localization is authoritative; never override it
        }
        match stripped.iter_mut().find(|(t, _)| *t == title) {
            None => stripped.push((title, pname.clone())),
            Some((_, existing)) => {
                if pname.ends_with("_POWER") && !existing.ends_with("_POWER") {
                    existing.clone_from(pname);
                }
            }
        }
    }
    // `_prune_against_registry`: drop every pair whose (stripped) id the
    // engine does not know (`dict.pop` keeps the survivors' order).
    let known: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    stripped.retain(|(_, id)| known.contains(id.as_str()));
    stripped
}

/// Registry id -> display title for powers (the inversion of
/// `POWER_TITLE_TO_ID`, last title wins).
pub fn power_id_to_title() -> &'static BTreeMap<String, String> {
    static MAP: OnceLock<BTreeMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut out = BTreeMap::new();
        for (title, id) in power_title_to_id() {
            out.insert(id, title);
        }
        out
    })
}

/// `StrikeIronclad` -> `STRIKE_IRONCLAD`; idempotent on snake/all-caps.
///
/// Inserts an underscore before each uppercase letter that follows an
/// alphanumeric character (so capital runs split too, `ExpectAFight` ->
/// `EXPECT_A_FIGHT`), then upper-cases — mirroring the game's
/// `StringHelper.Slugify`. A string that already contains `_` (or has no
/// lowercase letters) is assumed to be SCREAMING_SNAKE already and is
/// returned upper-cased as-is.
#[must_use]
pub fn pascal_to_screaming_snake(s: &str) -> String {
    if s.contains('_') || !s.chars().any(char::is_lowercase) {
        return s.to_uppercase();
    }
    let mut out = String::with_capacity(s.len() + 8);
    let mut prev: Option<char> = None;
    for c in s.chars() {
        if c.is_ascii_uppercase() && prev.is_some_and(|p| p.is_ascii_alphanumeric()) {
            out.push('_');
        }
        out.push(c);
        prev = Some(c);
    }
    out.to_uppercase()
}

/// Normalise any engine id to the SCREAMING_SNAKE loc-key/art stem.
///
/// Strips a registry prefix (`RELIC.WHETSTONE` -> `WHETSTONE`), converts
/// PascalCase (`StrikeIronclad` -> `STRIKE_IRONCLAD`), and drops the
/// disambiguating `_CARD` / `_STATUS` suffix a few card ids carry that the
/// loc keys and art slugs omit (`STAMPEDE_CARD` -> `STAMPEDE`,
/// `SLOTH_STATUS` -> `SLOTH`). A `RELIC.`-prefixed id keeps its suffix: the
/// Membership Card relic's loc key is `MEMBERSHIP_CARD`.
#[must_use]
pub fn loc_id(raw_id: &str) -> String {
    let (prefix, bare) = raw_id.split_once('.').unwrap_or(("", raw_id));
    let stem = pascal_to_screaming_snake(bare);
    if prefix.eq_ignore_ascii_case("RELIC") {
        return stem;
    }
    for suffix in ["_CARD", "_STATUS"] {
        if let Some(s) = stem.strip_suffix(suffix) {
            return s.to_string();
        }
    }
    stem
}

/// Display name for a power id (`STRENGTH` -> `Strength`), falling back to a
/// title-cased form of the id when the power isn't in the curated map.
#[must_use]
pub fn power_title(power_id: &str) -> String {
    let pid = power_id.strip_suffix("_POWER").unwrap_or(power_id);
    match power_id_to_title().get(pid) {
        Some(title) if !title.is_empty() => title.clone(),
        _ => py_title(&pid.replace('_', " ")),
    }
}

/// `\s*#\S*\{[^}]+\}.*$` -> `""`: strip a trailing specimen designation
/// carrying an unresolved loc template variable (Test Subject's
/// `"Test Subject #C{Count}"`).
fn strip_specimen_template(title: &str) -> String {
    let chars: Vec<char> = title.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '#' {
            // `\S*\{[^}]+\}`: a `{` reachable without whitespace, then at
            // least one non-`}` and a `}`.
            let mut j = i + 1;
            while j < chars.len() && !chars[j].is_whitespace() {
                if chars[j] == '{' && has_var_close(&chars, j) {
                    // Extend the match start back over the leading `\s*`.
                    let mut start = i;
                    while start > 0 && chars[start - 1].is_whitespace() {
                        start -= 1;
                    }
                    return chars[..start].iter().collect();
                }
                j += 1;
            }
        }
        i += 1;
    }
    title.to_string()
}

/// True when `chars[open] == '{'` is followed by `[^}]+` and then `}`.
fn has_var_close(chars: &[char], open: usize) -> bool {
    let mut k = open + 1;
    while k < chars.len() && chars[k] != '}' {
        k += 1;
    }
    k < chars.len() && k > open + 1
}

/// `\s*\{[^}]+\}` -> `""` everywhere: the safety net for any other
/// unresolved `{Var}` token.
fn strip_loc_template_vars(title: &str) -> String {
    let chars: Vec<char> = title.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    let mut last_match_end = 0;
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '{' && has_var_close(&chars, i) {
            // Drop the whitespace run just appended since the last match.
            while out.len() > last_match_end && out.last().is_some_and(|c| c.is_whitespace()) {
                out.pop();
            }
            let mut k = i + 1;
            while chars[k] != '}' {
                k += 1;
            }
            i = k + 1;
            last_match_end = out.len();
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out.into_iter().collect()
}

/// Display name for a monster kind (`Nibbit` -> `Nibbit`), falling back to
/// spacing the CamelCase id when the monster isn't curated. A curated title
/// holding an unresolved loc variable (`"Test Subject #C{Count}"`) is
/// stripped to the bare species name.
#[must_use]
pub fn monster_title(monster_id: &str) -> String {
    if let Some(title) = monster_id_to_title().get(monster_id) {
        let title = strip_specimen_template(title);
        return strip_loc_template_vars(&title).trim().to_string();
    }
    // `(?<=[a-z0-9])(?=[A-Z])` -> " "
    let mut out = String::with_capacity(monster_id.len() + 4);
    let mut prev: Option<char> = None;
    for c in monster_id.chars() {
        if c.is_ascii_uppercase()
            && prev.is_some_and(|p| p.is_ascii_lowercase() || p.is_ascii_digit())
        {
            out.push(' ');
        }
        out.push(c);
        prev = Some(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::slugify;

    // ── tests/test_display_maps.py ─────────────────────────────────

    #[test]
    fn loc_id_normalises_pascal_and_snake() {
        assert_eq!(loc_id("StrikeIronclad"), "STRIKE_IRONCLAD");
        assert_eq!(loc_id("STRIKE_IRONCLAD"), "STRIKE_IRONCLAD");
        assert_eq!(loc_id("RELIC.WHETSTONE"), "WHETSTONE");
    }

    #[test]
    fn loc_id_splits_capital_runs_like_game_slugify() {
        assert_eq!(loc_id("ExpectAFight"), "EXPECT_A_FIGHT");
        assert_eq!(loc_id("IAmInvincible"), "I_AM_INVINCIBLE");
        assert_eq!(loc_id("EXPECT_A_FIGHT"), "EXPECT_A_FIGHT");
    }

    #[test]
    fn loc_id_strips_card_suffix_to_match_loc_key() {
        assert_eq!(loc_id("StampedeCard"), "STAMPEDE");
        assert_eq!(loc_id("STAMPEDE_CARD"), "STAMPEDE");
        assert_eq!(loc_id("INFERNO_CARD"), "INFERNO");
        assert_eq!(loc_id("FOCUSED_STRIKE_CARD"), "FOCUSED_STRIKE");
        assert_eq!(loc_id("RELIC.MEMBERSHIP_CARD"), "MEMBERSHIP_CARD");
        // A real `_IRONCLAD` suffix (or any non-_CARD tail) is untouched.
        assert_eq!(loc_id("DEFEND_IRONCLAD"), "DEFEND_IRONCLAD");
    }

    #[test]
    fn loc_id_strips_status_suffix_to_match_loc_key() {
        assert_eq!(loc_id("SLOTH_STATUS"), "SLOTH");
        assert_eq!(loc_id("SlothStatus"), "SLOTH");
    }

    #[test]
    fn monster_title_strips_unresolved_specimen_template() {
        let title = monster_title("TestSubject");
        assert_eq!(title, "Test Subject");
        assert!(!title.contains('{') && !title.contains('#'));
        assert_eq!(slugify(&title), "test_subject");
    }

    #[test]
    fn monster_title_leaves_plain_names_untouched() {
        assert_eq!(monster_title("Nibbit"), "Nibbit");
        assert_eq!(monster_title("TheForgotten"), "The Forgotten");
        assert_eq!(monster_title("AssassinRubyRaider"), "Assassin Raider");
        // Uncurated id falls back to CamelCase spacing.
        assert_eq!(monster_title("SomeNewMonster"), "Some New Monster");
        assert_eq!(monster_title("BattleFriendV1"), "Battle Friend V1.0");
        // The alias title is what the inversion keeps for ToughEgg (last wins).
        assert_eq!(monster_title("ToughEgg"), "Hatchling");
    }

    #[test]
    fn monster_title_decimillipede_segments_share_base_name() {
        for kind in [
            "DecimillipedeSegmentFront",
            "DecimillipedeSegmentMiddle",
            "DecimillipedeSegmentBack",
        ] {
            let title = monster_title(kind);
            assert_eq!(title, "Decimillipede", "{kind}");
            assert_eq!(slugify(&title), "decimillipede", "{kind}");
        }
    }

    #[test]
    fn template_strippers_match_the_python_regexes() {
        assert_eq!(
            strip_specimen_template("Test Subject #C{Count}"),
            "Test Subject"
        );
        assert_eq!(
            strip_specimen_template("Test Subject #C{Count} extra"),
            "Test Subject"
        );
        assert_eq!(
            strip_specimen_template("Test Subject #C {Count}"),
            "Test Subject #C {Count}"
        );
        assert_eq!(strip_specimen_template("Name #{}"), "Name #{}");
        assert_eq!(strip_specimen_template("Plain"), "Plain");
        assert_eq!(
            strip_loc_template_vars("Test Subject #C {Count}"),
            "Test Subject #C"
        );
        assert_eq!(strip_loc_template_vars("A {X} B {Y}"), "A B");
        assert_eq!(strip_loc_template_vars("A {} B"), "A {} B");
        assert_eq!(strip_loc_template_vars("{X}A"), "A");
    }

    #[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
    #[test]
    fn power_title_prefers_the_curated_title() {
        assert_eq!(power_title("STRENGTH"), "Strength");
        assert_eq!(power_title("STRENGTH_POWER"), "Strength");
        assert_eq!(power_title("NO_SUCH_THING"), "No Such Thing");
        assert_eq!(power_title("NO_SUCH_THING_POWER"), "No Such Thing");
        // Every loc row: the curated title when the engine knows the (stripped)
        // id, the title-cased fallback when it does not — the server prunes
        // the map against the engine registry, so a loc-only power has no
        // curated title: `DECREE_OF_ENTROPY_POWER` renders "Decree Of
        // Entropy", capital O, exactly like the Python side.
        let known: BTreeSet<String> = sts2_wire::state_dict::registry_names_for("power")
            .unwrap()
            .into_iter()
            .collect();
        let (mut curated, mut pruned) = (0, 0);
        for (key, title) in &data::data().loc_powers.0 {
            let Some(stem) = key.strip_suffix(".title") else {
                continue;
            };
            let bare = stem.strip_suffix("_POWER").unwrap_or(stem);
            let fallback = py_title(&bare.replace('_', " "));
            let got = power_title(stem);
            if known.contains(bare) {
                // The registry's bare/`_POWER` twin rule re-points the id
                // (`FREE` + `FREE_POWER` -> "Free"), checked explicitly below.
                if bare
                    .strip_suffix("_POWER")
                    .is_some_and(|b| known.contains(b))
                {
                    continue;
                }
                // The curated title, unless an earlier row claimed the same
                // title first (first wins).
                assert!(got == *title || got == fallback, "{stem}: {got}");
                curated += 1;
            } else {
                assert_eq!(got, fallback, "{stem}");
                pruned += 1;
            }
        }
        assert!(
            curated > 100 && pruned > 0,
            "{curated} curated, {pruned} pruned"
        );
        assert_eq!(power_title("DECREE_OF_ENTROPY_POWER"), "Decree Of Entropy");
        assert_eq!(power_title("HEATSINKS_POWER"), "Heatsinks");
        assert_eq!(
            power_title("HUNGER_OF_THE_VOID_POWER"),
            "Hunger Of The Void"
        );
        // "FREE_POWER_POWER.title" is "Free Power", but the registry holds
        // both `FREE` and `FREE_POWER`, and the `_POWER` twin wins the
        // fallback title "Free" — which then wins the inversion.
        assert_eq!(power_title("FREE_POWER_POWER"), "Free");
        assert_eq!(power_title("FREE_POWER"), "Free");
    }

    // ── tests/workbench/test_monster_map_sync.py ───────────────────

    /// Engine monster ids intentionally absent from a title->id map.
    const EXCLUDED: &[&str] = &[
        // Synthetic test monsters (no localization entry).
        "OneHpMonster",
        "TenHpMonster",
        "SingleAttackMoveMonster",
        "MultiAttackMoveMonster",
        // Display-name collisions a title->id map cannot disambiguate.
        "DecimillipedeSegmentBack",
        "DecimillipedeSegmentFront",
        "DecimillipedeSegmentMiddle",
        "DenseVegetationWriggler",
    ];

    fn engine_monsters() -> BTreeSet<String> {
        sts2_wire::state_dict::registry_names_for("monster")
            .unwrap()
            .into_iter()
            .collect()
    }

    /// `ScrollOfBiting` -> `SCROLL_OF_BITING` (localization key stem).
    #[cfg(workbench_game_data)] // only the game-data-gated tests use it
    fn screaming(camel: &str) -> String {
        let mut out = String::new();
        for (i, c) in camel.chars().enumerate() {
            if i > 0 && c.is_ascii_uppercase() {
                out.push('_');
            }
            out.push(c);
        }
        out.to_uppercase()
    }

    /// Regenerate the canonical map from the engine registry × localization.
    #[cfg(workbench_game_data)] // only the game-data-gated tests use it
    fn derive_expected() -> BTreeMap<String, String> {
        let loc = &data::data().loc_monsters;
        let mut expected = BTreeMap::new();
        for mid in engine_monsters() {
            if EXCLUDED.contains(&mid.as_str()) {
                continue;
            }
            let title = loc
                .get(&format!("{}.name", screaming(&mid)))
                .unwrap_or_else(|| {
                    panic!("engine monster {mid:?} has no localization '.name' and is not EXCLUDED")
                });
            expected.insert(title.clone(), mid);
        }
        for (title, id) in MONSTER_TITLE_ALIASES {
            expected.insert((*title).to_string(), (*id).to_string());
        }
        expected
    }

    #[test]
    fn every_modeled_monster_is_mapped() {
        let mapped: BTreeSet<&str> = MONSTER_TITLE_TO_ID.iter().map(|(_, id)| *id).collect();
        let unmapped: Vec<String> = engine_monsters()
            .into_iter()
            .filter(|m| !mapped.contains(m.as_str()) && !EXCLUDED.contains(&m.as_str()))
            .collect();
        assert!(
            unmapped.is_empty(),
            "engine monster(s) missing from MONSTER_TITLE_TO_ID: {unmapped:?}"
        );
    }

    #[test]
    fn no_stale_or_unknown_ids() {
        let engine = engine_monsters();
        let unknown: Vec<&str> = MONSTER_TITLE_TO_ID
            .iter()
            .map(|(_, id)| *id)
            .filter(|id| !engine.contains(*id))
            .collect();
        assert!(
            unknown.is_empty(),
            "map ids not in the engine registry: {unknown:?}"
        );
    }

    #[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
    #[test]
    fn map_matches_localization_derivation() {
        let actual: BTreeMap<String, String> = MONSTER_TITLE_TO_ID
            .iter()
            .map(|(t, id)| ((*t).to_string(), (*id).to_string()))
            .collect();
        assert_eq!(derive_expected(), actual);
    }

    #[test]
    fn ids_are_unique() {
        let alias_titles: BTreeSet<&str> = MONSTER_TITLE_ALIASES.iter().map(|(t, _)| *t).collect();
        let alias_ids: BTreeSet<&str> = MONSTER_TITLE_ALIASES.iter().map(|(_, id)| *id).collect();
        let ids: Vec<&str> = MONSTER_TITLE_TO_ID
            .iter()
            .filter(|(t, _)| !alias_titles.contains(t))
            .map(|(_, id)| *id)
            .collect();
        let unique: BTreeSet<&str> = ids.iter().copied().collect();
        assert_eq!(
            ids.len(),
            unique.len(),
            "duplicate id(s) in MONSTER_TITLE_TO_ID"
        );
        assert!(
            alias_ids.is_subset(&unique),
            "MONSTER_TITLE_ALIASES id(s) not otherwise present: {:?}",
            alias_ids.difference(&unique).collect::<Vec<_>>()
        );
    }

    // ── tests/workbench/test_orb_map_sync.py ───────────────────────

    /// Defect orb localization-id stem (`DARK_ORB.title`) -> registry id.
    const ORB_TITLE_TO_ID: &[(&str, &str)] = &[
        ("Dark", "DARK"),
        ("Frost", "FROST"),
        ("Glass", "GLASS"),
        ("Lightning", "LIGHTNING"),
        ("Plasma", "PLASMA"),
        ("Orb Slot", "EMPTY"), // Empty slot's display name
    ];

    #[cfg(workbench_game_data)] // only the game-data-gated tests use it
    fn orb_loc_stem(orb_id: &str) -> String {
        if orb_id == "EMPTY" {
            "EMPTY_SLOT".to_string()
        } else {
            format!("{orb_id}_ORB")
        }
    }

    fn engine_orbs() -> BTreeSet<String> {
        sts2_wire::state_dict::registry_names_for("orb")
            .unwrap()
            .into_iter()
            .collect()
    }

    #[test]
    fn every_modeled_orb_is_mapped() {
        let mapped: BTreeSet<&str> = ORB_TITLE_TO_ID.iter().map(|(_, id)| *id).collect();
        let unmapped: Vec<String> = engine_orbs()
            .into_iter()
            .filter(|o| !mapped.contains(o.as_str()))
            .collect();
        assert!(
            unmapped.is_empty(),
            "engine orb(s) missing from ORB_TITLE_TO_ID: {unmapped:?}"
        );
    }

    #[test]
    fn orb_map_has_no_stale_or_unknown_ids() {
        let engine = engine_orbs();
        let unknown: Vec<&str> = ORB_TITLE_TO_ID
            .iter()
            .map(|(_, id)| *id)
            .filter(|id| !engine.contains(*id))
            .collect();
        assert!(
            unknown.is_empty(),
            "map ids not in the engine registry: {unknown:?}"
        );
    }

    #[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
    #[test]
    fn orb_map_matches_localization_derivation() {
        let loc = &data::data().loc_orbs;
        let expected: BTreeMap<String, String> = engine_orbs()
            .into_iter()
            .map(|orb_id| {
                let stem = orb_loc_stem(&orb_id);
                let title = loc.get(&format!("{stem}.title")).unwrap_or_else(|| {
                    panic!("engine orb {orb_id:?} has no localization '.title' under {stem:?}")
                });
                (title.clone(), orb_id)
            })
            .collect();
        let actual: BTreeMap<String, String> = ORB_TITLE_TO_ID
            .iter()
            .map(|(t, id)| ((*t).to_string(), (*id).to_string()))
            .collect();
        assert_eq!(expected, actual);
    }

    #[cfg(workbench_game_data)] // asserts on real game text (see build.rs)
    #[test]
    fn orb_titles_are_the_wire_spelling() {
        let loc = &data::data().loc_orbs;
        let titles: BTreeSet<&str> = loc
            .iter()
            .filter(|(k, _)| {
                k.strip_suffix(".title").is_some_and(|stem| {
                    !stem.is_empty() && stem.chars().all(|c| c.is_ascii_uppercase() || c == '_')
                })
            })
            .map(|(_, v)| v.as_str())
            .collect();
        for (title, _) in ORB_TITLE_TO_ID {
            assert!(titles.contains(title), "{title}");
        }
    }
}
