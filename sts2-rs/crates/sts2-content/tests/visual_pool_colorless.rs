//! Regression: `is_visual_pool_colorless` mirrors C#
//! `CardModel.VisualCardPool.IsColorless` — true for the Colorless, Token and
//! Event pools, false for character pools, Status and Curse, and false for
//! the Event cards whose `VisualCardPool` is a character pool.
//!
//! HeirloomHammer (HeirloomHammer.cs:33) filters hand cards by this property
//! before cloning one. The old `is_colorless_card_id` list omitted Token cards
//! like SOVEREIGN_BLADE, so HeirloomHammer cloned a different colorless hand
//! card than C# (Regent fuzz seeds 42 / 366 / 406). Bead `sts2-rs-1rv1q`.

use sts2_content::cards::colorless::is_visual_pool_colorless;
use sts2_core::cards::CardId;

#[test]
fn token_cards_are_visual_pool_colorless() {
    // SOVEREIGN_BLADE is a Token (CardRarity::Token) — TokenCardPool.IsColorless.
    assert!(
        is_visual_pool_colorless(CardId::SovereignBlade),
        "SOVEREIGN_BLADE (Token) must count as colorless for HeirloomHammer",
    );
}

#[test]
fn colorless_proper_cards_are_visual_pool_colorless() {
    // Both EQUILIBRIUM and SECRET_WEAPON are colorless-proper.
    assert!(is_visual_pool_colorless(CardId::Equilibrium));
    assert!(is_visual_pool_colorless(CardId::SecretWeapon));
    assert!(is_visual_pool_colorless(CardId::Alchemize));
}

#[test]
fn character_cards_are_not_visual_pool_colorless() {
    // Character-pool cards (RegentCardPool / IroncladCardPool) are NOT colorless.
    assert!(
        !is_visual_pool_colorless(CardId::StrikeIronclad),
        "an Ironclad card must not count as colorless",
    );
    assert!(
        !is_visual_pool_colorless(CardId::HeirloomHammer),
        "a Regent Rare must not count as colorless",
    );
}

#[test]
fn status_cards_are_not_visual_pool_colorless() {
    // StatusCardPool.IsColorless => false — must stay excluded even though the
    // sts2-core character map labels status cards "colorless".
    assert!(
        !is_visual_pool_colorless(CardId::Slimed),
        "SLIMED (Status) must not count as colorless",
    );
    assert!(
        !is_visual_pool_colorless(CardId::Dazed),
        "DAZED (Status) must not count as colorless",
    );
}

#[test]
fn ancient_event_cards_are_visual_pool_colorless() {
    // These have Ancient rarity but live in EventCardPool (IsColorless) and
    // keep it as their visual pool, so a rarity test misses them.
    for id in [
        CardId::Apotheosis,
        CardId::Apparition,
        CardId::BrightestFlame,
        CardId::Maul,
        CardId::NeowsFury,
        CardId::Relax,
        CardId::Wish,
        CardId::Whistle,
        CardId::Abundance,
    ] {
        assert!(is_visual_pool_colorless(id), "{id:?} is an Event card");
    }
}

#[test]
fn status_rarity_token_cards_are_visual_pool_colorless() {
    // Status rarity, but listed by TokenCardPool (IsColorless), not by
    // StatusCardPool.
    for id in [
        CardId::Disintegration,
        CardId::MindRot,
        CardId::SlothStatus,
        CardId::WasteAway,
    ] {
        assert!(is_visual_pool_colorless(id), "{id:?} is a Token card");
    }
}

#[test]
fn event_cards_drawn_in_a_character_color_are_not_visual_pool_colorless() {
    // EventCardPool cards that override `VisualCardPool` with a character
    // pool (e.g. Caltrops.cs:17 SilentCardPool, HelloWorld.cs:12
    // DefectCardPool), in both builds.
    for id in [
        CardId::Caltrops,
        CardId::Clash,
        CardId::Distraction,
        CardId::DualWield,
        CardId::Entrench,
        CardId::HelloWorldCard,
        CardId::Outmaneuver,
        CardId::Rebound,
        CardId::RipAndTear,
        CardId::Stack,
    ] {
        assert!(
            !is_visual_pool_colorless(id),
            "{id:?} wears a character pool's color"
        );
    }
}
