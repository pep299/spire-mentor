//! Dark Embrace's turn-end draw runs at its own place in the player's power
//! walk.
//!
//! C# `DarkEmbracePower.AfterSideTurnEnd` (decomp v0.111.0
//! DarkEmbracePower.cs:52-57, same body in v0.107.1) draws
//! `Amount * etherealCount` and is one listener of `Hook.AfterSideTurnEnd`,
//! which awaits each listener in turn over the creature's powers in
//! application order. So a power applied AFTER Dark Embrace still acts during
//! the draw: `NoDrawPower.ShouldDraw` vetoes it (NoDrawPower.cs, removed only
//! by its own later `AfterSideTurnEnd`), and a `TemporaryStrengthPower`
//! (TemporaryStrengthPower.cs:147-155) reverts whatever a Hellraiser-autoplayed
//! Setup Strike stacked onto it. A power applied BEFORE Dark Embrace has
//! already run. Rust used to apply every power's turn-end effect first and draw
//! afterwards, which is the second order for every power.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::hooks::{dispatch_after_turn_end, TurnSideCtx};
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid, Side,
};

const VERSIONS: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

fn fresh(version: GameVersion) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: 500,
        max_hp: 500,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(id));
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.character = Some(CharacterId::Ironclad);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerEndTurnPhaseTwo;
    state.side = Side::Player;
    state
}

fn push_draw(state: &mut CombatState, id: CardId) {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, false, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    state.player.draw.push(uid);
}

/// Adds a player power; `raw` is its `PowerUid`, so ascending `raw` in push
/// order is application order on both builds' walks.
fn add_player_power(state: &mut CombatState, raw: u32, kind: PowerKind, amount: i32) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(Power::new(pid, kind, amount));
    state.player.powers.push(PowerUid(pid));
}

fn set_ethereal_count(state: &mut CombatState, count: i32) {
    let de = state
        .powers
        .iter_mut()
        .find(|p| p.kind == PowerKind::DarkEmbrace)
        .expect("Dark Embrace");
    de.counter = count;
}

fn player_turn_end(state: &mut CombatState) {
    dispatch_after_turn_end(state, &TurnSideCtx { side: Side::Player });
}

/// Ironclad 4142 act 2, step 963: Dark Embrace applied before Battle Trance's
/// NoDraw. C# drew nothing for the 9 Ethereal exhausts; Rust drew 9.
#[test]
fn no_draw_applied_after_dark_embrace_vetoes_its_turn_end_draw() {
    for version in VERSIONS {
        let mut state = fresh(version);
        add_player_power(&mut state, 50, PowerKind::DarkEmbrace, 1);
        add_player_power(&mut state, 51, PowerKind::NoDraw, 1);
        set_ethereal_count(&mut state, 2);
        for _ in 0..3 {
            push_draw(&mut state, CardId::DefendIronclad);
        }

        player_turn_end(&mut state);

        assert!(
            state.player.hand.is_empty(),
            "{version:?}: NoDraw is still up when Dark Embrace draws"
        );
        assert_eq!(
            state.player_power_amount(PowerKind::NoDraw),
            0,
            "{version:?}"
        );
        let de = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::DarkEmbrace)
            .expect("Dark Embrace stays");
        assert_eq!(de.counter, 0, "{version:?}: etherealCount resets anyway");
    }
}

#[test]
fn no_draw_applied_before_dark_embrace_is_gone_when_it_draws() {
    for version in VERSIONS {
        let mut state = fresh(version);
        add_player_power(&mut state, 50, PowerKind::NoDraw, 1);
        add_player_power(&mut state, 51, PowerKind::DarkEmbrace, 1);
        set_ethereal_count(&mut state, 2);
        for _ in 0..3 {
            push_draw(&mut state, CardId::DefendIronclad);
        }

        player_turn_end(&mut state);

        assert_eq!(
            state.player.hand.len(),
            2,
            "{version:?}: 1 * etherealCount 2"
        );
    }
}

/// Dark Embrace, Hellraiser and an active Setup Strike (+3 temporary
/// Strength), then Dark Embrace draws a Setup Strike that Hellraiser
/// auto-plays. Returns (damage dealt, Strength left, SetupStrikePower left).
fn setup_strike_drawn_at_turn_end(
    version: GameVersion,
    dark_embrace_first: bool,
) -> (i32, i32, i32) {
    let mut state = fresh(version);
    let (de_raw, setup_raw) = if dark_embrace_first {
        (50, 52)
    } else {
        (52, 50)
    };
    add_player_power(&mut state, 51, PowerKind::Hellraiser, 1);
    add_player_power(&mut state, setup_raw, PowerKind::SetupStrike, 3);
    add_player_power(&mut state, 53, PowerKind::Strength, 3);
    add_player_power(&mut state, de_raw, PowerKind::DarkEmbrace, 1);
    // `state.player.powers` must list them in application (uid) order.
    state.player.powers.sort_unstable_by_key(|u| u.0);
    if let Some(c) = state
        .player
        .creature_uid
        .and_then(|uid| state.creatures.iter_mut().find(|c| c.id == uid.0))
    {
        c.powers.sort_unstable_by_key(|u| u.0);
    }
    set_ethereal_count(&mut state, 1);
    push_draw(&mut state, CardId::SetupStrikeCard);

    player_turn_end(&mut state);

    (
        500 - state.creatures[0].current_hp,
        state.player_power_amount(PowerKind::Strength),
        state.player_power_amount(PowerKind::SetupStrike),
    )
}

/// Ironclad 4011 act 2, step 445 (Entomancer, cs hp 27 vs rs 30): with Dark
/// Embrace first, the auto-played Setup Strike hits with the 3 temporary
/// Strength still up and stacks SetupStrikePower to 6, which then reverts
/// all 6.
#[test]
fn setup_strike_applied_after_dark_embrace_reverts_the_autoplayed_stack() {
    for version in VERSIONS {
        let base = if version >= GameVersion::V0_111_0 {
            3
        } else {
            2
        };
        let (dealt, strength, setup) = setup_strike_drawn_at_turn_end(version, true);
        let damage = card_pod_for(CardId::SetupStrikeCard, false, version)
            .expect("spec")
            .base_damage;
        assert_eq!(dealt, damage + 3, "{version:?}: hit with the 3 Strength");
        assert_eq!(strength, 0, "{version:?}: 3 + {base} reverted together");
        assert_eq!(setup, 0, "{version:?}: SetupStrikePower removed");
    }
}

/// The reverse order keeps the old behaviour: the revert runs first, and the
/// SetupStrikePower the auto-play creates is new, so this walk does not visit
/// it and its Strength survives the turn end.
#[test]
fn setup_strike_applied_before_dark_embrace_reverts_first() {
    for version in VERSIONS {
        let base = if version >= GameVersion::V0_111_0 {
            3
        } else {
            2
        };
        let (dealt, strength, setup) = setup_strike_drawn_at_turn_end(version, false);
        let damage = card_pod_for(CardId::SetupStrikeCard, false, version)
            .expect("spec")
            .base_damage;
        assert_eq!(dealt, damage, "{version:?}: hit with no Strength");
        assert_eq!(strength, base, "{version:?}: the fresh stack survives");
        assert_eq!(setup, base, "{version:?}");
    }
}
