//! GH #1054 (Ironclad 2185 act 3, 2003 act 2; v0.111.0): Havoc auto-plays the
//! Nostalgia power card off the draw top. C# fixes the played card's result
//! location at the TOP of `OnPlayWrapper` — `GetResultLocationForCardPlay` +
//! `Hook.ModifyCardPlayResultLocation` (v0.111.0 CardModel.cs:1881-1882;
//! v0.107.1 :1891), before `CardPlayStarted` / `OnPlay`. When Havoc's location
//! is decided there is no `NostalgiaPower` yet, so Havoc goes to discard and the
//! fresh Nostalgia(1) is left for the next Attack/Skill. Rust read the Nostalgia
//! amount live after `OnPlay` and sent Havoc to the draw top. The C# is the same
//! in both builds.

use sts2_content::cards::dispatch as content_dispatch;
use sts2_core::cards::CardId;
use sts2_core::{CardUid, CharacterId, CombatPhase, CombatState, GameVersion, MonsterKind, Side};

const VERSIONS: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

fn fresh(version: GameVersion) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    sts2_core::register_card_factory(sts2_content::cards::spec_index::card_pod_from_id);
    let mut state = CombatState {
        game_version: version,
        rng: sts2_core::dotnet_rng::Rng::new(7),
        run_rng: sts2_core::run_rng_set::RunRngSet::new(7),
        ..CombatState::default()
    };
    state.player.character = Some(CharacterId::Ironclad);
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 3;
    state.player.max_energy = 3;
    let _ = state.init_player_creature();
    let _ = state.spawn_enemy(MonsterKind::Nibbit);
    state.phase = CombatPhase::PlayerPlayPhase;
    state.side = Side::Player;
    state
}

fn mint(state: &mut CombatState, id: CardId) -> CardUid {
    let mut card = sts2_core::lookup_card_factory()
        .and_then(|f| f(id, false))
        .expect("card factory has the card");
    let uid = CardUid::new(u32::try_from(state.cards.len() + 1).expect("fits")).expect("nz");
    card.uid = uid;
    state.cards.push(card);
    uid
}

#[test]
#[serial_test::serial(global_registry)]
fn havoc_that_auto_plays_nostalgia_still_goes_to_discard() {
    for version in VERSIONS {
        let mut state = fresh(version);
        let nostalgia = mint(&mut state, CardId::NostalgiaCard);
        let filler = mint(&mut state, CardId::DefendIronclad);
        state.player.draw.push(nostalgia);
        state.player.draw.push(filler);
        let havoc = mint(&mut state, CardId::Havoc);
        state.player.hand.push(havoc);

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);

        assert_eq!(
            state.player_power_amount(sts2_core::PowerKind::Nostalgia),
            1,
            "{version:?}: the auto-played Nostalgia card applied its power"
        );
        assert!(
            state.player.discard.contains(&havoc),
            "{version:?}: Havoc's result location was fixed before its OnPlay \
             created Nostalgia, so it goes to discard (draw {:?}, discard {:?})",
            state.player.draw,
            state.player.discard
        );
        assert_eq!(
            state.player.draw.as_slice(),
            &[filler],
            "{version:?}: Havoc is not put back on the draw top"
        );
    }
}
