//! GH #1131: a select after a parked draw prompts the player.
//!
//! C# awaits `CardPileCmd.Draw` inside `OnPlay`, and with it the Stratagem
//! grid a reshuffle opens, then goes on with the rest of `OnPlay`: Thinking
//! Ahead's `CardSelectCmd.FromHand` put-back (ThinkingAhead.cs). Both prompt.
//!
//! Rust parks the draw on the auto-play's resume (`inner_draw`). After the
//! pick it used to finish that draw with every select drained inline, so the
//! put-back took option 0.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    ChoiceKind, CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory,
    MonsterId, MonsterKind, Power, PowerKind, PowerUid,
};

fn fresh() -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: GameVersion::V0_107_1,
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
    state.player.energy = 3;
    state.player.max_energy = 3;
    state.player.character = Some(CharacterId::Ironclad);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    let pid = NonZeroU32::new(50).expect("nz");
    state.powers.push(Power::new(pid, PowerKind::Stratagem, 1));
    state.player.powers.push(PowerUid(pid));
    state
}

fn new_card(state: &mut CombatState, id: CardId) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, false, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn resolve(state: &mut CombatState, idx: u8) {
    let _ = sts2_core::apply(
        state,
        sts2_core::Action::ResolveChoice {
            selection: smallvec::smallvec![idx],
        },
    );
}

fn assert_at_rest(state: &CombatState) {
    assert!(state.pending_choice.is_none());
    assert!(state.pending_collect_resume.is_none());
    assert!(state.pending_autoplay_resume.is_none());
    assert!(state.pending_play.is_none());
    assert!(
        state.player.play.is_empty(),
        "play: {:?}",
        state.player.play
    );
}

/// Havoc auto-plays Thinking Ahead off the draw pile. Its draw reshuffles
/// and prompts, and after the pick its put-back prompts too. Thinking Ahead's
/// and Havoc's tails wait for both.
#[test]
fn auto_played_put_back_after_parked_draw_prompts() {
    let mut state = fresh();
    for id in [CardId::StrikeIronclad, CardId::Bash, CardId::IronWave] {
        let uid = new_card(&mut state, id);
        state.player.discard.push(uid);
    }
    let thinking = new_card(&mut state, CardId::ThinkingAhead);
    state.player.draw.push(thinking);
    let havoc = new_card(&mut state, CardId::Havoc);
    state.player.hand.push(havoc);
    for _ in 0..2 {
        let uid = new_card(&mut state, CardId::DefendIronclad);
        state.player.hand.push(uid);
    }

    let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("Stratagem grid prompts");
    assert_eq!(pc.kind, ChoiceKind::AddFromDrawToHand);

    resolve(&mut state, 0);
    let pc = state.pending_choice.as_ref().expect("the put-back prompts");
    assert_eq!(pc.kind, ChoiceKind::PutBack);
    assert_eq!(pc.options.len(), 5, "2 Defends, the pick, 2 drawn");
    assert!(
        state.player.play.contains(&thinking),
        "Thinking Ahead waits"
    );

    resolve(&mut state, 0);
    assert_at_rest(&state);
    assert_eq!(state.player.hand.len(), 4);
    assert_eq!(state.player.draw.len(), 1, "the put-back card");
    assert!(state.player.exhaust.contains(&thinking));
    assert!(state.player.discard.contains(&havoc));
}
