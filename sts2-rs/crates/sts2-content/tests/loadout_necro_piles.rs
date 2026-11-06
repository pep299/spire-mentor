//! Necrobinder random-loadout parity: card pile routing (group `piles`).
//!
//! Source of truth: `sts2-wiki/decompiled/v0.111.0/`.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind,
};

const V111: GameVersion = GameVersion::V0_111_0;

fn fresh() -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: V111,
        ..CombatState::default()
    };
    let id = NonZeroU32::new(1).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: 100,
        max_hp: 100,
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
    state.player.energy = 5;
    state.player.max_energy = 3;
    state.player.character = Some(CharacterId::Necrobinder);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
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

fn add_relic(state: &mut CombatState, raw: u32, kind: sts2_core::RelicKind) {
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(raw).expect("nz"),
        kind,
        counter: kind.initial_counter(),
        is_wax: false,
        is_melted: false,
    });
}

/// An auto-played Exhaust card keeps the result pile C# fixed before its
/// `OnPlay` (`CardModel.OnPlayWrapper` → `GetResultLocationForCardPlay`,
/// v0.111.0 CardModel.cs:1880, :2084-2096). Razor Tooth's `AfterCardPlayed`
/// upgrade then strips Graveblast's Exhaust (`Graveblast.OnUpgrade`), but the
/// card still exhausts (loadout Necrobinder 1059: Catastrophe auto-plays it).
#[test]
fn auto_played_graveblast_exhausts_although_razor_tooth_upgrades_it() {
    let mut state = fresh();
    add_relic(&mut state, 1, sts2_core::RelicKind::RazorTooth);
    let uid = new_card(&mut state, CardId::Graveblast);
    state.player.draw.push(uid);
    assert!(sts2_core::combat_loop::auto_play_specific_card(
        &mut state, uid, false
    ));
    let card = &state.cards[uid.0.get() as usize - 1];
    assert!(card.upgraded, "Razor Tooth upgraded the played Graveblast");
    assert!(
        state.player.exhaust.contains(&uid),
        "exhausted per the pre-OnPlay result pile; discard={:?} exhaust={:?}",
        state.player.discard,
        state.player.exhaust
    );
    assert!(!state.player.discard.contains(&uid));
}

/// The hand-played twin: the play-start card snapshot already decides it.
#[test]
fn hand_played_graveblast_exhausts_although_razor_tooth_upgrades_it() {
    let mut state = fresh();
    add_relic(&mut state, 1, sts2_core::RelicKind::RazorTooth);
    let uid = new_card(&mut state, CardId::Graveblast);
    state.player.hand.push(uid);
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    assert!(state.cards[uid.0.get() as usize - 1].upgraded);
    assert!(state.player.exhaust.contains(&uid));
}

fn add_enemy(state: &mut CombatState, raw: u32, hp: i32) -> CreatureUid {
    let id = NonZeroU32::new(raw).expect("nz");
    state.creatures.push(Creature {
        id,
        current_hp: hp,
        max_hp: hp,
        block: 0,
        powers: SmallVec::new(),
        species: Some(MonsterKind::Nibbit),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId("Nibbit"),
        ai: None,
    });
    state.enemy_uids.push(CreatureUid(id));
    CreatureUid(id)
}

fn add_creature_power(
    state: &mut CombatState,
    uid: CreatureUid,
    raw: u32,
    kind: sts2_core::PowerKind,
    amount: i32,
) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(sts2_core::Power::new(pid, kind, amount));
    state
        .creatures
        .iter_mut()
        .find(|c| c.id == uid.0)
        .expect("creature")
        .powers
        .push(sts2_core::PowerUid(pid));
}

fn discard_pile_of(state: &mut CombatState, n: usize) {
    let ids = [
        CardId::StrikeNecrobinder,
        CardId::DefendNecrobinder,
        CardId::Bodyguard,
        CardId::Unleash,
        CardId::Bury,
        CardId::Reave,
        CardId::Sow,
    ];
    for i in 0..n {
        let uid = new_card(state, ids[i % ids.len()]);
        state.player.discard.push(uid);
    }
}

fn add_player_power(state: &mut CombatState, raw: u32, kind: sts2_core::PowerKind, amount: i32) {
    let pid = NonZeroU32::new(raw).expect("nz");
    state.powers.push(sts2_core::Power::new(pid, kind, amount));
    state.player.powers.push(sts2_core::PowerUid(pid));
}

/// Doom kills an enemy at the enemy side's turn end; Gremlin Horn's draw
/// reshuffles an empty draw pile and Stratagem's grid opens. C#
/// `Hook.AfterDeath` (v0.111.0 Hook.cs:606-620) only awaits a listener to its
/// first pause, so the enemy turn and the next player turn start (hand draw,
/// Mayhem) run first and the pick lands in the play phase — into a full hand
/// it goes to the discard (loadout Necrobinder 1070).
#[test]
fn an_after_death_stratagem_pick_waits_for_the_play_phase() {
    let mut state = fresh();
    let _second = add_enemy(&mut state, 100, 300);
    add_relic(&mut state, 1, sts2_core::RelicKind::GremlinHorn);
    add_player_power(&mut state, 50, sts2_core::PowerKind::Stratagem, 2);
    let doomed = state.enemy_uids[0];
    add_creature_power(&mut state, doomed, 51, sts2_core::PowerKind::Doom, 200);
    discard_pile_of(&mut state, 12);
    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert_eq!(
        state.phase,
        CombatPhase::PlayerPlayPhase,
        "the next turn started first"
    );
    assert_eq!(
        state.player.hand.len(),
        5,
        "the turn's hand draw ran before the pick"
    );
    let pc = state
        .pending_choice
        .as_ref()
        .expect("the Stratagem pick is open");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    // The grid was built at the reshuffle; pick two cards the hand draw left.
    let picks: Vec<u8> = pc
        .options
        .iter()
        .enumerate()
        .filter(|(_, o)| o.card_uid.is_some_and(|u| state.player.draw.contains(&u)))
        .map(|(i, _)| u8::try_from(i).expect("u8"))
        .take(2)
        .collect();
    let _ = sts2_core::resolve_choice_atomic_apply(&mut state, &picks);
    assert!(state.pending_choice.is_none());
    // The two picks, then Gremlin Horn's own card.
    assert_eq!(state.player.hand.len(), 8, "hand {:?}", state.player.hand);
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
}

/// A select that blocks the enemy side (an enemy hit's Centennial Puzzle draw
/// reshuffling into Stratagem, GH #1129) is captured, and the replay drivers
/// answer a two-card pick in one atomic set
/// (`RunManager::resolve_combat_choice_selection`). That must rerun the
/// captured enemy side, not resolve the pick on the captured state and leave
/// the combat parked in the enemy turn.
#[test]
fn an_atomic_answer_to_a_captured_enemy_side_select_resumes_the_turn() {
    let mut state = fresh();
    state.creatures[0].intent = Some(sts2_core::monsters::Intent::Attack { damage: 7 });
    add_relic(&mut state, 1, sts2_core::RelicKind::CentennialPuzzle);
    add_player_power(&mut state, 50, sts2_core::PowerKind::Stratagem, 2);
    discard_pile_of(&mut state, 12);
    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    let pc = state
        .pending_choice
        .as_ref()
        .expect("the Stratagem pick is captured");
    assert_eq!(pc.kind, sts2_core::ChoiceKind::AddFromDrawToHand);
    assert_ne!(
        state.phase,
        CombatPhase::PlayerPlayPhase,
        "the enemy side waits"
    );
    let _ = sts2_core::resolve_choice_atomic_apply(&mut state, &[0, 1]);
    assert!(state.pending_choice.is_none());
    assert_eq!(
        state.phase,
        CombatPhase::PlayerPlayPhase,
        "the next turn started"
    );
}

/// Soul's Power's deck pick lists the cards with the build's own Exhaust
/// keyword (`SoulsPower.CanEnchant`, v0.111.0 SoulsPower.cs:12-21). A deck
/// card keeps its v0.107.1 POD until it enters a combat, and v0.111.0 drops
/// Times Up's and Legion of Bone's Exhaust and gives Eidolon one: C# offered
/// [Eidolon, Wisp] (loadout Necrobinder 1172 / 1149 / 1075,
/// GraveOfTheForgotten).
#[test]
fn souls_power_pick_reads_the_builds_exhaust_keyword() {
    use sts2_core::run_manager::RunManager;
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut rm = RunManager::new(0, CharacterId::Necrobinder).with_game_version(V111);
    rm.run_state.master_deck.clear();
    for id in [
        CardId::TimesUp,
        CardId::Eidolon,
        CardId::Wisp,
        CardId::LegionOfBone,
    ] {
        let card = sts2_content::cards::spec_index::card_pod_from_id(id, false).expect("pod");
        rm.run_state.master_deck.push(card);
    }
    rm.run_state
        .pending_deck_enchants
        .push(sts2_core::runstate::PendingDeckEnchant {
            enchantment: sts2_core::cards::EnchantmentId::SoulsPower,
            amount: 1,
            type_restriction: None,
            secondary_type: None,
            optional: false,
        });
    assert_eq!(rm.enchant_eligible_deck_indices(), vec![1, 2]);
}

/// Sculpting Strike's pick skips a card already Ethereal from
/// `CardCmd.ApplyKeyword` — Music Box's clone (MusicBox.cs:78-79). C# filters
/// on local keywords (`SculptingStrike.cs:34`), so with [Fisticuffs (Music
/// Box clone), Scourge, Sow] only Scourge and Sow are offered (loadout
/// Necrobinder 1040).
#[test]
fn sculpting_strike_skips_a_music_box_ethereal_clone() {
    let mut state = fresh();
    let clone = new_card(&mut state, CardId::Fisticuffs);
    state.cards[clone.0.get() as usize - 1]
        .set_combat_var(sts2_core::cards::VarKey::EtherealThisCombat, 1);
    state.player.hand.push(clone);
    for id in [CardId::Scourge, CardId::Sow] {
        let uid = new_card(&mut state, id);
        state.player.hand.push(uid);
    }
    let ss = new_card(&mut state, CardId::SculptingStrike);
    state.player.hand.push(ss);
    let _ = sts2_core::combat_loop::play_card(&mut state, 3, Some(0));
    let pc = state
        .pending_choice
        .as_ref()
        .expect("the Ethereal pick opens");
    let offered: Vec<CardId> = pc.options.iter().map(|o| o.card_id).collect();
    assert_eq!(offered, vec![CardId::Scourge, CardId::Sow]);
}

/// `DoubleDamagePower.ModifyDamageMultiplicative` (v0.107.1
/// DoubleDamagePower.cs) doubles a powered card hit whose dealer is the owner
/// OR one of its pets, so the Osty's Fetch hit doubles too (loadout
/// Necrobinder 1019: Fetch+ killed a 17-HP Zapbot in C#, Rust left it at 8).
#[test]
fn double_damage_doubles_the_ostys_fetch() {
    let mut state = fresh();
    state.summon_osty(10).expect("osty summoned");
    let pid = NonZeroU32::new(60).expect("nz");
    state.powers.push(sts2_core::Power::new(
        pid,
        sts2_core::PowerKind::DoubleDamage,
        1,
    ));
    state.player.powers.push(sts2_core::PowerUid(pid));
    let fetch = new_card(&mut state, CardId::Fetch);
    state.player.hand.push(fetch);
    let before = state.creatures[0].current_hp;
    let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
    // Fetch: Osty deals 3, doubled.
    assert_eq!(before - state.creatures[0].current_hp, 6);
}

/// Mayhem 2 auto-plays the top two draw cards at turn start, both collected
/// onto Play first (`CardPileCmd.AutoPlayFromDrawPile`). Nostalgia 1 counts
/// `History.CardPlaysStarted`, so the first Scourge (no play started yet) goes
/// to the draw top, and the second Scourge's draw takes it back into hand.
/// Rust counted the not-yet-played sibling as started and discarded both
/// (loadout Necrobinder 1109).
#[test]
fn mayhem_batch_sibling_does_not_cost_the_first_card_its_nostalgia_redirect() {
    let mut state = fresh();
    for (raw, kind, amount) in [
        (70, sts2_core::PowerKind::Mayhem, 2),
        (71, sts2_core::PowerKind::Nostalgia, 1),
    ] {
        let pid = NonZeroU32::new(raw).expect("nz");
        state.powers.push(sts2_core::Power::new(pid, kind, amount));
        state.player.powers.push(sts2_core::PowerUid(pid));
    }
    for id in [
        CardId::StrikeNecrobinder,
        CardId::StrikeNecrobinder,
        CardId::DefendNecrobinder,
        CardId::DefendNecrobinder,
        CardId::Bodyguard,
        CardId::Scourge,
        CardId::Scourge,
        CardId::Unleash,
        CardId::Bury,
    ] {
        let uid = new_card(&mut state, id);
        state.player.draw.push(uid);
    }
    let _ = sts2_core::apply(&mut state, sts2_core::Action::EndTurn);
    assert_eq!(state.phase, CombatPhase::PlayerPlayPhase);
    let scourges_in = |pile: &[sts2_core::CardUid]| {
        pile.iter()
            .filter(|u| state.cards[u.0.get() as usize - 1].card_id == CardId::Scourge)
            .count()
    };
    assert_eq!(
        scourges_in(&state.player.discard),
        1,
        "only the second Scourge discards"
    );
    assert_eq!(
        scourges_in(&state.player.hand),
        1,
        "the first came back via the draw top"
    );
}
