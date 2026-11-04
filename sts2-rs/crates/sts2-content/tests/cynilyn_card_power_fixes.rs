//! Card / power / relic engine fixes first found by the cynilyn fork against
//! the real game, ported for BOTH builds (v0.107.1 and v0.111.0) unless the
//! two decomps differ. Each test cites the C# it pins.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, Power, PowerKind, PowerUid,
};

const V107: GameVersion = GameVersion::V0_107_1;
const V111: GameVersion = GameVersion::V0_111_0;
const BOTH: [GameVersion; 2] = [V107, V111];

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

/// Player with 5 energy in the play phase, no enemies yet.
fn fresh(version: GameVersion, character: CharacterId) -> CombatState {
    content_dispatch::ensure_installed();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        game_version: version,
        ..CombatState::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    state.player.energy = 5;
    state.player.max_energy = 3;
    state.player.character = Some(character);
    let _ = state.init_player_creature();
    state.phase = CombatPhase::PlayerPlayPhase;
    state
}

fn new_card(state: &mut CombatState, id: CardId, upgraded: bool) -> sts2_core::CardUid {
    let raw = u32::try_from(state.cards.len() + 1).expect("u32");
    let uid = sts2_core::CardUid::new(raw).expect("nz");
    let mut card = card_pod_for(id, upgraded, state.game_version).expect("spec");
    card.uid = uid;
    state.cards.push(card);
    uid
}

fn push_hand(state: &mut CombatState, id: CardId, upgraded: bool) -> u8 {
    let uid = new_card(state, id, upgraded);
    state.player.hand.push(uid);
    u8::try_from(state.player.hand.len() - 1).expect("u8")
}

fn hp_of(state: &CombatState, uid: CreatureUid) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| c.current_hp)
}

fn add_player_power_with_applier(
    state: &mut CombatState,
    raw: u32,
    kind: PowerKind,
    amount: i32,
    applier: CreatureUid,
) {
    let pid = NonZeroU32::new(raw).expect("nz");
    let mut p = Power::new(pid, kind, amount);
    p.applier = Some(applier);
    state.powers.push(p);
    state.player.powers.push(PowerUid(pid));
}

// ── 1. Thunderclap + Shrink whose applier dies in the same attack ──

/// Enemy B (1 HP, the Shrink applier) is hit first and dies; enemy A (100 HP)
/// is the second target. `CreatureCmd.Damage` resolves every target's
/// `ModifyDamage` before any death, and `ShrinkPower.cs:88` `AfterDeath`
/// removes the power only afterwards: A still takes floor(4 * 0.7) = 2.
#[test]
fn thunderclap_keeps_shrink_for_the_whole_attack() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let applier = add_enemy(&mut state, 2, 1);
        let other = add_enemy(&mut state, 3, 100);
        add_player_power_with_applier(&mut state, 900, PowerKind::Shrink, -1, applier);
        let i = push_hand(&mut state, CardId::Thunderclap, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert_eq!(hp_of(&state, applier), 0, "{v:?}");
        assert_eq!(hp_of(&state, other), 98, "{v:?}");
    }
}

// ── 2. LoseMaxHp below current HP deals damage first ──

fn add_relic(state: &mut CombatState, kind: sts2_core::RelicKind) {
    let raw = u32::try_from(state.relics.len() + 1).expect("u32");
    state.relics.push(sts2_core::Relic {
        id: NonZeroU32::new(raw).expect("nz"),
        kind,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
}

fn fill_draw(state: &mut CombatState, n: usize) {
    for _ in 0..n {
        let uid = new_card(state, CardId::StrikeIronclad, false);
        state.player.draw.push(uid);
    }
}

/// Full-HP player with Centennial Puzzle plays Brightest Flame (draw 2, lose
/// max HP). `CreatureCmd.LoseMaxHp` (CreatureCmd.cs:824, v0.111 :879): when
/// `newMaxHp < CurrentHp` it first deals `CurrentHp - newMaxHp`
/// Unblockable|Unpowered(|Move) damage, so Centennial Puzzle's after-damage
/// draw 3 fires: 2 + 3 cards in hand.
#[test]
fn lose_max_hp_below_current_hp_fires_after_damage_listeners() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let _ = add_enemy(&mut state, 2, 100);
        add_relic(&mut state, sts2_core::RelicKind::CentennialPuzzle);
        fill_draw(&mut state, 10);
        let i = push_hand(&mut state, CardId::BrightestFlame, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        let loss = if v >= V111 { 2 } else { 1 };
        assert_eq!(state.player.hand.len(), 5, "{v:?}");
        assert_eq!(state.player.max_hp, 80 - loss, "{v:?}");
        assert_eq!(state.player.current_hp, 80 - loss, "{v:?}");
    }
}

/// With current HP already under the new cap no damage is dealt: only the
/// cap shrinks, Centennial Puzzle stays quiet.
#[test]
fn lose_max_hp_above_current_hp_deals_no_damage() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let _ = add_enemy(&mut state, 2, 100);
        state.player.current_hp = 50;
        state.sync_player_creature();
        add_relic(&mut state, sts2_core::RelicKind::CentennialPuzzle);
        fill_draw(&mut state, 10);
        let i = push_hand(&mut state, CardId::BrightestFlame, false);
        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert_eq!(state.player.hand.len(), 2, "{v:?}");
        assert_eq!(state.player.current_hp, 50, "{v:?}");
    }
}

/// Enemies take the same path (nothing in `LoseMaxHp` is player-only). A
/// loss of the whole cap deals `CurrentHp - (MaxHp - amount)` >= CurrentHp
/// damage, which KILLS; the old clamp-only shrink left the creature at 1 HP
/// (`SetMaxHp` floors the cap at 1, CreatureCmd.cs:827).
#[test]
fn enemy_losing_its_whole_max_hp_dies() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let e = add_enemy(&mut state, 2, 10);
        let _keep_alive = add_enemy(&mut state, 3, 10);
        let _ = state.creature_lose_max_hp(e, 10);
        assert_eq!(hp_of(&state, e), 0, "{v:?}");
    }
}

// ── 3. Philosopher's Stone strengthens mid-combat summons ──

fn power_amount(state: &CombatState, uid: CreatureUid, kind: PowerKind) -> i32 {
    state
        .creatures
        .iter()
        .find(|c| c.id == uid.0)
        .map_or(0, |c| {
            c.powers
                .iter()
                .filter_map(|pu| state.powers.iter().find(|p| p.id == pu.0))
                .filter(|p| p.kind == kind)
                .map(|p| p.amount)
                .sum()
        })
}

/// `PhilosophersStone.cs:41` `AfterCreatureAddedToCombat`: an enemy added
/// mid-combat (`CreatureCmd.Add` -> `Hook.AfterCreatureAddedToCombat`) gains
/// the Stone's +1 Strength; the room-entry leg only covers the starters.
#[test]
fn philosophers_stone_strengthens_mid_combat_summons() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let _ = add_enemy(&mut state, 2, 50);
        add_relic(&mut state, sts2_core::RelicKind::PhilosophersStone);
        let summon = state.spawn_enemy(MonsterKind::Nibbit);
        assert_eq!(
            power_amount(&state, summon, PowerKind::Strength),
            1,
            "{v:?}"
        );

        // Without the Stone nothing is added.
        let mut plain = fresh(v, CharacterId::Ironclad);
        let _ = add_enemy(&mut plain, 2, 50);
        let summon = plain.spawn_enemy(MonsterKind::Nibbit);
        assert_eq!(
            power_amount(&plain, summon, PowerKind::Strength),
            0,
            "{v:?}"
        );
    }
}

// ── 4. Stomp / Pinpoint clones are not back-filled again ──

/// `Stomp.cs:53` / `Pinpoint.cs:43` `AfterCardEnteredCombat` return early on
/// `IsClone`: a copy of an already-discounted card keeps the cost it was
/// cloned with and is not shaved again for the plays finished this turn. A
/// freshly generated card (no bank) still gets the back-fill.
#[test]
fn stomp_and_pinpoint_clones_keep_their_cloned_cost() {
    use sts2_core::cards::{CardType, VarKey};
    let cases = [
        (
            CardId::Stomp,
            VarKey::StompCostReductionThisTurn,
            CardType::Attack,
        ),
        (
            CardId::Pinpoint,
            VarKey::PinpointCostReductionThisTurn,
            CardType::Skill,
        ),
    ];
    for v in BOTH {
        for (id, bank, ty) in cases {
            let mut state = fresh(v, CharacterId::Ironclad);
            let _ = add_enemy(&mut state, 2, 50);
            let orig = new_card(&mut state, id, false);
            let base = state.cards[orig.0.get() as usize - 1].cost;
            state.player.hand.push(orig);
            state.player.cards_played_this_turn[ty.index()] = 2;
            // The in-hand original was shaved by 2 by those plays.
            let c = &mut state.cards[orig.0.get() as usize - 1];
            c.cost = base - 2;
            c.set_combat_var(bank, 2);

            let clone = state.cards[orig.0.get() as usize - 1];
            let uid = sts2_core::card_effects::spawn_card_into_hand(&mut state, clone);
            let cost = state.cards[uid.0.get() as usize - 1].cost;
            assert_eq!(cost, base - 2, "{v:?} {id:?} clone");

            let fresh_card = card_pod_for(id, false, v).expect("spec");
            let uid = sts2_core::card_effects::spawn_card_into_hand(&mut state, fresh_card);
            let cost = state.cards[uid.0.get() as usize - 1].cost;
            assert_eq!(cost, base - 2, "{v:?} {id:?} fresh");
        }
    }
}

// ── 5. A Juggling copy of Stomp keeps its cloned cost (v0.111) ──

/// Stomp played as the turn's 3rd Attack under Juggling(1); returns the cost
/// of the Juggling copy in hand.
fn juggled_stomp_copy_cost(v: GameVersion) -> i8 {
    use sts2_core::cards::{CardType, VarKey};
    let mut state = fresh(v, CharacterId::Ironclad);
    let _ = add_enemy(&mut state, 2, 100);
    let p = state.player.creature_uid.expect("player");
    sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Juggling, 1);
    for pid in state.player_power_uids() {
        if let Some(pw) = state.powers.get_mut_by_id(pid.0) {
            if pw.kind == PowerKind::Juggling {
                pw.counter = 2;
            }
        }
    }
    state.player.cards_played_this_turn[CardType::Attack.index()] = 2;
    let i = push_hand(&mut state, CardId::Stomp, false);
    // Two Attacks already shaved it 3 -> 1.
    let c = &mut state.cards[0];
    c.cost = 1;
    c.set_combat_var(VarKey::StompCostReductionThisTurn, 2);
    let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
    assert_eq!(state.player.hand.len(), 1, "{v:?}: the Juggling copy");
    state.cards[state.player.hand[0].0.get() as usize - 1].cost
}

/// v0.111 `JugglingPower.cs:36` clones in `BeforeCardPlayed`; the hook's
/// listener list is snapshotted before it runs, so `Stomp.cs:62`
/// `BeforeCardPlayed` never visits the fresh copy: it keeps the cost it was
/// cloned with (1). v0.107.1 clones after the play, from the already-shaved
/// card (0).
#[test]
fn juggled_stomp_copy_keeps_its_cloned_cost() {
    assert_eq!(juggled_stomp_copy_cost(V111), 1);
    assert_eq!(juggled_stomp_copy_cost(V107), 0);
}

// ── 6. Dampen restores only the cards it downgraded ──

/// `DampenPower.cs:29` records the exact card instances it downgraded and
/// `AfterRemoved` re-upgrades only those. Anger's copy (made while the
/// original is downgraded) is not in that list and stays unupgraded.
#[test]
fn dampen_restores_the_original_but_not_its_clone() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let e = add_enemy(&mut state, 2, 100);
        let original = new_card(&mut state, CardId::Anger, true);
        state.player.hand.push(original);
        let oi = original.0.get() as usize - 1;
        assert!(state.cards[oi].upgraded);
        sts2_core::monster_power_effects::install_dampen_power_handlers();
        let player = state.player.creature_uid.expect("player");
        let _ = sts2_core::monster_power_effects::dispatch_after_applied_for_power(
            &mut state,
            player,
            PowerKind::Dampen,
            Some(e),
        );
        assert!(!state.cards[oi].upgraded, "{v:?}");

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));
        let clone = state
            .cards
            .iter()
            .find(|c| c.card_id == CardId::Anger && c.uid != original)
            .map(|c| c.uid)
            .expect("Anger clone");
        let _ = sts2_core::monster_power_effects::dispatch_after_removed_for_power(
            &mut state,
            player,
            PowerKind::Dampen,
        );
        assert!(state.cards[oi].upgraded, "{v:?}");
        assert!(!state.cards[clone.0.get() as usize - 1].upgraded, "{v:?}");
    }
}

/// `EchoingSlash.cs:31` passes `base.DynamicVars.Damage` to every wave, so a
/// wave that kills the Dampen caster re-upgrades the card (`DampenPower.cs:41`
/// `AfterDeath`) and the next wave hits for the upgraded damage. Rust read the
/// damage once before the loop: act23 Ironclad-3003 (act-2 checkpoint,
/// branch 3) left the survivor at 39 where C# had 33.
#[test]
fn echoing_slash_rewave_after_killing_the_dampen_caster_is_upgraded() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let caster = add_enemy(&mut state, 2, 5);
        let survivor = add_enemy(&mut state, 3, 100);
        let slash = new_card(&mut state, CardId::EchoingSlash, true);
        state.player.hand.push(slash);
        let si = slash.0.get() as usize - 1;
        let upgraded_damage = state.cards[si].base_damage;
        sts2_core::monster_power_effects::install_dampen_power_handlers();
        let player = state.player.creature_uid.expect("player");
        // As MagiKnight's DAMPEN move does: the power instance records its
        // caster, so the caster's death removes it and re-upgrades the cards.
        sts2_core::card_effects::apply_power_to_creature(&mut state, player, PowerKind::Dampen, 1);
        let pid = state
            .powers
            .iter()
            .find(|p| p.kind == PowerKind::Dampen)
            .map(|p| p.id)
            .expect("Dampen applied");
        state.powers.get_mut_by_id(pid).expect("Dampen").applier = Some(caster);
        let _ = sts2_core::monster_power_effects::dispatch_after_applied_for_power(
            &mut state,
            player,
            PowerKind::Dampen,
            Some(caster),
        );
        let dampened_damage = state.cards[si].base_damage;
        assert!(
            dampened_damage < upgraded_damage,
            "{v:?}: Dampen downgraded"
        );

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, None);
        let hp = state
            .creatures
            .iter()
            .find(|c| c.id == survivor.0)
            .expect("survivor")
            .current_hp;
        assert_eq!(
            100 - hp,
            dampened_damage + upgraded_damage,
            "{v:?}: wave 1 dampened, wave 2 re-upgraded"
        );
    }
}

// ── 8. Unceasing Top after a top-level auto-play empties the hand ──

/// Hand = one Defend; draw = [Setup Strike, Defend]; Hellraiser + Unceasing
/// Top. Playing the Defend empties the hand: Top draws Setup Strike, which
/// Hellraiser auto-plays. That auto-play's `OnPlayWrapper` ends with
/// `CheckForEmptyHand` (CardModel.cs:1993, v0.111 :2006) outside any card
/// effect (the outer play already left its effect body), so the empty hand
/// fires Top again and the second Defend is drawn.
#[test]
fn unceasing_top_redraws_after_a_hellraiser_auto_play_empties_the_hand() {
    for v in BOTH {
        let mut state = fresh(v, CharacterId::Ironclad);
        let _ = add_enemy(&mut state, 2, 100);
        add_relic(&mut state, sts2_core::RelicKind::UnceasingTop);
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Hellraiser, 1);
        let setup = new_card(&mut state, CardId::SetupStrikeCard, false);
        let defend = new_card(&mut state, CardId::DefendIronclad, false);
        state.player.draw.push(setup);
        state.player.draw.push(defend);
        let i = push_hand(&mut state, CardId::DefendIronclad, false);

        let _ = sts2_core::combat_loop::play_card(&mut state, i, None);
        assert!(!state.player.hand.contains(&setup), "{v:?}: auto-played");
        assert_eq!(state.player.hand.as_slice(), &[defend], "{v:?}");
        assert!(state.player.draw.is_empty(), "{v:?}");
    }
}

// ── 6. A Calamity-generated Stomp counts the Attack that generated it ──

/// C# `CardModel.OnPlayWrapper` records `History.CardPlayFinished` BEFORE
/// `Hook.AfterCardPlayed` (CardModel.cs:1959-1966, v0.107.1 :1957-1960), so
/// a Stomp that `CalamityPower.AfterCardPlayed` conjures sees the Attack being
/// played in `Stomp.AfterCardEnteredCombat`'s count: the first Attack of the
/// turn yields a Stomp at 3 - 1 = 2. Rust tallied the per-type bucket after the
/// dispatch, so the Stomp cost 3 (sts2-sim-2a, Ironclad v0.111 round).
#[test]
fn calamity_stomp_counts_the_attack_that_generated_it() {
    for v in BOTH {
        let mut found = false;
        for seed in 0..4000_i32 {
            let mut state = fresh(v, CharacterId::Ironclad);
            let _ = add_enemy(&mut state, 2, 500);
            state.run_rng.combat_card_generation = sts2_core::dotnet_rng::Rng::new(seed);
            let p = state.player.creature_uid.expect("player");
            sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Calamity, 1);
            let i = push_hand(&mut state, CardId::StrikeIronclad, false);
            let before: Vec<_> = state.player.hand.to_vec();
            let _ = sts2_core::combat_loop::play_card(&mut state, i, Some(0));
            let Some(stomp) = state.player.hand.iter().copied().find(|u| {
                !before.contains(u) && state.cards[u.0.get() as usize - 1].card_id == CardId::Stomp
            }) else {
                continue;
            };
            let cost = state.cards[stomp.0.get() as usize - 1].cost;
            assert_eq!(cost, 2, "{v:?} seed {seed}: Stomp 3 - 1 Attack played");
            found = true;
            break;
        }
        assert!(found, "{v:?}: no seed generated a Stomp");
    }
}
