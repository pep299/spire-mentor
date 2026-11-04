//! Order of the `Hook.AfterDeath` listener walk on an enemy kill.
//!
//! C# `CreatureCmd.KillWithoutCheckingWinCondition` awaits `Hook.AfterDeath`
//! (v0.111.0 `CreatureCmd.cs:552`, v0.107.1 `:519`), one sequential walk over
//! `CombatState.IterateHookListeners` (`Hook.cs:606-619`) in creature order:
//! allies first (player powers, relics, ...), then the enemies (their powers,
//! then the monster) (`CombatState.cs:115-170`). So a player relic's
//! `AfterDeath` and everything it awaits inline runs before the dying
//! enemy's own power reacts.

use smallvec::SmallVec;
use std::num::NonZeroU32;

use sts2_content::cards::dispatch as content_dispatch;
use sts2_content::cards::v0111::card_pod_for;
use sts2_core::cards::CardId;
use sts2_core::potions::CharacterId;
use sts2_core::{
    CombatPhase, CombatState, Creature, CreatureUid, GameVersion, IntentHistory, MonsterId,
    MonsterKind, PowerKind,
};

const BOTH: [GameVersion; 2] = [GameVersion::V0_107_1, GameVersion::V0_111_0];

fn fresh(version: GameVersion) -> CombatState {
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
    state.player.character = Some(CharacterId::Ironclad);
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

/// A Strike kills the only Axebot. Gremlin Horn's `AfterDeath` draws a Strike
/// (`GremlinHorn.cs:24-31`), which Hellraiser auto-plays inline. The Axebot's
/// own `StockPower.AfterDeath` (`StockPower.cs:17-28`) has not spawned the
/// replacement yet, so `HittableEnemies` is empty: `Rng.NextItem` returns
/// null without rolling and the Strike hits nothing (`CardCmd.cs:74-84`).
/// The replacement then spawns untouched. act23 Ironclad:2169 step 864.
#[test]
fn gremlin_horn_draw_resolves_before_stock_respawns_the_axebot() {
    for v in BOTH {
        let mut state = fresh(v);
        let id = NonZeroU32::new(2).expect("nz");
        state.creatures.push(Creature {
            id,
            current_hp: 6,
            max_hp: 40,
            block: 0,
            powers: SmallVec::new(),
            species: Some(MonsterKind::Axebot),
            intent: None,
            intent_history: IntentHistory::default(),
            monster_id: MonsterId("Axebot"),
            ai: None,
        });
        let bot = CreatureUid(id);
        state.enemy_uids.push(bot);
        sts2_core::card_effects::apply_power_to_creature(&mut state, bot, PowerKind::Stock, 2);
        add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Hellraiser, 1);
        let drawn = new_card(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(drawn);
        let strike = new_card(&mut state, CardId::StrikeIronclad);
        state.player.hand.push(strike);
        let targets_before = state.run_rng.combat_targets.counter();

        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(0));

        let spawned: Vec<&Creature> = state
            .creatures
            .iter()
            .filter(|c| c.id != id && state.enemy_uids.contains(&CreatureUid(c.id)))
            .collect();
        assert_eq!(spawned.len(), 1, "{v:?}: Stock respawned one Axebot");
        assert_eq!(
            spawned[0].current_hp, spawned[0].max_hp,
            "{v:?}: the auto-played Strike found no target"
        );
        assert!(
            !state.player.draw.contains(&drawn) && !state.player.hand.contains(&drawn),
            "{v:?}: Gremlin Horn drew the Strike and Hellraiser moved it on"
        );
        assert_eq!(
            state.run_rng.combat_targets.counter(),
            targets_before,
            "{v:?}: NextItem over no enemies does not roll"
        );
    }
}

fn enemy(id: u32, hp: i32, kind: MonsterKind, name: &'static str) -> Creature {
    Creature {
        id: NonZeroU32::new(id).expect("nz"),
        current_hp: hp,
        max_hp: 100,
        block: 0,
        powers: SmallVec::new(),
        species: Some(kind),
        intent: None,
        intent_history: IntentHistory::default(),
        monster_id: MonsterId(name),
        ai: None,
    }
}

/// A Strike kills Rocket while Crusher, its Kaiser Crab sibling, still holds
/// `CrabRagePower`. The player's listeners come before the enemies' in the
/// `AfterDeath` walk (`CombatState.cs:115-160`), so Gremlin Horn draws a
/// Strike that Hellraiser auto-plays into Crusher's bare HP before Crusher's
/// `CrabRagePower.AfterDeath` (`CrabRagePower.cs:27-38`) grants +6 Strength
/// and 99 block. act23 Ironclad:4187 step 690.
#[test]
fn gremlin_horn_draw_resolves_before_a_siblings_crab_rage() {
    for v in BOTH {
        let mut state = fresh(v);
        state
            .creatures
            .push(enemy(2, 40, MonsterKind::Crusher, "Crusher"));
        state
            .creatures
            .push(enemy(3, 6, MonsterKind::Rocket, "Rocket"));
        let crusher = CreatureUid(NonZeroU32::new(2).expect("nz"));
        let rocket = CreatureUid(NonZeroU32::new(3).expect("nz"));
        state.enemy_uids.push(crusher);
        state.enemy_uids.push(rocket);
        sts2_core::card_effects::apply_power_to_creature(
            &mut state,
            crusher,
            PowerKind::CrabRage,
            1,
        );
        add_relic(&mut state, sts2_core::RelicKind::GremlinHorn);
        let p = state.player.creature_uid.expect("player");
        sts2_core::card_effects::apply_power_to_creature(&mut state, p, PowerKind::Hellraiser, 1);
        let drawn = new_card(&mut state, CardId::StrikeIronclad);
        state.player.draw.push(drawn);
        let strike = new_card(&mut state, CardId::StrikeIronclad);
        state.player.hand.push(strike);

        // Target index 1 = Rocket.
        let _ = sts2_core::combat_loop::play_card(&mut state, 0, Some(1));

        let hp_block = |uid: CreatureUid| {
            let c = state.creatures.iter().find(|c| c.id == uid.0).expect("c");
            (c.current_hp, c.block)
        };
        assert_eq!(hp_block(rocket).0, 0, "{v:?}: the Strike killed Rocket");
        assert_eq!(
            hp_block(crusher),
            (34, 99),
            "{v:?}: the auto-played Strike hit Crusher before CrabRage's block"
        );
    }
}
