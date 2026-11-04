//! `Hook.AfterSideTurnStart` walks its listeners CREATURE-MAJOR: every ally
//! creature's powers, then (for a player) its relics, potions, orbs and
//! cards, and only then each enemy's powers and monster.
//!
//! C# `Hook.AfterSideTurnStart` (v0.111.0 `Hook.cs:1319-1331`, v0.107.1
//! `Hook.cs:1307-1319`) awaits each model of `IterateCombatHookListeners`,
//! whose iterator (`CombatState.cs` `IterateHookListeners`) walks
//! `Allies.Concat(Enemies)` and, per creature, yields `creature.Powers` then
//! the player's relics. So the player's `Sai` relic (`Sai.cs:19-25`, +7
//! block) and the `JuggernautPower` it triggers (`JuggernautPower.cs:17-26`,
//! damage to a random enemy on block gained) resolve BEFORE the enemy-owned
//! `RampartPower` (`RampartPower.cs:21-30`) re-grants 25 block to the
//! TurretOperator. Rust ran the whole power bus (enemy Rampart included)
//! before the relic bus, so Juggernaut's damage landed on the fresh 25 block.
//! Ironclad seed 2045 (v0.111.0, TURRET_OPERATOR, step 634) desynced here.

use std::num::NonZeroU32;

use sts2_core::combat_loop::start_player_turn;
use sts2_core::relics::RelicKind;
use sts2_core::{CombatPhase, CombatState, MonsterKind, Power, PowerKind, PowerUid, Relic};

fn turret_fight() -> (CombatState, sts2_core::CreatureUid) {
    sts2_content::cards::dispatch::install_default_registry();
    sts2_content::relics::dispatch::ensure_installed();
    let mut state = CombatState {
        rng: sts2_core::dotnet_rng::Rng::new(0_i32),
        ..Default::default()
    };
    state.player.current_hp = 80;
    state.player.max_hp = 80;
    let player_uid = state.init_player_creature();

    // One TurretOperator carrying Rampart(25) itself, so Juggernaut's random
    // enemy pick has a single candidate.
    let turret = state.spawn_enemy(MonsterKind::TurretOperator);
    let rampart = PowerUid::new(900).expect("nz");
    state.powers.push(Power {
        id: rampart.0,
        kind: PowerKind::Rampart,
        amount: 25,
        counter: 0,
        applier: None,
        skip_next_tick: false,
    });
    let c = state
        .creatures
        .iter_mut()
        .find(|c| c.id == turret.0)
        .expect("turret");
    c.current_hp = 41;
    c.max_hp = 41;
    c.block = 0;
    c.powers.push(rampart);

    sts2_core::card_effects::apply_power_to_creature(
        &mut state,
        player_uid,
        PowerKind::Juggernaut,
        6,
    );
    state.relics.push(Relic {
        id: NonZeroU32::new(1).expect("nz"),
        kind: RelicKind::Sai,
        counter: 0,
        is_wax: false,
        is_melted: false,
    });
    state.round = 2;
    (state, turret)
}

#[test]
fn player_relic_and_its_juggernaut_hit_resolve_before_enemy_rampart() {
    let (mut state, turret) = turret_fight();
    state.phase = CombatPhase::PlayerTurnStart;
    let _ = start_player_turn(&mut state);

    let t = state
        .creatures
        .iter()
        .find(|c| c.id == turret.0)
        .expect("turret");
    assert_eq!(state.player.block, 7, "Sai grants 7 block");
    assert_eq!(
        t.current_hp, 35,
        "Juggernaut's 6 lands on the unblocked turret before Rampart"
    );
    assert_eq!(t.block, 25, "Rampart's 25 block is granted afterwards");
}
