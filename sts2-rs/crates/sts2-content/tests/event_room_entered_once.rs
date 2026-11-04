//! An event room is entered once: `AfterRoomEntered` fires once per room.
//!
//! C# `RunManager.EnterRoom` runs `Hook.AfterRoomEntered` once per room, so
//! MawBank (`MawBank.cs`, 12 gold per room until a purchase) pays 12 on a
//! `?` node that resolves to an event. The engine's map move enters the room
//! (`RunManager::do_map_move` → `RunState::enter_room`), and binding the
//! event afterwards (`RunManager::bind_pending_event` → `begin_event` →
//! `RunState::enter_event_room`) entered it a second time. Every driver
//! that binds (full_run, PPO, the sts2-py agents, the workbench) saw room-entry
//! relics fire twice on every event room.

use sts2_core::map::MapPointType;
use sts2_core::potions::CharacterId;
use sts2_core::run_manager::{Phase, RunAction, RunManager};
use sts2_core::{GameVersion, RelicId};

/// Gold and HP right before the move, right after it, and after the bind.
struct EventEntry {
    seed: i32,
    before_move: (i32, i32),
    after_move: (i32, i32),
    after_bind: (i32, i32),
}

/// A fresh run on `seed` with MawBank, standing at the first map choice,
/// with every node of the first row turned into a `?` node. Tries each
/// first move on a clone and returns the first that resolves to an event.
fn first_event_entry(seed: i32, version: GameVersion) -> Option<EventEntry> {
    sts2_content::relics::dispatch::ensure_installed();
    let mut rm = RunManager::new(seed, CharacterId::Ironclad)
        .with_content_callbacks(sts2_content::run_callbacks::ironclad_content_callbacks())
        .with_game_version(version);
    rm.run_state.relics.push(RelicId::MawBank);
    // Planisphere heals 5 on entering a `?` point (Planisphere.cs).
    rm.run_state.relics.push(RelicId::Planisphere);
    rm.run_state.current_hp = rm.run_state.max_hp - 30;
    assert_eq!(
        rm.phase(),
        Phase::MapChoice,
        "a fresh run starts on the map"
    );
    let map = rm.run_state.current_act_map.as_mut().expect("act map");
    for p in &mut map.points {
        if p.coord.row == 1 {
            p.point_type = MapPointType::Unknown;
        }
    }
    for action in rm.get_available_actions() {
        if !matches!(action, RunAction::Move { .. }) {
            continue;
        }
        let mut probe = rm.clone();
        let before_move = (probe.run_state.gold, probe.run_state.current_hp);
        probe.take_action(action);
        if probe.phase() != Phase::Event || probe.run_state.current_event.is_some() {
            continue;
        }
        let after_move = (probe.run_state.gold, probe.run_state.current_hp);
        probe.bind_pending_event();
        assert!(
            probe.run_state.current_event.is_some(),
            "seed {seed}: event bound"
        );
        return Some(EventEntry {
            seed,
            before_move,
            after_move,
            after_bind: (probe.run_state.gold, probe.run_state.current_hp),
        });
    }
    None
}

fn check(version: GameVersion) {
    let entries: Vec<EventEntry> = (0..20)
        .filter_map(|seed| first_event_entry(seed, version))
        .collect();
    assert!(
        entries.len() >= 5,
        "{version:?}: need event-room entries to test, found {}",
        entries.len()
    );
    for e in &entries {
        assert_eq!(
            (
                e.after_move.0 - e.before_move.0,
                e.after_move.1 - e.before_move.1
            ),
            (12, 5),
            "{version:?} seed {}: the map move enters the room once \
             (MawBank +12 gold, Planisphere +5 HP)",
            e.seed
        );
        assert_eq!(
            e.after_bind, e.after_move,
            "{version:?} seed {}: binding the event must not fire \
             AfterRoomEntered again (gold, HP)",
            e.seed
        );
    }
}

#[test]
fn binding_an_event_room_does_not_enter_it_again_v0107() {
    check(GameVersion::V0_107_1);
}

#[test]
fn binding_an_event_room_does_not_enter_it_again_v0111() {
    check(GameVersion::V0_111_0);
}
