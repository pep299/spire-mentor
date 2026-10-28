//! GH #858 on the scenario builder — the path the combat pro bench
//! (`sts2_rs.build_scenario_state`) and the SimLab sweep build every fight
//! through. Each affected encounter must open with its C#
//! `AfterAddedToRoom` amount at the bench's ascension, on both builds.

use sts2_core::powers::PowerKind;
use sts2_core::{CombatState, GameVersion};

/// A fresh scenario for `encounter` at `ascension` on `version`.
fn fight(
    version: GameVersion,
    encounter: &str,
    ascension: u8,
    monsters: Option<&[String]>,
) -> CombatState {
    let deck: Vec<String> = ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    sts2_wire::scenario::build_scenario_state_for_version(
        version,
        1,
        "Ironclad",
        &deck,
        80,
        80,
        &[],
        &[],
        encounter,
        ascension,
        monsters,
        None,
    )
    .unwrap_or_else(|e| panic!("{encounter} A{ascension} {version:?}: {e}"))
}

/// `power` stacks on every live enemy, in `enemy_uids` order.
fn enemy_powers(state: &CombatState, power: PowerKind) -> Vec<i32> {
    state
        .enemy_uids
        .iter()
        .filter_map(|u| state.creatures.iter().find(|c| c.id == u.0))
        .filter(|c| c.current_hp > 0)
        .map(|c| {
            c.powers
                .iter()
                .filter_map(|pu| state.powers.get_by_id(pu.0))
                .filter(|p| p.kind == power)
                .map(|p| p.amount)
                .sum()
        })
        .collect()
}

/// `(encounter, power, enemies carrying it, A7 amount, A10 amount)`. C#:
/// `TerrorEel.cs:39` Shriek (ToughEnemies, 75, 70); `Vantom.cs:64` Slippery
/// (.., 9, 8); `PhantasmalGardener.cs:49` Skittish (.., 7, 6) on all four
/// gardeners; `FrogKnight.cs:43` Plating (.., 19, 15); `InfestedPrism.cs:40`
/// VitalSpark (DeadlyEnemies, 3, 2) — A8 is still 2 there.
const CASES: &[(&str, PowerKind, usize, i32, i32)] = &[
    ("TERROR_EEL_ELITE", PowerKind::Shriek, 1, 70, 75),
    ("VANTOM_BOSS", PowerKind::Slippery, 1, 8, 9),
    ("PHANTASMAL_GARDENERS_ELITE", PowerKind::Skittish, 4, 6, 7),
    ("FROG_KNIGHT_NORMAL", PowerKind::Plating, 1, 15, 19),
];

#[test]
fn scenario_encounters_open_with_scaled_on_spawn_powers_858() {
    for version in GameVersion::ALL {
        for &(encounter, power, count, base, scaled) in CASES {
            for (asc, want) in [(7, base), (8, scaled), (10, scaled)] {
                let state = fight(version, encounter, asc, None);
                let got: Vec<i32> = enemy_powers(&state, power)
                    .into_iter()
                    .filter(|&a| a != 0)
                    .collect();
                assert_eq!(
                    got.len(),
                    count,
                    "{encounter} A{asc} {version:?}: {count} enemies carry {power:?}"
                );
                assert!(
                    got.iter().all(|&a| a == want),
                    "{encounter} A{asc} {version:?}: {power:?} {got:?}, want {want} each"
                );
            }
        }
        // InfestedPrism is DeadlyEnemies: 2 at A8, 3 at A9+.
        for (asc, want) in [(8, 2), (9, 3), (10, 3)] {
            let state = fight(version, "INFESTED_PRISMS_ELITE", asc, None);
            let got = enemy_powers(&state, PowerKind::VitalSpark);
            assert!(
                !got.is_empty() && got.iter().all(|&a| a == want),
                "INFESTED_PRISMS_ELITE A{asc} {version:?}: VitalSpark {got:?}, want {want}"
            );
        }
    }
}

/// FrogKnight's Plating is also its opening block (`PlatingPower`
/// `BeforeSideTurnStart`, round 1), so the scenario the bench plays from
/// opens with 19 block at A8+.
#[test]
fn frog_knight_opening_block_follows_its_plating_858() {
    for version in GameVersion::ALL {
        for (asc, want) in [(7, 15), (10, 19)] {
            let state = fight(version, "FROG_KNIGHT_NORMAL", asc, None);
            let block: Vec<i32> = state
                .enemy_uids
                .iter()
                .filter_map(|u| state.creatures.iter().find(|c| c.id == u.0))
                .map(|c| c.block)
                .collect();
            assert_eq!(block, vec![want], "FROG_KNIGHT_NORMAL A{asc} {version:?}");
        }
    }
}

/// The bench's `monsters=` roster override resolves names through
/// `monster_kind_from_python_name` and spawns through the same
/// `spawn_roster`, so it scales too.
#[test]
fn monster_override_scales_on_spawn_powers_858() {
    let roster = ["TERROR_EEL".to_string()];
    let state = fight(GameVersion::V0_107_1, "TERROR_EEL_ELITE", 10, Some(&roster));
    assert_eq!(enemy_powers(&state, PowerKind::Shriek), vec![75]);
}
