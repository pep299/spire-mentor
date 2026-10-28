// port-pin-skip — DSL re-export glue (sts2-core monster_ai is canonical).
pub use sts2_core::monster_ai::{
    ConditionFn, ConditionalBranch, ConditionalBranchState, EffectFn, MonsterAi, MonsterState,
    MoveRepeatType, MoveState, RandomBranchState, StateId, WeightedBranch,
};

use sts2_core::monsters::Intent;
use sts2_core::{CombatState, CreatureUid};

/// Shared no-op [`EffectFn`] for telegraph-only moves whose behavior is
/// fully described by their intents.
pub(crate) fn noop_effect(_: &mut CombatState, _: CreatureUid) {}

/// Move-state builder: a telegraph-only move (no effect beyond its
/// intents) that transitions to `follow_up`.
pub(crate) fn mv(
    id: &'static str,
    intents: smallvec::SmallVec<[Intent; 4]>,
    follow_up: &'static str,
) -> MonsterState {
    MonsterState::Move(MoveState {
        id,
        effect: noop_effect,
        intents,
        follow_up: Some(follow_up),
        must_perform_once: false,
    })
}

/// Stun move-state builder: the prebuilt analogue of the
/// `MoveState("STUNNED", …, new StunIntent())` that C#
/// `Creature.StunInternal` synthesizes (Creature.cs:538-542). A no-op
/// `Intent::Stun` telegraph with `must_perform_once: true` — C# sets
/// `MustPerformOnceBeforeTransitioning = true` on every stun, so the machine
/// cannot roll away from it before the stunned turn has been served. Reach it
/// through `Effect::StunMonster`, never as a normal follow-up.
pub(crate) fn stun_mv(id: &'static str, follow_up: &'static str) -> MonsterState {
    MonsterState::Move(MoveState {
        id,
        effect: noop_effect,
        intents: smallvec::smallvec![Intent::Stun],
        follow_up: Some(follow_up),
        must_perform_once: true,
    })
}

/// [`mv`] variant that attaches a non-noop `effect_fn` (bead
/// `sts2-rs-vn0m`). Used by summoner moves (e.g. Fogmog ILLUSION, Vantom
/// CONSUME) whose effect must actually call `combat.spawn_enemy`, and by
/// telegraph-only Buff/Debuff moves that must apply a real power (bead
/// `sts2-rs-pdsdl`).
pub(crate) fn mv_eff(
    id: &'static str,
    intents: smallvec::SmallVec<[Intent; 4]>,
    follow_up: &'static str,
    effect: EffectFn,
) -> MonsterState {
    MonsterState::Move(MoveState {
        id,
        effect,
        intents,
        follow_up: Some(follow_up),
        must_perform_once: false,
    })
}

/// Weighted-branch builder for [`RandomBranchState`] (no cooldown).
pub(crate) fn br(
    state_id: &'static str,
    repeat: MoveRepeatType,
    max_times: u8,
    weight: f64,
) -> WeightedBranch {
    WeightedBranch {
        state_id,
        repeat_type: repeat,
        max_times,
        weight,
        cooldown: 0,
    }
}

/// Stateless [`ConditionalBranch`] builder for [`ConditionalBranchState`].
pub(crate) fn cb(condition: fn(&CombatState) -> bool, state_id: &'static str) -> ConditionalBranch {
    ConditionalBranch {
        condition: ConditionFn::Stateless(condition),
        state_id,
    }
}

/// Owner-aware [`ConditionalBranch`] builder — the predicate sees the creature
/// whose machine is branching. Use whenever the C# condition reads
/// `base.Creature` (`HasPower<…>`, HP thresholds), rather than reaching for a
/// combat-global proxy that cannot see per-creature state.
pub(crate) fn cb_owner(
    condition: fn(&CombatState, sts2_core::CreatureUid) -> bool,
    state_id: &'static str,
) -> ConditionalBranch {
    ConditionalBranch {
        condition: ConditionFn::OwnerAware(condition),
        state_id,
    }
}

/// Catch-all condition for the final [`cb`] fallback branch.
pub(crate) fn always_true(_: &CombatState) -> bool {
    true
}

/// One per-thread prototype per variant of a mid-combat spawn builder, for
/// [`proto_ai`]. Declare it inside the builder as
/// `thread_local! { static PROTO: ProtoAis<N> = const { proto_ais() }; }`.
pub(crate) type ProtoAis<const N: usize> = [std::cell::OnceCell<MonsterAi>; N];

/// An empty [`ProtoAis`].
pub(crate) const fn proto_ais<const N: usize>() -> ProtoAis<N> {
    [const { std::cell::OnceCell::new() }; N]
}

/// A clone of this thread's prototype for `variant`, built by `build` on the
/// thread's first call (perf J10).
///
/// A summon (PhrogParasite's four wrigglers, Fabricator bots, Ovicopter
/// eggs, …) used to rebuild its whole state machine — the `BTreeMap`, the
/// [`StateTable`](sts2_core::monster_ai::StateTable) and its `Arc` — on every
/// spawn, inside the search. The clone shares the prototype's table and
/// copies the per-instance fields as `MonsterAi::new` left them, so it is the
/// value a fresh build returns. A later `Arc::make_mut(&mut ai.states)`
/// (ascension scaling, WaterfallGiant's PRESSURE_GUN ramp and its
/// ABOUT_TO_BLOW rewrite of EXPLODE, TheForgotten's DREAD rewrite) copies on
/// write and never reaches the prototype.
///
/// Only for builders that are PURE in `variant`: no RNG, no run or combat
/// state. `variant` must identify the build exactly. The cache is per thread
/// because a process-wide one would make every rayon worker bump the same
/// refcounts on every `CombatState` clone. Debug builds compare every clone
/// with a fresh build.
pub(crate) fn proto_ai<const N: usize>(
    protos: &'static std::thread::LocalKey<ProtoAis<N>>,
    variant: usize,
    build: impl Fn() -> MonsterAi,
) -> MonsterAi {
    let ai = protos.with(|p| p[variant].get_or_init(&build).clone());
    debug_assert_eq!(
        format!("{ai:?}"),
        format!("{:?}", build()),
        "monster AI prototype {variant} is not what its builder returns"
    );
    ai
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{MonsterAi, MonsterState};
    use crate::monsters::act1_underdocks::TwoTailedRatSlot;
    use crate::monsters::shared::DenseVegetationWrigglerSlot;
    use crate::monsters::{act1_overgrowth, act1_underdocks, act2_hive, act3_glory, shared};
    use sts2_core::monsters::Intent;
    use sts2_core::MonsterKind;

    type Builder = Box<dyn Fn() -> MonsterAi>;

    /// Every [`super::proto_ai`]-cached builder, every variant: the cached
    /// entry point and the uncached build it must equal.
    fn cached_builders() -> Vec<(String, Builder, Builder)> {
        let mut all: Vec<(String, Builder, Builder)> = vec![
            (
                "eye_with_teeth".into(),
                Box::new(act1_overgrowth::create_eye_with_teeth_ai),
                Box::new(act1_overgrowth::build_eye_with_teeth_ai),
            ),
            (
                "parafright".into(),
                Box::new(act1_overgrowth::create_parafright_ai),
                Box::new(act1_overgrowth::build_parafright_ai),
            ),
            (
                "sneaky_gremlin".into(),
                Box::new(act1_underdocks::create_sneaky_gremlin_ai),
                Box::new(act1_underdocks::build_sneaky_gremlin_ai),
            ),
            (
                "fat_gremlin".into(),
                Box::new(act1_underdocks::create_fat_gremlin_ai),
                Box::new(act1_underdocks::build_fat_gremlin_ai),
            ),
            (
                "gas_bomb".into(),
                Box::new(act1_underdocks::create_gas_bomb_ai),
                Box::new(act1_underdocks::build_gas_bomb_ai),
            ),
            (
                "tough_egg".into(),
                Box::new(act2_hive::create_tough_egg_ai),
                Box::new(act2_hive::build_tough_egg_ai),
            ),
        ];
        for slot in [
            DenseVegetationWrigglerSlot::First,
            DenseVegetationWrigglerSlot::Second,
            DenseVegetationWrigglerSlot::Third,
            DenseVegetationWrigglerSlot::Fourth,
        ] {
            all.push((
                format!("infested_wriggler {slot:?}"),
                Box::new(move || shared::create_infested_wriggler_ai(slot)),
                Box::new(move || shared::build_infested_wriggler_ai(slot)),
            ));
        }
        for slot in [
            TwoTailedRatSlot::First,
            TwoTailedRatSlot::Second,
            TwoTailedRatSlot::Third,
            TwoTailedRatSlot::Other,
        ] {
            all.push((
                format!("two_tailed_rat {slot:?}"),
                Box::new(move || act1_underdocks::create_two_tailed_rat_ai(slot)),
                Box::new(move || act1_underdocks::build_two_tailed_rat_ai(slot)),
            ));
        }
        for boot_up in [false, true] {
            all.push((
                format!("axebot boot_up={boot_up}"),
                Box::new(move || act3_glory::create_axebot_ai(boot_up)),
                Box::new(move || act3_glory::build_axebot_ai(boot_up)),
            ));
        }
        let bots: [(MonsterKind, fn() -> MonsterAi); 4] = [
            (MonsterKind::Stabbot, act3_glory::create_stabbot_ai),
            (MonsterKind::Guardbot, act3_glory::create_guardbot_ai),
            (MonsterKind::Noisebot, act3_glory::create_noisebot_ai),
            (MonsterKind::Zapbot, act3_glory::create_zapbot_ai),
        ];
        for (kind, build) in bots {
            all.push((
                format!("bot {kind:?}"),
                Box::new(move || act3_glory::create_bot_ai(kind)),
                Box::new(build),
            ));
        }
        all
    }

    /// A cached summon AI is the value a fresh build returns, field for field
    /// (`Debug` covers the whole table, fn pointers included), and the second
    /// call really is a clone of the prototype rather than another build.
    #[test]
    fn a_cached_summon_ai_equals_a_fresh_build_for_every_variant() {
        for (name, cached, fresh) in cached_builders() {
            let want = format!("{:?}", fresh());
            let first = cached();
            let second = cached();
            assert_eq!(format!("{first:?}"), want, "{name}: first call");
            assert_eq!(format!("{second:?}"), want, "{name}: cached call");
            assert!(
                Arc::ptr_eq(&first.states, &second.states),
                "{name}: the second call did not clone the prototype"
            );
        }
    }

    /// A spawn-time / mid-combat `Arc::make_mut` on a cached clone copies the
    /// table and leaves the prototype, so the next spawn is still pristine.
    #[test]
    fn a_write_through_a_cached_clone_never_reaches_the_prototype() {
        let slot = DenseVegetationWrigglerSlot::First;
        let mut written = shared::create_infested_wriggler_ai(slot);
        let Some(MonsterState::Move(bite)) =
            Arc::make_mut(&mut written.states).get_mut("NASTY_BITE")
        else {
            panic!("the wriggler has a NASTY_BITE move");
        };
        bite.intents[0] = Intent::Attack { damage: 99 };
        let next = shared::create_infested_wriggler_ai(slot);
        assert!(!Arc::ptr_eq(&written.states, &next.states));
        assert_eq!(
            format!("{next:?}"),
            format!("{:?}", shared::build_infested_wriggler_ai(slot))
        );
        assert_ne!(format!("{written:?}"), format!("{next:?}"));
    }

    /// The prototype is per thread: a rayon worker never shares a table (and
    /// its refcount) with another thread's spawns.
    #[test]
    fn each_thread_builds_its_own_prototype() {
        let slot = DenseVegetationWrigglerSlot::Second;
        let here = shared::create_infested_wriggler_ai(slot);
        let there = std::thread::spawn(move || shared::create_infested_wriggler_ai(slot))
            .join()
            .expect("the spawning thread finishes");
        assert!(!Arc::ptr_eq(&here.states, &there.states));
        assert_eq!(format!("{here:?}"), format!("{there:?}"));
    }
}
