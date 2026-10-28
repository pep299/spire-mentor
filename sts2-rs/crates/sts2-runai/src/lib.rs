//! `sts2-runai` — run-level AI for full STS2 runs.
//!
//! Bead `sts2-rs-40ikw.4` (M0 of
//! `docs/plans/2026-06-12-full-run-ai-plan.md`): the crate that owns
//! everything *above* the combat engine on the way to a full-run agent —
//! benchmark seed panels (§3.1), the arena sweep harness with
//! completeness binning (§3.2), the frozen combat budget configs (§3.3),
//! the decision audit log (§3.6, minimal form), and the baseline
//! [`sts2_core::full_run::RunPolicy`] implementations (random /
//! first-option meta policies + the MCTS combat bridge).
//!
//! The `run_arena` binary in `sts2-cli` is a thin clap wrapper over
//! [`arena`]; the per-seed wall-clock-timeout (process-per-seed) layer
//! lives in the binary because it re-execs `current_exe()`.
//!
//! Bead `sts2-rs-40ikw.5` (M1, plan §5/M1 + §2 + §2.5): the tier-1
//! heuristic agent over all six meta surfaces — [`view`] (the
//! `VisibleRunView` information-discipline chokepoint), [`eval`]
//! (`HeuristicV` + TOML weights), [`projection`] (per-room-class state
//! projection for pathing), and [`policy::heuristic`] (the
//! `HeuristicPolicy` surface dispatcher).

//!
//! Bead `sts2-rs-40ikw.6` (M2, plan §5/M2 + §2.5 rule 3 + §6 E3/E4/E5):
//! rollout grounding — [`rollout`] (`fight_delta` paired synthetic-seed
//! probes + the bounded fight-strength cache), [`policy::tiered`]
//! (margin-gated tier-2/3 escalation), [`label`] (the offline rollout
//! labeler behind `run_arena label-campaign`), and [`calibrate`]
//! (the E3/E4/E5 budget/K/truncation calibration harness).

//! Bead `sts2-rs-40ikw.7` (M3, plan §5/M3) built a pick-ranker net on
//! top of [`eval::ranker_features`] (the versioned feature schema, which
//! the value/surrogate nets still share). The ranker net, its policy,
//! the feature export, and the self-play retraining loop were removed
//! once PPO ([`ppo`]) became the deployed approach — only the shared
//! feature schema survives.
//!
//! `sts2-pycli/docs/superpowers/plans/2026-07-07-ppo-rust-inference.md`
//! (Task 3+): [`ppo`] — Rust inference for the PPO `RunPolicyNet` (ONNX
//! export via tract), mirroring `sts2-pycli/python/ppo/{vocab,mapfeat,
//! env,encoder}.py` function-for-function and gated by exact
//! featurization + forward parity fixtures.

/// Content hash of THIS crate's build inputs, stamped at build time by
/// `build.rs`. Compiled INTO the crate so a stale `sts2-runai` rlib carries its
/// old hash; one of the parts of `sts2_rs.src_hash()`. See
/// `sts2_core::SRC_HASH`.
pub const SRC_HASH: &str = env!("STS2_RUNAI_SRC_HASH");

#[cfg(feature = "sweeps")]
pub mod arena;
#[cfg(feature = "sweeps")]
pub mod calibrate;
pub mod deck_archetype;
pub mod eval;
#[cfg(feature = "sweeps")]
pub mod fork_diagnosis;
#[cfg(feature = "sweeps")]
pub mod fork_label;
#[cfg(feature = "sweeps")]
pub mod harvest;
#[cfg(feature = "sweeps")]
pub mod label;
#[cfg(feature = "sweeps")]
pub mod panel;
pub mod planning;
pub mod planning_mcts;
pub mod policy;
pub mod ppo;
pub mod projection;
#[cfg(feature = "sweeps")]
pub mod proxy_study;
pub mod rollout;
#[cfg(feature = "sweeps")]
pub mod trajectory;
pub mod turn_plan_teacher;
pub mod view;
