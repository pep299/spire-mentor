//! Hook-callsite census.
//!
//! Records the list of `hooks::dispatch_*` callsites in
//! `crates/sts2-core/src/combat_loop.rs` and asserts it matches a golden
//! file. Catches the "Inflame subscribed to `on_card_played` for ALL
//! creatures" class of bug (per ROBUSTNESS §4.4 / D18 of the synthesized
//! plan) by pinning *which* hooks the combat loop calls — if a pool bead
//! adds a new listener subscription, the count of dispatch arms in
//! `combat_loop` shouldn't move; if a refactor accidentally removes a
//! dispatch call, the count drops and the assertion fires.
//!
//! Bead sts2-rs-pmf (P6-1b) seeds the golden at the current baseline (P5
//! shipping state). Downstream beads that legitimately add a `dispatch_*`
//! call site bump the golden deliberately.
//!
//! The census matches on the literal `hooks::dispatch_` prefix, so a hook
//! whose entry point is not named `dispatch_*` is invisible to it. There is
//! one: `hooks::block_clear_preventer` (GH #634), the predicate half of C#
//! `Hook.ShouldClearBlock`. Its census stand-in is
//! `dispatch_after_preventing_block_clear`, which the same `if let` arm calls
//! and which cannot be reached without it — delete the predicate and the
//! golden still moves.
//!
//! The "(power, hook, count)" tuples envisioned in ROBUSTNESS §4.4 require
//! a fully-wired combat trace; that lands with `fxw` (P6-10). At P6-1b
//! we lock the precursor — the static dispatch surface — so divergence is
//! caught earlier than first-fuzz.

use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates parent")
        .parent()
        .expect("repo root")
        .to_path_buf()
}

#[test]
fn hook_dispatch_callsites_match_golden() {
    let combat_loop = repo_root()
        .join("crates")
        .join("sts2-core")
        .join("src")
        .join("combat_loop.rs");
    let src =
        fs::read_to_string(&combat_loop).unwrap_or_else(|e| panic!("read combat_loop.rs: {e}"));

    // Collect every `hooks::dispatch_<name>` reference. The static-text scan
    // is intentionally syntactic — sub-100 LOC and zero-cost — so it stays
    // green even when the dispatcher signatures evolve in P6-3..P6-10.
    // Bead `sts2-rs-6ynt.11`: tighten the matcher to exclude
    // `enchantment_hooks::dispatch_*` (and any future `*_hooks::`
    // sibling-bus call). The census tracks the canonical
    // `hooks::dispatch_*` surface; sibling buses (relic_effects,
    // enchantment_hooks, run_effects) have their own discipline.
    let mut hooks: Vec<String> = src
        .lines()
        .filter_map(|line| {
            let i = line.find("hooks::dispatch_")?;
            if i > 0 && line.as_bytes()[i - 1] == b'_' {
                return None;
            }
            let tail = &line[i + "hooks::".len()..];
            let end = tail
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(tail.len());
            Some(tail[..end].to_string())
        })
        .collect();
    hooks.sort();
    hooks.dedup();

    let mut actual = String::new();
    for h in &hooks {
        actual.push_str(h);
        actual.push('\n');
    }

    let golden_path = repo_root()
        .join("tests")
        .join("parity")
        .join("hook_call_census.txt");
    let golden = fs::read_to_string(&golden_path)
        .unwrap_or_else(|e| panic!("missing golden {}: {e}", golden_path.display()));

    assert_eq!(
        actual,
        golden,
        "hooks::dispatch_* callsites in combat_loop.rs changed. \
         Re-run, eyeball, then bump {}.",
        golden_path.display()
    );
}
