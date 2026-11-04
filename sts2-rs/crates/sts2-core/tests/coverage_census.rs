//! Per-character card-dispatch coverage census.
//!
//! For each character, partition every `CardId` of that character into
//! buckets (`unreferenced` / `stubbed` / `implemented`) and assert the
//! resulting table matches a committed golden file. Catches silent partial
//! ports: a pool bead that lands 30 of 87 cards must bump its golden file
//! deliberately, surfacing the gap in PR review.
//!
//! Bead sts2-rs-pmf (P6-1b) seeds the goldens at the zero-implemented
//! baseline. Downstream P6-3..P6-7 beads bump them as cards land.
//!
//! Bucket rules:
//! * `unreferenced` — no dispatch handler registered (`lookup` returns None).
//! * `stubbed`      — handler registered but signals `EffectResult::NotImplemented`
//!                    or otherwise placeholder. (P6-1b: zero stubs — none of
//!                    today's registered handlers self-report stub state.)
//! * `implemented`  — handler registered and runs real logic.
//!
//! The character bucket of a `CardId` is derived from the Python suffix:
//! `_IRONCLAD` -> ironclad, `_SILENT` -> silent, `_DEFECT` -> defect, and so
//! on. Cards without a character suffix bucket into `colorless` per the
//! Python pool layout (curses, status, neutral). The Ironclad starter cards
//! and Bash are the only handlers registered at P6-1b baseline.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use sts2_core::card_effects::{self, lookup};
use sts2_core::cards::{character_of, CardId};

#[derive(Default)]
struct Buckets {
    unreferenced: usize,
    stubbed: usize,
    implemented: usize,
}

fn format_one(b: &Buckets) -> String {
    let total = b.unreferenced + b.stubbed + b.implemented;
    format!(
        "unreferenced: {}\nstubbed:      {}\nimplemented:  {}\ntotal:        {}\n",
        b.unreferenced, b.stubbed, b.implemented, total,
    )
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates parent")
        .parent()
        .expect("repo root")
        .join("tests")
        .join("parity")
}

#[test]
fn coverage_census_matches_golden() {
    // Install the core/starter handlers (Strike/Defend/Bash/AscendersBane).
    card_effects::install_default_registry();

    let mut buckets: BTreeMap<&str, Buckets> = BTreeMap::new();
    for ch in ["ironclad", "silent", "defect", "colorless"] {
        buckets.entry(ch).or_default();
    }

    for &id in CardId::all() {
        let ch = character_of(id);
        let entry = buckets.entry(ch).or_default();
        match lookup(id) {
            None => entry.unreferenced += 1,
            // P6-1b baseline: every registered handler runs real logic.
            // Downstream beads MAY introduce stub handlers; extend this
            // arm when they do.
            Some(_) => entry.implemented += 1,
        }
    }

    // One golden file per character, conflict-free for parallel pool beads.
    let dir = golden_dir();
    let mut failures = Vec::new();
    for (ch, b) in &buckets {
        let path = dir.join(format!("coverage_census_{ch}.txt"));
        let actual = format_one(b);
        let golden = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing golden {}: {e}", path.display()));
        if actual != golden {
            failures.push(format!(
                "--- {} ---\nexpected:\n{golden}\nactual:\n{actual}\n",
                path.display()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Coverage census changed for {} character(s):\n{}\nRe-run, eyeball the diff, and bump goldens deliberately.",
        failures.len(),
        failures.join("\n")
    );
}
