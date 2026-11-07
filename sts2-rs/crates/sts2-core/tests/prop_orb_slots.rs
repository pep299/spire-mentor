//! Property tests for [`sts2_core::orbs::OrbSlots`] and the pure orb
//! helpers ([`focus_scaled`], [`OrbKind::of`]).
//!
//! [`OrbSlots`] is the Defect orb belt: a bounded collection with FIFO
//! eviction at capacity, documented at `orbs.rs:116-234`. The combat
//! wrapper [`channel_orb`] layers evoke-on-overflow side-effects on top,
//! but the bare `OrbSlots` primitive has its own self-contained contract
//! that the wrapper leans on:
//!
//!   * `with_capacity(c)` / `add_slots(n)` keep `slots.len()` exactly
//!     equal to `capacity()` — the belt is always fully materialised
//!     (filled orbs + trailing `Empty` markers), never a short vec
//!     (`orbs.rs:152-188`).
//!   * `channel` fills the leftmost `Empty` first, only evicting the
//!     oldest real orb (front) once every slot is full
//!     (`orbs.rs:190-208`). So the filled orbs always occupy a
//!     contiguous prefix in channel order, and the belt behaves as a
//!     bounded FIFO queue of the last `capacity` channelled orbs.
//!   * `channel` into a zero-capacity belt is a no-op
//!     (`orbs.rs:195-197`).
//!   * `filled_len() <= capacity()` always, and the per-kind counts
//!     partition the filled orbs: `sum_k count_kind(k) == filled_len()`,
//!     with `distinct_kinds() <= min(filled_len(), 5)`
//!     (`orbs.rs:210-233`).
//!
//! The headline test is a stateful model check: a plain `VecDeque`
//! bounded to `capacity` is the oracle, and after every `channel` the
//! belt's non-empty prefix must equal the model in order.

use hegel::generators as gs;
use hegel::TestCase;
use std::collections::VecDeque;
use sts2_core::orbs::{focus_scaled, Orb, OrbKind, OrbSlots};

/// Draw one of the five real (non-`Empty`) orb kinds. `Dark`/`Glass`
/// carry payloads that the belt mechanics ignore — we vary them anyway
/// so the model's identity comparison is exercised against distinct
/// payload values, not just the discriminant.
fn arb_orb(tc: &TestCase) -> Orb {
    let pick = tc.draw(gs::integers::<u8>().min_value(0).max_value(4));
    match pick {
        0 => Orb::Lightning,
        1 => Orb::Frost,
        2 => Orb::Dark {
            stored: tc.draw(gs::integers::<i32>().min_value(0).max_value(99)),
        },
        3 => Orb::Plasma,
        _ => Orb::Glass {
            current_passive: tc.draw(gs::integers::<i32>().min_value(0).max_value(8)),
        },
    }
}

/// The non-empty orbs in channel order (front = oldest). The belt keeps
/// filled orbs as a contiguous prefix, so this is just the leading run
/// of non-empty slots, but filtering is robust to that assumption.
fn filled_in_order(belt: &OrbSlots) -> Vec<Orb> {
    belt.slots
        .iter()
        .copied()
        .filter(|o| !o.is_empty())
        .collect()
}

#[hegel::test]
fn channel_matches_bounded_fifo_model(tc: TestCase) {
    // Start from an arbitrary non-zero capacity so eviction is reachable
    // within a modest op count.
    let cap = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
    let mut belt = OrbSlots::with_capacity(cap);
    let mut model: VecDeque<Orb> = VecDeque::new();

    let ops = tc.draw(gs::integers::<u32>().min_value(0).max_value(40));
    for _ in 0..ops {
        let orb = arb_orb(&tc);
        belt.channel(orb);

        // Oracle: append, then drop the oldest if we exceeded capacity.
        model.push_back(orb);
        if model.len() > cap {
            model.pop_front();
        }

        // The belt's filled prefix must match the FIFO model exactly,
        // in order. This pins both the leftmost-empty fill rule and the
        // front-eviction rule simultaneously.
        let want: Vec<Orb> = model.iter().copied().collect();
        assert_eq!(filled_in_order(&belt), want);
    }
}

#[hegel::test]
fn slots_len_always_equals_capacity(tc: TestCase) {
    let cap = tc.draw(gs::integers::<usize>().min_value(0).max_value(8));
    let mut belt = OrbSlots::with_capacity(cap);
    assert_eq!(belt.slots.len(), belt.capacity());

    // Interleave add_slots and channel; the materialised-belt invariant
    // (len == capacity) must survive both.
    let ops = tc.draw(gs::integers::<u32>().min_value(0).max_value(30));
    for _ in 0..ops {
        if tc.draw(gs::booleans()) {
            let n = tc.draw(gs::integers::<usize>().min_value(0).max_value(3));
            belt.add_slots(n);
        } else {
            belt.channel(arb_orb(&tc));
        }
        assert_eq!(
            belt.slots.len(),
            belt.capacity(),
            "belt must stay fully materialised"
        );
        assert!(
            belt.filled_len() <= belt.capacity(),
            "filled {} exceeded capacity {}",
            belt.filled_len(),
            belt.capacity()
        );
    }
}

#[hegel::test]
fn channel_into_zero_capacity_is_noop(tc: TestCase) {
    let mut belt = OrbSlots::with_capacity(0);
    let n = tc.draw(gs::integers::<u32>().min_value(0).max_value(10));
    for _ in 0..n {
        belt.channel(arb_orb(&tc));
    }
    assert_eq!(belt.capacity(), 0);
    assert_eq!(belt.filled_len(), 0);
    assert!(belt.slots.is_empty());
}

#[hegel::test]
fn add_slots_grows_capacity_additively_up_to_ten(tc: TestCase) {
    let start = tc.draw(gs::integers::<usize>().min_value(0).max_value(20));
    let add = tc.draw(gs::integers::<usize>().min_value(0).max_value(20));
    let mut belt = OrbSlots::with_capacity(start);
    let before_filled = belt.filled_len();
    belt.add_slots(add);

    // Capacity grows by `add`, clamped to 10 total (C# `OrbCmd.AddSlots`
    // `Min(10 - Capacity, amount)`, never shrinking a belt already past it);
    // all new slots are empty, and no existing orb is disturbed.
    let want = start.max((start + add).min(sts2_core::orbs::MAX_ORB_SLOTS));
    assert_eq!(belt.capacity(), want);
    assert_eq!(belt.slots.len(), want);
    assert_eq!(belt.filled_len(), before_filled);
}

#[hegel::test]
fn count_kind_partitions_filled(tc: TestCase) {
    let cap = tc.draw(gs::integers::<usize>().min_value(0).max_value(8));
    let mut belt = OrbSlots::with_capacity(cap);
    let ops = tc.draw(gs::integers::<u32>().min_value(0).max_value(40));
    for _ in 0..ops {
        belt.channel(arb_orb(&tc));
    }

    // Per-kind counts over the real kinds must sum to filled_len: the
    // five real kinds partition every non-empty orb.
    let kinds = [
        OrbKind::Lightning,
        OrbKind::Frost,
        OrbKind::Dark,
        OrbKind::Plasma,
        OrbKind::Glass,
    ];
    let sum: usize = kinds.iter().map(|&k| belt.count_kind(k)).sum();
    assert_eq!(sum, belt.filled_len());

    // Empty is never counted as a real held orb beyond the trailing
    // markers; count_kind(Empty) is exactly capacity - filled_len.
    assert_eq!(
        belt.count_kind(OrbKind::Empty),
        belt.capacity() - belt.filled_len()
    );

    // distinct_kinds is bounded by both the held count and the 5-kind
    // universe, and is positive iff something is held.
    let distinct = belt.distinct_kinds();
    assert!(distinct <= belt.filled_len());
    assert!(distinct <= 5);
    assert_eq!(distinct == 0, belt.filled_len() == 0);
}

#[hegel::test]
fn orbkind_of_round_trips_constructors(tc: TestCase) {
    // OrbKind::of must project each constructor to its matching
    // discriminant regardless of payload.
    let orb = arb_orb(&tc);
    let want = match orb {
        Orb::Lightning => OrbKind::Lightning,
        Orb::Frost => OrbKind::Frost,
        Orb::Dark { .. } => OrbKind::Dark,
        Orb::Plasma => OrbKind::Plasma,
        Orb::Glass { .. } => OrbKind::Glass,
        Orb::Empty => OrbKind::Empty,
    };
    assert_eq!(OrbKind::of(&orb), want);
    assert_eq!(OrbKind::of(&Orb::Empty), OrbKind::Empty);
}

#[hegel::test]
fn focus_scaled_is_clamped_saturating_add(tc: TestCase) {
    // `focus_scaled(base, focus) == max(0, base + focus)`, computed
    // against an i64 oracle to dodge i32 overflow in the reference.
    let base = tc.draw(gs::integers::<i32>().min_value(-1000).max_value(1000));
    let focus = tc.draw(gs::integers::<i32>().min_value(-1000).max_value(1000));
    let got = focus_scaled(base, focus);

    let want = (i64::from(base) + i64::from(focus)).max(0);
    assert_eq!(i64::from(got), want);
    // Never negative — orbs under heavy Focus drain deal 0, not heal.
    assert!(got >= 0);
}

#[hegel::test]
fn focus_scaled_is_monotone_in_focus(tc: TestCase) {
    // Increasing Focus never decreases the realised value (clamp at 0 is
    // monotone, saturating add is monotone).
    let base = tc.draw(gs::integers::<i32>().min_value(-500).max_value(500));
    let f_lo = tc.draw(gs::integers::<i32>().min_value(-500).max_value(500));
    let bump = tc.draw(gs::integers::<i32>().min_value(0).max_value(500));
    let f_hi = f_lo.saturating_add(bump);
    assert!(focus_scaled(base, f_hi) >= focus_scaled(base, f_lo));
}
