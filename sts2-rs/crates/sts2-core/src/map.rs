//! Act-map data PODs + `ActMap` graph container.
//!
//! Bead `sts2-rs-zregk` (P9.1.a) ported the three core PODs
//! (`MapCoord`, `MapEdge`, `MapPoint`) from the Python
//! source-of-truth at
//! `sts2-rl-agent@01c094d:sts2_env/map/map_point.py`. Bead
//! `sts2-rs-h4jx9` (P9.1.c) added categorical helpers on
//! [`MapPointType`]. Bead `sts2-rs-72ipm` (P9.1.b — this file's
//! [`ActMap`]) ports the empty graph container + traversal helpers
//! from `sts2_env/map/generator.py:37-69`. The generator algorithm
//! itself (path generation, room-type assignment) lands in
//! `sts2-rs-wn4wc` (P9.2.b).
//!
//! Departure from Python: Python's `MapPoint.children`/`.parents`
//! hold direct object references and rely on identity (`is`) checks.
//! Rust uses index handles ([`MapPointIdx`]) into an `ActMap`-owned
//! `Vec<MapPoint>` so that `Clone` stays cheap and traversal doesn't
//! need `Rc`/`Box` cycles.

use serde::{Deserialize, Serialize};

/// (column, row) coordinate on the act-map grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MapCoord {
    pub col: i32,
    pub row: i32,
}

impl MapCoord {
    #[must_use]
    pub const fn new(col: i32, row: i32) -> Self {
        Self { col, row }
    }
}

/// One entry in `RunState.map_point_history` — the per-room visit log
/// the run-loop driver appends from
/// [`crate::runstate::RunState::add_visited_coord`] as the player
/// walks the act map. Mirrors the frozen dataclass at
/// `sts2-rl-agent@01c094d:sts2_env/run/run_state.py:54-57`:
///
/// ```python
/// @dataclass(frozen=True)
/// class MapPointHistoryEntry:
///     map_point_type: MapPointType
///     room_type: RoomType
/// ```
///
/// The `room_type` is the *resolved* room (e.g. unknown nodes have
/// their resolved kind here, not `MapPointType::Unknown`); the
/// `map_point_type` is the original node-typing on the map so the
/// audit-trail distinguishes a Monster room reached via an `?` node
/// from a Monster room reached via a `M` node. Consumers walk the
/// history via [`crate::runstate::RunState::count_map_point_history_entries`].
/// Bead `sts2-rs-2c0ze` (P9.2.m audit §2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MapPointHistoryEntry {
    /// Map-node type as rolled at generation. Mirrors Python
    /// `MapPointHistoryEntry.map_point_type`.
    pub map_point_type: MapPointType,
    /// Resolved room type for the visit. Mirrors Python
    /// `MapPointHistoryEntry.room_type`.
    pub room_type: crate::runstate::RoomType,
}

impl MapPointHistoryEntry {
    /// Constructor — frozen-dataclass shape, no defaulting.
    #[must_use]
    pub const fn new(map_point_type: MapPointType, room_type: crate::runstate::RoomType) -> Self {
        Self {
            map_point_type,
            room_type,
        }
    }
}

/// Directed edge between two grid coordinates.
///
/// Edges are stored on [`MapPoint`] adjacency lists ([`MapPoint::children`] /
/// [`MapPoint::parents`]); this POD is the standalone form used by
/// map-generator helpers and serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MapEdge {
    pub src: MapCoord,
    pub dst: MapCoord,
}

impl MapEdge {
    #[must_use]
    pub const fn new(src: MapCoord, dst: MapCoord) -> Self {
        Self { src, dst }
    }
}

/// Index handle into an `ActMap`-owned `Vec<MapPoint>`.
///
/// Replaces Python's direct `MapPoint` references in
/// `MapPoint.children` / `MapPoint.parents`. `u16` is enough for any
/// real act (Python pin has at most 15×7 = 105 points per act).
pub type MapPointIdx = u16;

/// Room-type discriminant assigned to each map node.
///
/// The categorical helpers (`is_combat()`, `is_rewards()`, …) live in
/// P9.1.c (bead `sts2-rs-h4jx9`); this enum is the bare discriminant
/// so [`MapPoint::point_type`] can compile.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub enum MapPointType {
    #[default]
    Unassigned,
    Unknown,
    Monster,
    Elite,
    Boss,
    Shop,
    RestSite,
    Treasure,
    /// Event-room point type. Python's `MapPointType.ANCIENT` —
    /// `run_state.py:1639` maps `RoomType::Event` ↔ this variant.
    Ancient,
}

impl MapPointType {
    /// True for room types that drive a combat encounter:
    /// `Monster`, `Elite`, `Boss`. Used by combat-vs-non-combat run-
    /// loop gating (e.g. post-combat reward fan-out, encounter-rng
    /// advancement, VenerableTeaSet's combat-end energy latch).
    #[must_use]
    pub const fn is_combat(self) -> bool {
        matches!(self, Self::Monster | Self::Elite | Self::Boss)
    }

    /// True iff this point is a Shop (merchant). Mirrors Python's
    /// implicit `point_type == MapPointType.SHOP` check used by
    /// LordsParasol-style merchant-room hooks.
    #[must_use]
    pub const fn is_merchant(self) -> bool {
        matches!(self, Self::Shop)
    }

    /// True iff this point is a Rest Site. Mirrors Python's implicit
    /// `point_type == MapPointType.REST_SITE` check used by
    /// rest-site dispatch in `run_manager._enter_rest_site`.
    #[must_use]
    pub const fn is_rest_site(self) -> bool {
        matches!(self, Self::RestSite)
    }

    /// True iff this point is the act boss. Useful for boss-relic
    /// offer gating and is_boss snapshotting at combat init.
    #[must_use]
    pub const fn is_boss(self) -> bool {
        matches!(self, Self::Boss)
    }

    /// True iff this point is an Elite encounter. Used by Tutorial
    /// elite-counter / `count_map_point_history_entries(ELITE)`
    /// callers in `run/reward_objects.py:1052`.
    #[must_use]
    pub const fn is_elite(self) -> bool {
        matches!(self, Self::Elite)
    }

    /// True iff this point is Treasure. Trivial chest-roll trigger.
    #[must_use]
    pub const fn is_treasure(self) -> bool {
        matches!(self, Self::Treasure)
    }

    /// True iff this point is an Unknown ('?') room awaiting
    /// resolution by the unknown-room resolver
    /// (`UnknownMapPointOdds`).
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// True iff this point is an Event (`Ancient` in Python).
    #[must_use]
    pub const fn is_event(self) -> bool {
        matches!(self, Self::Ancient)
    }

    /// True iff the type slot has been filled by the map generator
    /// — anything but `Unassigned`. Mirrors Python's
    /// `point_type is not MapPointType.UNASSIGNED` guard in
    /// `run_state.py:1691`.
    #[must_use]
    pub const fn is_assigned(self) -> bool {
        !matches!(self, Self::Unassigned)
    }

    /// Map this map-point type to the corresponding combat/non-combat
    /// [`crate::runstate::RoomType`]. `Unknown` and `Unassigned` map to
    /// `RoomType::Unknown` (the catch-all) — the caller is expected to
    /// resolve `?` rooms via
    /// [`crate::runstate::RunState::resolve_unknown_room`] BEFORE the
    /// `enter_room` step; see
    /// [`crate::run_manager::RunManager::enter_room_for_point`] for the
    /// driver-side wiring.
    ///
    /// `Ancient` ↔ `RoomType::Event` matches Python's `run_state.py:1639`
    /// mapping. Bead `sts2-rs-2h627` (P9.2.h).
    #[must_use]
    pub const fn to_room_type(self) -> crate::runstate::RoomType {
        use crate::runstate::RoomType;
        match self {
            Self::Monster => RoomType::Monster,
            Self::Elite => RoomType::Elite,
            Self::Boss => RoomType::Boss,
            Self::Shop => RoomType::Shop,
            Self::RestSite => RoomType::RestSite,
            Self::Treasure => RoomType::Treasure,
            Self::Ancient => RoomType::Event,
            Self::Unknown | Self::Unassigned => RoomType::Unknown,
        }
    }
}

/// Node on the act-map grid.
///
/// `children` / `parents` are index handles into the parent
/// `ActMap`'s point list rather than Python's object refs. The
/// graph-walking methods (`add_child`, `remove_child`, `siblings`,
/// `has_child_at`) live on `ActMap` in P9.1.b so they can read both
/// endpoints atomically without aliasing.
#[derive(Debug, Clone, PartialEq)]
pub struct MapPoint {
    pub coord: MapCoord,
    pub point_type: MapPointType,
    pub children: Vec<MapPointIdx>,
    pub parents: Vec<MapPointIdx>,
    /// Visual horizontal offset for spreading (Python: `x_offset`).
    /// Map-generation only; combat/run logic should ignore.
    pub x_offset: f32,
}

impl MapPoint {
    /// Construct a fresh node at `coord` with no edges, unassigned type.
    ///
    /// Mirrors Python's `MapPoint(coord)` dataclass default — children/
    /// parents empty, `x_offset = 0.0`, `point_type = Unassigned`.
    #[must_use]
    pub fn new(coord: MapCoord) -> Self {
        Self {
            coord,
            point_type: MapPointType::Unassigned,
            children: Vec::new(),
            parents: Vec::new(),
            x_offset: 0.0,
        }
    }

    #[must_use]
    pub const fn col(&self) -> i32 {
        self.coord.col
    }

    #[must_use]
    pub const fn row(&self) -> i32 {
        self.coord.row
    }
}

/// Empty act-map graph container.
///
/// Stores nodes in a flat `Vec<MapPoint>` and indexes them by
/// [`MapCoord`] for O(1) lookup; adjacency lives on each
/// [`MapPoint`]'s `children`/`parents` (built via [`Self::add_child`]).
///
/// `start` / `boss` are `Option<MapPointIdx>` until the generator
/// fills them in; an empty `ActMap::new(n)` has neither set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ActMap {
    pub points: Vec<MapPoint>,
    /// Coord → index into `points`. Mirrors Python's `_grid` dict.
    /// `BTreeMap` over `HashMap` matches the rest of `sts2-core`
    /// (workspace `disallowed-methods` lint blocks std `HashMap`).
    pub grid: std::collections::BTreeMap<MapCoord, MapPointIdx>,
    pub start: Option<MapPointIdx>,
    pub boss: Option<MapPointIdx>,
    /// A10 DoubleBoss second-boss node — `None` everywhere except the
    /// FINAL act of an ascension-10 run. Mirrors C#
    /// `StandardActMap.SecondBossMapPoint` (`StandardActMap.cs:79-82`):
    /// a standalone Boss-typed point one row above the boss node,
    /// wired as the boss node's only child
    /// (`StandardActMap.cs:208-210` `BossMapPoint.AddChildPoint`), so
    /// the two boss fights run back-to-back. Attached AFTER generation
    /// + post-processing via [`Self::attach_second_boss`] — its
    /// creation consumes NO RNG in C# either (it is constructed
    /// directly, never drawn from the type pool). Bead sts2-rs-40ikw.2.
    pub second_boss: Option<MapPointIdx>,
    /// `num_rooms + 1` per Python: row 0 = start, rows 1..N = rooms,
    /// row N+1 = boss. Stored as the row index of the boss row.
    pub map_length: i32,
    /// Map-layer room markings. Coord → set of [`crate::RelicId`]s
    /// whose effect has marked that room. Bead `sts2-rs-mxbai`.
    ///
    /// **Purpose.** Some Python relics (`FurCoat`, …) pick a subset
    /// of rooms when first obtained and remember those coordinates
    /// for later combat-side consumers. In Python this state lives
    /// on the `RelicInstance` (e.g. `FurCoat._marked_coords`). The
    /// Rust port has no per-instance relic state by default, so the
    /// marking is lifted up to a small map-layer registry keyed by
    /// the marking relic. A future `before_combat_start` consumer
    /// (filed as a follow-up FurCoat bead) reads
    /// `is_marked_by(current_coord, RelicId::FurCoat)` and clamps
    /// enemy HP to 1 when true.
    ///
    /// **Storage shape.** `BTreeMap<MapCoord, BTreeSet<RelicId>>`:
    /// many relics can mark the same coord; one relic can mark many
    /// coords. `BTreeSet`/`BTreeMap` mirrors the rest of `sts2-core`'s
    /// determinism discipline (the workspace `disallowed-methods`
    /// lint blocks `HashMap`/`HashSet`).
    ///
    /// **Empty default.** Skipping serialization-shape entirely for
    /// the empty-map case stays parity-stable with Python's
    /// `_marked_coords: list = []` initial state on every fresh
    /// `ActMap`.
    pub marked_rooms:
        std::collections::BTreeMap<MapCoord, std::collections::BTreeSet<crate::RelicId>>,
}

impl ActMap {
    /// Construct an empty `ActMap` sized for `num_rooms` non-meta
    /// rows. `map_length = num_rooms + 1`, matching Python.
    #[must_use]
    pub fn new(num_rooms: i32) -> Self {
        Self {
            points: Vec::new(),
            grid: std::collections::BTreeMap::new(),
            start: None,
            boss: None,
            second_boss: None,
            map_length: num_rooms + 1,
            marked_rooms: std::collections::BTreeMap::new(),
        }
    }

    /// A10 DoubleBoss: add the second-boss node directly above the boss
    /// node and wire it as the boss node's child. Mirrors C#
    /// `StandardActMap` (`StandardActMap.cs:79-82` placement at
    /// `(columnCount / 2, rowCount + 1)`, `:208-210` edge, `:285-288`
    /// `PointType = Boss`). No-op when the boss node is unset or a
    /// second boss is already attached. Consumes no RNG — call it after
    /// generation/post-processing, gated by the run's `has_double_boss`
    /// flag on the FINAL act only (C# `RunManager.cs:501`). Bead
    /// sts2-rs-40ikw.2.
    pub fn attach_second_boss(&mut self) {
        if self.second_boss.is_some() {
            return;
        }
        let Some(boss_idx) = self.boss else {
            return;
        };
        let boss_coord = self.point(boss_idx).coord;
        let sb_idx = self.get_or_create(boss_coord.col, boss_coord.row + 1);
        self.point_mut(sb_idx).point_type = MapPointType::Boss;
        self.add_child(boss_idx, sb_idx);
        self.second_boss = Some(sb_idx);
    }

    /// Look up or insert the [`MapPoint`] at `(col, row)`. Returns
    /// its index; the node itself can be borrowed via
    /// [`Self::point`]/[`Self::point_mut`].
    pub fn get_or_create(&mut self, col: i32, row: i32) -> MapPointIdx {
        let coord = MapCoord::new(col, row);
        if let Some(&idx) = self.grid.get(&coord) {
            return idx;
        }
        let idx: MapPointIdx = self
            .points
            .len()
            .try_into()
            .expect("ActMap point count exceeds MapPointIdx (u16) range");
        self.points.push(MapPoint::new(coord));
        self.grid.insert(coord, idx);
        idx
    }

    /// Index a node by coordinate. Mirrors Python's
    /// `ActMap.get_point` (returns `None` for absent coords).
    #[must_use]
    pub fn get_point(&self, coord: MapCoord) -> Option<&MapPoint> {
        self.grid.get(&coord).map(|&i| &self.points[i as usize])
    }

    /// Borrow a node by its index. Panics if out of range — indices
    /// returned by [`Self::get_or_create`] are always valid.
    #[must_use]
    pub fn point(&self, idx: MapPointIdx) -> &MapPoint {
        &self.points[idx as usize]
    }

    pub fn point_mut(&mut self, idx: MapPointIdx) -> &mut MapPoint {
        &mut self.points[idx as usize]
    }

    /// All nodes on `row`, sorted by column. Mirrors Python's
    /// `ActMap.get_row`.
    #[must_use]
    pub fn get_row(&self, row: i32) -> Vec<&MapPoint> {
        let mut v: Vec<&MapPoint> = self.points.iter().filter(|p| p.row() == row).collect();
        v.sort_by_key(|p| p.col());
        v
    }

    /// Topmost (boss) row index. Equal to `map_length`.
    #[must_use]
    pub const fn top_row(&self) -> i32 {
        self.map_length
    }

    /// Bottom (start) row index — always 0 by construction.
    #[must_use]
    pub const fn bottom_row(&self) -> i32 {
        0
    }

    /// All nodes excluding the start/boss meta-rows. Mirrors
    /// Python's `ActMap.room_points`.
    #[must_use]
    pub fn room_points(&self) -> Vec<&MapPoint> {
        self.points
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let idx = *i as MapPointIdx;
                Some(idx) != self.start && Some(idx) != self.boss
            })
            .map(|(_, p)| p)
            .collect()
    }

    /// Direct children of the node at `coord`. Empty if `coord` is
    /// not in the grid or the node has no outgoing edges.
    #[must_use]
    pub fn children_of(&self, coord: MapCoord) -> Vec<&MapPoint> {
        let Some(p) = self.get_point(coord) else {
            return Vec::new();
        };
        p.children
            .iter()
            .map(|&i| &self.points[i as usize])
            .collect()
    }

    /// Every grid-live node at map row `row`, sorted by column.
    /// "Grid-live" drops pruned ghost nodes that linger in `points`
    /// but were removed from `grid` (see `prune_and_repair` / bead
    /// `sts2-rs-l8hcp`), matching the filter `next_map_children_cols`
    /// and `full_map_view` apply. Mirrors C# `Map.GetPointsInRow`,
    /// used by the `WingedBoots` off-path map-choice surface.
    #[must_use]
    pub fn points_in_row(&self, row: i32) -> Vec<&MapPoint> {
        let mut pts: Vec<&MapPoint> = self
            .grid
            .iter()
            .filter(|(coord, _)| coord.row == row)
            .map(|(_, &idx)| &self.points[idx as usize])
            .collect();
        pts.sort_unstable_by_key(|p| p.col());
        pts
    }

    // -- Room markings (bead `sts2-rs-mxbai`) --------------------------
    //
    // Marks are tagged by [`crate::RelicId`] so multiple relics can
    // own non-overlapping mark sets on the same map without aliasing.
    // The marking primitives are intentionally `coord`-keyed rather
    // than `idx`-keyed because Python's `FurCoat._marked_coords`
    // stores tuples and the future consumer reads off
    // `rs.visited_map_coords[-1]` — a coord, not an idx.

    /// Mark the room at `coord` as owned by `relic`. Idempotent — the
    /// underlying set's `insert` returns `false` on duplicates and
    /// this method is a no-op in that case. The marking is allowed
    /// even on coords that don't exist in `grid` (i.e. before a
    /// downstream generator fills the map); the marked-set is pure
    /// metadata and does not validate against the live point graph.
    /// Bead `sts2-rs-mxbai`.
    pub fn mark_room(&mut self, coord: MapCoord, relic: crate::RelicId) {
        self.marked_rooms.entry(coord).or_default().insert(relic);
    }

    /// Mark every coord in `coords` as owned by `relic`. Convenience
    /// wrapper around [`Self::mark_room`].
    pub fn mark_rooms<I>(&mut self, coords: I, relic: crate::RelicId)
    where
        I: IntoIterator<Item = MapCoord>,
    {
        for coord in coords {
            self.mark_room(coord, relic);
        }
    }

    /// Drop `relic`'s mark on `coord`. Returns `true` iff the mark
    /// was present. The set entry is removed entirely if the last
    /// owner is dropped, so an empty `marked_rooms` value never
    /// lingers in the map.
    pub fn unmark_room(&mut self, coord: MapCoord, relic: crate::RelicId) -> bool {
        let Some(set) = self.marked_rooms.get_mut(&coord) else {
            return false;
        };
        let removed = set.remove(&relic);
        if set.is_empty() {
            self.marked_rooms.remove(&coord);
        }
        removed
    }

    /// Clear every mark owned by `relic` across the entire map. Used
    /// when the marking relic decides to re-roll its mark set (e.g.
    /// FurCoat re-marks if its prior coords no longer point to
    /// MONSTER / ELITE rooms after a `modify_generated_map` pass).
    pub fn clear_marks_for(&mut self, relic: crate::RelicId) {
        self.marked_rooms.retain(|_coord, owners| {
            owners.remove(&relic);
            !owners.is_empty()
        });
    }

    /// `true` iff `relic` has marked `coord`.
    #[must_use]
    pub fn is_marked_by(&self, coord: MapCoord, relic: crate::RelicId) -> bool {
        self.marked_rooms
            .get(&coord)
            .is_some_and(|owners| owners.contains(&relic))
    }

    /// Every coord marked by `relic`, in deterministic
    /// (`BTreeMap`-iteration) order. Empty when the relic owns no
    /// marks.
    #[must_use]
    pub fn marked_coords_for(&self, relic: crate::RelicId) -> Vec<MapCoord> {
        self.marked_rooms
            .iter()
            .filter_map(|(coord, owners)| owners.contains(&relic).then_some(*coord))
            .collect()
    }

    /// Wire a directed edge `parent → child` and the reverse
    /// `parent` link on `child`. Idempotent: re-adding an existing
    /// edge is a no-op, matching Python's `MapPoint.add_child`.
    pub fn add_child(&mut self, parent: MapPointIdx, child: MapPointIdx) {
        if !self.points[parent as usize].children.contains(&child) {
            self.points[parent as usize].children.push(child);
        }
        if !self.points[child as usize].parents.contains(&parent) {
            self.points[child as usize].parents.push(parent);
        }
    }

    /// Drop the directed edge `parent → child` and the reverse
    /// link. No-op if the edge isn't present. Mirrors C#
    /// `MapPoint.RemoveChildPoint` which clears both endpoints.
    /// Bead `sts2-rs-l8hcp` (MapPathPruning port).
    pub fn remove_child(&mut self, parent: MapPointIdx, child: MapPointIdx) {
        self.points[parent as usize]
            .children
            .retain(|&c| c != child);
        self.points[child as usize].parents.retain(|&p| p != parent);
    }

    /// Remove a node from the live map: clear its grid slot and drop
    /// every incident edge (both directions). The underlying
    /// `MapPoint` entry stays in [`Self::points`] so existing index
    /// handles remain valid — callers test live-ness with
    /// [`Self::is_in_map`] (mirrors C# `IsInMap`).
    ///
    /// Used by `MapPathPruning::PruneSegment` to delete duplicate-path
    /// nodes in-place. Bead `sts2-rs-l8hcp`.
    pub fn remove_point(&mut self, idx: MapPointIdx) {
        let coord = self.points[idx as usize].coord;
        // Drop grid slot iff it still points at us. C# nulls the slot
        // unconditionally; our equality guard prevents accidentally
        // clearing a re-used coord (defensive — should never trigger).
        if self.grid.get(&coord) == Some(&idx) {
            self.grid.remove(&coord);
        }
        let children: Vec<MapPointIdx> = self.points[idx as usize].children.clone();
        for child in children {
            self.remove_child(idx, child);
        }
        let parents: Vec<MapPointIdx> = self.points[idx as usize].parents.clone();
        for parent in parents {
            self.remove_child(parent, idx);
        }
    }

    /// `true` iff `idx` currently has a grid slot pointing at it.
    /// C# `MapPathPruning.IsInMap` exempts Boss/Ancient meta-nodes;
    /// expose that policy here so callers don't have to repeat it.
    /// Bead `sts2-rs-l8hcp`.
    #[must_use]
    pub fn is_in_map(&self, idx: MapPointIdx) -> bool {
        let coord = self.points[idx as usize].coord;
        if self.grid.get(&coord) == Some(&idx) {
            return true;
        }
        let pt = self.points[idx as usize].point_type;
        matches!(pt, MapPointType::Boss | MapPointType::Ancient)
    }

    /// Direct children of the node at `idx`, by index. Companion to
    /// [`Self::children_of`] for callers that already hold an idx
    /// (UI peek listeners, [`GoldenCompass`]). Bead
    /// `sts2-rs-70anj.3.18` (P9.0 map plumbing).
    #[must_use]
    pub fn children_idx_of(&self, idx: MapPointIdx) -> &[MapPointIdx] {
        &self.points[idx as usize].children
    }

    /// Single-child peek: returns the sole next-room idx when the
    /// node at `idx` has exactly one outgoing edge (true on the
    /// thin-slice linear chain). Returns `None` on branch points or
    /// terminal nodes — those callers must enumerate
    /// [`Self::children_idx_of`] and rely on a `MapChoice` phase.
    /// Bead `sts2-rs-70anj.3.18`.
    #[must_use]
    pub fn peek_next_room(&self, idx: MapPointIdx) -> Option<MapPointIdx> {
        let kids = &self.points[idx as usize].children;
        match kids.len() {
            1 => Some(kids[0]),
            _ => None,
        }
    }

    /// `true` iff `idx` references this act's boss meta-point.
    /// Mirrors Python's `floor.is_boss_room` snapshot used by
    /// boss-gated relic logic (e.g. `FurCoat`, post-boss rewards).
    /// Bead `sts2-rs-70anj.3.18`.
    #[must_use]
    pub fn is_boss_room(&self, idx: MapPointIdx) -> bool {
        Some(idx) == self.boss
    }

    /// `point_type` at `idx`. Convenience for listener reads that
    /// already hold an idx (avoids the coord round-trip). Bead
    /// `sts2-rs-70anj.3.18`.
    #[must_use]
    pub fn point_type_at(&self, idx: MapPointIdx) -> MapPointType {
        self.points[idx as usize].point_type
    }

    /// Build the P9.0 thin-slice stand-in: a single-column linear
    /// chain `start → Monster → Monster → Boss` for vertical-slice
    /// integration before the real generator (P9.2.b) lands.
    ///
    /// Layout: column 0, rows 0..=3.
    /// - row 0: start meta-point (`Unassigned`)
    /// - row 1: `Monster`
    /// - row 2: `Monster`
    /// - row 3: `Boss` (also `map_length`)
    ///
    /// The `MapChoice` phase trivially picks 'advance' since each
    /// node has exactly one child. Bead `sts2-rs-5fsbh` (P9.0.e).
    #[must_use]
    pub fn linear_thin_slice() -> Self {
        let mut m = Self::new(3);
        let start = m.get_or_create(0, 0);
        let r1 = m.get_or_create(0, 1);
        let r2 = m.get_or_create(0, 2);
        let boss = m.get_or_create(0, 3);
        m.point_mut(r1).point_type = MapPointType::Monster;
        m.point_mut(r2).point_type = MapPointType::Monster;
        m.point_mut(boss).point_type = MapPointType::Boss;
        m.add_child(start, r1);
        m.add_child(r1, r2);
        m.add_child(r2, boss);
        m.start = Some(start);
        m.boss = Some(boss);
        m
    }
}

/// One node in a [`FullMapView`]. `children` are the live forward edges as
/// raw coords; `visited`/`current` are run-scope flags filled by the builder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullMapNodeView {
    pub coord: MapCoord,
    pub point_type: MapPointType,
    pub children: Vec<MapCoord>,
    pub visited: bool,
    pub current: bool,
}

/// The boss meta-node, surfaced separately from [`FullMapView::rows`] to
/// mirror the C# `full_map` schema (`RunSimulator.cs:3876`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullMapBossView {
    pub coord: MapCoord,
    pub point_type: MapPointType,
}

/// Render-ready view of an [`ActMap`]: nodes grouped into dense, row-ordered
/// rows (boss excluded), plus the boss meta-node and the player's current
/// coord. Mirrors the C# `full_map` JSON the web UI consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullMapView {
    pub rows: Vec<Vec<FullMapNodeView>>,
    pub boss: Option<FullMapBossView>,
    pub current_coord: Option<MapCoord>,
}

impl ActMap {
    /// Build a [`FullMapView`] for UI rendering. `current` is the player's
    /// coord (flags the `current` node + sets `current_coord`); `visited` is
    /// the run's visited-coord history (flags `visited` per node). Children
    /// are filtered to live grid nodes or the boss meta-node, exactly as the
    /// `act_map()` PyO3 dump does, so pruned ghosts never leak as edges.
    #[must_use]
    pub fn full_map_view(&self, current: Option<MapCoord>, visited: &[MapCoord]) -> FullMapView {
        use std::collections::BTreeMap;

        let boss_coord = self.boss.map(|b| self.point(b).coord);

        let child_is_live =
            |c: &MapCoord| -> bool { self.grid.contains_key(c) || Some(*c) == boss_coord };

        // Group live grid nodes by row into a dense, row-ordered Vec.
        let mut by_row: BTreeMap<i32, Vec<FullMapNodeView>> = BTreeMap::new();
        for (coord, &idx) in &self.grid {
            // The boss is surfaced separately as a meta-node, never as a row
            // entry. Some generator paths (`generate_act_map`) insert the boss
            // into `grid`; the parity-matched runtime map keeps it grid-exempt.
            // Skipping by index handles both layouts.
            if Some(idx) == self.boss {
                continue;
            }
            let p = self.point(idx);
            let children: Vec<MapCoord> = p
                .children
                .iter()
                .map(|&ci| self.point(ci).coord)
                .filter(child_is_live)
                .collect();
            by_row.entry(coord.row).or_default().push(FullMapNodeView {
                coord: *coord,
                point_type: p.point_type,
                children,
                visited: visited.contains(coord),
                current: current == Some(*coord),
            });
        }

        // The grid BTreeMap iterates col-major (MapCoord Ord: col first), but
        // by_row is keyed on the i32 row, so it's row-ordered; flatten into a
        // dense Vec<Vec<_>> indexed 0..=max_row for bottom-up UI rendering.
        let max_row = by_row.keys().copied().max().unwrap_or(0);
        let mut rows: Vec<Vec<FullMapNodeView>> = (0..=max_row).map(|_| Vec::new()).collect();
        for (row, nodes) in by_row {
            let r = usize::try_from(row).expect("map row must be non-negative");
            rows[r] = nodes;
        }

        let boss = self.boss.map(|b| {
            let p = self.point(b);
            FullMapBossView {
                coord: p.coord,
                point_type: p.point_type,
            }
        });

        FullMapView {
            rows,
            boss,
            current_coord: current,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn full_map_view_groups_rows_and_flags_current_visited() {
        use crate::dotnet_rng::Rng;
        use crate::map_generator::generate_act_map;

        let mut rng = Rng::new(42);
        let act = generate_act_map(15, &mut rng, 0, false, 0, None);

        // Pick a real row-1 entrance node as "current" and mark it visited.
        let entry = act
            .grid
            .keys()
            .find(|c| c.row == 1)
            .copied()
            .expect("row-1 node exists");
        let visited = vec![
            MapCoord { col: 3, row: 0 }, // stale entrance history entry — need not be a live grid node
            entry,
        ];

        let view = act.full_map_view(Some(entry), &visited);

        // rows[0] is the entrance row (row 0), rows are dense and row-ordered.
        assert!(view.rows.len() >= 2, "at least entrance + row 1");
        for (i, row) in view.rows.iter().enumerate() {
            for n in row {
                assert_eq!(n.coord.row as usize, i, "row {i} holds only row-{i} nodes");
            }
        }

        // Boss is surfaced separately, never inside `rows`.
        let boss = view.boss.expect("boss present");
        assert_eq!(boss.point_type, MapPointType::Boss);
        assert!(
            view.rows.iter().flatten().all(|n| n.coord != boss.coord),
            "boss meta-node is excluded from rows",
        );

        // current_coord echoes the argument; the chosen node is flagged.
        assert_eq!(view.current_coord, Some(entry));
        let flagged = view
            .rows
            .iter()
            .flatten()
            .find(|n| n.coord == entry)
            .unwrap();
        assert!(flagged.current && flagged.visited);

        // children only reference live grid nodes or the boss.
        for n in view.rows.iter().flatten() {
            for ch in &n.children {
                let live = act.grid.contains_key(ch);
                let is_boss = view.boss.as_ref().is_some_and(|b| b.coord == *ch);
                assert!(live || is_boss, "child {ch:?} must be live or boss");
            }
        }
    }

    #[test]
    fn map_point_type_to_room_type_round_trip() {
        use crate::runstate::RoomType;
        assert_eq!(MapPointType::Monster.to_room_type(), RoomType::Monster);
        assert_eq!(MapPointType::Elite.to_room_type(), RoomType::Elite);
        assert_eq!(MapPointType::Boss.to_room_type(), RoomType::Boss);
        assert_eq!(MapPointType::Shop.to_room_type(), RoomType::Shop);
        assert_eq!(MapPointType::RestSite.to_room_type(), RoomType::RestSite);
        assert_eq!(MapPointType::Treasure.to_room_type(), RoomType::Treasure);
        assert_eq!(MapPointType::Ancient.to_room_type(), RoomType::Event);
        // Unknown / Unassigned both fall to the engine-side catch-all;
        // callers route Unknown through the resolver before mapping.
        assert_eq!(MapPointType::Unknown.to_room_type(), RoomType::Unknown);
        assert_eq!(MapPointType::Unassigned.to_room_type(), RoomType::Unknown);
    }

    #[test]
    fn mapcoord_eq_and_hash() {
        let a = MapCoord::new(2, 5);
        let b = MapCoord::new(2, 5);
        let c = MapCoord::new(2, 6);
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }

    #[test]
    fn mapedge_construct_and_eq() {
        let e1 = MapEdge::new(MapCoord::new(0, 0), MapCoord::new(0, 1));
        let e2 = MapEdge::new(MapCoord::new(0, 0), MapCoord::new(0, 1));
        let e3 = MapEdge::new(MapCoord::new(0, 1), MapCoord::new(0, 0));
        assert_eq!(e1, e2);
        assert_ne!(e1, e3, "edges are directed: src/dst swap is not equal");
    }

    #[test]
    fn mappoint_default_construction() {
        let p = MapPoint::new(MapCoord::new(3, 4));
        assert_eq!(p.coord, MapCoord::new(3, 4));
        assert_eq!(p.point_type, MapPointType::Unassigned);
        assert!(p.children.is_empty());
        assert!(p.parents.is_empty());
        assert!((p.x_offset - 0.0).abs() < f32::EPSILON);
        assert_eq!(p.col(), 3);
        assert_eq!(p.row(), 4);
    }

    #[test]
    fn mappoint_type_default_is_unassigned() {
        assert_eq!(MapPointType::default(), MapPointType::Unassigned);
    }

    #[test]
    fn is_combat_covers_monster_elite_boss_only() {
        use MapPointType::*;
        for t in [Monster, Elite, Boss] {
            assert!(t.is_combat(), "{t:?} should be combat");
        }
        for t in [Unassigned, Unknown, Shop, RestSite, Treasure, Ancient] {
            assert!(!t.is_combat(), "{t:?} should not be combat");
        }
    }

    #[test]
    fn singleton_predicates_are_singletons() {
        use MapPointType::*;
        assert!(Shop.is_merchant() && !Monster.is_merchant());
        assert!(RestSite.is_rest_site() && !Shop.is_rest_site());
        assert!(Boss.is_boss() && !Elite.is_boss());
        assert!(Elite.is_elite() && !Monster.is_elite());
        assert!(Treasure.is_treasure() && !Ancient.is_treasure());
        assert!(Unknown.is_unknown() && !Unassigned.is_unknown());
        assert!(Ancient.is_event() && !Unknown.is_event());
    }

    #[test]
    fn is_assigned_inverts_unassigned() {
        use MapPointType::*;
        assert!(!Unassigned.is_assigned());
        for t in [
            Unknown, Monster, Elite, Boss, Shop, RestSite, Treasure, Ancient,
        ] {
            assert!(t.is_assigned(), "{t:?} should be assigned");
        }
    }

    #[test]
    fn actmap_new_is_empty() {
        let m = ActMap::new(15);
        assert_eq!(m.map_length, 16);
        assert!(m.points.is_empty());
        assert!(m.grid.is_empty());
        assert!(m.start.is_none());
        assert!(m.boss.is_none());
        assert_eq!(m.bottom_row(), 0);
        assert_eq!(m.top_row(), 16);
    }

    #[test]
    fn get_or_create_is_idempotent_per_coord() {
        let mut m = ActMap::new(5);
        let a = m.get_or_create(3, 1);
        let b = m.get_or_create(3, 1);
        let c = m.get_or_create(4, 1);
        assert_eq!(a, b, "same coord returns same idx");
        assert_ne!(a, c, "different coord returns different idx");
        assert_eq!(m.points.len(), 2);
        assert_eq!(m.get_point(MapCoord::new(3, 1)).unwrap().coord.col, 3);
        assert!(m.get_point(MapCoord::new(9, 9)).is_none());
    }

    #[test]
    fn add_child_wires_both_directions_and_dedupes() {
        let mut m = ActMap::new(5);
        let p = m.get_or_create(3, 0);
        let c = m.get_or_create(3, 1);
        m.add_child(p, c);
        m.add_child(p, c);
        assert_eq!(m.point(p).children, vec![c]);
        assert_eq!(m.point(c).parents, vec![p]);
    }

    #[test]
    fn get_row_returns_sorted_by_col() {
        let mut m = ActMap::new(3);
        m.get_or_create(5, 2);
        m.get_or_create(1, 2);
        m.get_or_create(3, 2);
        m.get_or_create(0, 1); // different row, should be excluded
        let row2 = m.get_row(2);
        assert_eq!(row2.len(), 3);
        let cols: Vec<i32> = row2.iter().map(|p| p.col()).collect();
        assert_eq!(cols, vec![1, 3, 5]);
    }

    #[test]
    fn room_points_excludes_start_and_boss() {
        let mut m = ActMap::new(2);
        let s = m.get_or_create(3, 0);
        let r = m.get_or_create(3, 1);
        let b = m.get_or_create(3, 2);
        m.start = Some(s);
        m.boss = Some(b);
        let rooms = m.room_points();
        assert_eq!(rooms.len(), 1);
        assert_eq!(rooms[0].coord, m.point(r).coord);
    }

    #[test]
    fn points_in_row_returns_all_grid_live_cols_sorted() {
        // Row 1 has three nodes at cols 4, 0, 6 (insertion order
        // deliberately unsorted); row 2 has one. WingedBoots offers
        // every row-1 node regardless of edges, so `points_in_row`
        // must surface all three in col order — not just children.
        let mut m = ActMap::new(3);
        let s = m.get_or_create(3, 0);
        let a = m.get_or_create(4, 1);
        let _b = m.get_or_create(0, 1);
        let _c = m.get_or_create(6, 1);
        let d = m.get_or_create(6, 2);
        m.start = Some(s);
        // Only one edge wired (3,0)->(4,1); the other row-1 nodes are
        // unconnected but still grid-live.
        m.add_child(s, a);
        m.add_child(a, d);
        let cols: Vec<i32> = m.points_in_row(1).iter().map(|p| p.col()).collect();
        assert_eq!(cols, vec![0, 4, 6], "all row-1 cols, sorted");
        // Connected children of the start are only col 4.
        let child_cols: Vec<i32> = m
            .children_of(MapCoord::new(3, 0))
            .iter()
            .map(|p| p.col())
            .collect();
        assert_eq!(child_cols, vec![4], "children are a strict subset");
        assert!(m.points_in_row(9).is_empty(), "empty row");
    }

    #[test]
    fn children_of_returns_outgoing_neighbors() {
        let mut m = ActMap::new(3);
        let p = m.get_or_create(3, 0);
        let c1 = m.get_or_create(2, 1);
        let c2 = m.get_or_create(4, 1);
        m.add_child(p, c1);
        m.add_child(p, c2);
        let kids = m.children_of(MapCoord::new(3, 0));
        let mut cols: Vec<i32> = kids.iter().map(|p| p.col()).collect();
        cols.sort_unstable();
        assert_eq!(cols, vec![2, 4]);
        assert!(m.children_of(MapCoord::new(9, 9)).is_empty());
    }

    #[test]
    fn golden_path_like_chain_constructs() {
        // Mirrors generate_golden_path_map's column-3 spine.
        let mut m = ActMap::new(3);
        let s = m.get_or_create(3, 0);
        let r1 = m.get_or_create(3, 1);
        let r2 = m.get_or_create(3, 2);
        let b = m.get_or_create(3, 3);
        m.start = Some(s);
        m.boss = Some(b);
        m.add_child(s, r1);
        m.add_child(r1, r2);
        m.add_child(r2, b);
        assert_eq!(m.point(s).children, vec![r1]);
        assert_eq!(m.point(b).parents, vec![r2]);
        assert_eq!(m.room_points().len(), 2);
    }

    #[test]
    fn mappoint_clone_is_independent() {
        let mut a = MapPoint::new(MapCoord::new(0, 0));
        a.children.push(1);
        a.parents.push(2);
        a.x_offset = 1.5;
        a.point_type = MapPointType::Monster;
        let b = a.clone();
        a.children.push(7);
        assert_eq!(b.children, vec![1]);
        assert_eq!(b.parents, vec![2]);
        assert_eq!(b.point_type, MapPointType::Monster);
    }

    #[test]
    fn linear_thin_slice_has_four_nodes_in_chain() {
        let m = ActMap::linear_thin_slice();
        assert_eq!(m.points.len(), 4);
        assert_eq!(m.map_length, 4);
        let start = m.start.expect("start set");
        let boss = m.boss.expect("boss set");
        assert_eq!(m.point(start).coord, MapCoord::new(0, 0));
        assert_eq!(m.point(boss).coord, MapCoord::new(0, 3));
        assert_eq!(m.point(boss).point_type, MapPointType::Boss);
    }

    #[test]
    fn linear_thin_slice_rooms_are_two_monsters() {
        let m = ActMap::linear_thin_slice();
        let rooms = m.room_points();
        assert_eq!(rooms.len(), 2);
        assert!(rooms.iter().all(|p| p.point_type == MapPointType::Monster));
        let rows: Vec<i32> = rooms.iter().map(|p| p.row()).collect();
        assert!(rows.contains(&1) && rows.contains(&2));
    }

    #[test]
    fn linear_thin_slice_each_node_has_single_child_until_boss() {
        let m = ActMap::linear_thin_slice();
        for row in 0..3 {
            let kids = m.children_of(MapCoord::new(0, row));
            assert_eq!(kids.len(), 1, "row {row} should have exactly one child");
        }
        assert!(m.children_of(MapCoord::new(0, 3)).is_empty());
    }

    #[test]
    fn linear_thin_slice_advance_traversal_reaches_boss() {
        // MapChoice phase will trivially pick the single child each
        // step; verify the chain is walkable start → … → boss.
        let m = ActMap::linear_thin_slice();
        let mut cur = m.point(m.start.unwrap()).coord;
        for _ in 0..3 {
            let kids = m.children_of(cur);
            assert_eq!(kids.len(), 1);
            cur = kids[0].coord;
        }
        assert_eq!(cur, m.point(m.boss.unwrap()).coord);
    }

    #[test]
    fn peek_next_room_single_child_returns_some() {
        let m = ActMap::linear_thin_slice();
        let start = m.start.unwrap();
        let next = m.peek_next_room(start).expect("single child on thin slice");
        assert_eq!(m.point(next).coord, MapCoord::new(0, 1));
    }

    #[test]
    fn peek_next_room_terminal_returns_none() {
        let m = ActMap::linear_thin_slice();
        let boss = m.boss.unwrap();
        assert!(m.peek_next_room(boss).is_none());
    }

    #[test]
    fn peek_next_room_branch_returns_none() {
        let mut m = ActMap::new(2);
        let p = m.get_or_create(0, 0);
        let a = m.get_or_create(0, 1);
        let b = m.get_or_create(1, 1);
        m.add_child(p, a);
        m.add_child(p, b);
        assert!(
            m.peek_next_room(p).is_none(),
            "branch points must require a choice"
        );
    }

    #[test]
    fn is_boss_room_flags_boss_idx_only() {
        let m = ActMap::linear_thin_slice();
        let start = m.start.unwrap();
        let boss = m.boss.unwrap();
        assert!(m.is_boss_room(boss));
        assert!(!m.is_boss_room(start));
        for p in m.room_points() {
            let idx = m.grid[&p.coord];
            assert!(!m.is_boss_room(idx));
        }
    }

    #[test]
    fn point_type_at_matches_node_point_type() {
        let m = ActMap::linear_thin_slice();
        let boss = m.boss.unwrap();
        assert_eq!(m.point_type_at(boss), MapPointType::Boss);
        let r1 = m.grid[&MapCoord::new(0, 1)];
        assert_eq!(m.point_type_at(r1), MapPointType::Monster);
    }

    #[test]
    fn children_idx_of_matches_children_of() {
        let m = ActMap::linear_thin_slice();
        let start = m.start.unwrap();
        let by_idx = m.children_idx_of(start).to_vec();
        let by_coord: Vec<MapPointIdx> = m
            .children_of(m.point(start).coord)
            .iter()
            .map(|p| m.grid[&p.coord])
            .collect();
        assert_eq!(by_idx, by_coord);
    }

    // -- Room markings (bead `sts2-rs-mxbai`) --------------------------

    #[test]
    fn marked_rooms_starts_empty() {
        let m = ActMap::new(15);
        assert!(m.marked_rooms.is_empty());
        assert!(!m.is_marked_by(MapCoord::new(0, 0), crate::RelicId::FurCoat));
        assert!(m.marked_coords_for(crate::RelicId::FurCoat).is_empty());
    }

    #[test]
    fn mark_room_records_ownership_and_is_idempotent() {
        let mut m = ActMap::new(15);
        let c = MapCoord::new(2, 5);
        m.mark_room(c, crate::RelicId::FurCoat);
        assert!(m.is_marked_by(c, crate::RelicId::FurCoat));
        // Re-mark is a no-op.
        m.mark_room(c, crate::RelicId::FurCoat);
        assert_eq!(m.marked_coords_for(crate::RelicId::FurCoat), vec![c]);
    }

    #[test]
    fn mark_rooms_bulk_inserts_all_coords() {
        let mut m = ActMap::new(15);
        let coords = vec![
            MapCoord::new(0, 1),
            MapCoord::new(3, 2),
            MapCoord::new(6, 4),
        ];
        m.mark_rooms(coords.clone(), crate::RelicId::FurCoat);
        let mut got = m.marked_coords_for(crate::RelicId::FurCoat);
        got.sort();
        let mut want = coords.clone();
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn multiple_relics_can_mark_same_coord_without_aliasing() {
        let mut m = ActMap::new(15);
        let c = MapCoord::new(1, 2);
        m.mark_room(c, crate::RelicId::FurCoat);
        m.mark_room(c, crate::RelicId::GoldenCompass);
        assert!(m.is_marked_by(c, crate::RelicId::FurCoat));
        assert!(m.is_marked_by(c, crate::RelicId::GoldenCompass));
        assert!(!m.is_marked_by(c, crate::RelicId::BurningBlood));
    }

    #[test]
    fn unmark_room_removes_owner_and_returns_prev_present() {
        let mut m = ActMap::new(15);
        let c = MapCoord::new(2, 2);
        m.mark_room(c, crate::RelicId::FurCoat);
        m.mark_room(c, crate::RelicId::GoldenCompass);

        assert!(m.unmark_room(c, crate::RelicId::FurCoat));
        assert!(!m.is_marked_by(c, crate::RelicId::FurCoat));
        assert!(m.is_marked_by(c, crate::RelicId::GoldenCompass));

        // Dropping the last owner removes the coord entry entirely so
        // an empty BTreeSet never lingers.
        assert!(m.unmark_room(c, crate::RelicId::GoldenCompass));
        assert!(!m.marked_rooms.contains_key(&c));

        // Re-unmark on absent coord returns false.
        assert!(!m.unmark_room(c, crate::RelicId::FurCoat));
    }

    #[test]
    fn clear_marks_for_only_removes_target_relic() {
        let mut m = ActMap::new(15);
        let c1 = MapCoord::new(0, 1);
        let c2 = MapCoord::new(3, 4);
        m.mark_room(c1, crate::RelicId::FurCoat);
        m.mark_room(c1, crate::RelicId::GoldenCompass);
        m.mark_room(c2, crate::RelicId::FurCoat);

        m.clear_marks_for(crate::RelicId::FurCoat);

        assert!(m.marked_coords_for(crate::RelicId::FurCoat).is_empty());
        assert!(m.is_marked_by(c1, crate::RelicId::GoldenCompass));
        // c2 had only FurCoat — entry should be gone entirely.
        assert!(!m.marked_rooms.contains_key(&c2));
    }

    #[test]
    fn marked_coords_for_iterates_in_btreemap_order() {
        let mut m = ActMap::new(15);
        // Inserted out of order.
        m.mark_room(MapCoord::new(3, 2), crate::RelicId::FurCoat);
        m.mark_room(MapCoord::new(0, 1), crate::RelicId::FurCoat);
        m.mark_room(MapCoord::new(6, 4), crate::RelicId::FurCoat);
        let got = m.marked_coords_for(crate::RelicId::FurCoat);
        // BTreeMap iterates by key — MapCoord's `Ord` is row-major
        // via its derive (col, row); confirm the sequence is
        // deterministic and sorted.
        let mut sorted = got.clone();
        sorted.sort();
        assert_eq!(
            got, sorted,
            "marked_coords_for must yield BTreeMap-sorted order"
        );
        assert_eq!(got.len(), 3);
    }
}
