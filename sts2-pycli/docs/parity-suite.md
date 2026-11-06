# The C#-vs-Rust parity suite

This document explains how the parity suite works: what it compares, how it is
driven, the tools that make it up, and how to run them. It is the map for anyone
chasing a "does the Rust engine match the game?" question.

## 1. What "parity" means here

`sts2-rs` is a from-scratch Rust reimplementation of the Slay the Spire 2 combat
and run engine. The **authoritative behavior** is the decompiled C# game
(`sts2-wiki/decompiled/v0.103.2/`), run headlessly via
`sts2-pycli/src/Sts2Headless/`. The parity suite's job is to prove the Rust
engine produces **bit-identical state** to the C# game given the same seed and
the same sequence of actions — and, when it doesn't, to point a finger at the
*first* place they diverged and the card/monster/encounter responsible.

Two engines are driven in lockstep:

- **C# (the oracle / reference):** the real game logic, driven through a .NET
  headless subprocess (`sts2_client.Game`) over stdio, one JSON command at a
  time. Authoritative but slow (IPC + .NET startup per trial).
- **Rust (under test):** the `sts2_rs` PyO3 extension built out of
  `../sts2-rs/crates/sts2-py`. Microsecond-fast; mirrors the C#-chosen actions.

The action sequence is **C#-driven** (`rng.choice(cs_legal_actions)`); Rust only
ever replays the same actions. This makes the C# half a pure function of
`(scenario, seed, max_turns)` — the foundation for the trace cache (§6).

## 2. The state surface — what actually gets compared

`python/parity_fuzz_state.py` defines the comparison:

- `normalize_csharp(state)` reshapes a C# `get_state` response into a canonical
  dict.
- `normalize_rust(state)` reshapes a `sts2_rs.PyCombatState` (via its
  `state_to_dict()`) into the **same** canonical dict.
- `diff(a, b)` recursively, structurally compares two canonical dicts and
  returns a list of `{path, csharp, rust}` divergence records.

The canonical state carries:

- **player:** hp, max_hp, block, energy, powers
- **monsters:** per-slot hp, alive, block, powers
- **piles:** hand / draw / discard / exhaust — both **counts** and **ordered
  card-id lists** (the ordered lists are what `parity_audit` adds on top of raw
  `parity_fuzz`; without them a shuffle/draw-order RNG bug is invisible until it
  cascades into a damage diff)
- **per-hand-card resolved cost** (`hand_costs` — catches Confused / cost
  reduction / cost-override bugs)
- turn, relics, combat_terminal, player_won

> **Known NOT compared:** stance, orb slots, per-card afflictions, and intent
> are not in the canonical surface. If you chase a divergence that lives only in
> one of those, the suite will only catch it once it cascades into hp/block/
> powers. Extending the surface is the lever (see item#1 history in §7).

## 3. The classification taxonomy

`parity_audit.py` classifies each **first** divergence into a category so real
engine bugs are separated from cascade noise (a single early draw-order bug
otherwise lights up dozens of downstream hp/block diffs):

| Category | Meaning |
| --- | --- |
| `DRAW_ORDER` | ordered card-id list diverges, counts equal, hands matched going in → shuffle/draw RNG order bug (high value, single root) |
| `PILE_COUNT` | a pile *count* diverges with matching hands → a card moved to the wrong pile |
| `EXHAUST_FATE` | exhaust-pile membership diverges → discard-vs-exhaust routing bug |
| `HAND_COST` | a hand card's resolved cost diverges → cost-override / Confused / reduction bug |
| `PLAYER_HP` | player.hp diverges after end_turn → monster intent / damage formula bug |
| `PLAYER_STATE` | player.block/energy/powers diverges after a play_card → card-effect bug (attributed to the card) |
| `MONSTER_STATE` | monster hp/alive/block/powers diverges → card-damage or monster bug (attributed to the card) |
| `INVARIANT` | the Rust engine tripped an internal `check_combat_invariants` assertion (opt-in, `--invariants`) |
| `UPGRADE_CHOICE` | hand_ids diverge right after an upgrade-choice card (ARMAMENTS…); known-acceptable auto-pick asymmetry |
| `CASCADE` | hands already differed BEFORE this action → downstream of an earlier (undetected) bug |

A clean sweep reports only the `clean` bucket. Anything else is a triage
candidate (real bug, RNG-branch artifact, or coverage gap).

## 4. The main driver — `parity_audit.py`

This is the entry point for the random-seed sweep. Two sweep modes:

- `--sweep v1` (default): the original fuzz — `--trials N` random scenarios.
- `--sweep v2`: the structured sweep, assembled by `assemble_v2_trials`.

### v2 axes and tiers

**Axes** (`--axes`, comma-separated, default all four):

- `upgraded` — every card once, base and upgraded, vs a clean arena.
- `arenas` — card × arena cross-product (full cross is a nightly/full-tier axis).
- `fuzz` — N seeded random scenarios. N scales with tier.
- `pairwise` — Ironclad-only hook-interaction pairs (`HOOK_GROUPS` taxonomy;
  silently skipped for other characters).

**Tiers** (`--tier`) set the `fuzz` seed count: `smoke`=5, `quick`=500,
`mid`=2000, `full`=5000.

**Character** (`--sweep-character`, default `Ironclad`) parameterizes the
per-class axes to that character's pool. `Ironclad` reproduces the original
sweep byte-for-byte, keeping the committed trace cache valid.

### Depth and breadth knobs

- `--max-turns N` (default 6): how many turns each trial runs. Raise it (e.g.
  30) to reach late-combat scaling bugs (Demon Form / Juggernaut / Barricade
  stacks) the 6-turn horizon never reaches. Part of the trace-cache key.
- `--paths K` (default 1): independent action trajectories per scenario. Each
  path gets a distinct action seed, so the SAME scenario is explored down K
  different random play-lines — surfacing combos one random path misses.

### Coverage reporting

- `--coverage-report` writes `coverage_report.json`: which pool cards were
  actually **played** (not merely in-deck), plus powers/monsters seen.
- `--coverage-min N` turns it into a gate (exit 2 if fewer than N% of the pool
  was exercised).

### Invariant checking

- `--invariants` (default OFF) runs the Rust engine's `check_combat_invariants`
  after each step (HP/block/energy bounds, PowerStore integrity, deck-partition
  conservation, hand cap, enemy mirror). Off by default because the synthetic
  scenario builder trips a few pre-existing artifacts (block-mirror desync,
  instanced-power "dups") that would drown the signal.

### Typical invocations

```bash
# always go through the wheel venv so the right sts2_rs.so is on the path
scripts/wheel-venv.sh run -- python python/parity_audit.py --sweep v2 --tier quick --sweep-character Ironclad
scripts/wheel-venv.sh run -- python python/parity_audit.py --sweep v2 --tier full --axes fuzz --max-turns 30
scripts/wheel-venv.sh run -- python python/parity_audit.py --coverage    # single-variable content coverage sweep
```

## 5. Freshness gates — the two ways a sweep silently lies

A sweep run against the WRONG binary on either side produces confident garbage:
a real fix looks like a no-op, or a real bug looks fixed. Two symmetric gates,
run at the top of `main()`, abort instead:

- **`assert_wheel_fresh()`** (Rust side): aborts if the installed `sts2_rs`
  `.so` was not compiled from the current Rust source. Backed by
  `sts2_rs.src_hash()` and a content probe. See
  `feedback_maturin_reuses_stale_core_artifacts` in agent memory for the full
  staleness history.
- **`assert_oracle_fresh()`** (C# side): aborts if any `src/**/*.cs` is newer
  than the built `Sts2Headless.dll`. This closes the **stale-binary trap**: the
  trace cache is namespaced by a hash of the `.cs` SOURCE (so a recorded bundle
  replays across checkouts at the same commit), which means a stale built binary
  serves an OLD oracle under a namespace that *claims* freshness. This bit a
  full session — #23 added `hand_costs` to `RunSimulator.GetStateImpl`, but the
  on-disk dll predated the merge, so every sweep on main silently compared
  against a pre-#23 oracle. **Always `just build-cs` after pulling C# changes.**
  Bypass with `STS2_SKIP_ORACLE_CHECK=1` (e.g. mid C#-harness edit).

## 6. The trace cache

`--sweep` is process-bound on the C# reference: every trial spins a .NET
subprocess and drives it over stdio. But the C# half of a trial is a pure
function of `(scenario, seed, max_turns)`, so the suite **records** each trial's
full C# reference trace (init dump + per-step legal actions, chosen action, and
post-action state with `rng_counters`) once, then replays **Rust-only** against
it on every later sweep. Minutes → seconds.

Invalidation rides the cache **namespace** = `cs_version_namespace()`,
`<game version>-<hash>` (e.g. `v0.107.1-…`): a content hash of the selected
build's `lib/<ver>/*.dll` (game version + headless patch) + `src/**/*.cs` /
`*.csproj` (harness logic, excluding `bin/`/`obj/`). The build is
`STS2_GAME_VERSION` / `--game-version` (default v0.107.1, GH #789), so the two
oracle builds never share traces. Any change → new namespace → automatic full
re-record. The SOURCE is hashed, not the built dll, because a .NET build is not
byte-reproducible (fresh MVID per build) — hashing the binary would give every
worktree its own namespace and break the committed-trace bundle in
`data/parity-cache/` used by cloud sessions.

Controls: on by default; `--no-cache` / `STS2_PARITY_NOCACHE=1` to disable;
`--cache-dir` / `STS2_PARITY_CACHE_DIR` to relocate.

## 7. The complementary scripts

The combat sweep starts *inside* a synthetic combat, so it can't see the
run-setup RNG cascade or a masked spawn-roster bug. These scripts fill the gaps:

| Script | What it checks |
| --- | --- |
| `parity_fuzz.py` | single-trial driver — drives both engines from one seeded `Scenario`, reports the first divergence. The primitive `parity_audit` is built on. |
| `parity_fuzz_scenario.py` | seeded `Scenario` generator (`generate`, `generate_v2`, coverage scenarios). All ids canonical against the content crate. |
| `parity_fuzz_state.py` | the normalizers + structural `diff` (§2). |
| `parity_walk.py` | walks both **workbench** backends in lockstep (map → card reward → end_turn …) picking option 0 at every decision point; reports the first state-signature divergence. Run-level, not combat-level. |
| `r3_map_parity.py` | map-structure parity vs the **scraped `.run` corpus** — asserts each scraped visited-type path appears as a root→boss path in the Rust map. Corpus-bound. |
| `r3_neow_parity.py` | Neow `ancient_choice` parity vs the scraped corpus (relic-offer roll). |
| `run_setup_parity.py` | **(item#5)** map-gen parity over **arbitrary** seeds vs LIVE C# (`StartRun(seed)→get_map` vs `RunManager(string_seed).act_map()`). Compares root→boss path SETS (coordinate-independent — the two dumps use different origins). Fresh `Game` per seed (StartRun refuses a 2nd run). **Act 1 only** (reads the live `current_act_map`); for acts 2/3 see `act23_map_parity.py`. |
| `act23_map_parity.py` | **ALL-ACTS map-gen parity** — the acts-2/3 generalization of `run_setup_parity`. An act's map graph is a pure function of `(seed, act)` (both engines seed it `Rng(seed,"act_{N}_map")`, independent of run progress), so acts 2/3 are generated COMBAT-FREE: C# `dump_maps` reconstructs each act's `StandardActMap`; Rust `RunManager.act_map_for(act)` generates an arbitrary act's map without advancing. Compares the same root→boss path SETS per act. Act-1 verdict MUST match `run_setup_parity` (faithfulness check on the new oracles). `--all-characters -n N`. |
| `event_parity.py` | **EVENT parity across ALL acts**, combat-free. (a) POOL: each act's shuffled `RoomSet.events` (C# `dump_rooms`) vs Rust `generated_rooms()["events_shuffled"]`, order-exact — catches a wrong event set OR shuffle. (b) OPTION: forces every event on both engines (C# `enter_room {type:event}`; Rust `force_event(name)` + `event_option_infos()`) and diffs option COUNT + enabled/locked vector; option_id string drift is advisory (Rust short id vs C# loc-key tail). `CONTEXT_DEPENDENT_EVENTS` (potion/relic/gold/stock-gated by `IsAllowed`) are advisory under force-entry — their options legitimately differ in the impossible bare-run state. (c) **EFFECT (`--effect`)**: option count+enabled is necessary but NOT sufficient — a handler can present an identical menu while granting the wrong relic, no-opping a deck mutation, finishing a multi-page chain early, or skipping a nested combat. With `--effect` it additionally RESOLVES every option enabled on both engines (one fresh forced entry per option, via the new `RustGame.force_event` + `choose_option`/C# `Game.choose_option`) and diffs the resulting state through `run_parity`'s `Snap`/`diff_snaps` — relics, deck multiset, potions, gold, hp, the landed phase, and the multi-page follow-up option count. This is the check that catches the `sts2-rs-m3n0g` event-side stubs that (b) is blind to; the 57/83-Ironclad clean rate (reward-granting events included) confirms event-outcome RNG IS aligned under force-entry, so the flagged diffs are real (deterministic + identical across characters), modulo a few post-event decision-granularity phase splits (the #158 class) left for triage. `--all-characters --pool-seeds N [--effect]`. |
| `act23_state_parity.py` | **DEEP act-2/3 whole-run state sampling** (coverage gap (c)). The random-policy whole-run lockstep dies early in Act 1, so later-act carried state is never sampled. This pins both engines to `INVINCIBLE_HP` (mirror of `sts2_core::full_run::INVINCIBLE_HP`) **out of combat only** (via C# `set_player` / Rust new `RunManager.set_player` → `debug_set_player`), so the next combat *seeds* from full HP and the random walk survives deep — while in-combat HP decay stays an un-masked parity check (no mid-fight re-pin; sidesteps the Rust lazy mirror-creature). HP/gold writes consume NO RNG, so the stream stays faithful; benign both-engine act-rejects re-pick instead of truncating. Reuses `run_parity`'s `Snap`/`diff_*`/`enumerate_choices`/`run_signature` (run_parity.py untouched), plus three local additions from issue #158: (1) **per-step nav instrumentation** — `Nav` captures act/floor/current map-node `(col,row)` for BOTH engines every step in the verbose trace, with a BLOCKING `diff_nav` (act + current-node) gated to the shared `map` phase only (off the map C# legitimately leads by one act/room across the transition, so flagging there would false-diverge); `_act_of` now also reads C#'s nested `context.act` (was always `None` → `max_act` had been Rust-only); raw `floor` is trace-only (C# `ActFloor` per-act vs Rust cumulative `total_floor`); (2) **C# mid-combat-snapshot HP artifact classifier** — `_is_combat_hp_artifact` peels a small (`≤60`) player-HP-only delta in the combat / combat-exit window (`combat`/`card_reward`/`card_select`, NOT `event`/`rest`), since the out-of-combat pin wipes it at the next step so it can't perturb the trajectory; `--no-hp-peel` disables it for A/B; (3) phase-split decision detail in the trace. **Status (2026-06-27, fresh main): reach ~72% Act 2 all-characters / 92% Ironclad (was 42%), Act 3 0%.** The Act-1→Act-2 boundary nav cascade is RESOLVED (intervening engine work + the off-map nav gate); the instrumentation reclassified the residual `cs=map rs=event`/`card_select`/`card_reward` class as a real **event/room decision-granularity divergence** (Rust surfaces an `event_choice` C# auto-resolves), pinpointed at its node — the `sts2-rs-1hiht` thread, NOT an uncompared nav desync. HP-artifact peeling cut HP-only terminations 16→9 and un-masked deeper signal (real `enemies_hp` findings 26→31). Act 3 is now walled by **genuine Act-2 divergences** the probe newly reaches and surfaces cleanly: combat-damage `enemies_hp` (~31), `deck_size` (~12), map/event decision-granularity, the relic/gold combat-reward skews. `--all-characters -n N [--max-steps M] [--no-hp-peel]`; `--seed S --verbose` for single-run triage. **`--random-loadout`** fuzzes the run's *inputs* too: at the first map both engines take the same random deck, potions, gold and extra relics (`parity_fuzz_scenario.generate_loadout`, the `generate_v2` pools), through `set_player`, which fires no on-obtain hook and draws no RNG. So the walk reaches relic and card combinations that seeded rewards rarely assemble, on the rest-site, shop, reward and map surfaces the single-combat `fuzz` axis never touches. `--seed S --random-loadout` reproduces a run's loadout. The relics leave out the Event-rarity ones (`EVENT_RELICS`): an event grants those by name, so a grafted copy plus the event's would be a duplicate the real game never holds, and C# fires it per copy while Rust fires it per id. On-obtain effects are deliberately out of scope: they're bypassed on both sides. |
| `spawn_roster_parity.py` | **(item#6)** NATIVE Rust roster (`monsters=None`, no C# injection) vs C# per encounter. The combat sweep injects C#'s roster on mismatch to stop RNG-branched rosters cascading — which MASKS a genuine wrong native roster. This removes the mask: diffs per-slot kind, max_hp (HP roll), on-spawn powers, count, and order. |
| `mutation_test.py` | **(item#7)** sensitivity test. Injects compile-safe source mutations into the Rust engine one at a time, rebuilds, runs a fast slice, and checks the sweep CATCHES each (survivors = blind spots). A differential suite is only as good as what it can detect. |
| `run_parity.py` | **whole-RUN** lockstep (Neow → map → combats → events → shops → rewards → act transitions → boss), C# as the oracle. Component checks via flags: `--map-parity` / `--encounter-parity` / `--reward-parity` / `--relic-reward-parity` / `--shop-parity` / `--roster-parity` / `--rng-parity` / `--counters`. **`--sweep` is the productionized triage surface (§7a).** |
| `draw_sequence_trace.py` | **draw-order cascade tracer.** `run_trial` stops at the FIRST diverging snapshot; for a `PILE_COUNT\|end_turn` / `DRAW_ORDER` / `HAND_COST` cascade the real desync (a reshuffle/draw a few steps earlier) is long past. This re-runs the same lockstep driver but records the ORDERED piles of both engines at EVERY step, classifies each pile per step (`match` / `order` / `content`), reports whether the WHOLE-deck multiset is conserved (draw-order desync) or differs (real card loss), and flags which RNG stream diverges — separating benign run-setup offsets (`up_front`, potion/orb gen) from a real COMBAT `shuffle`/draw desync. `--seed N --character Silent [--max-turns T] [--full]`. |

### Running them

```bash
scripts/wheel-venv.sh run -- python python/run_setup_parity.py --character Ironclad -n 100
scripts/wheel-venv.sh run -- python python/act23_map_parity.py --all-characters -n 50
scripts/wheel-venv.sh run -- python python/event_parity.py --all-characters --pool-seeds 30
scripts/wheel-venv.sh run -- python python/event_parity.py --all-characters --effect  # resolve every option + diff the EFFECT
scripts/wheel-venv.sh run -- python python/spawn_roster_parity.py --character Ironclad
scripts/wheel-venv.sh run -- python python/mutation_test.py --tier quick --axes fuzz
```

All three exit non-zero on a real divergence / surviving mutation, so they slot
into CI.

## 7a. The whole-run sweep — `run_parity.py --sweep` (M6)

The productionized run-parity sweep, modelled on `parity_audit.py` so multiple
agents can divide the work (e.g. one agent per character) without stepping on
each other:

```bash
# Per-character worklist (the common case): N seeds, parallel, cached.
scripts/wheel-venv.sh run -- python python/run_parity.py \
    --sweep --character Silent --seed RP --seeds 50 -j 8

# Full-roster triage overview (buckets are character-tagged):
scripts/wheel-venv.sh run -- python python/run_parity.py \
    --sweep --all-characters --seeds 20
```

What it gives you:

- **Bucketed worklist.** Each run is reduced to a stable *divergence signature*
  — `phase/kind/<field>` with seed-specific numbers stripped (e.g.
  `combat/state/hp`, `card_select/state/phase:cs=card_select…`,
  `rest/act_reject/act:choose_option`) — so the SAME root cause across many
  seeds collapses into one ranked line with a representative example. Clean runs
  bucket as `clean:max_steps` / `clean:game_over_match`. Written to
  `--out/<ts>/{buckets,examples,rows}.json` (default `/tmp/run_parity`).
- **Freshness gates.** Reuses `parity_audit.assert_wheel_fresh()` +
  `assert_oracle_fresh()` (run once in the parent before forking), so a sweep
  never silently runs against a stale Rust wheel or a stale C# dll — the two
  ways a differential sweep lies (§5). Same bypass env vars apply.
- **Parallelism.** `-j N` worker processes (`ProcessPoolExecutor`); `0` =
  `os.cpu_count()`.
- **C# trajectory cache (ON by default).** The C# side of the lockstep is
  *Rust-independent* (every choice is enumerated off the C# oracle and picked by
  the seed-seeded `random.Random`), so the whole C# trajectory is recorded once
  and **replayed** against Rust on later sweeps — the .NET subprocess is never
  spawned again. Namespaced by `cs_version_namespace()` **plus a hash of
  `run_parity.py`** (the recorded trajectories bake in the choice picker /
  enumeration logic, so any harness edit auto-invalidates). Crucially the
  namespace does **not** include the Rust engine — so a per-character agent
  *fixing Rust* replays the same cached C# truth and just watches its bucket go
  green; the bundle stays valid across Rust work. `--no-cache` /
  `STS2_RUNPARITY_NOCACHE=1` forces a fully-live sweep; `--cache-dir` /
  `STS2_RUNPARITY_CACHE_DIR` relocates it (default `~/.cache/sts2_run_parity`).

### Replay-only — for 4-core / no-game agents

A low-core agent that must **never spawn the C# game** runs `--replay-only`
(env `STS2_RUNPARITY_REPLAY_ONLY=1`): every run is served from a pre-baked
bundle and a cache **miss fails closed** (reported as the `MISS:not_in_cache`
bucket) instead of falling through to `.NET`. The C#-oracle freshness gate is
skipped (no built dll to check); only the Rust wheel gate runs.

```bash
# Warm the committed bundle once (git-lfs + zstd), then replay-only:
git lfs pull --include 'sts2-pycli/data/run-parity-cache/*'
scripts/wheel-venv.sh run -- data/run-parity-cache/unpack.sh
scripts/wheel-venv.sh run -- python python/run_parity.py \
    --sweep --character Silent --seed RP --seeds 50 --steps 400 --replay-only -j 4
```

The `lib/<ver>/*.dll` are committed (git-lfs), so `cs_version_namespace()` reproduces
the same namespace on a no-game machine — the bundle a clone unpacks matches the
namespace it recomputes at the same commit. The replay params **must match the
baked grid** (same `--seed`/`--steps`, `--seeds ≤` baked); the bundle, the
canonical grid, the coordinator re-bake recipe, and the (currently manual)
`unpack.sh` warming all live in **`data/run-parity-cache/README.md`**. Baking is
hardened against the known C# headless stuck-turn (`STS2_RUNPARITY_RECORD_TIMEOUT`
abandons a wedged record as `incomplete:record_timeout` rather than hanging the
bake).

Engine-free unit coverage in `tests/test_run_parity_sweep.py` (signature
bucketing, cache key/namespace/hit-miss/format-invalidation, replay-only
fail-closed, record→replay equivalence with a fake Rust backend).

## 8. Wheel/venv discipline (read before running anything)

The harness imports `sts2_rs` from a venv whose `.so` must match the engine you
intend to test. **Always build and run through `scripts/wheel-venv.sh` with NO
src arg** so it auto-detects your checkout (main → shared `.venv`; each worktree
→ its own content-addressed `.venv-h/<engine-hash>`). Never build another
agent's venv. Full rules and the staleness history are in `CLAUDE.md` §"Building
the `sts2_rs` wheel".

## 9. Tests

The suite has its own pytest coverage (run via the wheel venv):

- `tests/test_parity_diff_surface.py` — the diff compares exhaust fate + hand
  cost.
- `tests/test_parity_gates.py` — oracle-freshness gate, invariant flag, coverage
  helpers, `assemble_v2_trials` paths/max_turns behavior.
- `tests/test_run_setup_parity.py` — map path-set comparison.

```bash
scripts/wheel-venv.sh run -- python -m pytest tests/test_parity_*.py
```

## 10. Current known divergences (triage candidates)

As of 2026-06-20, on the rebuilt C# oracle (namespace `ac231b05`), the suite is
**clean across every surface** — combat full tier 6103/6104 and whole-run
250/250 (all 5 characters). The trace bundles in `data/parity-cache/` and
`data/run-parity-cache/` were rebaked against this oracle.

The four `2026-06-10` regressions below are **all resolved** (fixed in the work
that landed after that date — the act-boss relic decision, event/all-acts
map-gen checks, PARITY132 non-combat reward odds, the .NET 9 pin, etc.):

- ~~`HAND_COST` seed 326 — a hand card costs 2 in C#, 1 in Rust~~
- ~~`MONSTER_STATE` CONFLAGRATION — `monsters[0].hp` 49 vs 53~~
- ~~`PLAYER_STATE` FIGHT_ME — `player.powers` diverge~~
- ~~PILLAGE — `combat_terminal` cs=True rs=False~~

The lone full-tier outlier is a single `PLAYER_STATE|CASCADE` on
`player.powers.len` (trial idx 3057) — downstream of an earlier RNG-branch
difference (the `CASCADE` taxonomy bucket), absent from the quick/mid tiers; an
artifact, not a root bug.

### Deep Act-2 carried-state divergences (issue #158, `act23_state_parity.py`)

The clean 250/250 whole-run sweep above runs a random policy that dies in Act 1,
so it only diffs Act-1 content. `act23_state_parity.py` (invincible lockstep)
now penetrates **~72% into Act 2 all-characters (92% Ironclad)** with per-step
nav instrumentation and the C# mid-combat-snapshot HP artifact classified out —
exposing a real Act-2 carried-state divergence list the shallow sweep never
reached. As of 2026-06-27 (fresh main, `--all-characters -n 20 --max-steps
4000`) the surviving buckets, in rough order, are triage candidates **for
follow-up beads** (out of scope for #158, which delivered the coverage/instrument
work that surfaces them):

- `combat/state/enemies_hp` (~31 runs) — Act-2 combat-damage skew (one card/power
  deals different damage C#↔Rust); the dominant wall to Act 3.
- `map/state/deck_size` + `card_reward/state/deck` (~14) — an Act-2 card
  acquisition/removal lands on one engine but not the other.
- `map/state/phase: cs=map rs=event|card_select|card_reward` (~12) — **event/room
  decision-granularity** (Rust surfaces an `event_choice` C# auto-resolves and
  steps past). The `sts2-rs-1hiht` class; the nav instrumentation pins it to the
  exact node. Companion #157 (event-effect parity) covers the effect side.
- `combat/state/phase: cs=combat rs=card_reward` (~4) + relic/gold reward skews —
  combat-length / reward-stream granularity.
- residual `combat|map/state/hp` (~9) — HP deltas **above** the 60 artifact
  tolerance (real or large-artifact; deliberately NOT peeled).

Act-3 carried-state parity stays **0%-reached**: every run hits one of the above
Act-2 divergences first. Reproduce a single class with
`python python/act23_state_parity.py --character <C> --seed <S> --verbose`.

A divergence a sweep shows but the same seed alone does not is a harness
race until proven otherwise (GH #1147). A sweep prints the C# stack of any
one-sided C# reject (`cs_stack|` lines), and `STS2_THREAD_PROBE=<dir>` makes
every oracle log game-state writes made off its game thread to
`<dir>/probe-<pid>.log` (`ThreadProbe` in `src/Sts2Headless/RunSimulator.cs`). Writes during
`end_turn`, `play_card`, `use_potion` or `cmd=-(idle)` are a race; writes
during the handler that started a pool task (`choose_option`,
`select_map_node`, shop buys) are expected. Harness code that starts game
code on the pool goes through `RunOnPool`, which installs the game's sync
context there.

See agent memory `project_parity_gap_closure_8_tools` and
`project_ic_random_seed_parity_complete` for the full history.

#### Checkpoints and the survival walk (2026-11)

Two changes make act 2 and act 3 cheap to sample.

- **Checkpoints.** `--checkpoint-dir D --checkpoint-acts 2,3` saves a checkpoint
  at the first map of each act where both engines still agree: the C# oracle's
  continue save plus the logged Rust calls. `--from-checkpoints D --branches N`
  starts N differently seeded walks from each one: every checkpoint in `D`,
  unless `--character`, `--ascension` or `--checkpoint-act` narrows it. Rust replays its calls (well
  under a second); C# loads the save (about a second). On v0.111.0 that gave 40
  act-2 trials in 123 s, against 376 s for 10 runs from Neow.
  `save_load_identity.py` is the gate that a C# reload is lossless (25/25 act-2
  and 8/8 act-3 runs identical for 400 steps). A checkpoint goes stale when the
  Rust engine changes; the run then ends `checkpoint_stale`.
- **The survival walk.** The Insatiable (act-2 boss) applies Sandpit, which
  force-kills its target at 0 whatever the HP, so the invincible walk used to
  die there almost every time and act 3 was never reached. The walk now always
  plays Frantic Escape and doesn't end a turn early while Sandpit is up
  (`--uniform-policy` restores the old walk).

One oracle limit applies to checkpoint runs only: a reloaded run can't finish
a win. `load_save` sets `ShouldSave`, so `WinRun` runs the profile progress
update, which throws under the oracle's NGame stub. The probe counts that
case as a win; the oracle fix (clear `ShouldSave` after a load) is a `.cs`
change to batch with GH #1054.
