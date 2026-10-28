"""`python -m ppo.combat_bench report`: the pilot-vs-human gap and paired arm deltas.

Reads one fight set and 1..N played files (record shapes in `SCHEMA.md`) and
prints text tables; `--json OUT` also writes a machine-readable summary.

    python -m ppo.combat_bench report --set data/combat_bench/ironclad_pro_v1.jsonl \\
        played.deployed.jsonl played.rl3.jsonl --baseline deployed --json report.json

An arm's name is the played file's stem unless given as `NAME=PATH`.

What it measures, and why each choice:

- **Loss.** Per rep `L = min(dmg_c + dmg_unlogged, hp_before)` for a win and
  `L = hp_before + DEATH_HP` for a death (HP equivalents; `--death-hp`,
  default 65). `dmg_c + dmg_unlogged` is the C# `DamageTaken` definition (see
  `rep_damage`). Clamping at `hp_before` stops overkill on a death from
  counting twice. The human's `L_h` is the same with `damage_taken`. A death
  always counts all the HP the fighter had: some deaths lose HP through a
  path `DamageTaken` does not count (The Insatiable's Sandpit calls
  `CreatureCmd.Kill`), and `min(damage_taken, hp_before)` would understate
  those.
- **Quits.** A human who abandoned the run mid-fight neither won nor died.
  Their counterfactual `L_h` lies in `[min(damage_taken, hp_before),
  hp_before + DEATH_HP]`: survived with the damage taken so far, or died.
  Every pilot-vs-human number is reported at both ends of that bracket
  (`quit_as_survived`, `quit_as_death`) and with the quit fights dropped
  (`quit_excluded`, a missing-at-random point estimate that can sit inside
  or outside the bracket; the per-stratum, per-player and per-encounter
  tables use it).
- **Scoring.** Only `end in {won, died}` counts. A capped, crashed or
  watchdog-killed fight is not a survival, so it is dropped and the report
  shows a worst/best-case bound with every dropped cell scored as the
  worst/best possible `L`. A `watchdog` end depends on the machine, and a
  `sims_cap` end truncates the arm's longest fights, so both get a loud
  warning. A cell retried with `run --retry-ends` replaces its earlier
  `watchdog` / `error` record.
- **Interim reads.** Cells are written in completion order inside a block, so
  the first-finished cells of a block favour short fights. A block with
  fewer cells than planned (in progress or aborted) gets a loud warning;
  `--complete-blocks-only` drops such blocks for an unbiased interim read.
- **Cost.** `sims_spent` is the primary cost metric. CPU ratios between arms
  are printed only when the arms ran on the same host with the same wheel
  kind (PGO or plain); load differences get a warning (`cpu_s` is
  `process_time`, which contention inflates by +50% on a loaded box).
- **Pilot vs human.** Fight-level `D_f = mean(L over scored reps) - L_h`.
  Fights in one run share a deck lineage, so the SE clusters by run (CR1); the
  pro set has only ~12 players, so the CI is a wild cluster bootstrap by
  PLAYER (Webb weights). CR1 by run ignores the player effect (on the pro
  data a player SD of ~2.5 HP): a +-1.96 CR1 interval covers only ~80-89% on
  the real split layouts, so every table, the splits included, shows the
  wild-by-player CI. On the death co-primaries (rare binary outcomes over ~12
  unbalanced players) the wild CI covers ~93% rather than 95% (calibrated on
  the pro boss layout); a death CI that only just excludes 0 is not decisive.
  An equal-player-weight variant (mean of per-player means) shows whether one
  prolific player drives the mean.
- **Arm vs arm.** Paired on fights AND seeds: each fight keeps the
  intersection of the two arms' scored rep ids, so both arms saw the same
  draws (common random numbers). Fight-level sign-flip permutation p, CR1 by
  run as robustness, Holm across the non-baseline arms.
- **Pro gap of arm A** = gap(B0) + delta(A - B0), computed per fight and
  bootstrapped as one quantity, never as two CIs added: B0 can carry many
  more reps than a screened arm, and the paired delta is far less noisy than
  a second human comparison.
- **Self-null** (`--self-null FILE`). One rep of each fight plays the "human"
  and the other reps the pilot. The true gap is exactly 0, so this checks
  the estimators end to end; `--self-null-coverage` rotates the "human" rep
  and reports how often each 95% CI covers 0.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from scipy import stats as sps

from ppo.combat_bench import stats as S

DEATH_HP = 65
PLAYED_SCHEMA = "combat_bench.played.v1"
REPORT_SCHEMA = "combat_bench.report.v1"
SCORED_ENDS = frozenset({"won", "died"})
ENDS = ("won", "died", "round_cap", "step_cap", "sims_cap", "no_action", "watchdog", "error")
ROOMS = ("monster", "elite", "boss")
VARIANTS = ("quit_excluded", "quit_as_survived", "quit_as_death")
QLABEL = {"quit_excluded": "excl", "quit_as_survived": "surv", "quit_as_death": "death"}
# Ends a `run --retry-ends` may replay; a later record replaces the earlier one.
RETRYABLE_ENDS = frozenset({"watchdog", "error"})
METRICS = ("L", "died", "dmg", "turns")


class BenchError(Exception):
    """Inputs that must not be compared (header mismatch, unknown schema, ...)."""


# ── loading and validation ──────────────────────────────────────────────


def file_sha256(path: str) -> str:
    """sha256 of the raw file bytes (what `run` records as `set_sha256`)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_set(path: str) -> dict[str, dict[str, Any]]:
    """The fight set, keyed by `key`.

    Raises:
        BenchError: on a duplicate key.
    """
    fights: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            f = json.loads(line)
            if f["key"] in fights:
                raise BenchError(f"{path}: duplicate key {f['key']}")
            fights[f["key"]] = f
    return fights


@dataclass
class Played:
    """One played file: its header and its cells.

    Attributes:
        name: Arm name.
        path: File path.
        header: The header object.
        cells: `(key, rep id) -> rep record`, every end status, with the cell's
            `block` copied into the record.
        duplicates: Repeated `(key, rep)` lines (the first one is kept),
            not counting retries.
        retried: `watchdog` / `error` records replaced by a later record of
            the same cell (`run --retry-ends`).
        unknown_keys: Cells whose key is not in the set (dropped).
        blocks_dropped: Cells dropped by `--complete-blocks-only`.
        fight_keys: The set keys the run planned to play (`selected_keys`), or
            None for the whole set.
    """

    name: str
    path: str
    header: dict[str, Any]
    cells: dict[tuple[str, int], dict[str, Any]]
    duplicates: int = 0
    retried: int = 0
    unknown_keys: int = 0
    blocks_dropped: int = 0
    notes: list[str] = field(default_factory=list)
    fight_keys: list[str] | None = None

    def planned(self, fights: dict[str, dict[str, Any]]) -> list[str]:
        """The fights this file promises cells for, in set order."""
        if self.fight_keys is None:
            return list(fights)
        return [k for k in self.fight_keys if k in fights]

    def expected_reps(self) -> list[int]:
        """Rep ids the header promises for every fight."""
        reps, off = self.header.get("reps"), self.header.get("rep_offset", 0)
        if isinstance(reps, int):
            return list(range(off, off + reps))
        return sorted({r for _, r in self.cells})

    def scored(self) -> dict[str, dict[int, dict[str, Any]]]:
        """`key -> rep id -> record` for the scored (`won` / `died`) cells only."""
        out: dict[str, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
        for (k, r), rec in self.cells.items():
            if rec.get("end") in SCORED_ENDS:
                out[k][r] = rec
        return dict(out)


def _arm_spec(arg: str) -> tuple[str, str]:
    """`NAME=PATH` or `PATH` (name = the file stem)."""
    if "=" in arg and not os.path.exists(arg):
        name, path = arg.split("=", 1)
        return name, path
    base = os.path.basename(arg)
    return (base[: -len(".jsonl")] if base.endswith(".jsonl") else base), arg


def load_played(path: str, name: str, fights: dict[str, dict[str, Any]]) -> Played:
    """Read a played file.

    Raises:
        BenchError: when line 1 is not a `combat_bench.played.v1` header.
    """
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    if not lines:
        raise BenchError(f"{path}: empty played file")
    head = json.loads(lines[0])
    if "header" not in head:
        raise BenchError(f"{path}: line 1 is not a header")
    header = head["header"]
    if header.get("schema") != PLAYED_SCHEMA:
        raise BenchError(f"{path}: schema {header.get('schema')!r}, expected {PLAYED_SCHEMA!r}")
    p = Played(name, path, header, {})
    for ln in lines[1:]:
        c = json.loads(ln)
        key, rep = c["key"], c["rep"]
        if key not in fights:
            p.unknown_keys += 1
            continue
        rid = int(rep["rep"])
        if rep.get("end") in SCORED_ENDS and not isinstance(rep.get("dmg_c"), (int, float)):
            raise BenchError(f"{path}: scored cell {key} rep {rid} has no dmg_c")
        old = p.cells.get((key, rid))
        if old is not None:
            # `run --retry-ends` appends a fresh record for a machine-dependent
            # end; the last one stands. Any other repeat keeps the first.
            if old.get("end") not in RETRYABLE_ENDS:
                p.duplicates += 1
                continue
            p.retried += 1
        p.cells[key, rid] = {**rep, "block": c.get("block")}
    return p


def selected_keys(header: dict[str, Any], fights: dict[str, dict[str, Any]]) -> list[str] | None:
    """The set keys a played file planned to play, from its header's `select`.

    `run` can play part of a set (`--keys-file`, `--room`, `--limit`). The
    expected cells are the PLANNED fights times the rep ids; counting the
    whole set instead would score a 24-fight run on the 1,547-fight pro set
    as 98% dropped. A keys file travels in `select.keys`; room and limit are
    re-applied to the set in its file order, exactly as `run.select_fights`
    applies them, and the result must hash to `select.fights_sha256`.

    Args:
        header: The played file's header.
        fights: The set, in file order.

    Returns:
        The planned keys in set order, or None when the header selects the
        whole set (no `select`, or one without keys, room or limit).

    Raises:
        BenchError: When the selection cannot be rebuilt from the header or
            does not hash to `select.fights_sha256`.
    """
    sel = header.get("select")
    if not sel:
        return None
    keys = list(fights)
    if sel.get("keys") is not None:
        want = set(sel["keys"])
        missing = want - set(fights)
        if missing:
            raise BenchError(f"select.keys names {len(missing)} keys the set does not have")
        keys = [k for k in keys if k in want]
    elif sel.get("keys_file_sha256"):
        raise BenchError("the header selects by a keys file but does not list its keys")
    if sel.get("room"):
        rooms = set(sel["room"])
        keys = [k for k in keys if fights[k]["room_type"] in rooms]
    if sel.get("limit") is not None:
        keys = keys[: int(sel["limit"])]
    want_sha = sel.get("fights_sha256")
    got_sha = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    if want_sha and got_sha != want_sha:
        raise BenchError(
            f"the rebuilt selection ({len(keys)} fights) does not hash to select.fights_sha256 "
            f"({sel.get('n_fights')} fights)"
        )
    if len(keys) == len(fights):
        return None
    return keys


def resolve_selection(
    p: Played, fights: dict[str, dict[str, Any]], allow_set_mismatch: bool = False
) -> None:
    """Set `p.fight_keys` from its header.

    With `allow_set_mismatch` an unrebuildable selection (usually: the set
    file changed) falls back to the keys the file has cells for, with a note.

    Raises:
        BenchError: When the selection cannot be rebuilt and no fallback is allowed.
    """
    try:
        p.fight_keys = selected_keys(p.header, fights)
    except BenchError as e:
        if not allow_set_mismatch:
            raise BenchError(
                f"{p.path}: {e} (--allow-set-mismatch expects the fights it has cells for)"
            ) from None
        played = {k for k, _ in p.cells}
        p.fight_keys = [k for k in fights if k in played]
        p.notes.append(f"!!! {e}; expected fights = the {len(p.fight_keys)} with cells")


def block_counts(p: Played, n_fights: int) -> tuple[dict[Any, tuple[int, int]], int | None]:
    """Per started block: `(cells present, cells planned)`, and the block size used.

    The block size is the header's `block_size` when present, otherwise the
    largest per-block count (correct once any block has completed); the
    block size is None when it had to be inferred.
    """
    total = n_fights * len(p.expected_reps())
    counts = collections.Counter(rec.get("block") for rec in p.cells.values())
    if not counts:
        return {}, None
    bs = p.header.get("block_size")
    header_bs = isinstance(bs, int) and bs > 0
    if not header_bs:
        bs = max(counts.values())
    n_blocks = max(1, math.ceil(total / bs))

    def expected(b: Any) -> int:
        if not isinstance(b, int):
            return -1
        return bs if b < n_blocks - 1 else total - bs * (n_blocks - 1)

    return {b: (n, expected(b)) for b, n in counts.items()}, bs if header_bs else None


def partial_blocks(p: Played, n_fights: int) -> list[tuple[Any, int, int]]:
    """Started blocks with fewer cells than planned: `(block, present, planned)`.

    Such a block is in progress or was aborted, and its cells are the ones
    that finished first, i.e. the short fights.
    """
    counts, _ = block_counts(p, n_fights)
    return sorted(
        ((b, n, want) for b, (n, want) in counts.items() if n != want),
        key=lambda x: (not isinstance(x[0], int), x[0] if isinstance(x[0], int) else 0),
    )


def complete_blocks_only(p: Played, n_fights: int) -> Played:
    """Keep only the blocks whose cells are all present.

    Cells are submitted in seeded-random order in fixed-size blocks, so a
    complete block is a random sample of the planned cells. An interim read
    on whatever finished first is biased whenever a fight's cost depends on
    its outcome (long fights finish last); one on complete blocks is not.
    """
    counts, bs = block_counts(p, n_fights)
    if not counts:
        return p
    notes = list(p.notes)
    if bs is None:
        notes.append(
            f"block size inferred as {max(n for n, _ in counts.values())} "
            "(header has no block_size)"
        )
    keep = {b for b, (n, want) in counts.items() if n == want}
    cells = {k: v for k, v in p.cells.items() if v.get("block") in keep}
    out = replace(p, cells=cells, blocks_dropped=len(p.cells) - len(cells))
    out.notes = [*notes, f"complete blocks kept: {len(keep)} of {len(counts)} started"]
    return out


def cell_hosts(p: Played) -> set[str]:
    """Hosts the arm's cells ran on (per-cell `host`, else the header's)."""
    hosts = {c["host"] for c in p.cells.values() if c.get("host")}
    return hosts or ({p.header["host"]} if p.header.get("host") else set())


def load_per_cpu(p: Played) -> float | None:
    """Median 1-minute load average per CPU over the arm's cells, if recorded."""
    loads = sorted(c["load1"] for c in p.cells.values() if c.get("load1") is not None)
    cpus = p.header.get("cpus")
    if not loads or not isinstance(cpus, int) or cpus <= 0:
        return None
    return loads[len(loads) // 2] / cpus


def cpu_comparable(played: list[Played]) -> str | None:
    """Why `cpu_s` ratios between these arms mean nothing, or None if they compare.

    `cpu_s` is `process_time`. A PGO wheel and a plain one share a
    `src_hash`, and two hosts run at different speeds, so a CPU ratio across
    either is not the agents' cost.
    """
    pgo = {p.name: p.header.get("pgo") for p in played}
    if len(set(pgo.values())) > 1:
        return "wheel kinds differ (pgo " + ", ".join(f"{n}={v}" for n, v in pgo.items()) + ")"
    hosts = {p.name: sorted(cell_hosts(p)) for p in played}
    if len({tuple(h) for h in hosts.values()}) > 1 or any(len(h) > 1 for h in hosts.values()):
        return "hosts differ (" + ", ".join(f"{n}={','.join(h)}" for n, h in hosts.items()) + ")"
    return None


def validate(
    set_sha: str,
    played: list[Played],
    *,
    allow_engine_mismatch: bool = False,
    allow_set_mismatch: bool = False,
) -> list[str]:
    """Refuse to compare files that do not describe the same experiment.

    Args:
        set_sha: sha256 of the fight-set file given to `report`.
        played: The played files.
        allow_engine_mismatch: Turn a `src_hash` mismatch into a warning.
        allow_set_mismatch: Turn a `set_sha256` mismatch into a warning.

    Returns:
        Warnings to print.

    Raises:
        BenchError: on a mismatch that was not allowed.
    """
    warns: list[str] = []
    shas = {p.name: p.header.get("set_sha256") for p in played}
    bad = {n: s for n, s in shas.items() if s != set_sha}
    if bad:
        msg = f"set_sha256 mismatch: the set file hashes to {set_sha[:12]}, but " + ", ".join(
            f"{n}={(s or 'None')[:12]}" for n, s in bad.items()
        )
        if not allow_set_mismatch:
            raise BenchError(msg + " (--allow-set-mismatch to override)")
        warns.append("!!! " + msg)
    srcs = {p.name: p.header.get("src_hash") for p in played}
    if len(set(srcs.values())) > 1:
        msg = "src_hash differs between arms (different engines): " + ", ".join(
            f"{n}={s}" for n, s in srcs.items()
        )
        if not allow_engine_mismatch:
            raise BenchError(msg + " (--allow-engine-mismatch to override)")
        warns.append(
            "!!! ENGINE MISMATCH — " + msg + ". Arm deltas mix agent and engine differences."
        )
    caps = {p.name: json.dumps(p.header.get("caps"), sort_keys=True) for p in played}
    if len(set(caps.values())) > 1:
        # A tighter cap drops more long fights unscored, and long fights are
        # the ones that go badly, so deltas would partly measure the caps.
        warns.append(
            "!!! caps differ between arms (dropped cells are not comparable): "
            + ", ".join(f"{n}={c}" for n, c in caps.items())
        )
    for p in played:
        if p.duplicates:
            warns.append(f"{p.name}: {p.duplicates} duplicate (key, rep) lines (first kept)")
        if p.retried:
            warns.append(f"{p.name}: {p.retried} watchdog/error cells retried (last record kept)")
        if p.unknown_keys:
            warns.append(f"{p.name}: {p.unknown_keys} cells whose key is not in the set (dropped)")
        n_wd = sum(1 for c in p.cells.values() if c.get("end") == "watchdog")
        if n_wd:
            warns.append(
                f"!!! WATCHDOG: {p.name} has {n_wd} watchdog ends. They depend on machine load, "
                "not on the agent; rerun them on an idle host (`run --retry-ends watchdog`) "
                "before trusting the arm."
            )
        n_cap = sum(1 for c in p.cells.values() if c.get("end") == "sims_cap")
        if n_cap:
            warns.append(
                f"!!! SIMS CAP: {p.name} has {n_cap} sims_cap ends, budget-truncated: "
                "--max-sims-per-fight cut its longest fights, which are the ones that go "
                "badly; raise the cap and rerun those cells before trusting the arm."
            )
        warns.extend(f"{p.name}: {n}" for n in p.notes)
    if len(played) > 1:
        why = cpu_comparable(played)
        if why:
            warns.append(f"!!! cpu x base is n/a: {why}. sims and decisions ratios still hold.")
        loads = {p.name: load_per_cpu(p) for p in played}
        known = {n: v for n, v in loads.items() if v is not None}
        if len(known) > 1 and max(known.values()) - min(known.values()) > 0.5:
            warns.append(
                "!!! LOAD differs between arms (median 1-min load per CPU: "
                + ", ".join(f"{n}={v:.2f}" for n, v in known.items())
                + "); cpu_s is process_time, which contention inflates, so cpu x base is "
                "confounded."
            )
    return warns


def interim_warnings(played: list[Played], fights: dict[str, dict[str, Any]]) -> list[str]:
    """A loud warning for every arm with a partial block (an interim read)."""
    out = []
    for p in played:
        pb = partial_blocks(p, len(p.planned(fights)))
        if pb:
            shown = ", ".join(f"{b} ({n}/{want})" for b, n, want in pb[:4])
            out.append(
                f"!!! PARTIAL BLOCK: {p.name} has {len(pb)} block(s) with fewer cells than "
                f"planned: {shown}{' ...' if len(pb) > 4 else ''}. This interim read is biased "
                "toward short fights; pass --complete-blocks-only."
            )
    return out


# ── per-fight quantities ────────────────────────────────────────────────


def rep_damage(rep: dict[str, Any]) -> float:
    """The rep's C# `DamageTaken` equivalent: `dmg_c + dmg_unlogged`.

    `dmg_c` sums the player damage rows of the engine's combat log, but the
    engine writes no row for HP loss through `Effect::DealHpLoss` (Thorns,
    Inferno, Combust, Crimson Mantle). The game routes all of those through
    `CreatureCmd.Damage`, so the human's `damage_taken` counts them
    (`CreatureCmd.cs:334`). `run` books that loss as `dmg_unlogged`; leaving
    it out would credit the pilot ~4% of its hallway damage the human pays.
    A file without the field (an older run) counts it as 0.
    """
    return rep["dmg_c"] + (rep.get("dmg_unlogged") or 0)


def loss(damage: float, died: bool, hp_before: float, death_hp: float) -> float:
    """HP-equivalent loss: clamped damage for a win, all HP plus `death_hp` for a death.

    A death counts `hp_before` whatever the damage rows say, for both sides:
    the C# Kill path (The Insatiable's Sandpit) takes HP that `DamageTaken`
    never counts, and the engine's HP ledger would book it as unlogged.
    """
    return hp_before + death_hp if died else min(damage, hp_before)


def rep_loss(rep: dict[str, Any], fight: dict[str, Any], death_hp: float) -> float:
    """HP-equivalent loss of one scored rep."""
    return loss(rep_damage(rep), bool(rep["died"]), fight["hp_before"], death_hp)


def rep_metrics(rep: dict[str, Any], fight: dict[str, Any], death_hp: float) -> dict[str, float]:
    """`L`, `died`, `dmg` (`rep_damage`) and `turns` of one scored rep."""
    return {
        "L": rep_loss(rep, fight, death_hp),
        "died": float(bool(rep["died"])),
        "dmg": float(rep_damage(rep)),
        "turns": float(rep.get("turns") or 0),
    }


def mean_metrics(
    reps: Iterable[dict[str, Any]], fight: dict[str, Any], death_hp: float
) -> dict[str, float]:
    """Per-metric mean over reps."""
    ms = [rep_metrics(r, fight, death_hp) for r in reps]
    return {k: sum(m[k] for m in ms) / len(ms) for k in METRICS}


def human_metrics(fight: dict[str, Any], death_hp: float, variant: str) -> dict[str, float] | None:
    """The human's outcome, or None when this variant excludes the fight.

    A mid-fight quit is dropped by `quit_excluded`, scored as a survival with
    the damage taken so far by `quit_as_survived` (the bracket's low end) and
    as a death that lost all the HP the human had by `quit_as_death` (its
    high end).
    """
    h = fight["human"]
    if h.get("damage_taken") is None:
        return None
    died = bool(h.get("died"))
    if h.get("quit"):
        if variant == "quit_excluded":
            return None
        died = variant == "quit_as_death"
    return {
        "L": loss(h["damage_taken"], died, fight["hp_before"], death_hp),
        "died": float(died),
        "dmg": float(h["damage_taken"]),
        "turns": float(h.get("turns") or 0),
    }


@dataclass(frozen=True)
class GapRow:
    """One fight in a pilot-vs-human comparison."""

    key: str
    run: str
    player: str
    room: str
    act: int
    encounter: str
    reps: int
    p: dict[str, float]
    h: dict[str, float]

    def d(self, metric: str = "L") -> float:
        """Pilot minus human on `metric` (deaths in percentage points)."""
        scale = 100.0 if metric == "died" else 1.0
        return scale * (self.p[metric] - self.h[metric])


def gap_rows(
    fights: dict[str, dict[str, Any]],
    pilot: dict[str, dict[int, dict[str, Any]]],
    human_of: Callable[[str], dict[str, float] | None],
    death_hp: float,
) -> list[GapRow]:
    """Fight rows for every fight with a scored pilot rep and a human outcome."""
    rows = []
    for key in sorted(pilot):
        reps = pilot[key]
        h = human_of(key)
        if not reps or h is None:
            continue
        f = fights[key]
        rows.append(
            GapRow(
                key,
                f["file"],
                f["player"],
                f["room_type"],
                int(f["act"]),
                f["encounter"],
                len(reps),
                mean_metrics(reps.values(), f, death_hp),
                h,
            )
        )
    return rows


@dataclass(frozen=True)
class PairRow:
    """One fight in an arm-vs-arm comparison, on the common rep ids."""

    key: str
    run: str
    player: str
    room: str
    act: int
    encounter: str
    reps: tuple[int, ...]
    a: dict[str, float]
    b: dict[str, float]

    def d(self, metric: str) -> float:
        """Arm minus baseline on `metric` (deaths in percentage points)."""
        scale = 100.0 if metric == "died" else 1.0
        return scale * (self.a[metric] - self.b[metric])


def pair_rows(
    fights: dict[str, dict[str, Any]],
    a: dict[str, dict[int, dict[str, Any]]],
    b: dict[str, dict[int, dict[str, Any]]],
    death_hp: float,
) -> list[PairRow]:
    """Paired fights: each keeps the intersection of the two arms' scored rep ids."""
    rows = []
    for key in sorted(set(a) & set(b)):
        common = tuple(sorted(set(a[key]) & set(b[key])))
        if not common:
            continue
        f = fights[key]
        rows.append(
            PairRow(
                key,
                f["file"],
                f["player"],
                f["room_type"],
                int(f["act"]),
                f["encounter"],
                common,
                mean_metrics((a[key][r] for r in common), f, death_hp),
                mean_metrics((b[key][r] for r in common), f, death_hp),
            )
        )
    return rows


# ── estimators over rows ────────────────────────────────────────────────


@dataclass(frozen=True)
class Opts:
    """Knobs shared by every estimator call."""

    death_hp: float = DEATH_HP
    boot: int = 9999
    perm: int = 20000
    seed: int = 0


def _est(vals: list[float], runs: list[str], players: list[str], o: Opts) -> dict[str, Any]:
    """Mean with plain SE, CR1-by-run SE and the wild-by-player CI."""
    plain = S.mean_se(vals)
    c = S.cr1(vals, runs)
    w = S.wild_cluster_ci(vals, players, draws=o.boot, seed=o.seed)
    return {
        "mean": plain.mean,
        "se": plain.se,
        "se_cr1": c.se,
        "ci_lo": w.lo,
        "ci_hi": w.hi,
        "p0": w.p0,
        "n": plain.n,
        "runs": c.g,
        "players": w.g,
        "boot_exact": w.exact,
    }


def _equal_player(vals: list[float], players: list[str], o: Opts) -> dict[str, Any]:
    """Mean of per-player means; the CI treats each player as one observation."""
    by: dict[str, list[float]] = collections.defaultdict(list)
    for v, p in zip(vals, players):
        by[p].append(v)
    means = [sum(v) / len(v) for v in by.values()]
    w = S.wild_cluster_ci(means, list(by), draws=o.boot, seed=o.seed)
    return {"mean": w.mean, "ci_lo": w.lo, "ci_hi": w.hi, "players": len(means)}


def summarize_gap(rows: list[GapRow], o: Opts) -> dict[str, Any]:
    """Overall pilot-vs-human numbers for one arm and one quit variant."""
    if not rows:
        return {"fights": 0}
    runs = [r.run for r in rows]
    players = [r.player for r in rows]
    gap = [r.d("L") for r in rows]
    boss = [r for r in rows if r.room == "boss"]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else math.nan

    out: dict[str, Any] = {
        "fights": len(rows),
        "runs": len(set(runs)),
        "players": len(set(players)),
        "cells": sum(r.reps for r in rows),
        "pilot": {m: mean([r.p[m] for r in rows]) for m in METRICS},
        "human": {m: mean([r.h[m] for r in rows]) for m in METRICS},
        "gap": _est(gap, runs, players, o),
        "gap_equal_player": _equal_player(gap, players, o),
        "deaths_pp": _est([r.d("died") for r in rows], runs, players, o),
        "dmg": _est([r.d("dmg") for r in rows], runs, players, o),
        "turns": asdict(S.mean_se([r.d("turns") for r in rows])),
    }
    if boss:
        out["boss_deaths_pp"] = _est(
            [r.d("died") for r in boss], [r.run for r in boss], [r.player for r in boss], o
        )
        out["boss_pilot_died"] = mean([r.p["died"] for r in boss])
        out["boss_human_died"] = mean([r.h["died"] for r in boss])
        out["boss_fights"] = len(boss)
    return out


def strata_table(
    rows: list[GapRow], o: Opts, strata: list[tuple[str, Callable[[GapRow], bool]]]
) -> list[dict[str, Any]]:
    """Gap and its components per stratum (room, act, ...)."""
    out = []
    seen: set[frozenset[str]] = set()
    for label, pred in strata:
        sel = [r for r in rows if pred(r)]
        keys = frozenset(r.key for r in sel)
        # A stratum holding exactly the fights of an earlier one (room=boss on
        # an all-boss set, A1 boss == act=1 ...) only repeats that row.
        if not sel or keys in seen:
            continue
        seen.add(keys)
        runs = [r.run for r in sel]
        players = [r.player for r in sel]
        n = len(sel)
        out.append(
            {
                "stratum": label,
                "fights": n,
                "pilot": {m: sum(r.p[m] for r in sel) / n for m in METRICS},
                "human": {m: sum(r.h[m] for r in sel) / n for m in METRICS},
                "gap": _est([r.d("L") for r in sel], runs, players, o),
                "deaths_pp": _est([r.d("died") for r in sel], runs, players, o),
                "deaths_pp_cr1": asdict(S.cr1([r.d("died") for r in sel], runs)),
            }
        )
    return out


def player_table(rows: list[GapRow]) -> list[dict[str, Any]]:
    """Per-player gap with a CR1-by-run CI (t with runs-1 df).

    A player with one run falls back to the plain SE over fights (t with
    fights-1 df). These are forest-plot intervals for reading spread, not tests.
    """
    by: dict[str, list[GapRow]] = collections.defaultdict(list)
    for r in rows:
        by[r.player].append(r)
    out = []
    for player, sel in by.items():
        vals = [r.d("L") for r in sel]
        runs = [r.run for r in sel]
        c = S.cr1(vals, runs)
        df = c.g - 1
        if c.g < 2:
            c = S.mean_se(vals)
            df = c.n - 1
        q = float(sps.t.ppf(0.975, df)) if df >= 1 else math.nan
        out.append(
            {
                "player": player,
                "fights": len(sel),
                "runs": len(set(runs)),
                "gap": c.mean,
                "se": c.se,
                "ci_lo": c.mean - q * c.se,
                "ci_hi": c.mean + q * c.se,
                "pilot_died": sum(r.p["died"] for r in sel) / len(sel),
                "human_died": sum(r.h["died"] for r in sel) / len(sel),
            }
        )
    out.sort(key=lambda d: (-d["fights"], d["player"]))
    return out


def encounter_table(rows: list[GapRow]) -> list[dict[str, Any]]:
    """Per-encounter gap, empirical-Bayes shrunk toward its room's mean gap.

    Each encounter's sampling variance is the room's pooled within-encounter
    variance over its fight count, so a one-fight encounter still gets one.
    With that variance the precision-weighted prior mean equals the room's
    fight-weighted mean gap, which is the room row of the strata table.
    """
    out = []
    for room in sorted({r.room for r in rows}):
        by: dict[str, list[float]] = collections.defaultdict(list)
        for r in rows:
            if r.room == room:
                by[r.encounter].append(r.d("L"))
        encs = sorted(by, key=lambda e: (-len(by[e]), e))
        n_tot = sum(len(v) for v in by.values())
        room_mean = sum(sum(v) for v in by.values()) / n_tot
        ss = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in by.values())
        dof = n_tot - len(by)
        if dof > 0 and ss > 0:
            s2 = ss / dof
        else:
            allv = [x for v in by.values() for x in v]
            s2 = float(np.var(allv, ddof=1)) if len(allv) > 1 and np.var(allv) > 0 else 1.0
        est = [sum(by[e]) / len(by[e]) for e in encs]
        var = [s2 / len(by[e]) for e in encs]
        eb = S.eb_shrink(est, var, mu=room_mean)
        for i, e in enumerate(encs):
            out.append(
                {
                    "room": room,
                    "encounter": e,
                    "fights": len(by[e]),
                    "raw": est[i],
                    "se": math.sqrt(var[i]),
                    "shrunk": eb.shrunk[i],
                    "post_sd": eb.post_sd[i],
                    "room_mean": room_mean,
                    "tau": math.sqrt(eb.tau2),
                }
            )
    return out


def compare(rows: list[PairRow], metric: str, o: Opts) -> dict[str, Any]:
    """Paired arm-vs-baseline delta on one metric."""
    if not rows:
        return {"fights": 0}
    d = [r.d(metric) for r in rows]
    scale = 100.0 if metric == "died" else 1.0
    plain = S.mean_se(d)
    c = S.cr1(d, [r.run for r in rows])
    perm = S.sign_flip_test(d, n_perm=o.perm, seed=o.seed)
    return {
        "fights": len(rows),
        "cells": sum(len(r.reps) for r in rows),
        "arm": scale * sum(r.a[metric] for r in rows) / len(rows),
        "base": scale * sum(r.b[metric] for r in rows) / len(rows),
        "delta": plain.mean,
        "se": plain.se,
        "se_cr1": c.se,
        "perm_p": perm.p,
        "perm_exact": perm.exact,
    }


PAIRED_METRICS: tuple[tuple[str, str, str | None], ...] = (
    ("L", "L", None),
    ("L (boss)", "L", "boss"),
    ("deaths pp", "died", None),
    ("deaths pp (boss)", "died", "boss"),
    ("dmg (C#)", "dmg", None),
    ("turns", "turns", None),
)


def combined_pro_gap(
    pr: list[PairRow],
    base_scored: dict[str, dict[int, dict[str, Any]]],
    fights: dict[str, dict[str, Any]],
    variant: str,
    o: Opts,
) -> dict[str, Any]:
    """Pro gap of an arm as gap(B0) + delta(A - B0), one quantity per fight.

    `gap(B0)` uses every scored B0 rep of the fight; the delta uses the common
    reps. Summing per fight and bootstrapping the sum keeps the correlation
    between the two terms, which adding two separate CIs would ignore.
    """
    out: dict[str, Any] = {}
    for metric, label, room in (
        ("L", "gap", None),
        ("died", "deaths_pp", None),
        ("died", "boss_deaths_pp", "boss"),
    ):
        vals, runs, players = [], [], []
        for r in pr:
            if room is not None and r.room != room:
                continue
            f = fights[r.key]
            h = human_metrics(f, o.death_hp, variant)
            if h is None:
                continue
            b0 = mean_metrics(base_scored[r.key].values(), f, o.death_hp)
            scale = 100.0 if metric == "died" else 1.0
            vals.append(scale * (b0[metric] - h[metric]) + r.d(metric))
            runs.append(r.run)
            players.append(r.player)
        out[label] = _est(vals, runs, players, o) if vals else {"n": 0}
    return out


def drop_bounds(
    p: Played,
    fights: dict[str, dict[str, Any]],
    death_hp: float,
) -> dict[str, Any]:
    """Mean L and the human gap if every dropped cell had the best/worst L.

    Expected cells are every planned fight times the header's rep ids. A dropped
    cell is any expected cell that is not scored: a non-scored end or a cell
    that is missing. Best is `L = 0`; worst is `hp_before + DEATH_HP`.
    """
    reps = p.expected_reps()
    if not reps:
        nan2 = [math.nan, math.nan]
        return {"L_scored": math.nan, "L_bound": nan2, "gap_bound_quit_excluded": nan2}
    lo, hi, scored_only, gap_lo, gap_hi = [], [], [], [], []
    for key in p.planned(fights):
        f = fights[key]
        ls = []
        for r in reps:
            rec = p.cells.get((key, r))
            ls.append(rep_loss(rec, f, death_hp) if rec and rec.get("end") in SCORED_ENDS else None)
        worst = f["hp_before"] + death_hp
        fl = sum(0.0 if x is None else x for x in ls) / len(ls)
        fh = sum(worst if x is None else x for x in ls) / len(ls)
        lo.append(fl)
        hi.append(fh)
        got = [x for x in ls if x is not None]
        if got:
            scored_only.append(sum(got) / len(got))
        h = human_metrics(f, death_hp, "quit_excluded")
        if h is not None:
            gap_lo.append(fl - h["L"])
            gap_hi.append(fh - h["L"])

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else math.nan

    return {
        "L_scored": mean(scored_only),
        "L_bound": [mean(lo), mean(hi)],
        "gap_bound_quit_excluded": [mean(gap_lo), mean(gap_hi)],
    }


def delta_bounds(
    a: Played, b: Played, fights: dict[str, dict[str, Any]], death_hp: float
) -> list[float]:
    """Range of the L delta over all expected common cells when dropped cells are best/worst."""
    reps = sorted(set(a.expected_reps()) & set(b.expected_reps()))
    if not reps:
        return [math.nan, math.nan]
    lo, hi = [], []
    common = set(b.planned(fights))
    for key in (k for k in a.planned(fights) if k in common):
        f = fights[key]
        worst = f["hp_before"] + death_hp
        dl = dh = 0.0
        for r in reps:
            ra, rb = a.cells.get((key, r)), b.cells.get((key, r))
            la = rep_loss(ra, f, death_hp) if ra and ra.get("end") in SCORED_ENDS else None
            lb = rep_loss(rb, f, death_hp) if rb and rb.get("end") in SCORED_ENDS else None
            dl += (0.0 if la is None else la) - (worst if lb is None else lb)
            dh += (worst if la is None else la) - (0.0 if lb is None else lb)
        lo.append(dl / len(reps))
        hi.append(dh / len(reps))
    if not lo:
        return [math.nan, math.nan]
    return [sum(lo) / len(lo), sum(hi) / len(hi)]


def cell_summary(p: Played, fights: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """End-status counts, missing and dropped cells."""
    ends = collections.Counter(c.get("end") for c in p.cells.values())
    expected = len(p.planned(fights)) * len(p.expected_reps())
    scored = sum(ends[e] for e in SCORED_ENDS)
    errors = [c.get("error") for c in p.cells.values() if c.get("end") == "error"][:3]
    return {
        "expected": expected,
        "present": len(p.cells),
        "missing": expected - len(p.cells),
        "scored": scored,
        "dropped": expected - scored,
        "ends": {e: ends.get(e, 0) for e in ENDS}
        | {e: n for e, n in ends.items() if e not in ENDS},
        "duplicates": p.duplicates,
        "retried": p.retried,
        "unknown_keys": p.unknown_keys,
        "blocks_dropped": p.blocks_dropped,
        "error_examples": errors,
    }


def cost_summary(p: Played, scored: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Search cost over the arm's scored cells."""
    recs = [rec for reps in scored.values() for rec in reps.values()]
    dec = sum(int(r.get("decisions") or 0) for r in recs)
    sims = [r.get("sims_spent") for r in recs]
    cpu = [r.get("cpu_s") for r in recs]
    tot_sims = None if not recs or any(s is None for s in sims) else sum(sims)
    tot_cpu = None if not recs or any(c is None for c in cpu) else sum(cpu)
    return {
        "cells": len(recs),
        "decisions": dec,
        "sims_per_decision": tot_sims / dec if tot_sims is not None and dec else None,
        "cpu_s_per_decision": tot_cpu / dec if tot_cpu is not None and dec else None,
        "cpu_s_per_fight": tot_cpu / len(recs) if tot_cpu is not None and recs else None,
    }


def cost_ratio(
    a: dict[str, dict[int, dict[str, Any]]], b: dict[str, dict[int, dict[str, Any]]]
) -> dict[str, float | None]:
    """Sims and CPU of arm `a` over baseline `b` on their common scored cells."""
    out: dict[str, float | None] = {}
    for fld in ("sims_spent", "cpu_s", "decisions"):
        sa = sb = 0.0
        ok = True
        for key in set(a) & set(b):
            for r in set(a[key]) & set(b[key]):
                va, vb = a[key][r].get(fld), b[key][r].get(fld)
                if va is None or vb is None:
                    ok = False
                    break
                sa += va
                sb += vb
            if not ok:
                break
        out[fld] = sa / sb if ok and sb else None
    return out


def split_defs(fights: dict[str, dict[str, Any]]) -> list[tuple[str, Callable[[str], bool]]]:
    """Fight subsets the report repeats its headline numbers on."""
    flags = sorted({fl for f in fights.values() for fl in f.get("flags") or []})
    out: list[tuple[str, Callable[[str], bool]]] = [
        ("all", lambda k: True),
        ("clean (no flags)", lambda k: not fights[k].get("flags")),
    ]
    for fl in flags:
        out.append((f"flag {fl}", lambda k, fl=fl: fl in (fights[k].get("flags") or [])))
    out.append(("belt_unused", lambda k: bool(fights[k].get("belt_unused"))))
    out.append(("not belt_unused", lambda k: not fights[k].get("belt_unused")))
    for fl in ("in_rl2_train", "in_rl2_gen"):
        out.append((f"not {fl}", lambda k, fl=fl: fl not in (fights[k].get("flags") or [])))
    cohorts = sorted({f.get("cohort") for f in fights.values() if f.get("cohort")})
    if len(cohorts) > 1:
        for c in cohorts:
            out.append((f"cohort {c}", lambda k, c=c: fights[k].get("cohort") == c))
    return out


# ── self-null ───────────────────────────────────────────────────────────


def self_null_rows(
    fights: dict[str, dict[str, Any]],
    scored: dict[str, dict[int, dict[str, Any]]],
    death_hp: float,
    pick: Callable[[str, list[int]], int],
) -> list[GapRow]:
    """Rows where one rep plays the human and the fight's other reps the pilot.

    Args:
        fights: The set.
        scored: The file's scored cells.
        death_hp: HP equivalent of a death.
        pick: `(key, sorted rep ids) -> rep id` that plays the human.
    """
    pilot: dict[str, dict[int, dict[str, Any]]] = {}
    human: dict[str, dict[str, float]] = {}
    for key, reps in scored.items():
        if len(reps) < 2:
            continue
        ids = sorted(reps)
        hr = pick(key, ids)
        human[key] = rep_metrics(reps[hr], fights[key], death_hp)
        pilot[key] = {r: v for r, v in reps.items() if r != hr}
    return gap_rows(fights, pilot, human.get, death_hp)


def self_null_coverage(
    fights: dict[str, dict[str, Any]],
    scored: dict[str, dict[int, dict[str, Any]]],
    o: Opts,
    trials: int,
) -> dict[str, Any]:
    """How often each 95% CI covers the true gap 0 across 'human' assignments.

    Trials are the deterministic rotations (the k-th scored rep of every fight
    plays the human) followed by `trials` random per-fight picks. Each trial
    is read on every stratum (all, each room, each act). Trials share data,
    so coverage fractions are correlated; read them as a calibration check,
    not as an exact coverage estimate.
    """
    max_reps = max((len(v) for v in scored.values()), default=0)
    picks: list[Callable[[str, list[int]], int]] = [
        (lambda k, ids, j=j: ids[j % len(ids)]) for j in range(max_reps)
    ]
    rng = np.random.default_rng(o.seed + 1)
    for _ in range(trials):
        choice = {k: int(rng.integers(0, 1 << 30)) for k in scored}
        picks.append(lambda k, ids, ch=choice: ids[ch[k] % len(ids)])
    strata: list[tuple[str, Callable[[GapRow], bool]]] = [("all", lambda r: True)]
    strata += [(f"room={x}", lambda r, x=x: r.room == x) for x in ROOMS]
    strata += [(f"act={a}", lambda r, a=a: r.act == a) for a in (1, 2, 3)]
    hits = {"plain": 0, "cr1": 0, "wild_player": 0}
    n = 0
    gaps = []
    for pick in picks:
        rows = self_null_rows(fights, scored, o.death_hp, pick)
        seen: set[frozenset[str]] = set()
        for label, pred in strata:
            sel = [r for r in rows if pred(r)]
            keys = frozenset(r.key for r in sel)
            if len(sel) < 2 or keys in seen:
                continue
            seen.add(keys)
            e = _est([r.d("L") for r in sel], [r.run for r in sel], [r.player for r in sel], o)
            if any(math.isnan(e[k]) for k in ("se", "se_cr1", "ci_lo", "ci_hi")):
                continue
            if label == "all":
                gaps.append(e["mean"])
            n += 1
            hits["plain"] += abs(e["mean"]) <= S.Z95 * e["se"]
            hits["cr1"] += abs(e["mean"]) <= S.Z95 * e["se_cr1"]
            hits["wild_player"] += e["ci_lo"] <= 0.0 <= e["ci_hi"]
    return {
        "trials": len(picks),
        "cases": n,
        "coverage": {k: v / n if n else math.nan for k, v in hits.items()},
        "gap_all_mean": sum(gaps) / len(gaps) if gaps else math.nan,
        "gap_all_min": min(gaps) if gaps else math.nan,
        "gap_all_max": max(gaps) if gaps else math.nan,
    }


# ── text output ─────────────────────────────────────────────────────────


def _f(x: Any, spec: str = "+.2f") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return format(x, spec)


def _ci(lo: Any, hi: Any, spec: str = "+.2f") -> str:
    return f"[{_f(lo, spec)}, {_f(hi, spec)}]"


def _ci_short(lo: Any, hi: Any) -> str:
    """A narrow CI for the splits table; bounds past +-999 (two clusters) print as such."""

    def one(x: Any) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        if abs(x) > 999:
            return ">+999" if x > 0 else "<-999"
        return format(x, "+.1f")

    return f"[{one(lo)}, {one(hi)}]"


def _p(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if x <= 0:
        return "0"
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}"


def _forest(lo: float, mean: float, hi: float, lim: float, width: int = 31) -> str:
    """One ASCII forest-plot row on the symmetric axis [-lim, +lim]; `|` marks 0."""
    if not all(map(math.isfinite, (lo, mean, hi))) or lim <= 0:
        return ""

    def pos(x: float) -> int:
        return round((max(-lim, min(lim, x)) + lim) / (2 * lim) * (width - 1))

    row = [" "] * width
    for i in range(pos(lo), pos(hi) + 1):
        row[i] = "-"
    row[pos(0.0)] = "|"
    row[pos(mean)] = "*"
    return "".join(row)


def print_report(rep: dict[str, Any], out: Any = None) -> None:
    """Render the JSON summary as text tables (to `out`, default the current stdout)."""
    out = sys.stdout if out is None else out

    def w(s: str = "") -> None:
        print(s, file=out)

    st = rep["set"]
    aw = max([14, *(len(a["name"]) for a in rep["arms"])])
    w(f"=== combat_bench report: {st['path']} ===")
    of_set = st.get("fights_in_set")
    w(
        f"set sha256 {st['sha256'][:16]}  fights {st['fights']}"
        + (f" (of {of_set} in the set)" if of_set and of_set != st["fights"] else "")
        + f"  runs {st['runs']}  players {st['players']}  rooms {st['rooms']}"
    )
    w(
        f"L = min(dmg_c + dmg_unlogged, hp_before), or hp_before + {rep['death_hp']:g} on a "
        f"death; human L the same with damage_taken; human quits in set: {st['quits']} "
        "(excl = dropped; surv / death = the two ends of the quit bracket)"
    )
    w(f"baseline arm: {rep['baseline']}")
    # Without quits both variants are the same numbers; print one.
    variants = VARIANTS if st["quits"] else VARIANTS[:1]
    for a in rep["arms"]:
        h = a["header"]
        w(
            f"  {a['name']:{aw}s} {a['path']}\n"
            f"  {'':{aw}s} spec {json.dumps(h.get('spec'), sort_keys=True)}\n"
            f"  {'':{aw}s} src_hash {h.get('src_hash')} pgo {h.get('pgo')} reps {h.get('reps')} "
            f"rep_offset {h.get('rep_offset')} caps {json.dumps(h.get('caps'))}"
        )
    for x in rep["warnings"]:
        w(f"WARNING: {x}")
    w()

    w("== cells (only won/died are scored) ==")
    w(
        f"  {'arm':{aw}s} {'expected':>8s} {'scored':>7s} {'missing':>7s} "
        + " ".join(f"{e:>9s}" for e in ENDS)
        + f" {'dropped':>8s}"
    )
    for a in rep["arms"]:
        c = rep["cells"][a["name"]]
        w(
            f"  {a['name']:{aw}s} {c['expected']:8d} {c['scored']:7d} {c['missing']:7d} "
            + " ".join(f"{c['ends'].get(e, 0):9d}" for e in ENDS)
            + f" {c['dropped'] / max(1, c['expected']):8.2%}"
        )
        for e in c["error_examples"]:
            w(f"  {'':{aw}s} error: {str(e)[:160]}")
    w("  bound if every dropped cell were the best (L=0) / worst (L=hp_before+death) outcome:")
    for a in rep["arms"]:
        b = rep["drop_bounds"][a["name"]]
        w(
            f"  {a['name']:{aw}s} mean L scored {_f(b['L_scored'], '.2f')}  "
            f"all-cells L in {_ci(*b['L_bound'], spec='.2f')}  "
            f"gap (quit excl) in {_ci(*b['gap_bound_quit_excluded'])}"
        )
    w()

    w("== pilot vs human: gap = L_pilot - L_human per fight (+ = pilot loses more) ==")
    w(
        "   SE = fight-level; CR1 = clustered by run; CI = 95% wild cluster bootstrap by player "
        "(Webb, restricted, inverted); p0 = wild p for gap == 0 (0: no draw as extreme)"
    )
    w(
        f"  {'arm':{aw}s} {'quit':5s} {'fights':>6s} {'runs':>5s} {'plyr':>4s} {'pilot L':>8s} "
        f"{'human L':>8s} {'gap':>7s} {'SE':>5s} {'CR1':>5s} {'95% CI (player)':>18s} "
        f"{'p0':>7s}   {'equal-player gap [CI]':>24s}"
    )
    for a in rep["arms"]:
        for v in variants:
            s = rep["human_gap"][a["name"]][v]["overall"]
            if not s.get("fights"):
                continue
            g, eq = s["gap"], s["gap_equal_player"]
            w(
                f"  {a['name']:{aw}s} {QLABEL[v]:5s} {s['fights']:6d} {s['runs']:5d} "
                f"{s['players']:4d} {s['pilot']['L']:8.2f} {s['human']['L']:8.2f} "
                f"{_f(g['mean']):>7s} {_f(g['se'], '.2f'):>5s} {_f(g['se_cr1'], '.2f'):>5s} "
                f"{_ci(g['ci_lo'], g['ci_hi']):>18s} {_p(g['p0']):>7s}   "
                f"{_f(eq['mean'])} {_ci(eq['ci_lo'], eq['ci_hi'])} ({eq['players']} pl)"
            )
    w()
    w("  deaths (pp), all rooms and bosses only (boss deaths are co-primary on bosses):")
    w(
        f"  {'arm':{aw}s} {'quit':5s} {'pilot':>6s} {'human':>6s} {'gap pp':>7s} {'CR1':>5s} "
        f"{'95% CI (player)':>18s} | {'boss n':>6s} {'pilot':>6s} {'human':>6s} "
        f"{'gap pp':>7s} {'CR1':>5s} {'95% CI (player)':>18s}"
    )
    for a in rep["arms"]:
        for v in variants:
            s = rep["human_gap"][a["name"]][v]["overall"]
            if not s.get("fights"):
                continue
            d = s["deaths_pp"]
            line = (
                f"  {a['name']:{aw}s} {QLABEL[v]:5s} {100 * s['pilot']['died']:5.1f}% "
                f"{100 * s['human']['died']:5.1f}% {_f(d['mean']):>7s} "
                f"{_f(d['se_cr1'], '.2f'):>5s} {_ci(d['ci_lo'], d['ci_hi']):>18s} |"
            )
            if "boss_deaths_pp" in s:
                b = s["boss_deaths_pp"]
                line += (
                    f" {s['boss_fights']:6d} {100 * s['boss_pilot_died']:5.1f}% "
                    f"{100 * s['boss_human_died']:5.1f}% {_f(b['mean']):>7s} "
                    f"{_f(b['se_cr1'], '.2f'):>5s} {_ci(b['ci_lo'], b['ci_hi']):>18s}"
                )
            w(line)
    w(
        "  death CIs cover ~93%, not 95% (rare deaths over ~12 unbalanced players; calibrated "
        "on the pro boss layout): a CI that only just excludes 0 is not decisive."
    )
    w()

    for a in rep["arms"]:
        w(f"-- {a['name']}: by room / act (quit excluded) --")
        w(
            f"  {'stratum':12s} {'fights':>6s} {'pilot L':>8s} {'human L':>8s} {'gap':>7s} "
            f"{'CR1':>5s} {'95% CI (player)':>18s} {'died p/h %':>12s} {'dmg p/h':>12s} "
            f"{'turns p/h':>10s}"
        )
        for r in rep["human_gap"][a["name"]]["quit_excluded"]["strata"]:
            g, p, h = r["gap"], r["pilot"], r["human"]
            w(
                f"  {r['stratum']:12s} {r['fights']:6d} {p['L']:8.2f} {h['L']:8.2f} "
                f"{_f(g['mean']):>7s} {_f(g['se_cr1'], '.2f'):>5s} "
                f"{_ci(g['ci_lo'], g['ci_hi']):>18s} "
                f"{100 * p['died']:5.1f}/{100 * h['died']:5.1f} "
                f"{p['dmg']:5.1f}/{h['dmg']:5.1f} {p['turns']:4.1f}/{h['turns']:4.1f}"
            )
        w()

    for a in rep["arms"]:
        pl = rep["human_gap"][a["name"]]["quit_excluded"]["players"]
        if not pl:
            continue
        shown = [x for x in pl if x["fights"] >= rep["forest_min_fights"]][:40]
        # Axis: the largest |gap| plus the median half-width, so one player with
        # two runs (t with 1 df) does not squash every other row; wider CIs clip.
        half = sorted(
            (x["ci_hi"] - x["ci_lo"]) / 2 for x in shown if math.isfinite(x["ci_hi"] - x["ci_lo"])
        )
        lim = max([abs(x["gap"]) for x in shown if math.isfinite(x["gap"])] or [0.0]) + (
            half[len(half) // 2] if half else 1.0
        )
        w(
            f"-- {a['name']}: per-player gap (quit excluded; CR1-by-run t CI; "
            f"{len(shown)} of {len(pl)} players with >= {rep['forest_min_fights']} fights; "
            f"axis +-{lim:.0f}) --"
        )
        for x in shown:
            w(
                f"  {x['player']:16s} {x['fights']:5d} f {x['runs']:3d} r  "
                f"{_f(x['gap']):>7s} {_ci(x['ci_lo'], x['ci_hi']):>18s}  "
                f"died p/h {100 * x['pilot_died']:5.1f}/{100 * x['human_died']:5.1f}  "
                f"{_forest(x['ci_lo'], x['gap'], x['ci_hi'], lim)}"
            )
        w()

    names = [a["name"] for a in rep["arms"]]
    enc = {n: rep["human_gap"][n]["quit_excluded"]["encounters"] for n in names}
    base = rep["baseline"]
    if enc[base]:
        w(
            "-- per-encounter gap (quit excluded): raw and empirical-Bayes shrunk toward the "
            "room mean --"
        )
        w(
            f"  {'room':8s} {'encounter':30s} {'n':>4s} {'raw ' + base[:9]:>14s} "
            + " ".join(f"{'EB ' + n[:11]:>14s}" for n in names)
        )
        idx = {n: {(e["room"], e["encounter"]): e for e in enc[n]} for n in names}
        for e in enc[base]:
            k = (e["room"], e["encounter"])
            cells = [
                f"{_f(idx[n][k]['shrunk']):>8s}±{idx[n][k]['post_sd']:4.1f}"
                if k in idx[n]
                else f"{'':>14s}"
                for n in names
            ]
            w(
                f"  {e['room']:8s} {e['encounter'][:30]:30s} {e['fights']:4d} "
                f"{_f(e['raw']):>8s}±{e['se']:4.1f} " + " ".join(f"{c:>14s}" for c in cells)
            )
        w()

    if rep["paired"]:
        w(
            f"== arm vs arm: paired on the common scored (key, rep) cells; baseline {base} "
            "(+ = arm higher) =="
        )
        w(
            "   perm p = fight-level sign-flip (two-sided); Holm across the non-baseline arms, "
            "per metric"
        )
        w(
            f"  {'metric':17s} {'arm':{aw}s} {'fights':>6s} {'cells':>6s} {'base':>8s} "
            f"{'arm':>8s} {'delta':>8s} {'SE':>6s} {'CR1':>6s} {'perm p':>8s} {'Holm p':>8s}"
        )
        for label, _, _ in PAIRED_METRICS:
            for n in names:
                if n not in rep["paired"]:
                    continue
                c = rep["paired"][n]["metrics"][label]
                if not c.get("fights"):
                    continue
                w(
                    f"  {label:17s} {n:{aw}s} {c['fights']:6d} {c['cells']:6d} "
                    f"{c['base']:8.2f} {c['arm']:8.2f} {_f(c['delta'], '+8.2f')} "
                    f"{_f(c['se'], '6.2f')} {_f(c['se_cr1'], '6.2f')} {_p(c['perm_p']):>8s} "
                    f"{_p(c['holm_p']):>8s}"
                )
        w("  L delta if every dropped cell were best/worst (arm best + base worst .. reverse):")
        for n in names:
            if n in rep["paired"]:
                w(f"  {n:{aw}s} {_ci(*rep['paired'][n]['delta_bound_L'])}")
        w()

    w("== pro gap via the baseline: gap(B0) + delta(A - B0) per fight, one bootstrap ==")
    w(
        f"  {'arm':{aw}s} {'quit':5s} {'fights':>6s} {'gap':>7s} {'CR1':>5s} "
        f"{'95% CI (player)':>18s} {'deaths pp':>9s} {'95% CI (player)':>18s} "
        f"{'boss d pp':>9s} {'95% CI (player)':>18s}"
    )
    for n in names:
        for v in variants:
            c = rep["pro_gap_via_baseline"][n][v]
            g, d, bd = c["gap"], c["deaths_pp"], c["boss_deaths_pp"]
            if not g.get("n"):
                continue
            boss = (
                f" {_f(bd['mean']):>9s} {_ci(bd['ci_lo'], bd['ci_hi']):>18s}" if bd.get("n") else ""
            )
            w(
                f"  {n:{aw}s} {QLABEL[v]:5s} {g['n']:6d} {_f(g['mean']):>7s} "
                f"{_f(g['se_cr1'], '.2f'):>5s} {_ci(g['ci_lo'], g['ci_hi']):>18s} "
                f"{_f(d['mean']):>9s} {_ci(d['ci_lo'], d['ci_hi']):>18s}" + boss
            )
    w()

    w("== cost (scored cells; sims is the primary cost metric) ==")
    if rep.get("cost_cpu_not_comparable"):
        w(f"  cpu x base is n/a: {rep['cost_cpu_not_comparable']}")
    w(
        f"  {'arm':{aw}s} {'cells':>6s} {'decisions':>9s} {'sims/dec':>9s} {'cpu ms/dec':>10s} "
        f"{'cpu s/fight':>11s} {'sims x base':>11s} {'cpu x base':>10s} {'load/cpu':>8s}  host"
    )
    for n in names:
        c = rep["cost"][n]
        r = c.get("vs_base", {})
        cpu_dec = c["cpu_s_per_decision"]
        w(
            f"  {n:{aw}s} {c['cells']:6d} {c['decisions']:9d} "
            f"{_f(c['sims_per_decision'], '.0f'):>9s} "
            f"{_f(None if cpu_dec is None else 1000 * cpu_dec, '.2f'):>10s} "
            f"{_f(c['cpu_s_per_fight'], '.2f'):>11s} {_f(r.get('sims_spent'), '.2f'):>11s} "
            f"{_f(r.get('cpu_s'), '.2f'):>10s} {_f(c.get('load_per_cpu'), '.2f'):>8s}  "
            f"{','.join(c.get('hosts') or []) or '?'} (pgo {c.get('pgo')})"
        )
    w()

    w(
        "== splits: gap (quit excluded) [95% wild CI by player] and paired L delta vs "
        "baseline +- SE =="
    )
    w(
        f"  {'split':26s} {'fights':>6s} "
        + " ".join(f"{'gap ' + n[:10]:>23s}" for n in names)
        + " "
        + " ".join(f"{'d ' + n[:10]:>15s}" for n in names if n != base)
    )
    for sp in rep["splits"]:
        gaps = [
            f"{_f(sp['gap'][n]['mean'], '+.2f')} "
            f"{_ci_short(sp['gap'][n].get('ci_lo'), sp['gap'][n].get('ci_hi'))}"
            if sp["gap"].get(n, {}).get("n")
            else "-"
            for n in names
        ]
        deltas = [
            f"{_f(sp['delta'][n]['mean'], '+.2f')}±{_f(sp['delta'][n]['se'], '.2f')}"
            if sp["delta"].get(n, {}).get("n")
            else "-"
            for n in names
            if n != base
        ]
        w(
            f"  {sp['split'][:26]:26s} {sp['fights']:6d} "
            + " ".join(f"{g:>23s}" for g in gaps)
            + " "
            + " ".join(f"{d:>15s}" for d in deltas)
        )
    w()

    sn = rep.get("self_null")
    if sn:
        w(f"== self-null: {sn['path']} (lowest scored rep = 'human', others = pilot; truth 0) ==")
        for s in sn["strata"]:
            g = s["gap"]
            w(
                f"  {s['stratum']:12s} {s['fights']:6d} fights  gap {_f(g['mean'])}  "
                f"SE {_f(g['se'], '.2f')}  CR1 {_f(g['se_cr1'], '.2f')}  "
                f"CI {_ci(g['ci_lo'], g['ci_hi'])}  p0 {_p(g['p0'])}"
            )
        cov = sn.get("coverage")
        if cov:
            c = cov["coverage"]
            w(
                f"  coverage of 0 by the 95% CIs over {cov['trials']} trials x strata = "
                f"{cov['cases']} cases: plain {c['plain']:.1%}  CR1 {c['cr1']:.1%}  "
                f"wild-player {c['wild_player']:.1%}"
            )
            w(
                f"  overall gap across trials: mean {_f(cov['gap_all_mean'])}  "
                f"range {_ci(cov['gap_all_min'], cov['gap_all_max'])}"
            )
        w()


# ── driver ──────────────────────────────────────────────────────────────


def _human_strata() -> list[tuple[str, Callable[[GapRow], bool]]]:
    out: list[tuple[str, Callable[[GapRow], bool]]] = [("all", lambda r: True)]
    out += [(f"room={x}", lambda r, x=x: r.room == x) for x in ROOMS]
    out += [(f"act={a}", lambda r, a=a: r.act == a) for a in (1, 2, 3)]
    out += [
        (f"A{a} {x}", lambda r, a=a, x=x: r.act == a and r.room == x)
        for a in (1, 2, 3)
        for x in ROOMS
    ]
    return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def build_report(
    set_path: str,
    fights: dict[str, dict[str, Any]],
    played: list[Played],
    baseline: str,
    o: Opts,
    warnings: list[str],
    self_null: Played | None = None,
    self_null_trials: int | None = None,
    forest_min_fights: int = 5,
) -> dict[str, Any]:
    """Every number the report prints, as one JSON-ready dict."""
    scored = {p.name: p.scored() for p in played}
    by_name = {p.name: p for p in played}
    rep: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "set": {
            "path": set_path,
            "sha256": file_sha256(set_path),
            "fights": len(fights),
            "runs": len({f["file"] for f in fights.values()}),
            "players": len({f["player"] for f in fights.values()}),
            "rooms": dict(collections.Counter(f["room_type"] for f in fights.values())),
            "quits": sum(1 for f in fights.values() if f["human"].get("quit")),
        },
        "death_hp": o.death_hp,
        "boot": o.boot,
        "perm": o.perm,
        "seed": o.seed,
        "baseline": baseline,
        "forest_min_fights": forest_min_fights,
        "arms": [{"name": p.name, "path": p.path, "header": p.header} for p in played],
        "warnings": warnings,
        "cells": {p.name: cell_summary(p, fights) for p in played},
        "drop_bounds": {p.name: drop_bounds(p, fights, o.death_hp) for p in played},
        "human_gap": {},
        "paired": {},
        "pro_gap_via_baseline": {},
        "cost": {},
        "splits": [],
    }
    rows_by: dict[str, dict[str, list[GapRow]]] = {}
    for p in played:
        rows_by[p.name] = {}
        rep["human_gap"][p.name] = {}
        for v in VARIANTS:
            rows = gap_rows(
                fights,
                scored[p.name],
                lambda k, v=v: human_metrics(fights[k], o.death_hp, v),
                o.death_hp,
            )
            rows_by[p.name][v] = rows
            entry: dict[str, Any] = {"overall": summarize_gap(rows, o)}
            if v == "quit_excluded":
                entry["strata"] = strata_table(rows, o, _human_strata())
                entry["players"] = player_table(rows)
                entry["encounters"] = encounter_table(rows)
            rep["human_gap"][p.name][v] = entry

    pairs: dict[str, list[PairRow]] = {}
    for p in played:
        if p.name == baseline:
            continue
        pr = pair_rows(fights, scored[p.name], scored[baseline], o.death_hp)
        pairs[p.name] = pr
        metrics = {}
        for label, metric, room in PAIRED_METRICS:
            sel = [r for r in pr if room is None or r.room == room]
            metrics[label] = compare(sel, metric, o)
        rep["paired"][p.name] = {
            "metrics": metrics,
            "delta_bound_L": delta_bounds(p, by_name[baseline], fights, o.death_hp),
        }
    for label, _, _ in PAIRED_METRICS:
        arms = [n for n in rep["paired"] if rep["paired"][n]["metrics"][label].get("fights")]
        adj = S.holm([rep["paired"][n]["metrics"][label]["perm_p"] for n in arms])
        for n, a in zip(arms, adj):
            rep["paired"][n]["metrics"][label]["holm_p"] = a
        for n in rep["paired"]:
            rep["paired"][n]["metrics"][label].setdefault("holm_p", math.nan)

    self_pairs = pair_rows(fights, scored[baseline], scored[baseline], o.death_hp)
    cpu_why = cpu_comparable(played)
    rep["cost_cpu_not_comparable"] = cpu_why
    for p in played:
        pr = pairs.get(p.name, self_pairs)
        rep["pro_gap_via_baseline"][p.name] = {
            v: combined_pro_gap(pr, scored[baseline], fights, v, o) for v in VARIANTS
        }
        c = cost_summary(p, scored[p.name])
        c["hosts"] = sorted(cell_hosts(p))
        c["pgo"] = p.header.get("pgo")
        c["load_per_cpu"] = load_per_cpu(p)
        if p.name != baseline:
            c["vs_base"] = cost_ratio(scored[p.name], scored[baseline])
            if cpu_why:
                c["vs_base"]["cpu_s"] = None
        rep["cost"][p.name] = c

    for label, pred in split_defs(fights):
        keys = {k for k in fights if pred(k)}
        if not keys:
            continue
        sp: dict[str, Any] = {"split": label, "fights": len(keys), "gap": {}, "delta": {}}
        for p in played:
            rows = [r for r in rows_by[p.name]["quit_excluded"] if r.key in keys]
            if not rows:
                sp["gap"][p.name] = {"n": 0}
                continue
            runs, players = [r.run for r in rows], [r.player for r in rows]
            # The wild-by-player CI, as everywhere else: CR1 by run ignores
            # the player effect and under-covers on these layouts.
            g = _est([r.d("L") for r in rows], runs, players, o)
            dd = _est([r.d("died") for r in rows], runs, players, o)
            sp["gap"][p.name] = {
                "mean": g["mean"],
                "se_cr1": g["se_cr1"],
                "ci_lo": g["ci_lo"],
                "ci_hi": g["ci_hi"],
                "n": g["n"],
                "players": g["players"],
                "deaths_pp": dd["mean"],
                "deaths_pp_se_cr1": dd["se_cr1"],
                "deaths_pp_ci_lo": dd["ci_lo"],
                "deaths_pp_ci_hi": dd["ci_hi"],
            }
            if p.name in pairs:
                prs = [r for r in pairs[p.name] if r.key in keys]
                e = S.mean_se([r.d("L") for r in prs])
                ed = S.mean_se([r.d("died") for r in prs])
                sp["delta"][p.name] = {
                    "mean": e.mean,
                    "se": e.se,
                    "n": e.n,
                    "deaths_pp": ed.mean,
                    "deaths_pp_se": ed.se,
                }
        rep["splits"].append(sp)

    if self_null is not None:
        sc = self_null.scored()
        rows = self_null_rows(fights, sc, o.death_hp, lambda k, ids: ids[0])
        sn: dict[str, Any] = {
            "path": self_null.path,
            "strata": strata_table(rows, o, _human_strata()[:7]),
        }
        if self_null_trials is not None:
            sn["coverage"] = self_null_coverage(fights, sc, o, self_null_trials)
        rep["self_null"] = sn
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ppo.combat_bench report",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--set", required=True, dest="set_path", help="fight set JSONL")
    ap.add_argument("played", nargs="*", help="played files, [NAME=]PATH")
    ap.add_argument("--baseline", default=None, help="baseline arm name or path (default: first)")
    ap.add_argument("--death-hp", type=float, default=DEATH_HP, help="HP equivalent of a death")
    ap.add_argument("--allow-engine-mismatch", action="store_true")
    ap.add_argument("--allow-set-mismatch", action="store_true")
    ap.add_argument("--complete-blocks-only", action="store_true")
    ap.add_argument("--boot", type=int, default=9999, help="wild bootstrap draws")
    ap.add_argument("--perm", type=int, default=20000, help="random sign flips (exact if fewer)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forest-min-fights", type=int, default=5)
    ap.add_argument("--self-null", default=None, help="played file for the self-null check")
    ap.add_argument("--self-null-coverage", action="store_true")
    ap.add_argument("--self-null-trials", type=int, default=20, help="random trials on top")
    ap.add_argument("--json", default=None, help="write the summary JSON here")
    args = ap.parse_args(argv)
    if not args.played and not args.self_null:
        ap.error("give at least one played file or --self-null")
    if args.self_null_coverage and not args.self_null:
        ap.error("--self-null-coverage needs --self-null")

    o = Opts(death_hp=args.death_hp, boot=args.boot, perm=args.perm, seed=args.seed)
    try:
        fights = load_set(args.set_path)
        set_sha = file_sha256(args.set_path)
        specs = [_arm_spec(a) for a in args.played]
        if len({n for n, _ in specs}) != len(specs):
            raise BenchError("two arms share a name; name them with NAME=PATH")
        played = [load_played(path, name, fights) for name, path in specs]
        sn = load_played(args.self_null, "self_null", fights) if args.self_null else None
        for p in played + ([sn] if sn else []):
            resolve_selection(p, fights, args.allow_set_mismatch)
        if args.complete_blocks_only:
            played = [complete_blocks_only(p, len(p.planned(fights))) for p in played]
            sn = complete_blocks_only(sn, len(sn.planned(fights))) if sn else None
            interim = []
        else:
            interim = interim_warnings(played + ([sn] if sn else []), fights)
        warns = interim + validate(
            set_sha,
            played + ([sn] if sn else []),
            allow_engine_mismatch=args.allow_engine_mismatch,
            allow_set_mismatch=args.allow_set_mismatch,
        )
        plans = {p.name: p.planned(fights) for p in played + ([sn] if sn else [])}
        if len({tuple(v) for v in plans.values()}) > 1:
            warns.append(
                "arms planned different fights ("
                + ", ".join(f"{n}={len(v)}" for n, v in plans.items())
                + "); each arm's cells are counted on its own plan, deltas on the common fights"
            )
        # Report on the fights some arm planned, not the whole set: the set
        # line, the splits and the self-null all read `fights`.
        planned = {k for v in plans.values() for k in v}
        n_set = len(fights)
        fights = {k: f for k, f in fights.items() if k in planned}
        if not played:
            assert sn is not None
            played = [sn]
            sn_only = True
        else:
            sn_only = False
        names = [p.name for p in played]
        baseline = names[0]
        if args.baseline:
            hit = [p.name for p in played if args.baseline in (p.name, p.path)]
            if not hit:
                raise BenchError(f"--baseline {args.baseline!r} is none of {names}")
            baseline = hit[0]
    except BenchError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    for x in warns:
        print(f"WARNING: {x}", file=sys.stderr)
    rep = build_report(
        args.set_path,
        fights,
        played,
        baseline,
        o,
        warns,
        self_null=sn,
        self_null_trials=args.self_null_trials if args.self_null_coverage else None,
        forest_min_fights=args.forest_min_fights,
    )
    rep["self_null_only"] = sn_only
    rep["set"]["fights_in_set"] = n_set
    print_report(rep)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(rep), fh, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
