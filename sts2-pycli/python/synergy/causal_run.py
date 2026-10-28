"""Parallel orchestration for the causal synergy layers.

Drives the two causal estimators over a *set* of card pairs and writes the
score files the catalog join (:mod:`synergy.cli` ``emit-catalog``) consumes:

* :func:`compute_econ_table` — value-net second difference (cheap; Ironclad
  only, since the bundled value net is Ironclad-trained) over every pair.
* :func:`compute_combat_table` — four-cell forced-deck fight interaction
  (character-agnostic) over a *curated* pair subset.

Both fan out across processes with :class:`concurrent.futures.ProcessPoolExecutor`.
Each worker re-imports the per-pair estimator and reads its shared inputs
(contexts / baselines) from module-level globals seeded by the pool
``initializer`` — so the heavy inputs are pickled once per worker, not once
per task.

Pair selection (:func:`select_pairs`) deliberately spans the observational
spectrum: the top-PMI head surfaces *confirmed* synergies (high-obs ∧ causal),
and a random below-median sample surfaces *undiscovered* ones (low-obs ∧
causal).  Without the low-obs sample the catalog can never populate the
undiscovered quadrant.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from synergy.causal_combat import CombatRow, combat_synergy
from synergy.causal_econ import State, econ_synergy, sample_contexts
from synergy.corpus import Deck, iter_runs, parse_deck
from synergy.observational import Pair

# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------


def select_pairs(
    obs_pairs: list[dict],
    top_k: int,
    sample_k: int,
    seed: int = 0,
) -> list[Pair]:
    """Select a pair subset spanning the observational PMI spectrum.

    Takes the ``top_k`` highest-PMI pairs (the *confirmed* candidates) plus a
    random ``sample_k`` drawn from the pairs at or below the median PMI (the
    *undiscovered* candidates).  Deduplicates while preserving order.

    Args:
        obs_pairs: ``pairs`` list from an ``obs_catalog`` document; each entry
            has ``"pair"`` and ``"pmi"`` keys.
        top_k: Number of top-PMI pairs to take.
        sample_k: Number of below-median pairs to sample.
        seed: RNG seed for the below-median sample.

    Returns:
        Ordered, de-duplicated list of canonical :data:`Pair` tuples.  When the
        corpus has fewer pairs than requested the result is simply shorter.
    """
    ranked = sorted(obs_pairs, key=lambda p: -float(p["pmi"]))
    head = ranked[:top_k]

    if ranked:
        mid = len(ranked) // 2
        below = ranked[mid:]
    else:
        below = []
    rng = random.Random(seed)
    tail = rng.sample(below, min(sample_k, len(below))) if below else []

    seen: set[Pair] = set()
    out: list[Pair] = []
    for entry in [*head, *tail]:
        pair: Pair = tuple(sorted(entry["pair"]))  # type: ignore[assignment]
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


# ---------------------------------------------------------------------------
# Baseline sampling (combat)
# ---------------------------------------------------------------------------


def sample_baselines(
    character: str,
    n: int,
    root: Path,
    min_cards: int = 14,
    max_cards: int = 18,
    seed: int = 0,
    wins_only: bool = True,
) -> list[list[str]]:
    """Sample *n* real, competitive decks to use as combat baselines.

    A bare 10-card starter deck dies in every encounter (``hp_lost`` floors at
    max HP → the four-cell second difference is identically zero).  Real
    mid-size decks instead *win with measurable HP loss*, giving the injected
    pair room to show a marginal effect.

    Args:
        character: Corpus character sub-directory (e.g. ``"silent"``).
        n: Number of baseline decks to return.
        root: Corpus root directory.
        min_cards: Minimum deck size to qualify (inclusive).
        max_cards: Maximum deck size to qualify (inclusive).
        seed: Reservoir-sampling RNG seed.
        wins_only: Keep only winning runs (more competitive decks).

    Returns:
        Up to *n* decks, each a list of engine card tokens.  Fewer than *n* if
        the corpus lacks enough qualifying decks.
    """
    rng = random.Random(seed)
    reservoir: list[list[str]] = []
    count = 0
    for run in iter_runs(root, character=character):
        if wins_only and not run.get("win"):
            continue
        deck = parse_deck(run)
        if not (min_cards <= len(deck) <= max_cards):
            continue
        count += 1
        if len(reservoir) < n:
            reservoir.append(deck)
        else:
            j = rng.randrange(count)
            if j < n:
                reservoir[j] = deck
    return reservoir


# ---------------------------------------------------------------------------
# Econ worker (process pool)
# ---------------------------------------------------------------------------

_ECON_CONTEXTS: list[State] = []
_ECON_NET: str = "gbm"
_ECON_CHARACTER: str = "ironclad"


def _econ_init(
    contexts: list[State],
    net_path: str | None,
    net: str,
    character: str,
) -> None:
    """Pool initializer: stash shared contexts + scorer backend selection.

    *net_path* pins ``STS2_RUNAI_VALUE`` for the GBM backend; *net*/*character*
    select the ``synergy_score`` backend (``gbm``/``ppo``) and character.
    """
    global _ECON_CONTEXTS, _ECON_NET, _ECON_CHARACTER
    _ECON_CONTEXTS = contexts
    _ECON_NET = net
    _ECON_CHARACTER = character
    if net_path:
        os.environ["STS2_RUNAI_VALUE"] = net_path


def _econ_one(pair: Pair) -> tuple[str, dict]:
    """Worker: econ second difference for one pair over the shared contexts."""
    row = econ_synergy(
        pair, _ECON_CONTEXTS, no_build=True, net=_ECON_NET, character=_ECON_CHARACTER
    )
    return (f"{pair[0]}|{pair[1]}", asdict(row))


def compute_econ_table(
    pairs: list[Pair],
    contexts: list[State],
    workers: int,
    net_path: str | None = None,
    *,
    net: str = "gbm",
    character: str = "ironclad",
) -> dict[str, dict]:
    """Compute the econ second difference for every pair, in parallel.

    Args:
        pairs: Canonical pairs to score.
        contexts: Shared mid-run deck contexts (sampled once by the caller).
        workers: Process-pool size.
        net_path: Optional value-net path; pinned into each worker's
            ``STS2_RUNAI_VALUE`` env (GBM backend only).
        net: Scorer backend passed to ``synergy_score`` — ``"gbm"`` (default,
            Ironclad-only GBM) or ``"ppo"`` (per-character PPO value head).
        character: Character for per-character base stats / PPO champion.

    Returns:
        Mapping ``"A|B" → {"median","p25","p75","n"}`` (the ``emit-catalog``
        econ-file format).
    """
    if not pairs:
        return {}
    out: dict[str, dict] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_econ_init,
        initargs=(contexts, net_path, net, character),
    ) as ex:
        for key, row in ex.map(_econ_one, pairs, chunksize=4):
            out[key] = row
    return out


# ---------------------------------------------------------------------------
# Combat worker (process pool)
# ---------------------------------------------------------------------------

_COMBAT_CFG: dict = {}


def _combat_init(
    character: str,
    baselines: list[list[str]],
    encounters: list[str],
    seeds: list[int],
    sims: int,
    depth: int,
) -> None:
    """Pool initializer: stash the shared combat panel config."""
    global _COMBAT_CFG
    _COMBAT_CFG = {
        "character": character,
        "baselines": baselines,
        "encounters": encounters,
        "seeds": seeds,
        "sims": sims,
        "depth": depth,
    }


def _combat_one(pair: Pair) -> tuple[str, dict | None]:
    """Worker: four-cell combat interaction for one pair over the panel.

    Returns ``(key, None)`` if any fight raises (e.g. an unported card factory)
    so the orchestrator can skip the pair rather than abort the whole run.
    """
    cfg = _COMBAT_CFG
    try:
        row: CombatRow = combat_synergy(
            pair,
            baseline_decks=cfg["baselines"],
            encounters=cfg["encounters"],
            seeds=cfg["seeds"],
            sims=cfg["sims"],
            depth=cfg["depth"],
            character=cfg["character"],
            no_build=True,
        )
    except Exception:
        return (f"{pair[0]}|{pair[1]}", None)
    return (
        f"{pair[0]}|{pair[1]}",
        {
            "median_interaction": row.median_interaction,
            "ci": [row.ci[0], row.ci[1]],
            "n": row.n,
        },
    )


def compute_combat_table(
    pairs: list[Pair],
    character: str,
    baselines: list[list[str]],
    encounters: list[str],
    seeds: list[int],
    sims: int,
    workers: int,
    depth: int = 50,
    log_every: int = 0,
    checkpoint_path: Path | None = None,
) -> dict[str, dict]:
    """Compute the four-cell combat interaction for every pair, in parallel.

    Pairs whose fights raise (unported card factory, etc.) are skipped.

    Args:
        pairs: Curated pair subset (see :func:`select_pairs`).
        character: Character tag forwarded to ``synergy_fight``.
        baselines: Competitive baseline decks (see :func:`sample_baselines`).
        encounters: Encounter-id panel.
        seeds: Engine seeds (shared across all four cells per RULE 4).
        sims: MCTS simulations per move.
        workers: Process-pool size.
        depth: Max rollout depth.
        log_every: If > 0, print a ``done/total`` progress line to stderr every
            *log_every* completed pairs (this is a long, opaque job — surfacing
            progress lets a watcher tell it apart from a hang).
        checkpoint_path: If set, write the partial result table to this path
            every *log_every* pairs (and once at the end), so a crash or kill
            mid-run does not discard all completed fights.

    Returns:
        Mapping ``"A|B" → {"median_interaction","ci","n"}`` (the ``emit-catalog``
        combat-file format), excluding skipped pairs.
    """
    if not pairs:
        return {}
    out: dict[str, dict] = {}
    total = len(pairs)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_combat_init,
        initargs=(character, baselines, encounters, seeds, sims, depth),
    ) as ex:
        for done, (key, row) in enumerate(ex.map(_combat_one, pairs, chunksize=1), 1):
            if row is not None:
                out[key] = row
            if log_every and (done % log_every == 0 or done == total):
                print(
                    f"[combat] {done}/{total} pairs done ({len(out)} scored)",
                    file=sys.stderr,
                    flush=True,
                )
                if checkpoint_path is not None:
                    checkpoint_path.write_text(json.dumps(out, indent=2))
    if checkpoint_path is not None:
        checkpoint_path.write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# Context sampling helper (econ)
# ---------------------------------------------------------------------------


def sample_econ_contexts(
    character: str,
    n: int,
    root: Path,
    seed: int = 0,
) -> list[State]:
    """Sample *n* mid-run econ contexts from a character's corpus decks.

    Wraps :func:`synergy.causal_econ.sample_contexts` with corpus decks loaded
    for *character* (rather than the env-default corpus root).

    Args:
        character: Corpus character sub-directory.
        n: Number of contexts.
        root: Corpus root directory.
        seed: RNG seed.

    Returns:
        A list of :class:`State` contexts.
    """
    decks = [
        Deck(cards=parse_deck(run), won=bool(run.get("win")), player_id=None, skill=None)
        for run in iter_runs(root, character=character)
    ]
    return sample_contexts(n, decks=decks, seed=seed)


# ---------------------------------------------------------------------------
# Run-level injection A/B (spec §8)
# ---------------------------------------------------------------------------


@dataclass
class InjectionResult:
    """Run-level injection A/B verdict (spec §8).

    Attributes:
        wr_delta: Paired win-rate difference ``WR(package) − WR(control)``.
        ci: Paired bootstrap 95% CI on ``wr_delta``.
        wr_pkg: Absolute win rate of the archetype-package arm.
        wr_ctrl: Absolute win rate of the matched-control arm.
        seeds: Number of paired seeds evaluated.
    """

    wr_delta: float
    ci: tuple[float, float]
    wr_pkg: float
    wr_ctrl: float
    seeds: int


def paired_wr_delta(
    wins_pkg: list[bool],
    wins_ctrl: list[bool],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """Paired (same-seed) win-rate delta with a paired bootstrap 95% CI.

    The two arms are paired by seed index (same seed list), so the bootstrap
    resamples seed *pairs* — the variance-reduced estimator the campaign uses.
    """
    n = min(len(wins_pkg), len(wins_ctrl))
    if n == 0:
        return 0.0, (0.0, 0.0)
    p = np.array([1.0 if w else 0.0 for w in wins_pkg[:n]])
    c = np.array([1.0 if w else 0.0 for w in wins_ctrl[:n]])
    diff = p - c
    delta = float(diff.mean())
    rng = np.random.default_rng(seed)
    boots = [float(diff[rng.integers(0, n, n)].mean()) for _ in range(n_boot)]
    lo, hi = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return delta, (lo, hi)


def injection_ab(
    package: list[str],
    control: list[str],
    seeds: list[int],
    *,
    run_arm: Callable[[list[str], list[int]], list[bool]],
    seed: int = 0,
) -> InjectionResult:
    """Run the package vs matched-control arms over paired seeds (spec §8).

    Args:
        package: The archetype card package to inject (weight-sampled upstream).
        control: The matched control package (embedding-proximity, §7).
        seeds: Paired engine seeds shared by both arms.
        run_arm: ``(package, seeds) -> per-seed wins`` runner — shells to
            ``run_arena`` in production (see :func:`run_arena_arm`); a mock in tests.
        seed: Bootstrap RNG seed.

    Returns:
        An :class:`InjectionResult`.
    """
    wins_pkg = run_arm(package, seeds)
    wins_ctrl = run_arm(control, seeds)
    delta, ci = paired_wr_delta(wins_pkg, wins_ctrl, seed=seed)
    n = min(len(wins_pkg), len(wins_ctrl))
    return InjectionResult(
        wr_delta=delta,
        ci=ci,
        wr_pkg=float(np.mean([1.0 if w else 0.0 for w in wins_pkg[:n]])) if n else 0.0,
        wr_ctrl=float(np.mean([1.0 if w else 0.0 for w in wins_ctrl[:n]])) if n else 0.0,
        seeds=n,
    )


# ---------------------------------------------------------------------------
# run_arena arm driver (the production `run_arm`)
# ---------------------------------------------------------------------------

_RUNAI_RS_ROOT = (Path(__file__).resolve().parent / ".." / ".." / ".." / "sts2-rs").resolve()
_RUN_ARENA_BINARY = _RUNAI_RS_ROOT / "target" / "release" / "run_arena"


def _parse_report_wins(report_path: Path, seeds: list[int]) -> list[bool]:
    """Read a ``run_arena`` report.jsonl and return per-seed wins in *seeds* order.

    Missing seeds (e.g. a timed-out cell) count as a loss so the paired arrays
    stay aligned.
    """
    won_by_seed: dict[int, bool] = {}
    for line in report_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "seed" in row and "won" in row:
            won_by_seed[int(row["seed"])] = bool(row["won"])
    return [won_by_seed.get(s, False) for s in seeds]


def run_arena_arm(
    package: list[str],
    seeds: list[int],
    *,
    character: str = "ironclad",
    ascension: int = 0,
    policy: str = "ppo",
    champion: str | None = None,
    no_build: bool = False,
) -> list[bool]:
    """Production ``run_arm``: sweep *seeds* with the injected *package* via run_arena.

    Sets ``STS2_RUNAI_INJECT_CARDS`` to the package (empty → the no-op / control
    baseline arm), runs the deployed *policy*, and parses per-seed wins.  Seeds
    must be a contiguous range (``run_arena``'s ad-hoc ``--seeds`` mode is
    ``seed_start .. seed_start+N``); non-contiguous panels need a panel file.
    """
    if not seeds:
        return []
    if character != "ironclad":
        # run_arena's ad-hoc --seeds range is Ironclad-A0 only; other characters
        # need a seed-panel file (out of scope for this ad-hoc driver).
        raise NotImplementedError("run_arena_arm ad-hoc seeds support ironclad only")
    binary = _ensure_run_arena_binary(no_build=no_build)
    seed_start, count = min(seeds), len(seeds)
    if sorted(seeds) != list(range(seed_start, seed_start + count)):
        raise ValueError("run_arena_arm requires a contiguous seed range")

    env = dict(os.environ)
    env["STS2_RUNAI_INJECT_CARDS"] = ",".join(package)
    if champion:
        env["STS2_RUNAI_PPO"] = champion

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "report.jsonl"
        cmd = [
            str(binary),
            "sweep",
            "--seeds",
            str(count),
            "--seed-start",
            str(seed_start),
            "--policy",
            policy,
            "--ascension",
            str(ascension),
            "--out",
            str(out),
        ]
        subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
        return _parse_report_wins(out, list(range(seed_start, seed_start + count)))


def _ensure_run_arena_binary(no_build: bool = False) -> Path:
    """Return the run_arena binary path, building it if missing (unless *no_build*)."""
    if _RUN_ARENA_BINARY.exists():
        return _RUN_ARENA_BINARY
    if no_build:
        raise RuntimeError(
            f"run_arena binary missing ({_RUN_ARENA_BINARY}); build with "
            "`cargo build --release -p sts2-cli --bin run_arena`."
        )
    print(f"[run_arena] building {_RUN_ARENA_BINARY} …", file=sys.stderr)
    subprocess.run(
        ["cargo", "build", "--release", "-p", "sts2-cli", "--bin", "run_arena"],
        cwd=_RUNAI_RS_ROOT,
        check=True,
    )
    return _RUN_ARENA_BINARY
