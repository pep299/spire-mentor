"""CLI entry-point for the card-synergy catalog pipeline.

Usage::

    python -m synergy.cli emit-observational [options]
    python -m synergy.cli emit-catalog [options]

Subcommands
-----------
emit-observational
    Run the full observational pipeline (PMI + wins-minus-losses) on the
    spire-codex corpus and write an ``obs_catalog.json`` file.

emit-catalog
    Join an existing ``obs_catalog.json`` with optional econ/combat score
    files and write the final ``catalog.json``.

Options (emit-observational)
----------------------------
--root PATH
    Corpus root directory (default: ``CODEX_ROOT`` env / hard-coded default).
--build-ids v0.103.2,v0.103.3
    Comma-separated accepted build ids.
--freq-floor INT
    Minimum pair co-occurrence to include (default: 50).
--skill-bucket {high,}
    Restrict to top-quartile players (``high``), or all players (omit).
--out PATH
    Output file path (default: ``obs_catalog.json``).
--character ironclad
    Character sub-directory (default: ``ironclad``).

Options (emit-catalog)
----------------------
--obs PATH
    Input obs_catalog.json from emit-observational (default: ``obs_catalog.json``).
--econ PATH
    Optional econ scores file (JSON, keyed by "A|B" pair strings).
    Format: ``{"A|B": {"median": 0.5, "p25": 0.3, "p75": 0.7, "n": 200}}``.
    Omit or pass an empty/nonexistent file to leave econ_S null.
--combat PATH
    Optional combat scores file (JSON, keyed by "A|B" pair strings).
    Format: ``{"A|B": {"median_interaction": 1.2, "ci": [0.4, 2.1], "n": 90}}``.
    Omit or pass an empty/nonexistent file to leave combat_S null.
--out PATH
    Output file path (default: ``catalog.json``).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from synergy.card_filter import NON_DECKBUILDABLE
from synergy.catalog import Thresholds, build_catalog
from synergy.causal_combat import CombatRow
from synergy.causal_econ import EconRow
from synergy.corpus import CODEX_ROOT, load_decks
from synergy.multiplicity import multiplicity_layer
from synergy.observational import (
    InteractionRow,
    ObsRow,
    Pair,
    fm_interactions,
    interaction_lr,
    pmi_table,
)


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list (q in [0,1])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo])


# ---------------------------------------------------------------------------
# emit-observational implementation
# ---------------------------------------------------------------------------


def _emit_observational(args: argparse.Namespace) -> None:
    """Run the full observational pipeline and write obs_catalog.json."""
    root = Path(args.root)
    build_ids = tuple(b.strip() for b in args.build_ids.split(",") if b.strip())
    freq_floor: int = args.freq_floor
    skill_bucket: str | None = args.skill_bucket or None
    out_path = Path(args.out)
    character: str = args.character

    # Materialize the corpus once — pmi_table, interaction_lr and the FM all
    # consume the same list.
    print(f"Loading decks from {root} …", file=sys.stderr)
    decks = list(
        load_decks(
            root=root,
            skill_bucket=skill_bucket,
            build_ids=build_ids,
            character=character,
        )
    )
    run_count = len(decks)
    print(f"Loaded {run_count} decks.", file=sys.stderr)

    # Per-card frequency (fraction of decks containing each card, presence-based).
    # Non-deckbuildable tokens (curses / statuses / combat tokens / quest cards)
    # are excluded to match the filtered pair universe (see card_filter).
    card_present: dict[str, int] = defaultdict(int)
    for deck in decks:
        for card in set(deck.cards) - NON_DECKBUILDABLE:
            card_present[card] += 1
    card_frequency: dict[str, float] = (
        {card: count / run_count for card, count in card_present.items()} if run_count > 0 else {}
    )

    # PMI (human-intent axis), interaction_lr (controlled candidate signal, the
    # W-L replacement) and the factorization machine — all over the same list.
    print("Computing PMI table …", file=sys.stderr)
    obs = pmi_table(decks, freq_floor=freq_floor)

    print(f"Computing interaction_lr (n_boot={args.n_boot}) …", file=sys.stderr)
    inter = interaction_lr(decks, freq_floor=freq_floor, n_boot=args.n_boot)

    fm_map: dict[Pair, float] = {}
    if args.fm:
        print(f"Fitting factorization machine (dim={args.fm_dim}) …", file=sys.stderr)
        fm_map = fm_interactions(
            decks, freq_floor=freq_floor, dim=args.fm_dim, epochs=args.fm_epochs
        ).interactions

    # Build output pairs, sorted by PMI descending.
    pairs_out = []
    for pair in sorted(obs, key=lambda p: -obs[p].pmi):
        row = obs[pair]
        inter_row = inter.get(pair)
        pairs_out.append(
            {
                "pair": list(pair),
                "pmi": row.pmi,
                "interaction_lr": inter_row.beta_ab if inter_row is not None else None,
                "interaction_ci": list(inter_row.ci) if inter_row is not None else None,
                "interaction_n_both": inter_row.n_both if inter_row is not None else None,
                "interaction_n": inter_row.n if inter_row is not None else None,
                "fm_S": fm_map.get(pair),
                "cooccur": row.cooccur,
                "p_a": row.p_a,
                "p_b": row.p_b,
                "p_ab": row.p_ab,
            }
        )

    catalog = {
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "character": character,
        "run_count": run_count,
        "freq_floor": freq_floor,
        "skill_bucket": skill_bucket,
        "build_ids": list(build_ids),
        "card_frequency": card_frequency,
        "pairs": pairs_out,
    }

    out_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Wrote {len(pairs_out)} pairs to {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# emit-catalog implementation
# ---------------------------------------------------------------------------


def _load_econ_file(path: str) -> dict[Pair, EconRow] | None:
    """Load an optional econ scores JSON file.

    File format: JSON object keyed by ``"A|B"`` pair strings::

        {"A|B": {"median": 0.5, "p25": 0.3, "p75": 0.7, "n": 200}, ...}

    Args:
        path: File path string.  Empty string or non-existent file → ``None``.

    Returns:
        Mapping from canonical :data:`~synergy.observational.Pair` to
        :class:`~synergy.causal_econ.EconRow`, or ``None`` when the path is
        absent/empty.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    raw: dict = json.loads(p.read_text(encoding="utf-8"))
    if not raw:
        return None
    result: dict[Pair, EconRow] = {}
    for key, val in raw.items():
        a, b = key.split("|", 1)
        pair: Pair = tuple(sorted((a, b)))  # type: ignore[assignment]
        result[pair] = EconRow(
            median=float(val["median"]),
            p25=float(val["p25"]),
            p75=float(val["p75"]),
            n=int(val["n"]),
        )
    return result or None


def _load_combat_file(path: str) -> dict[Pair, CombatRow] | None:
    """Load an optional combat scores JSON file.

    File format: JSON object keyed by ``"A|B"`` pair strings::

        {"A|B": {"median_interaction": 1.2, "ci": [0.4, 2.1], "n": 90}, ...}

    Args:
        path: File path string.  Empty string or non-existent file → ``None``.

    Returns:
        Mapping from canonical :data:`~synergy.observational.Pair` to
        :class:`~synergy.causal_combat.CombatRow`, or ``None`` when the path is
        absent/empty.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    raw: dict = json.loads(p.read_text(encoding="utf-8"))
    if not raw:
        return None
    result: dict[Pair, CombatRow] = {}
    for key, val in raw.items():
        a, b = key.split("|", 1)
        pair: Pair = tuple(sorted((a, b)))  # type: ignore[assignment]
        ci_raw = val["ci"]
        result[pair] = CombatRow(
            median_interaction=float(val["median_interaction"]),
            ci=(float(ci_raw[0]), float(ci_raw[1])),
            n=int(val["n"]),
        )
    return result or None


def _emit_catalog(args: argparse.Namespace) -> None:
    """Join obs_catalog.json with optional econ/combat files and write catalog.json."""
    obs_path = Path(args.obs)
    out_path = Path(args.out)

    print(f"Reading obs catalog from {obs_path} …", file=sys.stderr)
    raw = json.loads(obs_path.read_text(encoding="utf-8"))

    # Reconstruct obs, interaction and FM dicts from the obs_catalog pairs list.
    obs: dict[Pair, ObsRow] = {}
    interaction: dict[Pair, InteractionRow] = {}
    fm: dict[Pair, float] = {}
    for entry in raw.get("pairs", []):
        pair: Pair = tuple(sorted(entry["pair"]))  # type: ignore[assignment]
        obs[pair] = ObsRow(
            pmi=float(entry["pmi"]),
            cooccur=int(entry["cooccur"]),
            p_a=float(entry["p_a"]),
            p_b=float(entry["p_b"]),
            p_ab=float(entry["p_ab"]),
        )
        beta = entry.get("interaction_lr")
        if beta is not None:
            ci_raw = entry.get("interaction_ci") or [float("nan"), float("nan")]
            interaction[pair] = InteractionRow(
                beta_ab=float(beta),
                ci=(float(ci_raw[0]), float(ci_raw[1])),
                n_both=int(entry.get("interaction_n_both") or 0),
                n=int(entry.get("interaction_n") or 0),
            )
        fm_val = entry.get("fm_S")
        if fm_val is not None:
            fm[pair] = float(fm_val)

    card_frequency: dict[str, float] = {
        k: float(v) for k, v in raw.get("card_frequency", {}).items()
    }
    run_count: int = int(raw.get("run_count", 0))
    character: str = str(raw.get("character", "Ironclad")).title()

    econ = _load_econ_file(args.econ or "")
    combat = _load_combat_file(args.combat or "")

    if econ is not None:
        print(f"Loaded econ scores for {len(econ)} pairs.", file=sys.stderr)
    if combat is not None:
        print(f"Loaded combat scores for {len(combat)} pairs.", file=sys.stderr)

    # Econ second differences are a noisy signal centred near zero — a bare
    # median>0 gate flags ~half of all pairs causal-high, ballooning the
    # `undiscovered` quadrant into thousands of weak pairs. Gate on a data-driven
    # quantile of the econ distribution so only *strongly* positive pairs count.
    econ_cut = 0.0
    if econ is not None and 0.0 < args.econ_quantile < 1.0:
        medians = sorted(row.median for row in econ.values())
        if len(medians) >= 2:
            econ_cut = max(0.0, _quantile(medians, args.econ_quantile))
            print(
                f"Econ causal-high gate: median > {econ_cut:.5f} "
                f"(quantile {args.econ_quantile}); "
                f"{sum(1 for m in medians if m > econ_cut)} pairs pass.",
                file=sys.stderr,
            )

    # Combat is run only on a curated pair subset, so its median distribution is
    # already selective (most pairs sit at/below 0). The default gate is
    # median>0; --combat-quantile tightens it further to the top of the combat
    # effect-size distribution.
    combat_cut = 0.0
    if combat is not None and 0.0 < args.combat_quantile < 1.0:
        cmeds = sorted(row.median_interaction for row in combat.values())
        if len(cmeds) >= 2:
            combat_cut = max(0.0, _quantile(cmeds, args.combat_quantile))
            print(
                f"Combat causal-high gate: median > {combat_cut:.4f} "
                f"(quantile {args.combat_quantile}); "
                f"{sum(1 for m in cmeds if m > combat_cut)} pairs pass.",
                file=sys.stderr,
            )

    thresholds = Thresholds(
        econ_median_min=econ_cut, combat_median_min=combat_cut, q_max=args.q_max
    )

    # Multiplicity control (§9b): EB shrinkage + permutation p + BH-FDR q.  Needs
    # the corpus decks (for the permutation null) and re-uses the same
    # interaction_lr statistic as a point estimator (no bootstrap → cheap).
    multiplicity = None
    if args.multiplicity:
        build_ids = tuple(b.strip() for b in args.build_ids.split(",") if b.strip())
        print(f"Loading decks for multiplicity null from {args.root} …", file=sys.stderr)
        decks = list(
            load_decks(
                root=Path(args.root),
                skill_bucket=args.skill_bucket or None,
                build_ids=build_ids,
                character=args.character,
            )
        )
        freq_floor = args.freq_floor

        def _point_estimator(ds: list) -> dict[Pair, float]:
            return {
                p: r.beta_ab for p, r in interaction_lr(ds, freq_floor=freq_floor, n_boot=0).items()
            }

        print(
            f"Computing multiplicity layer (n_perm={args.n_perm}) over {len(decks)} decks …",
            file=sys.stderr,
        )
        multiplicity = multiplicity_layer(interaction, decks, _point_estimator, n_perm=args.n_perm)
        n_sig = sum(1 for r in multiplicity.values() if r.qvalue <= args.q_max)
        print(
            f"Multiplicity: {n_sig}/{len(multiplicity)} pairs pass BH-FDR q<={args.q_max}.",
            file=sys.stderr,
        )

    catalog = build_catalog(
        obs=obs,
        interaction=interaction,
        fm=fm or None,
        multiplicity=multiplicity,
        econ=econ,
        combat=combat,
        card_frequency=card_frequency,
        run_count=run_count,
        character=character,
        thresholds=thresholds,
    )

    out_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    n_edges = len(catalog["edges"])
    n_nodes = len(catalog["nodes"])
    print(f"Wrote catalog.json: {n_nodes} nodes, {n_edges} edges → {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# emit-geometry implementation (§5 View 3)
# ---------------------------------------------------------------------------


def _emit_geometry(args: argparse.Namespace) -> None:
    """Build card_embed_geometry_<char>.json from the champion embedding table."""
    from synergy.embed_geometry import build_geometry

    obs_path = Path(args.obs)
    print(f"Reading card frequencies from {obs_path} …", file=sys.stderr)
    raw = json.loads(obs_path.read_text(encoding="utf-8"))
    card_frequency: dict[str, float] = {
        k: float(v) for k, v in raw.get("card_frequency", {}).items()
    }

    print(f"Building embedding geometry for {args.character} …", file=sys.stderr)
    doc = build_geometry(
        args.character,
        card_frequency,
        k=args.k,
        min_freq=args.min_freq,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    n_clusters = len(set(doc["clusters"].values()))
    print(
        f"Wrote {out_path}: {doc['n_cards']} cards, {n_clusters} clusters (k={doc['k']}).",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# emit-archetypes implementation (§5 discovery)
# ---------------------------------------------------------------------------


def _emit_archetypes(args: argparse.Namespace) -> None:
    """Discover + tag archetypes from a corpus + catalog and write JSON."""
    from dataclasses import asdict

    from synergy.archetype_discover import discover_archetypes

    catalog_path = Path(args.catalog)
    print(f"Reading catalog edges from {catalog_path} …", file=sys.stderr)
    edges = json.loads(catalog_path.read_text(encoding="utf-8")).get("edges", [])

    build_ids = tuple(b.strip() for b in args.build_ids.split(",") if b.strip())
    print(f"Loading decks from {args.root} …", file=sys.stderr)
    decks = list(
        load_decks(
            root=Path(args.root),
            skill_bucket=args.skill_bucket or None,
            build_ids=build_ids,
            character=args.character,
        )
    )
    card_text = None
    if args.card_text:
        card_text = json.loads(Path(args.card_text).read_text(encoding="utf-8"))
    print(f"Discovering archetypes over {len(decks)} decks + {len(edges)} edges …", file=sys.stderr)
    candidates = discover_archetypes(
        decks,
        edges,
        n_components=args.n_components,
        top_k=args.top_k,
        signal_key=args.signal_key,
        qvalue_max=args.qvalue_max,
        jaccard_min=args.jaccard_min,
        resolution=args.resolution,
        n_restarts=args.n_restarts,
        stability_min=args.stability_min,
        card_text=card_text,
        seed=args.seed,
    )

    doc = {"character": args.character, "candidates": [asdict(c) for c in candidates]}
    out_path = Path(args.out)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    by_tag: dict[str, int] = defaultdict(int)
    for c in candidates:
        by_tag[c.tag] += 1
    print(
        f"Wrote {out_path}: {len(candidates)} candidates ({dict(by_tag)}).",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# emit-dose implementation (§7 dose-response panels)
# ---------------------------------------------------------------------------


def _emit_dose(args: argparse.Namespace) -> None:
    """Run archetype dose-response panels over the champion geometry and write JSON."""
    from dataclasses import asdict

    from synergy.causal_combat import DEFAULT_ENCOUNTERS, run_fight
    from synergy.causal_dose import dose_archetype
    from synergy.embed_geometry import champion_onnx_path, load_card_embeddings

    archetypes = json.loads(Path(args.archetypes).read_text(encoding="utf-8")).get("candidates", [])
    clusters: dict[str, int] = json.loads(Path(args.geometry).read_text(encoding="utf-8")).get(
        "clusters", {}
    )
    print(f"Loading champion embeddings for {args.character} …", file=sys.stderr)
    emb = load_card_embeddings(champion_onnx_path(args.character))

    encounters = args.encounters.split(",") if args.encounters else DEFAULT_ENCOUNTERS
    seeds = list(range(args.seeds))

    def fight_fn(base: list[str], inject: list[str], enc: str, seed: int) -> dict:
        return run_fight(
            base,
            inject,
            enc,
            seed,
            sims=args.sims,
            no_build=args.no_build,
            character=args.character,
        )

    doses = []
    for cand in archetypes:
        cards = cand["cards"]
        print(f"Dose-response for {cand['tag']} {cards[:4]}… …", file=sys.stderr)
        res = dose_archetype(
            fight_fn,
            cards,
            emb,
            clusters,
            [[]],  # per-character starter base (run_fight resolves [] → starter)
            encounters,
            seeds,
            K=args.k,
            target_k=args.target_k,
            n_draws=args.n_draws,
            n_null=args.n_null,
        )
        doses.append({"cards": cards, "tag": cand["tag"], **asdict(res)})

    doc = {"character": args.character, "doses": doses}
    Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}: {len(doses)} archetype dose panels.", file=sys.stderr)


# ---------------------------------------------------------------------------
# emit-injection implementation (§8 run-level A/B)
# ---------------------------------------------------------------------------


def _emit_injection(args: argparse.Namespace) -> None:
    """Run archetype-package injection A/B vs matched controls and write JSON."""
    from dataclasses import asdict

    from synergy import causal_run
    from synergy.causal_run import injection_ab
    from synergy.embed_geometry import champion_onnx_path, control_match, load_card_embeddings

    archetypes = json.loads(Path(args.archetypes).read_text(encoding="utf-8")).get("candidates", [])
    clusters: dict[str, int] = json.loads(Path(args.geometry).read_text(encoding="utf-8")).get(
        "clusters", {}
    )
    champion = str(champion_onnx_path(args.character))
    print(f"Loading champion embeddings for {args.character} …", file=sys.stderr)
    emb = load_card_embeddings(champion_onnx_path(args.character))

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    def run_arm(pkg: list[str], sds: list[int]) -> list[bool]:
        return causal_run.run_arena_arm(
            pkg, sds, character=args.character, champion=champion, no_build=args.no_build
        )

    injections = []
    for cand in archetypes[: args.top]:
        cards = cand["cards"][: args.k]
        control = control_match(cards, emb, clusters, len(cards))
        print(f"Injection A/B for {cand['tag']} {cards} …", file=sys.stderr)
        res = injection_ab(cards, control, seeds, run_arm=run_arm)
        injections.append({"id": cand.get("id"), "cards": cards, "tag": cand["tag"], **asdict(res)})

    doc = {"character": args.character, "injections": injections}
    Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}: {len(injections)} injection A/B verdicts.", file=sys.stderr)


# ---------------------------------------------------------------------------
# emit-archetype-catalog implementation (§9/§10 UI assembly)
# ---------------------------------------------------------------------------


def _emit_archetype_catalog(args: argparse.Namespace) -> None:
    """Merge archetypes + dose curves into the UI archetypes_<char>.json file."""
    from synergy.archetype_catalog import assemble_archetypes

    archetypes_doc = json.loads(Path(args.archetypes).read_text(encoding="utf-8"))
    dose_doc = json.loads(Path(args.dose).read_text(encoding="utf-8")) if args.dose else None
    doc = assemble_archetypes(archetypes_doc, dose_doc)
    Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(
        f"Wrote {args.out}: {len(doc['archetypes'])} archetypes for {doc['character']}.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m synergy.cli",
        description="Card synergy catalog pipeline tools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    eo = sub.add_parser(
        "emit-observational",
        help="Run PMI + interaction_lr (+ optional FM) pipeline and write obs_catalog.json.",
    )
    eo.add_argument(
        "--root",
        default=str(CODEX_ROOT),
        help="Corpus root directory (default: CODEX_ROOT).",
    )
    eo.add_argument(
        "--build-ids",
        default="v0.107.1",
        help="Comma-separated accepted build ids (default: v0.107.1).",
    )
    eo.add_argument(
        "--n-boot",
        type=int,
        default=200,
        help="Bootstrap resamples for interaction_lr CIs (default: 200).",
    )
    eo.add_argument(
        "--fm",
        action="store_true",
        help="Also fit the factorization machine and emit fm_S per pair.",
    )
    eo.add_argument("--fm-dim", type=int, default=16, help="FM embedding dimension.")
    eo.add_argument("--fm-epochs", type=int, default=60, help="FM training epochs.")
    eo.add_argument(
        "--freq-floor",
        type=int,
        default=50,
        help="Minimum co-occurrence count to include a pair (default: 50).",
    )
    eo.add_argument(
        "--skill-bucket",
        default=None,
        choices=["high"],
        help="Restrict to top-quartile players ('high'), or all players (omit).",
    )
    eo.add_argument(
        "--out",
        default="obs_catalog.json",
        help="Output file path (default: obs_catalog.json).",
    )
    eo.add_argument(
        "--character",
        default="ironclad",
        help="Character sub-directory name (default: ironclad).",
    )

    ec = sub.add_parser(
        "emit-catalog",
        help="Join obs_catalog.json with optional econ/combat files → catalog.json.",
    )
    ec.add_argument(
        "--obs",
        default="obs_catalog.json",
        help="Input obs_catalog.json (from emit-observational). Default: obs_catalog.json.",
    )
    ec.add_argument(
        "--econ",
        default="",
        help=(
            "Optional econ scores file (JSON keyed by 'A|B' pair strings). "
            "Omit to leave econ_S null."
        ),
    )
    ec.add_argument(
        "--combat",
        default="",
        help=(
            "Optional combat scores file (JSON keyed by 'A|B' pair strings). "
            "Omit to leave combat_S null."
        ),
    )
    ec.add_argument(
        "--out",
        default="catalog.json",
        help="Output file path (default: catalog.json).",
    )
    ec.add_argument(
        "--econ-quantile",
        type=float,
        default=0.9,
        help="Econ causal-high gate: a pair counts as causal-high (econ fallback) "
        "only if its econ median exceeds this quantile of the econ distribution "
        "(default 0.9 → top 10%%). Keeps 'undiscovered' selective. Set 0 to "
        "disable (bare median>0 gate).",
    )
    ec.add_argument(
        "--combat-quantile",
        type=float,
        default=0.0,
        help="Combat causal-high gate: tighten beyond the default median>0 to the "
        "given quantile of the combat effect-size distribution. Default 0 "
        "(median>0; combat is already run on a curated, selective pair subset).",
    )
    ec.add_argument(
        "--multiplicity",
        action="store_true",
        help="Run the §9b multiplicity layer (EB shrinkage + permutation p + "
        "BH-FDR q) and emit shrunk_S/pvalue/qvalue per edge, FDR-gating the "
        "'undiscovered' quadrant. Requires --root/--build-ids/--character to "
        "load the corpus for the permutation null.",
    )
    ec.add_argument(
        "--n-perm",
        type=int,
        default=200,
        help="Permutation replicates for the multiplicity null (default: 200).",
    )
    ec.add_argument(
        "--q-max",
        type=float,
        default=0.1,
        help="BH-FDR target for the 'undiscovered' gate (default: 0.1).",
    )
    ec.add_argument(
        "--root",
        default=str(CODEX_ROOT),
        help="Corpus root for the multiplicity null (default: CODEX_ROOT).",
    )
    ec.add_argument(
        "--build-ids",
        default="v0.107.1",
        help="Comma-separated accepted build ids for the null (default: v0.107.1).",
    )
    ec.add_argument(
        "--character",
        default="ironclad",
        help="Character sub-directory for the null corpus (default: ironclad).",
    )
    ec.add_argument(
        "--skill-bucket",
        default=None,
        choices=["high"],
        help="Restrict the null corpus to top-quartile players ('high').",
    )
    ec.add_argument(
        "--freq-floor",
        type=int,
        default=50,
        help="Min co-occurrence for the multiplicity point estimator (default: 50).",
    )

    eg = sub.add_parser(
        "emit-geometry",
        help="Build card_embed_geometry_<char>.json from the PPO champion embedding table (§5 View 3).",
    )
    eg.add_argument(
        "--character",
        default="ironclad",
        help="Character whose PPO champion to read (ironclad/silent/defect).",
    )
    eg.add_argument(
        "--obs",
        default="obs_catalog.json",
        help="obs_catalog.json supplying card_frequency for the gate (default: obs_catalog.json).",
    )
    eg.add_argument(
        "--out",
        default="card_embed_geometry.json",
        help="Output file path (default: card_embed_geometry.json).",
    )
    eg.add_argument("--k", type=int, default=15, help="kNN neighbours per card (default: 15).")
    eg.add_argument(
        "--min-freq",
        type=float,
        default=0.01,
        help="Frequency floor: drop undertrained rare-card rows (default: 0.01).",
    )
    eg.add_argument("--seed", type=int, default=0, help="Louvain seed (default: 0).")

    ea = sub.add_parser(
        "emit-archetypes",
        help="Discover + tag archetypes (NMF View 1 + causal communities View 2 + merge, §5).",
    )
    ea.add_argument(
        "--catalog",
        default="catalog.json",
        help="catalog.json supplying causal edges for View 2 (default: catalog.json).",
    )
    ea.add_argument(
        "--root",
        default=str(CODEX_ROOT),
        help="Corpus root for the NMF deck matrix (default: CODEX_ROOT).",
    )
    ea.add_argument(
        "--build-ids",
        default="v0.107.1",
        help="Comma-separated accepted build ids (default: v0.107.1).",
    )
    ea.add_argument(
        "--character",
        default="ironclad",
        help="Character sub-directory / document tag (default: ironclad).",
    )
    ea.add_argument(
        "--skill-bucket",
        default=None,
        choices=["high"],
        help="Restrict NMF corpus to top-quartile players ('high').",
    )
    ea.add_argument("--n-components", type=int, default=8, help="NMF component count (default: 8).")
    ea.add_argument("--top-k", type=int, default=10, help="Top cards per component (default: 10).")
    ea.add_argument(
        "--signal-key",
        default="pmi",
        help="Catalog edge key for the View-2 co-play graph (default: pmi — the "
        "observational co-play strength that defines an archetype; combat_S is "
        "too noisy per-pair to cluster on and is used only to validate).",
    )
    ea.add_argument(
        "--resolution",
        type=float,
        default=2.0,
        help="Louvain resolution for View-2 communities; >1 → tighter, "
        "single-archetype clusters (default: 2.0).",
    )
    ea.add_argument(
        "--card-text",
        default=None,
        help="Optional card_text.json (token→{title,description}) for mechanic "
        "labelling of discovered clusters.",
    )
    ea.add_argument(
        "--qvalue-max",
        type=float,
        default=None,
        help="Optional §9b FDR gate on View-2 edges (default: none).",
    )
    ea.add_argument(
        "--jaccard-min",
        type=float,
        default=0.3,
        help="Weighted-Jaccard threshold for NMF↔community match (default: 0.3).",
    )
    ea.add_argument(
        "--n-restarts", type=int, default=10, help="NMF stability restarts (default: 10)."
    )
    ea.add_argument(
        "--stability-min",
        type=float,
        default=0.6,
        help="Min across-restart top-k Jaccard to keep a component (default: 0.6).",
    )
    ea.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    ea.add_argument(
        "--out",
        default="archetypes.json",
        help="Output file path (default: archetypes.json).",
    )

    ed = sub.add_parser(
        "emit-dose",
        help="Run archetype combat dose-response panels over the champion geometry (§7).",
    )
    ed.add_argument(
        "--archetypes",
        default="archetypes.json",
        help="archetypes.json (from emit-archetypes). Default: archetypes.json.",
    )
    ed.add_argument(
        "--geometry",
        default="card_embed_geometry.json",
        help="card_embed_geometry.json (from emit-geometry) supplying clusters. "
        "Default: card_embed_geometry.json.",
    )
    ed.add_argument(
        "--character",
        default="ironclad",
        help="Character (champion + starter base + fight class). Default: ironclad.",
    )
    ed.add_argument("--k", type=int, default=5, help="Max dose level K (default: 5).")
    ed.add_argument("--target-k", type=int, default=3, help="Verdict dose level (default: 3).")
    ed.add_argument(
        "--n-draws",
        type=int,
        default=4,
        help="Weight-sampled packages per dose level (default: 4).",
    )
    ed.add_argument(
        "--n-null",
        type=int,
        default=50,
        help="Random strength-matched null packages (default: 50).",
    )
    ed.add_argument("--sims", type=int, default=1500, help="MCTS sims per move (default: 1500).")
    ed.add_argument("--seeds", type=int, default=6, help="Seed count per arm (default: 6).")
    ed.add_argument(
        "--encounters",
        default="",
        help="Comma-separated encounter ids (default: the built-in 3-encounter panel).",
    )
    ed.add_argument(
        "--no-build",
        action="store_true",
        help="Fail if the synergy_fight binary is stale/missing instead of rebuilding.",
    )
    ed.add_argument(
        "--out",
        default="dose.json",
        help="Output file path (default: dose.json).",
    )

    ei = sub.add_parser(
        "emit-injection",
        help="Run-level card-injection A/B for top archetypes vs matched controls (§8).",
    )
    ei.add_argument(
        "--archetypes",
        default="archetypes.json",
        help="archetypes.json (from emit-archetypes). Default: archetypes.json.",
    )
    ei.add_argument(
        "--geometry",
        default="card_embed_geometry.json",
        help="card_embed_geometry.json (clusters) for the control matcher. "
        "Default: card_embed_geometry.json.",
    )
    ei.add_argument(
        "--character",
        default="ironclad",
        help="Character (champion + arena runs; ad-hoc seeds are ironclad-only). "
        "Default: ironclad.",
    )
    ei.add_argument("--top", type=int, default=5, help="Top archetypes to test (default: 5).")
    ei.add_argument("--k", type=int, default=3, help="Package size to inject (default: 3).")
    ei.add_argument("--seeds", type=int, default=500, help="Paired seeds per arm (default: 500).")
    ei.add_argument("--seed-start", type=int, default=0, help="First seed (default: 0).")
    ei.add_argument(
        "--no-build",
        action="store_true",
        help="Fail if the run_arena binary is missing instead of building it.",
    )
    ei.add_argument(
        "--out",
        default="injection.json",
        help="Output file path (default: injection.json).",
    )

    ac = sub.add_parser(
        "emit-archetype-catalog",
        help="Merge archetypes + dose curves into the UI archetypes_<char>.json (§9/§10).",
    )
    ac.add_argument(
        "--archetypes",
        default="archetypes.json",
        help="archetypes.json (from emit-archetypes). Default: archetypes.json.",
    )
    ac.add_argument(
        "--dose",
        default="",
        help="Optional dose.json (from emit-dose) supplying the curves. Omit for no curves.",
    )
    ac.add_argument(
        "--out",
        default="archetypes_ui.json",
        help="Output file path (default: archetypes_ui.json).",
    )

    emb = sub.add_parser(
        "embed-catalog",
        help="Bake UMAP x/y/hue into catalog.json using a dense obs_catalog as the PMI source.",
    )
    emb.add_argument(
        "--obs",
        default="obs_catalog.json",
        help="Dense obs_catalog.json (freq-floor-50) used to build the PMI matrix.",
    )
    emb.add_argument(
        "--catalog",
        default="catalog.json",
        help="Catalog.json to augment with x/y/hue (default: catalog.json).",
    )
    emb.add_argument(
        "--out",
        default="catalog.json",
        help="Output path (default: overwrite catalog.json in place).",
    )
    emb.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors.")
    emb.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")

    ord_ = sub.add_parser(
        "order-catalog",
        help=(
            "Bake hierarchical-clustering card_order into catalog.json. "
            "Uses the dense obs_catalog (freq-floor-50) to build the PMI matrix, "
            "restricts to signal cards (those with ≥1 edge in the display catalog), "
            "and writes a card_order list + signal flag per node."
        ),
    )
    ord_.add_argument(
        "--obs",
        default="obs_catalog.json",
        help="Dense obs_catalog.json (freq-floor-50) used to build the PMI matrix.",
    )
    ord_.add_argument(
        "--catalog",
        default="catalog.json",
        help="Catalog.json to augment with card_order (default: catalog.json).",
    )
    ord_.add_argument(
        "--out",
        default="catalog.json",
        help="Output path (default: overwrite catalog.json in place).",
    )

    # --- emit-econ -----------------------------------------------------------
    en = sub.add_parser(
        "emit-econ",
        help="Compute the value-net second-difference econ score table (JSON).",
    )
    en.add_argument("--obs", default="obs_catalog.json", help="Input obs_catalog.json.")
    en.add_argument("--out", default="econ.json", help="Output econ scores file.")
    en.add_argument("--character", default="ironclad", help="Character (context source).")
    en.add_argument("--root", default=str(CODEX_ROOT), help="Corpus root directory.")
    en.add_argument("--n-contexts", type=int, default=200, help="Mid-run contexts to sample.")
    en.add_argument("--workers", type=int, default=_default_workers(), help="Process-pool size.")
    en.add_argument("--seed", type=int, default=0, help="Sampling RNG seed.")
    en.add_argument(
        "--net",
        default="",
        help="GBM value-net path (pinned into STS2_RUNAI_VALUE for all workers). "
        "Empty → bin default (Ironclad human-outcome net). Ignored for --backend ppo.",
    )
    en.add_argument(
        "--backend",
        default="gbm",
        choices=["gbm", "ppo"],
        help="Scorer backend: 'gbm' (legacy Ironclad GBM) or 'ppo' (deployed "
        "per-character PPO value head). Default: gbm.",
    )

    # --- emit-combat ---------------------------------------------------------
    cb = sub.add_parser(
        "emit-combat",
        help="Compute the four-cell forced-deck combat score table (JSON).",
    )
    cb.add_argument("--obs", default="obs_catalog.json", help="Input obs_catalog.json.")
    cb.add_argument("--out", default="combat.json", help="Output combat scores file.")
    cb.add_argument("--character", default="ironclad", help="Character tag for synergy_fight.")
    cb.add_argument("--root", default=str(CODEX_ROOT), help="Corpus root directory.")
    cb.add_argument("--top-k", type=int, default=150, help="Top-PMI pairs (confirmed candidates).")
    cb.add_argument(
        "--sample-k", type=int, default=150, help="Below-median pairs (undiscovered candidates)."
    )
    cb.add_argument("--n-baselines", type=int, default=3, help="Competitive baseline decks.")
    cb.add_argument(
        "--encounters",
        default="SLIMES_NORMAL,EXOSKELETONS_NORMAL,TUNNELER_NORMAL",
        help="Comma-separated encounter-id panel.",
    )
    cb.add_argument("--seeds", type=int, default=8, help="Number of engine seeds (range(N)).")
    cb.add_argument("--sims", type=int, default=500, help="MCTS simulations per move.")
    cb.add_argument("--depth", type=int, default=50, help="Max rollout depth.")
    cb.add_argument("--workers", type=int, default=_default_workers(), help="Process-pool size.")
    cb.add_argument("--seed", type=int, default=0, help="Pair-sample + baseline RNG seed.")
    cb.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print a done/total progress line every N completed pairs (0 = silent).",
    )
    cb.add_argument(
        "--checkpoint",
        default="",
        help="Write the partial combat table here every --log-every pairs "
        "(crash/kill-safe). Omit to disable incremental checkpointing.",
    )

    return parser


def _order_catalog(args: argparse.Namespace) -> None:
    """Bake card_order (hierarchical clustering seriation) into catalog.json."""
    import json

    from synergy.order import compute_card_order

    obs_path = Path(args.obs)
    catalog_path = Path(args.catalog)
    out_path = Path(args.out)

    print(f"Reading dense obs catalog from {obs_path} …", file=sys.stderr)
    obs_raw = json.loads(obs_path.read_text(encoding="utf-8"))
    obs_pairs = obs_raw.get("pairs", [])

    print(f"Reading display catalog from {catalog_path} …", file=sys.stderr)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    # Signal cards: those that appear in at least one edge of the display catalog
    signal_ids: set[str] = set()
    for edge in catalog.get("edges", []):
        signal_ids.update(edge["pair"])

    print(
        f"Computing card_order for {len(signal_ids)} signal cards "
        f"(from {len(catalog.get('edges', []))} edges) …",
        file=sys.stderr,
    )

    card_order = compute_card_order(obs_pairs, signal_ids=signal_ids)
    print(f"card_order: {len(card_order)} cards ordered.", file=sys.stderr)

    # Mark signal on each node
    signal_order_set = set(card_order)
    for node in catalog.get("nodes", []):
        node["signal"] = node["id"] in signal_order_set

    # Bake card_order at the top level
    catalog["card_order"] = card_order

    out_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Wrote updated catalog to {out_path}", file=sys.stderr)


def _default_workers() -> int:
    """Default process-pool size: leave 2 cores free, cap at 16."""
    return max(1, min(16, (os.cpu_count() or 4) - 2))


def _emit_econ(args: argparse.Namespace) -> None:
    """Compute the econ (value-net second-difference) score table → JSON."""
    from synergy.causal_econ import ensure_binary_fresh
    from synergy.causal_run import compute_econ_table, sample_econ_contexts

    # Build the scorer once in the parent so all no_build=True workers find it.
    # `touch` afterward: the mtime-freshness check is per-crate (any sibling .rs
    # edit flags every bin stale even when cargo correctly no-ops the relink),
    # so bump the verified-fresh binary's mtime past all sources.
    binary = ensure_binary_fresh()
    binary.touch()

    obs_raw = json.loads(Path(args.obs).read_text(encoding="utf-8"))
    pairs = [tuple(sorted(p["pair"])) for p in obs_raw.get("pairs", [])]
    print(f"Econ: {len(pairs)} pairs from {args.obs}", file=sys.stderr)

    root = Path(args.root)
    print(f"Sampling {args.n_contexts} contexts ({args.character}) …", file=sys.stderr)
    contexts = sample_econ_contexts(args.character, args.n_contexts, root, seed=args.seed)
    print(f"Sampled {len(contexts)} contexts.", file=sys.stderr)

    net_path = args.net or None
    table = compute_econ_table(
        pairs,
        contexts,
        workers=args.workers,
        net_path=net_path,
        net=args.backend,
        character=args.character,
    )
    Path(args.out).write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"Wrote econ scores for {len(table)} pairs → {args.out}", file=sys.stderr)


def _emit_combat(args: argparse.Namespace) -> None:
    """Compute the combat (four-cell fight interaction) score table → JSON."""
    from synergy.causal_combat import _ensure_fight_binary
    from synergy.causal_run import (
        compute_combat_table,
        sample_baselines,
        select_pairs,
    )

    # Build the fight bin once in the parent so all no_build=True workers find it
    # (see _emit_econ for why the touch is needed).
    fight_binary = _ensure_fight_binary()
    fight_binary.touch()

    obs_raw = json.loads(Path(args.obs).read_text(encoding="utf-8"))
    pairs = select_pairs(
        obs_raw.get("pairs", []),
        top_k=args.top_k,
        sample_k=args.sample_k,
        seed=args.seed,
    )
    print(
        f"Combat: {len(pairs)} pairs selected (top {args.top_k} + sample {args.sample_k}).",
        file=sys.stderr,
    )

    root = Path(args.root)
    baselines = sample_baselines(args.character, args.n_baselines, root, seed=args.seed)
    print(
        f"Sampled {len(baselines)} baseline decks for {args.character}.",
        file=sys.stderr,
    )
    if not baselines:
        print("No competitive baseline decks found — aborting.", file=sys.stderr)
        sys.exit(1)

    encounters = [e.strip() for e in args.encounters.split(",") if e.strip()]
    seeds = list(range(args.seeds))
    print(
        f"Panel: {len(encounters)} encounters × {len(seeds)} seeds × "
        f"{len(baselines)} baselines @ {args.sims} sims, {args.workers} workers.",
        file=sys.stderr,
    )

    table = compute_combat_table(
        pairs,
        character=args.character,
        baselines=baselines,
        encounters=encounters,
        seeds=seeds,
        sims=args.sims,
        workers=args.workers,
        depth=args.depth,
        log_every=args.log_every,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
    )
    Path(args.out).write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(
        f"Wrote combat scores for {len(table)}/{len(pairs)} pairs → {args.out}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry-point; callable from tests via ``main(argv)``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "emit-observational":
        _emit_observational(args)
    elif args.command == "emit-catalog":
        _emit_catalog(args)
    elif args.command == "embed-catalog":
        from synergy.embed import embed_catalog as _embed_catalog

        _embed_catalog(
            obs_path=Path(args.obs),
            catalog_path=Path(args.catalog),
            out_path=Path(args.out),
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )
    elif args.command == "order-catalog":
        _order_catalog(args)
    elif args.command == "emit-geometry":
        _emit_geometry(args)
    elif args.command == "emit-archetypes":
        _emit_archetypes(args)
    elif args.command == "emit-dose":
        _emit_dose(args)
    elif args.command == "emit-injection":
        _emit_injection(args)
    elif args.command == "emit-archetype-catalog":
        _emit_archetype_catalog(args)
    elif args.command == "emit-econ":
        _emit_econ(args)
    elif args.command == "emit-combat":
        _emit_combat(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
