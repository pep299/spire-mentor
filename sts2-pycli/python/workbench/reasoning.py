"""Adapter from a combat state + config to per-action MCTS scores.

Wraps sts2_hint.recommender.recommend. The native sts2_rs backend may be
unavailable (e.g. CI without a build); on any failure this returns an
``error`` payload with empty scores instead of raising.
"""

from __future__ import annotations

from typing import Any

# Sentinel stable_idx values mirrored from sts2_hint.recommender.
_END_TURN = -1
_POTION_BASE = -100  # potion slot s -> -100 - s, i.e. range (-200, -100]
_POTION_MIN = -200  # exclusive lower bound for the potion range


def _potion_name(state: dict[str, Any], slot: int) -> str | None:
    """Name of the potion in player slot *slot*, or None if absent."""
    potions = (state.get("player") or {}).get("potions") or state.get("potions") or []
    if 0 <= slot < len(potions):
        p = potions[slot]
        name = p.get("name") if isinstance(p, dict) else p
        if isinstance(name, dict):  # bilingual payload
            return name.get("en") or name.get("zh") or "?"
        return name or f"slot {slot}"
    return None


def _recommend(state: dict[str, Any], cfg: Any) -> dict[int, Any]:
    """Thin indirection so tests can monkeypatch without importing sts2_rs."""
    from sts2_hint.recommender import RecommendConfig, recommend  # type: ignore[import]

    rc = RecommendConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
    return recommend(state, rc)


def _action_label(stable_idx: int, state: dict[str, Any]) -> str:
    """Return a human-readable label for a stable action index.

    Args:
        stable_idx: The stable action index from the recommender. ``-1``
            means end turn.
        state: The combat state dict containing a ``hand`` list.

    Returns:
        ``"end turn"`` for -1, ``"drink <potion>"`` for a potion sentinel,
        the matching card name from the hand, or ``"action {stable_idx}"``
        as a fallback.
    """
    if stable_idx == _END_TURN:
        return "end turn"
    if _POTION_MIN < stable_idx <= _POTION_BASE:
        name = _potion_name(state, _POTION_BASE - stable_idx)
        return f"drink {name}" if name else f"drink potion slot {_POTION_BASE - stable_idx}"
    for card in state.get("hand", []) or []:
        if card.get("index") == stable_idx:
            return card.get("name", f"card {stable_idx}")
    return f"action {stable_idx}"


def _opt_float(v: Any) -> float | None:
    return None if v is None else float(v)


def score(state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return per-action MCTS scores or a graceful error payload.

    Args:
        state: A ``combat_play`` decision-point state dict.
        config: Optional recommender config (e.g. ``{"sims": 4000}``).

    Returns:
        On success: ``{"scores": [...], "pv": None, "config": {...}}``.
        On failure: ``{"error": str, "scores": [], "pv": None, "config": {...}}``.
    """
    config = dict(config or {})
    try:
        tags = _recommend(state, config)
    except Exception as exc:
        return {"error": str(exc), "scores": [], "pv": None, "config": config}

    rows: list[dict[str, Any]] = []
    for stable_idx, tag in tags.items():
        rows.append(
            {
                "stable_idx": int(stable_idx),
                "action": _action_label(int(stable_idx), state),
                "win_prob": float(getattr(tag, "win_prob", 0.0)),
                "q": float(getattr(tag, "q", 0.0)),
                "visits": int(getattr(tag, "visits", 0)),
                "is_best": bool(getattr(tag, "is_best", False)),
                "q_saturated": bool(getattr(tag, "q_saturated", False)),
                "low_confidence": bool(getattr(tag, "low_confidence", False)),
                "target_index": getattr(tag, "target_index", None),
                "mean_final_hp": _opt_float(getattr(tag, "mean_final_hp", None)),
                "hp_sd": _opt_float(getattr(tag, "hp_sd", None)),
                "hp_n": int(getattr(tag, "hp_n", 0) or 0),
            }
        )
    rows.sort(key=lambda r: (not r["is_best"], -r["win_prob"]))
    # PV is search-global and not exposed at this boundary today (spec Phase 4).
    return {"scores": rows, "pv": None, "config": config}


def rows_from_mcts(result: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Build score rows from a raw ``sts2_rs.mcts_recommend`` result.

    The native combat state's action space uses ``card_index`` = hand
    position, which the Rust backend also uses as each hand card's
    ``index`` — so the stable index is the card index directly. Sentinels
    mirror :mod:`sts2_hint.recommender`.

    ``state`` must be the position the search ROOTED at, not a later one:
    labels are resolved against ``state["hand"]`` by index, so passing a
    post-action state silently renames every row. The second caller — the
    run-mode stepper, which reports the search that produced the move it
    just applied — therefore passes its *pre*-action snapshot.

    Also accepts a ``HeuristicRunAgent`` step record's ``combat_search``
    dict: the pyo3 layer builds both from ``action_score_pydict``, so the
    ``{"actions": [...]}`` shape is identical by construction.
    """
    rows: list[dict[str, Any]] = []
    for entry in result.get("actions", []):
        a = entry["action"]
        kind = a.get("action")
        if kind == "play":
            stable_idx = int(a["card_index"])
        elif kind == "end_turn":
            stable_idx = _END_TURN
        elif kind in ("potion", "drink_potion"):
            stable_idx = _POTION_BASE - int(a["slot"])
        else:
            continue  # channel/evoke/choose not surfaced at this boundary yet
        target = a.get("target_index")
        rows.append(
            {
                "stable_idx": stable_idx,
                "action": _action_label(stable_idx, state),
                "win_prob": float(entry.get("p_win", 0.0)),
                "q": float(entry.get("q", entry.get("value", 0.0))),
                "visits": int(entry.get("visits", 0)),
                "is_best": bool(entry.get("is_best", False)),
                "q_saturated": bool(entry.get("q_saturated", False)),
                "low_confidence": bool(entry.get("low_confidence", False)),
                "target_index": int(target) if target is not None else None,
                "wins": int(entry.get("wins", 0)),
                "losses": int(entry.get("losses", 0)),
                "ties": int(entry.get("ties", 0)),
                "mean_final_hp": _opt_float(entry.get("mean_final_hp")),
                "hp_sd": _opt_float(entry.get("hp_sd")),
                "hp_n": int(entry.get("hp_n", 0) or 0),
            }
        )
    rows.sort(key=lambda r: (not r["is_best"], -r["win_prob"]))
    return rows


def score_native(
    cs: Any, state: dict[str, Any], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Per-action MCTS scores from a native ``sts2_rs`` combat state.

    Used by the Rust backend, which already holds the engine ``CombatState``
    and so skips the lossy C#-dict → wire translation that :func:`score` does.
    Same return shape as :func:`score`.
    """
    config = dict(config or {})
    try:
        import sts2_rs
        from sts2_hint.recommender import RecommendConfig

        cfg = RecommendConfig.from_dict(config)
        result = sts2_rs.mcts_recommend(
            cs,
            sims=cfg.sims,
            c_puct=cfg.resolved_c_puct(),
            depth=cfg.depth,
            seed=cfg.seed,
            threads=cfg.threads,
            fair=cfg.fair,
            rollout_policy=cfg.rollout_policy,
            # Forwarded so the panel can run the DEPLOYED honest search rather
            # than the leaky clairvoyant default. Dropping them here silently
            # pinned every hint to `hidden_info="clairvoyant"` no matter what
            # the caller asked for -- advice from a search that can see the
            # draw order the human can't.
            hidden_info=cfg.hidden_info,
            draw_widening_c=cfg.resolved_draw_widening_c(),
            endturn_widening_c=cfg.endturn_widening_c,
            endturn_widening_alpha=cfg.endturn_widening_alpha,
            dominance_pruning=cfg.dominance_pruning,
        )
    except Exception as exc:
        return {"error": str(exc), "scores": [], "pv": None, "config": config}
    return {"scores": rows_from_mcts(result, state), "pv": None, "config": config}


def rank_choice_native(cs: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fast per-option MCTS ranking for an in-combat pending choice.

    Runs a single combat MCTS search over a native ``sts2_rs`` combat state that
    sits on an open ``pending_choice`` (e.g. Headbutt's "put a card on top of the
    draw pile"). When a choice is open the engine's only legal actions are the
    ``ResolveChoice`` options, so the root children *are* the choice options and
    each carries its own visit / Q / win-probability stats — exactly the signal
    the slow :mod:`workbench.decision_advisor` re-derives with full-act rollouts,
    but from one combat search instead of (options × rollouts) playouts.

    Returns ``{"by_key": {choice_key: {win_prob, q, visits, is_best}}, "config"}``
    where ``choice_key`` is ``"choice:skip"`` for the empty-selection skip and
    ``f"choice:{opt}"`` otherwise — the same keys :mod:`workbench.decision_advisor`
    and the frontend join on. On failure: ``{"error": str, "by_key": {}, ...}``.
    """
    config = dict(config or {})
    try:
        import sts2_rs
        from sts2_hint.recommender import RecommendConfig

        cfg = RecommendConfig.from_dict(config)
        result = sts2_rs.mcts_recommend(
            cs,
            sims=cfg.sims,
            c_puct=cfg.resolved_c_puct(),
            depth=cfg.depth,
            seed=cfg.seed,
            threads=cfg.threads,
            fair=cfg.fair,
            rollout_policy=cfg.rollout_policy,
            hidden_info=cfg.hidden_info,
            draw_widening_c=cfg.resolved_draw_widening_c(),
            endturn_widening_c=cfg.endturn_widening_c,
            endturn_widening_alpha=cfg.endturn_widening_alpha,
            dominance_pruning=cfg.dominance_pruning,
        )
    except Exception as exc:
        return {"error": str(exc), "by_key": {}, "config": config}

    by_key: dict[str, dict[str, Any]] = {}
    for entry in result.get("actions", []):
        a = entry["action"]
        if a.get("action") != "choose":
            continue
        key = _choice_key(a)
        by_key[key] = {
            "win_prob": float(entry.get("p_win", 0.0)),
            "q": float(entry.get("q", entry.get("value", 0.0))),
            "visits": int(entry.get("visits", 0)),
            "is_best": bool(entry.get("is_best", False)),
        }
    return {"by_key": by_key, "config": config}


def _choice_key(action: dict[str, Any]) -> str:
    """``choice:skip`` / ``choice:{opt}`` for a ``choose`` action dict.

    The engine serializes a ``ResolveChoice`` with its full ``selection`` list
    (``[]`` for a skip, ``[i]`` for picking option *i*); PyO3 ships that ``Vec<u8>``
    as ``bytes``. The ``selection`` is what distinguishes a skip from option 0 —
    both collapse to ``index == 0`` — so prefer it, falling back to ``index`` only
    on an older wheel that predates the field (where skip is indistinguishable).
    """
    sel = action.get("selection")
    if sel is not None:
        opts = list(sel)  # bytes -> [int]; empty -> [] (skip)
        return "choice:skip" if not opts else f"choice:{opts[0]}"
    return f"choice:{int(action.get('index', 0))}"


def search_tree_native(cs: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full MCTS search tree from a native ``sts2_rs`` combat state."""
    config = dict(config or {})
    try:
        import sts2_rs
        from sts2_hint.recommender import RecommendConfig

        cfg = RecommendConfig.from_dict(config)
        tree = sts2_rs.mcts_search_tree(
            cs,
            sims=cfg.sims,
            c_puct=cfg.resolved_c_puct(),
            depth=cfg.depth,
            seed=cfg.seed,
            rollout_policy=cfg.rollout_policy,
            # Forwarded for the same reason :func:`score_native` forwards them:
            # without these the tree inspector ran the leaky clairvoyant search
            # while the hint panel beside it ran the deployed honest one, so
            # the two disagreed on the same position and no UI knob could
            # reconcile them.
            hidden_info=cfg.hidden_info,
            draw_widening_c=cfg.resolved_draw_widening_c(),
            endturn_widening_c=cfg.endturn_widening_c,
            endturn_widening_alpha=cfg.endturn_widening_alpha,
            dominance_pruning=cfg.dominance_pruning,
        )
    except Exception as exc:
        return {"error": str(exc), "nodes": [], "root": None, "pv": [], "config": config}
    tree["config"] = config
    return tree


def _replay_cs(fixture_path: str, actions: list[dict[str, Any]]) -> Any:
    """Reload a fixture and replay native actions to reproduce the exact state.

    ``combat_state_from_scenario_toml`` is deterministic (fixed scenario seed)
    and the recorded ``actions`` are the raw engine actions the recorder
    applied, so the rebuilt state is bit-identical to the one at that decision
    — including the RNG cursor and intent history a dict reconstruction can't
    recover.
    """
    import sts2_rs

    cs = sts2_rs.combat_state_from_scenario_toml(fixture_path)
    for a in actions:
        cs.apply(a)
    return cs


def search_tree_fixture_replay(
    fixture_path: str, actions: list[dict[str, Any]], config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Faithful MCTS tree from a fixture-sourced recorded run.

    Reloads ``fixture_path`` and replays ``actions`` to the decision, then runs
    the native search — so the result matches the recorder's saved reasoning
    (at matching sims/seed) instead of diverging like the dict reconstruction.
    """
    config = dict(config or {})
    try:
        cs = _replay_cs(fixture_path, actions)
    except Exception as exc:
        return {
            "error": f"fixture replay failed: {exc}",
            "nodes": [],
            "root": None,
            "pv": [],
            "config": config,
        }
    result = search_tree_native(cs, config)
    if not result.get("error"):
        result["faithful"] = True
    return result


def score_fixture_replay(
    fixture_path: str,
    actions: list[dict[str, Any]],
    state: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Faithful per-action scores from a fixture-sourced recorded run (see
    :func:`search_tree_fixture_replay`)."""
    config = dict(config or {})
    try:
        cs = _replay_cs(fixture_path, actions)
    except Exception as exc:
        return {
            "error": f"fixture replay failed: {exc}",
            "scores": [],
            "pv": None,
            "config": config,
        }
    result = score_native(cs, state, config)
    if not result.get("error"):
        result["faithful"] = True
    return result


def _search_tree(state: dict[str, Any], cfg: Any) -> dict[str, Any]:
    """Thin indirection so tests can monkeypatch without importing sts2_rs."""
    from sts2_hint.recommender import RecommendConfig, search_tree  # type: ignore[import]

    rc = RecommendConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
    return search_tree(state, rc)


def search_tree(state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the agent's full MCTS search tree, or a graceful error payload.

    Args:
        state: A ``combat_play`` decision-point state dict.
        config: Optional recommender config (e.g. ``{"sims": 3000}``).

    Returns:
        On success the raw tree dict from the native backend
        (``{"nodes", "root", "pv", "sims", "c_puct"}``) plus the echoed
        ``config``. On failure ``{"error": str, "nodes": [], "config": {...}}``.
    """
    config = dict(config or {})
    try:
        tree = _search_tree(state, config)
    except Exception as exc:
        return {"error": str(exc), "nodes": [], "root": None, "pv": [], "config": config}
    tree["config"] = config
    return tree
