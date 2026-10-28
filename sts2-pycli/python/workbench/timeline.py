"""In-memory timeline of decision steps loaded from a recorded run.

A step is one decision-point state plus the action that followed it. This
module is the single source of truth the web layer projects; it has no HTTP
or game-engine knowledge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from workbench.seed import json_exact_seed


class Step(TypedDict):
    idx: int
    decision: str
    state: dict[str, Any]
    # legal_actions and scores start empty; populated by the reasoning adapter.
    legal_actions: list[dict[str, Any]]
    scores: list[dict[str, Any]]
    pv: list[str] | None
    chosen_action: dict[str, Any] | None
    # Per-decision PPO value recorded live ({"value", "taken_label"}), when the
    # run was played in the workbench with value recording available.
    wpa: dict[str, Any] | None
    # This step's action's combat-log display rows, when the recording carried
    # them (workbench-saved runs). Empty otherwise; /api/timeline falls back to
    # slicing ``combat_log_raw`` off consecutive states for bridge logs.
    combat_log: list[dict[str, Any]]
    source: str
    meta: dict[str, Any]


def _compact_advice(advice: Any) -> dict[str, Any] | None:
    """A recorded advice object as the small ``meta.advice`` display row.

    The visit-backed ranking already becomes ``scores``; this is the fallback
    surface for POLICY-ranked advice (screen decisions have no visits) plus
    the advisor's one-line note. None when there is nothing to show.
    """
    if not isinstance(advice, dict) or not advice:
        return None
    raw = [o for o in (advice.get("options") or []) if isinstance(o, dict)]
    total_visits = sum(o.get("visits") or 0 for o in raw)
    opts: list[dict[str, Any]] = []
    for o in raw:
        share = o.get("visit_share")
        if share is None and total_visits:
            share = (o.get("visits") or 0) / total_visits
        opts.append(
            {
                "label": o.get("label") or o.get("action") or "?",
                "share": share,
                "is_best": bool(o.get("is_best")),
            }
        )
    best = advice.get("best") if isinstance(advice.get("best"), dict) else {}
    out = {
        "best": best.get("label"),
        "note": advice.get("note"),
        "options": opts[:12],
    }
    if not (out["best"] or out["note"] or opts):
        return None
    return out


def _graft_path_plan(state: dict[str, Any], advice: Any) -> None:
    """Draw the advisor's planned route on the recorded map.

    Map advice carries the hop list in ``advice["path"]``; MapView renders a
    route only from ``full_map.path_plan`` (the Rust backend's optionality
    surface). Synthesize the minimal plan -- route only, no per-column
    optionality stats -- so live-game replays show the predicted path.
    """
    if not isinstance(advice, dict):
        return
    fm = state.get("full_map")
    path = advice.get("path")
    if not (isinstance(fm, dict) and isinstance(path, list) and fm.get("path_plan") is None):
        return
    route = [
        {"col": h.get("col"), "row": h.get("row")}
        for h in path
        if isinstance(h, dict) and h.get("col") is not None and h.get("row") is not None
    ]
    if not route:
        return
    fm["path_plan"] = {
        "best_col": route[0]["col"],
        "baseline_col": route[0]["col"],
        "diverged": False,
        "route": route,
        "columns": [],
    }


class _CoachCollector:
    """The live coach's records in a run log (``type: "coach"`` lines,
    ``realgame_coach``). Twin of ``CoachCollector`` in ``runlog.rs``.

    Records attach to the step their ``log_idx`` names, never to the preceding
    ``state`` line: a turn's grade lands after later states, the autopsy after
    the terminal one. A ``withdraw {ids}`` drops those ids wherever it sits; an
    id emitted again after it comes back.
    """

    def __init__(self) -> None:
        # id -> record in file order; a re-emitted id keeps its slot.
        self.records: dict[str, dict[str, Any]] = {}
        self.autopsy: dict[str, Any] | None = None
        self._anon = 0

    def push(self, data: dict[str, Any]) -> None:
        kind = data.get("kind")
        if kind == "withdraw":
            ids = {i for i in data.get("ids") or [] if isinstance(i, str)}
            for i in ids:
                self.records.pop(i, None)
            if self.autopsy is not None and self.autopsy.get("id") in ids:
                self.autopsy = None
        elif kind == "autopsy":
            self.autopsy = data
        elif kind in ("turn", "choice", "recap"):
            rid = data.get("id")
            if not isinstance(rid, str):
                self._anon += 1
                rid = f"anon:{self._anon}"
            self.records[rid] = data

    def to_dict(self) -> dict[str, Any]:
        """``{records, by_step, autopsy}``: ``by_step`` maps a step index to the
        ids of the graded records on it (an ``ungraded`` turn is listed, not
        stepped)."""
        by_step: dict[int, list[str]] = {}
        for rid, r in self.records.items():
            idx = r.get("log_idx")
            if r.get("verdict") == "ungraded" or not isinstance(idx, int) or idx < 0:
                continue
            by_step.setdefault(idx, []).append(rid)
        return {
            "records": list(self.records.values()),
            "by_step": {str(k): v for k, v in sorted(by_step.items())},
            "autopsy": self.autopsy,
        }


class TimelineStore:
    """Ordered, seekable collection of :class:`Step`."""

    def __init__(
        self,
        steps: list[Step],
        head: dict[str, Any] | None = None,
        coach: dict[str, Any] | None = None,
    ) -> None:
        self._steps = steps
        self._head = head or {}
        self._coach = coach or _CoachCollector().to_dict()

    def coach(self) -> dict[str, Any]:
        """The run log's live-coach records: ``{records, by_step, autopsy}``."""
        return self._coach

    def __len__(self) -> int:
        return len(self._steps)

    def fixture_path(self) -> str | None:
        """Absolute path to the source fixture, when this run was recorded from
        one (``workbench.record --fixture``). Enables a faithful native
        re-search (reload + replay) instead of a lossy dict reconstruction.
        """
        return self._head.get("fixture_path")

    def mcts_seed(self) -> int:
        """The MCTS seed the recorder used, so a re-search reproduces the same
        RNG path (defaults to 1, the recorder default)."""
        seed = self._head.get("mcts_seed")
        return int(seed) if seed is not None else 1

    def character(self) -> str:
        """Display character of the recorded run (for the play-from-here board);
        defaults to Ironclad."""
        return self._head.get("character") or "Ironclad"

    def mcts_config(self) -> dict[str, Any]:
        """The full MCTS config the recorder used (sims / c_puct / depth / seed /
        rollout_policy). Used as the base for a faithful re-search so it matches
        the saved reasoning instead of falling back to RecommendConfig defaults
        (whose c_puct=1.4 silently changes the recommended move)."""
        cfg = dict(self._head.get("mcts_config") or {})
        cfg.setdefault("seed", self.mcts_seed())
        return cfg

    def run_identity(self) -> dict[str, Any] | None:
        """The run parameters needed to REBUILD this run from its seed, or None.

        A recorded line only replays through the same constructor
        (``run_mode``), Act-1 model and engine that produced it — the same
        triple a named save carries (``workbench.saves``). Only workbench
        ``BranchingLiveSession`` logs record them (they land in the head meta),
        so ``None`` here means "this recording cannot be resumed", not "resume
        it on the defaults": guessing replays the line against a different game
        and the engine rejects it several actions in.
        """
        head = self._head
        if head.get("run_mode") is None or not head.get("character"):
            return None
        ident = {
            "character": head["character"],
            "seed": str(head.get("seed", "")),
            "ascension": int(head.get("ascension") or 0),
            "run_mode": bool(head["run_mode"]),
            "act1": head.get("act1") or "overgrowth",
            "backend": head.get("backend") or "rust",
        }
        # GH #801: the build the run simulated. The log writer omits it at the
        # default (v0.107.1), so its absence means the default; only a
        # non-default build is carried here.
        if head.get("game_version"):
            ident["game_version"] = head["game_version"]
        return ident

    def replay_line(self, idx: int) -> list[dict[str, Any]] | None:
        """The ``[{"action", "args"}]`` line that reaches step ``idx``, or None.

        This is the shape :meth:`~workbench.branching.BranchingLiveSession.load_line`
        re-applies — NOT the step's ``chosen_action.args``, which the writer
        normalizes to the native shape the regret review keys off (a verb's
        args lose their ``action``; a native RunAction loses ``hand_idx``) and
        which therefore cannot drive a restore. Returns ``None`` when any step
        before *idx* lacks the field — logs written before it existed, and every
        non-workbench recording.
        """
        out: list[dict[str, Any]] = []
        for j in range(idx):
            rec = (self._steps[j].get("chosen_action") or {}).get("replay")
            if not isinstance(rec, dict) or not rec.get("action"):
                return None
            out.append({"action": rec["action"], "args": dict(rec.get("args") or {})})
        return out

    def native_actions_before(self, idx: int) -> list[dict[str, Any]] | None:
        """The native engine actions applied to *reach* step ``idx``.

        Each combat step's ``chosen_action`` carries the raw ``args`` the
        recorder applied; replaying ``args[0..idx)`` over a fresh fixture load
        reproduces the exact native state at step ``idx``. Returns ``None`` if
        any earlier step lacks a replayable native action (e.g. a full-run
        recording with non-combat phases), signalling the caller to fall back
        to dict reconstruction.
        """
        out: list[dict[str, Any]] = []
        for j in range(idx):
            args = (self._steps[j].get("chosen_action") or {}).get("args")
            if not isinstance(args, dict):
                return None
            out.append(args)
        return out

    def get(self, idx: int) -> Step:
        """Return the step at *idx*.

        Args:
            idx: Zero-based step index.

        Raises:
            IndexError: If *idx* is out of range.
        """
        if idx < 0 or idx >= len(self._steps):
            raise IndexError(f"step {idx} out of range (0..{len(self._steps) - 1})")
        return self._steps[idx]

    def all(self) -> list[Step]:
        """Return a shallow copy of all steps."""
        return list(self._steps)

    @classmethod
    def load_jsonl(cls, path: str | Path) -> TimelineStore:
        """Parse a recorded run JSONL file into a :class:`TimelineStore`.

        Each ``state`` line becomes one :class:`Step`; the ``action`` line
        immediately following it (if any) is attached as ``chosen_action``.

        Args:
            path: Path to a ``.jsonl`` file produced by ``game_log.py``.

        Returns:
            A new :class:`TimelineStore` with one step per decision point.
        """
        states: list[dict[str, Any]] = []
        actions: dict[int, dict[str, Any]] = {}
        mcts_by_state: dict[int, dict[str, Any]] = {}
        wpa_by_state: dict[int, dict[str, Any]] = {}
        clog_by_state: dict[int, list[dict[str, Any]]] = {}
        coach = _CoachCollector()
        head: dict[str, Any] = {}
        # errors="replace": a single mojibake byte (e.g. a cp1252-encoded
        # middle dot from a pre-2026-08-21 live-advisor log) must not make a
        # whole run unloadable -- the bad char degrades to U+FFFD in one note.
        for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            entry = json.loads(raw)
            if entry.get("type") == "state":
                states.append(entry)
            elif entry.get("type") == "action":
                # Attach to the most-recently-seen state by its position.
                if states:  # ignore actions logged before the first state
                    actions[len(states) - 1] = entry.get("data")
            elif entry.get("type") == "mcts":
                if states:
                    mcts_by_state[len(states) - 1] = entry.get("data") or {}
            elif entry.get("type") == "wpa":
                # Per-decision PPO value recorded live (workbench WPA review) —
                # attaches to the preceding state like an mcts record.
                if states:
                    wpa_by_state[len(states) - 1] = entry.get("data") or {}
            elif entry.get("type") == "combat_log":
                # This action's display-ready combat-log rows, written between
                # the action record and the state it produced — so they attach
                # to the preceding state: the decision the action answered.
                if states:
                    clog_by_state[len(states) - 1] = entry.get("data") or []
            elif entry.get("type") == "coach":
                coach.push(entry.get("data") or {})
            elif entry.get("type") == "meta":
                data = entry.get("data") or {}
                # The head meta (not the trailing summary) carries fixture_path /
                # mcts_seed for the faithful re-search.
                if not data.get("summary"):
                    head.update(data)

        # WHICH MAP NODE each floor's room sits on. The game rolls a combat
        # room's layer variants from `Rng(runSeed + row + col * 747)`
        # (`NCombatRoom.GenerateBackgroundRngForCurrentPoint`), so a fight can
        # only reproduce the real one's barrels, netting and windows if it knows
        # its node.
        #
        # Collected in a PASS OF ITS OWN because the coord arrives AFTER the room
        # it belongs to: a floor's `map_select` reports the node you are standing
        # on now, and it is logged once the floor's fight is over. Carrying it
        # forward the way floor/act are carried would label every fight with the
        # PREVIOUS floor's node. `row` also restarts each act while `floor` is
        # cumulative, so it cannot be derived either.
        coord_by_floor: dict[Any, dict[str, int]] = {}
        for entry in states:
            data = entry.get("data") or {}
            coord = (data.get("full_map") or {}).get("current_coord")
            if isinstance(coord, dict) and data.get("floor") is not None:
                coord_by_floor[data.get("floor")] = coord

        # floor/act/act_name only appear on map_select & game_over states, so
        # carry the last-seen values forward to label intervening combat steps.
        floor = act = act_name = None
        # The run's master RNG seed, from whichever states carry the rng block.
        run_seed: int | None = None
        steps: list[Step] = []
        for i, entry in enumerate(states):
            data = entry.get("data") or {}
            if data.get("floor") is not None:
                floor = data.get("floor")
            if data.get("act") is not None:
                act = data.get("act")
            if data.get("act_name") is not None:
                act_name = data.get("act_name")
            seed_now = (data.get("rng") or {}).get("run_seed")
            if isinstance(seed_now, int):
                run_seed = seed_now
            node = coord_by_floor.get(floor)
            m = mcts_by_state.get(i) or {}
            advice = m.get("advice")
            _graft_path_plan(data, advice)
            # Two writers produce `root_outcomes`, and they disagree on two
            # keys, so read both shapes:
            #   * `workbench.record` (the offline MCTS recorder) writes
            #     `action_label` and no win-prob, leaving it to be derived.
            #   * `BranchingLiveSession.quit` writes the live agent's OWN root
            #     rows (`reasoning.rows_from_mcts`), which carry `action` and
            #     the search's `win_prob` verbatim.
            # Prefer a recorded `win_prob`: `wins / visits` is only a proxy for
            # it, and re-deriving would quietly regrade the agent's own numbers.
            scores = [
                {
                    # The HP-aware grade needs these; only the live writers
                    # record them (#850).
                    **{k: r[k] for k in ("mean_final_hp", "hp_sd", "hp_n") if k in r},
                    "stable_idx": r.get("stable_idx"),
                    "action": r.get("action") or r.get("action_label"),
                    "target_index": r.get("target_index"),
                    "win_prob": (
                        float(r["win_prob"])
                        if r.get("win_prob") is not None
                        else ((r["wins"] / r["visits"]) if r.get("visits") else 0.0)
                    ),
                    "q": r.get("q", 0.0),
                    "visits": r.get("visits", 0),
                    "is_best": r.get("is_best", False),
                    "wins": r.get("wins", 0),
                    "losses": r.get("losses", 0),
                    "ties": r.get("ties", 0),
                }
                for r in m.get("root_outcomes", [])
            ]
            steps.append(
                Step(
                    idx=i,
                    decision=data.get("decision", "unknown"),
                    state=data,
                    legal_actions=[],
                    scores=scores,
                    pv=(m.get("pv") or None),
                    chosen_action=actions.get(i),
                    wpa=wpa_by_state.get(i),
                    combat_log=clog_by_state.get(i, []),
                    source="log",
                    meta={
                        "turn": data.get("round"),
                        "step_no": entry.get("step"),
                        "floor": floor,
                        "act": act,
                        "act_name": act_name,
                        # Absent together: the room falls back to its stable
                        # per-floor hash unless it has all three.
                        "bg_seed": None if run_seed is None else json_exact_seed(run_seed),
                        "bg_row": node.get("row") if node else None,
                        "bg_col": node.get("col") if node else None,
                        "advice": _compact_advice(advice),
                    },
                )
            )
        return cls(steps, head=head, coach=coach.to_dict())
