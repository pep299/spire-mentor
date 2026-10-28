"""Oracle turn planner: exhaustive turn expectimax over CRN playout leaves (GH #830).

Deep bets item 6. The plan (`deepbets/plan.md`, Step 2, "G-B") asks whether
planning a whole turn, scored where the turn ends, would play differently from
the deployed pilot (`MctsCombatAgent`). This module is the planner; the probe
that compares the two is `ppo.combat_bench.p6_probe`, and `make` is the bench
factory for the G-C arm.

One planning point:

1. **Honest root.** `b = cs.belief_view(plan_seed)`. Everything below reads
   `b` and states reached from it, never `cs`, so two worlds the player cannot
   tell apart (`cs.sample_world(r)`) give the same plan.
2. **Exhaustive DFS over the turn** (every legal non-EndTurn action, potions
   and card choices included). Positions are merged on `obs_key`
   (`observation_hash` plus the potion belt and player powers, which the
   engine hash misses): playing either of two identical Strikes is one move
   (`search_hash` keys piles on card uids and would keep them apart).
3. **Chance edges** are the transitions that move `rng_fingerprint` (the
   deployed search's own chance detector; valid on a belief view). A first
   chance edge averages `k_chance` re-planned outcomes: outcome `i` is
   `x.belief_view(c_i)` with the action applied, and its rest of the turn is
   enumerated again. A chance edge below one becomes a `belief` rollout leaf.
   An outcome whose turn already ended (a card that ends the turn) becomes a
   rollout leaf too.
4. **Leaves.** Every node where EndTurn is legal is an "end the turn here"
   leaf: the mean of `n` `values_many(mode="belief_endturn")` playouts with
   the SAME seeds `S_1..S_n` at every leaf (common random numbers: seed `j`
   deals every plan the same next turn). A rollout leaf is the playout from
   `x.belief_view(S_j)` after the action, seed `S_j` (the forced action's own
   randomness varies with `j`). A won / lost position is scored exactly.
5. **Backup.** A node takes the child with the highest mean, keeping its
   per-seed vector, so two plans compare as paired samples; a chance edge
   is the per-seed mean of its outcomes.
6. **Above `leaf_cap` leaves** (or `node_cap` decision nodes) the planner
   declines and the agent falls back to `MctsCombatAgent`.

Actions are matched across states by card identity, never by index: a
deterministic move by the observation it leads to, a chance move by the
content of the card (or potion, or chosen cards) it plays.

Cost: `n` playouts per leaf; the leaf count is logged per planning point.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# The planner's numpy work is tiny vectors; an OpenBLAS pool would start one
# thread per core at import. Bench cells and probe tasks budget their CPUs.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

N_DEFAULT = 64
K_CHANCE = 4
LEAF_CAP = 2000
# Decision nodes; a turn this wide declines like one over `leaf_cap`.
NODE_CAP = 20_000
# Jobs per `values_many` call: each job clones its state in the engine.
VALUES_CHUNK = 4096
# The seed of the belief view the action mapper probes a target state with.
# Deterministic moves are matched by the observation they lead to, which
# does not depend on it.
MAP_SEED = 0x5EED_0F_0AC1E
ENDTURN = -1
MASK63 = (1 << 63) - 1

# `pending_choice()["cards"][i]["pile"]` -> the `state_to_dict` deck key.
PILE_KEYS = {
    "hand": "hand",
    "draw": "draw_pile",
    "discard": "discard_pile",
    "exhaust": "exhaust_pile",
}


class PlannerDeclined(Exception):
    """The turn is wider than the planner's caps; the caller falls back."""


class MappingError(Exception):
    """A planned action has no counterpart in the state it must be played in."""


def stable_u64(*parts: Any) -> int:
    """A process-independent 64-bit hash of `parts` (Python's `hash` is salted)."""
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def obs_key(state: Any, d: dict[str, Any] | None = None) -> int:
    """What the player observes of `state`: `observation_hash` plus belt and powers.

    The engine's hashes (`observation_hash`, `search_hash`, `state_hash`)
    miss two observable things (wheel d0abba2e0): the potion belt
    (`PlayerCombatState.potions`), and the player powers applied through
    the creature path (`apply_power_to_creature`), which is how every
    self-buff potion lands: Strength, Dexterity, Flex, Speed, Buffer (Lucky
    Tonic), Retain Hand (Stable Serum), Regen, Plating, Thorns, Ritual,
    Duplication, Gigantification. Drinking one leaves every hash unchanged,
    so merging on `observation_hash` alone turned it into a self-loop (and
    the P6-0 backup into a RecursionError). The belt and the player's
    powers from `state_to_dict` are folded in here.

    Args:
        state: A `sts2_rs.CombatState`.
        d: `state_to_dict(state)` if the caller has it.
    """
    import sts2_rs

    if d is None:
        d = sts2_rs.state_to_dict(state)
    p = d["player"]
    extra = (
        tuple(str(x) for x in p.get("potions") or []),
        tuple(sorted(json.dumps(pw, sort_keys=True) for pw in p.get("powers") or [])),
    )
    return stable_u64(state.observation_hash(), extra)


# ── identity ─────────────────────────────────────────────────────────────


def card_sig(card: dict[str, Any]) -> tuple:
    """A card's content: every `state_to_dict` field but its instance id."""
    return tuple(
        sorted((k, json.dumps(v, sort_keys=True)) for k, v in card.items() if k != "stable_idx")
    )


def _option_sig(opt: dict[str, Any], d: dict[str, Any]) -> tuple:
    # A choice option names the card's pile and index in THAT state's pile
    # order; the content is what identifies it across states.
    key = PILE_KEYS.get(opt.get("pile") or "")
    idx = opt.get("index")
    if key is not None and isinstance(idx, int):
        pile = d["deck"].get(key) or []
        if 0 <= idx < len(pile) and pile[idx].get("id") == opt.get("id"):
            return ("card", card_sig(pile[idx]))
    return ("opt", opt.get("id"), opt.get("upgraded"), opt.get("type"), opt.get("pile"))


def action_identity(state: Any, action: dict[str, Any], d: dict[str, Any] | None = None) -> tuple:
    """What `action` plays in `state`, independent of hand / pile positions.

    Args:
        state: A `sts2_rs.CombatState`.
        action: One of `state.legal_actions()`.
        d: `state_to_dict(state)` if the caller has it.

    Returns:
        `("play", card content, target)`, `("potion", potion id, target)`,
        `("choose", choice kind, sorted chosen contents)` or `("end_turn",)`.
    """
    import sts2_rs

    kind = action["action"]
    if kind == "end_turn":
        return ("end_turn",)
    if d is None:
        d = sts2_rs.state_to_dict(state)
    if kind == "play":
        return (
            "play",
            card_sig(d["deck"]["hand"][action["card_index"]]),
            action.get("target_index"),
        )
    if kind in ("drink_potion", "potion"):
        return ("potion", d["player"]["potions"][action["slot"]], action.get("target_index"))
    if kind == "choose":
        pc = state.pending_choice() or {}
        cards = pc.get("cards") or []
        sel = action.get("selection") or []
        return ("choose", pc.get("kind"), tuple(sorted(_option_sig(cards[i], d) for i in sel)))
    return (kind, json.dumps(action, sort_keys=True))


def identity_label(ident: tuple) -> str:
    """A short human label for an `action_identity`."""
    kind = ident[0]
    if kind == "play":
        fields = dict(ident[1])
        cid = json.loads(fields.get("id", '"?"'))
        up = "+" if json.loads(fields.get("upgraded", "false")) else ""
        tgt = "" if ident[2] is None else f"->{ident[2]}"
        return f"{cid}{up}{tgt}"
    if kind == "potion":
        tgt = "" if ident[2] is None else f"->{ident[2]}"
        return f"potion:{ident[1]}{tgt}"
    if kind == "choose":
        names = []
        for sig in ident[2]:
            if sig[0] == "card":
                f = dict(sig[1])
                names.append(json.loads(f.get("id", '"?"')))
            else:
                names.append(str(sig[1]))
        return f"choose:{ident[1]}:{'+'.join(names) or 'skip'}"
    return kind


def _norm(action: dict[str, Any]) -> tuple:
    kind = action["action"]
    if kind == "play":
        return ("play", action.get("card_index"), action.get("target_index"))
    if kind in ("drink_potion", "potion"):
        return ("drink_potion", action.get("slot"), action.get("target_index"))
    if kind == "choose":
        return ("choose", tuple(action.get("selection") or []))
    return (kind,)


def rm_to_combat(action: dict[str, Any]) -> tuple:
    """A `RunManager` combat action dict as the `_norm` key of a `legal_actions` dict."""
    kind = action["action"]
    if kind == "play_card":
        return ("play", action.get("hand_idx"), action.get("target_index"))
    if kind == "drink_potion":
        return ("drink_potion", action.get("potion_slot"), action.get("target_index"))
    if kind == "combat_resolve_choice":
        opt = action.get("option")
        return ("choose", () if opt is None else (int(opt),))
    if kind == "combat_resolve_choice_take_first":
        return ("choose", tuple(range(int(action["count"]))))
    if kind == "combat_resolve_choice_selection":
        return ("choose", tuple(int(i) for i in action.get("selection") or []))
    return (kind,)


def combat_to_rm(action: dict[str, Any]) -> dict[str, Any]:
    """A `legal_actions` dict as the `RunManager.take_action` dict."""
    kind = action["action"]
    if kind == "play":
        return {
            "action": "play_card",
            "hand_idx": action["card_index"],
            "target_index": action.get("target_index"),
        }
    if kind in ("drink_potion", "potion"):
        return {
            "action": "drink_potion",
            "potion_slot": action["slot"],
            "target_index": action.get("target_index"),
        }
    if kind == "choose":
        sel = list(action.get("selection") or [])
        if len(sel) <= 1:
            return {"action": "combat_resolve_choice", "option": sel[0] if sel else None}
        return {"action": "combat_resolve_choice_selection", "selection": sel}
    if kind == "end_turn":
        return {"action": "end_turn"}
    raise MappingError(f"no run-action form for {action!r}")


def legal_match(state: Any, key: tuple) -> dict[str, Any]:
    """The legal action of `state` whose `_norm` key is `key`.

    Raises:
        MappingError: None is.
    """
    for a in state.legal_actions():
        if _norm(a) == key:
            return a
    raise MappingError(f"{key!r} is not legal here")


# ── the plan ─────────────────────────────────────────────────────────────


@dataclass
class Leaf:
    """A position the planner scores by playouts (or exactly, when terminal)."""

    kind: str  # "endturn" | "rollout" | "terminal"
    state: Any
    action: dict[str, Any] | None = None  # rollout: the forced chance action
    ident: tuple | None = None  # rollout: `action_identity` of `action`


@dataclass
class Edge:
    """One move out of a decision node."""

    action: dict[str, Any]  # legal at the parent's state
    kind: str  # "det" | "terminal" | "chance" | "rollout"
    obs: int | None = None  # det / terminal: the child's `obs_key`
    ident: tuple | None = None  # chance / rollout (and root moves): the identity
    node: DNode | None = None  # det
    leaf: int | None = None  # terminal / rollout
    outcomes: list[Any] = field(default_factory=list)  # chance: DNode | int (leaf)


@dataclass
class DNode:
    """A decision position: end the turn here, or make one of `edges`."""

    obs: int
    chance_depth: int
    endturn: int | None = None
    edges: list[Edge] = field(default_factory=list)
    stuck: int | None = None  # a leaf for a node with no move at all


@dataclass
class PlannedStep:
    """One action of the argmax plan, in the planner's coordinates."""

    action: dict[str, Any]
    kind: str  # "endturn" | "det" | "terminal" | "chance" | "rollout"
    pre_obs: int
    post_obs: int | None = None
    ident: tuple | None = None


@dataclass
class RootMove:
    """A first move from the root, as the probe compares it."""

    key: tuple  # ("end_turn",) | ("obs", child observation) | ("id", identity)
    label: str
    action: dict[str, Any]
    kind: str


class Plan:
    """The enumerated turn of one planning point and, once evaluated, its values."""

    def __init__(self, root_state: Any, plan_seed: int, n: int, k_chance: int) -> None:
        import sts2_rs

        self.root_state = root_state
        self.plan_seed = plan_seed
        self.n = n
        self.k_chance = k_chance
        self.seeds = [stable_u64(plan_seed, "leaf", j) for j in range(n)]
        self.round0 = int(sts2_rs.state_to_dict(root_state)["round"])
        self.leaves: list[Leaf] = []
        self.leaf_index: dict[tuple, int] = {}
        self.n_playout_leaves = 0
        self.nodes: dict[tuple[int, int], DNode] = {}
        self.root: DNode | None = None
        self.values: np.ndarray | None = None
        self.declined: str | None = None
        self.stats: dict[str, Any] = {
            "nodes": 0,
            "applies": 0,
            "det_edges": 0,
            "chance_edges": 0,
            "chance_outcomes": 0,
            "turn_ended_outcomes": 0,
        }

    # -- counts --

    def leaf_counts(self) -> dict[str, int]:
        out = {"endturn": 0, "rollout": 0, "terminal": 0}
        for leaf in self.leaves:
            out[leaf.kind] += 1
        return out

    def playout_leaves(self) -> int:
        """Leaves that cost `n` playouts each (end-turn and rollout leaves)."""
        return self.n_playout_leaves

    # -- backup --

    def _backup(self, cols: Any) -> tuple[dict[int, np.ndarray], dict[int, int]]:
        assert self.values is not None and self.root is not None
        V = self.values[:, cols]
        memo: dict[int, np.ndarray] = {}
        choice: dict[int, int] = {}
        # Guard: an edge back to a position on the current line (a cycle in
        # the merged turn graph) is skipped, since returning to a position
        # cannot beat it. With `obs_key` no cycle is known to occur; the
        # count is logged so one would show.
        on_line: set[int] = set()
        cycles = [0]

        def node_val(nd: DNode) -> np.ndarray | None:
            got = memo.get(id(nd))
            if got is not None:
                return got
            if id(nd) in on_line:
                cycles[0] += 1
                return None
            on_line.add(id(nd))
            best: np.ndarray | None = None
            best_mean = -math.inf
            best_i = ENDTURN
            if nd.endturn is not None:
                best = V[nd.endturn]
                best_mean = float(best.mean())
            for i, e in enumerate(nd.edges):
                v = edge_val(e)
                if v is None:
                    continue
                m = float(v.mean())
                if m > best_mean:
                    best, best_mean, best_i = v, m, i
            if best is None:
                assert nd.stuck is not None
                best = V[nd.stuck]
            on_line.discard(id(nd))
            memo[id(nd)] = best
            choice[id(nd)] = best_i
            return best

        def edge_val(e: Edge) -> np.ndarray | None:
            if e.kind == "det":
                assert e.node is not None
                return node_val(e.node)
            if e.kind in ("terminal", "rollout"):
                assert e.leaf is not None
                return V[e.leaf]
            parts = [node_val(o) if isinstance(o, DNode) else V[o] for o in e.outcomes]
            parts = [p for p in parts if p is not None]
            return np.mean(parts, axis=0) if parts else None

        self._edge_val = edge_val  # for root_values
        node_val(self.root)
        self.stats["cycle_edges"] = cycles[0]
        return memo, choice

    def root_values(self, cols: Any = slice(None)) -> list[tuple[RootMove, np.ndarray]]:
        """Each first move with its per-seed value vector over `cols`."""
        assert self.values is not None and self.root is not None
        self._backup(cols)
        V = self.values[:, cols]
        out = []
        for mv in self.root_moves():
            if mv.kind == "endturn":
                assert self.root.endturn is not None
                out.append((mv, V[self.root.endturn]))
            else:
                out.append((mv, self._edge_val(self._root_edge[mv.key])))
        return out

    def root_moves(self) -> list[RootMove]:
        """The root's distinct first moves (EndTurn first when it is legal)."""
        import sts2_rs

        if getattr(self, "_root_moves", None) is not None:
            return self._root_moves
        assert self.root is not None
        moves: list[RootMove] = []
        self._root_edge: dict[tuple, Edge] = {}
        if self.root.endturn is not None:
            moves.append(RootMove(("end_turn",), "end_turn", {"action": "end_turn"}, "endturn"))
        d = sts2_rs.state_to_dict(self.root_state)
        for e in self.root.edges:
            ident = e.ident or action_identity(self.root_state, e.action, d)
            # A deterministic move (turn-ending ones included) is the position it
            # reaches; a chance move is what it plays (`move_key` agrees).
            key = ("obs", e.obs) if e.obs is not None else ("id", ident)
            label = identity_label(ident) + ("!" if e.kind == "terminal" else "")
            moves.append(RootMove(key, label, e.action, e.kind))
            self._root_edge[key] = e
        self._root_moves = moves
        return moves

    def best_line(self) -> list[PlannedStep]:
        """The argmax plan (all seeds) up to its first EndTurn / chance / end."""
        assert self.root is not None
        _, choice = self._backup(slice(None))
        steps: list[PlannedStep] = []
        nd = self.root
        while True:
            c = choice[id(nd)]
            if c == ENDTURN:
                if nd.endturn is None:  # a stuck node: nothing to play
                    break
                steps.append(PlannedStep({"action": "end_turn"}, "endturn", nd.obs))
                break
            e = nd.edges[c]
            steps.append(PlannedStep(e.action, e.kind, nd.obs, e.obs, e.ident))
            if e.kind != "det":
                break
            assert e.node is not None
            nd = e.node
        # Every step also carries what it plays, so it can be matched by
        # identity when the observation route fails (`map_step`).
        s = self.root_state.clone()
        for st in steps:
            if st.ident is None:
                st.ident = action_identity(s, st.action)
            if st.kind == "det":
                s.apply(st.action)
        return steps


def near_best_class(
    moves: Sequence[tuple[RootMove, np.ndarray]],
) -> tuple[list[tuple], dict[tuple, dict[str, float]], tuple]:
    """The first moves within one paired SE of the top move.

    Args:
        moves: `Plan.root_values` output.

    Returns:
        `(class keys, per-move {"mean", "gap", "se"}, top key)`. `gap` is
        `mean(top) - mean(move)` and `se` its paired standard error over the
        seeds; a move is in the class iff `gap <= se`.
    """
    means = [float(v.mean()) for _, v in moves]
    top_i = int(np.argmax(means))
    top_v = moves[top_i][1]
    stats: dict[tuple, dict[str, float]] = {}
    cls: list[tuple] = []
    for (mv, v), m in zip(moves, means, strict=True):
        diff = (top_v - v).astype(np.float64)
        gap = float(diff.mean())
        se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
        stats[mv.key] = {"mean": m, "gap": gap, "se": se}
        if gap <= se:
            cls.append(mv.key)
    return cls, stats, moves[top_i][0].key


def _same_choice(y: Any, a: dict[str, Any], ident: tuple) -> dict[str, Any]:
    """`a` as it is spelled in `y`, a re-seeded view of the state `a` was legal in.

    `belief_view` keeps the hand and the potion belt in place, so a play or a
    potion keeps its index. A choice lists its options in the state's own
    pile order, so the chosen card is found again by content.
    """
    if a["action"] != "choose":
        return a
    import sts2_rs

    d = sts2_rs.state_to_dict(y)
    for z in y.legal_actions():
        if z["action"] == "choose" and action_identity(y, z, d) == ident:
            return z
    raise MappingError(f"the re-seeded state lost the choice {ident!r}")


class Planner:
    """Builds and evaluates `Plan`s.

    Args:
        n: Playouts per leaf (the CRN seed count).
        k_chance: Re-planned outcomes per first chance edge of a turn.
        leaf_cap: Playout leaves above which the planner declines.
        node_cap: Decision nodes above which the planner declines.
        depth: `values_many` playout depth override (None: the deployed one).
    """

    def __init__(
        self,
        n: int = N_DEFAULT,
        k_chance: int = K_CHANCE,
        leaf_cap: int = LEAF_CAP,
        node_cap: int = NODE_CAP,
        depth: int | None = None,
    ) -> None:
        self.n = n
        self.k_chance = k_chance
        self.leaf_cap = leaf_cap
        self.node_cap = node_cap
        self.depth = depth

    # -- enumeration --

    def enumerate(self, state: Any, plan_seed: int) -> Plan:
        """Enumerate the turn from `state.belief_view(plan_seed)`.

        `plan.declined` is set (and nothing is evaluated) when the turn is
        wider than the caps.
        """
        b = state.belief_view(plan_seed)
        plan = Plan(b, plan_seed, self.n, self.k_chance)
        t0 = time.perf_counter()
        try:
            plan.root = self._expand(plan, b, 0)
        except PlannerDeclined as e:
            plan.declined = str(e)
        plan.stats["enum_s"] = time.perf_counter() - t0
        return plan

    def _add_leaf(self, plan: Plan, key: tuple, leaf: Leaf) -> int:
        got = plan.leaf_index.get(key)
        if got is not None:
            return got
        plan.leaves.append(leaf)
        plan.leaf_index[key] = len(plan.leaves) - 1
        if leaf.kind != "terminal":
            plan.n_playout_leaves += 1
            if plan.n_playout_leaves > self.leaf_cap:
                raise PlannerDeclined(f"leaf_cap: more than {self.leaf_cap} playout leaves")
        return len(plan.leaves) - 1

    def _rollout_leaf(self, plan: Plan, x: Any, obs: int, a: dict, ident: tuple) -> int:
        return self._add_leaf(plan, ("rollout", obs, ident), Leaf("rollout", x.clone(), a, ident))

    def _expand(self, plan: Plan, x: Any, depth_c: int, obs: int | None = None) -> DNode:
        import sts2_rs

        if obs is None:
            obs = obs_key(x)
        nd = plan.nodes.get((obs, depth_c))
        if nd is not None:
            return nd
        nd = DNode(obs, depth_c)
        plan.nodes[obs, depth_c] = nd
        plan.stats["nodes"] += 1
        if plan.stats["nodes"] > self.node_cap:
            raise PlannerDeclined(f"node_cap: more than {self.node_cap} decision nodes")
        fp = x.rng_fingerprint()
        seen_obs: set[int] = set()
        seen_ident: set[tuple] = set()
        d: dict | None = None
        for a in x.legal_actions():
            if a["action"] == "end_turn":
                nd.endturn = self._add_leaf(plan, ("endturn", obs), Leaf("endturn", x.clone()))
                continue
            c = x.clone()
            res = c.apply_step(a)
            plan.stats["applies"] += 1
            if res == "invalid_target":
                continue
            if c.rng_fingerprint() != fp:
                if d is None:
                    d = sts2_rs.state_to_dict(x)
                ident = action_identity(x, a, d)
                if ident in seen_ident:
                    continue
                seen_ident.add(ident)
                plan.stats["chance_edges"] += 1
                if depth_c >= 1:
                    leaf = self._rollout_leaf(plan, x, obs, a, ident)
                    nd.edges.append(Edge(a, "rollout", ident=ident, leaf=leaf))
                    continue
                edge = self._chance_edge(plan, x, obs, a, ident)
                nd.edges.append(edge)
                continue
            dc = sts2_rs.state_to_dict(c)
            cobs = obs_key(c, dc)
            if cobs in seen_obs:
                continue
            seen_obs.add(cobs)
            plan.stats["det_edges"] += 1
            if c.terminal_value()[0]:
                leaf = self._add_leaf(plan, ("terminal", cobs), Leaf("terminal", c))
                nd.edges.append(Edge(a, "terminal", obs=cobs, leaf=leaf))
                continue
            if int(dc["round"]) != plan.round0:
                # Deterministic, yet the turn is over: score the playout.
                if d is None:
                    d = sts2_rs.state_to_dict(x)
                ident = action_identity(x, a, d)
                leaf = self._rollout_leaf(plan, x, obs, a, ident)
                nd.edges.append(Edge(a, "rollout", obs=cobs, ident=ident, leaf=leaf))
                continue
            child = self._expand(plan, c, depth_c, cobs)
            nd.edges.append(Edge(a, "det", obs=cobs, node=child))
        if nd.endturn is None and not nd.edges:
            nd.stuck = self._add_leaf(plan, ("stuck", obs), Leaf("rollout", x.clone(), None))
        return nd

    def _chance_edge(self, plan: Plan, x: Any, obs: int, a: dict, ident: tuple) -> Edge:
        import sts2_rs

        edge = Edge(a, "chance", ident=ident)
        for i in range(plan.k_chance):
            y = x.belief_view(stable_u64(plan.plan_seed, "chance", obs, ident, i))
            a_y = _same_choice(y, a, ident)
            y.apply(a_y)
            plan.stats["chance_outcomes"] += 1
            dy = sts2_rs.state_to_dict(y)
            yobs = obs_key(y, dy)
            if y.terminal_value()[0]:
                edge.outcomes.append(self._add_leaf(plan, ("terminal", yobs), Leaf("terminal", y)))
                continue
            if int(dy["round"]) != plan.round0:
                # The chance action ended the turn: a rollout leaf for the edge.
                plan.stats["turn_ended_outcomes"] += 1
                leaf = self._rollout_leaf(plan, x, obs, a, ident)
                return Edge(a, "rollout", ident=ident, leaf=leaf)
            edge.outcomes.append(self._expand(plan, y, 1, yobs))
        return edge

    # -- evaluation --

    def evaluate(self, plan: Plan) -> None:
        """Fill `plan.values` (leaves x seeds) with playout values."""
        import sts2_rs

        t0 = time.perf_counter()
        L, n = len(plan.leaves), plan.n
        vals = np.zeros((L, n), dtype=np.float32)
        kw: dict[str, Any] = {} if self.depth is None else {"depth": self.depth}
        endturn = [i for i, leaf in enumerate(plan.leaves) if leaf.kind == "endturn"]
        rollout = [i for i, leaf in enumerate(plan.leaves) if leaf.kind == "rollout"]
        terminal = [i for i, leaf in enumerate(plan.leaves) if leaf.kind == "terminal"]

        def run(rows: list[int], make_state: Any, mode: str) -> None:
            jobs: list[tuple[int, int, Any]] = []

            def flush() -> None:
                if not jobs:
                    return
                out = sts2_rs.values_many(
                    [s for _, _, s in jobs], [plan.seeds[j] for _, j, _ in jobs], mode=mode, **kw
                )
                for (r, j, _), o in zip(jobs, out, strict=True):
                    vals[r, j] = o["search_value"]
                jobs.clear()

            for r in rows:
                for j in range(n):
                    jobs.append((r, j, make_state(plan.leaves[r], plan.seeds[j])))
                    if len(jobs) >= VALUES_CHUNK:
                        flush()
            flush()

        run(endturn, lambda leaf, s: leaf.state, "belief_endturn")

        def forced(leaf: Leaf, s: int) -> Any:
            v = leaf.state.belief_view(s)
            if leaf.action is not None:
                assert leaf.ident is not None
                v.apply(_same_choice(v, leaf.action, leaf.ident))
            return v

        run(rollout, forced, "raw")
        if terminal:
            out = sts2_rs.values_many(
                [plan.leaves[i].state for i in terminal], [0] * len(terminal), mode="belief_endturn"
            )
            for i, o in zip(terminal, out, strict=True):
                vals[i, :] = o["search_value"]
        plan.values = vals
        plan.stats["eval_s"] = time.perf_counter() - t0
        plan.stats["playouts"] = n * (len(endturn) + len(rollout))

    def plan(self, state: Any, plan_seed: int) -> Plan:
        """Enumerate and, unless declined, evaluate one planning point."""
        plan = self.enumerate(state, plan_seed)
        if plan.declined is None:
            self.evaluate(plan)
        return plan


def det_dag_depth(view: Any, node_cap: int = 5000) -> int:
    """Longest run of deterministic non-EndTurn actions from `view` (a belief view).

    The turn DAG with chance edges cut, merged on `observation_hash`; the
    selection filter of the P6-0 positions ("DAG depth >= 2"). Returns
    `node_cap` when the DAG has more nodes than that. It keeps the engine
    hash (not `obs_key`) so the drawn P6-0 position set stays as drawn: a
    self-buff potion drink is a self-loop here and adds no depth.
    """
    memo: dict[int, int] = {}

    def walk(x: Any) -> int:
        obs = x.observation_hash()
        got = memo.get(obs)
        if got is not None:
            return got
        if len(memo) >= node_cap:
            raise PlannerDeclined("node_cap")
        memo[obs] = 0
        fp = x.rng_fingerprint()
        best = 0
        for a in x.legal_actions():
            if a["action"] == "end_turn":
                continue
            c = x.clone()
            if c.apply_step(a) == "invalid_target" or c.rng_fingerprint() != fp:
                continue
            best = max(best, 1 if c.terminal_value()[0] else 1 + walk(c))
        memo[obs] = best
        return best

    try:
        return walk(view)
    except PlannerDeclined:
        return node_cap


# ── matching moves across states ─────────────────────────────────────────


def move_key(view: Any, action: dict[str, Any]) -> tuple:
    """The `RootMove.key` of `action` (legal in the belief view `view`)."""
    if action["action"] == "end_turn":
        return ("end_turn",)
    c = view.clone()
    fp = view.rng_fingerprint()
    c.apply_step(action)
    if c.rng_fingerprint() != fp:
        return ("id", action_identity(view, action))
    return ("obs", obs_key(c))


def map_step(step: PlannedStep, state: Any) -> dict[str, Any]:
    """The legal action of `state` that plays `step`, matched by identity.

    `state` (the live combat, raw or a belief view) must be the position the
    step was planned in (the same observation); its hand and pile ORDER may
    differ from the planner's. A deterministic step is matched by the
    observation it leads to, a chance step by `action_identity`. The result is
    spelled in `state`'s own indices.

    Raises:
        MappingError: `state` is not the planned position, or no legal action
            plays the step.
    """
    import sts2_rs

    if obs_key(state) != step.pre_obs:
        raise MappingError("the state is not the position the step was planned in")
    legal = state.legal_actions()
    if step.kind == "endturn":
        for a in legal:
            if a["action"] == "end_turn":
                return a
        raise MappingError("end_turn is not legal")
    want = step.action["action"]
    if step.post_obs is not None:
        # Deterministic in the planner's honest view, so it is deterministic
        # here too and lands on the same observation.
        for a in legal:
            if a["action"] != want:
                continue
            c = state.clone()
            c.apply_step(a)
            if obs_key(c) == step.post_obs:
                return a
        if step.ident is None:
            raise MappingError(f"no legal action reaches the planned position ({step.action!r})")
        # An open card choice hashes its options in listing order, which
        # follows the hand's order; fall back to what the step plays.
    d = sts2_rs.state_to_dict(state)
    for a in legal:
        if a["action"] == want and action_identity(state, a, d) == step.ident:
            return a
    raise MappingError(f"no legal action plays {step.ident!r}")


def to_view_action(state: Any, view: Any, action: dict[str, Any]) -> dict[str, Any]:
    """`action` (legal in `state`) as the legal action of `view` that plays the same card.

    `view` is a belief view of `state`; matched by `action_identity`.

    Raises:
        MappingError: No legal action of `view` plays it.
    """
    import sts2_rs

    if action["action"] == "end_turn":
        return legal_match(view, ("end_turn",))
    ident = action_identity(state, action)
    d = sts2_rs.state_to_dict(view)
    for a in view.legal_actions():
        if a["action"] == action["action"] and action_identity(view, a, d) == ident:
            return a
    raise MappingError(f"the view has no legal action playing {ident!r}")


# ── the bench agent (arm G-C) ────────────────────────────────────────────


class OracleAgent:
    """`MctsCombatAgent.step`-compatible agent that plays the oracle's plans.

    At a planning point (a turn start, or the first decision after a chance
    event) it plans, plays the argmax plan's first action and queues the rest
    of its deterministic prefix; a queued action is played only while the
    live observation is the one the plan expected. A forced move (one legal
    action) is taken without planning. A declined plan falls back to the
    deployed `MctsCombatAgent` for that decision.
    """

    def __init__(
        self,
        seed: int,
        n: int = N_DEFAULT,
        k_chance: int = K_CHANCE,
        leaf_cap: int = LEAF_CAP,
        sims: int = 1000,
        boss_sims: int = 4000,
    ) -> None:
        import sts2_rs

        self.seed = int(seed)
        self.planner = Planner(n=n, k_chance=k_chance, leaf_cap=leaf_cap)
        self.fallback = sts2_rs.MctsCombatAgent(seed=seed, sims=sims, boss_sims=boss_sims)
        self.queue: list[PlannedStep] = []
        self.stats: dict[str, Any] = {
            "plans": 0,
            "declined": 0,
            "queued_played": 0,
            "queue_broken": 0,
            "forced": 0,
            "playouts": 0,
            "leaves": 0,
            "max_leaves": 0,
        }

    def planner_stats(self) -> dict[str, Any]:
        return dict(self.stats)

    def combat_budget_spent(self) -> tuple[int, int, int, int]:
        """`(planned decisions, fallback decisions, 0, playouts + fallback sims)`."""
        fb = self.fallback.combat_budget_spent()
        return (self.stats["plans"], int(fb[0] + fb[1]), 0, self.stats["playouts"] + int(fb[3]))

    def step(self, rm: Any) -> dict[str, Any] | None:
        acts = rm.available_actions()
        if not acts:
            return None
        cs = rm.combat_state()
        if rm.phase() != "combat" or cs is None:
            return self.fallback.step(rm)
        step = None
        if self.queue:
            nxt = self.queue.pop(0)
            if obs_key(cs) == nxt.pre_obs:
                step = nxt
            else:
                self.stats["queue_broken"] += 1
                self.queue.clear()
        if len(acts) == 1:
            # A forced move (a one-option choice, say) may be the planned one.
            self.stats["forced"] += 1
            if step is None:
                self.queue.clear()
            return self._take(rm, acts[0], None)
        if step is not None:
            self.stats["queued_played"] += 1
        search: dict[str, Any] | None = None
        if step is None:
            plan = self.planner.plan(
                cs, stable_u64(self.seed, "plan", self.stats["plans"]) & MASK63
            )
            self.stats["plans"] += 1
            if plan.declined is not None:
                self.stats["declined"] += 1
                self.queue.clear()
                return self.fallback.step(rm)
            self.stats["playouts"] += plan.stats["playouts"]
            self.stats["leaves"] += plan.playout_leaves()
            self.stats["max_leaves"] = max(self.stats["max_leaves"], plan.playout_leaves())
            line = plan.best_line()
            if not line:
                self.stats["declined"] += 1
                return self.fallback.step(rm)
            step, self.queue = line[0], line[1:]
            search = {
                "actions": [
                    {"action": mv.action, "label": mv.label, "visits": plan.n, "q": float(v.mean())}
                    for mv, v in plan.root_values()
                ],
                "recommended": step.action,
            }
        action = map_step(step, cs)
        rm_action = combat_to_rm(action)
        if rm_action["action"] != "combat_resolve_choice_selection" and rm_action not in acts:
            raise MappingError(f"{rm_action!r} is not an available run action")
        if search is None:
            search = {"actions": [{"action": action, "visits": 1}], "recommended": action}
        return self._take(rm, rm_action, search)

    def _take(
        self, rm: Any, action: dict[str, Any], search: dict[str, Any] | None
    ) -> dict[str, Any]:
        result = rm.take_action(action)
        return {
            "phase": "combat",
            "is_combat": True,
            "over": rm.is_over(),
            "won": bool(rm.player_won),
            "action": action,
            "combat_search": search,
            "result": result,
        }


def make(
    seed: int,
    n: int = N_DEFAULT,
    k_chance: int = K_CHANCE,
    leaf_cap: int = LEAF_CAP,
    sims: int = 1000,
    boss_sims: int = 4000,
) -> OracleAgent:
    """The `ppo.combat_bench run --agent ppo.combat_bench.oracle:make` factory."""
    return OracleAgent(
        seed, n=n, k_chance=k_chance, leaf_cap=leaf_cap, sims=sims, boss_sims=boss_sims
    )
