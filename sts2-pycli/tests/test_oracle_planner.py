"""GH #830 deep bets G-B: the oracle turn planner (`ppo.combat_bench.oracle`).

What P6-0 relies on: plays of identical card copies are one move, the
planner's output depends only on what the player can see, its chance edges
are the deployed search's chance nodes, and a planned action is played on the
live `RunManager` by card identity, not by index.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import sts2_rs
from ppo.combat_bench import agents
from ppo.combat_bench.oracle import (
    MappingError,
    OracleAgent,
    PlannedStep,
    Planner,
    RootMove,
    _same_choice,
    action_identity,
    card_sig,
    map_step,
    move_key,
    near_best_class,
    obs_key,
)
from ppo.combat_bench.p6_probe import plan_signature

# A hand with both kinds of edge: Pommel Strike / Shrug It Off draw from a
# heterogeneous hidden pile (chance), Burning Pact opens a choice (a
# deterministic play) whose resolution draws (chance), Anger / Strike / the
# potion are deterministic. A five-card turn of this kind is ~1,100 leaves
# already (draws re-planned four ways, second draws as rollout leaves).
DRAW_DECK = [
    "POMMEL_STRIKE",
    "SHRUG_IT_OFF",
    "BURNING_PACT",
    "STRIKE_IRONCLAD",
    "DEFEND_IRONCLAD",
    "DEFEND_IRONCLAD",
    "BASH",
    "INFLAME",
    "IRON_WAVE",
    "ANGER",
]


@pytest.fixture(autouse=True)
def engine_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No inherited engine seam leaks in; one rayon thread."""
    for key in list(os.environ):
        if agents.is_engine_env(key):
            monkeypatch.delenv(key)
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")


def _draw_state(seed: int = 1) -> sts2_rs.CombatState:
    """Opening of `DRAW_DECK` vs a Nibbit; seed 1 deals Pommel Strike, Shrug
    It Off, Anger, Strike, Burning Pact."""
    s = sts2_rs.build_scenario_state(
        seed,
        "Ironclad",
        DRAW_DECK,
        60,
        80,
        ["BURNING_BLOOD"],
        "NIBBITS_WEAK",
        ascension=10,
        potions=["BLOCK_POTION"],
    )
    hand = [c["id"] for c in sts2_rs.state_to_dict(s)["deck"]["hand"]]
    assert {"POMMEL_STRIKE", "SHRUG_IT_OFF", "BURNING_PACT", "ANGER"} <= set(hand), hand
    return s


def _hand(s: sts2_rs.CombatState) -> list[str]:
    return [c["id"] for c in sts2_rs.state_to_dict(s)["deck"]["hand"]]


def _norm_name(name: str) -> str:
    return name.replace("_", "").replace("+", "").lower()


# ── dedup ────────────────────────────────────────────────────────────────


def _reachable(view: sts2_rs.CombatState) -> tuple[set[int], set[int]]:
    """Every position deterministic plays reach from `view`, without any
    merging: (observation hashes, search hashes)."""
    obs: set[int] = set()
    search: set[int] = set()
    seen: set[int] = set()

    def walk(s: sts2_rs.CombatState) -> None:
        if s.ordered_hash() in seen:
            return
        seen.add(s.ordered_hash())
        obs.add(s.observation_hash())
        search.add(s.search_hash())
        fp = s.rng_fingerprint()
        for a in s.legal_actions():
            if a["action"] == "end_turn":
                continue
            c = s.clone()
            c.apply(a)
            if c.rng_fingerprint() == fp and not c.terminal_value()[0]:
                walk(c)

    walk(view)
    return obs, search


def test_dedup_collapses_plays_of_duplicate_copies() -> None:
    s = sts2_rs.CombatState(seed=1, encounter="Nibbit")
    assert _hand(s).count("STRIKE_IRONCLAD") == 4
    plan = Planner(n=2).enumerate(s, 3)
    assert plan.declined is None
    labels = [mv.label for mv in plan.root_moves()]
    # Four Strike plays on the one enemy are one move; so is everything below.
    assert sorted(labels) == ["BASH->0", "STRIKE_IRONCLAD->0", "end_turn"]
    strikes = [
        a
        for a in plan.root_state.legal_actions()
        if a["action"] == "play" and _hand(s)[a["card_index"]] == "STRIKE_IRONCLAD"
    ]
    assert len(strikes) == 4
    assert len({move_key(plan.root_state, a) for a in strikes}) == 1
    # One decision node per observable position: what `observation_hash`
    # merges, and far fewer than the uid-keyed `search_hash` would keep.
    obs, search = _reachable(plan.root_state)
    assert plan.stats["nodes"] == len(obs)
    assert plan.leaf_counts()["endturn"] == len(obs)
    assert len(obs) * 2 < len(search)


# ── honesty ──────────────────────────────────────────────────────────────


def test_planner_output_is_identical_on_every_sampled_world() -> None:
    """The plan depends on the observation and the plan seed only."""
    for s in (_draw_state(), sts2_rs.CombatState(seed=2, encounter="ShrinkerBeetles")):
        planner = Planner(n=4)
        plan = planner.plan(s, 77)
        assert plan.declined is None
        sig = plan_signature(plan)
        for r in range(5):
            w = s.sample_world(1000 + r)
            # Control: a genuinely different world (hidden draw order).
            assert w.ordered_hash() != s.ordered_hash()
            assert plan_signature(planner.plan(w, 77)) == sig
        # Non-vacuous: the plan seed is the planner's randomness.
        assert plan_signature(planner.plan(s, 78)) != sig
    # The draw state exercises the chance path (re-planned outcomes).
    assert Planner(n=2).enumerate(_draw_state(), 77).stats["chance_outcomes"] > 0


# ── chance edges ─────────────────────────────────────────────────────────


def _search_root_kinds(view: sts2_rs.CombatState) -> dict[tuple, str]:
    tree = sts2_rs.mcts_search_tree(view, sims=400, seed=0, hidden_info="beliefdraw")
    out = {}
    for cid in tree["nodes"][0]["children"]:
        node = tree["nodes"][cid]
        assert node["visits"] > 0
        if node["kind"] == "play":
            key = ("play", _norm_name(node["label"]), node["target_index"])
        elif node["kind"] == "choose":
            key = ("choose", int(node["label"].split()[-1]))
        else:
            key = (node["kind"],)
        out[key] = node["node_kind"]
    return out


def _planner_root_kinds(state: sts2_rs.CombatState) -> dict[tuple, str]:
    plan = Planner(n=2, k_chance=1).enumerate(state, 5)
    assert plan.declined is None and plan.root is not None
    hand = sts2_rs.state_to_dict(plan.root_state)["deck"]["hand"]
    out = {}
    for e in plan.root.edges:
        a = e.action
        if a["action"] == "play":
            c = hand[a["card_index"]]
            key = ("play", _norm_name(c["id"]), a["target_index"])
        elif a["action"] == "choose":
            key = ("choose", a["selection"][0])
        else:
            key = (a["action"],)
        out[key] = "chance" if e.kind in ("chance", "rollout") and e.obs is None else "decision"
    return out


def test_chance_edges_are_the_searchs_chance_nodes() -> None:
    s = _draw_state()
    pact = _hand(s).index("BURNING_PACT")
    chose = s.belief_view(9)
    chose.apply({"action": "play", "card_index": pact, "target_index": None})
    assert chose.pending_choice() is not None
    states = [s, chose, sts2_rs.CombatState(seed=3, encounter="Mawler")]
    n_chance = n_det = 0
    for st in states:
        view = st.belief_view(11)
        mine = _planner_root_kinds(view)
        search = _search_root_kinds(view)
        search.pop(("end_turn",), None)
        # Every distinct root move of the search is one of the planner's,
        # with the same kind (the planner also splits identical-id cards
        # whose content differs, which the search's edges merge).
        assert set(search) == set(mine), set(search) ^ set(mine)
        assert mine == search
        n_chance += sum(v == "chance" for v in mine.values())
        n_det += sum(v == "decision" for v in mine.values())
    # Non-vacuous: both kinds occur (draw cards, choice-then-draw, plain plays).
    assert n_chance >= 4 and n_det >= 4


# ── the near-best class ──────────────────────────────────────────────────


@pytest.mark.parametrize("potion", ["LUCKY_TONIC", "STRENGTH_POTION", "STABLE_SERUM"])
def test_a_self_buff_potion_is_a_move_not_a_self_loop(potion: str) -> None:
    """The engine hashes miss the belt and creature-path player powers, so
    a self-buff drink left `observation_hash` unchanged: a self-loop that
    sent the P6-0 backup into a RecursionError on 26 of 300 positions.
    `obs_key` sees the drink; the plan has it as a move of its own."""
    deck = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]
    s = sts2_rs.build_scenario_state(
        1,
        "Ironclad",
        deck,
        60,
        80,
        ["BURNING_BLOOD"],
        "NIBBITS_WEAK",
        ascension=10,
        potions=[potion],
    )
    plan = Planner(n=8).plan(s, 5)
    b = plan.root_state
    drink = next(a for a in b.legal_actions() if a["action"] in ("drink_potion", "potion"))
    c = b.clone()
    c.apply_step(drink)
    assert obs_key(c) != obs_key(b)
    assert plan.declined is None
    rv = plan.root_values()  # no RecursionError
    assert plan.stats["cycle_edges"] == 0
    keys = [mv.key for mv, _ in rv]
    assert move_key(b, drink) in keys
    assert move_key(b, drink) != ("obs", obs_key(b))
    assert len(set(keys)) == len(keys)


def test_near_best_class_uses_the_paired_se() -> None:
    def mv(k: str) -> RootMove:
        return RootMove((k,), k, {"action": k}, "det")

    top = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    rows = [
        (mv("top"), top),
        (mv("noisy"), np.array([0.9, 1.1, 0.95, 1.05], dtype=np.float32)),  # gap 0
        (mv("worse"), top - 0.05),  # constant gap, paired SE 0: out
        (mv("far"), np.zeros(4, dtype=np.float32)),
    ]
    cls, stats, best = near_best_class(rows)
    assert best == ("top",)
    assert set(cls) == {("top",), ("noisy",)}
    assert stats["worse",]["se"] == pytest.approx(0.0, abs=1e-6)


# ── mapping by identity ──────────────────────────────────────────────────


def _permuted_pair(s: sts2_rs.CombatState) -> tuple[sts2_rs.CombatState, sts2_rs.CombatState]:
    """The same position twice, the second with its hand in reverse order."""
    d = sts2_rs.state_to_dict(s)
    rev = json.loads(json.dumps(d))
    rev["deck"]["hand"] = list(reversed(rev["deck"]["hand"]))
    a = sts2_rs.state_from_dict(json.loads(json.dumps(d)))
    b = sts2_rs.state_from_dict(rev)
    assert a.observation_hash() == b.observation_hash()
    assert _hand(a) == list(reversed(_hand(b))) != _hand(b)
    return a, b


def _card(s: sts2_rs.CombatState, a: dict) -> tuple:
    return card_sig(sts2_rs.state_to_dict(s)["deck"]["hand"][a["card_index"]])


def test_planned_steps_map_by_card_identity_not_index() -> None:
    a, b = _permuted_pair(_draw_state())
    width = len(_hand(a))
    assert len(set(_hand(a))) == width  # distinct cards: the reversal moves all but one
    plan = Planner(n=4).plan(a, 21)
    assert plan.declined is None and plan.root is not None
    checked_det = checked_chance = 0
    for e in plan.root.edges:
        if e.action["action"] != "play":
            continue
        ident = action_identity(plan.root_state, e.action)
        step = PlannedStep(e.action, e.kind, plan.root.obs, e.obs, ident)
        got = map_step(step, b)
        # The same card, at its own index in the reversed hand.
        assert _card(b, got) == _card(plan.root_state, e.action)
        assert got["target_index"] == e.action["target_index"]
        assert got["card_index"] == width - 1 - e.action["card_index"]
        if e.obs is not None:
            c = b.clone()
            c.apply(got)
            if c.pending_choice() is None:
                # (An open choice lists its options in hand order, which the
                # observation hash keeps.)
                assert obs_key(c) == e.obs
            checked_det += 1
        else:
            checked_chance += 1
    assert checked_det >= 2 and checked_chance >= 2
    # A position the step was not planned in is refused.
    other = b.clone()
    other.apply({"action": "end_turn"})
    step = PlannedStep(plan.root.edges[0].action, "det", plan.root.obs, plan.root.edges[0].obs)
    with pytest.raises(MappingError, match="not the position"):
        map_step(step, other)


def test_a_choice_is_found_again_by_the_chosen_card() -> None:
    """Burning Pact's exhaust choice (then draw 2: a chance edge) lists the
    hand in hand order; the planner re-finds the chosen card by content."""
    a, b = _permuted_pair(_draw_state())
    sa, sb = a.belief_view(4), b.belief_view(4)
    for st in (sa, sb):
        st.apply(
            {"action": "play", "card_index": _hand(st).index("BURNING_PACT"), "target_index": None}
        )
    ids = [[c["id"] for c in st.pending_choice()["cards"]] for st in (sa, sb)]
    assert ids[0] == list(reversed(ids[1])) != ids[1]
    moved = 0
    for z in sa.legal_actions():
        assert z["action"] == "choose"
        ident = action_identity(sa, z)
        assert move_key(sa, z) == ("id", ident)  # exhaust, then draw: chance
        got = _same_choice(sb, z, ident)
        assert action_identity(sb, got) == ident
        moved += got["selection"] != z["selection"]
    assert moved >= 2


def test_a_draw_pile_choice_maps_from_the_planners_view_to_the_live_state() -> None:
    """Secret Weapon picks an Attack from the draw pile. The live (raw) state
    and the planner's belief view order the hidden pile differently, so the
    options point at different pile indices; the step lands on the same card."""
    deck = [
        "SECRET_WEAPON",
        "STRIKE_IRONCLAD",
        "BASH",
        "POMMEL_STRIKE",
        "TWIN_STRIKE",
        "HEADBUTT",
        "DEFEND_IRONCLAD",
        "DEFEND_IRONCLAD",
        "SHRUG_IT_OFF",
        "IRON_WAVE",
        "CLASH",
        "ANGER",
    ]
    s = sts2_rs.build_scenario_state(
        1, "Ironclad", deck, 60, 80, ["BURNING_BLOOD"], "CULTISTS_NORMAL", ascension=10
    )
    play = {"action": "play", "card_index": _hand(s).index("SECRET_WEAPON"), "target_index": None}
    live, view = s.clone(), s.belief_view(5)
    live.apply(play)
    view.apply(play)
    assert obs_key(live) == obs_key(view)
    idx = [[o["index"] for o in st.pending_choice()["cards"]] for st in (live, view)]
    assert idx[0] != idx[1]
    n = 0
    for z in view.legal_actions():
        c = view.clone()
        c.apply(z)
        step = PlannedStep(z, "det", obs_key(view), obs_key(c))
        step.ident = action_identity(view, z)
        got = map_step(step, live)
        assert action_identity(live, got) == step.ident
        r = live.clone()
        r.apply(got)
        assert obs_key(r) == step.post_obs
        n += 1
    assert n >= 3


def test_oracle_agent_plays_a_fight_through_the_run_manager() -> None:
    """Every planned action is legal on the live run, and the planned
    deterministic prefix lands where the plan said it would."""
    state = sts2_rs.CombatState(seed=4, encounter="Nibbit")
    rm = sts2_rs.RunManager(seed=4, character="Ironclad", ascension_level=0)
    rm.debug_enter_combat(state, "monster")
    agent = OracleAgent(seed=4, n=4)
    steps = 0
    while rm.phase() == "combat" and not rm.is_over() and steps < 200:
        before = rm.available_actions()
        rec = agent.step(rm)
        assert rec is not None
        assert (
            rec["action"] in before or rec["action"]["action"] == "combat_resolve_choice_selection"
        )
        steps += 1
    assert rm.is_over() or rm.phase() != "combat"
    st = agent.planner_stats()
    assert st["plans"] >= 2
    assert st["queued_played"] >= 1  # multi-action plans were executed
    assert st["queue_broken"] == 0
    assert agent.combat_budget_spent()[3] >= st["playouts"] > 0
