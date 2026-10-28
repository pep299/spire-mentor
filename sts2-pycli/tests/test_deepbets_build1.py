"""GH #830 deep bets build 1: the Python surface of the engine additions.

`values_many` (recording-free search playouts), `CombatState.belief_view` /
`sample_world` / `search_hash` / `observation_hash` / `rng_fingerprint`, and
`mcts_recommend`'s `leaf_playouts`. The engine-side semantics (in-search identity of a `raw`
playout, the k-playout mean, the fingerprint against the search's own chance
nodes) are pinned in `sts2-mcts/src/leaf_playouts_tests.rs`; these check the
bindings expose exactly those functions.
"""

from __future__ import annotations

import json
import os

import pytest
import sts2_rs
from ppo.combat_bench import agents

END_TURN = {"action": "end_turn"}


@pytest.fixture(autouse=True)
def engine_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No inherited engine seam (`STS2_RUNAI_*`, `STS2_ROLLOUT_*`) leaks in."""
    for key in list(os.environ):
        if agents.is_engine_env(key):
            monkeypatch.delenv(key)


def _states() -> list[sts2_rs.CombatState]:
    """Opening positions (Ironclad starter deck) and a few plays later."""
    out = []
    for seed, enc in ((1, "Nibbit"), (2, "ShrinkerBeetles"), (3, "Mawler")):
        s = sts2_rs.CombatState(seed=seed, encounter=enc)
        out.append(s.clone())
        for _ in range(2):
            plays = [a for a in s.legal_actions() if a["action"] == "play"]
            if not plays:
                break
            s.apply(plays[0])
        out.append(s.clone())
    return out


def _vals(rows: list[dict]) -> list[tuple]:
    return [(r["search_value"], r["won"], r["hp"], r["terminal"]) for r in rows]


def test_values_many_rows_and_determinism() -> None:
    states = _states()
    seeds = list(range(len(states)))
    rows = sts2_rs.values_many(states, seeds)
    assert len(rows) == len(states)
    for r in rows:
        # `search_value`: the key `LearnedPlayout.play_many` uses for the same number.
        assert set(r) == {"search_value", "won", "hp", "terminal"}
        assert -1.5 <= r["search_value"] <= 1.5
        # Depth 200 from a healthy opening: every playout finishes the fight.
        assert r["terminal"]
        assert r["won"] == (r["search_value"] > 0.5)
    assert _vals(sts2_rs.values_many(states, seeds)) == _vals(rows)
    # The seed is the playout's randomness: other seeds give other values.
    many = sts2_rs.values_many([states[0]] * 16, list(range(16)))
    assert len({r["search_value"] for r in many}) > 1


def test_values_many_belief_is_raw_on_the_belief_view() -> None:
    states = _states()
    for i, s in enumerate(states):
        seed = 100 + i
        belief = sts2_rs.values_many([s], [seed], mode="belief")
        raw_on_view = sts2_rs.values_many([s.belief_view(seed)], [seed], mode="raw")
        assert _vals(belief) == _vals(raw_on_view)


def test_values_many_belief_endturn_is_the_afterstate() -> None:
    for i, s in enumerate(_states()):
        seed = 200 + i
        view = s.belief_view(seed)
        view.apply(END_TURN)
        assert _vals(sts2_rs.values_many([s], [seed], mode="belief_endturn")) == _vals(
            sts2_rs.values_many([view], [seed], mode="raw")
        )


def test_values_many_overrides_and_errors() -> None:
    s = _states()[0]
    # A one-action horizon cannot finish the fight: the depth-cut leaf scores it.
    short = sts2_rs.values_many([s], [0], depth=1)[0]
    assert not short["terminal"] and -1.0 <= short["search_value"] <= 1.0
    sts2_rs.values_many([s], [0], rollout_policy="heuristic")
    with pytest.raises(ValueError, match="mode must be"):
        sts2_rs.values_many([s], [0], mode="clairvoyant")
    with pytest.raises(ValueError, match="differ in length"):
        sts2_rs.values_many([s, s], [0])
    with pytest.raises(ValueError, match="rollout_policy"):
        sts2_rs.values_many([s], [0], rollout_policy="greedy")


def test_belief_view_is_idempotent_and_forgets_the_view_it_starts_from() -> None:
    for s in _states():
        view = s.belief_view(7)
        key = (view.ordered_hash(), view.state_hash(), view.rng_fingerprint())
        again = view.belief_view(7)
        assert (again.ordered_hash(), again.state_hash(), again.rng_fingerprint()) == key
        other = s.belief_view(3).belief_view(7)
        assert (other.ordered_hash(), other.state_hash(), other.rng_fingerprint()) == key
        # Non-vacuous: the seed matters.
        assert s.belief_view(8).state_hash() != view.state_hash()


def test_belief_view_of_every_sampled_world_is_one_state() -> None:
    """Honesty: what the belief view keeps is what the player can see."""
    for s in _states():
        truth = s.belief_view(4242)
        worlds = [s.sample_world(r) for r in range(6)]
        views = [w.belief_view(4242) for w in worlds]
        for v in views:
            assert v.ordered_hash() == truth.ordered_hash()
            assert v.state_hash() == truth.state_hash()
            assert v.search_hash() == truth.search_hash()
        # Control: the worlds themselves differ in draw order and RNG streams...
        assert len({w.ordered_hash() for w in worlds}) == len(worlds)
        assert len({w.state_hash() for w in worlds}) == len(worlds)
        # ...which the search's dedup hash cannot see at all.
        assert len({w.search_hash() for w in worlds}) == 1


def test_rng_fingerprint_moves_on_chance_and_not_on_deterministic_plays() -> None:
    checked = 0
    for i, s in enumerate(_states()):
        view = s.belief_view(i)
        fp = view.rng_fingerprint()
        assert isinstance(fp, tuple) and len(fp) == 10
        # The starter deck's plays (Strike, Defend, Bash) consume no randomness.
        for a in view.legal_actions():
            if a["action"] != "play":
                continue
            after = view.clone()
            after.apply(a)
            assert after.rng_fingerprint() == fp, a
            checked += 1
        # Ending the turn rolls intents and draws from the hidden pile.
        ended = view.clone()
        ended.apply(END_TURN)
        assert ended.rng_fingerprint() != fp
        # The search hash ignores the cursors; it moves only with the position.
        assert view.clone().search_hash() == view.search_hash()
    assert checked >= 10


def test_mcts_recommend_leaf_playouts() -> None:
    s = sts2_rs.CombatState(seed=5, encounter="Nibbit")

    def q(**kw) -> list[tuple]:
        out = sts2_rs.mcts_recommend(s, sims=64, threads=1, seed=3, **kw)
        return [(repr(a["action"]), a["q"]) for a in out["actions"]]

    assert q() == q(leaf_playouts=1)
    assert q(leaf_playouts=4) != q()
    with pytest.raises(ValueError, match="leaf_playouts"):
        sts2_rs.mcts_recommend(s, sims=8, threads=1, leaf_playouts=0)


def test_rng_fingerprint_refuses_a_raw_state() -> None:
    """On a raw state a draw reads the materialized pile and consumes no RNG,
    so the fingerprint would miss a chance edge the deployed search makes."""
    raw = sts2_rs.CombatState(seed=1, encounter="Nibbit")
    with pytest.raises(ValueError, match="belief_view"):
        raw.rng_fingerprint()
    assert len(raw.belief_view(0).rng_fingerprint()) == 10


def _plays(view: sts2_rs.CombatState) -> dict[tuple, list[sts2_rs.CombatState]]:
    """The positions after each card play from `view`, by (card id, target)."""
    hand = view.get_state()["hand"]
    out: dict[tuple, list[sts2_rs.CombatState]] = {}
    for a in view.legal_actions():
        if a["action"] != "play":
            continue
        after = view.clone()
        after.apply(a)
        out.setdefault((hand[a["card_index"]]["id"], a.get("target_index")), []).append(after)
    return out


def test_observation_hash_merges_plays_of_identical_copies() -> None:
    plays = _plays(sts2_rs.CombatState(seed=1, encounter="Nibbit").belief_view(3))
    strikes = plays["STRIKE_IRONCLAD", 0]
    assert len(strikes) == 4
    # `search_hash` keys the piles on card uids: each copy is its own position.
    assert len({t.search_hash() for t in strikes}) == 4
    # The player sees one position.
    assert len({t.observation_hash() for t in strikes}) == 1
    # Control: different cards (this hand: Strike, Defend), and one card on
    # different targets, stay apart.
    firsts = [ts[0].observation_hash() for ts in plays.values()]
    assert len(set(firsts)) == len(firsts) >= 2
    beetles = _plays(sts2_rs.CombatState(seed=2, encounter="ShrinkerBeetles").belief_view(3))
    on_targets = {
        t: ts[0].observation_hash() for (cid, t), ts in beetles.items() if cid == "STRIKE_IRONCLAD"
    }
    assert len(on_targets) == 2 and len(set(on_targets.values())) == 2


def _observable_key(s: sts2_rs.CombatState) -> str:
    """The position with every pile a sorted multiset of card contents (card
    instance ids dropped) and no RNG: what the player can tell apart."""
    d = sts2_rs.state_to_dict(s)
    piles = {
        name: sorted(
            json.dumps({k: v for k, v in c.items() if k != "stable_idx"}, sort_keys=True)
            for c in pile
        )
        for name, pile in d["deck"].items()
        if isinstance(pile, list)
    }
    rest = {k: v for k, v in d.items() if k not in ("deck", "rng", "rng_counters")}
    return json.dumps([piles, rest], sort_keys=True, default=str)


def _turn_afterstates(view: sts2_rs.CombatState) -> list[sts2_rs.CombatState]:
    """Every position one turn of deterministic card plays reaches from `view`
    at which the turn can end (a play that moves the fingerprint is a chance
    edge and ends the walk)."""
    out: list[sts2_rs.CombatState] = []
    seen: set[int] = set()

    def walk(s: sts2_rs.CombatState) -> None:
        if s.ordered_hash() in seen:
            return
        seen.add(s.ordered_hash())
        for a in s.legal_actions():
            if a["action"] == "end_turn":
                out.append(s)
            elif a["action"] == "play":
                t = s.clone()
                fp = t.rng_fingerprint()
                t.apply(a)
                if t.rng_fingerprint() == fp and not t.terminal_value()[0]:
                    walk(t)

    walk(view)
    return out


def test_observation_hash_is_exactly_the_observable_position() -> None:
    """Over a turn's plans, `observation_hash` puts two afterstates together
    exactly when they differ only in card instance ids, and that merges most
    of what `search_hash` keeps apart."""
    for seed, enc in ((1, "Nibbit"), (2, "ShrinkerBeetles"), (3, "Mawler")):
        after = _turn_afterstates(sts2_rs.CombatState(seed=seed, encounter=enc).belief_view(5))
        by_obs: dict[str, set[int]] = {}
        by_hash: dict[int, set[str]] = {}
        for s in after:
            key, h = _observable_key(s), s.observation_hash()
            by_obs.setdefault(key, set()).add(h)
            by_hash.setdefault(h, set()).add(key)
        assert all(len(v) == 1 for v in by_obs.values()), (
            f"{enc}: one observable position, many hashes"
        )
        assert all(len(v) == 1 for v in by_hash.values()), (
            f"{enc}: one hash, many observable positions"
        )
        n_search = len({s.search_hash() for s in after})
        assert len(by_hash) * 2 < n_search, (
            f"{enc}: {len(by_hash)} observable vs {n_search} search_hash"
        )
