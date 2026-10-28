"""GH #830 G-D: the value-net state sampler, labels and data path on the engine.

A hand-made trajectory record (a starter deck plus Pommel Strike, a draw card,
against Nibbits) stands in for a pilot trajectory: specs are sampled from it,
rebuilt, hash-checked and labelled the way `value_label` does at scale.
"""

from __future__ import annotations

import collections
import os
import random

import numpy as np
import pytest
import sts2_rs
from ppo import rollout_starts
from ppo.combat_bench import agents
from ppo.deepbets import value_data as vd
from ppo.deepbets import value_label as vl

FIGHT = {
    "file": "unit.run.gz",
    "floor": 3,
    "deck": ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH", "POMMEL_STRIKE"],
    "hp_before": 70,
    "max_hp": 80,
    "relics": ["BURNING_BLOOD"],
    "encounter": "NIBBITS_WEAK",
    "ascension": 10,
    "potions": [],
    "room_type": "elite",
}
CFG = {
    "seed": 1,
    "n_desc": 2,
    "desc_min": 3,
    "desc_max": 6,
    "deep_frac": 1.0,
    "max_plans": 6,
    "plan_cap": 500,
}


@pytest.fixture(autouse=True)
def engine_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if agents.is_engine_env(key):
            monkeypatch.delenv(key)


def _record(seed: int = 7, n: int = 14) -> dict:
    """A trajectory record: first legal run action each step (plays, then EndTurn)."""
    rm = rollout_starts.build_fight(FIGHT, seed)
    actions, hashes = [], []
    while rollout_starts.in_combat(rm) and len(actions) < n:
        hashes.append(int(rm.combat_state().ordered_hash()))
        a = rm.available_actions()[0]
        rm.take_action(a)
        actions.append(a)
    return {
        "key": "unit.run.gz:3", "file": "unit.run.gz", "rep": 0, "seed": seed, "room": "elite",
        "fight": FIGHT, "actions": actions, "hashes": hashes, "searched": [True] * len(actions),
        "_idx": 0,
    }  # fmt: skip


def _turn_starts(rec: dict) -> list[int]:
    a = rec["actions"]
    return [t for t in range(len(a)) if t == 0 or a[t - 1].get("action") == "end_turn"]


def test_split_is_sha1_mod_10() -> None:
    names = [f"{i:016x}.run.gz" for i in range(2000)]
    c = collections.Counter(vd.split_of(n) for n in names)
    assert set(c) == {"test", "dev", "train"}
    assert 150 <= c["test"] <= 250 and 150 <= c["dev"] <= 250
    assert vd.split_of(names[0]) == vd.split_of(names[0])


def test_walk_record_verifies_hashes_and_stops_at_a_mismatch() -> None:
    rec = _record()
    states, verified = vd.walk_record(rec, range(len(rec["actions"])))
    assert verified == len(rec["actions"]) and len(states) == verified
    for t, cs in states.items():
        assert int(cs.ordered_hash()) == rec["hashes"][t]
    bad = dict(rec, hashes=[*rec["hashes"][:5], 0, *rec["hashes"][6:]])
    states, verified = vd.walk_record(bad, range(len(rec["actions"])))
    assert verified == 5 and set(states) == set(range(5))


def test_features_are_the_honest_view_only() -> None:
    """Every input the net reads is identical across worlds the player cannot
    tell apart (re-dealt hidden draw order, reseeded RNG)."""
    rec = _record()
    states, _ = vd.walk_record(rec, [0, 3, 6])
    for cs in states.values():
        ref = vd.features(cs, False)
        for r in range(4):
            assert vd.features(cs.sample_world(r), False) == ref
        # Control: the worlds do differ.
        assert len({cs.sample_world(r).ordered_hash() for r in range(4)}) == 4


def test_sampler_specs_rebuild_and_label() -> None:
    rec = _record()
    turns = _turn_starts(rec)
    res = vl.sample_job((rec, [0, 2], turns[:1], CFG))
    specs = res["specs"]
    kinds = collections.Counter(s["kind"] for s in specs)
    assert kinds["decision"] == 2 and kinds["child"] > 0 and kinds["endturn"] >= 1
    assert kinds["desc"] >= 2 and kinds["after"] >= 2
    assert all(s["mode"] in vd.HONEST_MODES for s in specs)
    assert {s["mode"] for s in specs if s["kind"] in ("endturn", "deep_endturn", "after")} == {
        "belief_endturn"
    }
    assert len({s["id"] for s in specs}) == len(specs)
    # Children are distinct observable positions: the opening hand's Strikes
    # (several copies) give one child.
    for gid in {s["group"] for s in specs if s["kind"] == "child"}:
        hs = [s["ohash"] for s in specs if s["group"] == gid and s["kind"] == "child"]
        assert len(hs) == len(set(hs))
    # Pommel Strike draws from the hidden pile: a chance edge; Strike is not.
    chance = {s["chance"] for s in specs if s["kind"] in ("child", "deep_child")}
    assert False in chance
    # Afterstates of one turn share the CRN group and the root plan is kept.
    after = [s for s in specs if s["kind"] == "after"]
    assert len({s["group"] for s in after}) == 1 and any(not s["path"] for s in after)
    # Every spec rebuilds to its recorded position (observation_hash checked).
    vl._RECS = [rec]
    built = vl.rebuild_specs("unused", specs)
    assert len(built) == len(specs)
    out = vl.label_job(("unused", specs[:6], 8))
    assert len(out["rows"]) == 6 and all("label" in r for r in out["rows"])
    lab = out["rows"][0]["label"]
    assert lab["n"] == 8 and lab["n_term"] == 8 and -1.5 <= lab["mean"] <= 1.5
    vl._RECS = None


def test_endturn_member_is_the_afterstate_label() -> None:
    """The sibling group's EndTurn member scores the parent in belief_endturn
    mode: the same numbers as raw playouts from belief_view(seed) + EndTurn."""
    rec = _record()
    states, _ = vd.walk_record(rec, [0])
    cs = states[0]
    seeds = vd.seeds(vl.LABEL_SALT, "x", 4)
    got, _, _ = vl.run_playouts(cs, "belief_endturn", seeds)
    want = []
    for s in seeds:
        v = cs.belief_view(s)
        v.apply({"action": "end_turn"})
        want.append(sts2_rs.values_many([v], [s], mode="raw")[0]["search_value"])
    assert np.allclose(got, want)
    with pytest.raises(AssertionError):
        vl.run_playouts(cs, "raw", seeds)


def test_enumerate_plans_dedups_on_the_observation_and_cuts_chance() -> None:
    rec = _record()
    states, _ = vd.walk_record(rec, [0])
    bv = states[0].belief_view(5)
    nodes, capped, _ = vl.enumerate_plans(bv, 10_000, random.Random(0))
    assert not capped and nodes
    hs = [int(s.observation_hash()) for s, _ in nodes]
    assert len(hs) == len(set(hs))
    fp0 = bv.rng_fingerprint()
    for s, path in nodes:
        # Deterministic: replaying the path reproduces the node, no RNG moved.
        r = bv.clone()
        for a in path:
            r.apply(a)
        assert int(r.observation_hash()) == int(s.observation_hash())
        assert s.rng_fingerprint() == fp0 or vl._terminal(s)
        assert vl._terminal(s) or any(a["action"] == "end_turn" for a in s.legal_actions())
    # The cap bounds the walk.
    _, capped_small, seen_small = vl.enumerate_plans(bv, 3, random.Random(0))
    assert capped_small and seen_small == 3


def test_descend_never_returns_a_terminal_state() -> None:
    rec = _record()
    states, _ = vd.walk_record(rec, [0])
    rng = random.Random(3)
    for pol in ("T1", "T2", "U"):
        path, s = vl.descend(states[0], rng, 8, pol)
        assert path and not vl._terminal(s)
        assert int(vd.apply_path(states[0], None, path).observation_hash()) == int(
            s.observation_hash()
        )


def test_playout_stats() -> None:
    v = np.array([1.2, -1.3, 1.1, -1.1], np.float32)
    won = np.array([True, False, True, False])
    st = vl.playout_stats(v, won, np.ones(4, bool))
    assert st["n_won"] == 2 and st["mean_win"] == pytest.approx(1.15)
    assert st["mean_loss"] == pytest.approx(-1.2) and st["mean"] == pytest.approx(-0.025)


def test_holdout_union_and_source_assertions(tmp_path) -> None:
    import json

    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    m1.write_text(json.dumps({"held_out_runs": ["a.run.gz", "b.run.gz"]}))
    m2.write_text(json.dumps({"held_out_runs": ["b.run.gz", "c.run.gz"]}))
    held, counts = vd.holdout_union([str(m1), str(m2)])
    assert held == {"a.run.gz", "b.run.gz", "c.run.gz"} and counts == {"m1.json": 2, "m2.json": 2}
    rec = _record()
    traj = tmp_path / "traj.jsonl"
    rows = [{"schema": rollout_starts.SCHEMA}]
    for f in ("a.run.gz", "z.run.gz"):
        r = {k: v for k, v in rec.items() if k != "_idx"}
        rows.append(dict(r, file=f, key=f"{f}:3"))
    traj.write_text("".join(json.dumps(r) + "\n" for r in rows))
    bank = tmp_path / "hbank.jsonl"
    bank.write_text(
        json.dumps({"file": "z.run.gz"}) + "\n" + json.dumps({"file": "a.run.gz"}) + "\n"
    )
    _, kept, meta = vd.load_source(str(traj), str(bank), [str(m1), str(m2)])
    assert [r["file"] for r in kept] == ["z.run.gz"] and meta["dropped"]["bench"] == 1
    bank.write_text(json.dumps({"file": "a.run.gz"}) + "\n")
    with pytest.raises(AssertionError, match="not human-bank"):
        vd.load_source(str(traj), str(bank), [str(m1), str(m2)])


def test_label_shards_round_trip_into_training_tensors(tmp_path) -> None:
    """FeatureShard -> load_label_data: per-zone token counts survive, the
    vocabulary comes from the train split only, and the net runs on it."""
    torch = pytest.importorskip("torch")
    from ppo.deepbets import value_train as vt

    rec = _record()
    res = vl.sample_job((rec, [0, 2], [], CFG))
    vl._RECS = [rec]
    out = vl.label_job(("unused", res["specs"], 4))
    vl._RECS = None
    rows = out["rows"]
    for i, r in enumerate(rows):  # the sampled record is one fight: fake two splits
        r["spec"] = dict(r["spec"], split="train" if i % 3 else "dev")
    shard = vl.FeatureShard()
    for r in rows:
        shard.add(r)
    shard.write(str(tmp_path / "label-00000.npz"))
    d = vt.load_label_data(str(tmp_path))
    assert len(d.ids) == len(rows)
    for i, r in enumerate(rows):
        lo, hi = d.row_off[i], d.row_off[i + 1]
        got = collections.Counter()
        for z, t, c in zip(d.f_zone[lo:hi], d.f_tok[lo:hi], d.f_cnt[lo:hi], strict=True):
            got[int(z), d.vocab[t]] += int(c)
        want = collections.Counter()
        for z, cards in enumerate([*r["feats"]["zones"], r["feats"]["potions"]]):
            for card in cards:
                want[z, card if card in d.vocab else "<unk>"] += 1
        assert got == want
    norm = vt.norm_stats(d, np.arange(len(d.ids)))
    model = vt.build_model(torch, len(d.vocab), len(d.species), 32, True)
    bt = vt.Batcher(torch, d, norm, True, "cpu")
    p, mu_w, mu_l, v = model(*bt.inputs(torch.arange(len(d.ids))))
    assert v.shape == (len(d.ids),)
    assert bool(((mu_w >= 1.0) & (mu_w <= 1.5) & (mu_l >= -1.45) & (mu_l <= -1.05)).all())
    gidx = vt.group_index(d.group)
    ys = {
        k: torch.as_tensor(getattr(d, k).astype(np.float32))
        for k in ("n", "n_won", "mean", "mean_win", "mean_loss")
    }
    terms = vt.loss_terms(
        torch, (p, mu_w, mu_l, v), ys, torch.as_tensor(gidx), torch.ones(len(d.ids))
    )
    assert all(bool(torch.isfinite(t)) for t in terms.values())
    assert float(terms["sib"].detach()) > 0
