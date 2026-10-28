"""`ppo.turnplan.data` / `ppo.turnplan.train`: the deck-aware plan scorer (GH #830).

The scorer ranks a turn's plans against the teacher's prices. These tests pin
that it can learn a ranking at all, that `--no-deck` really ignores the deck,
that the vocabulary never sees the holdout split, and that padding a batch
does not change any position's loss. Every record is synthetic; no engine.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from ppo.turnplan import data, train
from ppo.turnplan.label import split_of

PLAN_DIM = data.PLAN_DIM


def _file_in(split: str, salt: int = 0) -> str:
    """A run file name that `split_of` puts in `split`."""
    for i in range(salt * 1000, salt * 1000 + 1000):
        f = f"{i:016x}.run.gz"
        if split_of(f) == split:
            return f
    raise AssertionError(split)


def _plan(features: list[float], q: float, cards: list[str]) -> dict[str, Any]:
    return {
        "actions": [],
        "end": "end_turn",
        "features": features,
        "cards": cards,
        "logp_heur": -1.0,
        "logp_rl2": None,
        "anchors": {
            "lethal": False,
            "max_block": False,
            "max_damage": False,
            "heuristic": False,
            "rl2": False,
        },
        "q": q,
        "visits": 10,
        "q_var": 0.1,
        "final_hp": None,
    }


def _record(file: str, turn: int, plans: list[dict[str, Any]], cards: list[str]) -> dict[str, Any]:
    return {
        "key": f"{file}:1",
        "rep": 0,
        "file": file,
        "room_type": "boss",
        "turn": turn,
        "split": split_of(file),
        "label_s": 0.1,
        "cap_hit": False,
        "n_plans": len(plans),
        "deck": {
            "cards": cards,
            "relics": ["BurningBlood"],
            "powers": [["Strength", 2]],
            "ctx": [float(turn), 0.5, 3.0, 3.0, 0.0, 1.0],
        },
        "plans": plans,
    }


def _write_dir(out: Path, records: list[dict[str, Any]]) -> None:
    """`records` in the label generator's on-disk format: one shard per cell + ledger."""
    out.mkdir(parents=True)
    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in records:
        cells.setdefault((r["key"], r["rep"]), []).append(r)
    with open(out / "cells.jsonl", "w", encoding="utf-8") as ledger:
        for n, ((key, rep), recs) in enumerate(cells.items()):
            shard = f"shard-0-{n}.jsonl.gz"
            with gzip.open(out / shard, "wt") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            row = {"key": key, "rep": rep, "end": "won", "positions": len(recs), "shard": shard}
            ledger.write(json.dumps(row) + "\n")


def _synthetic(n_pos: int, seed: int) -> list[dict[str, Any]]:
    """Positions whose best plan, and only it, has feature 4 set."""
    rng = random.Random(seed)
    recs = []
    for i in range(n_pos):
        n = rng.randint(2, 12)
        best = rng.randrange(n)
        plans = []
        for j in range(n):
            f = [rng.gauss(0.0, 1.0) for _ in range(PLAN_DIM)]
            f[4] = 1.0 if j == best else 0.0
            q = rng.uniform(0.3, 0.5) if j == best else rng.uniform(-0.5, 0.2)
            plans.append(_plan(f, q, rng.sample(["Strike", "Defend", "Bash"], rng.randint(0, 2))))
        split = "holdout" if i % 4 == 0 else "train"
        recs.append(_record(_file_in(split, salt=i), 1, plans, ["Strike", "Defend"]))
    return recs


def test_learns_synthetic_best_feature(tmp_path: Path) -> None:
    _write_dir(tmp_path / "d", _synthetic(240, seed=0))
    vocab = data.build_vocab([str(tmp_path / "d")])
    tr = data.load([str(tmp_path / "d")], "train", vocab)
    ho = data.load([str(tmp_path / "d")], "holdout", vocab)
    assert len(tr) > 100 and len(ho) > 30

    fit_part, val = train.split_val(tr)
    model, meta = train.fit(fit_part, val, vocab, train.TrainConfig(epochs=30, seed=0))

    scores = train.score(model, ho)
    top1 = np.mean([int(np.argmax(s) == np.argmax(p.q)) for s, p in zip(scores, ho, strict=True)])
    assert top1 >= 0.9, (top1, meta["val_loss"])


def test_no_deck_ignores_deck() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(5, PLAN_DIM)).astype(np.float32)
    base = data.Position(
        key="k",
        rep=0,
        file="f",
        room_type="boss",
        cap_hit=False,
        deck_cards=[1, 2, 2],
        relics=[1],
        powers=[(1, 2.0)],
        ctx=np.arange(6, dtype=np.float32),
        plan_feats=feats,
        plan_cards=[[1], [2], [], [1, 2], [3]],
        q=np.zeros(5, np.float32),
        q_var=np.zeros(5, np.float32),
        visits=np.zeros(5, np.int64),
        logp_heur=np.zeros(5, np.float32),
        logp_rl2=np.zeros(5, np.float32),
        anchors=np.zeros((5, 5), bool),
    )
    other = data.Position(
        **{**base.__dict__, "deck_cards": [3, 3, 3, 1], "relics": [2, 3], "powers": [(2, 5.0)]}
    )
    norm = data.Normaliser.fit([base])
    sizes = {"card": 4, "relic": 4, "power": 3}

    # One position per batch, so both go through bit-identical arithmetic.
    deck_model = train.PlanScorer(sizes, norm, no_deck=False)
    (s_base,), (s_other,) = train.score(deck_model, [base]), train.score(deck_model, [other])
    assert not np.allclose(s_base, s_other), "the deck branch must matter when it is on"

    model = train.PlanScorer(sizes, norm, no_deck=True)
    (s_base,), (s_other,) = train.score(model, [base]), train.score(model, [other])
    np.testing.assert_array_equal(s_base, s_other)


def test_vocab_from_train_only(tmp_path: Path) -> None:
    plans = [
        _plan([0.0] * PLAN_DIM, 0.0, ["Strike"]),
        _plan([1.0] * PLAN_DIM, 1.0, ["HoldoutOnly"]),
    ]
    recs = [
        _record(_file_in("train"), 1, plans[:1], ["Strike", "Defend"]),
        _record(_file_in("holdout"), 1, plans, ["Strike", "HoldoutOnly"]),
    ]
    _write_dir(tmp_path / "d", recs)

    vocab = data.build_vocab([str(tmp_path / "d")])
    assert "HoldoutOnly" not in vocab.card
    (ho,) = data.load([str(tmp_path / "d")], "holdout", vocab)
    assert ho.deck_cards == [vocab.card["Strike"], 0]
    assert ho.plan_cards == [[vocab.card["Strike"]], [0]]
    assert ho.logp_rl2.tolist() == [-np.inf, -np.inf]

    vocab.save(tmp_path / "vocab.json")
    assert data.Vocab.load(tmp_path / "vocab.json") == vocab


def test_padding_does_not_leak() -> None:
    rng = np.random.default_rng(1)
    scores = torch.tensor(rng.normal(size=(3, 4)), dtype=torch.float32)
    q = torch.tensor(rng.normal(size=(3, 4)), dtype=torch.float32)
    mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False], [True, False, False, False]]
    )
    loss = train.listwise_loss(scores, q, mask, tau=0.05)

    # Three more padded columns holding garbage scores and prices.
    pad_s = torch.tensor(rng.normal(size=(3, 3)) * 100, dtype=torch.float32)
    pad_q = torch.tensor(rng.normal(size=(3, 3)) * 100, dtype=torch.float32)
    loss_padded = train.listwise_loss(
        torch.cat([scores, pad_s], 1),
        torch.cat([q, pad_q], 1),
        torch.cat([mask, torch.zeros(3, 3, dtype=torch.bool)], 1),
        tau=0.05,
    )
    torch.testing.assert_close(loss, loss_padded)
    assert torch.isfinite(loss).all()

    # A single-plan position has zero loss whatever its score.
    assert loss[2].item() == pytest.approx(0.0, abs=1e-6)


def test_val_fold_is_by_run_file_and_inside_train() -> None:
    files = [_file_in("train", salt=i) for i in range(200)]
    val = [f for f in files if train.is_val_file(f)]
    assert 20 <= len(val) <= 60, len(val)  # ~20% of train files
    assert [train.is_val_file(f) for f in files] == [train.is_val_file(f) for f in files]

    # Every position of one file lands on one side; holdout files are refused.
    recs = [r for r in _synthetic(80, seed=3) if r["split"] == "train"]
    recs += [{**r, "turn": 2} for r in recs]
    positions = [data.to_position(r, data.Vocab.build(recs)) for r in recs]
    fit_part, val_part = train.split_val(positions)
    assert len(fit_part) + len(val_part) == len(positions)
    assert {p.file for p in fit_part}.isdisjoint({p.file for p in val_part})
    assert all(train.is_val_file(p.file) for p in val_part)
    holdout = data.to_position(
        _record(_file_in("holdout"), 1, recs[0]["plans"], ["Strike"]), data.Vocab.build(recs)
    )
    with pytest.raises(ValueError, match="holdout"):
        train.split_val([holdout])


def test_main_never_reads_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_dir(tmp_path / "d", _synthetic(160, seed=2))
    real = data.to_position

    def guard(rec: dict[str, Any], vocab: data.Vocab) -> data.Position:
        assert rec["split"] != "holdout", "train.py mapped a holdout position"
        return real(rec, vocab)

    monkeypatch.setattr(data, "to_position", guard)
    real_build = data.Vocab.build.__func__

    def build(cls: type, records: Any) -> data.Vocab:
        records = list(records)
        assert all(r["split"] == "train" for r in records)
        return real_build(cls, records)

    monkeypatch.setattr(data.Vocab, "build", classmethod(build))

    out = tmp_path / "m"
    assert train.main(["--data", str(tmp_path / "d"), "--out", str(out), "--epochs", "2"]) == 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_val"] > 0 and meta["n_val_files"] > 0
    assert meta["n_train"] > meta["n_val"]
    assert len(meta["val_loss"]) == 2 and len(meta["val_top1"]) == 2
    assert not any("holdout" in k for k in meta), sorted(meta)
