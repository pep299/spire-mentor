"""Turn-plan label positions as arrays, for the plan scorer (GH #830).

`load` reads the label generator's output through `label.read_positions`
(each cell only from the shard its ledger names, each turn once), keeps one
split, and maps names to ids with a `Vocab` built from the TRAIN split only
(`build_vocab`), so the scorer never sees a holdout-only card. Id 0 is UNK.

Kept as recorded:

- capped positions (`cap_hit`, a partial plan list): flagged, not dropped;
  the trainer trains on them (a partial list is still a valid listwise
  target), the recall measurement excludes them.
- positions of `watchdog` / `error` cells: each is complete on its own.
- terminal plans: `visits == 0`, `q_var == 0`.
- `logp_rl2 is None` (no rl2 prior): `-inf`, so it ranks last under the rl2
  filter.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ppo.turnplan.label import read_positions

PLAN_DIM = 18
CTX_DIM = 6
ANCHORS = ("lethal", "max_block", "max_damage", "heuristic", "rl2")
UNK = 0


@dataclass
class Vocab:
    """Name -> id maps for cards, relics and powers; id 0 is UNK."""

    card: dict[str, int]
    relic: dict[str, int]
    power: dict[str, int]

    @classmethod
    def build(cls, records: Iterable[dict[str, Any]]) -> Vocab:
        """The vocab of `records`, ids in sorted-name order from 1."""
        cards: set[str] = set()
        relics: set[str] = set()
        powers: set[str] = set()
        for r in records:
            deck = r["deck"]
            cards.update(deck["cards"])
            relics.update(deck["relics"])
            powers.update(name for name, _ in deck["powers"])
            for p in r["plans"]:
                cards.update(p["cards"])

        def ids(names: set[str]) -> dict[str, int]:
            return {n: i for i, n in enumerate(sorted(names), start=1)}

        return cls(ids(cards), ids(relics), ids(powers))

    def sizes(self) -> dict[str, int]:
        """Embedding table sizes (UNK included)."""
        return {
            "card": len(self.card) + 1,
            "relic": len(self.relic) + 1,
            "power": len(self.power) + 1,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"card": self.card, "relic": self.relic, "power": self.power}) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["card"], d["relic"], d["power"])


@dataclass
class Position:
    """One labelled turn start: the deck context and its `n` plans."""

    key: str
    rep: int
    file: str
    room_type: str
    cap_hit: bool
    deck_cards: list[int]
    relics: list[int]
    powers: list[tuple[int, float]]
    ctx: np.ndarray  # [6] float32
    plan_feats: np.ndarray  # [n, 18] float32
    plan_cards: list[list[int]]
    q: np.ndarray  # [n] float32
    q_var: np.ndarray  # [n] float32
    visits: np.ndarray  # [n] int64
    logp_heur: np.ndarray  # [n] float32
    logp_rl2: np.ndarray  # [n] float32, -inf where the record has None
    anchors: np.ndarray  # [n, 5] bool, columns in `ANCHORS` order


@dataclass
class Normaliser:
    """Train-split mean / std of the context and plan features."""

    ctx_mean: np.ndarray
    ctx_std: np.ndarray
    plan_mean: np.ndarray
    plan_std: np.ndarray

    @classmethod
    def fit(cls, positions: list[Position]) -> Normaliser:
        """Moments over `positions` (plan moments over every plan); std floored to 1."""
        ctx = np.stack([p.ctx for p in positions]).astype(np.float64)
        plans = np.concatenate([p.plan_feats for p in positions]).astype(np.float64)

        def std(x: np.ndarray) -> np.ndarray:
            s = x.std(0)
            return np.where(s < 1e-6, 1.0, s).astype(np.float32)

        return cls(
            ctx.mean(0).astype(np.float32),
            std(ctx),
            plans.mean(0).astype(np.float32),
            std(plans),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> Normaliser:
        return cls(**{k: np.asarray(v, np.float32) for k, v in d.items()})


def _records(dirs: list[str], split: str) -> Iterator[dict[str, Any]]:
    for d in dirs:
        for rec in read_positions(d):
            if rec["split"] == split and rec["plans"]:
                yield rec


def build_vocab(dirs: list[str]) -> Vocab:
    """The vocab of the TRAIN split of `dirs`."""
    return Vocab.build(_records(dirs, "train"))


def to_position(rec: dict[str, Any], vocab: Vocab) -> Position:
    """One shard record as a `Position`, names mapped through `vocab`."""
    deck = rec["deck"]
    plans = rec["plans"]

    def card_ids(names: list[str]) -> list[int]:
        return [vocab.card.get(n, UNK) for n in names]

    return Position(
        key=rec["key"],
        rep=int(rec["rep"]),
        file=rec["file"],
        room_type=rec["room_type"],
        cap_hit=bool(rec["cap_hit"]),
        deck_cards=card_ids(deck["cards"]),
        relics=[vocab.relic.get(n, UNK) for n in deck["relics"]],
        powers=[(vocab.power.get(n, UNK), float(a)) for n, a in deck["powers"]],
        ctx=np.asarray(deck["ctx"], np.float32),
        plan_feats=np.asarray([p["features"] for p in plans], np.float32).reshape(-1, PLAN_DIM),
        plan_cards=[card_ids(p["cards"]) for p in plans],
        q=np.asarray([p["q"] for p in plans], np.float32),
        q_var=np.asarray([p["q_var"] for p in plans], np.float32),
        visits=np.asarray([p["visits"] for p in plans], np.int64),
        logp_heur=np.asarray([p["logp_heur"] for p in plans], np.float32),
        logp_rl2=np.asarray(
            [-np.inf if p["logp_rl2"] is None else p["logp_rl2"] for p in plans], np.float32
        ),
        anchors=np.asarray([[p["anchors"][a] for a in ANCHORS] for p in plans], bool).reshape(
            -1, len(ANCHORS)
        ),
    )


def load(dirs: list[str], split: str, vocab: Vocab | None = None) -> list[Position]:
    """Every position of `split` in the label dirs `dirs`.

    Args:
        dirs: Label generator output dirs.
        split: `"train"` or `"holdout"`.
        vocab: The id maps; default: `build_vocab(dirs)` (train split only).
    """
    if split not in ("train", "holdout"):
        raise ValueError(f"split must be train or holdout, not {split!r}")
    vocab = build_vocab(dirs) if vocab is None else vocab
    return [to_position(r, vocab) for r in _records(dirs, split)]
