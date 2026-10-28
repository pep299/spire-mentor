"""Per-fight log of each enemy's DISPLAYED intents (GH #830 step 0, DESIGN §a.2).

The deployed BeliefDraw search keeps the monsters' AI state (`belief_root`
does not touch it), so it knows where each monster is in its move pattern. A
human player infers the same thing from the intents they have already seen.
The wire cannot carry this: `intent_history` is always empty, and `intents`
holds only the head of a multi-intent move (the KNOWN GAP in
`sts2-wire/src/emit/combat.rs`). `IntentLog` records what was shown, and only
that:

* On the first step of each new `round` it appends, per enemy (keyed by the
  opaque `eid`), that enemy's displayed head intent from the honest snapshot:
  kind, amount, effective_amount, times. An enemy with no intent gets kind
  `NONE`.
* `history(snap)` returns, aligned with `snap["enemies"]`, the last
  `MAX_HIST` entries from rounds before the current one, each with
  `lag` = current round - its round. The current round's intent is already
  on the ENEMY token.
* An enemy's history resets when the species under its `eid` changes;
  resets are counted.

It reads only `snap["round"]` and each enemy's `eid`, `kind` and `intents`
(test 12). The tail of a multi-intent move is NOT filled in from
`sts2_rs.monster_moves`: that needs the move id, which is hidden, and about 26
movesets are under-declared in content. gen and the play driver use this one
class, so their logs agree.
"""

from __future__ import annotations

from typing import Any

MAX_HIST = 3
NONE_KIND = "NONE"


def head_intent(enemy: dict[str, Any]) -> dict[str, Any]:
    ints = enemy.get("intents") or []
    if not ints:
        return {"kind": NONE_KIND}
    i = ints[0]
    out: dict[str, Any] = {"kind": i.get("kind", "Unknown")}
    for k in ("amount", "effective_amount", "times"):
        if k in i:
            out[k] = i[k]
    return out


class IntentLog:
    """Displayed-intent history of one fight, built from honest snapshots."""

    def __init__(self) -> None:
        self.last_round: int | None = None
        self.log: dict[Any, list[tuple[int, dict[str, Any]]]] = {}
        self.species: dict[Any, str] = {}
        self.resets = 0

    def update(self, snap: dict[str, Any]) -> None:
        """Record the snapshot's intents if it is the first step of a new round."""
        rnd = int(snap["round"])
        if rnd == self.last_round:
            return
        self.last_round = rnd
        for m in snap["enemies"]:
            eid = m["eid"]
            if self.species.get(eid, m["kind"]) != m["kind"]:
                self.log[eid] = []
                self.resets += 1
            self.species[eid] = m["kind"]
            self.log.setdefault(eid, []).append((rnd, head_intent(m)))

    def history(self, snap: dict[str, Any]) -> list[list[dict[str, Any]]]:
        """Per enemy of `snap` (same order): up to `MAX_HIST` earlier entries,
        most recent first. A dead enemy keeps its history (its alive flag is on
        the token)."""
        rnd = int(snap["round"])
        out = []
        for m in snap["enemies"]:
            eid = m["eid"]
            hist = []
            if self.species.get(eid) == m["kind"]:
                past = [(r, i) for r, i in self.log.get(eid, []) if r < rnd]
                for r, i in reversed(past[-MAX_HIST:]):
                    hist.append({"lag": rnd - r, **i})
            out.append(hist)
        return out

    def observe(self, snap: dict[str, Any]) -> dict[str, Any]:
        """`update`, then attach `snap["ehist"]`; returns `snap`."""
        self.update(snap)
        snap["ehist"] = self.history(snap)
        return snap
