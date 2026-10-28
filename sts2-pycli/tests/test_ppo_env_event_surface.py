"""GH #302: ordinary event rooms must reach the policy as real decisions.

`PpoRunEnv._advance` auto-resolves any decision with a single legal action.
The event used to be bound inside `observe()`, which runs only *past* that
check — so on arriving at an event room `current_event` was still `None`, the
engine offered the lone placeholder `event_choice` that binds it, and the room
was swallowed as ``single-option-auto``. Every ordinary event was resolved at
option 0 without the policy ever seeing it.

The cost was invisible until the checkpoints were diffed: across Regent's
whole ``it220 -> it800`` leg (580 iterations x 256 games) ``emb_event`` moved
**0.000000**, with 3 of 71 rows touched — ``Neow`` (pre-bound by
``neow_start=True``), ``TinkerTime`` (multi-page events linger with
``current_event`` set, so page 2+ does surface), and the OOV row by 3.1e-5.
Meanwhile ``emb_relic`` moved 0.0084 and ``emb_potion`` 0.0071, so this was
never sparsity. The deployed Rust arm binds before choosing
(``PpoPolicy::prepare``, policy.rs:673), so the policy met event choices at
deploy that it had never trained on — the mechanism behind the #300
SlipperyBridge suicide loop.

These tests pin the contract at the env boundary rather than the fix's
mechanics: an event room is a multi-option decision, and it carries an
identity the encoder can resolve.
"""

from __future__ import annotations

import os
import random

import pytest

PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))

# Enough seeds to walk past act 1 on several runs under random play. A
# 40-seed sweep of this same walk yields ~3.6 event decisions per run, ~2.6 of
# them non-Neow, so six seeds clear every "at least" assertion below by a wide
# margin; the walks dominate runtime, hence not more. Each walk is its own
# test so xdist spreads them over workers: as one module fixture the six ran
# back to back (~40 s) on every worker that drew a test from this file, the
# pytest gate's long pole.
SEEDS = range(1000, 1006)


@pytest.fixture(autouse=True)
def _syspath(monkeypatch):
    monkeypatch.syspath_prepend(PY)


def _walk(seed: int):
    """Random-policy walk of one run; returns (event decisions, resolution log)."""
    from ppo.env import Decision, PpoRunEnv

    env = PpoRunEnv(seed=seed, ascension=0, character="Ironclad")
    rng = random.Random(seed)
    events = []
    out = env.reset()
    while isinstance(out, Decision):
        if out.surface == "event":
            events.append((out.obs.get("event_name"), len(out.actions)))
        out = env.step(rng.randrange(len(out.actions)))
    return events, env.resolution_log


@pytest.fixture(scope="module", params=SEEDS, ids=lambda s: f"seed{s}")
def walk(request):
    return _walk(request.param)


def test_no_event_room_is_auto_resolved(walk) -> None:
    """The bug, stated directly: an event must never be `single-option-auto`.

    This is the assertion that fails against the pre-fix env — every ordinary
    event room took exactly this branch.
    """
    _, log = walk
    swallowed = [
        (phase, surface, klass)
        for _, phase, surface, klass in log
        if phase == "event" and klass == "single-option-auto"
    ]
    assert not swallowed, f"{len(swallowed)} event rooms auto-resolved without a decision"


def test_ordinary_events_reach_the_policy() -> None:
    """Not just Neow. Pre-fix this set was exactly {'Neow'}.

    Across SEEDS, as before, but walked lazily: the first seed that reaches an
    ordinary event settles it (today that is the first seed)."""
    names: set = set()
    for seed in SEEDS:
        events, _ = _walk(seed)
        names |= {name for name, _ in events}
        if "Neow" in names and names - {"Neow"}:
            break
    assert "Neow" in names, "Neow is pre-bound by neow_start and must still surface"
    assert names - {"Neow"}, "no ordinary event room ever became a decision"


def test_every_event_decision_carries_a_resolvable_identity(walk) -> None:
    """An unbound event yields `event_name=None` -> the encoder's pad row, and
    a name outside the registry yields the shared OOV row. Both make the
    decision anonymous to the net; neither may happen."""
    from ppo import vocab

    reg = vocab.events()
    events, _ = walk
    for name, n_actions in events:
        assert name, "event decision with no bound event id"
        assert reg.index(name) != reg.oov, f"event id {name!r} lands on the OOV row"
        assert n_actions > 1, f"{name} yielded a decision with {n_actions} action(s)"
