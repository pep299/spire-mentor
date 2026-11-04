"""Tests for :mod:`combat_mirror` — the persistent per-fight engine mirror.

Two tiers:

* **Unit** — synthetic wire dicts drive one mirror; a doctored field must
  produce a DESYNC, STRICT must raise, OBSERVE must tag, and ``search`` must
  refuse a clairvoyant profile.
* **Oracle** — a mirror driven alongside a *native* ``RunManager`` run (see
  :mod:`diag_mirror_oracle`), fed the true engine state at every combat
  decision as if it had come off the wire. Ground truth is exact here, so any
  drift is a mirror-machinery bug with **zero content-parity confound** — the
  isolation the DLL grind can never give us, and the reason this is the gate
  for the machinery.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
from combat_mirror import (
    CombatMirror,
    MirrorDivergence,
    MirrorPolicy,
    diff_observables,
)
from diag_mirror_oracle import drive_run, sweep
from sts2_hint.recommender import RecommendConfig

sts2_rs = pytest.importorskip("sts2_rs")

# The honest profile every mirror search must run under. Matches the deployed
# `MctsCombatPolicy` (BeliefDraw + c_puct 2.5 + draw_widening_c 2.0).
HONEST = RecommendConfig(sims=40, hidden_info="beliefdraw", c_puct=2.5, draw_widening_c=2.0)


def _mirror(tmp_path, policy=MirrorPolicy.STRICT, label="test") -> CombatMirror:
    return CombatMirror(HONEST, policy=policy, label=label, log_path=tmp_path / "divergences.jsonl")


def _fixture_state(seed: int = 7):
    """A real Ironclad-vs-Nibbit combat, mid-turn-1."""
    return sts2_rs.CombatState(seed=seed, encounter="Nibbit")


# ---------------------------------------------------------------------------
# Observables / diff
# ---------------------------------------------------------------------------


def test_identical_states_have_no_desync():
    wire = sts2_rs.state_to_dict(_fixture_state())
    assert diff_observables(wire, json.loads(json.dumps(wire))) == []


def test_diff_names_the_field_that_moved():
    wire = sts2_rs.state_to_dict(_fixture_state())
    doctored = json.loads(json.dumps(wire))
    doctored["player"]["hp"] -= 7
    doctored["encounter"][0]["block"] += 3

    paths = {d.path for d in diff_observables(wire, doctored)}
    # Enemies are keyed by identity, not slot — see `enemy_slot_key`.
    assert paths == {"$.player.hp", "$.encounter[Nibbit#0].block"}


def test_a_telegraph_is_compared_baked_not_raw():
    """The mirror's raw base must be diffed against the world's baked number.

    A real wire reports C# ``AttackIntent.GetSingleDamage`` — already run
    through ``ModifyDamage`` — while ``Intent::Attack`` stores the raw base
    from the move table. Comparing those directly flags every Strength / Weak /
    Shrink / player-Vulnerable board as a divergence; that was 161 of the 213
    unequal attack telegraphs in the v17 corpus.
    """
    wire = sts2_rs.state_to_dict(_fixture_state())
    mirror = json.loads(json.dumps(wire))
    world = json.loads(json.dumps(wire))
    # The mirror carries base 9 that bakes to 6 (a Weak dealer); the world,
    # which has no `effective_amount`, reports the baked 6 in `amount`.
    mirror["encounter"][0]["intents"] = [
        {"kind": "Attack", "amount": 9, "effective_amount": 6, "times": None}
    ]
    world["encounter"][0]["intents"] = [{"kind": "Attack", "amount": 6, "times": None}]
    assert diff_observables(mirror, world) == []

    # ...and a genuinely different telegraph still diverges.
    world["encounter"][0]["intents"] = [{"kind": "Attack", "amount": 7, "times": None}]
    assert "$.encounter[Nibbit#0].intents" in {d.path for d in diff_observables(mirror, world)}


def test_absent_times_is_the_same_telegraph_as_times_one():
    """A single-hit attack must not diverge on the ``times`` spelling.

    ``state_to_dict`` (``sts2-rs/crates/sts2-py/src/state_dict.rs``) emits
    ``times`` ONLY for ``Intent::MultiAttack``, while the bridge always emits
    ``times: 1``. Before normalisation ``_intent_key`` compared ``None``
    against ``1`` and flagged every single-hit attack telegraph as a
    divergence when a stored wire pair was re-diffed.
    """
    wire = sts2_rs.state_to_dict(_fixture_state())
    mirror = json.loads(json.dumps(wire))
    world = json.loads(json.dumps(wire))

    # Engine dialect: no `times` key at all on a single-hit attack.
    mirror["encounter"][0]["intents"] = [{"kind": "Attack", "amount": 6}]
    # Bridge dialect: an explicit `times: 1`.
    world["encounter"][0]["intents"] = [{"kind": "Attack", "amount": 6, "times": 1}]
    assert diff_observables(mirror, world) == []

    # An explicit `None` (the older bridge spelling) folds in the same way.
    world["encounter"][0]["intents"] = [{"kind": "Attack", "amount": 6, "times": None}]
    assert diff_observables(mirror, world) == []

    # ...but a REAL hit-count difference still diverges.
    world["encounter"][0]["intents"] = [{"kind": "Attack", "amount": 6, "times": 3}]
    assert "$.encounter[Nibbit#0].intents" in {d.path for d in diff_observables(mirror, world)}


def _two_nibbits(*, first_is_dead: bool) -> dict[str, Any]:
    """A two-Nibbit board where the engine still holds the first as a corpse."""
    wire = sts2_rs.state_to_dict(_fixture_state())
    alive = json.loads(json.dumps(wire["encounter"][0]))
    alive["hp"], alive["is_dead"] = 40, False
    corpse = json.loads(json.dumps(alive))
    corpse["hp"], corpse["is_dead"] = 0, True
    wire["encounter"] = [corpse, alive] if first_is_dead else [alive]
    return wire


def test_a_reaped_corpse_does_not_renumber_the_survivor():
    """The engine keeps a corpse the game removed; the boards still agree.

    The survivor sits at mirror slot 1 and world slot 0. Numbering the living
    and the dead separately is what lets those two name the same creature —
    the miss this key was written to prevent, and 11% of the v14 corpus.
    """
    mirror = _two_nibbits(first_is_dead=True)
    world = _two_nibbits(first_is_dead=False)
    assert diff_observables(mirror, world) == []


def test_a_survivor_the_world_reaped_is_still_a_divergence():
    """Reaping excuses a CORPSE the mirror kept, never a live enemy."""
    mirror = _two_nibbits(first_is_dead=True)
    mirror["encounter"][0]["is_dead"] = False  # the mirror thinks both live
    world = _two_nibbits(first_is_dead=False)
    assert "$.encounter.roster" in {d.path for d in diff_observables(mirror, world)}


def test_the_transplant_aligns_a_survivor_past_its_own_corpse():
    """The carried mirror's slot for the world's enemy 0 is the SURVIVOR.

    Alignment shares the identity key, so the old numbering pointed the
    world's only living enemy at the mirror's corpse — silently transplanting
    one creature's carried history onto another.
    """
    from combat_mirror import _enemy_alignment

    assert _enemy_alignment(
        _two_nibbits(first_is_dead=False), _two_nibbits(first_is_dead=True)
    ) == [1]


def test_a_mispredicted_kill_refuses_the_transplant_rather_than_guessing():
    """The one case ``enemy_slot_key`` genuinely cannot identify.

    The mirror thinks both Nibbits live; the world killed the first and reaped
    it. Numbering is per-namespace, so the world's survivor is ``Nibbit#0`` and
    the mirror's survivor is ``Nibbit#1`` — the keys cross and the world's
    survivor names the mirror's OTHER creature. Carrying ``intent_history``
    across that is exactly the mis-transplant the identity key exists to
    prevent, so the alignment must decline instead.
    """
    from combat_mirror import _enemy_alignment

    mirror = _two_nibbits(first_is_dead=True)
    mirror["encounter"][0]["is_dead"] = False  # the mirror mispredicted the kill
    world = _two_nibbits(first_is_dead=False)

    assert _enemy_alignment(world, mirror) == [None], (
        "a shifted same-kind roster must carry nothing, not the wrong creature's history"
    )


def test_power_order_is_not_a_desync():
    """The arena's power order is an implementation detail, not a board."""
    wire = sts2_rs.state_to_dict(_fixture_state())
    wire["player"]["powers"] = [
        {"name": "Strength", "amount": 2},
        {"name": "Vulnerable", "amount": 1},
    ]
    shuffled = json.loads(json.dumps(wire))
    shuffled["player"]["powers"].reverse()
    assert diff_observables(wire, shuffled) == []


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------


def _end_turn(state) -> str:
    return state.apply_step({"action": "end_turn"})


def test_reconcile_is_clean_when_the_mirror_predicted_reality(tmp_path):
    """The whole point: same board + same action ⇒ nothing to report."""
    world = _fixture_state()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))

    for _ in range(3):
        _end_turn(world)
        mirror.advance({"action": "end_turn"})
        assert mirror.reconcile(sts2_rs.state_to_dict(world)) == []

    assert mirror.stats.desyncs == 0
    assert mirror.stats.reconciles == 3


def test_strict_raises_on_a_doctored_observable(tmp_path):
    world = _fixture_state()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))

    _end_turn(world)
    mirror.advance({"action": "end_turn"})
    doctored = sts2_rs.state_to_dict(world)
    doctored["player"]["hp"] -= 5

    with pytest.raises(MirrorDivergence, match=r"\$\.player\.hp"):
        mirror.reconcile(doctored)


def test_observe_tags_the_run_and_restarts_from_reality(tmp_path):
    """OBSERVE never lets a proven-wrong state keep answering — but it does
    not leave the driver blind for the rest of the fight either: the mirror is
    dropped and immediately re-seeded from the wire, losing only the carried
    half (RNG cursors, pins, intent history)."""
    world = _fixture_state()
    log = tmp_path / "divergences.jsonl"
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(sts2_rs.state_to_dict(world))

    _end_turn(world)
    mirror.advance({"action": "end_turn"})
    doctored = sts2_rs.state_to_dict(world)
    doctored["player"]["hp"] -= 5
    mirror.reconcile(doctored)

    assert mirror.tainted, "an OBSERVE run must be tagged, not silently reported clean"
    assert "$.player.hp" in mirror.drop_reason
    assert mirror.alive, "the fight is still going; the mirror restarts from reality"
    # Reality, not the prediction — and a BARE reconstruction, so the carried
    # half really was surrendered rather than quietly kept.
    assert sts2_rs.state_to_dict(mirror.state)["player"]["hp"] == doctored["player"]["hp"]
    assert (
        mirror.state.shuffle_stream_counter()
        == sts2_rs.state_from_dict(doctored).shuffle_stream_counter()
    )

    record = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["kind"] == "desync"
    assert [d["path"] for d in record["desyncs"]] == ["$.player.hp"]
    assert record["world_wire"]["player"]["hp"] == doctored["player"]["hp"]


def test_a_pile_the_mirror_did_not_predict_is_a_divergence(tmp_path):
    """A draw pile that lost a card without the mirror playing one is a
    desync, reported by the observable diff.

    Deliberately NOT a transplant refusal: pins become untranslatable when
    the piles disagree and are dropped, but the RNG cursors are still valid,
    so the transplant proceeds and the DIFF is what raises. Refusing there
    too would have thrown away the cursors to report something already
    reported.
    """
    world = _fixture_state()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))

    shrunk = sts2_rs.state_to_dict(world)
    shrunk["deck"]["draw_pile"] = shrunk["deck"]["draw_pile"][:-1]
    with pytest.raises(MirrorDivergence, match=r"\$\.deck\.draw_pile"):
        mirror.reconcile(shrunk)


def test_a_world_card_in_flight_defers_the_reconcile(tmp_path):
    """A world wire whose ``play_pile`` holds a card the mirror has nothing
    in flight for is a MID-RESOLUTION snapshot (the live producer serialized
    between "the play started" and "the play finished"), not a divergence.

    The mid-resolution board is transiently wrong on every field the card
    touches — its power missing, the card not yet in the discard — so STRICT
    must not raise, the diff is deferred, and the carried mirror survives
    untouched (rebuilding from such a wire would adopt a play-pile card that
    can never resume).
    """
    world = _fixture_state()
    mirror = _mirror(tmp_path)  # STRICT
    mirror.seed(sts2_rs.state_to_dict(world))
    predicted = sts2_rs.state_to_dict(mirror.state)

    midres = sts2_rs.state_to_dict(world)
    # The just-played card sits in the play pile; its effects have not landed
    # (hand shrank, but block/powers/discard are still pre-play) — doctored
    # hp stands in for "a field the in-flight card already/not-yet touched".
    midres["deck"]["play_pile"] = [midres["deck"]["hand"].pop()]
    midres["player"]["hp"] -= 6

    assert mirror.reconcile(midres) == []
    assert mirror.stats.in_flight_skips == 1
    assert mirror.stats.desyncs == 0
    assert not mirror.tainted
    assert mirror.alive
    # The carried state was kept, not rebuilt from the half-applied board.
    assert sts2_rs.state_to_dict(mirror.state) == predicted
    # And because no reseed adopted the half-applied board, the resolution
    # finishing leaves nothing behind: the next stable wire diffs clean. (The
    # old rebuild adopted the in-flight card into a play_pile it could never
    # leave, whose absence from the mirror's discard then echoed as the
    # discard_pile-ALONE class — 20/22 of those live2 records were exactly
    # the stuck card matching the world's discard surplus.)
    assert mirror.reconcile(sts2_rs.state_to_dict(world)) == []
    assert mirror.stats.desyncs == 0


def test_the_deferred_diff_is_still_graded_at_the_next_stable_wire(tmp_path):
    """The in-flight skip DEFERS the comparison; it does not grant amnesty.

    A real misprediction that hid behind a mid-resolution snapshot is caught
    one decision later, when the world's play pile has emptied — an
    explicitly EMPTY (or absent) pile is a stable board and diffs as always.
    """
    world = _fixture_state()
    mirror = _mirror(tmp_path)  # STRICT
    mirror.seed(sts2_rs.state_to_dict(world))

    midres = sts2_rs.state_to_dict(world)
    midres["deck"]["play_pile"] = [midres["deck"]["hand"].pop()]
    midres["player"]["hp"] -= 6
    assert mirror.reconcile(midres) == []

    stable = sts2_rs.state_to_dict(world)
    stable["deck"]["play_pile"] = []  # emptied: the play finished resolving
    stable["player"]["hp"] -= 6
    with pytest.raises(MirrorDivergence, match=r"\$\.player\.hp"):
        mirror.reconcile(stable)
    assert mirror.stats.in_flight_skips == 1
    assert mirror.stats.desyncs == 1


def test_reconcile_takes_the_worlds_cursor_when_the_wire_carries_it(tmp_path):
    """Reality owns the board — and, when it says so, the dice too.

    A producer that ships ``$.rng.run_seed`` is handing over the run's actual
    RNG lineage, and the reconstruction rebuilds every stream from it. Carrying
    the mirror's own cursors over that would replace known truth with the
    mirror's drift, which is exactly what makes an enemy intent a guess instead
    of a prediction.
    """
    world = _fixture_state()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    _end_turn(world)
    mirror.advance({"action": "end_turn"})

    wire = sts2_rs.state_to_dict(world)
    assert "run_seed" in wire["rng"], "the Rust producer must carry the run's dice"
    mirror.reconcile(wire)
    assert mirror.state.shuffle_stream_counter() == wire["rng"]["counters"]["shuffle"]


def test_a_cursor_the_wire_does_not_carry_is_still_carried_forward(tmp_path):
    """Older producers say nothing about the dice, and there the mirror's own
    cursors remain the only continuity there is — dropping them would restart
    every stream at roll zero on every decision."""
    world = _fixture_state()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    _end_turn(world)
    mirror.advance({"action": "end_turn"})
    carried = mirror.state.shuffle_stream_counter()

    wire = sts2_rs.state_to_dict(world)
    del wire["rng"]["run_seed"]
    del wire["rng"]["counters"]
    assert sts2_rs.state_from_dict(wire).shuffle_stream_counter() == 0
    mirror.reconcile(wire)
    assert mirror.state.shuffle_stream_counter() == carried


def test_a_cursor_the_world_moved_and_the_mirror_did_not_is_counted(tmp_path):
    """Consuming a different NUMBER of rolls is the divergence no field diff
    can see: the board still looks right this turn and is wrong for the rest
    of the fight.

    Counted, not raised. The cursor is hidden state rather than one of the
    deterministic observables STRICT is defined over, and a drift is a lead —
    it says the two engines spent different amounts of randomness, not yet
    that any board is wrong. Promoting it to a crash is a decision to take
    once the leads it produces have been worked through.
    """
    world = _fixture_state()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    _end_turn(world)
    mirror.advance({"action": "end_turn"})

    wire = sts2_rs.state_to_dict(world)
    wire["rng"]["counters"]["monster_ai"] += 1
    before = mirror.stats.cursor_drifts
    mirror.reconcile(wire)  # STRICT, and it must NOT raise
    assert mirror.stats.cursor_drifts == before + 1
    assert mirror.alive, "a cursor lead must not drop the mirror"


def test_only_the_relic_latches_the_wire_withheld_are_carried():
    """A relic counter the world reported is reality's; one it withheld is
    hidden state, and hidden state is what the transplant exists to carry.

    Vambrace's "the first card-granted block this combat doubles" lives in
    `counter`, and both producers only emit a counter for relics that SHOW
    one — so it arrived as None and every reconstruction re-armed it.
    """
    from combat_mirror import _undisclosed_relic_counters

    wire = {
        "player": {
            "relics": [
                {"id": "VAMBRACE", "counter": None},  # a latch, withheld
                {"id": "PAELS_LEGION", "counter": 2},  # shows a counter
                {"id": "BURNING_BLOOD"},  # absent entirely
            ]
        }
    }
    assert _undisclosed_relic_counters(wire) == ["VAMBRACE", "BURNING_BLOOD"]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_refuses_a_clairvoyant_profile(tmp_path):
    """The mirror holds the true draw order, so it is the one object that
    must never be searched clairvoyantly."""
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(_fixture_state()))
    with pytest.raises(ValueError, match="clairvoyant"):
        mirror.search(RecommendConfig(sims=8, hidden_info="clairvoyant"))


def test_search_never_mutates_ground_truth(tmp_path):
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(_fixture_state()))
    before = mirror.state.state_hash()
    result = mirror.search()
    assert result["actions"], "the fixture must have legal actions to search"
    assert mirror.state.state_hash() == before


# ---------------------------------------------------------------------------
# Multi-pick: one search per pick
# ---------------------------------------------------------------------------


def _two_pick_fixture(seed: int = 3):
    """A Silent state parked on Hidden Daggers' discard-2 (min = max = 2)."""
    state = sts2_rs.build_scenario_state(
        seed,
        "Silent",
        ["HIDDEN_DAGGERS"] * 4 + ["STRIKE_SILENT"] * 6,
        80,
        80,
        [],
        "SHRINKER_BEETLE_WEAK",
    )
    hand = sts2_rs.state_to_dict(state)["deck"]["hand"]
    idx = next(i for i, c in enumerate(hand) if c["id"] == "HIDDEN_DAGGERS")
    return state, idx


def test_multi_pick_searches_once_per_pick(tmp_path):
    """A discard-2 must be searched twice, not read off one root ranking.

    The engine decomposes a multi-pick sequentially (bead `sts2-rs-sopgf`):
    an incremental `ResolveChoice` fires no effect and re-opens the same
    choice. So the root's second-ranked row answers "which card FIRST",
    runner-up — not "which card second, given the first". Taking the top-k
    off one search asked the wrong question for every pick after the first.
    """
    world, play_idx = _two_pick_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    assert mirror.advance({"action": "play", "card_index": play_idx}) == "pending_choice"
    pc = mirror.parked_choice()
    assert (pc["min_select"], pc["max_select"]) == (2, 2), pc

    before = mirror.state.state_hash()
    searches_before = mirror.stats.searches
    picks = mirror.search_choice_picks(
        hand=sts2_rs.state_to_dict(mirror.state)["deck"]["hand"],
        n_options=len(pc["cards"]),
        min_select=2,
        max_select=2,
    )

    assert len(picks) == 2 and len(set(picks)) == 2, picks
    assert all(0 <= i < len(pc["cards"]) for i in picks), picks
    assert mirror.stats.searches - searches_before == 2, "one search per pick"
    # The walk runs on a private clone: the caller still answers the world
    # first and only then advances the mirror.
    assert mirror.state.state_hash() == before
    assert mirror.state.pending_choice() is not None


def test_single_pick_still_costs_exactly_one_search(tmp_path):
    """`max_select == 1` must not pay for the sequential walk."""
    world = _choice_fixture()  # Armaments: upgrade_in_hand, min = max = 1
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    assert mirror.advance({"action": "play", "card_index": 0}) == "pending_choice"
    pc = mirror.parked_choice()
    assert pc["max_select"] == 1, pc

    searches_before = mirror.stats.searches
    picks = mirror.search_choice_picks(
        hand=sts2_rs.state_to_dict(mirror.state)["deck"]["hand"],
        n_options=len(pc["cards"]),
        min_select=1,
        max_select=1,
    )
    assert len(picks) == 1
    assert mirror.stats.searches - searches_before == 1


# ---------------------------------------------------------------------------
# Injected choices
# ---------------------------------------------------------------------------


def _choice_fixture(seed: int = 3):
    """A hand of nothing but Armaments: playing one opens an ``upgrade_in_hand``
    choice over the four identical cards left behind.

    Identical options are the point — a name-matched option list would be
    ambiguous here, and the ``pile``/``index`` locator is what disambiguates.
    """
    return sts2_rs.build_scenario_state(
        seed,
        "Ironclad",
        ["ARMAMENTS"] * 10,
        80,
        80,
        [],
        "SHRINKER_BEETLE_WEAK",
    )


def _play_into_a_choice(state) -> dict[str, Any]:
    """Play hand[0] and return the choice it parks on."""
    assert state.apply_step({"action": "play", "card_index": 0}) == "pending_choice"
    pc = state.pending_choice()
    assert pc is not None
    return pc


def test_injection_accepts_the_engines_own_derivation(tmp_path):
    """The uniform path: re-open the live options over a mirror that already
    derived the same ones. Nothing to grade means nothing raised."""
    world = _choice_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))

    pc = _play_into_a_choice(world)
    assert mirror.advance({"action": "play", "card_index": 0}) == "pending_choice"
    assert mirror.parked_choice() is not None

    mirror.reconcile(sts2_rs.state_to_dict(world))
    assert mirror.parked_choice() is not None, "reconcile must not wipe the parked choice"
    assert (
        mirror.inject_choice(
            pc["kind"], pc["cards"], pc["min_select"], pc["max_select"], can_skip=pc["can_skip"]
        )
        == []
    )
    assert mirror.stats.injections == 1

    world.apply_step({"action": "choose", "selection": [1]})
    assert mirror.advance_through_choice([1]) == "advanced"
    assert mirror.reconcile(sts2_rs.state_to_dict(world)) == []


def _potion_fixture(potion: str = "ATTACK_POTION", seed: int = 3):
    """A belt holding one potion whose use opens a choose-a-card screen."""
    return sts2_rs.build_scenario_state(
        seed,
        "Ironclad",
        ["STRIKE_IRONCLAD"] * 10,
        80,
        80,
        [],
        "SHRINKER_BEETLE_WEAK",
        0,
        None,
        [potion],
    )


def _drink_into_a_choice(state) -> dict[str, Any]:
    """Drink the only potion in the belt and return the action that did it."""
    action = next(a for a in state.legal_actions() if a.get("action") in ("drink_potion", "potion"))
    assert state.apply_step(action) == "pending_choice"
    assert state.pending_choice() is not None
    return action


def test_reconcile_keeps_a_drink_parked_on_its_own_screen(tmp_path):
    """A rebuild deleted the potion's screen AND its deferred relic hook.

    C# `PotionModel.OnUseWrapper` (PotionModel.cs:291-342) awaits `OnUse` --
    the Attack/Skill/Power/Colorless choose-a-card screen included -- before
    firing `Hook.AfterPotionUsed` at `:338`, so the engine parks the deferred
    dispatch on `pending_potion_hook_resume`. Nothing on the wire carries it,
    and it was in NEITHER `has_pending_play` nor `has_suspended_walk`: a
    reconcile while a potion screen was open rebuilt from the wire and threw
    both away.

    That is why a potion screen could not be searched at all. The drink replay
    would land, park correctly, and then the very next reconcile -- the first
    thing `pick_card_select_mirrored` does -- undid it, leaving the injection
    to fail against a board with no open choice.
    """
    world = _potion_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))

    action = _drink_into_a_choice(world)
    assert mirror.advance(action) == "pending_choice"
    assert mirror.state.has_pending_potion_hook(), "the AfterPotionUsed tail is suspended"
    assert not mirror.state.has_pending_play(), "a drink is not a card play"
    assert not mirror.state.has_suspended_walk(), "nor a turn-transition walk"

    mirror.reconcile(sts2_rs.state_to_dict(world))

    assert mirror.parked_choice() is not None, "reconcile must not wipe the potion's screen"
    assert mirror.state.has_pending_potion_hook(), "nor the hook deferred behind it"


def test_injection_grades_a_deterministic_kind(tmp_path):
    """A pile-select the engine derived differently is a real divergence."""
    world = _choice_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    pc = _play_into_a_choice(world)
    mirror.advance({"action": "play", "card_index": 0})
    mirror.reconcile(sts2_rs.state_to_dict(world))

    trimmed = list(pc["cards"])[:-1]
    with pytest.raises(MirrorDivergence, match=r"\$\.choice\.options"):
        mirror.inject_choice(pc["kind"], trimmed, pc["min_select"], pc["max_select"])


def test_injection_does_not_grade_a_generative_kind(tmp_path):
    """Discover-class options were ROLLED — the engine rolled its own three
    and the game rolled a different three. That is not a fidelity signal."""
    world = _choice_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    _play_into_a_choice(world)
    mirror.advance({"action": "play", "card_index": 0})
    mirror.reconcile(sts2_rs.state_to_dict(world))

    rolled = [{"id": "STRIKE_IRONCLAD", "upgraded": False}, {"id": "BASH", "upgraded": True}]
    assert mirror.inject_choice("add_generated_to_hand", rolled, 1, 1) == []
    assert mirror.state.pending_choice()["kind"] == "add_generated_to_hand"


def test_an_unlocatable_deterministic_injection_is_refused_not_counted(tmp_path):
    """The one unparked shape that must not be injected at all.

    A deterministic kind whose options carry no ``pile``/``index`` points at no
    board position, and there is no parked choice to resolve them against. Every
    deterministic resolver bails when ``selected_card_uid`` comes back ``None``,
    and ``ChoiceKind::CardSelect`` — the kind a driver reaches for when it has
    nothing better — is an explicit no-op placeholder arm in the engine, not a
    working catch-all. Injecting anyway would have the world exhaust a card while
    the mirror closed its choice having applied nothing, with no grading (grading
    needs ``parked``) and nothing raised: a silent, permanent desync dressed as
    success.

    So this is a refusal, not a tally — which is why the counter below is
    measured on a shape that CAN be injected.
    """
    world = _choice_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    assert mirror.parked_choice() is None, "no card was played, so nothing parked"

    with pytest.raises(MirrorDivergence, match="no locator"):
        mirror.inject_choice("card_select", [{"id": "STRIKE_IRONCLAD"}], 1, 1)
    assert mirror.stats.injections == 0, "a refused screen was never opened"


def test_an_unparked_injection_is_counted(tmp_path):
    """A screen the engine never opened is a measurable modelling gap.

    Counting them separates "the engine mispredicted" from "the engine was never
    asked" — the metric the v8 → v15 series drove from 255 to 73.

    Measured on a *generative* kind, because that is the unparked screen the
    mirror can legitimately open: its options rolled their own content, so they
    are synthetic by design and carry no locator to miss. The deterministic
    unlocatable case is refused outright, see the test above.
    """
    world = _choice_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))

    # An injection over the engine's own parked choice is NOT unparked.
    pc = _play_into_a_choice(world)
    mirror.advance({"action": "play", "card_index": 0})
    mirror.reconcile(sts2_rs.state_to_dict(world))
    assert mirror.parked_choice() is not None
    mirror.inject_choice(pc["kind"], pc["cards"], pc["min_select"], pc["max_select"])
    assert mirror.stats.injections == 1
    assert mirror.stats.unparked_injections == 0

    # ... whereas a screen the engine never opened is. Deliberately last: the
    # grant lands a card in the mirror's hand that the world never received, so
    # any reconcile after this one would (correctly) diverge on `$.deck.hand`.
    mirror.advance_through_choice([0])
    world.apply_step({"action": "choose", "selection": [0]})
    assert mirror.parked_choice() is None
    rolled = [{"id": "STRIKE_IRONCLAD", "upgraded": False}, {"id": "BASH", "upgraded": True}]
    assert mirror.inject_choice("add_generated_to_hand", rolled, 1, 1) == []
    assert mirror.stats.injections == 2
    assert mirror.stats.unparked_injections == 1


def test_a_parked_mirror_is_verified_not_rebuilt(tmp_path):
    """A card suspended mid-play must survive its own reconcile.

    Its ``OnPlay`` opened a screen and the REST of it — AfterCardPlayed,
    replays, disposal — is parked in ``pending_play``. No wire dialect
    expresses that (the parser refuses a dict that even claims to), so
    rebuilding here silently deletes the tail and the pick resolves into
    nothing. Reality still wins on every observable; what the mirror does not
    do is throw away the half-executed card in order to say so.
    """
    world = _choice_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    _play_into_a_choice(world)
    mirror.advance({"action": "play", "card_index": 0})
    assert mirror.state.has_pending_play(), "Armaments suspends its tail on the screen"

    before = mirror.state
    assert mirror.reconcile(sts2_rs.state_to_dict(world)) == []
    assert mirror.stats.parked_verifies == 1
    assert mirror.state is before, "the parked state is KEPT, not replaced"
    assert mirror.state.has_pending_play(), "the suspended tail must survive the verify"


def test_replacing_a_parked_choice_keeps_the_suspended_card(tmp_path):
    """``replace_parked`` swaps the option list and nothing else.

    The mirror is sitting on its own derived choice and the world's list wins.
    Clearing ``pending_choice`` to make room is the whole operation —
    ``pending_play`` must come through untouched, which is the entire reason
    the mirror was left parked instead of rebuilt.
    """
    world = _choice_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    pc = _play_into_a_choice(world)
    mirror.advance({"action": "play", "card_index": 0})
    mirror.reconcile(sts2_rs.state_to_dict(world))
    assert mirror.parked_choice() is not None

    mirror.inject_choice(pc["kind"], pc["cards"], pc["min_select"], pc["max_select"])
    assert mirror.state.pending_choice() is not None, "the world's screen is open"
    assert mirror.state.has_pending_play(), (
        "replacing the choice must not take the suspended card with it"
    )

    # And the tail actually runs: resolving lands the upgrade AND closes the play.
    mirror.advance_through_choice([0])
    assert not mirror.state.has_pending_play(), "the pick finishes the suspended card"


def test_a_parked_choice_survives_the_observe_reseed(tmp_path):
    """The OBSERVE reseed must salvage the engine's derived choice.

    A parked-verify divergence drops the mirror and reseeds from the wire —
    but the divergence proved the state's *prediction* wrong, not the
    derived choice's kind/arity, and only that kind lets the next
    ``inject_choice`` resolve a locator-less world list at all. Clearing it
    made 25 of the 32 v34wr ``unlocatable_injection`` records: the engine
    had the right answer and threw it away microseconds before it was
    needed. The stale locators indexed the dropped state, so they are
    re-located against the world's own piles by identity.
    """
    world = _choice_fixture()
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(sts2_rs.state_to_dict(world))
    pc = _play_into_a_choice(world)
    mirror.advance({"action": "play", "card_index": 0})
    assert mirror.parked_choice() is not None

    doctored = sts2_rs.state_to_dict(world)
    doctored["player"]["hp"] -= 5
    mirror.reconcile(doctored)
    assert mirror.tainted and mirror.alive

    parked = mirror.parked_choice()
    assert parked is not None, "the reseed must salvage the derived choice"
    assert parked["kind"] == pc["kind"]
    assert all(c.get("pile") is not None and c.get("index") is not None for c in parked["cards"]), (
        "salvaged locators must point at the freshly seeded (world) piles"
    )

    # The DLL driver only knows card names — exactly the shape that used to
    # be refused as an unlocatable_injection after a reseed.
    name_only = [{"id": c["id"], "upgraded": c.get("upgraded", False)} for c in pc["cards"]]
    assert (
        mirror.inject_choice(
            pc["kind"], name_only, pc["min_select"], pc["max_select"], can_skip=pc["can_skip"]
        )
        == []
    )
    world.apply_step({"action": "choose", "selection": [0]})
    assert mirror.advance_through_choice([0]) == "advanced"


def test_a_salvaged_choice_keeps_the_options_the_wire_spells_differently(tmp_path):
    """The reseed salvage must relocate against the ENGINE, not the wire.

    A produced wire and a parked option are two dialects for one card:
    Sts2Headless / SpireMentor send ``curse:Decay``, the engine's derivation
    says ``DECAY``. ``_relocate_parked`` matched the two as raw strings, so
    every prefixed card failed to relocate and was dropped from the carried
    option list -- silently, because a dropped option is indistinguishable
    from an option the world never offered.

    The bug is invisible on an ordinary board and total on a cursed one, and
    the fixtures could not see it at all: both sides of every other test are
    ``state_to_dict`` output and share a dialect by construction, which is
    the same blind spot ``reconcile``'s parked-verify round-trip exists to
    close. Live, it is all six ``unlocatable_option`` records in
    ``logs/f7-mirror-check16`` -- the world offered a whole 8-card hand, the
    salvage came back with 7, and the Survivor / Dagger Throw discard fell
    to the rarity sort with a searchable choice one line away.
    """
    world = sts2_rs.build_scenario_state(
        1, "Silent", ["SURVIVOR"] * 5 + ["DECAY"] * 5, 70, 70, [], "SHRINKER_BEETLE_WEAK"
    )
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(sts2_rs.state_to_dict(world))
    pc = _play_into_a_choice(world)
    assert "DECAY" in [c["id"] for c in pc["cards"]], "the fixture must offer the curse"
    assert mirror.advance({"action": "play", "card_index": 0}) == "pending_choice"

    # Reality, in the producer's spelling, and wrong about the HP -- so the
    # mirror is dropped and has to salvage its derivation across the reseed.
    doctored = sts2_rs.state_to_dict(world)
    doctored["player"]["hp"] -= 5
    for pile in doctored["deck"].values():
        for card in pile:
            if card.get("id") == "DECAY":
                card["id"] = "curse:Decay"
    mirror.reconcile(doctored)
    assert mirror.tainted and mirror.alive

    parked = mirror.parked_choice()
    assert parked is not None
    assert [c["id"] for c in parked["cards"]] == [c["id"] for c in pc["cards"]], (
        "a card the wire spells differently is still the same card"
    )
    # And the salvage is usable: the world's own list injects clean.
    name_only = [{"id": c["id"], "upgraded": c.get("upgraded", False)} for c in pc["cards"]]
    assert (
        mirror.inject_choice(
            pc["kind"], name_only, pc["min_select"], pc["max_select"], can_skip=pc["can_skip"]
        )
        == []
    )


def test_a_fork_is_detached_from_the_fight_it_came_from(tmp_path):
    """Work done on a fork must not reach the mirror it was forked from.

    This is the whole point: answering a `card_select` is destructive by
    design (reconcile, replay, inject) and ends parked on a choice only the
    party that EXECUTES the picks can resolve. An advisor never executes, so
    it must do that work somewhere it can throw away.
    """
    world = _choice_fixture()
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(sts2_rs.state_to_dict(world))
    before = mirror.state.state_hash()

    twin = mirror.fork("test")
    assert twin is not None
    assert twin.state.state_hash() == before, "a fork starts as the same fight"

    # Advance the fork into a parked choice, then destroy it outright.
    assert twin.advance({"action": "play", "card_index": 0}) == "pending_choice"
    twin.drop("done")

    assert not twin.alive
    assert mirror.alive, "dropping a fork must not drop the fight"
    assert mirror.state.state_hash() == before, "the fight must not have moved"
    assert mirror.state.pending_choice() is None


def test_a_fork_does_not_share_the_fights_stats(tmp_path):
    """A throwaway's work is not the fight's history.

    Every rate the run log reports -- desyncs per decision above all -- is
    computed from these counters, so folding a fork's reconciles into them
    would quietly corrupt the measurement rather than fail it.
    """
    world = _choice_fixture()
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(sts2_rs.state_to_dict(world))
    twin = mirror.fork("test")
    assert twin is not None

    twin.reconcile(sts2_rs.state_to_dict(world))
    twin.advance({"action": "play", "card_index": 0})

    assert mirror.stats.reconciles == 0, "the fight did not reconcile; the fork did"
    assert twin.stats is not mirror.stats


def test_forking_a_dropped_mirror_yields_nothing(tmp_path):
    """A fork is an optimisation over declining, so it fails quietly."""
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    assert mirror.fork("test") is None
    mirror.seed(sts2_rs.state_to_dict(_choice_fixture()))
    mirror.drop("gone")
    assert mirror.fork("test") is None


def test_a_parked_divergence_is_returned_and_not_swallowed(tmp_path):
    """``reconcile`` must REPORT a parked-path divergence, like every other.

    The branch above reseeds and then used to
    ``return self._reseed_after_observe(wire)`` -- that helper's ``None``,
    from a method annotated ``-> list[Desync]``. Two bugs in one line: the
    report was thrown away, so a divergence found while a card was suspended
    mid-play was invisible to any caller that counts them; and the ``None``
    was a live landmine for the first caller that touched the value.

    It stayed hidden because every production caller ignored the return and
    ``test_a_parked_choice_survives_the_observe_reseed`` walks this exact
    branch without looking at it. The advisor's mirror was the first to read
    it, and died on ``len(None)`` -- taking one live decision with it, and
    then the run: the advisor answered "advisor error", autoplay had nothing
    to click, and the run stalled out at the screen
    (001-ironclad-a8-0824-2138, killed at 180s).
    """
    world = _choice_fixture()
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(sts2_rs.state_to_dict(world))
    _play_into_a_choice(world)
    mirror.advance({"action": "play", "card_index": 0})

    doctored = sts2_rs.state_to_dict(world)
    doctored["player"]["hp"] -= 5
    desyncs = mirror.reconcile(doctored)

    assert desyncs is not None, "a list-returning method must not return None"
    assert [d.path for d in desyncs] == ["$.player.hp"], (
        "the divergence that caused the reseed is what must be reported"
    )
    # The reseed still happened -- reporting it does not cost the recovery.
    assert mirror.tainted and mirror.alive


def test_a_reshuffle_does_not_fail_a_parked_verify(tmp_path):
    """The reshuffle blind spot applies to the parked path too.

    A card that reshuffles the draw pile and THEN opens a screen lands on the
    parked branch, and the world rolled its own order for whatever was
    reshuffled — so ``$.deck.*`` is UNVERIFIABLE, not wrong. Diffing it anyway
    is a false STRICT crash, and this is the one path that cannot recover by
    re-seeding, because re-seeding is what would delete the suspended card.
    """
    world = _choice_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    _play_into_a_choice(world)
    mirror.advance({"action": "play", "card_index": 0})

    # Stage the state the guard exists for: parked mid-play, and the advance
    # since the last reconcile consumed the shuffle stream.
    mirror._crossed_shuffle = True
    doctored = sts2_rs.state_to_dict(world)
    doctored["deck"]["draw_pile"].reverse()

    assert mirror.reconcile(doctored) == [], "a reshuffled pile is unverifiable, not a desync"
    assert mirror.stats.reshuffle_skips == 1
    assert mirror._crossed_shuffle, (
        "a parked verify keeps its own piles, so the blind spot is not resolved yet — "
        "only a rebuild earns the right to clear it"
    )

    # ...and a NON-pile field still diverges on the same board.
    doctored["player"]["hp"] -= 5
    with pytest.raises(MirrorDivergence, match=r"\$\.player\.hp"):
        mirror.reconcile(doctored)


def test_a_partial_multi_pick_is_finalized_not_left_open(tmp_path):
    """A world that picks fewer than ``max_select`` still lands on a closed
    choice.

    ``resolve_choice_action`` ACCUMULATES a single index on a multi-pick and
    re-opens, waiting for more; the world's ``resolve_combat_choice_selection``
    resolves in one shot. Left alone, the mirror would sit on a choice the
    world had already closed and every following decision would be answered
    against a board that no longer existed.
    """
    world = _choice_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    _play_into_a_choice(world)
    mirror.advance({"action": "play", "card_index": 0})
    mirror.reconcile(sts2_rs.state_to_dict(world))

    rolled = [{"id": "STRIKE_IRONCLAD"}, {"id": "BASH"}, {"id": "DEFEND_IRONCLAD"}]
    mirror.inject_choice("add_generated_to_hand", rolled, 1, 2)
    mirror.advance_through_choice([0])
    assert mirror.state.pending_choice() is None
    assert "STRIKE_IRONCLAD" in [
        c["id"] for c in sts2_rs.state_to_dict(mirror.state)["deck"]["hand"]
    ]


def test_a_grant_still_in_flight_defers_the_reconcile(tmp_path):
    """The choice channel's twin of the mid-card-resolution snapshot.

    ``SimCardSelector.GetSelectedCards`` returns the instant the pick arrives
    and nothing waits for the ``CardSelectCmd`` that awaited it to finish, so
    the next serialized board can show the screen closed and the card it
    granted still on its way into the hand. The mirror resolved the pick
    atomically, so it is AHEAD — not wrong — and STRICT must not raise.

    27 of the 49 pure ``$.deck.hand`` records in the ``mirror_live3`` corpus
    were exactly this: an ``add_generated_free_to_hand`` pick sitting in the
    mirror's hand against a world that had it in no pile at all.
    """
    world = _choice_fixture()
    pre = sts2_rs.state_to_dict(world)
    mirror = _mirror(tmp_path)  # STRICT
    mirror.seed(json.loads(json.dumps(pre)))

    mirror.inject_choice("add_generated_free_to_hand", [{"id": "BASH", "upgraded": False}], 0, 1)
    mirror.advance_through_choice([0])
    granted = sts2_rs.state_to_dict(mirror.state)
    assert "BASH" in [c["id"] for c in granted["deck"]["hand"]]

    # The world's next board still predates the grant.
    assert mirror.reconcile(json.loads(json.dumps(pre))) == []
    assert mirror.stats.choice_in_flight_skips == 1
    assert mirror.stats.desyncs == 0
    assert not mirror.tainted
    # The carried mirror survived: no reseed adopted the half-applied board,
    # which is what used to DELETE the grant and manufacture the mirror-BEHIND
    # echo one decision later (16 more of those 49 records).
    assert sts2_rs.state_to_dict(mirror.state) == granted
    # And the world catching up diffs clean.
    assert mirror.reconcile(json.loads(json.dumps(granted))) == []
    assert mirror.stats.desyncs == 0


def test_a_cost_the_world_has_not_zeroed_yet_defers_the_reconcile(tmp_path):
    """The same deferral for the OTHER grant flavour.

    ``make_free_in_hand`` leaves the picked card where it is and rewrites its
    cost, so the in-flight board shows both hands holding it — the mirror's at
    0, the world's still at the printed cost (6 of the 49 live3 records).
    """
    world = _choice_fixture()
    pre = sts2_rs.state_to_dict(world)
    mirror = _mirror(tmp_path)  # STRICT
    mirror.seed(json.loads(json.dumps(pre)))

    mirror.inject_choice(
        "make_free_in_hand",
        [{"id": "ARMAMENTS", "upgraded": False, "pile": "hand", "index": 2}],
        1,
        1,
    )
    mirror.advance_through_choice([0])
    granted = sts2_rs.state_to_dict(mirror.state)
    assert [c.get("cost_override") for c in granted["deck"]["hand"]][2] == 0

    assert mirror.reconcile(json.loads(json.dumps(pre))) == []
    assert mirror.stats.choice_in_flight_skips == 1
    assert sts2_rs.state_to_dict(mirror.state) == granted


def test_the_deferred_grant_is_still_graded_at_the_next_wire(tmp_path):
    """The grant deferral defers; it does not grant amnesty.

    Nothing re-arms the flag, so a world that never actually made the grant —
    a real modelling divergence hiding behind the same shape — is reported one
    decision later.
    """
    world = _choice_fixture()
    pre = sts2_rs.state_to_dict(world)
    mirror = _mirror(tmp_path)  # STRICT
    mirror.seed(json.loads(json.dumps(pre)))
    mirror.inject_choice("add_generated_free_to_hand", [{"id": "BASH", "upgraded": False}], 0, 1)
    mirror.advance_through_choice([0])

    assert mirror.reconcile(json.loads(json.dumps(pre))) == []
    with pytest.raises(MirrorDivergence, match=r"\$\.deck\.hand"):
        mirror.reconcile(json.loads(json.dumps(pre)))
    assert mirror.stats.choice_in_flight_skips == 1
    assert mirror.stats.desyncs == 1


def test_a_mirror_BEHIND_the_world_is_never_deferred(tmp_path):
    """Only the mirror-AHEAD shape is a snapshot artifact.

    A world holding a card the mirror does not is indistinguishable from the
    post-reseed echo this deferral exists to prevent and from a genuine missed
    grant, so it stays a divergence even in the grace window.
    """
    world = _choice_fixture()
    pre = sts2_rs.state_to_dict(world)
    mirror = _mirror(tmp_path)  # STRICT
    mirror.seed(json.loads(json.dumps(pre)))
    mirror.inject_choice("add_generated_free_to_hand", [{"id": "BASH", "upgraded": False}], 0, 1)
    mirror.advance_through_choice([])  # skipped: the mirror grants nothing

    ahead = json.loads(json.dumps(pre))
    ahead["deck"]["hand"].append(
        dict(ahead["deck"]["hand"][0], id="BASH", stable_idx=999, cost_override=0)
    )
    with pytest.raises(MirrorDivergence, match=r"\$\.deck\.hand"):
        mirror.reconcile(ahead)
    assert mirror.stats.choice_in_flight_skips == 0


def test_a_card_the_pick_never_named_is_not_deferred(tmp_path):
    """The grace window covers the picked card and nothing else.

    Widening it to "any single hand difference right after a choose" would
    give every screen a free pass on the one decision where the engine is most
    likely to be wrong about what the screen did.
    """
    world = _choice_fixture()
    pre = sts2_rs.state_to_dict(world)
    mirror = _mirror(tmp_path)  # STRICT
    mirror.seed(json.loads(json.dumps(pre)))
    mirror.inject_choice("add_generated_free_to_hand", [{"id": "BASH", "upgraded": False}], 0, 1)
    mirror.advance_through_choice([0])

    behind = json.loads(json.dumps(pre))
    behind["deck"]["hand"].pop()  # an ARMAMENTS the world lost, not the grant
    with pytest.raises(MirrorDivergence, match=r"\$\.deck\.hand"):
        mirror.reconcile(behind)
    assert mirror.stats.choice_in_flight_skips == 0


def test_any_other_action_closes_the_grant_grace_window(tmp_path):
    """The window is one board wide, and a committed action ends it.

    ``advance`` clears the flag, so a play between the pick and the reconcile
    cannot smuggle its own hand divergence through the deferral.
    """
    world = _choice_fixture()
    pre = sts2_rs.state_to_dict(world)
    mirror = _mirror(tmp_path)  # STRICT
    mirror.seed(json.loads(json.dumps(pre)))
    mirror.inject_choice("add_generated_free_to_hand", [{"id": "BASH", "upgraded": False}], 0, 1)
    mirror.advance_through_choice([0])
    assert mirror._choice_in_flight is not None
    mirror.advance({"action": "play", "card_index": 0})
    assert mirror._choice_in_flight is None


def test_the_live3_hand_class_replays_as_deferred(tmp_path):
    """Offline replay of the divergence class this closes.

    Every ``mirror_live3`` record whose ONLY diverging path was
    ``$.deck.hand`` and whose last action was an injected ``choose`` is fed
    back through the predicate. All 30 must defer; the 19 that must not are
    the post-reseed echoes (last action ``play``), which stop being produced
    at all once the direct records no longer reseed.
    """
    import combat_mirror
    from combat_mirror import Desync, _choice_effect_in_flight

    corpus = (
        pathlib.Path(combat_mirror.__file__).resolve().parents[1]
        / "logs"
        / "diag"
        / "mirror_live3.jsonl"
    )
    if not corpus.exists():  # pragma: no cover - corpus is not shipped everywhere
        pytest.skip(f"{corpus} not present")

    def _key(card: Any) -> tuple[Any, ...]:
        return tuple(tuple(v) if isinstance(v, list) else v for v in card)

    deferred = after_a_choose = 0
    for line in corpus.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if tuple(sorted(rec.get("paths") or [])) != ("$.deck.hand",):
            continue
        action = rec.get("last_action") or {}
        injection = rec.get("last_injection") or {}
        picks, options = injection.get("picks"), injection.get("options") or []
        if action.get("action") != "choose" or picks is None:
            continue
        after_a_choose += 1
        d = rec["desyncs"][0]
        one = [
            Desync(
                path=d["path"],
                mirror=[_key(c) for c in d["mirror"]],
                world=[_key(c) for c in d["world"]],
            )
        ]
        named = [options[i] for i in picks if 0 <= i < len(options)]
        deferred += bool(_choice_effect_in_flight(one, named))

    assert after_a_choose == 30
    assert deferred == 30


requires_turn_start_resume = pytest.mark.skipif(
    not hasattr(sts2_rs.CombatState, "arm_turn_start_resume"),
    reason="installed sts2_rs wheel predates CombatState.arm_turn_start_resume "
    "(mirror-fidelity round-4 defect #13)",
)


@requires_turn_start_resume
def test_gambling_chip_mulligan_resumes_the_rest_of_turn_setup(tmp_path):
    """Answering the world's turn-1 mulligan must also run what C# deferred.

    ``GamblingChip.AfterPlayerTurnStart`` opens a ``CardSelectCmd`` that C#
    ``Hook.AfterPlayerTurnStart`` *awaits*, so the wire the driver seeds from at
    that decision predates the rest of turn setup — Akabeko's ``VigorPower(8)``
    on the outer loop's ``Hook.AfterSideTurnStart`` among it. Injecting a bare
    ``discard_from_hand_and_draw`` and resolving it left the mirror permanently
    short that batch (18 v34wr records, seeds 1004 / 1047).
    """
    world = sts2_rs.build_scenario_state(
        11,
        "Ironclad",
        ["STRIKE_IRONCLAD"] * 10,
        70,
        80,
        ["GAMBLING_CHIP", "AKABEKO"],
        "SHRINKER_BEETLE_WEAK",
    )
    wire = sts2_rs.state_to_dict(world)
    # The engine parks the tail itself, so the world board it hands out is the
    # same mid-turn-setup snapshot the DLL serializes.
    # C# StartTurn runs Hook.AfterSideTurnStart once SetupPlayerTurn pauses
    # on the screen (CombatManager.cs:779-785 v0.111.0, :516-524 v0.107.1),
    # so Akabeko's Vigor is already on the board the screen is shown with.
    assert [(p["name"], p["amount"]) for p in wire["player"]["powers"]] == [("VIGOR", 8)], (
        "Akabeko's Vigor lands while the mulligan screen is up"
    )

    mirror = _mirror(tmp_path)
    mirror.seed(wire)
    hand = wire["deck"]["hand"]
    assert hand, "the mulligan needs an opening hand to offer"
    options = [
        {"id": c["id"], "upgraded": bool(c["upgraded"]), "pile": "hand", "index": i}
        for i, c in enumerate(hand)
    ]
    mirror.inject_choice("discard_from_hand_and_draw", options, 0, len(options), can_skip=True)
    assert mirror.arm_turn_start_resume("GAMBLING_CHIP")
    mirror.advance_through_choice([])

    assert mirror.state.pending_choice() is None
    powers = {
        (p["name"], p["amount"]) for p in sts2_rs.state_to_dict(mirror.state)["player"]["powers"]
    }
    assert ("VIGOR", 8) in powers, "the deferred turn-start batch must run on resolve"
    assert ("VIGOR", 16) not in powers, "the side walk that already ran must not run again"


requires_power_turn_start_resume = pytest.mark.skipif(
    not hasattr(sts2_rs.CombatState, "arm_turn_start_resume_after_power"),
    reason="installed sts2_rs wheel predates CombatState.arm_turn_start_resume_after_power "
    "(GH #355)",
)


def _tyranny_turn_start_world():
    """A world parked at Tyranny's turn-2 exhaust screen, mid-turn-setup.

    Both v0.107.1 relics with a turn-start card select are gated to turn 1, so
    a POWER is the only way to reach this shape on a later turn — which is
    exactly why the relic-only resume was a partial fix (GH #355).
    """
    world = sts2_rs.build_scenario_state(
        11,
        "Ironclad",
        ["TYRANNY"] + ["STRIKE_IRONCLAD"] * 9,
        70,
        80,
        [],
        "SHRINKER_BEETLE_WEAK",
    )
    world.apply_step({"action": "play", "card_index": 4})  # Tyranny
    assert world.apply_step({"action": "end_turn"}) == "pending_choice"
    return world


@requires_power_turn_start_resume
@pytest.mark.parametrize("armed", [False, True])
def test_tyranny_turn_start_screen_resumes_the_rest_of_turn_setup(tmp_path, armed):
    """Answering a POWER's turn-start screen must also run what C# deferred.

    ``TyrannyPower.AfterPlayerTurnStart`` opens a ``CardSelectCmd`` that C#
    ``Hook.AfterPlayerTurnStart`` *awaits*, so a mirror seeded from that wire
    is the same mid-turn-setup snapshot Gambling Chip's mulligan produces —
    on any turn, with no relic in sight. Both arms resolve the identical pick
    on mirror and world; only the armed one lands on the world's board.
    """
    world = _tyranny_turn_start_world()
    wire = sts2_rs.state_to_dict(world)
    assert wire["round"] == 2, "the point of this case is that it is NOT turn 1"

    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(wire)
    hand = wire["deck"]["hand"]
    options = [
        {"id": c["id"], "upgraded": bool(c["upgraded"]), "pile": "hand", "index": i}
        for i, c in enumerate(hand)
    ]
    mirror.inject_choice("exhaust_from_hand", options, 1, 1, can_skip=False)
    if armed:
        assert mirror.arm_turn_start_resume_after_power("TYRANNY")
    mirror.advance_through_choice([0])
    world.apply_step({"action": "choose", "selection": [0]})

    diffs = diff_observables(sts2_rs.state_to_dict(mirror.state), sts2_rs.state_to_dict(world))
    if armed:
        assert diffs == [], "the deferred turn-start tail must run on resolve"
    else:
        # The control: without the marker the mirror answers the screen and
        # then just stops — stranded in `PlayerDraw` with the rest of turn
        # setup unrun, which is the shape the corpus reported as "the mirror
        # is a round behind the world".
        assert [d.path for d in diffs] == ["$.phase"]


@requires_power_turn_start_resume
def test_arm_turn_start_resume_after_power_declines_a_board_without_the_power(tmp_path):
    """A no-op, not a double-fire, when the recognition was wrong.

    Same contract as the relic twin: parking a resume against a board that
    never suspended would re-run the whole turn-start batch.
    """
    world = _tyranny_turn_start_world()
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(sts2_rs.state_to_dict(world))
    # A choice IS open here, so this isolates the ownership check.
    hand = sts2_rs.state_to_dict(world)["deck"]["hand"]
    mirror.inject_choice(
        "exhaust_from_hand",
        [
            {"id": c["id"], "upgraded": bool(c["upgraded"]), "pile": "hand", "index": i}
            for i, c in enumerate(hand)
        ],
        1,
        1,
        can_skip=False,
    )
    assert mirror.arm_turn_start_resume_after_power("TOOLS_OF_THE_TRADE") is False


@requires_turn_start_resume
def test_arm_turn_start_resume_declines_a_board_that_never_suspended(tmp_path):
    """A no-op, not a double-fire, when the recognition was wrong.

    Parking a resume against a board whose turn setup already completed would
    re-run the whole `after_side_turn_start` batch — worse than the gap it fixes
    — so the engine refuses when the relic is not equipped.
    """
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(_fixture_state()))
    # No choice open at all.
    assert mirror.arm_turn_start_resume("GAMBLING_CHIP") is False


def test_answering_a_choice_that_is_not_open_is_a_divergence(tmp_path):
    """The world showing a selection the engine never opened means the two
    disagree about what the last card did."""
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(_fixture_state()))
    with pytest.raises(MirrorDivergence, match="no choice open"):
        mirror.advance_through_choice([0])


def test_grading_catches_an_arity_disagreement(tmp_path):
    """The earlier catch: a deterministic kind whose arity the world states
    differently is graded before anything is injected."""
    world = _choice_fixture()
    mirror = _mirror(tmp_path)
    mirror.seed(sts2_rs.state_to_dict(world))
    pc = _play_into_a_choice(world)
    mirror.advance({"action": "play", "card_index": 0})
    mirror.reconcile(sts2_rs.state_to_dict(world))
    with pytest.raises(MirrorDivergence, match=r"\$\.choice\.max_select"):
        mirror.inject_choice(pc["kind"], pc["cards"], 2, 2)


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 4, 11])
def test_mirror_tracks_a_native_engine_run_under_strict(tmp_path, seed):
    """A whole run, reconciled under STRICT, with nothing to report.

    Random play reaches a wide spread of boards fast — reshuffles, multi-enemy
    fights, mid-combat card selects — and every decision is checked. A failure
    here names the exact field that drifted.
    """
    mirror = _mirror(tmp_path, label=f"oracle-seed-{seed}")
    fights = drive_run(seed, "Ironclad", mirror, max_steps=1500)

    assert fights >= 1, "the oracle must actually reach a fight"
    assert mirror.stats.reconciles >= 20, "too few decisions to be a meaningful gate"
    assert mirror.stats.desyncs == 0
    assert not mirror.tainted


def test_oracle_exercises_the_injected_choice_path(tmp_path):
    """Teeth for the test above: a run with no mid-combat selection in it
    would pass ``test_mirror_tracks_a_native_engine_run_under_strict`` while
    leaving the entire choice path — the reason this project exists —
    completely untouched.
    """
    mirror = _mirror(tmp_path, label="oracle-injection")
    for seed in (1, 4, 11, 23, 42):
        drive_run(seed, "Ironclad", mirror, max_steps=1500)
        if mirror.stats.injections:
            break
    assert mirror.stats.injections > 0
    assert mirror.stats.desyncs == 0


@pytest.mark.slow
def test_oracle_sweep_finds_no_unclassified_divergence(tmp_path):
    """The broad gate: 5 characters × 20 seeds, every divergence classified.

    The narrow STRICT tests above stop at the first divergence on three
    hand-picked seeds, which is not a gate — it is a spot check. This runs the
    whole matrix and holds every divergence against
    :data:`KNOWN_RECONSTRUCTION_GAPS`, the documented list of fields
    ``state_dict.rs`` cannot yet reconstruct. A new mirror-machinery bug names
    a field that is not on that list and fails here.

    The ceilings are the measured frontier as of this commit. They exist so a
    *volume* regression fails too — a change that leaves the field set alone
    but doubles how often the mirror mispredicts. Tightening them as gaps
    close is the point; loosening them needs a reason in the commit message.

    They are also SEED-FROZEN: `clean_runs >= 95` / `divergences <= 5` were
    measured for exactly ``range(1, 21)`` × the five characters. Changing the
    seed range re-tunes both, and can also surface UNKNOWN classifications
    that are only an artifact of ``Gap.requires`` ordering (a run diverging on
    ``$.encounter[].powers`` without ``$.encounter[].intents`` has no chain to
    match). Re-measure rather than relax.

    Marked ``slow``: it drives 100 full runs, ~25 s, and ``addopts =
    -m 'not slow'`` would otherwise put all of that in the `just test` fast
    gate. It runs under `just test-slow` / `just test-all`.
    """
    report = sweep(seeds=range(1, 21), log_path=str(tmp_path / "divergences.jsonl"))

    assert report.runs == 100
    assert report.reconciles > 4000, "the sweep must actually reach deep boards"
    assert report.injections > 50, "the sweep must exercise mid-combat selections"
    assert not report.unknown, (
        "unclassified divergence(s) — these are mirror-machinery bugs, not "
        f"known wire gaps:\n{report.render()}"
    )
    # Measured on the whole v7 -> v20 series: 95 clean / 5 divergences over
    # 5372 reconciles. Nothing before the Osty creature-id aliasing fix is a
    # comparable baseline — until then a reconstructed Necrobinder aliased its
    # pet onto another creature's id, leaving a dangling power handle that
    # failed serialization outright and aborted the run mid-sweep, so the sweep
    # did not REACH this assert and every earlier ceiling was tuned against a
    # truncated matrix.
    #
    # The `$.encounter[].intents` class that dominated the residual at v15 (11
    # of 15) is gone: the four telegraph layers closed it.
    #
    # Re-measured 2026-08-26 at 98 clean / 2 divergences, both `$.deck.*`
    # draw-order (Silent-20, Necrobinder-2). The 95/5 ceiling above it was
    # taken before the reviver-roster slot STAMP in `state_dict.rs`: a
    # reconstructed board left every enemy slot-unknown, so a lone Fogmog
    # summoned its Eye behind itself instead of in front and
    # `$.encounter.roster` diverged (Ironclad-5, Defect-17). Tightening rather
    # than banking the slack is the point — a stale ceiling hides the next
    # three-run volume regression.
    assert report.clean_runs >= 98, f"clean-run count regressed:\n{report.render()}"
    assert report.divergences <= 2, f"divergence count regressed:\n{report.render()}"


def test_the_osty_is_a_compared_observable():
    """The Necrobinder pet absorbs damage aimed at the player, so a mirror
    whose Osty disagrees is planning against the wrong soak. It was not in the
    observable set at all, which is why a reconstruction that dropped the pet
    entirely could only surface indirectly as ``$.player.hp``."""
    wire = sts2_rs.state_to_dict(_fixture_state())
    mirror = json.loads(json.dumps(wire))
    world = json.loads(json.dumps(wire))
    mirror["player"]["osty"] = {"alive": True, "hp": 1, "max_hp": 1, "block": 0}
    world["player"]["osty"] = {"alive": True, "hp": 5, "max_hp": 11, "block": 0}
    assert "$.player.osty" in {d.path for d in diff_observables(mirror, world)}

    # A pet that agrees is not a divergence, and neither is no pet at all.
    world["player"]["osty"] = dict(mirror["player"]["osty"])
    assert diff_observables(mirror, world) == []


def _knight_board(order: list[tuple[str, bool]]) -> dict[str, Any]:
    """A multi-kind board built from the fixture's enemy, in the given order."""
    wire = sts2_rs.state_to_dict(_fixture_state())
    template = json.loads(json.dumps(wire["encounter"][0]))
    encounter = []
    for kind, dead in order:
        e = json.loads(json.dumps(template))
        e["kind"] = kind
        e["hp"], e["is_dead"] = (0, True) if dead else (40, False)
        if dead:
            e["intents"] = []
        encounter.append(e)
    wire["encounter"] = encounter
    return wire


def test_where_a_corpse_sits_in_the_roster_is_not_a_desync():
    """C# keeps dead-but-reviving creatures in their `Enemies` slots, but the
    headless producer serializes them into a separate `dead_enemies` list the
    adapter re-appends dead-LAST, while the engine wire keeps corpses inline
    at their spawn slots. Both interleavings describe the same board — the d1
    corpus flagged every post-death Parafright/Decimillipede fight over it."""
    mirror = _knight_board([("Parafright", True), ("TheObscura", False)])
    world = _knight_board([("TheObscura", False), ("Parafright", True)])
    assert diff_observables(mirror, world) == []


def test_the_alive_orders_stay_load_bearing_after_canonicalization():
    """Only corpses float; a re-ordered LIVING pair is still the ±8 Juggernaut
    mis-target the positional roster comparison exists to catch."""
    mirror = _knight_board([("SpectralKnight", False), ("MagiKnight", False)])
    world = _knight_board([("MagiKnight", False), ("SpectralKnight", False)])
    assert "$.encounter.roster" in {d.path for d in diff_observables(mirror, world)}


def test_a_reap_does_not_resort_the_surviving_roster():
    """Dropping a reaped corpse must FILTER the mirror's roster, not re-sort
    it: survivors in non-lexical order (SpectralKnight before MagiKnight)
    diverged on every decision after the reap, for the rest of the fight."""
    mirror = _knight_board(
        [("FlailKnight", True), ("SpectralKnight", False), ("MagiKnight", False)]
    )
    world = _knight_board([("SpectralKnight", False), ("MagiKnight", False)])
    assert diff_observables(mirror, world) == []


# ---------------------------------------------------------------------------
# GH #368.1 — the damage-source trace attached to a divergence record.
# ---------------------------------------------------------------------------


def test_a_divergence_record_names_the_effect_that_spent_the_hp(tmp_path, monkeypatch):
    """The whole point of the trace: an HP desync record must carry WHICH of
    the mirror's effects dealt the damage, not just how much the two boards
    disagree by. Without it, a reproducible magnitude (round 7d's ``+6`` on
    the Defect arm) is unattributable, because the mirror is re-imposed from
    reality one decision later and the evidence expires.
    """
    monkeypatch.setenv("STS2_MIRROR_DAMAGE_TRACE", "1")
    world = _fixture_state()
    log = tmp_path / "divergences.jsonl"
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(sts2_rs.state_to_dict(world))

    _end_turn(world)
    mirror.advance({"action": "end_turn"})
    doctored = sts2_rs.state_to_dict(world)
    doctored["player"]["hp"] -= 5
    mirror.reconcile(doctored)

    record = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    trace = record.get("damage_trace")
    assert trace, "an end-turn that took a Nibbit attack must produce rows"
    # The enemy's turn ran inside the mirror's step, so its intent frame is
    # what names the damage.
    assert any(r["source"].startswith("intent:") for r in trace), trace
    row = next(r for r in trace if r["source"].startswith("intent:"))
    assert row["target_id"] == "player"
    assert row["hp_before"] - row["hp_after"] == row["hp_loss"]


def test_the_trace_is_absent_unless_the_sweep_armed_it(tmp_path, monkeypatch):
    """Off by default. A sweep that did not ask for traces must not pay the
    record-size cost — and `diag_damage_source` distinguishes "not armed" from
    "no damage" by the key's absence, so a stray empty list would mislead it.
    """
    monkeypatch.delenv("STS2_MIRROR_DAMAGE_TRACE", raising=False)
    world = _fixture_state()
    log = tmp_path / "divergences.jsonl"
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(sts2_rs.state_to_dict(world))

    _end_turn(world)
    mirror.advance({"action": "end_turn"})
    doctored = sts2_rs.state_to_dict(world)
    doctored["player"]["hp"] -= 5
    mirror.reconcile(doctored)

    record = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert "damage_trace" not in record


def test_a_clean_decision_does_not_leak_its_damage_into_the_next_record(tmp_path, monkeypatch):
    """The window is per-decision. A turn that reconciled clean must not have
    its rows show up in the NEXT turn's divergence — that would attribute
    damage to a step that never dealt it, which is worse than no trace.
    """
    monkeypatch.setenv("STS2_MIRROR_DAMAGE_TRACE", "1")
    world = _fixture_state()
    log = tmp_path / "divergences.jsonl"
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(sts2_rs.state_to_dict(world))

    # Two clean turns first — their damage must be discarded at each reconcile.
    for _ in range(2):
        _end_turn(world)
        mirror.advance({"action": "end_turn"})
        assert mirror.reconcile(sts2_rs.state_to_dict(world)) == []

    _end_turn(world)
    mirror.advance({"action": "end_turn"})
    doctored = sts2_rs.state_to_dict(world)
    doctored["player"]["hp"] -= 5
    mirror.reconcile(doctored)

    record = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    trace = record["damage_trace"]
    turns = {r["turn"] for r in trace}
    assert len(turns) == 1, f"one decision's window, but saw turns {turns}"


def test_the_echo_form_budget_survives_a_reconcile(tmp_path):
    """GH #368.1: `cards_first_in_series_this_turn` is hidden state.

    No producer emits it, so `state_from_dict` rebuilds it as 0 and EchoForm's
    "have I already echoed this turn?" gate re-opens at every decision — on the
    mirrored arms, once per card. Measured on a pre-fix DLL sweep: 23 of 23
    `+6` enemy-HP desyncs had EchoForm live, and 19 showed one card hitting
    the same creature twice.

    The transplant is the seam that has to carry it, and it is the only half
    that can carry it in full: a per-power counter cannot count the Echo Form
    card's OWN play, because the power did not exist yet — and that is exactly
    the case that survived the first attempt at this fix (9 of the 28 positive
    enemy-HP desyncs left on the post-fix sweep).
    """
    world = _fixture_state()
    mirror = _mirror(tmp_path, policy=MirrorPolicy.OBSERVE)
    mirror.seed(sts2_rs.state_to_dict(world))

    action = {"action": "play", "card_index": 0, "target_index": 0}
    world.apply_step(action)
    mirror.advance(action)
    assert mirror.state.cards_first_in_series_this_turn() == 1

    assert mirror.reconcile(sts2_rs.state_to_dict(world)) == []
    assert mirror.state.cards_first_in_series_this_turn() == 1, (
        "the reconstruction re-zeroed the EchoForm gate; the transplant must carry it"
    )


def test_a_disclosed_turn_history_is_reality_not_carried():
    """GH #989: the turn's exhausts and card block gains ride the wire from
    newer mods (zero included); then the reconcile lets reality win on them,
    and only an older producer's wire leaves them to the carried mirror."""
    from combat_mirror import _turn_history_disclosed

    assert not _turn_history_disclosed({"player": {}})
    assert not _turn_history_disclosed({"player": {"cards_exhausted_this_turn": 0}})
    assert _turn_history_disclosed(
        {"player": {"cards_exhausted_this_turn": 0, "card_block_gains_this_turn": 0}}
    )
