#!/usr/bin/env python3
"""Assert that an unattended F7 (advisor autoplay) run did not stall.

Reads the game's ``godot.log`` and checks the failure signatures of the six
defects reported on 2026-08-06. Every check is written against a line the mod
actually prints, so a check that cannot see its evidence says INCONCLUSIVE
rather than PASS -- a run that never reached a shop proves nothing about the
shop, and reporting that as a pass is how a harness lies.

Usage::

    python spire-mentor-mod/tools/check_f7_log.py                 # default log path
    python spire-mentor-mod/tools/check_f7_log.py path/to/godot.log

Exit code 0 when nothing failed (inconclusive checks do not fail the run),
1 when any check failed, 2 when the log could not be read.

How to produce a log to check -- see ``SimConfig.AdvisorAutoplay``:

1. Deploy the mod (``dotnet build`` in spire-mentor-mod does it).
2. Write a SpireMentor.conf next to SpireMentor.dll with ``auto_start: true``,
   ``advisor_autoplay: true``, ``client_run_config: false``, ``num_runs: 1``
   and (to reach the deep screens with a deliberately weak advisor)
   ``debug_player_hp: 999``.
3. Point the advisor at a small search so screens arrive quickly, e.g. put
   ``--sims 8 --boss-sims 8`` in SimAdvisor.cmd.
4. Launch the game detached, REDIRECTING ITS STDOUT (it floods a terminal
   otherwise), wait, then run this script.
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_LOG = (
    Path(os.environ.get("APPDATA", "")) / "SlayTheSpire2" / "logs" / "godot.log"
)

DECISION_RE = re.compile(r"\[SimAdvisor\] Decision up: (\S+) \[(.*)\]$")
TAKEOVER_RE = re.compile(r"\[SimAdvisor\] Takeover executed on (\S+): (.*)$")
DESCRIBE_FAIL_RE = re.compile(r"\[SimScreenAdvice\] (\S+) failed: (.*)$")
RUNAWAY_RE = re.compile(r"\[SimAdvisor\] Takeover: (\S+) has not moved after")
PAUSE_RE = re.compile(r"\[SimAdvisor\] Paused for the player on (\S+): (.*)$")

# Screens where repeating the same choice is a real defect rather than a
# legitimately repeated step. The loot screen genuinely claims one reward per
# click, and the crystal sphere genuinely reveals one cell per click -- both
# carry a changing signature, so they are excluded by name rather than by
# accident.
REPEAT_SENSITIVE = (
    "NDeckUpgradeSelectScreen",
    "NDeckEnchantSelectScreen",
    "NDeckCardSelectScreen",
    "NDeckTransformSelectScreen",
    "NSimpleCardSelectScreen",
    "NCardRewardSelectionScreen",
    "NEventRoom",
    "NRestSiteRoom",
    "NTreasureRoom",
    "NMapScreen",
)

# More than this many identical (screen, label) choices in a row is the
# signature of the confirm bug: the click landed, the screen did not move, and
# the poll asked again. Two is allowed because a deck screen legitimately takes
# "pick" then "confirm", and a Min != Max screen takes "confirm" twice (once to
# open the preview, once to commit it).
REPEAT_LIMIT = 3


@dataclass
class Check:
    name: str
    status: str  # PASS / FAIL / INCONCLUSIVE
    detail: str = ""
    evidence: list[str] = field(default_factory=list)


def load(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def check_describe_errors(lines: list[str]) -> Check:
    """No describer should throw. The treasure room used to throw on every
    poll, and one shared catch made that blind every other screen."""
    fails = Counter()
    for line in lines:
        m = DESCRIBE_FAIL_RE.search(line)
        if m:
            fails[f"{m.group(1)}: {m.group(2)}"] += 1
    if not fails:
        return Check("describe-errors", "PASS", "no describer threw")
    worst = fails.most_common(3)
    return Check(
        "describe-errors",
        "FAIL",
        f"{sum(fails.values())} describer exceptions",
        [f"{n}x {what}" for what, n in worst],
    )


def check_no_repeats(lines: list[str]) -> Check:
    """The headline defect: the same choice executed over and over because the
    screen never closed (59 consecutive enchant picks in the reported run)."""
    runs: list[tuple[str, str, int]] = []
    prev: tuple[str, str] | None = None
    count = 0
    for line in lines:
        m = TAKEOVER_RE.search(line)
        if not m:
            continue
        key = (m.group(1), m.group(2).strip())
        if key == prev:
            count += 1
        else:
            if prev is not None:
                runs.append((prev[0], prev[1], count))
            prev, count = key, 1
    if prev is not None:
        runs.append((prev[0], prev[1], count))

    if not runs:
        return Check("no-repeated-choices", "INCONCLUSIVE", "no screen choices executed")
    bad = [
        r for r in runs if r[2] >= REPEAT_LIMIT and r[0].startswith(REPEAT_SENSITIVE)
    ]
    if bad:
        return Check(
            "no-repeated-choices",
            "FAIL",
            f"{len(bad)} screen(s) took the same choice {REPEAT_LIMIT}+ times in a row",
            [f"{screen}: '{label}' x{n}" for screen, label, n in bad],
        )
    worst = max(runs, key=lambda r: r[2])
    return Check(
        "no-repeated-choices",
        "PASS",
        f"{len(runs)} choice runs, longest {worst[2]}x ({worst[0]}: {worst[1]})",
    )


def check_no_cycles(lines: list[str]) -> Check:
    """Consecutive repeats are only the easy shape. The card-reward stall was a
    two-step CYCLE -- claim the reward, skip its screen, the loot screen
    re-offers it -- where neither choice ever repeated back to back, so the
    consecutive check sailed straight past it (2026-08-06). Count occurrences
    in a sliding window instead."""
    window: list[tuple[str, str]] = []
    offenders: Counter = Counter()
    size = 8
    limit = 3
    for line in lines:
        m = TAKEOVER_RE.search(line)
        if not m:
            continue
        key = (m.group(1), m.group(2).strip())
        window.append(key)
        del window[:-size]
        if window.count(key) > limit and key[0].startswith(REPEAT_SENSITIVE + ("NRewardsScreen",)):
            offenders[key] += 1
    if not offenders:
        return Check("no-choice-cycles", "PASS", "no decision came round repeatedly")
    return Check(
        "no-choice-cycles",
        "FAIL",
        f"{len(offenders)} decision(s) cycled",
        [f"{s}: '{lbl}' recurred {n}x in an 8-choice window" for (s, lbl), n in offenders.most_common(4)],
    )


def check_runaway_guard(lines: list[str]) -> Check:
    """The guard firing means something still stalls -- it is a backstop, not
    an expected outcome."""
    hits = [m.group(1) for line in lines if (m := RUNAWAY_RE.search(line))]
    if hits:
        return Check(
            "runaway-guard-silent",
            "FAIL",
            f"guard stopped autoplay {len(hits)} time(s)",
            sorted(set(hits)),
        )
    return Check("runaway-guard-silent", "PASS", "guard never had to fire")


def _pauses(lines: list[str]) -> list[str]:
    return [f"{m.group(1)}: {m.group(2)}" for line in lines if (m := PAUSE_RE.search(line))]


def check_no_pauses(lines: list[str]) -> Check:
    """`pause_on` handing a decision back is deliberate, not a stall (GH #874)
    -- but an unattended run has nobody to hand it to, and `advisor_autoplay`
    forces `pause_on` empty. A pause here means that forcing broke, so it is
    reported under its own name rather than as a run that went nowhere."""
    pauses = _pauses(lines)
    if pauses:
        return Check(
            "no-intentional-pause",
            "FAIL",
            f"autoplay handed back {len(pauses)} decision(s) ON PURPOSE (pause_on) -- "
            "not a stall, but an unattended run must never pause",
            pauses[:5],
        )
    return Check("no-intentional-pause", "PASS", "autoplay never handed a decision back")


def _decisions(lines: list[str]) -> list[tuple[str, str]]:
    return [
        (m.group(1), m.group(2))
        for line in lines
        if (m := DECISION_RE.search(line))
    ]


def check_card_select_confirms(lines: list[str]) -> Check:
    """Every deck screen that opened should have produced a `confirm`
    decision -- that is the fix, and its absence means the old `%Confirm`-only
    path silently did nothing again."""
    decisions = _decisions(lines)
    deck_screens = {
        "NDeckUpgradeSelectScreen",
        "NDeckEnchantSelectScreen",
        "NDeckCardSelectScreen",
        "NDeckTransformSelectScreen",
    }
    opened = [d for d in decisions if d[0] in deck_screens]
    if not opened:
        return Check(
            "card-select-confirm",
            "INCONCLUSIVE",
            "no deck-select screen opened (no smith / enchant / purge this run)",
        )
    picks = [d for d in opened if not d[1].startswith("card_select:confirm")]
    # Count the confirm CLICKS, not the "Decision up: ... [card_select:confirm]"
    # lines. A confirm reached through ExecuteScreenChoice's own re-ask never
    # passes through the poll, so it executes without ever printing a decision
    # line -- counting those undercounts a working run (3 picks, 2 lines, 3
    # confirms, observed 2026-08-06).
    confirms = [
        m.group(2).strip()
        for line in lines
        if (m := TAKEOVER_RE.search(line))
        and m.group(1) in deck_screens
        and m.group(2).strip().startswith("confirm")
    ]
    # One confirm does not mean one card. A screen with MinSelect == MaxSelect
    # > 1 (the Amalgamator's "combine two Strikes") takes N picks and then
    # commits them all at once, labelled `confirm (N cards)`. Comparing clicks
    # to picks failed such a run as "a screen was left open" when every screen
    # had in fact been finished -- count the cards each confirm accounted for.
    def _cards_committed(label: str) -> int:
        m = re.search(r"confirm \((\d+) cards\)", label)
        return int(m.group(1)) if m else 1

    committed = sum(_cards_committed(label) for label in confirms)
    if not confirms:
        return Check(
            "card-select-confirm",
            "FAIL",
            f"{len(picks)} deck-screen pick(s) but no confirm was ever clicked",
            [f"{s} [{sig}]" for s, sig in opened[:5]],
        )
    if committed < len(picks):
        return Check(
            "card-select-confirm",
            "FAIL",
            f"{len(picks)} pick(s) but only {committed} card(s) committed by "
            f"{len(confirms)} confirm(s) -- a screen was left open",
        )
    return Check(
        "card-select-confirm",
        "PASS",
        f"{len(picks)} pick(s) -> {committed} card(s) committed by {len(confirms)} confirm(s)",
    )


def check_treasure(lines: list[str]) -> Check:
    """The chest must be opened by the mod, not left for a human."""
    decisions = _decisions(lines)
    treasure = [sig for screen, sig in decisions if screen == "NTreasureRoom"]
    if not treasure:
        return Check("treasure-chest", "INCONCLUSIVE", "no treasure room this run")
    opened = any("open the chest" in sig for sig in treasure)
    took = any(
        "open the chest" not in sig and "proceed" not in sig for sig in treasure
    )
    if not opened:
        return Check(
            "treasure-chest",
            "FAIL",
            "treasure room described but never offered 'open the chest'",
            treasure[:5],
        )
    return Check(
        "treasure-chest",
        "PASS",
        f"chest opened{' and a relic was offered' if took else ''}",
    )


def check_shop_inventory(lines: list[str]) -> Check:
    """The shopfront has to be open, or the player watches the AI buy nothing
    visible."""
    decisions = _decisions(lines)
    shops = [sig for screen, sig in decisions if screen == "NMerchantRoom"]
    if not shops:
        return Check("shop-inventory", "INCONCLUSIVE", "no shop this run")
    opened = sum(
        1 for line in lines if "Opening the merchant inventory" in line
    )
    if not opened:
        return Check(
            "shop-inventory",
            "FAIL",
            "shop described but the inventory was never opened",
            shops[:3],
        )
    # Reopening the shopfront AFTER deciding to leave means the poll got
    # between the two clicks of the exit and re-armed the room -- and the
    # reopen disables the very Proceed the exit is about to click. Both live
    # exits did this before the in-flight flag moved into LeaveShopAsync
    # (2026-08-06); reading it out of the log by eye is not a test.
    reopens = 0
    leaving = False
    for line in lines:
        if "Takeover executed on NMerchantRoom" in line and "leave shop" in line.lower():
            leaving = True
            continue
        if leaving and "Opening the merchant inventory" in line:
            reopens += 1
            leaving = False
        elif "Decision up:" in line and "NMerchantRoom" not in line:
            leaving = False
    if reopens:
        return Check(
            "shop-inventory",
            "FAIL",
            f"inventory opened {opened}x, but reopened {reopens}x while leaving "
            "(the exit races the screen poll)",
        )
    return Check("shop-inventory", "PASS", f"inventory opened {opened}x, none during an exit")


def check_stale_rooms(lines: list[str]) -> Check:
    """A room decision must never appear AFTER the map has been travelled --
    that is the stale Proceed that reopened the map over the next room."""
    decisions = _decisions(lines)
    rooms = {"NRestSiteRoom", "NTreasureRoom", "NMerchantRoom", "NEventRoom"}
    offenders: list[str] = []
    # A room decision is stale when it is the FIRST thing described after a map
    # takeover and names a room whose signature says nothing is left but
    # 'proceed' -- the shape of every occurrence in the reported log.
    awaiting_travel = False
    for line in lines:
        if TAKEOVER_RE.search(line) and "on NMapScreen:" in line:
            awaiting_travel = True
            continue
        m = DECISION_RE.search(line)
        if not m:
            continue
        screen, sig = m.group(1), m.group(2)
        if awaiting_travel and screen in rooms and sig.endswith("proceed"):
            offenders.append(f"{screen} [{sig}] straight after a map travel")
        awaiting_travel = False
    if offenders:
        return Check(
            "no-stale-room-proceed",
            "FAIL",
            f"{len(offenders)} stale room decision(s) after a map click",
            offenders[:5],
        )
    travels = sum(1 for line in lines if "Takeover executed on NMapScreen:" in line)
    if travels == 0:
        return Check("no-stale-room-proceed", "INCONCLUSIVE", "no map travel this run")
    return Check(
        "no-stale-room-proceed", "PASS", f"{travels} map travels, none followed by a stale room"
    )


def check_progress(lines: list[str]) -> Check:
    """Did the run actually go anywhere? Everything above can pass vacuously
    on a log where autoplay never started."""
    armed = any("Arming autoplay" in line for line in lines)
    floors = [
        line for line in lines if "Takeover executed on NMapScreen:" in line
    ]
    if not armed:
        return Check(
            "run-progressed",
            "FAIL",
            "autoplay was never armed (advisor_autoplay did not fire)",
        )
    if len(floors) < 3:
        pauses = _pauses(lines)
        return Check(
            "run-progressed",
            "FAIL",
            f"autoplay armed but only {len(floors)} floor(s) travelled"
            + (" -- it was handed back by pause_on, not stalled" if pauses else ""),
            pauses[:1],
        )
    return Check("run-progressed", "PASS", f"{len(floors)} floors travelled")


CHECKS = (
    check_progress,
    check_describe_errors,
    check_no_repeats,
    check_no_cycles,
    check_runaway_guard,
    check_no_pauses,
    check_card_select_confirms,
    check_treasure,
    check_shop_inventory,
    check_stale_rooms,
)


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_LOG
    try:
        lines = load(path)
    except OSError as exc:
        print(f"could not read {path}: {exc}")
        return 2

    print(f"log: {path}  ({len(lines)} lines)\n")
    results = [check(lines) for check in CHECKS]
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"  {r.status:<13} {r.name:<{width}}  {r.detail}")
        for line in r.evidence:
            print(f"                {' ' * width}  - {line}")

    failed = [r for r in results if r.status == "FAIL"]
    unknown = [r for r in results if r.status == "INCONCLUSIVE"]
    print()
    print(
        f"{len(results) - len(failed) - len(unknown)} passed, "
        f"{len(failed)} failed, {len(unknown)} inconclusive"
    )
    if unknown:
        print("inconclusive checks never saw their screen -- run longer, not a pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
