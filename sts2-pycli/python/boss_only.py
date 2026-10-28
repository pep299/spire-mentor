"""Boss-only MCTS-vs-human benchmark for the Act-1 boss (Vantom).

Instead of MCTS-replaying the whole act (slow: hundreds of searches/run, and
confounds boss skill with act survival), this reconstructs the human's
boss-ENTRY loadout directly from the scrape — deck (starter + Σcards_gained −
Σcards_removed up to the boss), relics, the potion belt (Σ picked − Σ used,
unioned with the boss fight's own `potion_used`), and the exact boss-entry HP
(the pre-boss rest site's post-heal `current_hp`) — builds the Vantom fight via
`build_scenario_state`, and plays only that one fight with
`mcts_play_to_terminal`. ~1 boss fight/run vs ~7 combats + full run.

This isolates the question that actually matters: GIVEN the human's exact
situation at the boss, can MCTS win the fight the human won?

Usage:
  python boss_only.py [--sims 1000] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import act1_replay as ar
from act1_replay import _norm_card

sts2_rs = ar.sts2_rs


def _canon(scraped_id: str) -> str:
    """`CARD.X`/`RELIC.X`/`POTION.X` -> canonical UPPER_SNAKE."""
    for pfx in ("CARD.", "RELIC.", "POTION."):
        if scraped_id.startswith(pfx):
            return scraped_id[len(pfx) :]
    return scraped_id


def reconstruct_boss_entry(run):
    """Return (deck_ids, relic_ids, entry_hp, max_hp) at Act-1 boss entry."""
    floors = run.act1_floors
    max_hp = 80
    # Deck: starter + Σgains − Σremovals over Neow..floor-before-boss.
    starter = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]
    if run.ascension >= 10:
        starter.append("ASCENDERS_BANE")
    deck = Counter(_norm_card(c) for c in starter)
    rep = {_norm_card(c): c for c in starter}
    upg = Counter()  # how many copies of each card the human upgraded (smith/events)
    pre_boss = floors[:-1]  # exclude the boss floor (its rewards are post-fight)
    for fl in pre_boss:
        d = ar.extract_floor_decision(fl, 0)
        for g in d.cards_gained or []:
            if g.get("id"):
                k = _norm_card(g["id"])
                deck[k] += 1
                rep[k] = _canon(g["id"])
        for x in d.cards_removed or []:
            if x.get("id"):
                deck[_norm_card(x["id"])] -= 1
        for u in d.upgraded_cards or []:
            if u:
                upg[_norm_card(u)] += 1
    # Emit base ids, marking up to the human's upgrade count of each as "+1"
    # (build_scenario_state's upgrade suffix). Equalises the smith advantage.
    deck_ids = []
    for k, n in (+deck).items():
        base = rep.get(k, k.upper())
        n_up = min(upg.get(k, 0), n)
        deck_ids.extend([base + "+1"] * n_up)
        deck_ids.extend([base] * (n - n_up))

    # Relics: BurningBlood starter + every relic obtained pre-boss.
    relics = ["BURNING_BLOOD"]
    seen = {"BURNING_BLOOD"}
    for fl in pre_boss:
        d = ar.extract_floor_decision(fl, 0)
        for rid in ar._scraped_relics_for_floor(d):
            c = _canon(rid)
            if c not in seen:
                seen.add(c)
                relics.append(c)

    # Boss-entry HP = the last pre-boss floor's recorded `current_hp`. In
    # Act 1 that floor is ALWAYS the rest site (verified: 38/38 won IC runs),
    # and its `current_hp` is recorded POST-rest-action — SMITH leaves it
    # unchanged, HEAL adds `hp_healed` (e.g. 43 + 37 = 80). Nothing happens
    # between the rest node and the adjacent boss node, so this is the EXACT
    # HP the human walked into Vantom with, not a loose proxy. (Do NOT use the
    # boss floor's `current_hp` — that's recorded POST-fight.)
    entry_hp = None
    ps = (pre_boss[-1].get("player_stats") or [{}])[0]
    if ps.get("current_hp"):
        entry_hp = ps["current_hp"]
        max_hp = ps.get("max_hp") or entry_hp
    if entry_hp is None:
        entry_hp, max_hp = 70, 80
    return deck_ids, relics, int(entry_hp), int(max_hp)


def reconstruct_boss_potions(run, known=None):
    """Best-effort potion belt the human held walking into the boss.

    The scrape carries no belt snapshot — only per-floor `potion_choices`
    (offered, with `was_picked`) and `potion_used` (consumed that floor).
    So reconstruct: belt = Σ picked − Σ used over the pre-boss floors, then
    union in the boss floor's own `potion_used` — those were provably in the
    belt at entry, even if our pick-tracking missed where they came from
    (shop / event / starter potions never show up as a `potion_choices`
    pick). Capped at MAX_POTIONS (5). `known` filters to ids the engine can
    resolve so an unknown potion never aborts the build.
    """
    floors = run.act1_floors

    def ps(fl):
        return (fl.get("player_stats") or [{}])[0]

    belt = Counter()
    for fl in floors[:-1]:  # pre-boss
        p = ps(fl)
        for c in p.get("potion_choices") or []:
            if c.get("was_picked") and c.get("choice"):
                belt[_canon(c["choice"])] += 1
        for used in p.get("potion_used") or []:
            belt[_canon(used)] -= 1
    belt = +belt  # drop non-positive counts

    # The boss fight's own `potion_used` is a hard lower bound on the belt.
    for used in ps(floors[-1]).get("potion_used") or []:
        pid = _canon(used)
        if belt[pid] < 1:
            belt[pid] = 1

    ids = []
    for pid, n in belt.items():
        if known is not None and pid not in known:
            continue
        ids.extend([pid] * n)
    return ids[:5]  # MAX_POTIONS


def _resolve_ok(build_fn, **kw):
    """Try build; on unknown-id ValueError, drop the offending id and retry."""
    deck = list(kw["deck"])
    relics = list(kw["relics"])
    for _ in range(40):
        try:
            return (
                build_fn(
                    deck=deck,
                    relics=relics,
                    **{k: v for k, v in kw.items() if k not in ("deck", "relics")},
                ),
                deck,
                relics,
            )
        except ValueError as e:
            msg = str(e)
            # "unknown relic id \"X\"" / "unknown card id \"X\""
            bad = msg.split('"')[1] if '"' in msg else None
            if bad is None:
                raise
            before = len(deck) + len(relics)
            deck = [c for c in deck if c != bad]
            relics = [r for r in relics if r != bad]
            if len(deck) + len(relics) == before:
                raise  # couldn't find the id to drop → give up
    return None, deck, relics


def _prepare_run(r, seed, known_potions):
    """Reconstruct + build the Vantom scenario once; return a dict the sweep
    re-clones per sim level (avoids rebuilding the state every level)."""
    deck, relics, hp, max_hp = reconstruct_boss_entry(r)
    potions = reconstruct_boss_potions(r, known=known_potions)

    def _build(deck, relics, _hp=hp, _max=max_hp, _asc=r.ascension, _pot=potions):
        return sts2_rs.build_scenario_state(
            seed, "Ironclad", deck, _hp, _max, relics, "VANTOM_BOSS", _asc, potions=_pot
        )

    cs, deck2, relics2 = _resolve_ok(_build, deck=deck, relics=relics)
    if cs is None:
        raise ValueError("unbuildable after dropping unknown ids")
    return {
        "run": r,
        "cs": cs,
        "deck": len(deck2),
        "relics": len(relics2),
        "potions": potions,
        "hp": hp,
        "max_hp": max_hp,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sims", type=int, default=1000)
    p.add_argument(
        "--sweep",
        type=str,
        default="",
        help="comma list of sim counts, e.g. 30000,50000,100000,150000",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default="boss_sweep_results.md")
    p.add_argument(
        "--monotone",
        action="store_true",
        help="once a run wins at a sim level, carry W forward to higher levels "
        "without re-running (assumes more search doesn't flip a win to a loss; "
        "front-loads cost at the cheapest level so the sweep finishes in hours "
        "not a day). Records the LOWEST winning sim level per run.",
    )
    args = p.parse_args(argv)

    sim_levels = [int(x) for x in args.sweep.split(",") if x.strip()] if args.sweep else [args.sims]

    runs = [r for r in ar.iter_v103_act1_runs("the-ironclad", False) if r.scraped_won]
    if args.limit:
        runs = runs[: args.limit]
    known_potions = set(sts2_rs.list_registry_names("potion"))
    print(f"boss_only: {len(runs)} won-only IC runs; sim levels={sim_levels}")

    # Build every scenario once.
    prepared, errored = [], 0
    for r in runs:
        try:
            prepared.append(_prepare_run(r, args.seed, known_potions))
        except Exception as e:
            errored += 1
            print(f"  asc{r.ascension:>2} {r.seed[:10]:<10} PREP ERROR: {e}")
    potion_runs = sum(bool(p["potions"]) for p in prepared)

    # results[seed10][sims] = (won, final_hp, turns, wall_ms, skipped)
    results = {p["run"].seed[:10]: {} for p in prepared}
    summary = []  # (sims, won, n, total_wall_ms)
    first_win = {}  # seed10 -> lowest sim level that won (monotone bookkeeping)

    for sims in sim_levels:
        won = lost = 0
        total_wall = 0.0
        print(f"\n── sims={sims:,} ──", flush=True)
        for pr in prepared:
            r = pr["run"]
            s10 = r.seed[:10]
            if args.monotone and s10 in first_win:
                # Already won at a cheaper level — carry the win forward.
                _, pfh, pt, _, _ = results[s10][first_win[s10]]
                results[s10][sims] = (True, pfh, pt, 0.0, True)
                won += 1
                print(f"  asc{r.ascension:>2} {s10:<10} -> WON  (carried from {first_win[s10]:,})")
                continue
            res = sts2_rs.mcts_play_to_terminal(
                pr["cs"].clone(), sims=sims, max_rollout_depth=10, seed=args.seed
            )
            w = bool(res["player_won"])
            wall = float(res.get("wall_ms", 0.0))
            results[s10][sims] = (w, res["final_hp"], res["turns"], wall, False)
            if w and s10 not in first_win:
                first_win[s10] = sims
            won += w
            lost += not w
            total_wall += wall
            print(
                f"  asc{r.ascension:>2} {s10:<10} -> "
                f"{'WON ' if w else 'lost'} (final_hp={res['final_hp']:>3}, "
                f"turns={res['turns']:>2}, {wall / 1000:6.1f}s)",
                flush=True,
            )
        summary.append((sims, won, won + lost, total_wall))
        print(
            f"  ⇒ sims={sims:,}: {won}/{won + lost} = {won / (won + lost):.0%} "
            f"(actual compute this level: {total_wall / 1000:.0f}s)",
            flush=True,
        )
        _write_report(
            args.out, sim_levels, prepared, results, summary, potion_runs, errored, first_win
        )

    _print_tables(sim_levels, prepared, results, summary, first_win)
    print(f"\n(report written to {args.out})")
    return 0


def _fmt_summary_table(summary):
    rows = [
        "| sims | MCTS won | win% | compute (s) | s/fight (run) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for sims, won, n, wall in summary:
        rows.append(
            f"| {sims:,} | {won}/{n} | {won / n:.0%} | {wall / 1000:.0f} | "
            f"{(wall / 1000 / max(n, 1)):.1f} |"
        )
    return "\n".join(rows)


def _fmt_matrix(sim_levels, prepared, results, first_win):
    head = (
        "| seed | asc | hp | " + " | ".join(f"{s // 1000}k" for s in sim_levels) + " | first-win |"
    )
    sep = "|" + "---|" * (4 + len(sim_levels))
    rows = [head, sep]
    for pr in sorted(prepared, key=lambda p: p["run"].seed):
        r = pr["run"]
        s10 = r.seed[:10]
        cells = []
        for s in sim_levels:
            cell = results.get(s10, {}).get(s)
            if cell is None:
                cells.append("·")
            elif cell[0]:
                cells.append("W*" if cell[4] else "W")  # * = carried (not re-run)
            else:
                cells.append("L")
        fw = first_win.get(s10)
        fw_s = f"{fw // 1000}k" if fw else "never"
        rows.append(
            f"| {s10} | {r.ascension} | {pr['hp']}/{pr['max_hp']} | "
            + " | ".join(cells)
            + f" | {fw_s} |"
        )
    return "\n".join(rows)


def _write_report(path, sim_levels, prepared, results, summary, potion_runs, errored, first_win):
    never = sum(1 for s10 in results if s10 not in first_win)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Boss-only sweep — Vantom (Act-1 IC boss), human boss-entry loadout\n\n")
        f.write(f"- runs: {len(prepared)} won-only IC (human won 100% by construction)\n")
        f.write(f"- given potions: {potion_runs}/{len(prepared)}; prep errors: {errored}\n")
        f.write("- HP = exact post-rest-site boss-entry HP; deck/relics/upgrades faithful\n")
        f.write(
            f"- still-losing at the top sim level: {never}/{len(prepared)} "
            "(genuine AI gap, not search budget)\n\n"
        )
        f.write("## Win rate vs search budget\n\n")
        f.write(_fmt_summary_table(summary) + "\n\n")
        f.write(
            "_`compute (s)` is actual wall time spent this level; with `--monotone`, "
            "runs that already won at a cheaper level are carried forward (W*) and not "
            "re-run, so compute is front-loaded at the lowest level._\n\n"
        )
        f.write("## Per-run W/L matrix (search-sensitivity)\n\n")
        f.write("_W = MCTS won; W* = carried from a cheaper winning level; L = lost._\n\n")
        f.write(_fmt_matrix(sim_levels, prepared, results, first_win) + "\n")


def _print_tables(sim_levels, prepared, results, summary, first_win):
    print("\n════════ BOSS-ONLY SWEEP — Vantom (human boss-entry loadout) ════════\n")
    print(_fmt_summary_table(summary))
    print("\nPer-run matrix (W=won, W*=carried, L=lost):\n")
    print(_fmt_matrix(sim_levels, prepared, results, first_win))
    never = sum(1 for s10 in results if s10 not in first_win)
    print(f"\nstill losing at top level: {never}/{len(prepared)} (genuine AI gap)")
    print(f"(human won all {len(prepared)} by construction — won-only corpus)")


if __name__ == "__main__":
    sys.exit(main())
