"""Render one recorded MCTS run (JSONL) as Markdown for agent inspection.

CLI: ``python -m workbench.report <log.jsonl> [--out file.md]``. The Markdown
mirrors what the deprecated Rust ``fight_replay`` HTML showed, in a form that is
easy to Read/grep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from enemy_labels import enemy_labels


def _intent_text(enemy: dict[str, Any]) -> str:
    """One-line intent summary from a canonical enemy's ``intents`` list.

    Each intent is the ``{type, damage, hits}`` shape the workbench uses
    (e.g. ``Attack 7``, ``Attack 5x3``, ``Buff``). Falls back to the legacy
    singular ``intent.label`` for older logs.
    """
    intents = enemy.get("intents")
    if intents:
        parts = []
        for it in intents:
            kind = it.get("type", "?")
            dmg = it.get("damage")
            hits = it.get("hits")
            if dmg is None:
                parts.append(str(kind))
            elif hits and hits > 1:
                parts.append(f"{kind} {dmg}x{hits}")
            else:
                parts.append(f"{kind} {dmg}")
        return ", ".join(parts)
    return (enemy.get("intent") or {}).get("label", "?")


def _load(path: str) -> tuple[dict, list[dict], dict]:
    """Return (head_meta, steps, summary). Each step = {state, mcts, action}."""
    head: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    by_step: dict[int, dict[str, Any]] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        e = json.loads(raw)
        t, step, data = e.get("type"), e.get("step"), e.get("data") or {}
        if t == "meta" and data.get("summary"):
            summary = data
        elif t == "meta":
            head = data
        elif t in ("state", "mcts", "action"):
            by_step.setdefault(step, {})[t] = data
    steps = [by_step[k] for k in sorted(by_step) if "state" in by_step[k]]
    return head, steps, summary


def render_run_markdown(path: str) -> str:
    head, steps, summary = _load(path)
    out: list[str] = []
    stem = head.get("fixture_stem") or head.get("seed", "run")
    out.append(f"# MCTS run: {stem}")
    out.append(
        f"- encounter: {head.get('encounter')}  build: {head.get('build_id')}  sims: {head.get('sims')}"
    )
    if summary:
        verdict = "WON" if summary.get("final_won") else "LOST"
        out.append(
            f"- outcome: **{verdict}** at turn {summary.get('final_turn')}, "
            f"HP {summary.get('final_player_hp')}/{summary.get('final_player_max_hp')}"
        )
    # Relics / potions are constant across the combat — surface them once from
    # the first state so effects like Ice Cream (energy carryover) are visible.
    if steps:
        pl0 = steps[0].get("state", {}).get("player", {})
        relics = [r.get("name", r.get("id", "?")) for r in pl0.get("relics") or []]
        potions = [p.get("name", p.get("id", "?")) for p in pl0.get("potions") or [] if p]
        if relics:
            out.append(f"- relics: {', '.join(relics)}")
        if potions:
            out.append(f"- potions: {', '.join(potions)}")
    out.append("")
    for d in steps:
        st = d["state"]
        pl = st.get("player", {})
        who = pl.get("name") or ""
        who = f"{who} " if who else ""
        # Energy lives at the state top level in the canonical shape; older logs
        # nested it under ``player``.
        energy = st.get("energy", pl.get("energy"))
        out.append(
            f"## T{st.get('round')} · {st.get('decision')} · "
            f"{who}HP {pl.get('hp')}/{pl.get('max_hp')} blk {pl.get('block')} e {energy}"
        )
        hand = ", ".join(c.get("name", c.get("id", "?")) for c in st.get("hand", []))
        out.append(f"- hand: {hand}")
        enemies = st.get("enemies", [])
        # A name shared by several enemies carries its " #k" (enemy_labels).
        labels = enemy_labels(enemies)
        for m, label in zip(enemies, labels, strict=True):
            intent = _intent_text(m)
            name = label if m.get("name") else m.get("name")
            out.append(f"- {name} {m.get('hp')}/{m.get('max_hp')} blk {m.get('block')} → {intent}")
        sim = st.get("reward_sim")
        if sim:
            out.append(
                f"- reward picked by boss simulation vs {sim.get('encounter')} "
                f"({sim.get('sims')} sims/move):"
            )
            for c in sim.get("candidates") or []:
                tag = "WIN" if c.get("win") else "loss"
                out.append(f"  - {c.get('label')}: score {c.get('score'):+.0f} ({tag})")
        mcts = d.get("mcts")
        if mcts:
            out.append("")
            out.append("| rank | action | visits | share | Q | W/L/T |")
            out.append("|---|---|---:|---:|---:|---|")
            rows = mcts.get("root_outcomes", [])
            total = mcts.get("total_visits") or sum(r["visits"] for r in rows) or 1
            for i, r in enumerate(sorted(rows, key=lambda r: -r["visits"])):
                share = round(100 * r["visits"] / total)
                star = " ⭐" if r.get("is_best") else ""
                out.append(
                    f"| {i} | {r['action_label']}{star} | {r['visits']} | "
                    f"{share}% | {r['q']:+.2f} | {r['wins']}/{r['losses']}/{r['ties']} |"
                )
            if mcts.get("pv"):
                out.append(f"\npv: {' → '.join(mcts['pv'])}")
        act = d.get("action")
        if act:
            out.append(f"\nchosen: `{act.get('action')}` {act.get('args', {})}")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    md = render_run_markdown(a.log)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
