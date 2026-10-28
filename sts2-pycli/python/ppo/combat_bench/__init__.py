"""Combat strength benchmark against pro human players (GH #830).

Replays human A10 v0.107.1 Ironclad fights from the spire-codex corpus with a
combat agent and compares the outcome against the human's on the same entry
state, and against other agents on the same seeds. Record shapes are in
`SCHEMA.md` next to this file.

    python -m ppo.combat_bench build ...   # fight sets + MANIFEST
    python -m ppo.combat_bench run ...     # one agent spec on one set
    python -m ppo.combat_bench report ...  # human gap + paired arm deltas
"""
