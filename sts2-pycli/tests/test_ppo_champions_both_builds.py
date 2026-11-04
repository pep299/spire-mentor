"""Every deployed champion plays a whole run on both game builds (GH #799).

The champions were re-exported over a vocab that spans v0.107.1 and v0.111.0
(`scripts/reexport_champions_vocab.py`). These drive the deployed run agent
(`sts2_rs.PpoRunAgent`: the champion for every meta decision, the combat MCTS
for every fight) through one complete run per character on each build, at a
tiny combat budget so the suite stays fast. The v0.111.0 seed's Neow offer
carries DowsingRod, so the net is shown a v0.111.0-only id, which must now
have a row of its own instead of OOV.

The full-budget smoke, with the per-run record, is
`scripts/smoke_v0111_champions.py`; bit-identity on v0.107.1 is
`scripts/vocab_reexport_equivalence.py`.
"""

from __future__ import annotations

import champions
import pytest
import sts2_rs
from ppo import vocab

_CHARACTERS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]

#: Seed whose v0.111.0 Neow offer includes DowsingRod (any character).
_V0111_SEED = 22


def _play(character: str, seed: int, game_version: str | None) -> sts2_rs.RunManager:
    kwargs = {} if game_version is None else {"game_version": game_version}
    rm = sts2_rs.RunManager(
        seed=seed,
        character=character,
        neow_start=True,
        mode="agent",
        **kwargs,
    )
    champ = champions.get(character)
    assert champ is not None
    agent = sts2_rs.PpoRunAgent(seed=seed, sims=8, boss_sims=16, model_path=str(champ.onnx_path))
    for _ in range(5000):
        if rm.is_over() or agent.step(rm) is None:
            break
    assert rm.is_over(), f"{character} seed {seed}: run did not finish"
    return rm


@pytest.mark.parametrize("character", _CHARACTERS)
def test_every_champion_completes_a_run(character: str) -> None:
    rm = _play(character, 0, None)
    assert rm.game_version == "v0.107.1"


@pytest.mark.parametrize("character", _CHARACTERS)
def test_every_champion_completes_a_run_v0111(character: str) -> None:
    probe = sts2_rs.RunManager(
        seed=_V0111_SEED, character=character, neow_start=True, game_version="v0.111.0"
    )
    assert "DOWSING_ROD" in probe.neow_offers()
    # The id the net is shown has a real row, past every v0.107.1 row.
    row = vocab.relics().index("DOWSING_ROD")
    assert row != vocab.relics().oov
    assert row >= 297  # the v0.107.1 relic vocab had 297 ids
    rm = _play(character, _V0111_SEED, "v0.111.0")
    assert rm.game_version == "v0.111.0"
