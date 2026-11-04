"""The C# oracle's save/load round trip (``python/save_load_identity.py``).

The act-2/3 checkpoints in ``act23_state_parity`` rest on a reloaded save
continuing exactly like the process that wrote it.
"""

from __future__ import annotations

import pytest
import save_load_identity as sli


def test_normalize_save_drops_only_the_lossy_keys() -> None:
    save = {
        "players": [{"discovered_potions": ["P"], "gold": 5}],
        "acts": [{"saved_map": {"points": [{"coord": {}, "type": "monster", "can_modify": True}]}}],
        "other": {"can_modify": True},
    }
    out = sli._normalize_save(save)
    assert out["players"] == [{"gold": 5}]
    assert out["acts"][0]["saved_map"]["points"] == [{"coord": {}, "type": "monster"}]
    # ``can_modify`` is dropped only on map points.
    assert out["other"] == {"can_modify": True}


def test_strip_view_drops_chaotic_vars() -> None:
    view = {"options": [{"vars": {"EntrantNumber": 827, "Gold": 50}}]}
    assert sli._strip_view(view) == {"options": [{"vars": {"Gold": 50}}]}


@pytest.mark.slow
@pytest.mark.oracle
def test_reload_continues_identically() -> None:
    res = sli.check_one("Ironclad", "1000", act=2, steps=80, ascension=0, max_walk=4000)
    assert res.status == "identical", (res.detail, res.diffs)
