"""GH #893: the map-parity harnesses build their Rust ``RunManager`` for the
selected game build.

``act23_map_parity`` (and the Act-1 ``run_setup_parity`` it builds on) drive
the C# oracle of ``STS2_GAME_VERSION`` but used to construct the Rust side
with no ``game_version``, i.e. always v0.107.1. On v0.111.0 that compared the
v0.107.1 engine with the v0.111.0 oracle — the 0/15 it reported was a harness
artifact. These tests stub ``sts2_rs.RunManager`` and check what it is given.
"""

from __future__ import annotations

from typing import Any, ClassVar

import act23_map_parity as amp
import pytest
import run_setup_parity as rsp
import sts2_client


class _FakeRunManager:
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        _FakeRunManager.calls.append(kwargs)

    def act_map_for(self, act: int) -> list[dict]:
        return []

    def act_map(self) -> list[dict]:
        return []


@pytest.fixture
def fake_rm(monkeypatch: pytest.MonkeyPatch) -> type[_FakeRunManager]:
    _FakeRunManager.calls = []
    monkeypatch.setattr(amp.sts2_rs, "RunManager", _FakeRunManager)
    return _FakeRunManager


@pytest.mark.parametrize("version", sts2_client.SUPPORTED_GAME_VERSIONS)
def test_act23_rust_side_follows_the_selected_game_version(
    fake_rm: type[_FakeRunManager], monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(sts2_client, "GAME_VERSION", version)
    amp._rust_act_maps("Ironclad", "PARA", 0)
    assert fake_rm.calls[-1]["game_version"] == version


def test_act23_explicit_game_version_wins(fake_rm: type[_FakeRunManager]) -> None:
    amp._rust_act_maps("Ironclad", "PARA", 0, game_version="v0.111.0")
    assert fake_rm.calls[-1]["game_version"] == "v0.111.0"


@pytest.mark.parametrize("version", sts2_client.SUPPORTED_GAME_VERSIONS)
def test_run_setup_rust_side_follows_the_selected_game_version(
    fake_rm: type[_FakeRunManager], monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(rsp.sts2_rs, "RunManager", _FakeRunManager)
    monkeypatch.setattr(sts2_client, "GAME_VERSION", version)
    rsp._rust_act1_map("Ironclad", "PARA", 0)
    assert fake_rm.calls[-1]["game_version"] == version
