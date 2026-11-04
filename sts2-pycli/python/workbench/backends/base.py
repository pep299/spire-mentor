"""The run driver the workbench live session is built on.

A :class:`GameBackend` owns one run being driven — it starts the run, applies
one verb at a time and renders the C# state-dict shape keyed by ``decision``
that the server / render / diff layers consume. :class:`BranchingLiveSession`
talks to this surface and nothing else, so the session never has to ask a
backend *what it is* — every optional capability has an honest default.

Four adapters satisfy the interface today:

* :class:`workbench.backends.rust_game.RustGame` — the in-process ``sts2_rs``
  engine. The only adapter with a native run (:meth:`GameBackend.native_run`),
  which is what run-mode autoplay, the PPO panels and the advisors need.
* :class:`workbench.backends.combat_replay.CombatReplayBackend` — one recorded
  combat replayed to an exact decision ("play from here"). Combat only.
* :class:`workbench.backends.imported_combat.ImportedCombatBackend` — one
  combat rebuilt from a board copied out of the live game. Combat only; it
  shares everything but its materialization with the replay adapter (see
  :class:`workbench.backends.single_combat.SingleCombatBackend`).
* :class:`DuckTypedBackend` — any object that speaks the five C# verbs
  (``sts2_client.Game``, the test fakes). It has none of the optional
  capabilities, and says so through the defaults rather than by lacking
  attributes.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any


class UnsupportedOperation(RuntimeError):
    """The backend cannot do this — e.g. run-mode play without a native run.

    A ``RuntimeError`` so the server's existing ``RuntimeError → 400`` mapping
    turns it into a client-visible message rather than a 500.
    """


class GameBackend:
    """One run being driven.

    Subclasses implement the five required verbs. The optional capabilities
    below default to "not available" (``None`` / no-op / empty), so callers
    branch on the *returned value*, never on ``hasattr``.

    Contract for the required verbs: implementations are zero-arg
    constructible (run parameters arrive via :meth:`start`); every method
    returns the C# state-dict shape keyed by ``decision``; :meth:`act` returns
    ``{"type": "error", "message": ...}`` for an illegal move rather than
    raising.
    """

    # -- required ------------------------------------------------------------
    def start(
        self,
        *,
        character: str,
        seed: str,
        ascension: int,
        act1: str = "overgrowth",
        interactive_potion_swap: bool = False,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def act(self, action: str, **args: Any) -> dict[str, Any]:
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_map(self, first_col: int | None = None) -> dict[str, Any]:
        """The current act's connected map graph (``{}`` when there is none).

        *first_col* seeds the attached route plan with the column the driving
        run policy will take; adapters without a planner ignore it.
        """
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # -- optional capabilities ----------------------------------------------
    def native_run(self) -> Any | None:
        """The in-process ``sts2_rs.RunManager`` driving this run, or ``None``.

        Everything that forks, searches or evaluates the run natively (run-mode
        autoplay, the advisors, the PPO panels, native action replay) needs
        this; a backend without one simply can't offer those features.
        """
        return None

    def combat_state(self) -> Any | None:
        """The native ``sts2_rs`` combat state while a fight is live, else ``None``."""
        return None

    def start_faithful(self, *, character: str, seed: int, ascension: int = 0) -> dict[str, Any]:
        """Start a run the way the offline benchmark / PPO env does (integer
        seed, Neow start). Only the native engine can."""
        raise UnsupportedOperation("run-mode play requires the rust backend")

    def start_imported(self, imported: Any) -> dict[str, Any]:
        """Rebuild the run behind a state copied out of the live game.

        Takes a :class:`workbench.state_import.RunSnapshot`. Only the native
        engine can: the graft is built out of ``RunManager`` affordances the
        C# oracle has no analogue for.
        """
        raise UnsupportedOperation("importing a live state requires the rust backend")

    def import_notes(self) -> list[str]:
        """What the last :meth:`start_imported` could NOT reproduce.

        Empty for every other start path, so a caller can read it unguarded —
        and a half-restored position is never allowed to read as a whole one.
        """
        return []

    def recommended_action(self) -> tuple[str, dict[str, Any]] | None:
        """``(verb, args)`` the engine's own out-of-combat policy would take
        now, or ``None`` (combat, run over, or no such policy)."""
        return None

    def set_record_card_plays(self, on: bool) -> None:
        """Opt in to the engine card play-log (event-log labels)."""

    def set_record_combat_log(self, on: bool) -> None:
        """Opt in to the full game-style combat log (the log panel)."""

    def card_play_log(self) -> list[dict[str, Any]]:
        """The cumulative card play-log for the current combat (empty when not recorded)."""
        return []

    def combat_log(self) -> list[dict[str, Any]]:
        """The cumulative combat log for the current combat (empty when not recorded)."""
        return []


class DuckTypedBackend(GameBackend):
    """Adapter over any object that speaks the five C# verbs.

    Wraps ``sts2_client.Game`` (a subprocess, so no native run) and the test
    fakes that mimic it. Attribute reads for names :class:`GameBackend` does
    *not* declare fall through to the wrapped object, so harness code that
    drives the C# client's extras (``set_player`` in the parity walks) keeps
    working.

    The optional verbs are the other case: :class:`GameBackend` declares them
    as concrete methods with "engine can't do this" defaults, so those names
    resolve on the adapter and ``__getattr__`` never sees them. A duck-typed
    engine that *does* implement, say, ``combat_log`` gets the base's ``[]``
    anyway — silently. Nothing here needs it today (``sts2_client.Game``
    implements none of the eight), but an engine that grows one has to
    subclass :class:`GameBackend` rather than rely on the pass-through.
    """

    def __init__(self, game: Any) -> None:
        self._game = game

    def start(
        self,
        *,
        character: str,
        seed: str,
        ascension: int,
        act1: str = "overgrowth",
        interactive_potion_swap: bool = False,
    ) -> dict[str, Any]:
        # Forwarded only when set, so a duck-typed engine without the
        # keyword keeps working.
        extra = {"interactive_potion_swap": True} if interactive_potion_swap else {}
        return self._game.start(
            character=character, seed=seed, ascension=ascension, act1=act1, **extra
        )

    def act(self, action: str, **args: Any) -> dict[str, Any]:
        return self._game.act(action, **args)

    def get_state(self) -> dict[str, Any]:
        return self._game.get_state()

    def get_map(self, first_col: int | None = None) -> dict[str, Any]:
        return self._game.get_map()

    def close(self) -> None:
        self._game.close()

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not found on the adapter itself — which
        # excludes every verb GameBackend declares, optional ones included.
        return getattr(self._game, name)


def adapt_backend(obj: Any) -> GameBackend:
    """Return *obj* as a :class:`GameBackend`, wrapping duck-typed engines."""
    return obj if isinstance(obj, GameBackend) else DuckTypedBackend(obj)


def make_backend(name: str, **kwargs: Any) -> Callable[[], GameBackend]:
    """Return a zero-arg factory for backend *name* (``"csharp"`` / ``"rust"``).

    Extra keyword arguments are forwarded to the underlying constructor. The
    rust backend accepts ``target_convention`` (``"native"`` — the default, for
    the workbench/MCTS/standalone paths — or ``"csharp"`` for cross-engine
    parity replay; see :class:`~workbench.backends.rust_game.RustGame`).

    Raises:
        ValueError: if *name* is not a known backend.
    """
    if name == "csharp":
        from sts2_client import Game

        def _csharp() -> GameBackend:
            return DuckTypedBackend(Game(**kwargs))

        return _csharp
    if name == "rust":
        from workbench.backends.rust_game import RustGame

        if kwargs:
            return functools.partial(RustGame, **kwargs)
        return RustGame
    raise ValueError(f"unknown backend: {name!r} (expected 'csharp' or 'rust')")
