"""Ids and display names for everything a sandbox position can mention.

The engine registries are the source of truth for *which* ids exist
(``sts2_rs.list_registry_names``); the game's own localization
(``localization_eng/*.json``) and the pinned wiki manifests
(``sts2-wiki/data/v0.107.1``) supply the words a person reads. Where the two
sides spell an id differently the join rules live here, in one place:

* cards — ``loc_key == id``, else the id minus a ``_CARD`` suffix
  (``DEMON_FORM_CARD`` → ``DEMON_FORM``), else title-case;
* relics / potions — exact;
* powers — ``"<ID>_POWER"`` (engine ids drop the suffix), else title-case;
* monsters — engine ``PascalCase`` == wiki ``class_name``; the game names them
  under ``"<UPPER_SNAKE>.name"``; move titles best-effort under
  ``"<UPPER_SNAKE>.moves.<MOVE>.title"``.

Everything degrades to a title-cased id when the data files are absent (a
trimmed deploy, the test suite), so the catalog is never *missing* an entry
the engine knows.
"""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import Any

from simlab.position import CHARACTERS, ORB_BASE

#: Per-character starters: max HP, starter relic, starter deck (registry ids).
#: Cross-checked against the wiki ``characters.json`` in the tests.
STARTERS: dict[str, dict[str, Any]] = {
    "Ironclad": {
        "hp": 80,
        "relic": "BURNING_BLOOD",
        "deck": ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"],
        "orbSlots": 0,
    },
    "Silent": {
        "hp": 70,
        "relic": "RING_OF_THE_SNAKE",
        "deck": ["STRIKE_SILENT"] * 5 + ["DEFEND_SILENT"] * 5 + ["NEUTRALIZE", "SURVIVOR"],
        "orbSlots": 0,
    },
    "Defect": {
        "hp": 75,
        "relic": "CRACKED_CORE",
        "deck": ["STRIKE_DEFECT"] * 4 + ["DEFEND_DEFECT"] * 4 + ["ZAP", "DUALCAST"],
        "orbSlots": 3,
    },
    "Regent": {
        "hp": 75,
        "relic": "DIVINE_RIGHT",
        "deck": ["STRIKE_REGENT"] * 4 + ["DEFEND_REGENT"] * 4 + ["FALLING_STAR", "VENERATE"],
        "orbSlots": 0,
    },
    "Necrobinder": {
        "hp": 66,
        "relic": "BOUND_PHYLACTERY",
        "deck": ["STRIKE_NECROBINDER"] * 4 + ["DEFEND_NECROBINDER"] * 4 + ["BODYGUARD", "UNLEASH"],
        "orbSlots": 0,
    },
}

_ENGINE_WIKI_VERSION = "v0.107.1"


def starter_for(character: str) -> dict[str, Any]:
    """The starter row for *character* (PascalCase); KeyError-free."""
    return dict(STARTERS.get(character, STARTERS["Ironclad"]))


def _camel_to_upper_snake(name: str) -> str:
    out = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name)
    out = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", out)
    return out.upper()


def title_case(ident: str) -> str:
    """``FUZZY_WURM_CRAWLER`` / ``FuzzyWurmCrawler`` → ``Fuzzy Wurm Crawler``."""
    if "_" not in ident and not ident.isupper():
        ident = _camel_to_upper_snake(ident)
    return " ".join(w.capitalize() for w in ident.lower().split("_") if w)


# ---------------------------------------------------------------------------
# Data-file discovery
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """The ``sts2-pycli`` checkout this module lives in."""
    return Path(__file__).resolve().parents[2]


def localization_dir(game_version: str | None = None) -> Path | None:
    """``sts2-pycli/localization_eng`` (the Dockerfile copies it beside
    ``python/``); a non-default build's ``localization_eng_<ver>/`` (GH #877)."""
    name = (
        "localization_eng"
        if game_version in (None, _ENGINE_WIKI_VERSION)
        else f"localization_eng_{game_version}"
    )
    for candidate in (repo_root() / name, Path("/app") / name):
        if candidate.is_dir():
            return candidate
    return None


def wiki_data_dir() -> Path | None:
    """The wiki ``data/`` dir: ``STS2_WIKI_DATA``, else derived from
    ``STS2_WIKI_IMAGES`` the way the workbench does, else the sibling checkout."""
    explicit = os.environ.get("STS2_WIKI_DATA")
    if explicit and Path(explicit).is_dir():
        return Path(explicit)
    images = os.environ.get("STS2_WIKI_IMAGES")
    if images:
        p = Path(images)
        if len(p.parents) >= 3 and (p.parents[2] / "data").is_dir():
            return p.parents[2] / "data"
    sibling = repo_root().parent / "sts2-wiki" / "data"
    return sibling if sibling.is_dir() else None


def _load_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _loc(loc_dir: Path | None, name: str) -> dict[str, str]:
    data = _load_json(loc_dir / f"{name}.json") if loc_dir else None
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _wiki(
    wiki_dir: Path | None, name: str, version: str = _ENGINE_WIKI_VERSION
) -> list[dict[str, Any]]:
    data = _load_json(wiki_dir / version / f"{name}.json") if wiki_dir else None
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def _card_rows(
    ids: list[str], loc: dict[str, str], wiki: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {str(w.get("loc_key")): w for w in wiki if w.get("loc_key")}
    rows = []
    for cid in ids:
        key = cid if cid in by_key or f"{cid}.title" in loc else cid.removesuffix("_CARD")
        meta = by_key.get(key, {})
        name = loc.get(f"{key}.title") or str(meta.get("title") or "") or title_case(key)
        rows.append(
            {
                "id": cid,
                "name": name,
                "type": str(meta.get("type") or ""),
                "rarity": str(meta.get("rarity") or ""),
                "cost": meta.get("energy_cost"),
                "character": str(meta.get("character") or ""),
            }
        )
    return rows


def _named_rows(
    ids: list[str], loc: dict[str, str], wiki: list[dict[str, Any]], *, suffix: str = ""
) -> list[dict[str, Any]]:
    by_key = {str(w.get("loc_key")): w for w in wiki if w.get("loc_key")}
    rows = []
    for ident in ids:
        key = f"{ident}{suffix}"
        meta = by_key.get(key, {})
        name = loc.get(f"{key}.title") or str(meta.get("title") or "") or title_case(ident)
        row = {"id": ident, "name": name}
        if meta.get("rarity"):
            row["rarity"] = str(meta["rarity"])
        if meta.get("description"):
            row["description"] = str(meta["description"])
        rows.append(row)
    return rows


def _monster_rows(
    ids: list[str], loc: dict[str, str], wiki: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    import sts2_rs

    by_class = {str(w.get("class_name")): w for w in wiki if w.get("class_name")}
    rows = []
    for kind in ids:
        snake = _camel_to_upper_snake(kind)
        meta = by_class.get(kind, {})
        name = loc.get(f"{snake}.name") or str(meta.get("title") or "") or title_case(kind)
        try:
            moves = list(sts2_rs.monster_moves(kind, 0))
        except ValueError:
            moves = []
        move_rows = []
        for mv in moves:
            mid = str(mv["id"])
            title = (
                loc.get(f"{snake}.moves.{mid}.title")
                or loc.get(f"{snake}.moves.{mid.removesuffix('_MOVE')}.title")
                or title_case(mid.removesuffix("_MOVE"))
            )
            move_rows.append(
                {
                    "id": mid,
                    "title": title,
                    "initial": bool(mv.get("initial")),
                    "intent": mv.get("intent"),
                }
            )
        row: dict[str, Any] = {"id": kind, "name": name, "moves": move_rows}
        if meta.get("min_hp") is not None:
            row["minHp"] = meta.get("min_hp")
            row["maxHp"] = meta.get("max_hp")
        rows.append(row)
    return rows


def _encounter_rows(loc: dict[str, str]) -> list[dict[str, Any]]:
    import sts2_rs

    rows = []
    for e in sts2_rs.list_encounters():
        rows.append(
            {
                "id": str(e["id"]),
                "display": loc.get(f"{e['id']}.title")
                or str(e["display"])
                or title_case(str(e["id"])),
                "tier": str(e["tier"]),
                "act": int(e["act"]),
                "actName": str(e["act_name"]),
                "monsters": [str(m) for m in e["monsters"]],
            }
        )
    return rows


def build_catalog(
    loc_dir: Path | None, wiki_dir: Path | None, wiki_version: str = _ENGINE_WIKI_VERSION
) -> dict[str, Any]:
    """Assemble the catalog from the engine registries plus the data files of
    game build ``wiki_version`` (the wiki ``data/<ver>/`` dir)."""
    import sts2_rs

    from simlab.agents import AGENTS

    cards_loc = _loc(loc_dir, "cards")
    characters = [
        {
            "id": c,
            "name": {
                "Ironclad": "The Ironclad",
                "Silent": "The Silent",
                "Defect": "The Defect",
                "Regent": "The Regent",
                "Necrobinder": "The Necrobinder",
            }.get(c, c),
            **starter_for(c),
        }
        for c in CHARACTERS
    ]
    return {
        "characters": characters,
        "cards": _card_rows(
            sorted(sts2_rs.list_registry_names("card")),
            cards_loc,
            _wiki(wiki_dir, "cards", wiki_version),
        ),
        "relics": _named_rows(
            sorted(sts2_rs.list_registry_names("relic")),
            _loc(loc_dir, "relics"),
            _wiki(wiki_dir, "relics", wiki_version),
        ),
        "potions": _named_rows(
            sorted(sts2_rs.list_registry_names("potion")),
            _loc(loc_dir, "potions"),
            _wiki(wiki_dir, "potions", wiki_version),
        ),
        "powers": _named_rows(
            sorted(sts2_rs.list_registry_names("power")),
            _loc(loc_dir, "powers"),
            _wiki(wiki_dir, "powers", wiki_version),
            suffix="_POWER",
        ),
        "orbs": [{"id": k, "name": k, "passive": p, "evoke": e} for k, (p, e) in ORB_BASE.items()],
        "monsters": _monster_rows(
            sorted(sts2_rs.list_registry_names("monster")),
            _loc(loc_dir, "monsters"),
            _wiki(wiki_dir, "monsters", wiki_version),
        ),
        "encounters": _encounter_rows(_loc(loc_dir, "encounters")),
        "strategies": sorted(AGENTS),
    }


@functools.lru_cache(maxsize=4)
def _cached(
    loc_dir: str | None, wiki_dir: str | None, wiki_version: str = _ENGINE_WIKI_VERSION
) -> dict[str, Any]:
    return build_catalog(
        Path(loc_dir) if loc_dir else None, Path(wiki_dir) if wiki_dir else None, wiki_version
    )


def catalog(game_version: str | None = None) -> dict[str, Any]:
    """The catalog for this process's data files (built once per build).

    ``game_version`` names the build whose text names the rows (GH #877; ``None``
    = the default build). A build whose vendored loc is absent falls back to the
    default build's names. Rust twin: ``simlab_catalog::catalog_for``.
    """
    version = game_version or _ENGINE_WIKI_VERSION
    loc = localization_dir(version) or localization_dir()
    wiki = wiki_data_dir()
    return _cached(str(loc) if loc else None, str(wiki) if wiki else None, version)


def name_of(kind: str, ident: str, game_version: str | None = None) -> str:
    """Display name for one id (``kind`` in card/relic/potion/power/monster)."""
    cat = catalog(game_version)
    table = {
        "card": "cards",
        "relic": "relics",
        "potion": "potions",
        "power": "powers",
        "monster": "monsters",
    }
    for row in cat.get(table.get(kind, ""), []):
        if row["id"] == ident:
            return str(row["name"])
    return title_case(ident)
