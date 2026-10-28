#!/usr/bin/env python3
"""
sts2-pycli interactive player — play Slay the Spire 2 in your terminal.

Usage:
    python3 play.py                    # Interactive mode (you play)
    python3 play.py --auto             # Auto-play with simple AI
    python3 play.py --seed myseed      # Fixed seed for reproducibility
    python3 play.py --character Silent  # Choose character
"""

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import threading
import traceback

import sts2_client
from game_log import GameLogger

# Hint recommender (T9 wire-up). Imported lazily-safely: if the
# sts2_hint package isn't importable (e.g. sts2_rs wheel missing in a
# minimal dev env) we degrade silently and never render tags.
try:
    from sts2_hint import display as _hint_display
    from sts2_hint.errors import get_logger as _hint_get_logger
    from sts2_hint.recommender import RecommendConfig as _HintRecommendConfig
    from sts2_hint.recommender import recommend as _hint_recommend
    from sts2_hint.recommender import recommend_all as _hint_recommend_all

    _HINT_AVAILABLE = True
except Exception:  # pragma: no cover - import-time fallback
    _hint_display = None  # type: ignore[assignment]
    _hint_get_logger = None  # type: ignore[assignment]
    _HintRecommendConfig = None  # type: ignore[assignment]
    _hint_recommend = None  # type: ignore[assignment]
    _hint_recommend_all = None  # type: ignore[assignment]
    _HINT_AVAILABLE = False

# Default hint config. CLI flags in T10 override this via play(hint_cfg=...).
# When None, hint rendering is disabled entirely (no-hint path).
_HINT_CFG = _HintRecommendConfig() if _HINT_AVAILABLE else None

# Hint mode: "inline" (compute tags during show_combat), "off"
# (--no-hints, never compute), or "on_demand" (--hint-on-demand:
# only compute when the user types `/hint` in the REPL).
_HINT_MODE = "inline"

# Hard wall-clock cap (ms) on a single recommend() call. 0 = no cap.
# Enforced via threading in _compute_hint_tags.
_HINT_TIMEOUT_MS = 5000

# Last error surfaced from :func:`_compute_hint_tags`. The full
# traceback always lands in the rotating ``sts2_hint`` log; this short
# summary is what the REPL prints inline when ``go`` / ``a`` can't get
# a recommendation.
_LAST_HINT_ERROR: str | None = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.join(ROOT, "src", "Sts2Headless", "Sts2Headless.csproj")
# lib/<ver>/ for the selected game build (STS2_GAME_VERSION / --game-version).
LIB_DIR = sts2_client.LIB_DIR
SAVE_DIR = os.path.join(ROOT, "saves")
CURRENT_SAVE_SCHEMA_VERSION = 14


def _find_dotnet():
    """Find .NET SDK binary."""
    candidates = [
        os.path.expanduser("~/.dotnet-arm64/dotnet"),
        os.path.expanduser("~/.dotnet/dotnet"),
        "dotnet",
    ]
    for p in candidates:
        try:
            r = subprocess.run([p, "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return p
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


DOTNET = _find_dotnet()


def _find_game_dir():
    """Auto-detect STS2 Steam install directory."""
    import platform

    system = platform.system()
    candidates = []
    if system == "Darwin":
        base = os.path.expanduser(
            "~/Library/Application Support/Steam/steamapps/common/Slay the Spire 2/SlayTheSpire2.app/Contents/Resources"
        )
        candidates = [
            os.path.join(base, "data_sts2_macos_arm64"),
            os.path.join(base, "data_sts2_macos_x86_64"),
        ]
    elif system == "Linux":
        for steam in ["~/.steam/steam", "~/.local/share/Steam"]:
            candidates.append(os.path.expanduser(f"{steam}/steamapps/common/Slay the Spire 2"))
    elif system == "Windows":
        candidates = [r"C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"]

    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


def _copy_dlls(game_dir):
    """Copy required DLLs from game directory to lib/."""
    os.makedirs(LIB_DIR, exist_ok=True)
    dlls = [
        "sts2.dll",
        "SmartFormat.dll",
        "SmartFormat.ZString.dll",
        "Sentry.dll",
        "Steamworks.NET.dll",
        "MonoMod.Backports.dll",
        "MonoMod.ILHelpers.dll",
        "0Harmony.dll",
        "System.IO.Hashing.dll",
    ]
    import shutil

    for dll in dlls:
        src = os.path.join(game_dir, dll)
        dst = os.path.join(LIB_DIR, dll)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  ✓ {dll}")
        else:
            # Search subdirectories
            for root_d, _, files in os.walk(game_dir):
                if dll in files:
                    shutil.copy2(os.path.join(root_d, dll), dst)
                    print(f"  ✓ {dll}")
                    break
            else:
                print(f"  ✗ {dll} not found")

    # Backup original sts2.dll
    sts2 = os.path.join(LIB_DIR, "sts2.dll")
    backup = os.path.join(LIB_DIR, "sts2.dll.original")
    if os.path.isfile(sts2) and not os.path.isfile(backup):
        shutil.copy2(sts2, backup)


def _patch_dll():
    """Apply IL patches to sts2.dll using setup.sh (requires Mono.Cecil via dotnet)."""
    setup_sh = os.path.join(ROOT, "setup.sh")
    if not os.path.isfile(setup_sh):
        print("  ⚠ setup.sh not found, skipping IL patch")
        return
    # Run just the patching part via setup.sh
    subprocess.run(["bash", setup_sh], cwd=ROOT)


def _build():
    """Build the C# project."""
    if not DOTNET:
        return False
    r = subprocess.run(
        [DOTNET, "build", PROJECT, f"-p:GameVersion={sts2_client.GAME_VERSION}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.returncode == 0


def ensure_setup():
    """Check that everything is ready to run. Auto-setup if needed."""
    issues = []

    # Check .NET SDK
    if not DOTNET:
        print("❌ .NET SDK not found.")
        print("   Install .NET 9+ from https://dotnet.microsoft.com/download")
        sys.exit(1)

    # Check lib/sts2.dll exists
    sts2_dll = os.path.join(LIB_DIR, "sts2.dll")
    if not os.path.isfile(sts2_dll):
        print("📦 Game DLLs not found. Running first-time setup...")
        game_dir = _find_game_dir()
        if not game_dir:
            print("❌ Could not find Slay the Spire 2 installation.")
            print("   Install the game via Steam, then run again.")
            print("   Or run: ./setup.sh /path/to/game/data")
            sys.exit(1)
        print(f"  Found game at: {game_dir}")
        _copy_dlls(game_dir)
        if not os.path.isfile(sts2_dll):
            print("❌ Failed to copy sts2.dll")
            sys.exit(1)

    # Set STS2_GAME_DIR env var for runtime DLL resolution
    game_dir = _find_game_dir()
    if game_dir:
        os.environ["STS2_GAME_DIR"] = game_dir

    # Check if built
    exe_dir = os.path.dirname(sts2_client.headless_bin())
    exe = os.path.join(exe_dir, "Sts2Headless.dll")
    if (
        not os.path.isfile(exe)
        or pathlib.Path(sts2_dll).stat().st_mtime > pathlib.Path(exe).stat().st_mtime
    ):
        print("🏗️  Building...")
        if not _build():
            print("❌ Build failed. Try: ./setup.sh")
            sys.exit(1)
        print("  ✓ Build succeeded")


# ─── Native save file support ───


def _find_native_save_dir():
    """Auto-detect the game's save directory."""
    import glob as globmod
    import platform

    system = platform.system()
    patterns = []
    if system == "Darwin":
        patterns = [
            os.path.expanduser(
                "~/Library/Application Support/SlayTheSpire2/steam/*/profile*/saves"
            ),
        ]
    elif system == "Linux":
        patterns = [
            os.path.expanduser("~/.local/share/SlayTheSpire2/steam/*/profile*/saves"),
            os.path.expanduser(
                "~/.config/unity3d/MegaCrit/Slay the Spire 2/steam/*/profile*/saves"
            ),
        ]
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        localappdata = os.environ.get("LOCALAPPDATA", "")
        patterns = [
            os.path.join(appdata, "SlayTheSpire2", "steam", "*", "profile*", "saves"),
            os.path.join(localappdata, "SlayTheSpire2", "steam", "*", "profile*", "saves"),
        ]
    for pat in patterns:
        matches = globmod.glob(pat)
        for d in matches:
            if os.path.isfile(os.path.join(d, "current_run.save")):
                return d
        if matches:
            return matches[0]
    return None


def _id_to_name(model_id):
    """Convert model ID like 'CARD.STRIKE_NECROBINDER' to readable name."""
    if not model_id:
        return "?"
    parts = model_id.split(".", 1)
    name = parts[-1] if len(parts) > 1 else model_id
    return name.replace("_", " ").title()


def show_native_save(save_path):
    """Parse and display a native current_run.save file."""
    with open(save_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{'═' * 60}")
    print("  Native Save File")
    print(f"  {save_path}")
    print(f"{'═' * 60}")
    seed = data.get("rng", {}).get("seed", "?")
    ascension = data.get("ascension", 0)
    act_idx = data.get("current_act_index", 0)
    acts = data.get("acts", [])
    act_name = _id_to_name(acts[act_idx]["id"]) if act_idx < len(acts) else "?"
    run_time = data.get("run_time", 0)
    run_min = run_time // 60
    run_sec = run_time % 60

    print(f"\n  Seed: {seed}")
    print(f"  Ascension: {ascension}")
    print(f"  Act: {act_idx + 1} ({act_name})")
    print(f"  Time: {run_min}m{run_sec:02d}s")
    print(f"  Schema: v{data.get('schema_version', '?')}")
    room = data.get("pre_finished_room", {})
    if room:
        room_type = room.get("room_type", "?")
        enc = room.get("encounter_id") or room.get("event_id") or ""
        print(f"  Room: {room_type}" + (f" ({_id_to_name(enc)})" if enc else ""))

    visited = data.get("visited_map_coords", [])
    if visited:
        print(f"  Map: Floor {len(visited)} ({len(visited)} nodes visited)")

    for player in data.get("players", []):
        char_name = _id_to_name(player.get("character_id", "?"))
        hp = player.get("current_hp", 0)
        max_hp = player.get("max_hp", 0)
        gold = player.get("gold", 0)
        energy = player.get("max_energy", 3)

        print(f"\n  {'─' * 50}")
        print(f"  {char_name}  HP: {hp}/{max_hp}  Gold: {gold}  Energy: {energy}")

        deck = player.get("deck", [])
        print(f"\n  Deck ({len(deck)}):")
        card_counts = {}
        for card in deck:
            cid = _id_to_name(card.get("id", "?"))
            up = card.get("current_upgrade_level", 0)
            key = f"{cid}{'+' * up if up else ''}"
            card_counts[key] = card_counts.get(key, 0) + 1
        for name, cnt in sorted(card_counts.items()):
            print(f"    • {name}" + (f" x{cnt}" if cnt > 1 else ""))

        relics = player.get("relics", [])
        if relics:
            print(f"\n  Relics ({len(relics)}):")
            for r in relics:
                print(f"    🔶 {_id_to_name(r.get('id', '?'))}")

        potions = player.get("potions", [])
        if potions:
            print(f"\n  Potions ({len(potions)}):")
            for p in potions:
                print(f"    🧪 [{p.get('slot_index', '?')}] {_id_to_name(p.get('id', '?'))}")

    if acts:
        print(f"\n  {'─' * 50}")
        print("  Acts summary:")
        for i, act in enumerate(acts):
            act_id = _id_to_name(act.get("id", "?"))
            rooms_data = act.get("rooms", {})
            boss = _id_to_name(rooms_data.get("boss_id", ""))
            normals = rooms_data.get("normal_encounters_visited", 0)
            elites = rooms_data.get("elite_encounters_visited", 0)
            events = rooms_data.get("events_visited", 0)
            bosses = rooms_data.get("boss_encounters_visited", 0)
            marker = " ◀" if i == act_idx else ""
            print(
                f"    Act {i + 1}: {act_id}  Boss: {boss}  "
                f"[M{normals} E{elites} ?{events} B{bosses}]{marker}"
            )

    print(f"\n{'═' * 60}\n")


# ─── Display helpers ───


def n(obj):
    """Extract display name."""
    return str(obj) if obj is not None else "?"


def short_n(obj):
    """Short name only."""
    return str(obj) if obj is not None else "?"


def desc(obj):
    """Extract description, strip BBCode tags, clean SmartFormat vars."""
    if obj and isinstance(obj, str):
        import re

        text = obj
        text = re.sub(r"\[/?[^\]]+\]", "", text)  # strip BBCode [tags]

        # Handle SmartFormat expressions:
        # {IfUpgraded:show:text1|text2} → text2 (non-upgraded default)
        # {InCombat:text1|text2} → text1 (show combat version)
        # {energyPrefix:energyIcons(1)} → [E] (energy symbol)
        # {Stars:starIcons()} → [S] (star symbol)
        # {VarName:diff()} → [VarName] (simple var)
        # {VarName:choose(a|b)} → [VarName]

        def smart_replace(m):
            full = m.group(1)
            # Handle conditional: {IfUpgraded:show:textA|textB}
            if full.startswith("IfUpgraded:show:"):
                parts = full[len("IfUpgraded:show:") :].split("|")
                return parts[1] if len(parts) > 1 else parts[0]  # show non-upgraded
            if full.startswith("IfUpgraded:"):
                parts = full[len("IfUpgraded:") :].split("|")
                return parts[1] if len(parts) > 1 else parts[0]
            # {InCombat:text|alt} → show combat text
            if full.startswith("InCombat:"):
                parts = full[len("InCombat:") :].split("|")
                return parts[0].lstrip("\n")  # show combat version
            # Energy icons: {Energy:energyIcons()} → [Energy]E
            if "energyIcons" in full:
                var = full.split(":")[0]
                return f"[{var}]E"
            # Star icons: {Stars:starIcons()} → [Stars]⭐
            if "starIcons" in full:
                var = full.split(":")[0]
                return f"[{var}]⭐"
            # Plural: {Cards:plural:card|cards} → card/cards based on value
            if ":plural:" in full:
                parts = full.split(":")
                var = parts[0]
                plural_parts = ":".join(parts[2:]).split("|")
                return f"[{var}:{plural_parts[0]}|{plural_parts[1] if len(plural_parts) > 1 else plural_parts[0]}]"
            # Conditional: {IsMultiplayer:textA|textB} → textB (single player)
            if ":" in full and "|" in full:
                parts_after = ":".join(full.split(":")[1:]).split("|")
                return parts_after[-1]  # take the false/last branch
            # Simple var with format: {Damage:diff()} → [Damage]
            var = full.split(":")[0]
            return f"[{var}]"

        # Process from innermost braces outward (handle nesting)
        for _ in range(3):  # max 3 nesting levels
            text = re.sub(r"\{([^{}]+)\}", smart_replace, text)
        return text.strip()
    return ""


COLORS = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def c(text, color):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def bar(current, maximum, width=20):
    filled = int(current / max(maximum, 1) * width)
    return c("█" * filled, "red") + c("░" * (width - filled), "dim")


# ─── Game display ───

SPECIAL_VARS = {
    "energyprefix": "E",
    "energy": "E",
}


def resolve_template(text, vars_dict):
    """Replace [VarName] in text with actual values from vars dict.
    Matches case-insensitively against the vars dict keys.
    Also handles special vars like energyPrefix."""
    if not text:
        return text
    import re

    # Build case-insensitive lookup from stats + special vars
    lower_vars = {}
    if vars_dict:
        lower_vars = {k.lower(): v for k, v in vars_dict.items()}

    def replacer(m):
        key = m.group(1)
        # Handle plural: [Cards:card|cards]
        if ":" in key and "|" in key:
            var_name, plural_spec = key.split(":", 1)
            val = lower_vars.get(var_name.lower())
            if val is not None:
                forms = plural_spec.split("|")
                return forms[0] if int(val) == 1 else (forms[1] if len(forms) > 1 else forms[0])
            return f"[{key}]"
        kl = key.lower()
        val = lower_vars.get(kl)
        if val is not None:
            return str(val)
        # Special vars
        if kl == "energyprefix":
            return ""  # prefix only, unit already added by energyIcons handler in desc()
        return f"[{key}]"

    return re.sub(r"\[([^\]]+)\]", replacer, text)


def card_desc(card):
    """Get resolved card description using stats as template vars."""
    d = desc(card.get("description", {}))
    stats = card.get("stats") or {}
    return resolve_template(d, stats)  # always resolve (handles energyPrefix etc.)


def relic_str(r):
    """Format a relic with name and resolved description."""
    if isinstance(r, dict) and "name" in r:
        name = n(r["name"])
        d = desc(r.get("description", {}))
        # Resolve template vars with actual values
        vars_dict = r.get("vars") or {}
        d = resolve_template(d, vars_dict)
        return f"{name}" + (f": {c(d, 'dim')}" if d else "")
    return n(r)


def potion_str(p):
    """Format a potion with name and resolved description."""
    if isinstance(p, dict) and "name" in p:
        name = n(p["name"])
        d = desc(p.get("description", {}))
        vars_dict = p.get("vars") or {}
        d = resolve_template(d, vars_dict) if vars_dict else d
        idx = p.get("index", "?")
        return f"[{idx}] {name}" + (f": {c(d, 'dim')}" if d else "")
    return n(p)


def _resolved_relic_desc(r):
    """Return description text for a relic dict, with vars resolved."""
    d = desc(r.get("description", {}))
    return resolve_template(d, r.get("vars") or {})


def _resolved_potion_desc(p):
    """Return description text for a potion dict, with vars resolved."""
    d = desc(p.get("description", {}))
    vars_dict = p.get("vars") or {}
    return resolve_template(d, vars_dict) if vars_dict else d


def _show_relics_detail(player):
    """Print all owned relics, numbered, with name then indented description."""
    relics = player.get("relics", []) or []
    if not relics:
        print("  No relics.")
        return
    print(
        f"\n  {c('Relics:', 'bold')}  {c('(use ', 'dim')}{c('relic <N|name>', 'cyan')}{c(' for one)', 'dim')}"
    )
    for i, r in enumerate(relics):
        name = n(r.get("name", "?"))
        d = _resolved_relic_desc(r)
        print(f"  [{i}] 🔶 {c(name, 'bold')}")
        if d:
            for line in d.split("\n"):
                if line.strip():
                    print(f"      {c(line, 'dim')}")


def _resolve_relic_query(relics, query):
    """Resolve `query` (index or name substring) to a relic dict, or None."""
    if not relics or not query:
        return None
    if query.isdigit():
        idx = int(query)
        if 0 <= idx < len(relics):
            return relics[idx]
        return None
    q = query.lower()
    for r in relics:
        if q in n(r.get("name", "")).lower():
            return r
    return None


def _show_one_relic(player, query):
    """Show full info for one relic, identified by index or name substring."""
    relics = player.get("relics", []) or []
    r = _resolve_relic_query(relics, query)
    if not r:
        print("  No matching relic.")
        return
    name = n(r.get("name", "?"))
    print(f"\n  🔶 {c(name, 'bold')}")
    d = _resolved_relic_desc(r)
    if d:
        for line in d.split("\n"):
            print(f"    {line}")
    vars_dict = r.get("vars") or {}
    if vars_dict:
        parts = [f"{k}={v}" for k, v in vars_dict.items()]
        print(f"    {c('Values: ' + ', '.join(parts), 'dim')}")


def _show_potions_detail(player):
    """Print all owned potions, numbered, with description on its own line."""
    pots = [p for p in (player.get("potions", []) or []) if p]
    if not pots:
        print("  No potions.")
        return
    print(
        f"\n  {c('Potions:', 'bold')}  {c('(use ', 'dim')}{c('potion <N|name>', 'cyan')}{c(' for one)', 'dim')}"
    )
    for p in pots:
        name = n(p.get("name", "?"))
        idx = p.get("index", "?")
        d = _resolved_potion_desc(p)
        print(f"  [{idx}] 🧪 {c(name, 'bold')}")
        if d:
            for line in d.split("\n"):
                if line.strip():
                    print(f"      {c(line, 'dim')}")


def _show_one_potion(player, query):
    """Show full info for one potion, identified by index or name substring."""
    pots = [p for p in (player.get("potions", []) or []) if p]
    if not pots:
        print("  No potions.")
        return
    target = None
    if query.isdigit():
        i = int(query)
        for p in pots:
            if p.get("index") == i:
                target = p
                break
    if target is None:
        q = query.lower()
        for p in pots:
            if q and q in n(p.get("name", "")).lower():
                target = p
                break
    if not target:
        print("  No matching potion.")
        return
    name = n(target.get("name", "?"))
    idx = target.get("index", "?")
    print(f"\n  [{idx}] 🧪 {c(name, 'bold')}")
    d = _resolved_potion_desc(target)
    if d:
        for line in d.split("\n"):
            print(f"    {line}")


def _boss_monster_key(boss_id):
    """Map a boss encounter id (e.g. 'VANTOM_BOSS') to a monster localization key."""
    if not boss_id:
        return ""
    key = boss_id[:-5] if boss_id.endswith("_BOSS") else boss_id
    if key == "THE_KIN":
        key = "KIN_PRIEST"
    return key


def _boss_known_moves(monster_key):
    """Return a list of move-name strings for a boss, looked up in localization."""
    cache = _load_loc()
    prefix = f"monsters:{monster_key}.moves."
    suffix = ".title"
    moves = []
    seen = set()
    for k, v in cache.items():
        if not k.startswith(prefix) or not k.endswith(suffix):
            continue
        move_id = k[len(prefix) : -len(suffix)]
        if move_id in seen:
            continue
        seen.add(move_id)
        moves.append(v)
    return moves


def _show_boss(state, send_fn=None):
    """Show act boss info: name and known move titles from localization."""
    ctx = state.get("context", {}) if isinstance(state, dict) else {}
    boss = ctx.get("boss") or {}
    if not boss and send_fn is not None:
        map_data = send_fn({"cmd": "get_map"})
        if isinstance(map_data, dict):
            boss = map_data.get("boss") or {}
    if not boss:
        print("  No boss info available.")
        return
    boss_id = boss.get("id", "") or ""
    name = n(boss.get("name", boss_id or "?"))
    monster_key = _boss_monster_key(boss_id)
    act_name = n(ctx.get("act_name", "?"))
    print(f"\n  {c('Act Boss', 'bold')} ({act_name}): {c(name, 'red')}")
    if monster_key:
        moves = _boss_known_moves(monster_key)
        if moves:
            print(f"  Known moves: {', '.join(moves)}")
        else:
            print(f"  {c('No move info available.', 'dim')}")


def show_player(p, show_deck=False):
    hp, mhp = p.get("hp", 0), p.get("max_hp", 1)
    blk = p.get("block", 0)
    gold = p.get("gold", 0)
    deck = p.get("deck_size", 0)
    name = n(p.get("name", "?"))

    print(
        f"  {c(name, 'bold')}  HP {bar(hp, mhp)} {c(f'{hp}/{mhp}', 'red')}"
        + (f"  {c(str(blk), 'blue')} blk" if blk > 0 else "")
        + f"  Gold {c(str(gold), 'yellow')}  Deck {deck}"
    )
    for r in p.get("relics", []):
        print(f"    🔶 {relic_str(r)}")
    for pot in p.get("potions", []):
        if pot:
            print(f"    🧪 {potion_str(pot)}")
    if show_deck:
        cards = p.get("deck", [])
        if cards:
            print(f"  {c('Deck:', 'bold')}")
            for cd in cards:
                up = c("+", "green") if cd.get("upgraded") else ""
                kws = cd.get("keywords") or []
                kw_str = " ".join(c(k, "dim") for k in kws)
                kw_str = f" [{kw_str}]" if kw_str else ""
                cd_d = card_desc(cd)
                print(
                    f"    {n(cd['name'])}{up} ({cd.get('cost', '?')}) {c(cd.get('type', ''), 'dim')}{kw_str}"
                )
                if cd_d:
                    print(f"      {c(cd_d, 'dim')}")
                stats = cd.get("stats") or {}
                aug_parts = _format_upgrade_preview(stats, cd.get("after_upgrade"), cd.get("cost"))
                if aug_parts:
                    print(f"      {c('upgrade:', 'green')} {', '.join(aug_parts)}")


class _HintResult:
    """Bundle of the collapsed dict view and the full per-target list.

    ``by_idx`` maps ``stable_idx → HintTag`` (best entry per stable_idx,
    same shape the inline column has always rendered). ``all_tags`` is
    the full list with one entry per (action, target_index) pair,
    sorted with the MCTS-best first.
    """

    __slots__ = ("all_tags", "by_idx")

    def __init__(self, by_idx, all_tags):
        self.by_idx = by_idx
        self.all_tags = all_tags


def _compute_hint_tags(state, cfg=None, *, force=False, timeout_ms=None):
    """Return a :class:`_HintResult` for ``state``, or ``None``.

    Crash-isolated wrapper around :func:`sts2_hint.recommender.recommend`.
    Any exception (adapter error, MCTS panic, unmodelled content, …) is
    swallowed and logged with a full traceback to the ``sts2_hint``
    rotating logger so a hint bug never breaks combat rendering.

    Returns ``None`` when hints are disabled, the hint package is
    unavailable, the state is not a ``combat_play`` decision, or any
    error was caught.

    The hint mode (``_HINT_MODE``) gates the *inline* render path:

    * ``off``: never compute (corresponds to ``--no-hints``).
    * ``on_demand``: don't compute from ``show_combat`` — set
      ``force=True`` from the ``/hint`` REPL verb to compute on demand.
    * ``inline`` (default): compute on every render.

    The wall-clock timeout is enforced via a worker thread plus
    :class:`threading.Event`. On timeout we log a warning and return
    ``None``. ``timeout_ms <= 0`` disables the cap.
    """
    if not _HINT_AVAILABLE or _hint_recommend_all is None:
        return None
    if _HINT_MODE == "off":
        return None
    if _HINT_MODE == "on_demand" and not force:
        return None
    effective_cfg = cfg if cfg is not None else _HINT_CFG
    if effective_cfg is None:
        return None
    if not isinstance(state, dict) or state.get("decision") != "combat_play":
        return None

    effective_timeout_ms = _HINT_TIMEOUT_MS if timeout_ms is None else timeout_ms

    result: dict = {}

    def _worker():
        try:
            result["all_tags"] = _hint_recommend_all(state, effective_cfg)
        except Exception as e:
            result["exc"] = traceback.format_exc()
            result["exc_summary"] = f"{type(e).__name__}: {e}"

    if effective_timeout_ms and effective_timeout_ms > 0:
        thread = threading.Thread(target=_worker, name="hint-recommend", daemon=True)
        thread.start()
        thread.join(timeout=effective_timeout_ms / 1000.0)
        if thread.is_alive():
            # Timeout: leave the worker running (daemon) and bail out.
            try:
                if _hint_get_logger is not None:
                    _hint_get_logger().warning(
                        "hint recommend() exceeded timeout_ms=%d; rendering without tags this turn",
                        effective_timeout_ms,
                    )
            except Exception:
                print(
                    f"[hint] recommend() exceeded timeout_ms={effective_timeout_ms}",
                    file=sys.stderr,
                )
            return None
    else:
        _worker()

    if "exc" in result:
        global _LAST_HINT_ERROR
        _LAST_HINT_ERROR = result.get("exc_summary") or "hint recommend() failed"
        try:
            if _hint_get_logger is not None:
                _hint_get_logger().error(
                    "hint recommend() failed; rendering without tags\n%s",
                    result["exc"],
                )
        except Exception:
            print(
                f"[hint] recommend() failed and logger errored:\n{result['exc']}",
                file=sys.stderr,
            )
        return None
    _LAST_HINT_ERROR = None
    all_tags = result.get("all_tags")
    if all_tags is None:
        return None
    by_idx: dict[int, object] = {}
    for tag in all_tags:
        prior = by_idx.get(tag.stable_idx)
        if (
            prior is None
            or (tag.is_best and not prior.is_best)
            or (not prior.is_best and tag.win_prob > prior.win_prob)
        ):
            by_idx[tag.stable_idx] = tag
    return _HintResult(by_idx=by_idx, all_tags=all_tags)


def _show_hints_on_demand(state):
    """Compute and print hints for ``state`` as a standalone block.

    Used by the ``/hint`` REPL verb when ``--hint-on-demand`` is on.
    Renders a numbered list of cards with ``[<BUCKET> pct%] <arrow>``
    suffixes. Non-card actions (end-turn, potions, …) are listed under
    a separate heading. Always renders even if ``_HINT_MODE != "off"``;
    use ``--no-hints`` to disable hints entirely.
    """
    if _HINT_MODE == "off" or not _HINT_AVAILABLE:
        print(f"  {c('Hints disabled.', 'dim')}")
        return
    hr = _compute_hint_tags(state, force=True)
    if not hr or not hr.all_tags:
        print(f"  {c('No hints available.', 'dim')}")
        return
    _render_ranked_actions(state, hr.all_tags)


def _render_hint_cell(hint_result, stable_idx):
    """Return a tag column cell for ``stable_idx``.

    ``hint_result`` is the :class:`_HintResult` from
    :func:`_compute_hint_tags`. Returns an empty-string when hints are
    disabled entirely (so the no-hint render path is bit-identical to
    the pre-T9 output). When hints are *active* but this row has no
    tag, returns a blank cell of :data:`sts2_hint.display.TAG_WIDTH`
    width for column alignment.
    """
    if hint_result is None or _hint_display is None:
        return ""
    tag = hint_result.by_idx.get(stable_idx)
    if tag is None:
        return _hint_display.format_empty() + "  "
    return _hint_display.format_tag(tag) + "  "


def _action_label(state, tag):
    """Return a short human-readable label for one :class:`HintTag`.

    Examples: ``"play [0] Strike → 1 Nibbit"``, ``"end turn"``,
    ``"potion p0 Strength → 0 Nibbit"``.
    """
    from sts2_hint.recommender import (
        CHANNEL_STABLE_IDX,
        CHOOSE_SKIP_STABLE_IDX,
        CHOOSE_STABLE_IDX_BASE,
        END_TURN_STABLE_IDX,
        EVOKE_STABLE_IDX,
        POTION_STABLE_IDX_BASE,
    )

    sidx = tag.stable_idx
    enemies = state.get("enemies") or []
    tgt_str = (
        " " + _hint_display.format_target(tag.target_index, enemies)
        if tag.target_index is not None
        else ""
    )

    if sidx == END_TURN_STABLE_IDX:
        return "end turn"
    if sidx == CHANNEL_STABLE_IDX:
        return f"channel{tgt_str}"
    if sidx == EVOKE_STABLE_IDX:
        return f"evoke{tgt_str}"
    if POTION_STABLE_IDX_BASE >= sidx > POTION_STABLE_IDX_BASE - 16:
        slot = POTION_STABLE_IDX_BASE - sidx
        pots = (state.get("player", {}) or {}).get("potions", []) or []
        pname = "?"
        if 0 <= slot < len(pots):
            p = pots[slot]
            if isinstance(p, dict):
                pname = n(p.get("name", "?"))
        return f"potion p{slot} {pname}{tgt_str}"
    if sidx == CHOOSE_SKIP_STABLE_IDX:
        # The pending-choice FINALIZE/SKIP (`selection == []`) — a distinct
        # action, not "choose 0" (see recommender.CHOOSE_SKIP_STABLE_IDX).
        return "choose skip"
    if sidx <= CHOOSE_STABLE_IDX_BASE:
        return f"choose {CHOOSE_STABLE_IDX_BASE - sidx}"
    # Card play: ``sidx`` is the live C# hand index from the recommender
    # (the wire-dict adapter sets stable_idx from ``card.index``). Look
    # up by ``card["index"]`` — the Python-side ``StableHand.stable_idx``
    # is a separate per-slot namespace that drifts mid-turn.
    hand = state.get("hand", []) or []
    cname = "?"
    for card in hand:
        if card.get("index") == sidx:
            cname = n(card.get("name", "?"))
            break
    return f"play [{sidx}] {cname}{tgt_str}"


def _render_ranked_actions(state, all_tags, *, limit=None):
    """Print every action tag ranked by win_prob with target labels.

    ``all_tags`` is the ``HintResult.all_tags`` list (one entry per
    (action, target_index) pair, MCTS-best first). When ``limit`` is
    set, only the top N rows render.
    """
    if not all_tags or _hint_display is None:
        return
    print(f"  {c('MCTS hints (ranked):', 'bold')}")
    rows = all_tags if limit is None else all_tags[:limit]
    for tag in rows:
        label = _action_label(state, tag)
        cell = _hint_display.format_tag(tag)
        extras = _hint_display.format_extras(tag)
        print(f"    {cell}  {extras}  {label}")
    print()


def show_combat(state, hint_cfg=None):
    rnd = state.get("round", 0)
    energy = state.get("energy", 0)
    max_energy = state.get("max_energy", 0)
    draw = state.get("draw_pile_count", 0)
    discard = state.get("discard_pile_count", 0)

    print(f"\n{'─' * 60}")
    print(
        f"  {c(f'Round {rnd}', 'bold')}  Energy {c(f'{energy}/{max_energy}', 'cyan')}  Draw {draw}  Discard {discard}"
    )
    show_player(state.get("player", {}))

    # Player powers/buffs/debuffs
    ppowers = state.get("player_powers") or []
    if ppowers:
        for pw in ppowers:
            amt = pw.get("amount", 0)
            amt_str = f" {amt}" if amt and amt != 0 else ""
            pw_desc = desc(pw.get("description", ""))
            if pw_desc and amt:
                pw_desc = resolve_template(
                    pw_desc, {"Amount": abs(amt) if isinstance(amt, (int, float)) else amt}
                )
            is_debuff = isinstance(amt, (int, float)) and amt < 0
            color = "red" if is_debuff else "green"
            label = "Debuff" if is_debuff else "Buff"
            desc_str = f": {c(pw_desc, 'dim')}" if pw_desc else ""
            pw_name = n(pw.get("name", "?"))
            print(f"    {c(label, color)} {c(f'{pw_name}{amt_str}', color)}{desc_str}")

    # Character-specific: Necrobinder's Osty (show near player)
    osty = state.get("osty")
    if osty:
        if osty.get("alive"):
            ohp, omhp = osty.get("hp", 0), osty.get("max_hp", 1)
            oblk = osty.get("block", 0)
            print(
                f"    🦴 {n(osty.get('name', 'Osty'))}  {bar(ohp, omhp)} {ohp}/{omhp}"
                + (f"  {c(str(oblk), 'blue')}blk" if oblk else "")
            )
        else:
            print(f"    🦴 {c('Osty (dead)', 'dim')}")

    # Character-specific: Defect's Orbs
    orbs = state.get("orbs")
    if orbs:
        orb_icons = {"Lightning": "⚡", "Frost": "❄", "Dark": "🌑", "Plasma": "🔆", "Glass": "💠"}
        orb_parts = []
        for orb in orbs:
            otype = orb.get("type", "?")
            icon = orb_icons.get(otype, "○")
            pv, ev = orb.get("passive", 0), orb.get("evoke", 0)
            orb_parts.append(f"{icon}{n(orb.get('name', otype))}({pv}/{ev})")
        slots = state.get("orb_slots", len(orbs))
        print(f"    Orbs [{len(orbs)}/{slots}]: {' '.join(orb_parts)}")

    # Character-specific: Regent's Stars
    stars = state.get("stars")
    if stars is not None:
        print(f"    ⭐ Stars: {c(str(stars), 'yellow')}")

    print()
    for e in state.get("enemies", []):
        hp, mhp = e.get("hp", 0), e.get("max_hp", 1)
        blk = e.get("block", 0)

        # Build intent string from detailed intents
        intents = e.get("intents") or []
        intent_parts = []
        for it in intents:
            itype = it.get("type", "")
            dmg = it.get("damage")
            hits = it.get("hits")
            if itype == "Attack":
                if dmg is not None:
                    if hits and hits > 1:
                        intent_parts.append(c(f"⚔ {dmg}x{hits}", "red"))
                    else:
                        intent_parts.append(c(f"⚔ {dmg}", "red"))
                else:
                    intent_parts.append(c("⚔ ATK", "red"))
            elif itype == "Defend":
                intent_parts.append(c("🛡 DEF", "blue"))
            elif itype in ("Buff", "Heal"):
                intent_parts.append(c(f"⬆ {itype}", "magenta"))
            elif itype == "Debuff":
                intent_parts.append(c("⬇ Debuff", "yellow"))
            elif itype == "DebuffStrong":
                intent_parts.append(c("⬇ Strong", "yellow"))
            elif itype in ("CardDebuff", "StatusCard"):
                intent_parts.append(c("⬇ Cards", "yellow"))
            elif itype == "DeathBlow":
                if dmg is not None:
                    intent_parts.append(c(f"💀 {dmg}", "red"))
                else:
                    intent_parts.append(c("💀 KILL", "red"))
            elif itype == "Escape":
                intent_parts.append(c("🏃 Escape", "dim"))
            elif itype == "Summon":
                intent_parts.append(c("📢 Summon", "magenta"))
            elif itype == "Sleep":
                intent_parts.append(c("💤 Sleep", "dim"))
            elif itype == "Stun":
                intent_parts.append(c("⚡ Stun", "yellow"))
            elif itype == "Hidden":
                intent_parts.append(c("? ???", "dim"))
            elif itype:
                intent_parts.append(c(itype, "dim"))
        intent_str = " ".join(intent_parts) if intent_parts else c("? ???", "dim")

        # Enemy powers
        powers = e.get("powers") or []
        power_str = ""
        if powers:
            pw_parts = [f"{n(pw['name'])} {pw.get('amount', '')}" for pw in powers]
            power_str = "  " + c(", ".join(pw_parts), "dim")

        print(
            f"  [{e['index']}] {n(e['name'])}  {bar(hp, mhp)} {hp}/{mhp}"
            + (f"  {c(str(blk), 'blue')} blk" if blk else "")
            + f"  {intent_str}{power_str}"
        )

    print()
    hand = state.get("hand", [])
    # T9: compute hint tags once for this decision point. Crash-isolated
    # in _compute_hint_tags; returns None on disable / error.
    hint_tags = _compute_hint_tags(state, hint_cfg)
    for card in hand:
        cost = card.get("cost", 0)
        playable = card.get("can_play", False)
        ctype = card.get("type", "?")
        target = card.get("target_type", "")

        type_color = {
            "Attack": "red",
            "Skill": "blue",
            "Power": "magenta",
            "Status": "dim",
            "Curse": "dim",
        }.get(ctype, "reset")
        mark = c("●", "green") if playable else c("○", "dim")
        star_cost = card.get("star_cost", 0)
        cost_str = c(str(cost), "cyan")
        if star_cost > 0:
            cost_str += f"+{c(f'{star_cost}⭐', 'yellow')}"

        # Show damage/block inline, plus description for non-trivial cards
        stats = card.get("stats") or {}
        stat_parts = []
        if "damage" in stats:
            stat_parts.append(c(f"{stats['damage']}dmg", "red"))
        if "block" in stats:
            stat_parts.append(c(f"{stats['block']}blk", "blue"))
        stat_str = " ".join(stat_parts)

        # Show description if card has no damage/block (so effect isn't obvious from stats)
        cd_d = card_desc(card)
        extra_desc = ""
        if cd_d and "damage" not in stats and "block" not in stats:
            # No damage/block — show full description inline
            extra_desc = f"  {c(cd_d.replace(chr(10), ' '), 'dim')}"
        elif cd_d and any(k not in ("damage", "block") for k in stats):
            # Has extra effects beyond damage/block — show last line
            lines = cd_d.split("\n")
            if len(lines) > 1:
                extra_desc = f"  {c(lines[-1], 'dim')}"

        # Show keywords (Innate, Exhaust, Ethereal, etc.)
        kws = card.get("keywords") or []
        kw_str = " ".join(c(k, "dim") for k in kws) if kws else ""
        if kw_str:
            kw_str = f" [{kw_str}]"
        ench = card.get("enchantment")
        ench_str = f" {c(n(ench), 'magenta')}" if ench else ""

        # Prefer stable_idx (set by StableHand). Falls back to live index when
        # the snapshot isn't initialised yet (e.g., one-off renders).
        display_idx = card.get("stable_idx", card.get("index"))
        # The recommender's ``HintTag.stable_idx`` is the live C# hand
        # index (the wire-dict adapter copies ``card.index`` into the
        # wire ``stable_idx`` field). StableHand uses a separate
        # per-slot namespace that drifts after a play, so look the
        # hint up by ``card["index"]`` rather than ``display_idx``.
        hint_cell = _render_hint_cell(hint_tags, card.get("index", display_idx))
        print(
            f"  {hint_cell}{mark} [{display_idx}] {c(n(card['name']), type_color)}{ench_str} ({cost_str}) {stat_str}{kw_str}{extra_desc}"
            + (f"  {c('→', 'yellow')}" if target == "AnyEnemy" else "")
        )

    # Ranked-actions block: one row per (action, target_index) pair so
    # the user can see per-enemy win-prob for multi-target attacks and
    # see scores for potions / end-turn that don't fit on the hand list.
    if hint_tags is not None and hint_tags.all_tags:
        print()
        _render_ranked_actions(state, hint_tags.all_tags)


def show_map(state, send_fn=None):
    """Show map at map_select. Fetches full map if send_fn available."""
    choices = state.get("choices", [])
    choice_set = {(ch["col"], ch["row"]) for ch in choices}

    # Try to fetch full map for richer display
    if send_fn:
        map_data = send_fn({"cmd": "get_map"})
        if map_data and map_data.get("type") == "map":
            # Build index map: (col,row) → choice index
            choice_indices = {(ch["col"], ch["row"]): i for i, ch in enumerate(choices)}
            _render_map(map_data, choice_set, choice_indices)
            return

    # Fallback: simple list
    ctx = state.get("context", {})
    act_name = n(ctx.get("act_name", "?"))
    floor = ctx.get("floor", "?")
    print(f"\n{'═' * 60}")
    print(f"  {c(f'{act_name}', 'bold')} Floor {floor}")
    show_player(state.get("player", {}))
    print()
    type_icons = {
        "Monster": "⚔",
        "Elite": "💀",
        "Boss": "👹",
        "RestSite": "🏕",
        "Shop": "🏪",
        "Treasure": "💎",
        "Event": "❓",
        "Unknown": "❓",
        "Ancient": "🏛",
    }
    for i, ch in enumerate(choices):
        icon = type_icons.get(ch["type"], "?")
        ntype = ch["type"]
        print(f"  [{i}] {icon} {ntype}")


def _format_upgrade_preview(stats, aug, current_cost=None):
    """Format upgrade preview string."""
    if not aug:
        return None
    aug_stats = aug.get("stats") or {}
    parts = []
    # Cost change
    aug_cost = aug.get("cost")
    if current_cost is not None and aug_cost is not None and aug_cost != current_cost:
        parts.append(c(f"cost {current_cost}→{aug_cost}", "green"))
    # Compare all stats, show changed values with readable names
    all_keys = set(list(stats.keys()) + list(aug_stats.keys()))
    for k in sorted(all_keys):
        old = stats.get(k, 0)
        new_val = aug_stats.get(k, old)
        if new_val != old:
            if k == "damage":
                parts.append(c(f"dmg {old}→{new_val}", "red"))
            elif k == "block":
                parts.append(c(f"blk {old}→{new_val}", "blue"))
            else:
                parts.append(c(f"{old}→{new_val}", "green"))
    # Keyword changes (e.g., Discovery removes Exhaust)
    for kw in aug.get("removed_keywords") or []:
        parts.append(c(f"-{kw}", "green"))
    for kw in aug.get("added_keywords") or []:
        parts.append(c(f"+{kw}", "yellow"))
    return parts


def show_card_reward(state):
    print(f"\n{'─' * 60}")
    gold_earned = state.get("gold_earned", 0)
    if gold_earned > 0:
        print(f"  {c('Combat won!', 'green')} +{c(str(gold_earned), 'yellow')}g")
    print(f"  {c('Card Reward', 'bold')} — choose one (or skip)")
    show_player(state.get("player", {}))
    print()
    for card in state.get("cards", []):
        ctype = card.get("type", "?")
        rarity = card.get("rarity", "Common")
        cost = card.get("cost", "?")
        type_color = {"Attack": "red", "Skill": "blue", "Power": "magenta"}.get(ctype, "reset")
        rarity_label = rarity
        rarity_color = {"Rare": "yellow", "Uncommon": "cyan"}.get(rarity, "dim")
        stats = card.get("stats") or {}
        cd_desc = card_desc(card)

        print(
            f"  [{card['index']}] {c(n(card['name']), type_color)} ({cost}) {c(rarity_label, rarity_color)}"
        )
        if cd_desc:
            print(f"      {c(cd_desc, 'dim')}")
        # Show upgrade preview
        aug_parts = _format_upgrade_preview(stats, card.get("after_upgrade"), card.get("cost"))
        if aug_parts:
            print(f"      {c('upgrade:', 'green')} {', '.join(aug_parts)}")


def show_shop(state):
    print(f"\n{'─' * 60}")
    print(f"  {c('Shop', 'bold')}")
    show_player(state.get("player", {}))
    gold = state.get("player", {}).get("gold", 0)

    print(f"\n  {c('Cards:', 'bold')}")
    for card in state.get("cards", []):
        if not card.get("is_stocked"):
            continue
        cost = card.get("cost", 0)
        affordable = c(str(cost), "green") if cost <= gold else c(str(cost), "red")
        sale = c(" SALE", "yellow") if card.get("on_sale") else ""
        cc = card.get("card_cost", "?")
        print(
            f"  [{card['index']}] {n(card['name'])} ({cc}) {c(card.get('type', '?'), 'dim')} — {affordable}g{sale}"
        )
        cd_desc = card_desc(card)
        if cd_desc:
            print(f"      {c(cd_desc, 'dim')}")
        aug_parts = _format_upgrade_preview(
            card.get("stats") or {}, card.get("after_upgrade"), card.get("card_cost")
        )
        if aug_parts:
            print(f"      {c('upgrade:', 'green')} {', '.join(aug_parts)}")

    print(f"\n  {c('Relics:', 'bold')}")
    for r in state.get("relics", []):
        if not r.get("is_stocked"):
            continue
        cost = r.get("cost", 0)
        affordable = c(str(cost), "green") if cost <= gold else c(str(cost), "red")
        r_desc = desc(r.get("description", ""))
        print(f"  [r{r['index']}] {n(r['name'])} — {affordable}g")
        if r_desc:
            print(f"      {c(r_desc, 'dim')}")

    print(f"\n  {c('Potions:', 'bold')}")
    for p in state.get("potions", []):
        if not p.get("is_stocked"):
            continue
        cost = p.get("cost", 0)
        affordable = c(str(cost), "green") if cost <= gold else c(str(cost), "red")
        p_desc = desc(p.get("description", ""))
        print(f"  [p{p['index']}] {n(p['name'])} — {affordable}g")
        if p_desc:
            print(f"      {c(p_desc, 'dim')}")

    removal_cost = state.get("card_removal_cost")
    if removal_cost:
        affordable = (
            c(str(removal_cost), "green") if removal_cost <= gold else c(str(removal_cost), "red")
        )
        print(f"\n  [rm] Remove a card — {affordable}g")

    print("\n  [leave] Leave shop")


def show_rest_site(state):
    print(f"\n{'─' * 60}")
    ctx = state.get("context", {})
    if ctx:
        print(f"  {c(n(ctx.get('act_name', '?')), 'dim')} Floor {ctx.get('floor', '?')}")
    print(f"  {c('Rest Site', 'bold')}")
    show_player(state.get("player", {}))
    print()
    for opt in state.get("options", []):
        enabled = opt.get("is_enabled", True)
        mark = c("●", "green") if enabled else c("○", "dim")
        opt_id = opt.get("option_id", "?")
        opt_name = opt_id
        opt_desc = opt.get("name", "")
        print(
            f"  {mark} [{opt['index']}] {opt_name}"
            + (f" — {opt_desc}" if opt_desc and opt_desc != opt_id else "")
        )


def _load_loc():
    """Load localization data for resolving event option names."""
    cache = getattr(_load_loc, "_cache", None)
    if cache is None:
        cache = {}
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d = os.path.join(base, "localization_eng")
        if os.path.isdir(d):
            for f in os.listdir(d):
                if not f.endswith(".json"):
                    continue
                try:
                    data = json.load(open(os.path.join(d, f), encoding="utf-8"))
                except Exception:
                    continue
                table = f[:-5]
                cache.setdefault(table, {})
                for k, v in data.items():
                    cache.setdefault(f"{table}:{k}", v)
        _load_loc._cache = cache  # type: ignore[attr-defined]
    return cache


def loc_resolve(key):
    """Resolve a loc key like 'NEOW.pages.INITIAL.options.PRECISE_SCISSORS.title' to readable text."""
    cache = _load_loc()
    for table in ["events", "relics", "ancients", "cards", "potions", "monsters"]:
        val = cache.get(f"{table}:{key}")
        if val:
            return val
    parts = key.split(".")
    for p in reversed(parts):
        if p not in ("title", "description", "options", "pages", "INITIAL"):
            relic_name = cache.get(f"relics:{p}.title")
            if relic_name:
                d = desc(cache.get(f"relics:{p}.description", ""))
                return f"{relic_name}" + (f" — {c(d, 'dim')}" if d else "")
            return p.replace("_", " ").title()
    return key


def show_event(state):
    print(f"\n{'─' * 60}")
    event_name = state.get("event_name", "?")
    # event_name is now bilingual dict {"en": ..., "zh": ...} or plain string
    event_display = n(event_name) if isinstance(event_name, dict) else event_name
    event_desc = state.get("description", "")
    # Show context
    ctx = state.get("context", {})
    if ctx:
        act = n(ctx.get("act_name", "?"))
        floor = ctx.get("floor", "?")
        print(f"  {c(act, 'dim')} Floor {floor}")
    event_label = "Event"
    print(f"  {c(f'{event_label}: {event_display}', 'bold')}")
    # event_desc is usually a raw loc key — skip it (event name already in title)
    show_player(state.get("player", {}))
    print()
    for opt in state.get("options", []):
        locked = opt.get("is_locked", False)
        mark = c("○", "dim") if locked else c("●", "green")
        raw_title = opt.get("title", opt.get("text_key", f"Option {opt['index']}"))
        # title is now bilingual dict or loc key string
        if isinstance(raw_title, dict):
            title = n(raw_title)
        else:
            title = (
                loc_resolve(raw_title)
                if "." in str(raw_title) or str(raw_title).isupper()
                else raw_title
            )
        # Show option description with resolved template vars
        raw_desc = opt.get("description")
        opt_desc = desc(raw_desc) if raw_desc else ""
        # Resolve template vars like [MaxHp], [Gold], {Cards}
        opt_vars = opt.get("vars") or {}
        if opt_vars and opt_desc:
            opt_desc = resolve_template(opt_desc, opt_vars)
        desc_str = f" — {c(opt_desc, 'dim')}" if opt_desc else ""
        print(f"  {mark} [{opt['index']}] {title}{desc_str}")


# ─── Input handling ───


def _render_map(map_data, choice_set=None, choice_indices=None):
    """Render map as a grid with connection lines between rows."""
    if choice_set is None:
        choice_set = set()
    if choice_indices is None:
        choice_indices = {}

    ctx = map_data.get("context", {})
    act = n(ctx.get("act_name", "?"))
    floor_n = ctx.get("floor", "?")
    cur = map_data.get("current_coord")

    ICONS = {
        "Monster": "M",
        "Elite": "E",
        "Boss": "B",
        "RestSite": "R",
        "Shop": "$",
        "Treasure": "T",
        "Event": "?",
        "Unknown": "?",
        "Ancient": "A",
    }

    rows = map_data.get("rows", [])
    if not rows:
        return

    # Collect nodes and edges
    node_map = {}
    max_col = 0
    row_numbers = set()
    # edges_up[lower_row] = [(from_col, to_col), ...] where to is in the row above
    edges_up = {}
    for row in rows:
        for nd in row:
            col, rn = nd.get("col", 0), nd.get("row", 0)
            node_map[col, rn] = nd
            max_col = max(max_col, col)
            row_numbers.add(rn)
            for ch in nd.get("children") or []:
                edges_up.setdefault(rn, []).append((col, ch["col"]))

    row_numbers = sorted(row_numbers)
    total_cols = max_col + 1
    W = 4  # chars per column cell
    # Center of column c = c*W + W//2 = c*4 + 2

    width = W * total_cols + 6
    print(f"\n{'═' * width}")
    print(f"  {c(act, 'bold')} — Floor {floor_n}")
    # Show current position if it's not on the map grid (e.g., starting row 0)
    if cur and cur.get("row", -1) not in row_numbers:
        print(f"  {c('You are at the start', 'green')}")
    print()

    # Boss row
    boss = map_data.get("boss", {})
    boss_col = boss.get("col", 0)
    boss_row = boss.get("row", -1)
    boss_name = n(boss.get("name", "")) if boss else ""
    buf = list(" " * (W * total_cols))
    buf[boss_col * W + W // 2] = "B"
    line = "".join(buf)
    line = line[: boss_col * W + W // 2] + c("B", "red") + line[boss_col * W + W // 2 + 1 :]
    boss_label = f"  {c('B', 'dim')} | {line}"
    if boss_name:
        boss_label += f"   {c(boss_name, 'red')}"
    print(boss_label)

    # Connection from top row to boss
    top_rn = row_numbers[-1] if row_numbers else -1
    conn = list(" " * (W * total_cols))
    for fc, tc in edges_up.get(top_rn, []):
        if tc == boss_col:  # this edge's target row should be boss
            pass
    # Actually, edges_up[top_rn] has edges from top_rn to its children.
    # Children of top row nodes go to boss.
    for nd_row in rows:
        for nd in nd_row:
            if nd.get("row") == top_rn:
                for ch in nd.get("children") or []:
                    if ch.get("row") == boss_row:
                        fc, tc = nd["col"], ch["col"]
                        _draw_conn(conn, fc, tc, W)
    print(f"    | {c(''.join(conn), 'dim')}")

    # Map rows (top to bottom)
    for idx in range(len(row_numbers) - 1, -1, -1):
        rn = row_numbers[idx]

        # --- Node line ---
        buf = list(" " * (W * total_cols))
        color_subs = []  # (start_pos, end_pos, colored_str)
        for col in range(total_cols):
            nd = node_map.get((col, rn))
            if not nd:
                continue
            icon = ICONS.get(nd.get("type", "?"), "·")
            is_cur = cur and cur["col"] == col and cur["row"] == rn
            is_choice = (col, rn) in choice_set
            visited = nd.get("visited", False)

            center = col * W + W // 2
            choice_idx = choice_indices.get((col, rn))
            if is_cur:
                buf[center - 1] = "["
                buf[center] = icon
                buf[center + 1] = "]"
                color_subs.append((center - 1, center + 2, c(f"[{icon}]", "green")))
            elif choice_idx is not None:
                buf[center] = icon
                color_subs.append((center, center + 1, c(icon, "yellow")))
            elif visited:
                buf[center] = icon
                color_subs.append((center, center + 1, c(icon, "dim")))
            else:
                buf[center] = icon

        line = "".join(buf)
        # Apply colors right-to-left
        for start, end, colored in sorted(color_subs, key=lambda x: -x[0]):
            line = line[:start] + colored + line[end:]
        print(f"  {rn:>2}| {line}")

        # --- Choice index annotation line ---
        row_choices = {
            col: choice_indices[col, rn] for col in range(total_cols) if (col, rn) in choice_indices
        }
        if row_choices:
            ann = list(" " * (W * total_cols))
            ann_subs = []
            for col, idx in row_choices.items():
                label = f"[{idx}]"
                start = col * W + W // 2 - 1
                for j, ch in enumerate(label):
                    if 0 <= start + j < len(ann):
                        ann[start + j] = ch
                ann_subs.append((start, start + len(label), c(label, "yellow")))
            ann_line = "".join(ann)
            for start, end, colored in sorted(ann_subs, key=lambda x: -x[0]):
                ann_line = ann_line[:start] + colored + ann_line[end:]
            print(f"    | {ann_line}")

        # --- Connection line below this row (edges from row below going up to this row) ---
        if idx > 0:
            below_rn = row_numbers[idx - 1]
            conn = list(" " * (W * total_cols))
            for fc, tc in edges_up.get(below_rn, []):
                # fc is in below_rn, tc is the child row
                # We need edges where child row == rn
                pass
            # Rebuild: iterate edges from below_rn whose children are in rn
            for nd_row in rows:
                for nd in nd_row:
                    if nd.get("row") != below_rn:
                        continue
                    for ch in nd.get("children") or []:
                        if ch.get("row") == rn:
                            _draw_conn(conn, nd["col"], ch["col"], W)
            print(f"    | {c(''.join(conn), 'dim')}")

    # Legend
    print(f"  {'─' * width}")
    legend = (
        f"  M=Monster E=Elite R=Rest "
        f"$=Shop T=Treasure ?=Event "
        f"{c('[x]', 'green')}=You {c('0', 'yellow')}=Choice"
    )
    print(legend)
    # Show choice details
    if choice_indices:
        inv = {v: k for k, v in choice_indices.items()}
        parts = []
        for i in sorted(inv.keys()):
            col, row = inv[i]
            nd = node_map.get((col, row))
            if nd:
                ntype = nd.get("type", "?")
                parts.append(f"{c(str(i), 'yellow')}={ntype}")
        print(f"  {' '.join(parts)}")
    print()


def _draw_conn(buf, from_col, to_col, W):
    """Draw a connection between two columns on one line.
    from_col = lower row node, to_col = upper row node.
    Single char at midpoint: | for straight, / for up-right, \\ for up-left."""
    fc = from_col * W + W // 2
    tc = to_col * W + W // 2
    if from_col == to_col:
        if 0 <= fc < len(buf):
            buf[fc] = "|"
    else:
        mid = (fc + tc) // 2
        ch = "/" if from_col < to_col else "\\"
        if 0 <= mid < len(buf):
            buf[mid] = ch


def get_input(prompt, valid_options=None, state=None):
    """Get user input with validation. Supports meta-commands: help, map, deck, potions."""
    while True:
        try:
            raw = input(f"\n{c('>', 'green')} {prompt}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise _QuitRequested()

        if not raw:
            continue

        # Meta-commands available at any prompt
        if raw == "help":
            print(f"""
  {c("Commands:", "bold")}
    {c("help", "cyan")}     — show this help
    {c("map", "cyan")}      — show map
    {c("deck", "cyan")}     — show full deck list
    {c("draw", "cyan")}     — show draw pile (combat only)
    {c("discard", "cyan")}  — show discard pile (combat only)
    {c("exhaust", "cyan")}  — show exhaust pile (combat only)
    {c("potions", "cyan")}  — show potions ({c("potion <N|name>", "cyan")} for one)
    {c("relics", "cyan")}   — show relics ({c("relic <N|name>", "cyan")} for one)
    {c("boss", "cyan")}     — show act boss (name + known moves)
    {c("quit", "cyan")}     — quit
    {c("abandon", "cyan")}  — abandon run (forfeit)
    {c("save", "cyan")}     — save game
    {c("saves", "cyan")}    — list saves

  {c("Actions:", "bold")}
    Map:     path number (0, 1, 2)
    Combat:  {c("N", "yellow")} play card / {c("NtM", "yellow")} card N at enemy M / {c("e", "yellow")} end turn / {c("pN", "yellow")} potion
             {c("g", "yellow")} play one MCTS-recommended move / {c("a", "yellow")} MCTS plays the rest of the turn
             card indices are STABLE within a turn (don't shift after each play)
             batch: separate tokens with spaces, e.g. {c("0t1 1t0 e", "yellow")}
    Reward:  card index / {c("s", "yellow")} skip
    Rest:    option index
    Event:   option index / {c("leave", "yellow")} leave
    Shop:    {c("c0", "yellow")} card / {c("r0", "yellow")} relic / {c("p0", "yellow")} potion / {c("rm", "yellow")} remove / {c("leave", "yellow")} leave

  {c("Combat math:", "bold")}
    Vulnerable: enemy takes 50% more damage from Attacks while active.
    Block: resets at the start of your next turn unless a card/relic says
           otherwise. Player block also resets after enemy attacks resolve.
""")
            continue
        if raw == "deck" and state:
            p = state.get("player", {})
            show_player(p, show_deck=True)
            continue
        if raw in ("draw", "drawpile", "draw_pile") and state:
            _show_pile(state.get("draw_pile", []), "Draw pile", state)
            continue
        if raw in ("discard", "discardpile", "discard_pile") and state:
            _show_pile(state.get("discard_pile", []), "Discard pile", state)
            continue
        if raw in ("exhaust", "exhaustpile", "exhaust_pile") and state:
            _show_pile(state.get("exhaust_pile", []), "Exhaust pile", state)
            continue
        if raw == "potions" and state:
            _show_potions_detail(state.get("player", {}))
            continue
        if raw.startswith("potion ") and state:
            _show_one_potion(state.get("player", {}), raw[len("potion ") :].strip())
            continue
        if raw == "relics" and state:
            _show_relics_detail(state.get("player", {}))
            continue
        if raw.startswith("relic ") and state:
            _show_one_relic(state.get("player", {}), raw[len("relic ") :].strip())
            continue
        if raw == "boss" and state:
            send_fn = getattr(get_input, "_send", None)
            _show_boss(state, send_fn)
            continue
        if raw == "map":
            # Fetch full map from CLI
            if hasattr(get_input, "_send"):
                map_data = get_input._send({"cmd": "get_map"})
                if map_data and map_data.get("type") == "map":
                    _render_map(map_data)
                else:
                    print("  Map not available.")
            elif state:
                ctx = state.get("context", {})
                print(f"  {c(n(ctx.get('act_name', '?')), 'bold')} Floor {ctx.get('floor', '?')}")
            continue
        if raw == "save":
            if hasattr(get_input, "_save_fn"):
                get_input._save_fn()
            else:
                print("  Save not available.")
            continue
        if raw == "saves":
            saves = _list_saves()
            if saves:
                print(f"\n  {c('Saved games:', 'bold')}")
                for s in saves:
                    print(
                        f"    {c(s['file'], 'cyan')}  {s['character']}  Seed:{s['seed']}  Actions:{s['actions']}"
                    )
                print(f"\n  Load with: python3 play.py --load saves/{saves[0]['file']}")
            else:
                print("  No saves found.")
            continue
        if raw in ("/hint", "hint") and state:
            _show_hints_on_demand(state)
            continue
        if raw == "quit":
            raise _QuitRequested()
        if raw == "abandon":
            confirm = input("  Abandon this run? (y/n): ")
            if confirm.strip().lower() in ("y", "yes"):
                raise KeyboardInterrupt("abandon")
            continue

        if valid_options and raw not in valid_options:
            print(f"  Invalid. Options: {', '.join(sorted(valid_options))}")
            continue
        return raw


# ─── Main game loop ───


def _save_game(save_path, character, seed, action_log):
    """Write action replay save file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {"character": character, "seed": seed, "actions": action_log}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_game(save_path):
    """Read action replay save file. Returns (character, seed, actions)."""
    with open(save_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["character"], data["seed"], data["actions"]


def _list_saves():
    """List available save files."""
    if not os.path.isdir(SAVE_DIR):
        return []
    saves = []
    for f in sorted(os.listdir(SAVE_DIR)):
        if f.endswith(".json"):
            path = os.path.join(SAVE_DIR, f)
            try:
                with open(path, encoding="utf-8") as fh:
                    d = json.load(fh)
                saves.append(
                    {
                        "file": f,
                        "path": path,
                        "character": d.get("character", "?"),
                        "seed": d.get("seed", "?"),
                        "actions": len(d.get("actions", [])),
                    }
                )
            except Exception:
                pass
    return saves


class StableHand:
    """Tracks stable per-turn card indices in combat.

    The C# backend renumbers hand indices after every play (a card at live
    index 2 becomes live index 1 once the card before it leaves). That makes
    multi-card turns hard to plan since the user can't say "play card 0, then
    card 2, then card 4" up front. This class assigns each card a stable index
    when it enters the hand and remembers which stable slots are still in
    play, mapping them to live indices on demand.
    """

    def __init__(self):
        # Each slot: {stable_idx, card_id, name, in_hand, live_idx}
        self.slots = []
        self.last_round = -1
        self.in_combat = False

    def update(self, state):
        """Reconcile snapshot with the latest combat state. Mutates the
        hand entries in `state` to add a `stable_idx` field. Returns the hand."""
        if state.get("decision") != "combat_play":
            self.in_combat = False
            self.slots = []
            return state.get("hand", [])

        round_num = state.get("round", 0)
        hand = state.get("hand", [])

        if not self.in_combat or round_num != self.last_round:
            # Fresh turn: reset all stable indices.
            self.slots = [
                {
                    "stable_idx": i,
                    "card_id": c.get("id"),
                    "name": c.get("name", "?"),
                    "in_hand": True,
                    "live_idx": i,
                }
                for i, c in enumerate(hand)
            ]
            self.last_round = round_num
            self.in_combat = True
        else:
            self._reconcile(hand)

        # Annotate hand cards with their stable_idx.
        in_hand_slots = [s for s in self.slots if s["in_hand"]]
        for live_idx, card in enumerate(hand):
            if live_idx < len(in_hand_slots):
                card["stable_idx"] = in_hand_slots[live_idx]["stable_idx"]
            else:
                card["stable_idx"] = card.get("index", live_idx)
        return hand

    def _reconcile(self, new_hand):
        """Walk in-hand slots and new_hand together, marking removed slots
        consumed and appending newly-added cards as fresh stable slots."""
        in_hand_slots = [s for s in self.slots if s["in_hand"]]
        i = 0
        for live_idx, card in enumerate(new_hand):
            cid = card.get("id")
            j = i
            while j < len(in_hand_slots) and in_hand_slots[j]["card_id"] != cid:
                in_hand_slots[j]["in_hand"] = False
                j += 1
            if j < len(in_hand_slots):
                in_hand_slots[j]["live_idx"] = live_idx
                i = j + 1
            else:
                # Card added mid-turn (rare; e.g., card-effect hand insert).
                next_idx = max((s["stable_idx"] for s in self.slots), default=-1) + 1
                new_slot = {
                    "stable_idx": next_idx,
                    "card_id": cid,
                    "name": card.get("name", "?"),
                    "in_hand": True,
                    "live_idx": live_idx,
                }
                self.slots.append(new_slot)
                in_hand_slots.append(new_slot)
                i = len(in_hand_slots)
        while i < len(in_hand_slots):
            in_hand_slots[i]["in_hand"] = False
            i += 1

    def stable_to_live(self, stable_idx):
        """Resolve a stable index to the current live (C#) hand index, or None."""
        for s in self.slots:
            if s["stable_idx"] == stable_idx and s["in_hand"]:
                return s["live_idx"]
        return None

    def mark_consumed(self, stable_idx):
        """Force a slot to consumed state. Call before update() to give the
        reconcile algorithm an authoritative hint about which specific slot
        was played (matters when multiple cards share a card_id)."""
        for s in self.slots:
            if s["stable_idx"] == stable_idx:
                s["in_hand"] = False
                return


def _best_action_token(state, stable_hand):
    """Run MCTS on ``state`` and return the best action as a batch token.

    Returns a dict with the same shape :func:`_parse_batch_token` emits
    (``kind`` ∈ {``"card"``, ``"potion"``, ``"end"``}, plus ``idx`` and
    ``target``), or ``None`` when no hint is available.
    """
    from sts2_hint.recommender import END_TURN_STABLE_IDX, POTION_STABLE_IDX_BASE

    hr = _compute_hint_tags(state, force=True)
    if not hr or not hr.all_tags:
        return None
    best = next((tag for tag in hr.all_tags if tag.is_best), hr.all_tags[0])
    sidx = best.stable_idx
    if sidx == END_TURN_STABLE_IDX:
        return {"kind": "end", "idx": None, "target": None}
    if POTION_STABLE_IDX_BASE >= sidx > POTION_STABLE_IDX_BASE - 16:
        slot = POTION_STABLE_IDX_BASE - sidx
        return {"kind": "potion", "idx": slot, "target": best.target_index}
    if sidx >= 0:
        return {"kind": "card", "idx": sidx, "target": best.target_index}
    return None


def _format_action_token(tok, state):
    """One-line label for the executed action (used in `go`/`auto` echo).

    For ``card`` tokens emitted by :func:`_best_action_token`, ``idx``
    is the live C# hand index at MCTS-run time — look the card up via
    ``card["index"]`` rather than the Python-side ``stable_idx``, which
    reflects :class:`StableHand`'s separate per-slot numbering.
    """
    kind = tok["kind"]
    enemies = state.get("enemies") or []
    tgt_str = ""
    if tok.get("target") is not None and _hint_display is not None:
        tgt_str = " " + _hint_display.format_target(tok["target"], enemies)
    if kind == "end":
        return "end turn"
    if kind == "potion":
        slot = tok["idx"]
        pots = (state.get("player", {}) or {}).get("potions", []) or []
        pname = "?"
        if 0 <= slot < len(pots):
            p = pots[slot]
            if isinstance(p, dict):
                pname = n(p.get("name", "?"))
        return f"potion p{slot} {pname}{tgt_str}"
    if kind == "card":
        live_idx = tok["idx"]
        hand = state.get("hand", []) or []
        cname = "?"
        for card in hand:
            if card.get("index") == live_idx:
                cname = n(card.get("name", "?"))
                break
        return f"play [{live_idx}] {cname}{tgt_str}"
    return str(tok)


def _execute_single_token(state, tok, send, stable_hand):
    """Execute one parsed token and return the new state.

    Mirrors the per-token branches of :func:`_handle_combat_input` but
    skips the user-prompts (callers already chose targets) and the
    sibling-token batching. For ``card`` tokens, ``idx`` is the live
    C# hand index — pass it straight to the backend rather than
    routing through :class:`StableHand`, whose stable slots are a
    separate Python-side namespace.
    """
    kind = tok["kind"]
    if kind == "end":
        return send({"cmd": "action", "action": "end_turn"})
    if kind == "potion":
        args: dict = {"potion_index": tok["idx"]}
        if tok.get("target") is not None:
            args["target_index"] = tok["target"]
        new_state = send({"cmd": "action", "action": "use_potion", "args": args})
        if new_state and new_state.get("decision") == "combat_play":
            stable_hand.update(new_state)
        return new_state
    if kind == "card":
        live_idx = tok["idx"]
        args = {"card_index": live_idx}
        if tok.get("target") is not None:
            args["target_index"] = tok["target"]
        new_state = send({"cmd": "action", "action": "play_card", "args": args})
        if new_state and new_state.get("decision") == "combat_play":
            stable_hand.update(new_state)
        return new_state
    return state


def _parse_batch_token(tok):
    """Parse one batch token. Returns dict {kind, idx, target} or None on parse failure.

    Tokens:
      'e'                 → end turn
      'N'                 → play stable card N
      'NtM'               → play stable card N targeting enemy M
      'pN'                → use potion N
      'pNtM'              → use potion N targeting enemy M
    """
    tok = tok.strip().lower()
    if not tok:
        return None
    if tok == "e":
        return {"kind": "end", "idx": None, "target": None}
    if tok in ("go", "g"):
        return {"kind": "go", "idx": None, "target": None}
    if tok in ("a", "auto"):
        return {"kind": "auto", "idx": None, "target": None}
    is_potion = tok.startswith("p")
    body = tok[1:] if is_potion else tok
    if "t" in body:
        idx_part, _, tgt_part = body.partition("t")
    else:
        idx_part, tgt_part = body, ""
    if not idx_part.isdigit():
        return None
    target = None
    if tgt_part:
        if not tgt_part.isdigit():
            return None
        target = int(tgt_part)
    return {
        "kind": "potion" if is_potion else "card",
        "idx": int(idx_part),
        "target": target,
    }


def _show_pile(pile, label, state=None):
    """Print a pile (list of cards) with name, cost, and rough description."""
    if not pile:
        print(f"  {c(label, 'bold')}: empty")
        return
    from collections import Counter

    counts = Counter()
    for card in pile:
        counts[n(card.get("name", "?"))] += 1
    print(f"  {c(label, 'bold')} ({len(pile)}):")
    for name, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        suffix = f" x{cnt}" if cnt > 1 else ""
        print(f"    {name}{suffix}")


def _handle_combat_input(state, hand, enemies, energy, auto, send, stable_hand):
    """Drive one user turn-step in combat. Returns the new state.

    Supports batch syntax: multiple space-separated tokens executed in order.
    Each token can be a stable card index `N`, target form `NtM`, potion `pN`,
    potion-with-target `pNtM`, or `e` to end turn (must be the last token).
    """
    # Build a lookup from stable_idx → card so the user-facing display matches
    # what we resolve later. We deliberately don't filter by playability here;
    # validation happens token-by-token so we can give precise errors.
    stable_by_idx = {}
    for card in hand:
        stable_by_idx[card.get("stable_idx", card.get("index"))] = card

    if auto:
        playable = [c for c in hand if c.get("can_play") and c.get("cost", 99) <= energy]
        if playable:
            card = playable[0]
            tokens = [
                {"kind": "card", "idx": card.get("stable_idx", card["index"]), "target": None}
            ]
        else:
            tokens = [{"kind": "end", "idx": None, "target": None}]
    else:
        # Custom prompt: accept any string, parse ourselves.
        prompt_label = (
            "Play card [idx], (e)nd turn, (p0) potion, (g)o MCTS best, (a)uto play turn. "
            "Batch: '0t1 1t0 e'"
        )
        raw = get_input(prompt_label, valid_options=None, state=state)
        # Special bare commands handled in get_input (help, deck, draw, etc.).
        # If we got here, raw is a real action string.
        toks = raw.split()
        tokens = []
        for tok in toks:
            parsed = _parse_batch_token(tok)
            if parsed is None:
                print(f"  {c('Invalid token:', 'red')} {tok}")
                return state
            tokens.append(parsed)
        if not tokens:
            return state

    # `go` / `auto`: replace tokens with MCTS-recommended action(s).
    # `go` runs one best move; `auto` keeps running best moves until the
    # turn ends or the state leaves combat_play. Both ignore any sibling
    # tokens in the input — the user-facing behavior is "the rest is on
    # the engine."
    if tokens and tokens[0]["kind"] in ("go", "auto"):
        loop = tokens[0]["kind"] == "auto"
        while True:
            best_tok = _best_action_token(state, stable_hand)
            if best_tok is None:
                err = _LAST_HINT_ERROR
                msg = "No MCTS hint available."
                if err:
                    msg = f"{msg} ({err})"
                print(f"  {c(msg, 'dim')}")
                return state
            label = _format_action_token(best_tok, state)
            print(f"  {c('MCTS plays:', 'green')} {label}")
            state = _execute_single_token(state, best_tok, send, stable_hand)
            if not state or state.get("decision") != "combat_play":
                return state
            if not loop:
                return state
            if best_tok["kind"] == "end":
                return state

    # Execute tokens sequentially. State refreshes between actions; the stable
    # hand is reconciled so subsequent tokens see correct mappings.
    for ti, tok in enumerate(tokens):
        if tok["kind"] == "end":
            if ti != len(tokens) - 1:
                print(f"  {c('end-turn must be the last token', 'red')}")
                return state
            old_hand_names = [n(cd.get("name", "?")) for cd in state.get("hand", [])]
            old_discard = state.get("discard_pile_count", 0)
            state = send({"cmd": "action", "action": "end_turn"})
            if state and state.get("decision") == "combat_play":
                stable_hand.update(state)
                status_cards = [
                    n(cd.get("name", "?"))
                    for cd in state.get("hand", [])
                    if cd.get("type") in ("Status", "Curse")
                ]
                if status_cards:
                    sc_str = ", ".join(f"{c(name, 'red')}" for name in status_cards)
                    print(f"  ⚠ Status cards in hand:: {sc_str}")
            return state

        if tok["kind"] == "potion":
            args = {"potion_index": tok["idx"]}
            if enemies and tok["target"] is not None:
                args["target_index"] = tok["target"]
            elif enemies and len(tokens) == 1 and not auto:
                # Single-token interactive use: prompt for target as before.
                tgt = get_input(
                    "Target enemy [index] or self (s)",
                    state=state,
                )
                if tgt != "s" and tgt.isdigit():
                    args["target_index"] = int(tgt)
            state = send({"cmd": "action", "action": "use_potion", "args": args})
            if not state or state.get("decision") != "combat_play":
                return state
            # Refresh hand/enemies/energy + stable mapping for subsequent tokens.
            stable_hand.update(state)
            hand = state.get("hand", [])
            enemies = state.get("enemies", [])
            energy = state.get("energy", 0)
            continue

        # tok["kind"] == "card"
        live_idx = stable_hand.stable_to_live(tok["idx"])
        if live_idx is None:
            print(f"  {c('Stable card index unavailable:', 'red')} {tok['idx']}")
            return state
        # Sanity: ensure the live card exists and matches our snapshot.
        if live_idx >= len(hand):
            print(f"  {c('Card index out of range:', 'red')} {tok['idx']}")
            return state
        card = hand[live_idx]
        # Cost / playability check
        if not card.get("can_play"):
            print(f"  {c('Card cannot be played:', 'red')} {n(card.get('name', '?'))}")
            return state
        if card.get("cost", 99) > energy:
            print(f"  {c('Not enough energy.', 'red')}")
            return state
        args = {"card_index": live_idx}
        if card.get("target_type") == "AnyEnemy":
            if tok["target"] is not None:
                args["target_index"] = tok["target"]
            elif len(enemies) == 1:
                args["target_index"] = enemies[0]["index"]
            elif auto:
                args["target_index"] = min(enemies, key=lambda e: e.get("hp", 999))["index"]
            else:
                tgt = get_input(
                    "Target enemy [index]",
                    {str(e["index"]) for e in enemies},
                )
                args["target_index"] = int(tgt)
        state = send({"cmd": "action", "action": "play_card", "args": args})
        if not state or state.get("decision") != "combat_play":
            return state
        # Pre-mark the played slot as consumed so reconcile picks the right
        # specific slot when multiple share a card_id (e.g., 3 Strikes).
        stable_hand.mark_consumed(tok["idx"])
        stable_hand.update(state)
        hand = state.get("hand", [])
        enemies = state.get("enemies", [])
        energy = state.get("energy", 0)
    return state


class _QuitRequested(Exception):
    pass


def _quit_with_save(native_save_path, character, seed):
    """Return the save path to use on quit, or None to quit without saving."""
    print()
    try:
        ans = input("  Save before quitting? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"

    if ans not in ("y", "yes"):
        print("  Quitting without saving.")
        return None

    if native_save_path:
        return native_save_path

    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    char_tag = (character or "run").lower()
    seed_tag = seed or "random"
    return os.path.join(SAVE_DIR, f"{char_tag}_{seed_tag}_{ts}.save")


def _show_quit_save_result(result):
    """Print save confirmation or failure from a quit_result response."""
    save_result = result.get("save") if result else None
    if save_result and save_result.get("success"):
        sz = save_result.get("size", 0)
        save_path = save_result.get("path")
        print(f"  {c('Saved!', 'green')} ({sz // 1024}KB)")
        if save_path:
            print(f"  Save path: {c(save_path, 'cyan')}")
            print(f"  Continue later: python3 play.py --continue {save_path}")
    elif save_result:
        print(f"  {c('Save failed:', 'red')} {save_result.get('message', '?')}")


def _writeback_continue_save(send_fn, native_save_path):
    """Best-effort writeback for --continue sessions when a stable map checkpoint is reached."""
    if not native_save_path:
        return
    result = send_fn({"cmd": "write_continue_save", "path": native_save_path})
    if result and result.get("success"):
        sz = result.get("size", 0)
        print(f"  {c(f'Save written ({sz // 1024}KB)', 'dim')}")
    elif result:
        print(f"  {c('Save failed:', 'red')} {result.get('message', '?')}")


def play(
    character="Ironclad",
    seed=None,
    auto=False,
    ascension=0,
    log=True,
    load_path=None,
    native_save_path=None,
    hint_cfg=None,
    hint_mode=None,
    hint_timeout_ms=None,
):
    """Play one run; ``hint_cfg``/``hint_mode``/``hint_timeout_ms``
    install per-run hint settings into module globals so the REPL
    verb (``/hint``) and the inline render path see the same config.
    """
    global _HINT_CFG, _HINT_MODE, _HINT_TIMEOUT_MS
    if hint_cfg is not None:
        _HINT_CFG = hint_cfg
    if hint_mode is not None:
        _HINT_MODE = hint_mode
    if hint_timeout_ms is not None:
        _HINT_TIMEOUT_MS = hint_timeout_ms
    actual_seed = seed or f"cli_{random.randint(1000, 9999)}"
    replay_actions = None
    restart_requested = False
    quit_sent = False

    if load_path:
        character, actual_seed, replay_actions = _load_game(load_path)
        print(f"\n{c('Loading save...', 'yellow')} {os.path.basename(load_path)}")
        print(f"  Character: {character}  Seed: {actual_seed}  Actions: {len(replay_actions)}")

    logger = GameLogger(character, actual_seed, enabled=log, ascension=ascension)
    action_log = []
    proc = subprocess.Popen(
        [
            DOTNET,
            "run",
            "--no-build",
            "--project",
            PROJECT,
            f"--property:GameVersion={sts2_client.GAME_VERSION}",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def read():
        while True:
            l = proc.stdout.readline().strip()
            if not l:
                return None
            if l.startswith("{"):
                resp = json.loads(l)
                # On a map decision, capture the full map graph (a non-advancing
                # query) so replay tools can render the connected map.
                if resp.get("decision") == "map_select" and "full_map" not in resp:
                    try:
                        proc.stdin.write(json.dumps({"cmd": "get_map"}) + "\n")
                        proc.stdin.flush()
                        while True:
                            ml = proc.stdout.readline().strip()
                            if not ml:
                                break
                            if ml.startswith("{"):
                                resp["full_map"] = json.loads(ml)
                                break
                    except (OSError, ValueError):
                        pass
                logger.log_state(resp)
                return resp

    autosave_path = os.path.join(SAVE_DIR, f"autosave_{character}_{actual_seed}.json")

    def send(cmd, record=True):
        logger.log_action(cmd)
        if record and cmd.get("cmd") == "action":
            action_log.append(cmd)
            try:
                _save_game(autosave_path, character, actual_seed, action_log)
            except OSError as e:
                print(f"  {c('Autosave failed:', 'yellow')} {e}", file=sys.stderr)
        proc.stdin.write(json.dumps(cmd) + "\n")
        proc.stdin.flush()
        return read()

    # Wire send into get_input for map command
    get_input._send = send

    def do_save():
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{character}_{actual_seed}_{ts}.json"
        save_path = os.path.join(SAVE_DIR, fname)
        _save_game(save_path, character, actual_seed, action_log)
        print(f"  {c('Saved!', 'green')} {fname} ({len(action_log)} actions)")
        print(f"  Load with: python3 play.py --load {os.path.relpath(save_path, ROOT)}")

    get_input._save_fn = do_save

    stable_hand = StableHand()

    def refresh_state(s):
        """Apply stable-index annotations to a fresh combat state."""
        if s:
            stable_hand.update(s)
        return s

    try:
        ready = read()
        if not ready:
            print("Failed to start simulator")
            return

        if native_save_path:
            print("  Loading game save...")
            state = send({"cmd": "load_save", "path": native_save_path}, record=False)
            if state and state.get("type") == "error":
                print(f"  {c('Error:', 'red')} {state.get('message', '?')}")
                return
            p = state.get("player", {}) if state else {}
            char_name = p.get("name", {})
            if isinstance(char_name, dict):
                character = char_name.get("en", character)
            print(f"  {c('Save loaded!', 'green')}")
        else:
            state = send(
                {
                    "cmd": "start_run",
                    "character": character,
                    "seed": actual_seed,
                    "ascension": ascension,
                    "lang": "en",
                },
                record=False,
            )

            # Replay saved actions silently
            if replay_actions:
                total = len(replay_actions)
                for i, cmd in enumerate(replay_actions):
                    state = send(cmd, record=True)
                    pct = (i + 1) * 100 // total
                    print(
                        f"\r  Replaying... {pct}% ({i + 1}/{total})",
                        end="",
                        flush=True,
                    )
                    if not state:
                        print(f"\n{c('Replay failed at action', 'red')} {i + 1}")
                        return
                print(f"\r  {c('Replay complete!', 'green')}" + " " * 30)
                print()
        print(f"\n{c('Slay the Spire 2 — Headless CLI', 'bold')}")
        if native_save_path:
            p = state.get("player", {}) if state else {}
            ctx = state.get("context", {}) if state else {}
            print(
                f"Character: {n(p.get('name', '?'))}  "
                f"Act: {ctx.get('act', '?')} ({n(ctx.get('act_name', '?'))})  "
                f"HP: {p.get('hp', '?')}/{p.get('max_hp', '?')}  "
                f"Gold: {p.get('gold', '?')}"
            )
        else:
            asc_str = f"  Ascension: {ascension}" if ascension > 0 else ""
            print(f"Character: {character}  Seed: {actual_seed}{asc_str}")
        print(f"Type {c('help', 'cyan')} for available commands.\n")

        while True:
            if not state:
                print("Connection lost.")
                break

            if state.get("type") == "error":
                print(f"  {c('Error:', 'red')} {state.get('message', '?')}")
                state = send({"cmd": "action", "action": "proceed"})
                continue

            dec = state.get("decision", "")

            if dec == "game_over":
                victory = state.get("victory", False)
                p = state.get("player", {})
                ctx = state.get("context", {})

                print(f"\n{'═' * 60}")
                if victory:
                    print(
                        f"  {c('★ ★ ★', 'yellow')}  {c('VICTORY!', 'green')}  {c('★ ★ ★', 'yellow')}"
                    )
                else:
                    print(f"  {c('DEFEAT', 'red')}")
                print()

                act_name = n(ctx.get("act_name", "?"))
                floor = state.get("floor", "?")
                print(f"  Act: {state.get('act', '?')} ({act_name})  Floor: {floor}")
                print(f"  Character: {n(p.get('name', '?'))}")
                print(
                    f"  HP: {p.get('hp', '?')}/{p.get('max_hp', '?')}  Gold: {p.get('gold', '?')}"
                )

                deck = p.get("deck", [])
                if deck:
                    print(f"  Deck: {len(deck)} cards")

                relics = p.get("relics", [])
                if relics:
                    relic_names = [n(r.get("name", "?")) for r in relics]
                    print(f"  Relics ({len(relics)}): {', '.join(relic_names)}")

                print(f"{'═' * 60}")

                if auto:
                    break

                print(f"\n  {c('q', 'cyan')} Quit    {c('n', 'cyan')} New run")
                choice = input("  > ").strip().lower()
                if choice == "n":
                    restart_requested = True
                    break
                break

            elif dec == "map_select":
                _writeback_continue_save(send, native_save_path)
                show_map(state, send_fn=send)
                choices = state.get("choices", [])

                if auto:
                    if len(choices) == 1:
                        pick = choices[0]
                    else:
                        p = state.get("player", {})
                        hp_ratio = p.get("hp", 1) / max(p.get("max_hp", 1), 1)
                        if hp_ratio < 0.4:
                            pick = next(
                                (ch for ch in choices if ch["type"] == "RestSite"), choices[0]
                            )
                        else:
                            pick = choices[0]
                else:
                    valid = {str(i): ch for i, ch in enumerate(choices)}
                    key = get_input("Choose path [number]", set(valid.keys()), state=state)
                    pick = valid[key]

                state = send(
                    {
                        "cmd": "action",
                        "action": "select_map_node",
                        "args": {"col": pick["col"], "row": pick["row"]},
                    }
                )

            elif dec == "combat_play":
                stable_hand.update(state)
                show_combat(state)
                hand = state.get("hand", [])
                enemies = state.get("enemies", [])
                energy = state.get("energy", 0)
                state = _handle_combat_input(state, hand, enemies, energy, auto, send, stable_hand)

            elif dec == "card_reward":
                show_card_reward(state)
                cards = state.get("cards", [])
                valid = {str(c["index"]): c for c in cards}
                valid["s"] = None  # skip

                if auto:
                    choice = "0" if cards else "s"
                else:
                    choice = get_input(
                        "Pick card [index] or (s)kip",
                        set(valid.keys()),
                        state=state,
                    )

                if choice == "s":
                    state = send({"cmd": "action", "action": "skip_card_reward"})
                else:
                    state = send(
                        {
                            "cmd": "action",
                            "action": "select_card_reward",
                            "args": {"card_index": int(choice)},
                        }
                    )

            elif dec == "bundle_select":
                print(f"\n{'─' * 60}")
                ctx = state.get("context", {})
                if ctx:
                    print(
                        f"  {c(n(ctx.get('act_name', '?')), 'dim')} Floor {ctx.get('floor', '?')}"
                    )
                print(f"  {c('Choose a card pack', 'bold')}")
                show_player(state.get("player", {}))
                print()
                bundles = state.get("bundles", [])
                for b in bundles:
                    bidx = b["index"]
                    print(f"  {c(f'Pack [{bidx}]:', 'yellow')}")
                    for cd in b.get("cards", []):
                        cd_desc = card_desc(cd)
                        print(
                            f"    {n(cd['name'])} ({cd.get('cost', '?')}) {c(cd.get('type', ''), 'dim')}"
                        )
                        if cd_desc:
                            print(f"      {c(cd_desc, 'dim')}")
                valid = {str(b["index"]): b for b in bundles}
                if auto:
                    choice = "0"
                else:
                    choice = get_input(
                        "Choose pack [index]",
                        set(valid.keys()),
                        state=state,
                    )
                state = send(
                    {
                        "cmd": "action",
                        "action": "select_bundle",
                        "args": {"bundle_index": int(choice)},
                    }
                )

            elif dec == "card_select":
                print(f"\n{'─' * 60}")
                ctx = state.get("context", {})
                if ctx:
                    print(
                        f"  {c(n(ctx.get('act_name', '?')), 'dim')} Floor {ctx.get('floor', '?')}"
                    )
                min_sel = state.get("min_select", 1)
                max_sel = state.get("max_select", 1)
                card_count = len(state.get("cards", []) or [])
                # Render the selection range. "Any number" prompts (e.g.
                # Ashwater's Exhaust-N-cards) come through with a sentinel
                # max_sel like 999_999_999; collapse those to "any".
                if max_sel >= max(card_count, 100):
                    if min_sel == 0:
                        sel_label = "select any"
                    else:
                        sel_label = f"select at least {min_sel}"
                elif min_sel == max_sel:
                    sel_label = f"select {min_sel}"
                else:
                    sel_label = f"select {min_sel}-{max_sel}"
                print(f"  {c('Choose cards', 'bold')} ({sel_label})")
                show_player(state.get("player", {}))
                print()
                cards = state.get("cards", [])
                for cd in cards:
                    up = c("+", "green") if cd.get("upgraded") else ""
                    stats = cd.get("stats") or {}
                    ctype_label = cd.get("type", "")
                    cd_desc_text = card_desc(cd)
                    print(
                        f"  [{cd['index']}] {n(cd['name'])}{up} ({cd.get('cost', '?')}) {c(ctype_label, 'dim')}"
                    )
                    if cd_desc_text:
                        print(f"      {c(cd_desc_text, 'dim')}")
                    aug_parts = _format_upgrade_preview(
                        stats, cd.get("after_upgrade"), cd.get("cost")
                    )
                    if aug_parts:
                        print(f"      {c('upgrade:', 'green')} {', '.join(aug_parts)}")

                valid = {str(cd["index"]): cd for cd in cards}
                if min_sel == 0:
                    valid["s"] = None

                # Save state before selection to show diff
                old_deck_cards = [
                    n(cd.get("name", "?")) for cd in state.get("player", {}).get("deck", [])
                ]

                if auto:
                    choice = "0"
                else:
                    choice = get_input(
                        "Choose card(s) [index] or (s)kip",
                        set(valid.keys()),
                        state=state,
                    )

                if choice == "s":
                    state = send({"cmd": "action", "action": "skip_select"})
                else:
                    # Support comma-separated indices
                    state = send(
                        {"cmd": "action", "action": "select_cards", "args": {"indices": choice}}
                    )

                # Show what changed
                if state and state.get("player"):
                    from collections import Counter

                    new_deck_cards = [
                        n(cd.get("name", "?")) for cd in state["player"].get("deck", [])
                    ]
                    added = Counter(new_deck_cards) - Counter(old_deck_cards)
                    removed = Counter(old_deck_cards) - Counter(new_deck_cards)
                    if added or removed:
                        parts = []
                        for card_name, cnt in removed.items():
                            parts.append(c(f"-{card_name}" + (f"x{cnt}" if cnt > 1 else ""), "red"))
                        for card_name, cnt in added.items():
                            parts.append(
                                c(f"+{card_name}" + (f"x{cnt}" if cnt > 1 else ""), "green")
                            )
                        print(f"\n  {c('Changes:', 'yellow')} Deck: {' '.join(parts)}")

            elif dec == "shop":
                show_shop(state)

                if auto:
                    choice = "leave"
                else:
                    choice = get_input(
                        "Buy [index/r0/p0/rm] or (leave)",
                        state=state,
                    )

                if choice == "leave":
                    state = send({"cmd": "action", "action": "leave_room"})
                elif choice == "rm":
                    state = send({"cmd": "action", "action": "remove_card"})
                elif choice.startswith("r"):
                    state = send(
                        {
                            "cmd": "action",
                            "action": "buy_relic",
                            "args": {"relic_index": int(choice[1:])},
                        }
                    )
                elif choice.startswith("p"):
                    state = send(
                        {
                            "cmd": "action",
                            "action": "buy_potion",
                            "args": {"potion_index": int(choice[1:])},
                        }
                    )
                else:
                    state = send(
                        {"cmd": "action", "action": "buy_card", "args": {"card_index": int(choice)}}
                    )

            elif dec == "rest_site":
                show_rest_site(state)
                options = state.get("options", [])
                enabled = [o for o in options if o.get("is_enabled")]
                valid = {str(o["index"]): o for o in enabled}

                if auto:
                    hp = state.get("player", {}).get("hp", 1)
                    mhp = state.get("player", {}).get("max_hp", 1)
                    heal = next((o for o in enabled if o.get("option_id") == "HEAL"), None)
                    smith = next((o for o in enabled if o.get("option_id") == "SMITH"), None)
                    pick = (heal if hp < mhp * 0.7 else smith) or (
                        heal or (enabled[0] if enabled else None)
                    )
                    choice = str(pick["index"]) if pick else "0"
                else:
                    choice = get_input("Choose option [index]", set(valid.keys()), state=state)

                state = send(
                    {
                        "cmd": "action",
                        "action": "choose_option",
                        "args": {"option_index": int(choice)},
                    }
                )
                if state and state.get("type") == "error":
                    state = send({"cmd": "action", "action": "leave_room"})

            elif dec == "event_choice":
                show_event(state)
                options = state.get("options", [])
                unlocked = [o for o in options if not o.get("is_locked")]
                valid = {str(o["index"]): o for o in unlocked}
                valid["leave"] = None

                # Save state before choice to show diff
                old_relics = set(
                    n(r.get("name", "?")) for r in state.get("player", {}).get("relics", [])
                )
                old_deck_cards = [
                    n(cd.get("name", "?")) for cd in state.get("player", {}).get("deck", [])
                ]
                old_deck = state.get("player", {}).get("deck_size", 0)
                old_hp = state.get("player", {}).get("hp", 0)
                old_max_hp = state.get("player", {}).get("max_hp", 0)
                old_gold = state.get("player", {}).get("gold", 0)

                if auto:
                    choice = str(unlocked[0]["index"]) if unlocked else "leave"
                else:
                    choice = get_input(
                        "Choose option [index] or (leave)",
                        set(valid.keys()),
                        state=state,
                    )

                if choice == "leave":
                    state = send({"cmd": "action", "action": "leave_room"})
                else:
                    state = send(
                        {
                            "cmd": "action",
                            "action": "choose_option",
                            "args": {"option_index": int(choice)},
                        }
                    )
                    if state and state.get("type") == "error":
                        state = send({"cmd": "action", "action": "leave_room"})

                # Show what changed
                if state and state.get("player"):
                    new_p = state["player"]
                    new_relics = set(n(r.get("name", "?")) for r in new_p.get("relics", []))
                    gained_relics = new_relics - old_relics
                    new_deck_cards = [n(cd.get("name", "?")) for cd in new_p.get("deck", [])]
                    new_deck = new_p.get("deck_size", 0)
                    new_hp = new_p.get("hp", 0)
                    new_max_hp = new_p.get("max_hp", 0)
                    new_gold = new_p.get("gold", 0)
                    changes = []
                    if gained_relics:
                        changes.append(f"Relic: {', '.join(gained_relics)}")
                    # Show specific card changes
                    from collections import Counter

                    old_counts = Counter(old_deck_cards)
                    new_counts = Counter(new_deck_cards)
                    added = new_counts - old_counts
                    removed = old_counts - new_counts
                    if added or removed:
                        parts = []
                        for card_name, cnt in removed.items():
                            parts.append(c(f"-{card_name}" + (f"x{cnt}" if cnt > 1 else ""), "red"))
                        for card_name, cnt in added.items():
                            parts.append(
                                c(f"+{card_name}" + (f"x{cnt}" if cnt > 1 else ""), "green")
                            )
                        changes.append(f"Deck: {' '.join(parts)}")
                    elif new_deck != old_deck:
                        changes.append(f"Deck: {old_deck} → {new_deck}")
                    if new_hp != old_hp or new_max_hp != old_max_hp:
                        changes.append(f"HP: {old_hp}/{old_max_hp} → {new_hp}/{new_max_hp}")
                    if new_gold != old_gold:
                        diff = new_gold - old_gold
                        changes.append(f"Gold: {'+' if diff > 0 else ''}{diff}")
                    if changes:
                        print(f"\n  {c('Changes:', 'yellow')} {'; '.join(changes)}")

            else:
                print(f"  Unknown state: {dec}")
                state = send({"cmd": "action", "action": "proceed"})

    except _QuitRequested:
        quit_sent = False
        quit_save_path = _quit_with_save(native_save_path, character, actual_seed)
        # Retry loop: if save fails the process stays alive so we can try a different path.
        quit_sent = True
        while True:
            quit_cmd = {"cmd": "quit"}
            if quit_save_path:
                quit_cmd["path"] = quit_save_path
            result = send(quit_cmd)
            if result and result.get("type") == "save_error":
                save_detail = result.get("save") or {}
                msg = save_detail.get("message", "?")
                print(f"  {c('Save failed:', 'red')} {msg}")
                try:
                    ans = input("  New path (Enter = quit without saving): ").strip()
                except (EOFError, KeyboardInterrupt):
                    ans = ""
                quit_save_path = ans if ans else None
            else:
                _show_quit_save_result(result)
                break
    except KeyboardInterrupt:
        print(f"\n  {c('Run abandoned.', 'yellow')}")
    finally:
        if not quit_sent:
            try:
                result = send({"cmd": "quit"})
                _show_quit_save_result(result)
            except Exception:
                pass
        logger.close()
        if logger.path:
            print(f"\n  [log] Game log saved to {logger.path}")
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    return restart_requested


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Slay the Spire 2 in your terminal")
    parser.add_argument("--auto", action="store_true", help="Auto-play with simple AI")
    parser.add_argument("--seed", type=str, default=None, help="Random seed")
    parser.add_argument(
        "--character",
        type=str,
        default="Ironclad",
        choices=["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"],
        help="Character to play",
    )
    parser.add_argument(
        "--ascension",
        type=int,
        default=0,
        choices=range(0, 11),
        metavar="0-10",
        help="Ascension level (0-10)",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable game logging")
    parser.add_argument("--load", type=str, default=None, help="Load a save file (action replay)")
    parser.add_argument("--saves", action="store_true", help="List available saves and exit")
    parser.add_argument(
        "--save-info", type=str, default=None, help="Show info from a save file (provide path)"
    )
    parser.add_argument(
        "--continue",
        dest="continue_save",
        type=str,
        default=None,
        help="Continue playing from a save file (provide path)",
    )

    # ── MCTS hint flags (bead sts2-rs-vjd1, hint-T10) ──
    hint_group = parser.add_argument_group("MCTS hints")
    hint_group.add_argument(
        "--mcts-sims", type=int, default=300, help="MCTS simulation budget (default 300)"
    )
    hint_group.add_argument(
        "--mcts-depth", type=int, default=10, help="MCTS rollout depth (default 10)"
    )
    hint_group.add_argument(
        "--mcts-c-puct",
        type=float,
        default=1.4,
        help="PUCT exploration constant (default 1.4)",
    )
    hint_group.add_argument(
        "--mcts-seed",
        type=int,
        default=None,
        help="MCTS RNG seed (default: random per call)",
    )
    hint_group.add_argument(
        "--mcts-threads",
        type=int,
        default=1,
        help="MCTS worker threads (default 1). >1 parallelizes via "
        "sts2_rs.mcts_recommend's GIL-released path.",
    )
    hint_group.add_argument(
        "--mcts-fair",
        action="store_true",
        help="Use fair per-action MCTS (one root per action) instead of single-root visits.",
    )
    hint_group.add_argument(
        "--mcts-cheat-draw-order",
        action="store_true",
        help="Reveal the upcoming draw-pile order to MCTS rollouts.",
    )
    hint_group.add_argument(
        "--hint-timeout-ms",
        type=int,
        default=5000,
        help="Hard wall-clock cap (ms) per recommend() call (default 5000).",
    )
    hint_group.add_argument(
        "--mcts-rollout",
        choices=("heuristic", "uniform", "net"),
        default=None,
        help="MCTS rollout policy. Default: what the hidden-info mode deploys "
        "(the playout net under beliefdraw). 'heuristic' = softmax over expert "
        "play priorities; 'uniform' = unbiased random; 'net' = the playout net.",
    )
    hint_group.add_argument(
        "--mcts-log",
        type=str,
        default=None,
        help="Append one JSONL record per MCTS recommend() call to this "
        "path (state digest + per-action visits/wins/losses/p_win + PV).",
    )
    hint_mode_group = parser.add_mutually_exclusive_group()
    hint_mode_group.add_argument(
        "--no-hints",
        action="store_true",
        help="Disable MCTS hints entirely (no compute, no tags).",
    )
    hint_mode_group.add_argument(
        "--hint-on-demand",
        action="store_true",
        help="Don't render inline tags during combat; type '/hint' in the "
        "REPL to compute hints on demand.",
    )
    parser.add_argument(
        "--game-version",
        default=sts2_client.GAME_VERSION,
        choices=sts2_client.SUPPORTED_GAME_VERSIONS,
        help="game build the C# oracle simulates (default: $STS2_GAME_VERSION or "
        f"{sts2_client.DEFAULT_GAME_VERSION}); needs `just build-cs <version>`",
    )

    args = parser.parse_args()
    sts2_client.set_game_version(args.game_version)
    LIB_DIR = sts2_client.LIB_DIR

    if args.save_info is not None:
        p = args.save_info
        if not pathlib.Path(p).is_absolute():
            p = os.path.join(ROOT, p)
        if not os.path.isfile(p):
            print(f"Save file not found: {p}")
            sys.exit(1)
        show_native_save(p)
        sys.exit(0)

    if args.saves:
        saves = _list_saves()
        if saves:
            print(f"\n{'─' * 50}")
            for s in saves:
                print(f"  {s['file']}  {s['character']}  seed:{s['seed']}  actions:{s['actions']}")
            print(f"{'─' * 50}")
            print("  Load: python3 play.py --load saves/<file>")
        else:
            print("No saves found.")
        sys.exit(0)

    load_path = None
    if args.load:
        p = args.load
        if not pathlib.Path(p).is_absolute():
            p = os.path.join(ROOT, p)
        if not os.path.isfile(p):
            print(f"Save file not found: {args.load}")
            sys.exit(1)
        load_path = p

    native_save_path = None
    if args.continue_save is not None:
        native_save_path = args.continue_save
        if not pathlib.Path(native_save_path).is_absolute():
            native_save_path = os.path.join(ROOT, native_save_path)
        if not os.path.isfile(native_save_path):
            print(f"Save file not found: {native_save_path}")
            sys.exit(1)
        show_native_save(native_save_path)

    ensure_setup()

    # Build hint config from MCTS flags. When --no-hints is set we pass
    # ``hint_cfg=None`` AND ``hint_mode="off"`` — either alone would
    # disable; we set both so the REPL `/hint` verb also short-circuits.
    if _HINT_AVAILABLE and _HintRecommendConfig is not None:
        hint_cfg = _HintRecommendConfig(
            sims=args.mcts_sims,
            depth=args.mcts_depth,
            c_puct=args.mcts_c_puct,
            seed=args.mcts_seed if args.mcts_seed is not None else 0,
            threads=args.mcts_threads,
            fair=args.mcts_fair,
            cheat_draw_order=args.mcts_cheat_draw_order,
            timeout_ms=args.hint_timeout_ms,
            ascension=args.ascension,
            rollout_policy=args.mcts_rollout,
            log_path=args.mcts_log,
        )
    else:
        hint_cfg = None

    if args.no_hints:
        hint_mode = "off"
        hint_cfg = None
    elif args.hint_on_demand:
        hint_mode = "on_demand"
    else:
        hint_mode = "inline"

    if args.mcts_threads > 1:
        # sts2_rs.mcts_recommend releases the GIL (T4), so threads>1
        # should parallelize. Surface a warning anyway so users can
        # confirm against their own benches.
        print(
            f"  [hint] --mcts-threads={args.mcts_threads}: relies on "
            f"sts2_rs.mcts_recommend's GIL-released path. If you observe "
            f"no speedup, fall back to --mcts-threads=1.",
            file=sys.stderr,
        )

    next_seed = args.seed
    next_auto = args.auto
    next_load_path = load_path
    next_native_save_path = native_save_path
    while True:
        restart = play(
            character=args.character,
            seed=next_seed,
            auto=next_auto,
            ascension=args.ascension,
            log=not args.no_log,
            load_path=next_load_path,
            native_save_path=next_native_save_path,
            hint_cfg=hint_cfg,
            hint_mode=hint_mode,
            hint_timeout_ms=args.hint_timeout_ms,
        )
        if not restart:
            break
        next_seed = None
        next_auto = False
        next_load_path = None
        next_native_save_path = None
