#!/usr/bin/env python3
"""Put the art tree into the static build (``web/dist/assets``), as WebP.

The static site serves this tree itself (Cloudflare Pages; Firebase Hosting
before 2026-09-03, whose 360 MB/day transfer quota is why the tree was
shrunk — at PNG size it is ~300 MB, one fresh device per day). Every PNG is
therefore shipped as WebP (lossy q82 for painted art, lossless for the
pixel-crisp UI folders) **under its original ``.png`` path**: browsers decode
an image by its bytes, ``web/public/_headers`` labels ``/assets/*.png`` as
``image/webp``, and nothing that stamps art URLs (the wasm data bundle, the
Python resolver, the SPA) has to know. ~4 x smaller; visually identical at
tablet scale (compared side by side before this was adopted).

Conversion is cached under ``web/.art-cache`` keyed by the source's mtime +
size, so only changed files are re-encoded; the first run over ~3200 PNGs
takes a few minutes on all cores. Files land in dist as hard links when the
trees share a filesystem (instant, no extra space), copies otherwise.

Also writes ``web/dist/wb-art-list.json`` — every manifest path with its
shipped size — which the app's "Download for offline" button walks.

Stdlib only (plus the ``cwebp`` binary), like the rest of the deploy path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
DEFAULT_IMAGES = _REPO.parent / "sts2-wiki" / "site" / "public" / "images"
DEFAULT_AUDIO = _REPO.parent / "sts2-wiki" / "site" / "public" / "audio"
DEFAULT_MANIFEST = _REPO / "deploy" / "asset_manifest.json"
DEFAULT_DEST = _REPO / "web" / "dist" / "assets"
DEFAULT_CACHE = _REPO / "web" / ".art-cache"

#: Top-level folders encoded losslessly: icons, fonts and UI chrome, where a
#: hard edge or a 1-px line is the point. Everything else is painted art.
LOSSLESS_DIRS = frozenset(
    {"ui", "map_icons", "boss_icons", "intent_atlas", "sprite_fonts", "animations"}
)
LOSSY_QUALITY = 82
#: The one file the manifest excludes but the tree carries; shipped as-is.
EXTRA_FILES = ("creature_scales.json",)
#: What a Git LFS pointer file starts with. The wiki's art and audio trees
#: are LFS-tracked, and a checkout whose objects were never smudged (a fresh
#: worktree, a clone without git-lfs) holds ~130-byte pointer TEXT files
#: under the real names. Shipped verbatim, they deploy fine and fail only
#: in the browser: every sound is silent (``decodeAudioData`` rejects the
#: text) and every image is broken. That happened — the 2026-09-08 deploy
#: served pointers for the whole audio tree — so it is refused here.
LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def _is_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("rb") as fh:
            return fh.read(len(LFS_POINTER_MAGIC)) == LFS_POINTER_MAGIC
    except OSError:
        return False


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _source_of(rel: str, images: Path, audio: Path) -> Path:
    if rel == "audio" or rel.startswith("audio/"):
        return audio / rel[len("audio/") :]
    return images / rel


def _cache_is_fresh(cached: Path, src: Path) -> bool:
    try:
        c, s = cached.stat(), src.stat()
    except FileNotFoundError:
        return False
    return c.st_size > 0 and c.st_mtime >= s.st_mtime


def _encode(src: Path, cached: Path, lossless: bool) -> None:
    cached.parent.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(cached.suffix + ".tmp")
    args = ["cwebp", "-quiet", "-mt"]
    args += ["-lossless", "-z", "6"] if lossless else ["-q", str(LOSSY_QUALITY)]
    args += [str(src), "-o", str(tmp)]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    Path(tmp).replace(cached)


def convert_tree(
    files: list[str],
    *,
    images: Path,
    audio: Path,
    dest: Path,
    cache: Path,
    jobs: int,
) -> dict[str, int]:
    """Ship every manifest file into *dest*; return ``{rel: shipped bytes}``."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    jobs_png: list[tuple[str, Path, Path, bool]] = []
    shipped: dict[str, int] = {}
    encoded = 0
    pointers = [rel for rel in files if _is_lfs_pointer(_source_of(rel, images, audio))]
    if pointers:
        raise SystemExit(
            f"ERROR: {len(pointers)} art/audio files are Git LFS POINTERS, not content "
            f"(e.g. {pointers[:3]}) — the deploy would ship them as-is and every sound "
            "would be silent. Fetch the objects first: git -C ../sts2-wiki lfs pull"
        )
    for rel in files:
        src = _source_of(rel, images, audio)
        if not src.is_file():
            raise SystemExit(f"ERROR: manifest file missing from the art tree: {rel}")
        if src.suffix.lower() == ".png":
            cached = cache / (rel + ".webp")
            lossless = rel.split("/", 1)[0] in LOSSLESS_DIRS
            if not _cache_is_fresh(cached, src):
                jobs_png.append((rel, src, cached, lossless))
            shipped[rel] = -1  # filled after encoding
        else:
            _link_or_copy(src, dest / rel)
            shipped[rel] = src.stat().st_size
    if jobs_png:
        print(f">> art: encoding {len(jobs_png)} PNGs to WebP ({jobs} jobs)…", flush=True)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(lambda j: _encode(j[1], j[2], j[3]), jobs_png))
        encoded = len(jobs_png)
    for rel, size in list(shipped.items()):
        if size == -1:
            cached = cache / (rel + ".webp")
            _link_or_copy(cached, dest / rel)
            shipped[rel] = cached.stat().st_size
    for extra in EXTRA_FILES:
        src = images / extra
        if src.is_file():
            _link_or_copy(src, dest / extra)
    print(f">> art: {encoded} newly encoded, {len(files) - encoded} from cache/as-is")
    return shipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args(argv)
    if shutil.which("cwebp") is None:
        print("ERROR: cwebp not found (install libwebp) — the static build ships the art as WebP")
        return 1
    if not a.images.is_dir():
        print(f"ERROR: art tree not found at {a.images} (the sts2-wiki sibling checkout)")
        return 1
    files: list[str] = json.loads(a.manifest.read_text())
    shipped = convert_tree(
        files, images=a.images, audio=a.audio, dest=a.dest, cache=a.cache, jobs=a.jobs
    )
    missing = [f for f in files if not (a.dest / f).is_file()]
    if missing:
        print(f"ERROR: {len(missing)} manifest files missing from {a.dest}, e.g. {missing[:3]}")
        return 1
    total = sum(shipped.values())
    png_src = sum(
        _source_of(f, a.images, a.audio).stat().st_size for f in files if f.endswith(".png")
    )
    png_out = sum(s for f, s in shipped.items() if f.endswith(".png"))
    listing = {"files": [{"path": f, "bytes": shipped[f]} for f in files], "bytes": total}
    (a.dest.parent / "wb-art-list.json").write_text(json.dumps(listing, separators=(",", ":")))
    print(
        f">> art: {len(files)} files, {total / 1e6:.0f} MB in {a.dest} "
        f"(PNG {png_src / 1e6:.0f} MB -> WebP {png_out / 1e6:.0f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
