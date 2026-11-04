#!/usr/bin/env python
"""Regenerate the champion regions the manifest cannot be read from directly.

``sts2-pycli/data/ppo/champions.toml`` is the source of truth, and Python reads
it at runtime. Two consumers cannot:

* ``sts2-rs/crates/sts2-runai/src/ppo/net.rs`` — the crate is built from
  standalone worktrees where no sibling ``sts2-pycli/`` exists (its own resolver
  already guards with ``sibling.exists()``), so a ``build.rs`` that
  ``include_str!``\\ 'd the manifest would make those worktrees unbuildable.
* ``./Dockerfile`` — ``COPY`` is Dockerfile syntax; it cannot read a file.

So both carry a GENERATED region between markers, rewritten from the manifest by
this script. ``--check`` byte-compares instead of writing, which is the gate:
it does not ask "did you remember to update the other lists", it asks "does this
match what the generator produces", and forgetting cannot pass that.

Usage:
    python scripts/sync_champions.py            # rewrite both regions
    python scripts/sync_champions.py --check    # fail if either is stale
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PYCLI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PYCLI / "python"))

import champions  # noqa: E402

_REPO = _PYCLI.parent
_NET_RS = _REPO / "sts2-rs" / "crates" / "sts2-runai" / "src" / "ppo" / "net.rs"
_DOCKERFILE = _REPO / "Dockerfile"

BEGIN = "GENERATED FROM sts2-pycli/data/ppo/champions.toml -- DO NOT EDIT"
END = "END GENERATED CHAMPIONS"

#: manifest key -> Rust `CharacterId` variant.
_RUST_VARIANT = {
    "ironclad": "Ironclad",
    "silent": "Silent",
    "defect": "Defect",
    "regent": "Regent",
    "necrobinder": "Necrobinder",
}


def _rust_region() -> str:
    lines = []
    for key in champions.characters():
        variant = _RUST_VARIANT.get(key)
        if variant is None:
            raise SystemExit(
                f"champions.toml declares {key!r}, which has no CharacterId variant "
                f"mapped in {Path(__file__).name}'s _RUST_VARIANT. Add it there."
            )
        champ = champions.all_champions()[key]
        # Bare literal, not `Some(...)`: every character STS2 ships has a
        # champion, so `champion_onnx_filename` is total.
        #
        # The tuple's second slot is the encoder feature spec (`features` in
        # the manifest). It rides with the filename rather than in a region of
        # its own because they are one fact -- which net, at which schema --
        # and two regions is how they would drift apart. Serving a champion at
        # the wrong schema is silent: a gated feature rewrites an EXISTING cat
        # slot, so the tensor shapes match either way (GH #545).
        lines.append(f'        CharacterId::{variant} => ("{champ.onnx}", "{champ.features}"),')
    return "\n".join(lines)


def _docker_region() -> str:
    lines = []
    for key in champions.characters():
        champ = champions.all_champions()[key]
        rel = f"sts2-pycli/data/ppo/{champ.onnx}"
        if champ.gate:
            lines.append(f"# {key}: {champ.gate}")
        lines.append(f"COPY {rel} ./data/ppo/{champ.onnx}")
    return "\n".join(lines)


def _splice(path: Path, body: str, comment: str) -> str:
    """Replace the marked region in *path* with *body*, keeping the markers."""
    text = path.read_text(encoding="utf-8")
    begin_marker = f"{comment} {BEGIN}"
    end_marker = f"{comment} {END}"
    try:
        start = text.index(begin_marker)
        stop = text.index(end_marker)
    except ValueError:
        raise SystemExit(
            f"{path} has no generated-champion region. Expected the lines "
            f"{begin_marker!r} ... {end_marker!r}."
        ) from None
    if stop < start:
        raise SystemExit(f"{path}: champion markers are out of order")
    head = text[: start + len(begin_marker)]
    tail = text[stop:]
    return f"{head}\n{body}\n{tail}"


#: The Rust edition `rustfmt` must be told about — it does not read
#: `Cargo.toml` when invoked on a bare file, and the workspace pins 2021
#: (`sts2-rs/Cargo.toml:22`).
_RUST_EDITION = "2021"

#: `rustfmt` reads `rustfmt.toml` by walking up from the INPUT FILE's
#: directory, and this call feeds it stdin — so with no `--config-path` it
#: searches from the cwd and finds the workspace config only by luck.
#: `cargo fmt`, the gate this normalisation exists to agree with, always reads
#: it. Point at it explicitly so the two cannot disagree.
_RUSTFMT_CONFIG = _REPO / "sts2-rs" / "rustfmt.toml"


def _rustfmt(text: str) -> str:
    """`text` as `rustfmt` would write it, or unchanged if rustfmt is absent.

    The generator emits one match arm per line, which was identical to
    rustfmt's own output for as long as every arm fit inside the line width.
    Ironclad's arm stopped fitting when GH #545 gave it a non-empty feature
    spec (`"reward_upgrade"`), and the two gates then disagreed FOREVER:
    `cargo fmt --check` demanded the arm wrapped across four lines, while this
    script's `--check` demanded the one-line form it generates. Committing
    either one failed the other. Normalising through rustfmt here means the
    comparison is against what will actually sit in the file, so the two gates
    cannot diverge again the next time an arm crosses the width.

    Absent rustfmt degrades to the raw text rather than failing: the sync is
    useful without it (this is a formatting normalisation, not a correctness
    one), and `cargo fmt --check` is the gate that would catch the difference.

    "Cannot diverge again" is only true while this call and `cargo fmt` read
    the SAME config, which is why `--config-path` is passed: without it
    rustfmt searches upward from the cwd (there is no input file here — the
    text arrives on stdin), so it agreed with the workspace only incidentally.
    `max_width` happens to equal rustfmt's default today, so nothing was
    visibly broken; the first `sts2-rs/rustfmt.toml` edit that changes a
    width-affecting option would have re-opened the exact deadlock this
    function was written to close.
    """
    try:
        done = subprocess.run(
            [
                "rustfmt",
                "--edition",
                _RUST_EDITION,
                "--config-path",
                str(_RUSTFMT_CONFIG),
                "--emit",
                "stdout",
                "--quiet",
            ],
            input=text,
            capture_output=True,
            # utf-8 explicitly: Windows' default cp1252 mangles net.rs's "—",
            # rustfmt rejects the stream, and the raw one-line arm came back.
            encoding="utf-8",
            check=False,
        )
    except OSError:
        return text
    return done.stdout if done.returncode == 0 and done.stdout else text


_TARGETS = [
    (_NET_RS, _rust_region, "        //", _rustfmt),
    (_DOCKERFILE, _docker_region, "#", None),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if a generated region is stale",
    )
    args = ap.parse_args(argv)

    stale = []
    for path, build, comment, normalise in _TARGETS:
        if path == _DOCKERFILE and not path.exists():
            # The public export ships no Dockerfile (the image is private).
            print(f"skipped {path.relative_to(_REPO)}: not in this checkout")
            continue
        want = _splice(path, build(), comment)
        if normalise is not None:
            want = normalise(want)
        if path.read_text(encoding="utf-8") == want:
            continue
        if args.check:
            stale.append(path)
        else:
            path.write_text(want, encoding="utf-8", newline="\n")
            print(f"updated {path.relative_to(_REPO)}")

    if stale:
        names = ", ".join(str(p.relative_to(_REPO)) for p in stale)
        print(
            f"STALE champion region(s): {names}\n"
            f"data/ppo/champions.toml has moved on. Run `just sync-champions` "
            f"and commit the result -- do NOT hand-edit inside the markers.",
            file=sys.stderr,
        )
        return 1
    if args.check:
        print("champion regions are in sync with champions.toml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
