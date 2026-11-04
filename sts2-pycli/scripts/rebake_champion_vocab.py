#!/usr/bin/env python3
"""Re-export the deployed champions onto a widened PPO categorical vocabulary.

WHY THIS EXISTS
---------------
The PPO registries are derived from the engine content on purpose, so the vocab
cannot drift from what a run can produce. The cost is that adding content moves
embedding rows, and a champion's ``emb_*`` table is baked at the width it was
trained on. Porting Underdocks as the second Act 1 grew the boss vocab from 9
ids to 12 (GH #544), so every champion's ``[11, dim]`` boss table has to become
``[14, dim]`` before it will load at all.

WHY THIS IS SAFE (and why it is not a retrain)
----------------------------------------------
The vocab order is pinned append-only (``sts2-vocab/src/vocab_pins/``), so
the nine ids these nets were trained on keep rows 0..8. The re-bake is a pure
row re-lay, done by ``ppo.model._widen_padded_embedding``:

    row i (i < old_n_ids)  <- unchanged
    the new ids            <- the OLD OOV row (an id the net never saw IS OOV
                              to it, which is the honest answer)
    OOV, PAD               <- moved to the new tail

On any Overgrowth/Hive/Glory decision the widened net therefore reads exactly
the vectors it learned, and this script CHECKS that rather than asserting it:
it runs the pre- and post-re-bake graphs on the same inputs and requires
bit-identical logits and value.

It does NOT make the nets good at Underdocks. Their three new boss rows are the
OOV row, so a champion walking an Underdocks act sees "some boss I don't know",
exactly as it saw every boss before ``emb_boss`` existed. Fixing that needs a
retrain (GH #545), not a re-bake.

USAGE
-----
    uv run python scripts/rebake_champion_vocab.py            # re-bake + verify
    uv run python scripts/rebake_champion_vocab.py --check    # verify only

``--check`` re-exports to a temp file and compares, leaving the tree untouched;
use it to confirm the committed champions already match the current vocab.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

PPO_DIR = ROOT / "data" / "ppo"
CHAMPIONS_TOML = PPO_DIR / "champions.toml"

#: Boss rows the pre-Underdocks champions were trained on. Every one of these
#: indices must produce identical output before and after the re-bake; that is
#: the property that makes this a re-lay rather than a change of model.
N_TRAINED_BOSS_ROWS = 9


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def random_inputs(
    rng: np.random.Generator, t_max: int, a_max: int, cat_dim: int, n_cont: int
) -> dict:
    """One synthetic decision, valid for BOTH graph widths except the boss
    column, which the caller patches per-graph (its pad index moved).

    The observation occupies the first ``obs_len`` tokens with in-range ids;
    everything after is padding, masked out of attention and the value pool.

    ``cat_dim`` / ``n_cont`` come from the graph, not from ``ppo.schema``:
    champions are promoted at different schema generations (cat 10..14, cont
    19..26 today), and an input sized from the CURRENT schema is rejected by
    every older champion before the comparison runs (GH #576).
    """
    from ppo.schema import N_SECTION, empty_cat

    pad = np.array(empty_cat(), dtype=np.int64)[:cat_dim]
    obs_len = 24
    cat = np.tile(pad, (1, t_max, 1))
    for t in range(obs_len + a_max):
        for c in range(cat.shape[2]):
            # Stay strictly below the pad row so the id is valid under BOTH
            # widths; the boss column is overwritten below anyway.
            cat[0, t, c] = int(rng.integers(0, max(1, pad[c])))
        cat[0, t, 9] = int(rng.integers(0, N_TRAINED_BOSS_ROWS))
    pad_mask = np.ones((1, t_max), dtype=bool)
    pad_mask[0, : obs_len + a_max] = False
    return {
        "cat": cat,
        "cont": rng.standard_normal((1, t_max, n_cont)).astype(np.float32),
        "sec": rng.integers(0, N_SECTION, size=(1, t_max)).astype(np.int64),
        "pad_mask": pad_mask,
        "obs_len": np.array([obs_len], dtype=np.int64),
        "act_mask": np.ones((1, a_max), dtype=bool),
    }


def boss_rows(path: Path) -> int | None:
    """Height of this graph's ``emb_boss`` table, or ``None`` if it has none.

    A champion promoted before the boss-identity input exports ``cat_dim == 9``
    and carries no boss table at all — ``PpoNet::infer`` drops the trailing slot
    and the Rust height guard only checks the columns the graph consumes. Those
    nets are unaffected by a boss-vocab change and MUST NOT be re-exported here:
    doing so would silently widen their input contract to 10 columns, which is a
    model change, not a re-lay.
    """
    import onnx

    model = onnx.load(str(path))
    for init in model.graph.initializer:
        if init.name.endswith("emb_boss.weight"):
            return init.dims[0]
    return None


def input_contract(path: Path) -> dict[str, list]:
    """The graph's input names -> shapes (the contract the Rust host feeds)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return {i.name: list(i.shape) for i in sess.get_inputs()}


def compare_graphs(old: Path, new: Path, trials: int = 8) -> None:
    """Assert the two graphs answer identically on trained boss rows.

    Whether the table SHOULD have grown is the caller's call: on the mutating
    path a same-width re-export is "already current, nothing to do", and on the
    ``--check`` path it is the healthy state ``--check`` exists to confirm
    (GH #576 -- this used to assert ``new > old`` unconditionally, so
    ``--check`` could never pass once the migration was done).
    """
    import onnxruntime as ort

    old_rows, new_rows = boss_rows(old), boss_rows(new)
    print(f"    emb_boss rows {old_rows} -> {new_rows}")
    # The caller only screens the OLD graph for a boss table. A re-export that
    # dropped it would reach here as `None` and compare as a TypeError, so name
    # the failure instead of letting the `>=` below raise it.
    assert old_rows is not None, "old graph has no emb_boss table to compare"
    assert new_rows is not None, "re-bake dropped the emb_boss table"
    assert new_rows >= old_rows, (
        f"re-export NARROWED the boss table ({old_rows} -> {new_rows}); "
        "the current vocab is smaller than the committed champion's"
    )

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s_old = ort.InferenceSession(str(old), so, providers=["CPUExecutionProvider"])
    s_new = ort.InferenceSession(str(new), so, providers=["CPUExecutionProvider"])
    inputs = {i.name: i.shape for i in s_old.get_inputs()}
    t_max = inputs["cat"][1]
    a_max = inputs["act_mask"][1]
    cat_dim = inputs["cat"][2]
    n_cont = inputs["cont"][2]

    rng = np.random.default_rng(0)
    for i in range(trials):
        feed = random_inputs(rng, t_max, a_max, cat_dim, n_cont)
        # The boss pad row moved with the table, so each graph gets its own.
        feed_old = dict(feed)
        feed_new = dict(feed)
        cat_old, cat_new = feed["cat"].copy(), feed["cat"].copy()
        pad_col = feed["pad_mask"][0]
        cat_old[0, pad_col, 9] = old_rows - 1
        cat_new[0, pad_col, 9] = new_rows - 1
        feed_old["cat"], feed_new["cat"] = cat_old, cat_new

        lo, vo = (np.asarray(x) for x in s_old.run(None, feed_old))
        ln, vn = (np.asarray(x) for x in s_new.run(None, feed_new))
        if not (np.array_equal(lo, ln) and np.array_equal(vo, vn)):
            dl = float(np.max(np.abs(lo - ln)))
            dv = float(np.max(np.abs(vo - vn)))
            raise AssertionError(
                f"re-bake changed the model on trial {i}: max |dlogit|={dl}, "
                f"max |dvalue|={dv}. The row re-lay is wrong — do NOT ship this net."
            )
    print(f"    {trials} trials bit-identical on boss rows 0..{N_TRAINED_BOSS_ROWS - 1}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify only; re-export to a temp dir and leave the tree untouched",
    )
    args = ap.parse_args()

    from ppo import vocab
    from ppo.export_onnx import export

    print(
        f"boss vocab is now {len(vocab.bosses().names)} ids "
        f"(emb_boss must be {vocab.bosses().size + 1} rows)"
    )

    target_rows = vocab.bosses().size + 1
    champions = tomllib.loads(CHAMPIONS_TOML.read_text(encoding="utf-8"))
    updates: dict[str, str] = {}
    stale: list[str] = []
    blocked: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for character, entry in champions.items():
            onnx_path = PPO_DIR / entry["onnx"]
            ckpt_path = PPO_DIR / entry["ckpt"]
            print(f"\n[{character}] {entry['onnx']}")
            if not ckpt_path.exists():
                print(f"    SKIP: checkpoint {ckpt_path.name} is missing")
                continue
            old_rows = boss_rows(onnx_path)
            if old_rows is None:
                print(
                    "    SKIP: pre-boss champion (cat_dim 9, no emb_boss table) — "
                    "unaffected by the boss-vocab change"
                )
                continue
            # Always export to a temp file first: the committed champion is
            # only replaced after the re-export has been verified.
            out = tmpdir / f"{character}.after.onnx"
            export(str(ckpt_path), str(out))

            # `export` emits the CURRENT schema width (trailing cat/cont columns
            # zero-filled by `load_compat`), so for a champion promoted at an
            # older schema the re-export is a different input contract, not a
            # boss-row re-lay. Never install that; and its graphs cannot be fed
            # the same inputs, so there is nothing to compare bit-for-bit.
            old_in, new_in = input_contract(onnx_path), input_contract(out)
            if old_in != new_in:
                changed = {
                    k: (old_in.get(k), new_in.get(k))
                    for k in old_in
                    if old_in.get(k) != new_in.get(k)
                }
                if old_rows == target_rows:
                    print(
                        f"    OK: emb_boss already {old_rows} rows. (A re-export from the "
                        f"checkpoint would change the input contract {changed}; not compared.)"
                    )
                else:
                    blocked.append(character)
                    print(
                        f"    BLOCKED: emb_boss is {old_rows} rows (want {target_rows}), but a "
                        f"re-export would also change the input contract {changed}. That is a "
                        "model change, not a re-lay — re-bake this champion by hand."
                    )
                continue

            compare_graphs(onnx_path, out)
            if boss_rows(out) == old_rows:
                print("    OK: already at the current width")
                continue
            if args.check:
                stale.append(character)
                print("    STALE: committed champion is narrower than the current vocab")
                continue
            shutil.copy2(out, onnx_path)
            digest = sha256_of(onnx_path)
            if digest != entry["sha256"]:
                updates[character] = digest
                print(f"    sha256 {entry['sha256'][:12]} -> {digest[:12]}")

    if blocked:
        print(f"\nFAILED: {', '.join(blocked)} cannot be re-baked by re-export (see above)")
        return 1
    if args.check:
        print("\n--check: nothing written.")
        if stale:
            print(
                f"--check FAILED: {', '.join(stale)} need a re-bake "
                "(run this script without --check)"
            )
            return 1
        print("--check OK: every champion with a boss table is at the current width")
        return 0
    if updates:
        text = CHAMPIONS_TOML.read_text(encoding="utf-8")
        for character, digest in updates.items():
            old = champions[character]["sha256"]
            # Anchor on the character's own table so two champions that
            # somehow shared a digest could not cross-write.
            section = re.search(rf"(\[{re.escape(character)}\][^\[]*)", text, flags=re.S)
            assert section, f"no [{character}] table in champions.toml"
            text = text.replace(section.group(1), section.group(1).replace(old, digest), 1)
        CHAMPIONS_TOML.write_text(text, encoding="utf-8")
        print(f"\nchampions.toml: updated {len(updates)} sha256 entries")
    else:
        print("\nchampions.toml: nothing re-baked, all sha256 entries unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
