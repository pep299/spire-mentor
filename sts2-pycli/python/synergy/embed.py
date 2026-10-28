"""UMAP embedding for card-synergy catalog nodes.

Computes per-card 2-D UMAP coordinates from a PMI co-occurrence profile,
bakes them into an existing catalog.json, and derives an emergent hue from
the embedding angle.

Usage (CLI subcommand wired in synergy/cli.py)::

    python -m synergy.cli embed-catalog \\
        --obs obs_catalog.json \\
        --catalog catalog.json \\
        --out catalog.json          # overwrite in-place by default

Algorithm
---------
1. Load obs_catalog.json (dense freq-floor-50 run) to get the full PMI matrix.
2. For each node in the *target* catalog.json, build a feature vector:
   its PMI with every other card in the dense catalog (0 where absent).
3. Run UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
   on the feature matrix → 2-D coords.
4. Normalize x,y to [0,1].
5. Derive a `hue` (0-360) from atan2(y-0.5, x-0.5) mapped to 0-360 — this
   makes position-proximate cards share a similar hue, entirely from the data.
6. Bake `x`, `y`, `hue` onto each node dict.  Cards with no PMI profile row
   are skipped (no x/y/hue fields → frontend places them aside).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pmi_matrix(
    obs_pairs: list[dict],
) -> tuple[dict[str, int], np.ndarray]:
    """Build a symmetric card×card PMI matrix from obs_catalog pairs.

    Args:
        obs_pairs: The ``pairs`` list from an obs_catalog.json dict.  Each entry
            must have ``pair`` (list of two card tokens) and ``pmi`` (float).

    Returns:
        Tuple of (token_to_index dict, square numpy float32 PMI matrix).
        The matrix is symmetric: ``M[i,j] == M[j,i] == pmi``.
        Diagonal is 0.
    """
    # Collect all tokens
    tokens_seen: set[str] = set()
    for entry in obs_pairs:
        tokens_seen.update(entry["pair"])
    token_list = sorted(tokens_seen)
    tok2idx: dict[str, int] = {t: i for i, t in enumerate(token_list)}

    n = len(token_list)
    mat = np.zeros((n, n), dtype=np.float32)
    for entry in obs_pairs:
        a, b = entry["pair"]
        i, j = tok2idx[a], tok2idx[b]
        v = float(entry["pmi"])
        mat[i, j] = v
        mat[j, i] = v

    return tok2idx, mat


def embed_catalog(
    *,
    obs_path: Path,
    catalog_path: Path,
    out_path: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> None:
    """Compute UMAP embedding and bake x/y/hue into catalog.json.

    Reads obs_catalog.json for the dense PMI matrix and catalog.json for the
    node list.  Writes the augmented catalog to out_path (may be the same as
    catalog_path for in-place update).

    Args:
        obs_path: Path to obs_catalog.json (dense freq-floor-50 source).
        catalog_path: Path to catalog.json to augment with x/y/hue.
        out_path: Output path for the augmented catalog.
        n_neighbors: UMAP n_neighbors (default 15).
        min_dist: UMAP min_dist (default 0.1).
        metric: UMAP distance metric (default "cosine").
        random_state: RNG seed for reproducibility (default 42).
    """
    import umap  # local import — only needed when running embedding

    obs_raw = json.loads(obs_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    obs_pairs = obs_raw.get("pairs", [])
    tok2idx, pmi_mat = build_pmi_matrix(obs_pairs)

    nodes = catalog["nodes"]

    import sys

    zero_nodes = {n["id"] for n in nodes if n["id"] not in tok2idx}
    if zero_nodes:
        print(
            f"embed_catalog: {len(zero_nodes)} nodes have no PMI profile — "
            "embedding as zero-vectors (will cluster together in UMAP)",
            file=sys.stderr,
        )

    if len(nodes) < 2:
        out_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        return

    n_dense = len(tok2idx)
    # Feature matrix: each row = that card's PMI profile vs all cards in the dense matrix.
    # Cards absent from the PMI matrix get a zero vector.
    X = np.zeros((len(nodes), n_dense), dtype=np.float32)
    for i, node in enumerate(nodes):
        idx = tok2idx.get(node["id"])
        if idx is not None:
            X[i] = pmi_mat[idx]

    # Clamp n_neighbors to n_samples - 1 (handles tiny test inputs)
    actual_neighbors = min(n_neighbors, len(nodes) - 1)

    reducer = umap.UMAP(
        n_neighbors=actual_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2,
    )
    coords = reducer.fit_transform(X)  # shape: (len(nodes), 2)

    # Normalize to [0, 1]
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    xrange = xmax - xmin if xmax > xmin else 1.0
    yrange = ymax - ymin if ymax > ymin else 1.0
    coords_norm = np.column_stack(
        [
            (coords[:, 0] - xmin) / xrange,
            (coords[:, 1] - ymin) / yrange,
        ]
    )

    # Build fast lookup: node_id -> (x_norm, y_norm)
    id_to_coord: dict[str, tuple[float, float]] = {}
    for node, (x_n, y_n) in zip(nodes, coords_norm):
        id_to_coord[node["id"]] = (float(x_n), float(y_n))

    # Bake x, y, hue back onto nodes (in-place mutation of catalog dict)
    for node in nodes:
        coord = id_to_coord.get(node["id"])
        if coord is None:
            continue
        x_n, y_n = coord
        node["x"] = round(x_n, 6)
        node["y"] = round(y_n, 6)
        # Emergent hue: angle from center mapped to [0, 360)
        angle = math.atan2(y_n - 0.5, x_n - 0.5)
        node["hue"] = round((math.degrees(angle) + 360) % 360, 1)

    out_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
