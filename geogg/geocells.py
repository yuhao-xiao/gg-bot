"""Adaptive S2 geocells: turn the globe into a set of cells that are SMALL where
training data is dense and LARGE where it's sparse, then treat "which cell?" as a
classification problem.

Why adaptive (vs a uniform lat/lon grid)? Street View data is wildly uneven (the
US alone is ~1.2M points; Bhutan has 41). A uniform grid would give thousands of
empty cells and a few hugely overcrowded ones. Instead we subdivide a cell only
when it holds more than `max_per_cell` points -> dense regions (W. Europe, Japan)
get many fine cells; empty oceans/deserts get none. This is the same idea as the
quadtree cells OSV-5M ships and the "semantic geocells" in PIGEON.

We use Google's S2 library (s2sphere): it maps the sphere onto a quadtree where
each cell has 4 children one level deeper. Level 0 ~ a face of the cube (~85M
km^2); each extra level divides area by 4 (level 8 ~ 1300 km^2, level 12 ~ 5 km^2).

Build algorithm (equivalent to recursive top-down splitting, done efficiently):
  1. Snap every point to its S2 cell at `max_level` (the finest allowed) = "atoms".
  2. For each candidate level L from base_level (coarse) -> max_level (fine),
     count how many points fall under each cell at that level.
  3. A point's leaf cell = the COARSEST ancestor whose count <= max_per_cell
     (i.e. the first level, scanning coarse->fine, that is small enough). If even
     the finest level is too big, the leaf is that finest cell.
This yields a disjoint partition: no leaf is an ancestor of another leaf.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import s2sphere as s2
from tqdm import tqdm


def _cellid(lat: float, lon: float, level: int) -> s2.CellId:
    return s2.CellId.from_lat_lng(s2.LatLng.from_degrees(lat, lon)).parent(level)


@dataclass
class GridParams:
    base_level: int
    max_level: int
    max_per_cell: int


def build_adaptive_grid(
    lats: Sequence[float],
    lons: Sequence[float],
    base_level: int = 4,
    max_level: int = 12,
    max_per_cell: int = 2500,
    min_count: int = 0,
) -> tuple[list[dict], GridParams]:
    """Return a list of leaf-cell dicts (token, level, count, lat, lon, class_index).

    base_level = coarsest allowed (largest cell); max_level = finest allowed
    (smallest cell). min_count drops leaf cells with fewer than this many points
    (removes near-empty sparse cells)."""
    n = len(lats)

    # 1) atoms at max_level: count + lat/lon sums per finest cell
    atom_count: dict[str, int] = defaultdict(int)
    atom_sumlat: dict[str, float] = defaultdict(float)
    atom_sumlon: dict[str, float] = defaultdict(float)
    atom_cellid: dict[str, s2.CellId] = {}
    for lat, lon in tqdm(zip(lats, lons), total=n, desc="snapping points -> S2 atoms"):
        cid = _cellid(lat, lon, max_level)
        tok = cid.to_token()
        atom_count[tok] += 1
        atom_sumlat[tok] += lat
        atom_sumlon[tok] += lon
        if tok not in atom_cellid:
            atom_cellid[tok] = cid
    print(f"  {len(atom_cellid):,} unique atoms at level {max_level}")

    # 2) population under each cell at every candidate level
    levels = range(base_level, max_level + 1)
    counts: dict[int, dict[str, int]] = {L: defaultdict(int) for L in levels}
    for tok, cid in atom_cellid.items():
        c = atom_count[tok]
        for L in levels:
            counts[L][cid.parent(L).to_token()] += c

    # 3) assign each atom to its leaf (coarsest ancestor that is small enough)
    leaf_count: dict[str, int] = defaultdict(int)
    leaf_sumlat: dict[str, float] = defaultdict(float)
    leaf_sumlon: dict[str, float] = defaultdict(float)
    leaf_level: dict[str, int] = {}
    for tok, cid in atom_cellid.items():
        chosen, chosen_level = None, None
        for L in levels:
            ptok = cid.parent(L).to_token()
            if counts[L][ptok] <= max_per_cell:
                chosen, chosen_level = ptok, L
                break
        if chosen is None:  # even finest level exceeds the cap -> keep finest
            chosen, chosen_level = tok, max_level
        leaf_count[chosen] += atom_count[tok]
        leaf_sumlat[chosen] += atom_sumlat[tok]
        leaf_sumlon[chosen] += atom_sumlon[tok]
        leaf_level[chosen] = chosen_level

    cells = [
        {
            "token": tok,
            "level": leaf_level[tok],
            "count": leaf_count[tok],
            "lat": leaf_sumlat[tok] / leaf_count[tok],
            "lon": leaf_sumlon[tok] / leaf_count[tok],
        }
        for tok in leaf_count
        if leaf_count[tok] >= min_count
    ]
    cells.sort(key=lambda d: -d["count"])
    for i, c in enumerate(cells):
        c["class_index"] = i

    return cells, GridParams(base_level, max_level, max_per_cell)


class Grid:
    """A built geocell grid: maps (lat, lon) -> cell token -> class index."""

    def __init__(self, cells: list[dict], params: GridParams) -> None:
        self.cells = cells
        self.params = params
        self.token_to_index = {c["token"]: c["class_index"] for c in cells}
        self.tokens = set(self.token_to_index)
        self.index_to_cell = {c["class_index"]: c for c in cells}

    @property
    def num_classes(self) -> int:
        return len(self.cells)

    def assign_token(self, lat: float, lon: float) -> str | None:
        """Return the leaf-cell token a point falls in, or None if outside the grid."""
        cid = _cellid(lat, lon, self.params.max_level)
        for L in range(self.params.base_level, self.params.max_level + 1):
            tok = cid.parent(L).to_token()
            if tok in self.tokens:
                return tok
        return None

    def assign_index(self, lat: float, lon: float) -> int | None:
        tok = self.assign_token(lat, lon)
        return self.token_to_index[tok] if tok is not None else None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": {
                "base_level": self.params.base_level,
                "max_level": self.params.max_level,
                "max_per_cell": self.params.max_per_cell,
            },
            "cells": self.cells,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Grid":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        p = payload["params"]
        return cls(payload["cells"], GridParams(p["base_level"], p["max_level"], p["max_per_cell"]))


def cell_to_medium(grid: Grid, medium_level: int = 4) -> tuple[dict[int, int], int]:
    """Map each fine cell_index -> a coarser 'medium' index (its S2 level-`medium_level`
    ancestor). Used by the hierarchical multi-task head. Returns (mapping, n_mediums)."""
    tokens_to_medium: dict[str, int] = {}
    cell_to_medium_idx: dict[int, int] = {}
    for cell in grid.cells:
        parent_token = s2.CellId.from_token(cell["token"]).parent(medium_level).to_token()
        if parent_token not in tokens_to_medium:
            tokens_to_medium[parent_token] = len(tokens_to_medium)
        cell_to_medium_idx[cell["class_index"]] = tokens_to_medium[parent_token]
    return cell_to_medium_idx, len(tokens_to_medium)
