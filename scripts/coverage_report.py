"""Per-source + combined-coverage report across the indices we have. Helps pick
the balanced-sample cap before precomputing embeddings.

Usage:
    python scripts/coverage_report.py
    python scripts/coverage_report.py --names blalexa,train,europe,kaggle50k,paulchambaz
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from geogg.geocells import Grid
from geogg.paths import GRID_PATH, INDEX_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="blalexa,train,europe,kaggle50k,paulchambaz",
                    help="comma-separated index basenames under artifacts/index/")
    args = ap.parse_args()

    grid = Grid.load(GRID_PATH)
    n_cells = grid.num_classes
    names = args.names.split(",")

    print(f"{'source':<20}{'images':>12}{'cells':>10}{'countries':>12}")
    print("-" * 54)
    dfs = []
    for n in names:
        p = INDEX_DIR / f"{n}.parquet"
        if not p.exists():
            print(f"{n:<20}{'MISSING':>12}")
            continue
        df = pd.read_parquet(p, columns=["cell_index", "country"])
        dfs.append((n, df))
        print(f"{n:<20}{len(df):>12,}{df['cell_index'].nunique():>10,}{df['country'].nunique():>12,}")

    combined = pd.concat([df for _, df in dfs], ignore_index=True)
    counts = combined["cell_index"].value_counts()
    arr = counts.values
    print("-" * 54)
    print(f"{'COMBINED':<20}{len(combined):>12,}{len(counts):>10,}{combined['country'].nunique():>12,}")
    print(f"\ncell coverage: {len(counts)}/{n_cells} populated  ({n_cells - len(counts)} empty)")
    print(f"images/cell: min {arr.min()} | median {int(np.median(arr))} | mean {arr.mean():.0f} | max {arr.max()}")
    for thr in (10, 20, 50, 100, 200):
        below = (arr < thr).sum()
        print(f"  cells with < {thr:3d}: {below} ({100*below/len(counts):.0f}% of populated)")

    print("\ntotal images used if balanced cap N per cell:")
    for n in (40, 60, 80, 120, 160, 200, 300):
        print(f"  cap {n:3d}: {int(np.minimum(arr, n).sum()):>12,}")


if __name__ == "__main__":
    main()
