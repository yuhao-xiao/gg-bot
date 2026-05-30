"""Build + save the unified image index for the OSV-5M splits we've downloaded.

Usage:
    python scripts/build_index.py            # train + test
    python scripts/build_index.py --split train
"""

from __future__ import annotations

import argparse

from geogg.dataset import build_osv5m_index
from geogg.geocells import Grid
from geogg.paths import GRID_PATH, INDEX_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = ap.parse_args()

    grid = Grid.load(GRID_PATH)
    print(f"grid: {grid.num_classes} cells")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        df = build_osv5m_index(split, grid)
        out = INDEX_DIR / f"{split}.parquet"
        df.to_parquet(out, index=False)
        n_cells = df["cell_index"].nunique()
        print(f"saved {out} | {len(df):,} images | {n_cells:,}/{grid.num_classes} cells populated\n")


if __name__ == "__main__":
    main()
