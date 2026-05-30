"""Build geocell+country indices for the free external Google datasets
(saleha1wer Europe, Kaggle 50k, Muninn). Coordinates come from filenames/folders
(latitude first, then longitude); countries are reverse-geocoded.

Usage:
    python scripts/build_external_index.py --source europe
    python scripts/build_external_index.py --source kaggle50k
    python scripts/build_external_index.py --source muninn
"""

from __future__ import annotations

import argparse

from geogg.dataset import build_csv_indexed_index, build_europe_index, build_filename_coord_index
from geogg.geocells import Grid
from geogg.geocode import CountryLocator
from geogg.paths import DATA_ROOT, GRID_PATH, INDEX_DIR

SOURCES = {
    "europe": DATA_ROOT / "google_europe" / "images",
    "kaggle50k": DATA_ROOT / "kaggle_50k",
    "muninn": DATA_ROOT / "muninn",
    "paulchambaz": DATA_ROOT / "google_kaggle" / "dataset",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), required=True)
    ap.add_argument("--dir", default=None, help="override the dataset directory")
    args = ap.parse_args()

    directory = args.dir or SOURCES[args.source]
    grid = Grid.load(GRID_PATH)
    loc = CountryLocator("50m")
    print(f"indexing {args.source} from {directory} (grid: {grid.num_classes} cells)")

    if args.source == "europe":
        df = build_europe_index(directory, grid, loc)
    elif args.source == "paulchambaz":
        df = build_csv_indexed_index(directory, f"{directory}/coords.csv", grid, loc,
                                     source="paulchambaz", dataset="paulchambaz")
    else:
        df = build_filename_coord_index(directory, grid, loc, source=args.source, dataset=args.source)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out = INDEX_DIR / f"{args.source}.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {out} | {len(df):,} images | {df['cell_index'].nunique()} cells | {df['country'].nunique()} countries")


if __name__ == "__main__":
    main()
