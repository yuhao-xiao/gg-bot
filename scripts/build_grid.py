"""Build the adaptive S2 geocell grid from GSV-filtered OSV-5M coordinates,
save it, print a report, and render a density/adaptivity map.

Usage:
    python scripts/build_grid.py                       # full, default params
    python scripts/build_grid.py --max-points 500000   # fast preview on a subsample
    python scripts/build_grid.py --max-per-cell 5000   # coarser grid (fewer cells)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from geogg.data import load_metadata  # noqa: E402
from geogg.geocells import Grid, build_adaptive_grid  # noqa: E402
from geogg.paths import GOOGLE_COVERAGE_PARQUET  # noqa: E402


def load_points(args) -> tuple[np.ndarray, np.ndarray]:
    """Return (lats, lons) from the chosen coverage source."""
    if args.source == "osv5m":
        df = load_metadata("train", ["latitude", "longitude", "country"])
        print(f"loaded {len(df):,} OSV-5M (Mapillary) GSV points")
        return df["latitude"].to_numpy(), df["longitude"].to_numpy()
    # google coverage parquet from scripts/sample_google_coverage.py
    df = pd.read_parquet(args.google_path)
    if args.official_only and "official" in df.columns:
        df = df[df["official"]]
    print(f"loaded {len(df):,} Google coverage points from {args.google_path}")
    return df["lat"].to_numpy(), df["lon"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["osv5m", "google"], default="osv5m")
    ap.add_argument("--google-path", default=str(GOOGLE_COVERAGE_PARQUET))
    ap.add_argument("--official-only", action="store_true", help="(google) keep only Google-captured panos")
    ap.add_argument("--base-level", type=int, default=4, help="coarsest level (largest allowed cell)")
    ap.add_argument("--max-level", type=int, default=12, help="finest level (smallest allowed cell)")
    ap.add_argument("--max-per-cell", type=int, default=2500)
    ap.add_argument("--min-count", type=int, default=0, help="drop cells with fewer than this many points")
    ap.add_argument("--max-points", type=int, default=0, help="subsample N points for speed (0=all)")
    ap.add_argument("--out", default="artifacts/grid")
    args = ap.parse_args()

    lats_all, lons_all = load_points(args)
    df = pd.DataFrame({"latitude": lats_all, "longitude": lons_all})
    if args.max_points and args.max_points < len(df):
        df = df.sample(args.max_points, random_state=0).reset_index(drop=True)
        print(f"subsampled to {len(df):,} for this run")

    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()

    cells, params = build_adaptive_grid(
        lats, lons, args.base_level, args.max_level, args.max_per_cell, args.min_count
    )
    grid = Grid(cells, params)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    grid.save(outdir / "grid.json")

    counts = np.array([c["count"] for c in cells])
    lvls = np.array([c["level"] for c in cells])
    print(f"\n=== GRID: {grid.num_classes:,} cells ===")
    print(f"params: base={params.base_level} max={params.max_level} max_per_cell={params.max_per_cell}")
    print(
        f"points/cell: min={counts.min()} median={int(np.median(counts))} "
        f"mean={counts.mean():.0f} max={counts.max()}"
    )
    print("cells per S2 level (higher level = finer = denser area):")
    for L in range(params.base_level, params.max_level + 1):
        k = int((lvls == L).sum())
        if k:
            print(f"  level {L:2d}: {k:6d} cells")

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.hexbin(lons, lats, gridsize=300, bins="log", cmap="Greys", mincnt=1)
    sc = ax.scatter(
        [c["lon"] for c in cells], [c["lat"] for c in cells],
        c=lvls, s=5, cmap="viridis", linewidths=0,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 78)
    ax.set_title(
        f"Adaptive S2 geocells: {grid.num_classes} cells "
        f"(dot color = S2 level; grey = point density)"
    )
    fig.colorbar(sc, ax=ax, label="S2 level (higher = finer cell)")
    fig.savefig(outdir / "grid_map.png", dpi=120, bbox_inches="tight")
    print(f"\nsaved grid -> {outdir / 'grid.json'}")
    print(f"saved map  -> {outdir / 'grid_map.png'}")


if __name__ == "__main__":
    main()
