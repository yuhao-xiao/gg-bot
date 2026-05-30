"""Sanity-check the locked grid: stats + where real landmarks land."""

from __future__ import annotations

import argparse

import numpy as np

from geogg.geocells import Grid
from geogg.metrics import haversine_km
from geogg.paths import GRID_PATH

LANDMARKS = {
    "Paris": (48.8566, 2.3522), "London": (51.5074, -0.1278),
    "Tokyo": (35.6762, 139.6503), "New York": (40.7128, -74.0060),
    "Sydney": (-33.8688, 151.2093), "Sao Paulo": (-23.5505, -46.6333),
    "Cape Town": (-33.9249, 18.4241), "Reykjavik": (64.1466, -21.9426),
    "Singapore": (1.3521, 103.8198), "Toronto": (43.6532, -79.3832),
    "Mumbai": (19.0760, 72.8777), "Mexico City": (19.4326, -99.1332),
    "Nuuk (no GSV)": (64.18, -51.69), "mid-Pacific (ocean)": (0.0, -140.0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=str(GRID_PATH))
    args = ap.parse_args()
    grid = Grid.load(args.grid)
    counts = np.array([c["count"] for c in grid.cells])
    levels = np.array([c["level"] for c in grid.cells])
    print(f"grid: {grid.num_classes} cells | levels {levels.min()}-{levels.max()}")
    print(f"coverage pts/cell: min={counts.min()} median={int(np.median(counts))} max={counts.max()}")
    print("cells per level:", {int(L): int((levels == L).sum()) for L in range(levels.min(), levels.max() + 1)})

    print("\nlandmark -> cell (centroid offset = how far cell centre is from the city):")
    for name, (lat, lon) in LANDMARKS.items():
        ci = grid.assign_index(lat, lon)
        if ci is None:
            print(f"  {name:20s}: OUTSIDE grid (no cell)")
            continue
        cell = grid.index_to_cell[ci]
        off = float(haversine_km(lat, lon, cell["lat"], cell["lon"]))
        print(f"  {name:20s}: cell {ci:4d} | level {cell['level']} | {cell['count']:4d} pts | centroid {off:5.0f} km away")


if __name__ == "__main__":
    main()
