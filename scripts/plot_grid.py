"""Render a saved geocell grid as actual S2 cell SQUARES (not centroid dots),
in the style of the reference: red cell outlines on black, brighter where cells
are small and densely packed. Faint grey training points give landmass context.

Loads artifacts/grid/grid.json (no rebuild needed).

Usage:
    python scripts/plot_grid.py
    python scripts/plot_grid.py --grid artifacts/grid/grid.json --out artifacts/grid/grid_squares.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.io.shapereader as shpreader  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import s2sphere as s2  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from shapely.geometry import MultiPolygon  # noqa: E402

from geogg.data import load_metadata  # noqa: E402
from geogg.geocells import Grid  # noqa: E402


def cell_corners(token: str) -> list[tuple[float, float]]:
    """Return the 4 (lon, lat) corners of an S2 cell."""
    cell = s2.Cell(s2.CellId.from_token(token))
    pts = []
    for k in range(4):
        ll = s2.LatLng.from_point(cell.get_vertex(k))
        pts.append((ll.lng().degrees, ll.lat().degrees))
    return pts


def draw_borders(ax, color: str = "#777777", lw: float = 0.4) -> None:
    path = shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries")
    for rec in shpreader.Reader(path).records():
        geom = rec.geometry
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        for poly in polys:
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, lw=lw, zorder=1.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="artifacts/grid/grid.json")
    ap.add_argument("--out", default="artifacts/grid/grid_squares.png")
    ap.add_argument("--points", type=int, default=400000, help="faint background points (0=none)")
    ap.add_argument("--no-borders", action="store_true", help="don't draw country borders")
    ap.add_argument("--source", choices=["osv5m", "google"], default="osv5m",
                    help="which dataset to use for the faint background points")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                    help="zoom to a region, e.g. --bbox -12 35 30 60 for Europe")
    args = ap.parse_args()

    grid = Grid.load(args.grid)
    print(f"loaded {grid.num_classes} cells")

    quads, levels = [], []
    skipped = 0
    for c in grid.cells:
        corners = cell_corners(c["token"])
        lons = [p[0] for p in corners]
        if max(lons) - min(lons) > 180:  # antimeridian wrap -> skip the few offenders
            skipped += 1
            continue
        quads.append(corners)
        levels.append(c["level"])
    print(f"drawing {len(quads)} cells ({skipped} antimeridian cells skipped)")

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    if not args.no_borders:
        draw_borders(ax)

    if args.points:
        if args.source == "osv5m":
            df = load_metadata("train", ["latitude", "longitude"])
            lon_col, lat_col = "longitude", "latitude"
        else:
            import pandas as pd
            from geogg.paths import GOOGLE_COVERAGE_PARQUET
            df = pd.read_parquet(GOOGLE_COVERAGE_PARQUET)
            lon_col, lat_col = "lon", "lat"
        if args.points < len(df):
            df = df.sample(args.points, random_state=0)
        ax.scatter(df[lon_col], df[lat_col], s=0.2, c="#555555", linewidths=0, zorder=1)

    # color edges by level so adaptivity is visible, but keep the red reference vibe
    levels = np.array(levels)
    norm = (levels - levels.min()) / max(1, (levels.max() - levels.min()))
    colors = plt.cm.autumn(0.15 + 0.85 * norm)  # yellow(fine) -> red(coarse)-ish
    pc = PolyCollection(
        quads, facecolors="none", edgecolors=colors, linewidths=0.3, alpha=0.9, zorder=2
    )
    ax.add_collection(pc)

    if args.bbox:
        ax.set_xlim(args.bbox[0], args.bbox[2])
        ax.set_ylim(args.bbox[1], args.bbox[3])
    else:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 80)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{grid.num_classes} adaptive S2 geocells (color = level; brighter = finer)", color="white")
    fig.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="black")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
