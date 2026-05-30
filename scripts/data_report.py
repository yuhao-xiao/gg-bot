"""Report + visualize the training-data distribution across geocells: images per cell,
dead cells, densest cells, source/country breakdown, plus a world heatmap and a histogram.

Reads only the small artifacts/embeddings/<name>_index.parquet files (no embeddings .npy), so
it's fast. Use the same --names you train on.

Usage:
    python scripts/data_report.py
    python scripts/data_report.py --names blalexa_train,europe --balanced 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from geogg.geocells import Grid  # noqa: E402
from geogg.paths import EMB_DIR, GRID_PATH  # noqa: E402

try:  # reuse the cell-square + border drawing from plot_grid (sibling or package import)
    from plot_grid import cell_corners, draw_borders
except ImportError:
    from scripts.plot_grid import cell_corners, draw_borders

DEFAULT_NAMES = "blalexa_train,europe,kaggle50k,paulchambaz,topup"


def load_index(names: list[str]) -> pd.DataFrame:
    frames = []
    for n in names:
        f = EMB_DIR / f"{n}_index.parquet"
        if not f.exists():
            print(f"  (skip {n}: {f} not found)")
            continue
        frames.append(pd.read_parquet(f))
    if not frames:
        raise SystemExit("no index parquets found for the given --names")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=DEFAULT_NAMES, help="comma-separated embedding basenames")
    ap.add_argument("--balanced", type=int, default=0, help="show the effect of an images/cell cap")
    ap.add_argument("--top", type=int, default=25, help="how many densest cells to list")
    ap.add_argument("--out-dir", default="artifacts/data_report")
    args = ap.parse_args()

    grid = Grid.load(GRID_PATH)
    n_cells = grid.num_classes
    idx = load_index(args.names.split(","))
    print(f"\nloaded {len(idx):,} images from: {args.names}")

    counts = idx.groupby("cell_index").size()  # images per non-empty cell
    covered = counts.size
    dead = n_cells - covered
    c = counts.to_numpy()

    print(f"\n=== TRAINING DATA over {n_cells:,} grid cells ===")
    print(f"total images       : {len(idx):,}")
    print(f"cells covered      : {covered:,} ({covered / n_cells * 100:.1f}%)")
    print(f"dead cells (0 imgs): {dead:,}")
    print(f"images/cell        : min {c.min()} | p10 {np.percentile(c, 10):.0f} | "
          f"median {np.median(c):.0f} | mean {c.mean():.1f} | p90 {np.percentile(c, 90):.0f} | max {c.max()}")
    if args.balanced:
        eff = int(np.minimum(c, args.balanced).sum())
        at_cap = int((c >= args.balanced).sum())
        print(f"with --balanced {args.balanced}: {eff:,} images used | {at_cap:,} cells hit the cap")

    print(f"\nby source: {idx['source'].value_counts().to_dict()}")
    print("\ntop-20 countries by image count:")
    for cc, n in idx["country"].value_counts().head(20).items():
        print(f"  {cc:4s} {n:>9,}")

    print(f"\ntop-{args.top} densest cells (cell | images | country | centroid):")
    for ci, n in counts.sort_values(ascending=False).head(args.top).items():
        cell = grid.index_to_cell[int(ci)]
        cc = idx.loc[idx["cell_index"] == ci, "country"].mode()
        cc = cc.iloc[0] if len(cc) else "?"
        print(f"  {int(ci):5d} | {n:>7,} | {cc:3s} | {cell['lat']:7.2f},{cell['lon']:8.2f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- world map: each cell square colored by log10(images+1); dead cells dark grey ----
    quads, vals, dead_quads, skipped = [], [], [], 0
    for cell in grid.cells:
        ci = cell["class_index"]
        corners = cell_corners(cell["token"])
        lons = [pt[0] for pt in corners]
        if max(lons) - min(lons) > 180:  # antimeridian wrap -> skip the few offenders
            skipped += 1
            continue
        n = int(counts.get(ci, 0))
        if n > 0:
            quads.append(corners)
            vals.append(np.log10(n + 1))
        else:
            dead_quads.append(corners)

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    draw_borders(ax)
    if dead_quads:
        ax.add_collection(PolyCollection(dead_quads, facecolors="#222222",
                                         edgecolors="#333333", linewidths=0.2, zorder=2))
    pc = PolyCollection(quads, array=np.array(vals), cmap="viridis",
                        edgecolors="none", alpha=0.9, zorder=3)
    ax.add_collection(pc)
    cb = fig.colorbar(pc, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("log10(images per cell + 1)", color="white")
    cb.ax.tick_params(colors="white")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"training images per cell  ({len(idx):,} imgs | {covered}/{n_cells} cells | {dead} dead)",
                 color="white")
    map_png = out / "images_per_cell.png"
    fig.savefig(map_png, dpi=130, bbox_inches="tight", facecolor="black")
    print(f"\nsaved map  -> {map_png}  ({skipped} antimeridian cells skipped)")

    # ---- histogram of images-per-cell ----
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.hist(c, bins=60)
    ax2.set_yscale("log")
    ax2.set_xlabel("images in a cell")
    ax2.set_ylabel("number of cells (log)")
    if args.balanced:
        ax2.axvline(args.balanced, color="r", ls="--", label=f"--balanced {args.balanced}")
        ax2.legend()
    ax2.set_title("distribution of images per covered cell")
    hist_png = out / "images_per_cell_hist.png"
    fig2.savefig(hist_png, dpi=130, bbox_inches="tight")
    print(f"saved hist -> {hist_png}")


if __name__ == "__main__":
    main()
