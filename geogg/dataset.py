"""Build a unified image index: one row per usable image with its path,
coordinates, source tag, and assigned geocell. This is the join between the
image files we downloaded and the coordinate/label metadata, filtered to images
that fall inside our grid.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geogg.data import load_metadata
from geogg.geocells import Grid
from geogg.geocode import CountryLocator
from geogg.paths import OSV5M_DIR

IMG_EXTS = (".jpg", ".jpeg", ".png")


def build_filename_coord_index(image_dir, grid: Grid, locator: CountryLocator,
                               source: str, dataset: str) -> pd.DataFrame:
    """Index a folder of images named '{lat}_{lon}.ext' (e.g. Kaggle 50k, Muninn)."""
    rows, dropped = [], 0
    for p in Path(image_dir).rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        parts = p.stem.split("_")
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            dropped += 1
            continue
        cell = grid.assign_index(lat, lon)
        if cell is None:
            dropped += 1
            continue
        rows.append({"path": str(p), "lat": lat, "lon": lon,
                     "country": locator.country_of(lat, lon) or "UNK",
                     "source": source, "dataset": dataset, "cell_index": cell})
    print(f"  {dataset}: {len(rows):,} indexed, {dropped:,} dropped")
    return pd.DataFrame(rows)


def build_csv_indexed_index(image_dir, coords_csv, grid: Grid, locator: CountryLocator,
                            source: str, dataset: str, ext: str = ".png") -> pd.DataFrame:
    """Index a dataset where images are named '{row}.ext' and a headerless
    'lat,lon' CSV gives coords by row order (e.g. Kaggle Paul Chambaz)."""
    coords = pd.read_csv(coords_csv, header=None, names=["lat", "lon"])
    rows, dropped = [], 0
    for i, r in enumerate(coords.itertuples(index=False)):
        p = Path(image_dir) / f"{i}{ext}"
        if not p.exists():
            dropped += 1
            continue
        lat, lon = float(r.lat), float(r.lon)
        cell = grid.assign_index(lat, lon)
        if cell is None:
            dropped += 1
            continue
        rows.append({"path": str(p), "lat": lat, "lon": lon,
                     "country": locator.country_of(lat, lon) or "UNK",
                     "source": source, "dataset": dataset, "cell_index": cell})
    print(f"  {dataset}: {len(rows):,} indexed, {dropped:,} dropped")
    return pd.DataFrame(rows)


def build_europe_index(images_root, grid: Grid, locator: CountryLocator) -> pd.DataFrame:
    """Index the saleha1wer Europe set: subfolders named '{idx},{lat},{lon}',
    each holding per-heading images."""
    rows, dropped = [], 0
    for folder in Path(images_root).iterdir():
        if not folder.is_dir() or folder.name.startswith("__"):
            continue
        parts = folder.name.split(",")
        try:
            lat, lon = float(parts[-2]), float(parts[-1])  # last two fields
        except (ValueError, IndexError):
            continue
        cell = grid.assign_index(lat, lon)
        if cell is None:
            dropped += 1
            continue
        country = locator.country_of(lat, lon) or "UNK"
        for img in folder.iterdir():
            if img.suffix.lower() in IMG_EXTS:
                rows.append({"path": str(img), "lat": lat, "lon": lon, "country": country,
                             "source": "europe", "dataset": "saleha1wer_europe",
                             "heading": img.stem, "cell_index": cell})
    print(f"  europe: {len(rows):,} images indexed, {dropped:,} locations dropped")
    return pd.DataFrame(rows)


def _image_files(split: str) -> dict[str, str]:
    """Map OSV-5M image id (filename stem) -> absolute path."""
    img_dir = OSV5M_DIR / "images" / split
    out: dict[str, str] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for f in img_dir.rglob(ext):
            out[f.stem] = str(f)
    return out


def build_osv5m_index(split: str, grid: Grid) -> pd.DataFrame:
    """Index the downloaded OSV-5M (Mapillary) images for a split."""
    id_to_path = _image_files(split)
    if not id_to_path:
        raise SystemExit(f"no images found under {OSV5M_DIR / 'images' / split} - download shards first")

    meta = load_metadata(split, ["id", "latitude", "longitude", "country"], gsv_only=True)
    meta["id"] = meta["id"].astype(str)
    meta = meta[meta["id"].isin(id_to_path)].reset_index(drop=True)
    print(f"  {split}: {len(id_to_path):,} image files, {len(meta):,} matched to GSV metadata")

    rows = []
    dropped = 0
    for r in meta.itertuples(index=False):
        cell = grid.assign_index(r.latitude, r.longitude)
        if cell is None:  # point falls outside the (Google-coverage) grid
            dropped += 1
            continue
        rows.append({
            "path": id_to_path[r.id],
            "lat": r.latitude,
            "lon": r.longitude,
            "country": r.country,
            "source": "mapillary",
            "dataset": "osv5m",
            "cell_index": cell,
        })
    df = pd.DataFrame(rows)
    print(f"  {split}: {len(df):,} indexed, {dropped:,} dropped (outside grid)")
    return df


def balanced_sample(df: pd.DataFrame, max_total: int, cell_col: str = "cell_index", seed: int = 0) -> pd.DataFrame:
    """Round-robin sample across cells for a class-balanced subset (saleha1wer's
    'fixed number of locations per cell' idea). Pulls one image from each cell in
    turn until `max_total` is reached, so dense regions don't dominate."""
    import numpy as np

    groups = [g.sample(frac=1.0, random_state=seed).reset_index(drop=True) for _, g in df.groupby(cell_col)]
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    picks, pos, progressed = [], [0] * len(groups), True
    while len(picks) < max_total and progressed:
        progressed = False
        for gi, g in enumerate(groups):
            if pos[gi] < len(g):
                picks.append(g.iloc[pos[gi]])
                pos[gi] += 1
                progressed = True
                if len(picks) >= max_total:
                    break
    return pd.DataFrame(picks).reset_index(drop=True)


def cell_country_map(index: pd.DataFrame) -> dict[int, str]:
    """Majority country per cell (for country-accuracy scoring)."""
    return (
        index.groupby("cell_index")["country"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
