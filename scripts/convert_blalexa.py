"""Convert the blalexa global Google panorama dataset to perspective crops.

Resumable + crash-safe: writes a per-shard index 'part' after each shard and
skips shards/crops already done, so re-running continues where it left off and a
crash only costs the in-progress shard. Downloads one shard at a time and deletes
it afterward to keep disk use low.

Usage:
    python scripts/convert_blalexa.py --shards 235          # run / resume
    python scripts/convert_blalexa.py --merge-only          # rebuild index from parts
"""

from __future__ import annotations

import argparse
import io
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from tqdm import tqdm

from geogg.geocells import Grid
from geogg.geocode import CountryLocator
from geogg.panorama import pano_to_perspectives
from geogg.paths import DATA_ROOT, GRID_PATH, INDEX_DIR

REPO = "blalexa/google-streetview-panoramas-geotagged"
OUT_DIR = DATA_ROOT / "blalexa_perspective"
PARTS_DIR = INDEX_DIR / "blalexa_parts"
HEADINGS = (0.0, 90.0, 180.0, 270.0)
FOV, SIZE = 90.0, 384  # crops are resized to 224 for embedding, so 384 is plenty
MAX_SRC_WIDTH = 1536  # downscale equirect before projecting (4x crop = full detail)


def _img_bytes(field) -> bytes:
    return field["bytes"] if isinstance(field, dict) else field


def _crops_exist(subdir: Path, pid: str) -> bool:
    return all((subdir / f"{pid}_{int(h)}.jpg").exists() for h in HEADINGS)


def _convert_one(args) -> bool:
    """Worker: decode one panorama's bytes and save its perspective crops."""
    raw, pid, subdir = args
    try:
        crops = pano_to_perspectives(Image.open(io.BytesIO(raw)), HEADINGS, FOV, SIZE, max_src_width=MAX_SRC_WIDTH)
    except Exception:
        return False
    for h, crop in zip(HEADINGS, crops):
        crop.save(Path(subdir) / f"{pid}_{int(h)}.jpg", quality=90)
    return True


def _rows_for(subdir, pid, lat, lon, country, cell):
    return [{"path": str(Path(subdir) / f"{pid}_{int(h)}.jpg"), "lat": lat, "lon": lon,
             "country": country, "source": "blalexa", "dataset": "blalexa",
             "pano_id": pid, "heading": h, "cell_index": cell} for h in HEADINGS]


def merge_parts() -> None:
    parts = sorted(PARTS_DIR.glob("*.parquet"))
    if not parts:
        print("no parts to merge")
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    out = INDEX_DIR / "blalexa.parquet"
    df.to_parquet(out, index=False)
    print(f"merged {len(parts)} parts -> {out} | {len(df):,} crops | {df['cell_index'].nunique()} cells")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=235)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_parts()
        return

    grid = Grid.load(GRID_PATH)
    loc = CountryLocator("50m")
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))[: args.shards]
    shard_dir = DATA_ROOT / "blalexa_shards"
    todo = [(fi, fn) for fi, fn in enumerate(files) if not (PARTS_DIR / f"{fi:05d}.parquet").exists()]
    print(f"{len(files)} shards | {len(files) - len(todo)} done | {len(todo)} to convert | {args.workers} workers")

    dl = ThreadPoolExecutor(max_workers=1)  # prefetch shard downloads
    futures: dict[int, object] = {}

    def prefetch(k: int) -> None:
        if 0 <= k < len(todo) and k not in futures:
            futures[k] = dl.submit(hf_hub_download, REPO, filename=todo[k][1],
                                   repo_type="dataset", local_dir=str(shard_dir))

    prefetch(0)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:  # one persistent pool
        for k, (fi, fname) in enumerate(todo):
            local = futures.pop(k).result()  # wait for this shard
            prefetch(k + 1)                  # download next while we convert this one
            print(f"[{fi+1}/{len(files)}] {fname}")
            subdir = OUT_DIR / f"{fi:05d}"    # per-shard folder: keeps each dir small (exFAT-friendly)
            subdir.mkdir(parents=True, exist_ok=True)

            meta = pd.read_parquet(local, columns=["pano_id", "lat", "lon"])
            rows, job_meta, need_pos = [], [], []
            for pos, r in enumerate(meta.itertuples(index=False)):
                cell = grid.assign_index(float(r.lat), float(r.lon))
                if cell is None:
                    continue
                pid = str(r.pano_id)
                country = loc.country_of(float(r.lat), float(r.lon)) or "UNK"
                if _crops_exist(subdir, pid):  # recover already-converted crops without re-decoding
                    rows += _rows_for(subdir, pid, float(r.lat), float(r.lon), country, cell)
                else:
                    job_meta.append((pid, float(r.lat), float(r.lon), country, cell))
                    need_pos.append(pos)

            if need_pos:  # only load heavy image bytes for panos that need converting
                imgs = pd.read_parquet(local, columns=["image"])["image"].to_numpy()
                jobs = [(_img_bytes(imgs[p]), job_meta[i][0], str(subdir)) for i, p in enumerate(need_pos)]
                oks = list(tqdm(ex.map(_convert_one, jobs), total=len(jobs), desc=f"shard {fi+1}"))
                for (pid, lat, lon, country, cell), ok in zip(job_meta, oks):
                    if ok:
                        rows += _rows_for(subdir, pid, lat, lon, country, cell)

            pd.DataFrame(rows).to_parquet(PARTS_DIR / f"{fi:05d}.parquet", index=False)  # checkpoint
            Path(local).unlink(missing_ok=True)
    dl.shutdown()
    merge_parts()


if __name__ == "__main__":
    main()
