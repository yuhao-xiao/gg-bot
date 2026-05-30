"""Download official Google Street View IMAGES for a held-out slice of the
panorama ids we already found during coverage sampling. BILLABLE (~$0.007/image,
drawn from your $300 trial credit).

Safety: dry-run by default (prints the cost estimate). You must pass --yes to
actually spend. Hard-capped by --max-images. Resumable (skips existing files).
Test and train pools are disjoint (deterministic seeded split on pano id).

Usage:
    python scripts/download_google_images.py --split test --max-images 8000          # dry run
    python scripts/download_google_images.py --split test --max-images 8000 --yes    # download
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from geogg.dataset import balanced_sample
from geogg.geocells import Grid
from geogg.google_images import PRICE_PER_IMAGE_USD, fetch_static_image
from geogg.paths import DATA_ROOT, GOOGLE_COVERAGE_PARQUET, GRID_PATH, INDEX_DIR

TEST_FRACTION = 0.2  # first 20% of the seeded shuffle is the test pool, rest is train


class RateLimiter:
    def __init__(self, qps: float) -> None:
        self.min_interval = 1.0 / qps
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            sleep_for = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + self.min_interval
        if sleep_for:
            time.sleep(sleep_for)


def pool_for_split(cov: pd.DataFrame, split: str) -> pd.DataFrame:
    shuffled = cov.sample(frac=1.0, random_state=42).reset_index(drop=True)
    cut = int(len(shuffled) * TEST_FRACTION)
    return shuffled.iloc[:cut] if split == "test" else shuffled.iloc[cut:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train"], required=True)
    ap.add_argument("--max-images", type=int, required=True, help="hard cap on images to download")
    ap.add_argument("--qps", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--heading-mode", choices=["random", "zero"], default="random")
    ap.add_argument("--yes", action="store_true", help="actually spend credit (otherwise dry run)")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_MAPS_API_KEY not set (see .env).")

    cov = pd.read_parquet(GOOGLE_COVERAGE_PARQUET)
    grid = Grid.load(GRID_PATH)
    pool = pool_for_split(cov, args.split).copy()
    pool["cell_index"] = [grid.assign_index(lat, lon) for lat, lon in zip(pool["lat"], pool["lon"])]
    pool = pool[pool["cell_index"].notna()].reset_index(drop=True)
    pool["cell_index"] = pool["cell_index"].astype(int)
    if args.split == "train":  # class-balanced selection across cells
        pool = balanced_sample(pool, args.max_images, "cell_index", seed=42)
        print(f"balanced selection: {len(pool):,} panos across {pool['cell_index'].nunique()} cells")
    else:  # test reflects the natural coverage distribution
        pool = pool.head(args.max_images).reset_index(drop=True)

    out_dir = DATA_ROOT / "google_images" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in pool.itertuples(index=False) if not (out_dir / f"{r.pano_id}.jpg").exists()]

    est = len(todo) * PRICE_PER_IMAGE_USD
    print(f"split={args.split} pool={len(pool):,} | already have {len(pool)-len(todo):,} | to download {len(todo):,}")
    print(f"estimated cost: ${est:.2f}  (@ ${PRICE_PER_IMAGE_USD}/image)")
    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to actually download (spends credit).")
        return

    limiter = RateLimiter(args.qps)
    local = threading.local()
    lock = threading.Lock()
    done = {"n": 0, "err": 0}
    import random

    rng = random.Random(7)
    headings = {r.pano_id: (rng.uniform(0, 360) if args.heading_mode == "random" else 0.0) for r in todo}

    def session() -> requests.Session:
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s

    def work(row) -> None:
        limiter.wait()
        try:
            img = fetch_static_image(session(), row.pano_id, api_key, heading=headings[row.pano_id])
            (out_dir / f"{row.pano_id}.jpg").write_bytes(img)
            with lock:
                done["n"] += 1
        except Exception:
            with lock:
                done["err"] += 1

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(work, todo), total=len(todo), desc=f"google {args.split}"))

    spent = done["n"] * PRICE_PER_IMAGE_USD
    print(f"downloaded {done['n']:,} images ({done['err']} errors) | spent ~${spent:.2f}")

    # build index (cells already assigned during selection)
    rows = []
    for r in pool.itertuples(index=False):
        p = out_dir / f"{r.pano_id}.jpg"
        if not p.exists():
            continue
        rows.append({"path": str(p), "lat": r.lat, "lon": r.lon, "country": r.country,
                     "source": "google_api", "dataset": "coverage", "pano_id": r.pano_id,
                     "cell_index": int(r.cell_index)})
    idx = pd.DataFrame(rows)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out_index = INDEX_DIR / f"google_{args.split}.parquet"
    idx.to_parquet(out_index, index=False)
    print(f"saved index {out_index} | {len(idx):,} images | {idx['cell_index'].nunique()} cells")


if __name__ == "__main__":
    main()
