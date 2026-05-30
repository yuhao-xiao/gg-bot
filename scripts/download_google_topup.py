"""Top up sparse + empty cells with targeted Google Street View image downloads
from our coverage parquet, so every cell ends up with at least `--target` images.

Reads which cells are short across the existing source indices, picks panos from
google_coverage_land for those cells (deduped against blalexa's pano ids), and
downloads via the billable Static API. Dry-run by default.

Usage:
    python scripts/download_google_topup.py --target 20                  # dry run
    python scripts/download_google_topup.py --target 20 --yes            # spend
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from geogg.geocells import Grid
from geogg.geocode import CountryLocator
from geogg.google_images import PRICE_PER_IMAGE_USD, fetch_static_image
from geogg.paths import DATA_ROOT, GOOGLE_COVERAGE_PARQUET, GRID_PATH, INDEX_DIR


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20, help="per-cell floor")
    ap.add_argument("--sources", default="blalexa,europe,kaggle50k,paulchambaz",
                    help="existing source indices to count toward the floor")
    ap.add_argument("--max-images", type=int, default=50000, help="hard cap")
    ap.add_argument("--qps", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--heading-mode", choices=["random", "zero"], default="random")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_MAPS_API_KEY not set")

    grid = Grid.load(GRID_PATH)
    locator = CountryLocator("50m")

    # count what we already have per cell across all current sources
    counts = pd.Series(0, index=range(grid.num_classes), dtype=int)
    for n in args.sources.split(","):
        p = INDEX_DIR / f"{n}.parquet"
        if not p.exists():
            print(f"  warn: {p} missing, skipping"); continue
        c = pd.read_parquet(p, columns=["cell_index"])["cell_index"].value_counts()
        counts = counts.add(c, fill_value=0).astype(int)
    print(f"existing: {int(counts.sum()):,} images | {int((counts > 0).sum())} / {grid.num_classes} cells populated")
    print(f"cells below floor ({args.target}): {int((counts < args.target).sum())}")

    # dedup against blalexa pano ids (our coverage may overlap)
    blalexa_ids: set[str] = set()
    bp = INDEX_DIR / "blalexa.parquet"
    if bp.exists():
        blalexa_ids = set(pd.read_parquet(bp, columns=["pano_id"])["pano_id"].astype(str).unique())
        print(f"blalexa pano ids to exclude: {len(blalexa_ids):,}")

    # load coverage + assign cells
    print("assigning cells to coverage panos ...")
    cov = pd.read_parquet(GOOGLE_COVERAGE_PARQUET)
    cov["cell_index"] = [grid.assign_index(la, lo) for la, lo in zip(cov["lat"], cov["lon"])]
    cov = cov[cov["cell_index"].notna()].copy()
    cov["cell_index"] = cov["cell_index"].astype(int)
    cov = cov[~cov["pano_id"].astype(str).isin(blalexa_ids)].reset_index(drop=True)
    cov = cov.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle for variety
    print(f"usable coverage panos: {len(cov):,}")

    # pick up to (target - current) panos per under-filled cell
    selected: list[dict] = []
    by_cell = {c: g.reset_index(drop=True) for c, g in cov.groupby("cell_index")}
    for cell_idx, current in counts.items():
        need = args.target - int(current)
        if need <= 0:
            continue
        group = by_cell.get(cell_idx)
        if group is None or len(group) == 0:
            continue
        take = group.head(need)
        selected.extend(take.to_dict("records"))

    pool = pd.DataFrame(selected)
    if args.max_images and len(pool) > args.max_images:
        pool = pool.head(args.max_images).reset_index(drop=True)
    print(f"\nselected {len(pool):,} panos to download "
          f"(covers {pool['cell_index'].nunique()} cells)")
    est = len(pool) * PRICE_PER_IMAGE_USD
    print(f"estimated cost: ${est:.2f}  (@ ${PRICE_PER_IMAGE_USD}/image)")
    if not args.yes:
        print("\nDRY RUN. Pass --yes to actually download (spends credit).")
        return

    out_dir = DATA_ROOT / "google_images" / "topup"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in pool.itertuples(index=False) if not (out_dir / str(r.pano_id)[:2] / f"{r.pano_id}.jpg").exists()]
    print(f"to download (new): {len(todo):,} | already on disk: {len(pool) - len(todo):,}")

    import random as _r
    rng = _r.Random(7)
    heading = (lambda: rng.uniform(0, 360)) if args.heading_mode == "random" else (lambda: 0.0)

    limiter = RateLimiter(args.qps)
    local = threading.local()
    lock = threading.Lock()
    done = {"n": 0, "err": 0}

    def session() -> requests.Session:
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s

    def work(row) -> None:
        limiter.wait()
        try:
            img = fetch_static_image(session(), row.pano_id, api_key, heading=heading())
            p = out_dir / str(row.pano_id)[:2] / f"{row.pano_id}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(img)
            with lock:
                done["n"] += 1
        except Exception:
            with lock:
                done["err"] += 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(work, todo), total=len(todo), desc="topup"))
    print(f"downloaded {done['n']:,} ({done['err']} errors) | spent ~${done['n'] * PRICE_PER_IMAGE_USD:.2f}")

    # build index (reverse-geocode countries, since coverage was sampled in land mode)
    rows = []
    for r in pool.itertuples(index=False):
        p = out_dir / str(r.pano_id)[:2] / f"{r.pano_id}.jpg"
        if not p.exists():
            continue
        rows.append({"path": str(p), "lat": float(r.lat), "lon": float(r.lon),
                     "country": locator.country_of(float(r.lat), float(r.lon)) or "UNK",
                     "source": "google_api_topup", "dataset": "topup",
                     "pano_id": str(r.pano_id), "cell_index": int(r.cell_index)})
    idx = pd.DataFrame(rows)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out_index = INDEX_DIR / "topup.parquet"
    idx.to_parquet(out_index, index=False)
    print(f"saved {out_index} | {len(idx):,} images | {idx['cell_index'].nunique()} cells")


if __name__ == "__main__":
    main()
