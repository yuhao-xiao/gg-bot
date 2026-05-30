"""Download a SMALL subset of OSV-5M (osv5m/osv5m on Hugging Face) for local work.

The full dataset is ~5.1M images (hundreds of GB). For laptop-scale iteration we
only grab the metadata CSV(s) plus the first few image shards of one split.

The script discovers the repo's file layout at runtime (via list_repo_files) so we
don't hard-code paths that might change.

Usage:
    python scripts/download_osv5m.py --split test --shards 1
    python scripts/download_osv5m.py --split train --shards 2 --out datasets/osv5m
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from geogg.paths import OSV5M_DIR

REPO_ID = "osv5m/osv5m"
REPO_TYPE = "dataset"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--shards", type=int, default=1, help="number of image .zip shards to fetch")
    ap.add_argument("--out", default=str(OSV5M_DIR))
    ap.add_argument("--no-extract", action="store_true", help="download zips but don't unzip")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"listing files in {REPO_ID} ...")
    files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)

    # 1) metadata: any top-level / split csv files
    csvs = [f for f in files if f.endswith(".csv")]
    print(f"found {len(csvs)} csv file(s): {csvs}")
    for csv in csvs:
        if (out / csv).exists():
            print(f"  {csv} already present, skipping")
            continue
        hf_hub_download(REPO_ID, filename=csv, repo_type=REPO_TYPE, local_dir=str(out))
        print(f"  downloaded {csv}")

    # 2) image shards for the chosen split, e.g. images/test/00.zip
    prefix = f"images/{args.split}/"
    shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".zip"))
    if not shards:
        # fall back: any zip mentioning the split
        shards = sorted(f for f in files if args.split in f and f.endswith(".zip"))
    chosen = shards[: args.shards]
    print(f"found {len(shards)} {args.split} image shard(s); downloading {len(chosen)}: {chosen}")

    for shard in chosen:
        local = hf_hub_download(REPO_ID, filename=shard, repo_type=REPO_TYPE, local_dir=str(out))
        print(f"  downloaded {shard}")
        if not args.no_extract:
            dest = out / "images" / args.split
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(local) as zf:
                zf.extractall(dest)
            print(f"  extracted -> {dest}")

    # quick summary
    img_dir = out / "images" / args.split
    if img_dir.exists():
        n = sum(1 for _ in img_dir.rglob("*.jpg")) + sum(1 for _ in img_dir.rglob("*.png"))
        print(f"\n{n} images present under {img_dir}")
    print("done.")


if __name__ == "__main__":
    main()
