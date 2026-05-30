"""One-shot fix: move blalexa crops that live in the flat exFAT folder into
subdir-sharded subfolders (still on E:) and update the parquets to point there.

The problem isn't exFAT itself -- it's a single directory with 100k+ files.
Splitting into ~256 small subfolders (by pano-id prefix) makes lookups fast.
Stays on the same filesystem so each move is just a rename (no data copy).

Layout:
    E:/gg-data/blalexa_perspective/<filename>.jpg          (source, flat)
        ->  E:/gg-data/blalexa_perspective/legacy/<2char>/<filename>.jpg

Usage:
    python scripts/fix_blalexa_flat.py             # dry run (counts only)
    python scripts/fix_blalexa_flat.py --do-it     # actually move + update
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from geogg.paths import DATA_ROOT, INDEX_DIR

SRC_ROOT = DATA_ROOT / "blalexa_perspective"
DST_ROOT = SRC_ROOT / "legacy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--do-it", action="store_true", help="actually move + update parquets")
    args = ap.parse_args()

    print(f"scanning top-level of {SRC_ROOT} (one exFAT dir scan) ...")
    flat_files = [p for p in SRC_ROOT.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"]
    print(f"found {len(flat_files):,} flat-folder crops to relocate")

    if not flat_files:
        print("nothing to do.")
        return

    if not args.do_it:
        print("\nDRY RUN. Pass --do-it to actually move + update parquets.")
        return

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    flat_to_new: dict[str, str] = {}

    for src in tqdm(flat_files, desc="moving"):
        pano_id = src.stem.rsplit("_", 1)[0]
        subdir = DST_ROOT / pano_id[:2]
        subdir.mkdir(exist_ok=True)
        dst = subdir / src.name
        if dst.exists():  # idempotent: skip if already moved
            flat_to_new[str(src)] = str(dst)
            continue
        shutil.move(str(src), str(dst))  # same-filesystem rename -> fast
        flat_to_new[str(src)] = str(dst)

    print(f"\nmoved {len(flat_to_new):,} files into {DST_ROOT}")

    # update the blalexa parquets to point to the new locations
    for name in ("blalexa", "blalexa_train", "blalexa_test"):
        p = INDEX_DIR / f"{name}.parquet"
        if not p.exists():
            print(f"  skip {name}: parquet not found")
            continue
        df = pd.read_parquet(p)
        df["path"] = df["path"].map(lambda x: flat_to_new.get(x, x))
        n_in_legacy = int(df["path"].str.contains(r"\\legacy\\|/legacy/", regex=True).sum())
        df.to_parquet(p, index=False)
        print(f"  {name}: {len(df):,} rows | {n_in_legacy:,} now point to legacy/...")

    print("\ndone. flat top-level folder is now (mostly) empty; the parquets ignore it.")


if __name__ == "__main__":
    main()
