"""Random pano-disjoint split of blalexa for held-out Google-domain test set.

Splits by pano_id (not by row) so the 4 crops of a single location stay together
on the same side of train/test. Writes blalexa_train.parquet and blalexa_test.parquet.

Usage:
    python scripts/split_blalexa.py --test-panos 2000   # ~8k test crops
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from geogg.paths import INDEX_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-panos", type=int, default=2000, help="number of panos to hold out (~4 crops each)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = INDEX_DIR / "blalexa.parquet"
    df = pd.read_parquet(src)
    print(f"loaded {len(df):,} crops from {df['pano_id'].nunique():,} panos")

    rng = np.random.default_rng(args.seed)
    all_panos = df["pano_id"].unique()
    rng.shuffle(all_panos)
    test_panos = set(all_panos[: args.test_panos])

    is_test = df["pano_id"].isin(test_panos)
    train_df = df.loc[~is_test].reset_index(drop=True)
    test_df = df.loc[is_test].reset_index(drop=True)

    train_df.to_parquet(INDEX_DIR / "blalexa_train.parquet", index=False)
    test_df.to_parquet(INDEX_DIR / "blalexa_test.parquet", index=False)

    print(f"train: {len(train_df):,} crops | {train_df['pano_id'].nunique():,} panos | {train_df['cell_index'].nunique()} cells")
    print(f"test : {len(test_df):,} crops | {test_df['pano_id'].nunique():,} panos | {test_df['cell_index'].nunique()} cells")


if __name__ == "__main__":
    main()
