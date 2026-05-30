"""Load OSV-5M metadata, optionally filtered to the GSV country keep-set."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from geogg.countries import load_gsv_countries
from geogg.paths import OSV5M_DIR

META_PATHS = {
    "train": OSV5M_DIR / "train.csv",
    "test": OSV5M_DIR / "test.csv",
}


def load_metadata(
    split: str = "train",
    columns: Sequence[str] | None = None,
    gsv_only: bool = True,
) -> pd.DataFrame:
    cols = list(columns) if columns is not None else None
    if cols is not None and gsv_only and "country" not in cols:
        cols = cols + ["country"]

    df = pd.read_csv(META_PATHS[split], usecols=cols, low_memory=False)
    if gsv_only:
        keep = load_gsv_countries()
        df = df[df["country"].isin(keep)].reset_index(drop=True)
    return df
