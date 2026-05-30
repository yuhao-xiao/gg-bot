"""Configurable storage locations.

Big raw data (images, OSV-5M CSVs, coverage parquet) lives under DATA_ROOT, which
you can point at an external drive via the GG_DATA_ROOT env var (or a line in
.env, e.g. GG_DATA_ROOT=E:/gg-data). Small derived artifacts (embeddings, grid,
checkpoints) live under ARTIFACTS_ROOT, best kept on a fast local drive.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = Path(os.environ.get("GG_DATA_ROOT", "datasets"))
ARTIFACTS_ROOT = Path(os.environ.get("GG_ARTIFACTS_ROOT", "artifacts"))

OSV5M_DIR = DATA_ROOT / "osv5m"
# canonical coverage = the 416k all-land sample used to build the locked grid
GOOGLE_COVERAGE_PARQUET = DATA_ROOT / "google_coverage_land.parquet"

# locked geocell grid: all-land Google coverage, 5,880 cells, bands 70/145/290 km
GRID_PATH = ARTIFACTS_ROOT / "grid_final" / "grid.json"

INDEX_DIR = ARTIFACTS_ROOT / "index"
EMB_DIR = ARTIFACTS_ROOT / "embeddings"
MODELS_DIR = ARTIFACTS_ROOT / "models"
