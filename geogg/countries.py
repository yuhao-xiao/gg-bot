"""Load the set of Google-Street-View-covered countries used to scope the data.

The keep-set lives in configs/gsv_countries.txt (editable). This module just
parses it into a set of ISO alpha-2 codes.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_PATH = Path("configs/gsv_countries.txt")


def load_gsv_countries(path: str | Path = DEFAULT_PATH) -> set[str]:
    codes: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()  # drop comments + whitespace
        if line:
            codes.add(line.upper())
    return codes


def country_index(path: str | Path = DEFAULT_PATH) -> dict[str, int]:
    """Stable {ISO-alpha2 -> class index} for the auxiliary country head.
    Index len(countries) is reserved as 'UNK' for anything outside the list."""
    codes = sorted(load_gsv_countries(path))
    idx = {c: i for i, c in enumerate(codes)}
    idx["UNK"] = len(codes)
    return idx

