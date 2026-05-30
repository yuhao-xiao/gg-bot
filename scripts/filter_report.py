"""Report how the OSV-5M points split under our GSV country keep-set.

Shows: kept countries (with image counts), countries EXCLUDED but present in
OSV-5M with notable counts (so you can catch false exclusions), and GSV-list
countries ABSENT from OSV-5M (coverage gaps). Edit configs/gsv_countries.txt
and re-run to converge.

Usage:
    python scripts/filter_report.py
"""

from __future__ import annotations

import pandas as pd

from geogg.countries import load_gsv_countries

META = "datasets/osv5m/train.csv"
EXCLUDE_REPORT_MIN = 5000  # only flag excluded countries with at least this many images


def main() -> None:
    keep = load_gsv_countries()
    counts = pd.read_csv(META, usecols=["country"])["country"].value_counts()

    present = set(counts.index)
    kept = sorted((c for c in counts.index if c in keep), key=lambda c: -counts[c])
    excluded = sorted((c for c in counts.index if c not in keep), key=lambda c: -counts[c])
    missing = sorted(keep - present)

    kept_rows = int(counts[kept].sum())
    total_rows = int(counts.sum())

    print(f"GSV keep-set size: {len(keep)} countries")
    print(f"OSV-5M countries present: {len(present)}")
    print(
        f"\nKEPT: {len(kept)} countries | {kept_rows:,} images "
        f"({100*kept_rows/total_rows:.1f}% of {total_rows:,})"
    )
    for c in kept:
        print(f"  {c}  {counts[c]:>9,}")

    print(f"\nEXCLUDED but present with >= {EXCLUDE_REPORT_MIN:,} images "
          f"(verify none of these should be kept):")
    for c in excluded:
        if counts[c] >= EXCLUDE_REPORT_MIN:
            print(f"  {c}  {counts[c]:>9,}")

    print(f"\nIn GSV list but ABSENT from OSV-5M ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
