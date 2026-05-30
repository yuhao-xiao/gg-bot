"""Reverse-geocode coordinates to an ISO alpha-2 country via Natural Earth borders.

Used to label images that only have lat/lon (the all-land Google coverage, plus
the free Google datasets) so the auxiliary country head has targets.
"""

from __future__ import annotations

import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.strtree import STRtree


def _iso_a2(attrs: dict) -> str | None:
    for key in ("ISO_A2_EH", "ISO_A2"):
        v = attrs.get(key)
        if v and v != "-99":
            return v
    return None


class CountryLocator:
    """Fast point-in-country lookup using an STRtree over all NE countries."""

    def __init__(self, resolution: str = "50m") -> None:
        path = shpreader.natural_earth(resolution=resolution, category="cultural", name="admin_0_countries")
        self.geoms = []
        self.codes: list[str | None] = []
        for rec in shpreader.Reader(path).records():
            self.geoms.append(rec.geometry)
            self.codes.append(_iso_a2(rec.attributes))
        self.tree = STRtree(self.geoms)

    def country_of(self, lat: float, lon: float) -> str | None:
        p = Point(lon, lat)
        for idx in self.tree.query(p):  # bbox candidates
            if self.geoms[idx].contains(p):
                return self.codes[idx]
        return None

    def countries_of(self, lats, lons) -> list[str | None]:
        return [self.country_of(la, lo) for la, lo in zip(lats, lons)]
