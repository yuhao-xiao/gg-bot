"""Sample random coordinates that fall on land inside GSV-covered countries.

Used to feed the Google metadata sampler efficient query points (so we don't
waste calls on oceans / non-covered countries). Country polygons come from
Natural Earth via cartopy (auto-downloaded + cached on first use).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.prepared import PreparedGeometry, prep

from geogg.countries import load_gsv_countries


@dataclass
class _Country:
    code: str
    bounds: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    prepared: PreparedGeometry
    area: float


def _iso_a2(attrs: dict) -> str | None:
    for key in ("ISO_A2_EH", "ISO_A2"):
        v = attrs.get(key)
        if v and v != "-99":
            return v
    return None


def load_all_land_geom(resolution: str = "50m", drop_antarctica: bool = True) -> list[_Country]:
    """All land polygons (no country filter) so Google's metadata defines coverage.
    Splits the land multipolygon into landmasses/islands and drops Antarctica."""
    path = shpreader.natural_earth(resolution=resolution, category="physical", name="land")
    out: list[_Country] = []
    for rec in shpreader.Reader(path).records():
        geom = rec.geometry
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            if drop_antarctica and poly.bounds[3] < -55:  # maxy below ~Antarctica
                continue
            out.append(_Country("", poly.bounds, prep(poly), poly.area))
    print(f"  loaded {len(out)} land polygons (Antarctica dropped={drop_antarctica})")
    return out


def load_gsv_countries_geom(resolution: str = "50m") -> list[_Country]:
    keep = load_gsv_countries()
    path = shpreader.natural_earth(resolution=resolution, category="cultural", name="admin_0_countries")
    out: list[_Country] = []
    matched: set[str] = set()
    for rec in shpreader.Reader(path).records():
        code = _iso_a2(rec.attributes)
        if code in keep:
            geom = rec.geometry
            out.append(_Country(code, geom.bounds, prep(geom), geom.area))
            matched.add(code)
    missing = keep - matched
    if missing:
        print(f"  note: {len(missing)} GSV codes had no polygon at {resolution}: {sorted(missing)}")
    return out


class LandSampler:
    """Draw random (lat, lon, country) points uniformly-ish over GSV land area."""

    def __init__(self, countries: list[_Country], max_attempts: int = 50) -> None:
        self.countries = countries
        self.weights = [c.area for c in countries]  # area-weighted country choice
        self.max_attempts = max_attempts

    def sample(self) -> tuple[float, float, str]:
        c = random.choices(self.countries, weights=self.weights, k=1)[0]
        minx, miny, maxx, maxy = c.bounds
        for _ in range(self.max_attempts):
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            if c.prepared.contains(Point(x, y)):
                return y, x, c.code  # lat, lon, country
        return (miny + maxy) / 2, (minx + maxx) / 2, c.code  # fallback: bbox centre
