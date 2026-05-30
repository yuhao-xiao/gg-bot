"""Minimal client for the Google Street View *metadata* endpoint.

IMPORTANT: this module ONLY ever calls the metadata endpoint, which is FREE
(no charge, no quota cost for the image SKU). It NEVER calls the billable static
image endpoint. Keep it that way.

Metadata response of interest:
    status:    "OK" if a panorama exists near the query point, else ZERO_RESULTS / etc.
    location:  the snapped {lat, lng} of the actual panorama
    pano_id:   stable panorama id (used to dedupe)
    date:      capture month, e.g. "2021-05"
    copyright: "© 2021 Google" for official car coverage; other text for user photo spheres
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"


@dataclass
class PanoMeta:
    status: str
    lat: float | None = None
    lon: float | None = None
    pano_id: str | None = None
    date: str | None = None
    copyright: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    @property
    def is_google_official(self) -> bool:
        """True for Google-captured car coverage (what GeoGuessr's official maps use)."""
        return bool(self.copyright) and "google" in self.copyright.lower()


def query_metadata(
    session: requests.Session,
    lat: float,
    lon: float,
    api_key: str,
    radius: int = 1000,
    source: str = "outdoor",
    timeout: float = 10.0,
) -> PanoMeta:
    """Look up Street View coverage near (lat, lon). Free metadata call only."""
    params = {
        "location": f"{lat:.6f},{lon:.6f}",
        "key": api_key,
        "radius": radius,
        "source": source,  # 'outdoor' restricts to outdoor collections
    }
    resp = session.get(METADATA_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    loc = data.get("location") or {}
    return PanoMeta(
        status=data.get("status", "UNKNOWN"),
        lat=loc.get("lat"),
        lon=loc.get("lng"),
        pano_id=data.get("pano_id"),
        date=data.get("date"),
        copyright=data.get("copyright"),
    )
