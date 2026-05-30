"""Client for the Google Street View STATIC IMAGE endpoint.

WARNING: this endpoint is BILLABLE (~$0.007 per image). It is intentionally kept
in a separate module from the free metadata client (geogg.google_sv) so the
billable code path is easy to audit. Callers must cap how many images they fetch.
"""

from __future__ import annotations

import requests

STATIC_URL = "https://maps.googleapis.com/maps/api/streetview"
PRICE_PER_IMAGE_USD = 0.007


def fetch_static_image(
    session: requests.Session,
    pano_id: str,
    api_key: str,
    heading: float = 0.0,
    size: str = "640x640",
    fov: int = 90,
    pitch: int = 0,
    timeout: float = 15.0,
) -> bytes:
    """Download one Street View image for a specific panorama id. BILLABLE."""
    params = {
        "size": size,
        "pano": pano_id,
        "heading": f"{heading:.1f}",
        "fov": fov,
        "pitch": pitch,
        "key": api_key,
        "return_error_code": "true",  # 4xx instead of a grey 'no imagery' placeholder
    }
    resp = session.get(STATIC_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.content
