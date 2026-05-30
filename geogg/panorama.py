"""Convert equirectangular 360 panoramas into perspective crops that match the
Google Static API / GeoGuessr view (square, ~90 FOV, level pitch). This lets us
use panoramic datasets in the same pipeline as our perspective images.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import py360convert
from PIL import Image


def pano_to_perspectives(
    pano: Image.Image,
    headings: Sequence[float] = (0, 90, 180, 270),
    fov: float = 90.0,
    size: int = 640,
    pitch: float = 0.0,
    max_src_width: int = 0,
) -> list[Image.Image]:
    """Return one perspective crop per heading from an equirectangular panorama.

    max_src_width: if set, downscale the equirectangular first (faster projection
    + less memory). ~4x the crop size keeps full detail for the output FOV."""
    pano = pano.convert("RGB")
    if max_src_width and pano.width > max_src_width:
        pano = pano.resize((max_src_width, max_src_width // 2), Image.BILINEAR)
    arr = np.asarray(pano)
    crops = []
    for h in headings:
        persp = py360convert.e2p(arr, fov_deg=fov, u_deg=float(h), v_deg=pitch, out_hw=(size, size))
        crops.append(Image.fromarray(persp.astype(np.uint8)))
    return crops
