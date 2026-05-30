"""Geolocation metrics: great-circle distance, threshold accuracies, GeoGuessr score."""

from __future__ import annotations

import numpy as np

EARTH_R_KM = 6371.0

# Standard geolocation accuracy thresholds (km): street, city, region, country, continent
THRESHOLDS_KM = {"street_1km": 1, "city_25km": 25, "region_200km": 200, "country_750km": 750, "cont_2500km": 2500}


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def geoguessr_score(dist_km) -> np.ndarray:
    """Approximate GeoGuessr world-map score (max 5000, ~halves every ~1000 km)."""
    return 5000.0 * np.exp(-np.asarray(dist_km) / 1492.7)


def threshold_accuracies(dist_km: np.ndarray) -> dict[str, float]:
    return {name: float((dist_km <= km).mean()) for name, km in THRESHOLDS_KM.items()}


def haversine_soft_target_matrix(lats, lons, tau_km: float = 1000.0, epsilon: float = 0.1) -> np.ndarray:
    """Per-cell soft target matrix for Haversine label smoothing.

    Row c gives the smoothed target distribution when the true class is cell c:
      soft[c] = (1 - eps) * onehot(c) + eps * softmax(-d(c, .)^2 / tau^2)

    Near cells get a meaningful fraction of the smoothing mass; far cells get
    almost none. Returns (N, N) float32.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    n = len(lats)
    p1 = np.radians(lats)[:, None]
    p2 = np.radians(lats)[None, :]
    dphi = p2 - p1
    dlmb = np.radians(lons)[None, :] - np.radians(lons)[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    d_km = 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))  # (N, N)

    w = np.exp(-(d_km / float(tau_km)) ** 2)  # gaussian decay
    p = w / w.sum(axis=1, keepdims=True)
    eye = np.eye(n)
    soft = (1.0 - epsilon) * eye + epsilon * p
    return soft.astype(np.float32)
