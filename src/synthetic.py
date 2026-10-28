"""Synthetic Sentinel-2-like multi-temporal scene generator (no download needed).

Urban area grows outward from a city core, vegetation is lost at the fringe,
and a river + lake stay fixed. Ground truth labels are returned so the
classifier can be scored honestly.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter

# mean reflectance (Blue, Green, Red, NIR, SWIR1) per class
_SIG = {
    0: (0.06, 0.07, 0.05, 0.03, 0.02),   # water
    1: (0.04, 0.08, 0.04, 0.45, 0.20),   # vegetation
    2: (0.12, 0.14, 0.17, 0.22, 0.32),   # built-up
    3: (0.15, 0.20, 0.27, 0.33, 0.30),   # barren
}
# Default scene: peri-urban Kolkata, West Bengal (illustrative location only)
DEFAULT_BOUNDS = (22.60, 88.30, 22.80, 88.50)  # south, west, north, east


def _truth(size: int, year: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    noise = gaussian_filter(rng.normal(size=(size, size)), size / 18)
    noise = noise / np.abs(noise).max()
    lab = np.full((size, size), 1, np.uint8)                     # vegetation base
    barren = gaussian_filter(rng.normal(size=(size, size)), size / 12) > 0.015
    lab[barren] = 3
    t = (year - 2015) / 10.0
    radius = 0.14 + 0.20 * t                                      # city growth
    d = np.hypot(yy - 0.55, xx - 0.45) + 0.08 * noise
    lab[d < radius] = 2
    sat = np.hypot(yy - 0.2, xx - 0.8) + 0.06 * noise             # satellite town
    lab[sat < 0.04 + 0.07 * t] = 2
    river = np.abs(xx - (0.15 + 0.7 * yy) + 0.04 * np.sin(yy * 14)) < 0.012
    lake = np.hypot(yy - 0.8, xx - 0.25) < 0.07
    lab[river | lake] = 0
    return lab


def make_scene(size: int = 256, year: int = 2015, seed: int = 7):
    """Return (bands[5,H,W] float32, truth[H,W] uint8)."""
    truth = _truth(size, year, seed)
    rng = np.random.default_rng(seed * 100 + year)
    bands = np.zeros((5, size, size), np.float32)
    season = 1.0 + 0.04 * np.sin(year)                            # slight inter-year shift
    for k, sig in _SIG.items():
        m = truth == k
        for b in range(5):
            bands[b][m] = sig[b]
    texture = gaussian_filter(rng.normal(size=(5, size, size)), (0, 1.2, 1.2)) * 0.05
    bands = bands * season * (1 + texture) + rng.normal(0, 0.012, bands.shape)
    return np.clip(bands, 0.001, 1.0).astype(np.float32), truth


def make_timeseries(years=(2015, 2020, 2025), size=256, seed=7):
    return {y: make_scene(size, y, seed) for y in years}
