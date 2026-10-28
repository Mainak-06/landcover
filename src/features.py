"""Spectral indices and feature stack."""
from __future__ import annotations
import numpy as np

EPS = 1e-6


def nd(a, b):
    return (a - b) / (a + b + EPS)


def indices(bands: np.ndarray) -> dict:
    blue, green, red, nir, swir = bands[:5]
    return {
        "NDVI": nd(nir, red),
        "NDWI": nd(green, nir),
        "NDBI": nd(swir, nir),
        "MNDWI": nd(green, swir),
    }


def feature_stack(bands: np.ndarray) -> np.ndarray:
    """(5,H,W) reflectance -> (H,W,9) features: 5 bands + 4 indices."""
    idx = indices(bands)
    st = np.concatenate([bands[:5], np.stack(list(idx.values()))], axis=0)
    return np.moveaxis(st, 0, -1).astype(np.float32)


def weak_labels(bands: np.ndarray) -> np.ndarray:
    """Rule-based labels from indices, used to bootstrap training when the
    user has no hand-labelled samples. Returns 255 where ambiguous."""
    i = indices(bands)
    lab = np.full(bands.shape[1:], 255, np.uint8)
    water = (i["MNDWI"] > 0.2) & (i["NDVI"] < 0.1)
    veg = (i["NDVI"] > 0.45) & ~water
    built = (i["NDBI"] > 0.05) & (i["NDVI"] < 0.2) & (i["NDVI"] > -0.05) & ~water
    barren = (i["NDVI"] >= 0.0) & (i["NDVI"] < 0.2) & (i["NDBI"] <= 0.05) & ~water
    lab[barren] = 3
    lab[built] = 2
    lab[veg] = 1
    lab[water] = 0
    return lab
