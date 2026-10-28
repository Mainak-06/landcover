"""Change-detection analytics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import CLASS_NAMES, NODATA_CLASS

BUILT, VEG = 2, 1


def area_table(class_map, pixel_area_km2: float) -> pd.Series:
    valid = class_map != NODATA_CLASS
    counts = np.bincount(class_map[valid].ravel(), minlength=len(CLASS_NAMES))[:len(CLASS_NAMES)]
    return pd.Series(counts * pixel_area_km2, index=[CLASS_NAMES[i] for i in range(len(CLASS_NAMES))])


def transition_matrix(a, b, pixel_area_km2: float) -> pd.DataFrame:
    m = (a != NODATA_CLASS) & (b != NODATA_CLASS)
    n = len(CLASS_NAMES)
    mat = np.zeros((n, n))
    np.add.at(mat, (a[m].astype(int), b[m].astype(int)), 1)
    names = [CLASS_NAMES[i] for i in range(n)]
    return pd.DataFrame(mat * pixel_area_km2, index=[f"{x} (from)" for x in names],
                        columns=[f"{x} (to)" for x in names])


def change_map(a, b) -> np.ndarray:
    """0=no change, 1=urban expansion, 2=vegetation loss (non-urban), 3=vegetation gain, 4=other change, 255=nodata."""
    out = np.zeros(a.shape, np.uint8)
    changed = a != b
    out[changed] = 4
    out[(b == BUILT) & (a != BUILT)] = 1
    out[(a == VEG) & (b != VEG) & (b != BUILT)] = 2
    out[(b == VEG) & (a != VEG)] = 3
    out[(a == NODATA_CLASS) | (b == NODATA_CLASS)] = NODATA_CLASS
    return out


CHANGE_NAMES = {0: "No change", 1: "Urban expansion", 2: "Vegetation loss", 3: "Vegetation gain", 4: "Other change"}
CHANGE_COLORS = {0: None, 1: "#e31a1c", 2: "#ff7f00", 3: "#33a02c", 4: "#999999"}


def change_rgba(cm: np.ndarray) -> np.ndarray:
    out = np.zeros(cm.shape + (4,), np.uint8)
    for k, hexc in CHANGE_COLORS.items():
        if hexc:
            out[cm == k] = [int(hexc[i:i + 2], 16) for i in (1, 3, 5)] + [210]
    return out


def sprawl_summary(a, b, year_a, year_b, pixel_area_km2) -> dict:
    ba, bb = area_table(a, pixel_area_km2)["Built-up"], area_table(b, pixel_area_km2)["Built-up"]
    yrs = max(year_b - year_a, 1)
    cagr = ((bb / ba) ** (1 / yrs) - 1) * 100 if ba > 0 and bb > 0 else float("nan")
    valid = max(int(((a != NODATA_CLASS) & (b != NODATA_CLASS)).sum()), 1)
    new_urban = int(((a != BUILT) & (b == BUILT) & (a != NODATA_CLASS)).sum())
    return {"built_a": ba, "built_b": bb, "delta_km2": bb - ba,
            "pct_growth": (bb - ba) / ba * 100 if ba > 0 else float("nan"),
            "cagr_pct": cagr, "rate_km2_yr": (bb - ba) / yrs,
            "new_urban_km2": new_urban * pixel_area_km2,
            "changed_pct": float((a != b)[(a != NODATA_CLASS) & (b != NODATA_CLASS)].sum() / valid * 100)}


def ndvi_stats(ndvi_a, ndvi_b) -> dict:
    d = ndvi_b - ndvi_a
    return {"mean_a": float(np.nanmean(ndvi_a)), "mean_b": float(np.nanmean(ndvi_b)),
            "mean_delta": float(np.nanmean(d)),
            "pct_degraded": float((d < -0.2).mean() * 100), "pct_improved": float((d > 0.2).mean() * 100)}
