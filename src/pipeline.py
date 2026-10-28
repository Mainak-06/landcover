"""Glue: load scenes -> train -> classify every year."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .synthetic import make_timeseries, DEFAULT_BOUNDS
from .geo import grid_pixel_area_km2
from .classify import train, predict_map
from .features import indices


@dataclass
class Scenes:
    years: list
    bands: dict
    bounds: tuple
    pixel_km2: float
    valid: dict
    truth: dict = field(default_factory=dict)
    source: str = "demo"


def load_demo(years=(2015, 2020, 2025), size=256) -> Scenes:
    ts = make_timeseries(years, size)
    shape = next(iter(ts.values()))[0].shape[1:]
    return Scenes(list(years), {y: v[0] for y, v in ts.items()}, DEFAULT_BOUNDS,
                  grid_pixel_area_km2(DEFAULT_BOUNDS, shape),
                  {y: np.ones(shape, bool) for y in years}, {y: v[1] for y, v in ts.items()}, "demo")


def load_uploads(items, band_order, scale) -> Scenes:
    """items: list of (year, bytes). All rasters must share grid size."""
    from .geo import read_geotiff
    items = sorted(items, key=lambda t: t[0])
    if len({y for y, _ in items}) != len(items):
        raise ValueError("Each GeoTIFF needs a unique year.")
    bands, valid, ref = {}, {}, None
    for y, raw in items:
        b, bounds, px, v = read_geotiff(raw, band_order, scale)
        if ref is None:
            ref = (b.shape[1:], bounds, px)
        elif b.shape[1:] != ref[0]:
            raise ValueError(f"Year {y} has size {b.shape[1:]} but first raster is {ref[0]}. "
                             "Co-register / resample all images to the same grid first.")
        bands[y], valid[y] = b, v
    return Scenes([y for y, _ in items], bands, ref[1], ref[2], valid, {}, "upload")


def run(scenes: Scenes, model_kind: str, train_year=None):
    """Train on one reference year (weak labels) and classify all years."""
    ty = train_year or scenes.years[0]
    model, metrics = train(scenes.bands[ty], model_kind)
    maps = {y: predict_map(model, scenes.bands[y], scenes.valid[y]) for y in scenes.years}
    ndvi = {y: np.where(scenes.valid[y], indices(scenes.bands[y])["NDVI"], np.nan) for y in scenes.years}
    # honest external score in demo mode: compare against hidden ground truth
    ext = {}
    for y, t in scenes.truth.items():
        ext[y] = float((maps[y] == t).mean())
    return model, metrics, maps, ndvi, ext
