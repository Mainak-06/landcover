"""Raster I/O (GeoTIFF via rasterio) and ROI helpers."""
from __future__ import annotations
import io
import numpy as np


def read_geotiff(file_bytes: bytes, band_order=(1, 2, 3, 4, 5), scale: float | None = None):
    """Read a multi-band GeoTIFF. band_order = 1-based file bands for Blue,Green,Red,NIR,SWIR1.
    Returns (bands[5,H,W] float32 reflectance 0-1, bounds(south,west,north,east) in WGS84, pixel_area_km2, valid mask)."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds
    with MemoryFile(file_bytes) as mf, mf.open() as src:
        if src.count < max(band_order):
            raise ValueError(f"File has {src.count} bands but band order needs {max(band_order)}.")
        arr = src.read(list(band_order)).astype(np.float32)
        nod = src.nodata
        valid = np.isfinite(arr).all(0)
        if nod is not None:
            valid &= ~(arr == nod).any(0)
        if scale is None:  # auto: Sentinel-2 L2A is 0-10000
            scale = 10000.0 if np.nanmax(arr) > 2 else 1.0
        arr = np.clip(arr / scale, 0.0, 1.0)
        if src.crs is None:
            raise ValueError("GeoTIFF has no CRS; cannot place it on a map.")
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        if src.crs.is_projected:
            px_km2 = abs(src.res[0] * src.res[1]) / 1e6
        else:
            lat = (s + n) / 2
            px_km2 = abs(src.res[0] * 111.32 * np.cos(np.radians(lat)) * src.res[1] * 110.57)
    return arr, (s, w, n, e), px_km2, valid


def grid_pixel_area_km2(bounds, shape) -> float:
    s, w, n, e = bounds
    h, wd = shape
    lat = (s + n) / 2
    return ((n - s) * 110.57 / h) * ((e - w) * 111.32 * np.cos(np.radians(lat)) / wd)


def roi_window(bounds, shape, roi_bounds):
    """Convert (south,west,north,east) ROI to pixel slice (r0,r1,c0,c1) and snapped bounds."""
    s, w, n, e = bounds
    h, wd = shape
    rs, rw, rn, re_ = roi_bounds
    c0 = int(np.clip(np.floor((rw - w) / (e - w) * wd), 0, wd - 1))
    c1 = int(np.clip(np.ceil((re_ - w) / (e - w) * wd), c0 + 1, wd))
    r0 = int(np.clip(np.floor((n - rn) / (n - s) * h), 0, h - 1))
    r1 = int(np.clip(np.ceil((n - rs) / (n - s) * h), r0 + 1, h))
    nb = (n - r1 * (n - s) / h, w + c0 * (e - w) / wd, n - r0 * (n - s) / h, w + c1 * (e - w) / wd)
    return (r0, r1, c0, c1), nb


def class_map_to_geotiff(arr: np.ndarray, bounds) -> bytes:
    """Serialise a uint8 raster to an in-memory GeoTIFF (EPSG:4326) for download."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
    s, w, n, e = bounds
    h, wd = arr.shape
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=h, width=wd, count=1, dtype="uint8", crs="EPSG:4326",
                     transform=from_bounds(w, s, e, n, wd, h), nodata=255) as dst:
            dst.write(arr.astype(np.uint8), 1)
        return mf.read()
