"""Shared constants: class definitions, colours, band order."""
import numpy as np

CLASS_NAMES = {0: "Water", 1: "Vegetation", 2: "Built-up", 3: "Barren"}
CLASS_COLORS = {0: "#1f78b4", 1: "#33a02c", 2: "#e31a1c", 3: "#e6c35c"}
N_CLASSES = len(CLASS_NAMES)
# Band order used internally: Blue, Green, Red, NIR, SWIR1 (Sentinel-2 B2,B3,B4,B8,B11)
BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1"]
NODATA_CLASS = 255


def colour_rgb(class_map: np.ndarray, alpha: int = 255) -> np.ndarray:
    """Convert a class map to an RGBA uint8 image (nodata is transparent)."""
    out = np.zeros(class_map.shape + (4,), np.uint8)
    for k, hexc in CLASS_COLORS.items():
        rgb = [int(hexc[i:i + 2], 16) for i in (1, 3, 5)]
        m = class_map == k
        out[m, :3] = rgb
        out[m, 3] = alpha
    return out
