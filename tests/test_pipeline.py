import numpy as np
from src.pipeline import load_demo, run
from src import change as ch
from src.geo import roi_window, class_map_to_geotiff, read_geotiff


def test_end_to_end():
    s = load_demo(size=96)
    for kind in ("Random Forest", "SVM"):
        model, met, maps, ndvi, ext = run(s, kind)
        assert met["accuracy"] > 0.9
        assert ext[2025] > 0.85
    a, b = maps[2015], maps[2025]
    sm = ch.sprawl_summary(a, b, 2015, 2025, s.pixel_km2)
    assert sm["delta_km2"] > 0                      # city grew
    t = ch.transition_matrix(a, b, s.pixel_km2)
    assert abs(t.values.sum() - a.size * s.pixel_km2) < 1e-6


def test_roi_and_geotiff_roundtrip():
    (r0, r1, c0, c1), nb = roi_window((22.6, 88.3, 22.8, 88.5), (100, 100), (22.65, 88.35, 22.75, 88.45))
    assert 0 <= r0 < r1 <= 100 and 0 <= c0 < c1 <= 100
    s = load_demo(size=64)
    raw = class_map_to_geotiff(np.zeros((64, 64), np.uint8), s.bounds)
    assert len(raw) > 0
