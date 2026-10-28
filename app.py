"""AI-Powered Land Cover & Urban Sprawl Change Detection - Streamlit dashboard."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from folium import plugins
from matplotlib import colormaps
from streamlit_folium import st_folium

from src.config import CLASS_NAMES, CLASS_COLORS, colour_rgb, NODATA_CLASS
from src import change as ch
from src.geo import roi_window, class_map_to_geotiff
from src.pipeline import load_demo, load_uploads, run
from src.classify import feature_importance

st.set_page_config(page_title="Land Cover & Urban Sprawl Monitor", page_icon="🛰️", layout="wide")


# ------------------------------------------------------------------ cached compute
@st.cache_data(show_spinner="Generating demo scene…")
def _demo(size):
    return load_demo(size=size)


@st.cache_data(show_spinner="Reading GeoTIFFs…")
def _uploads(items, band_order, scale):
    return load_uploads(list(items), band_order, scale)


@st.cache_data(show_spinner="Training classifier & mapping all years…")
def _run(_scenes, key, model_kind):
    return run(_scenes, model_kind)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛰️ Controls")
mode = st.sidebar.radio("Data source", ["Demo (synthetic Sentinel-2)", "Upload GeoTIFFs"])
model_kind = st.sidebar.selectbox("Classifier", ["Random Forest", "SVM"])
scenes, key = None, None
try:
    if mode.startswith("Demo"):
        size = st.sidebar.select_slider("Scene size (pixels)", [128, 192, 256, 384], 256)
        scenes, key = _demo(size), f"demo{size}"
    else:
        st.sidebar.caption("Upload ≥2 multi-band GeoTIFFs (same grid) of the same area, one per year.")
        files = st.sidebar.file_uploader("GeoTIFFs", type=["tif", "tiff"], accept_multiple_files=True)
        order = st.sidebar.text_input("Band order in file: Blue,Green,Red,NIR,SWIR1", "2,3,4,8,11")
        scale = st.sidebar.number_input("Reflectance scale (0 = auto)", 0.0, 100000.0, 0.0)
        if files and len(files) >= 2:
            yrs = [int(st.sidebar.number_input(f"Year of {f.name}", 1980, 2100, 2018 + 3 * i, key=f"y{i}"))
                   for i, f in enumerate(files)]
            bo = tuple(int(x) for x in order.split(","))
            if len(bo) != 5:
                raise ValueError("Band order needs exactly 5 numbers.")
            items = tuple((y, f.getvalue()) for y, f in zip(yrs, files))
            scenes, key = _uploads(items, bo, scale or None), f"up{hash(items)}"
        else:
            st.info("Upload at least two GeoTIFFs in the sidebar, or switch to the demo data.")
            st.stop()
except Exception as e:  # show friendly error, never a traceback
    st.error(f"Could not load data: {e}")
    st.stop()

model, metrics, maps, ndvi, ext = _run(scenes, key, model_kind)

ya = st.sidebar.selectbox("From year", scenes.years, 0)
yb = st.sidebar.selectbox("To year", scenes.years, len(scenes.years) - 1)
if ya >= yb:
    st.warning("Choose a 'From' year earlier than the 'To' year.")
    st.stop()

# ROI
st.sidebar.subheader("Region of interest")
H, W = maps[ya].shape
roi_mode = st.sidebar.radio("ROI", ["Full scene", "Draw on map"], horizontal=True)
if roi_mode == "Draw on map" and "roi" in st.session_state:
    (r0, r1, c0, c1), rb = roi_window(scenes.bounds, (H, W), st.session_state["roi"])
elif roi_mode == "Draw on map":
    st.sidebar.info("Use the rectangle tool on the map tab to draw your ROI.")
    (r0, r1, c0, c1), rb = (0, H, 0, W), scenes.bounds
else:
    (r0, r1, c0, c1), rb = (0, H, 0, W), scenes.bounds
if st.sidebar.button("Reset ROI"):
    st.session_state.pop("roi", None)
    st.rerun()

A, B = maps[ya][r0:r1, c0:c1], maps[yb][r0:r1, c0:c1]
NA, NB = ndvi[ya][r0:r1, c0:c1], ndvi[yb][r0:r1, c0:c1]
pxa = scenes.pixel_km2
summ = ch.sprawl_summary(A, B, ya, yb, pxa)
cm = ch.change_map(A, B)

# ------------------------------------------------------------------ header & KPIs
st.title("Land Cover & Urban Sprawl Change Detection")
st.caption(f"{'Synthetic demo scene (illustrative location)' if scenes.source == 'demo' else 'User-supplied imagery'} · "
           f"{ya} → {yb} · ROI {(r1 - r0) * (c1 - c0) * pxa:,.1f} km² · {model_kind}")
k = st.columns(4)
k[0].metric(f"Built-up {yb}", f"{summ['built_b']:.2f} km²", f"{summ['delta_km2']:+.2f} km²")
k[1].metric("Urban growth", f"{summ['pct_growth']:+.1f} %", f"{summ['cagr_pct']:.2f} % / yr CAGR")
k[2].metric("Mean NDVI change", f"{np.nanmean(NB) - np.nanmean(NA):+.3f}", f"{np.nanmean(NB):.3f} now", delta_color="normal")
k[3].metric("Area changed", f"{summ['changed_pct']:.1f} %")

tab_map, tab_stats, tab_change, tab_model, tab_about = st.tabs(
    ["🗺️ Map", "📊 Land-cover stats", "🔁 Change analysis", "🤖 Model quality", "ℹ️ About"])

# ------------------------------------------------------------------ map
with tab_map:
    s, w, n, e = scenes.bounds
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=11, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer("Esri.WorldImagery", name="Satellite basemap").add_to(m)
    bounds_ll = [[s, w], [n, e]]
    for yr in (ya, yb):
        folium.raster_layers.ImageOverlay(colour_rgb(maps[yr], 200), bounds_ll, name=f"Land cover {yr}",
                                          show=(yr == yb), opacity=0.85).add_to(m)
    full_cm = ch.change_map(maps[ya], maps[yb])
    folium.raster_layers.ImageOverlay(ch.change_rgba(full_cm), bounds_ll, name="Change map", show=False).add_to(m)
    d = np.where(np.isfinite(ndvi[yb] - ndvi[ya]), ndvi[yb] - ndvi[ya], 0)
    rgba = (colormaps["RdYlGn"]((np.clip(d, -0.6, 0.6) + 0.6) / 1.2) * 255).astype(np.uint8)
    rgba[..., 3] = (np.abs(d) > 0.1) * 190
    folium.raster_layers.ImageOverlay(rgba, bounds_ll, name="NDVI difference", show=False).add_to(m)
    rr, cc = np.nonzero(full_cm == 1)
    if rr.size:
        step = max(1, rr.size // 6000)
        lat = n - (rr[::step] + 0.5) / H * (n - s)
        lon = w + (cc[::step] + 0.5) / W * (e - w)
        plugins.HeatMap(list(zip(lat, lon)), name="Urban-expansion heatmap", radius=14, blur=18, show=False).add_to(m)
    if roi_mode == "Draw on map":
        plugins.Draw(export=False, draw_options={"rectangle": True, "polyline": False, "polygon": False,
                                                 "circle": False, "marker": False, "circlemarker": False}).add_to(m)
    if rb != scenes.bounds:
        folium.Rectangle([[rb[0], rb[1]], [rb[2], rb[3]]], color="#000", weight=2, fill=False, tooltip="ROI").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    out = st_folium(m, height=560, use_container_width=True, returned_objects=["last_active_drawing"])
    drw = (out or {}).get("last_active_drawing")
    if roi_mode == "Draw on map" and drw and drw.get("geometry", {}).get("coordinates"):
        pts = np.array(drw["geometry"]["coordinates"][0])
        roi = (float(pts[:, 1].min()), float(pts[:, 0].min()), float(pts[:, 1].max()), float(pts[:, 0].max()))
        if st.session_state.get("roi") != roi:
            st.session_state["roi"] = roi
            st.rerun()
    lg = " ".join(f"<span style='color:{CLASS_COLORS[i]}'>■</span> {CLASS_NAMES[i]}" for i in CLASS_NAMES)
    st.markdown(lg + " &nbsp;|&nbsp; Change: <span style='color:#e31a1c'>■</span> urban expansion "
                "<span style='color:#ff7f00'>■</span> veg. loss <span style='color:#33a02c'>■</span> veg. gain",
                unsafe_allow_html=True)

# ------------------------------------------------------------------ stats
with tab_stats:
    rows = {y: ch.area_table(maps[y][r0:r1, c0:c1], pxa) for y in scenes.years}
    df = pd.DataFrame(rows).T
    df.index.name = "Year"
    c1_, c2_ = st.columns(2)
    long = df.reset_index().melt("Year", var_name="Class", value_name="km²")
    cmap = {CLASS_NAMES[i]: CLASS_COLORS[i] for i in CLASS_NAMES}
    c1_.plotly_chart(px.bar(long, x="Year", y="km²", color="Class", color_discrete_map=cmap,
                            title="Land-cover area by year").update_xaxes(type="category"), use_container_width=True)
    c2_.plotly_chart(px.line(long, x="Year", y="km²", color="Class", markers=True, color_discrete_map=cmap,
                             title="Trend").update_xaxes(type="category"), use_container_width=True)
    pct = df.div(df.sum(axis=1), axis=0) * 100
    st.subheader("Area (km²) and share (%)")
    st.dataframe(pd.concat({"km²": df.round(3), "%": pct.round(2)}, axis=1), use_container_width=True)
    st.download_button("⬇️ Download stats (CSV)", df.round(4).to_csv().encode(), "landcover_stats.csv", "text/csv")

# ------------------------------------------------------------------ change
with tab_change:
    t = ch.transition_matrix(A, B, pxa)
    st.subheader(f"Transition matrix {ya} → {yb} (km²)")
    st.plotly_chart(px.imshow(t.round(3), text_auto=".2f", color_continuous_scale="YlOrRd", aspect="auto"),
                    use_container_width=True)
    nd = ch.ndvi_stats(NA, NB)
    a1, a2, a3 = st.columns(3)
    a1.metric("Vegetation degraded (ΔNDVI < −0.2)", f"{nd['pct_degraded']:.1f} % of area")
    a2.metric("Vegetation improved (ΔNDVI > +0.2)", f"{nd['pct_improved']:.1f} % of area")
    a3.metric("Expansion rate", f"{summ['rate_km2_yr']:.3f} km²/yr")
    src_loss = t["Built-up (to)"].drop("Built-up (from)")
    fig = go.Figure(go.Bar(x=[i.replace(" (from)", "") for i in src_loss.index], y=src_loss.values,
                           marker_color=[CLASS_COLORS[i] for i in (0, 1, 3)]))
    fig.update_layout(title="What did new built-up area replace? (km²)", yaxis_title="km²")
    st.plotly_chart(fig, use_container_width=True)
    cname = pd.Series(np.bincount(cm[cm != NODATA_CLASS].ravel(), minlength=5)[:5] * pxa,
                      index=[ch.CHANGE_NAMES[i] for i in range(5)], name="km²")
    st.dataframe(cname.round(3).to_frame(), use_container_width=True)
    st.download_button("⬇️ Download change map (GeoTIFF)", class_map_to_geotiff(cm, rb), f"change_{ya}_{yb}.tif",
                       "image/tiff")

# ------------------------------------------------------------------ model
with tab_model:
    q1, q2, q3 = st.columns(3)
    q1.metric("Hold-out accuracy", f"{metrics['accuracy'] * 100:.1f} %")
    q2.metric("Cohen's kappa", f"{metrics['kappa']:.3f}")
    if ext:
        q3.metric(f"Accuracy vs ground truth ({yb})", f"{ext[yb] * 100:.1f} %")
    labels = [CLASS_NAMES[i] for i in metrics["labels"]]
    st.plotly_chart(px.imshow(metrics["confusion"], x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                              labels=dict(x="Predicted", y="Actual"), title="Confusion matrix (hold-out)"),
                    use_container_width=True)
    fi = feature_importance(model)
    if fi:
        st.plotly_chart(px.bar(x=list(fi), y=list(fi.values()), labels=dict(x="Feature", y="Importance"),
                               title="Feature importance"), use_container_width=True)
    st.caption("Training labels are bootstrapped from spectral-index rules (NDVI/NDWI/NDBI/MNDWI) on the first year; "
               "the classifier learns the full 9-feature signature and is then applied to every year.")

with tab_about:
    st.markdown("""
**Pipeline** — Rasterio read → reflectance scaling → spectral indices (NDVI, NDWI, NDBI, MNDWI) →
rule-based weak labels → class-balanced sampling → Random Forest / SVM → per-year classification →
transition matrix, urban-expansion map, ΔNDVI heatmap, CAGR.

**Using real data** — export Sentinel-2 L2A (or Landsat) composites for the same area for ≥2 years
(e.g. from Google Earth Engine / Copernicus Browser) as multi-band GeoTIFFs on an identical grid,
upload them, and set the band order of the file.

**Limits** — Weak labels are heuristic; for publication-grade results supply field-verified training polygons.
""")
