# 🛰️ AI-Powered Land Cover & Urban Sprawl Change Detection

A Streamlit + Folium dashboard that classifies multi-year satellite imagery into **Water, Vegetation, Built-up and Barren** land, then quantifies how the landscape changed, with a focus on urban expansion.

🔗 **Live demo:** [Land Cover & Urban Sprawl Monitor](https://landcover-jcfbm7hw8ktpetdz6e95vy.streamlit.app/)



---

## ✨ Features

- **GeoTIFF ingest** with Rasterio: any CRS is reprojected to WGS84, reflectance is scaled automatically, nodata is masked
- **9 features per pixel:** 5 spectral bands + NDVI, NDWI, NDBI and MNDWI
- **Weak-label bootstrapping** to generate training samples, then class-balanced **Random Forest** or **SVM** training
- **Accuracy reporting:** Cohen's kappa and confusion matrix
- **Interactive maps:** land cover per year, change map, NDVI-difference heatmap, urban-expansion heatmap, ROI drawing
- **Statistics:** area per class, transition matrix, CAGR, expansion rate (km²/yr)
- **Export:** CSV statistics and GeoTIFF classified rasters

## 🏷️ Land Cover Classes

| Class | Main indicator |
|-------|----------------|
| 💧 Water | NDWI / MNDWI |
| 🌿 Vegetation | NDVI |
| 🏙️ Built-up | NDBI |
| 🟫 Barren | Low NDVI, low NDWI |

## 🚀 Quick Start

bash
git clone https://github.com/Mainak-06/landcover.git
cd landcover
pip install -r requirements.txt
streamlit run app.py


Run the tests (3 tests, including a headless app run):

```bash
pytest
```

## 🌍 Using Real Data

The app can run on built-in synthetic data for a quick demo. To analyse a real area:

1. Export **Sentinel-2 L2A** composites for **two or more years** on the **same grid** (Google Earth Engine or Copernicus).
2. Include these bands: **Blue, Green, Red, NIR, SWIR1** (Sentinel-2 `B2, B3, B4, B8, B11`).
3. Upload the GeoTIFFs in the app, set the year for each file, and confirm the band order.

## 🗂️ Project Structure


landcover/
├── app.py                # Streamlit entry point
├── requirements.txt
├── .streamlit/           # Streamlit configuration
├── src/
│   ├── config.py         # Constants, class definitions
│   ├── synthetic.py      # Synthetic demo data generator
│   ├── features.py       # Bands + spectral indices
│   ├── classify.py       # Weak labels, RF / SVM training, metrics
│   ├── change.py         # Change detection, transition matrix, rates
│   ├── geo.py            # Raster I/O, reprojection, export
│   └── pipeline.py       # End-to-end workflow
└── tests/                # pytest tests


## 🔬 Method in Brief

1. **Load and align** rasters for each year.
2. **Compute features:** bands and the indices NDVI, NDWI, NDBI, MNDWI.
3. **Create weak labels** from index thresholds, then train a class-balanced RF or SVM on them.
4. **Classify** each year and **compare** the results pixel by pixel.
5. **Summarise** change as maps, a transition matrix, CAGR and expansion rate.

## ⚠️ Limitations

- Training labels come from index-based rules rather than ground-truth samples, so kappa can look better than real-world accuracy. For research use, validate with independent reference points.
- Results depend on cloud-free, consistently processed imagery across years.
- Spectral indices can confuse barren land with built-up areas in some regions.

## 🔮 Future Improvements

- Accept user-supplied ground-truth samples for supervised training and validation
- Add deep-learning classifiers (e.g. U-Net)
- Support more classes (cropland, forest, wetlands)
- Batch processing for multi-year time series

## 🛠️ Tech Stack

Python · Streamlit · Folium · Rasterio · scikit-learn · NumPy · pandas · pytest

## 👤 Author

**Mainak** · [GitHub @Mainak-06](https://github.com/Mainak-06)

