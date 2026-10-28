# AI-Powered Land Cover & Urban Sprawl Change Detection

Streamlit + Folium dashboard that classifies multi-year satellite imagery into
Water / Vegetation / Built-up / Barren with Random Forest or SVM, then quantifies change.

## Run
    pip install -r requirements.txt
    streamlit run app.py
    pytest            # 3 tests, incl. headless app run

## Features
- Rasterio GeoTIFF ingest (any CRS -> WGS84, auto reflectance scaling, nodata masks)
- Indices: NDVI, NDWI, NDBI, MNDWI + 5 bands = 9 features
- Weak-label bootstrapping -> class-balanced RF/SVM training, kappa + confusion matrix
- Map: land-cover per year, change map, NDVI-difference heatmap, urban-expansion heatmap, ROI drawing
- Stats: area per class, transition matrix, CAGR, expansion rate (km²/yr), CSV + GeoTIFF export

## Real data
Export Sentinel-2 L2A composites (Blue,Green,Red,NIR,SWIR1 = B2,B3,B4,B8,B11) for 2+ years on
the same grid (Google Earth Engine / Copernicus). Upload, set years and band order.

## Layout
app.py · src/{config,synthetic,features,classify,change,geo,pipeline}.py · tests/
