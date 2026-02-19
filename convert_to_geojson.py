import pandas as pd
import geopandas as gpd
from shapely import wkt
import json
import os

csv_file = r'E:\WEBSITE_KP\desa1_riau.csv'
print(f"Loading CSV from: {csv_file}")
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} rows from CSV")

if 'WKT' not in df.columns:
    print("WKT column not found. Available columns:", df.columns.tolist())
    exit(1)

print("Converting WKT to geometry...")
try:
    df['geometry'] = df['WKT'].apply(wkt.loads)
    print("WKT conversion successful")
except Exception as e:
    print(f"Error converting WKT: {e}")
    exit(1)

gdf = gpd.GeoDataFrame(df, geometry='geometry')

gdf.crs = "EPSG:4326"

static_dir = r'E:\WEBSITE_KP\static'
os.makedirs(static_dir, exist_ok=True)

geojson_file = os.path.join(static_dir, 'desa1_riau.geojson')
print(f"Saving GeoJSON to: {geojson_file}")
gdf.to_file(geojson_file, driver='GeoJSON')

print(f"Successfully converted {csv_file} to {geojson_file}")
print(f"GeoJSON contains {len(gdf)} features")
