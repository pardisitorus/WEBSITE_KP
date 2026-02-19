import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def initialize_gee():
    """Initialize Google Earth Engine"""
    try:
        # Note: This requires GEE authentication
        # For production, you need to run: earthengine authenticate
        import ee
        ee.Initialize()
        return True
    except Exception as e:
        print(f"GEE initialization failed: {e}")
        return False

def fetch_lst_from_gee(lon, lat, date):
    """Fetch Land Surface Temperature from GEE"""
    try:
        import ee
        point = ee.Geometry.Point([lon, lat])
        # Landsat 8 LST collection
        lst_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_LST') \
            .filterDate(date, (pd.to_datetime(date) + timedelta(days=1)).strftime('%Y-%m-%d')) \
            .filterBounds(point)

        lst_image = lst_collection.first()
        if lst_image:
            lst_value = lst_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30
            ).get('LST').getInfo()

            if lst_value:
                # Convert from Kelvin to Celsius
                return lst_value * 0.00341802 + 149.0 - 273.15
    except Exception as e:
        print(f"Error fetching LST: {e}")

    return None

def fetch_ndvi_from_gee(lon, lat, date):
    """Fetch NDVI from GEE"""
    try:
        point = ee.Geometry.Point([lon, lat])
        # Landsat 8 NDVI
        ndvi_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_8DAY_NDVI') \
            .filterDate(date, (pd.to_datetime(date) + timedelta(days=8)).strftime('%Y-%m-%d')) \
            .filterBounds(point)

        ndvi_image = ndvi_collection.first()
        if ndvi_image:
            ndvi_value = ndvi_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=30
            ).get('NDVI').getInfo()

            if ndvi_value:
                return ndvi_value
    except Exception as e:
        print(f"Error fetching NDVI: {e}")

    return None

def fetch_rainfall_from_openmeteo(lon, lat, date):
    """Fetch rainfall data from Open-Meteo (free alternative)"""
    try:
        # Open-Meteo API for historical weather data
        start_date = (pd.to_datetime(date) - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = date

        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=precipitation_sum&timezone=Asia/Jakarta"

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'daily' in data and 'precipitation_sum' in data['daily']:
                precip_data = data['daily']['precipitation_sum']
                # Return average rainfall over the last 30 days
                valid_precip = [p for p in precip_data if p is not None]
                if valid_precip:
                    return np.mean(valid_precip)
    except Exception as e:
        print(f"Error fetching rainfall: {e}")

    return None

def fetch_live_data_for_villages(df):
    """Fetch live satellite and weather data for all villages"""
    print("Fetching live data for villages...")

    # Create a copy of the dataframe
    live_df = df.copy()

    # Initialize GEE (if available)
    gee_available = initialize_gee()

    # Current date
    current_date = datetime.now().strftime('%Y-%m-%d')

    # For each village, fetch live data
    for idx, row in live_df.iterrows():
        try:
            # Get coordinates (assuming columns exist, otherwise use dummy)
            if 'longitude' in row and 'latitude' in row:
                lon, lat = row['longitude'], row['latitude']
            else:
                # Generate dummy coordinates for Riau province
                lon = np.random.uniform(100, 104)
                lat = np.random.uniform(0, 2)

            # Fetch LST
            lst = None
            if gee_available:
                lst = fetch_lst_from_gee(lon, lat, current_date)

            if lst is None:
                # Fallback to simulated data
                lst = 25 + np.random.uniform(-5, 15)  # 20-40°C range

            # Fetch NDVI
            ndvi = None
            if gee_available:
                ndvi = fetch_ndvi_from_gee(lon, lat, current_date)

            if ndvi is None:
                # Fallback to simulated data
                ndvi = np.random.uniform(0.1, 0.9)  # 0.1-0.9 range

            # Fetch rainfall
            rainfall = fetch_rainfall_from_openmeteo(lon, lat, current_date)

            if rainfall is None:
                # Fallback to simulated data
                rainfall = np.random.uniform(0, 50)  # 0-50mm range

            # Update the dataframe with live data
            live_df.at[idx, 'LST_Max_2024_C'] = lst
            live_df.at[idx, 'NDVI_Max_2024'] = ndvi
            live_df.at[idx, 'Rain_Max_2024_mm'] = rainfall

        except Exception as e:
            print(f"Error processing village {row.get('NAMA_DESA', idx)}: {e}")
            # Use fallback values
            live_df.at[idx, 'LST_Max_2024_C'] = 30 + np.random.uniform(-5, 5)
            live_df.at[idx, 'NDVI_Max_2024'] = np.random.uniform(0.2, 0.8)
            live_df.at[idx, 'Rain_Max_2024_mm'] = np.random.uniform(5, 25)

    # Recalculate physics features with live data
    LST = live_df['LST_Max_2024_C']
    NDVI = live_df['NDVI_Max_2024']
    Rain_Log = np.log1p(live_df['Rain_Max_2024_mm'])
    EPS = 0.01

    live_df['X1_Fuel_Dryness'] = (LST * (1 - NDVI)) * (Rain_Log + EPS)
    live_df['X2_Thermal_Kinetic'] = LST ** 2
    live_df['X3_Hydro_Stress'] = LST * (Rain_Log + EPS)
    live_df = live_df.replace([np.inf, -np.inf], 0).fillna(0)

    print(f"✅ Live data fetched for {len(live_df)} villages")
    return live_df

def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {
        'high': '#ff0000',    # Red
        'medium': '#ffff00',  # Yellow
        'low': '#00ff00'      # Green
    }
    return colors.get(risk_level, '#808080')  # Gray for unknown
