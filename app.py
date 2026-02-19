import streamlit as st
import pandas as pd
import numpy as np
import os
import pydeck as pdk
import shapely.wkt
import shapely.geometry
import ee
import altair as alt
from datetime import datetime, timedelta
import joblib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. KONFIGURASI AWAL
# ==============================================================================

# Path Data
DATA_URL = "https://drive.google.com/uc?id=1jmBB6Dv36aRnbDkj-cuZ154M0E3tzhOQ"
LOCAL_FILE = "desa1_riau.csv"
MODEL_FILE = "model_knn.pkl"

# Konfigurasi untuk optimasi
BATCH_SIZE = 10  # Process desa dalam batch untuk mempercepat
MAX_WORKERS = 5  # Jumlah thread paralel

# ==============================================================================
# 2. INISIALISASI GOOGLE EARTH ENGINE
# ==============================================================================

@st.cache_resource
def init_ee():
    """Koneksi Hybrid: Secrets (Cloud), refresh token, lalu auth lokal."""
    project_id = st.secrets.get("GEE_PROJECT", "website-kp")
    errors = []

    # 1) Service Account JSON untuk deployment cloud
    try:
        service_token = st.secrets.get("EARTHENGINE_TOKEN")
        if service_token:
            import json
            from google.oauth2.service_account import Credentials as ServiceAccountCredentials

            service_account_info = (
                json.loads(service_token) if isinstance(service_token, str) else dict(service_token)
            )
            credentials = ServiceAccountCredentials.from_service_account_info(
                service_account_info,
                scopes=ee.oauth.SCOPES,
            )
            ee.Initialize(credentials=credentials, project=project_id)
            return True
    except Exception as e:
        errors.append(f"Service account gagal: {e}")

    # 2) Refresh token (tanpa perlu `earthengine authenticate`)
    try:
        refresh_token = str(st.secrets.get("GEE_REFRESH_TOKEN", "")).strip()
        if refresh_token and "paste_your_refresh_token_here" not in refresh_token.lower():
            from google.oauth2.credentials import Credentials as UserCredentials

            credentials = UserCredentials(
                token=None,
                refresh_token=refresh_token,
                token_uri=ee.oauth.TOKEN_URI,
                client_id=ee.oauth.CLIENT_ID,
                client_secret=ee.oauth.CLIENT_SECRET,
                scopes=ee.oauth.SCOPES,
            )
            ee.Initialize(credentials=credentials, project=project_id)
            return True
    except Exception as e:
        errors.append(f"Refresh token gagal: {e}")

    # 3) Fallback auth lokal (laptop/CLI)
    try:
        ee.Initialize(project=project_id)
        return True
    except Exception as e:
        errors.append(f"Auth lokal gagal: {e}")

    try:
        ee.Initialize()
        return True
    except Exception as e:
        errors.append(f"Auth default gagal: {e}")

    st.sidebar.error("Gagal Login GEE. Tidak ada metode autentikasi yang berhasil.")
    for err in errors:
        st.sidebar.caption(f"- {err}")
    st.sidebar.warning(
        "Cloud/Server: isi `.streamlit/secrets.toml` dengan `EARTHENGINE_TOKEN` "
        "(service account JSON) atau `GEE_REFRESH_TOKEN`."
    )
    st.sidebar.warning("Laptop lokal: jalankan `earthengine authenticate`, lalu restart app.")
    return False

# ==============================================================================
# 3. LOAD MODEL KNN
# ==============================================================================

@st.cache_resource
def load_knn_model():
    """Load model KNN yang sudah dilatih"""
    try:
        if os.path.exists(MODEL_FILE):
            model = joblib.load(MODEL_FILE)
            st.sidebar.success("✅ Model KNN berhasil dimuat!")
            return model
        else:
            st.sidebar.warning(f"⚠️ File model '{MODEL_FILE}' tidak ditemukan di direktori kerja!")
            st.sidebar.info("💡 Model akan menggunakan metode heuristik sebagai fallback.")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        st.sidebar.info("💡 Menggunakan metode heuristik.")
        return None

# ==============================================================================
# 4. FUNGSI LOAD DATA DESA
# ==============================================================================

@st.cache_data
def load_data():
    """Load dan preprocessing data desa dari Google Drive"""
    
    # Download jika belum ada
    if not os.path.exists(LOCAL_FILE):
        try:
            import gdown
            with st.spinner("⬇️ Mengunduh layer desa dari Google Drive..."):
                gdown.download(DATA_URL, LOCAL_FILE, quiet=False, fuzzy=True)
        except Exception as e:
            st.error(f"Gagal download. Error: {e}")
            st.info("Pastikan gdown terinstall: pip install gdown")
            return None

    try:
        # Load CSV
        df = pd.read_csv(LOCAL_FILE)
        df.columns = [c.strip().upper() for c in df.columns]

        # Standarisasi nama kolom
        col_map = {
            'WADMKD': 'nama_desa',
            'NAMOBJ': 'nama_desa',
            'DESA': 'nama_desa',
            'WADMKK': 'kabupaten',
            'KABUPATEN': 'kabupaten'
        }
        df = df.rename(columns=col_map)
        df = df.loc[:, ~df.columns.duplicated()]

        # Pastikan kolom nama_desa ada
        if 'nama_desa' not in df.columns:
            df['nama_desa'] = "Desa Tanpa Nama"

        # Konversi WKT ke geometry
        df['geometry'] = df['WKT'].apply(
            lambda x: shapely.wkt.loads(str(x)) if pd.notnull(x) else None
        )
        df = df.dropna(subset=['geometry']).reset_index(drop=True)

        # Hitung centroid untuk setiap desa
        df['lat'] = df['geometry'].apply(lambda g: g.centroid.y)
        df['lon'] = df['geometry'].apply(lambda g: g.centroid.x)

        return df

    except Exception as e:
        st.error(f"❌ Gagal load layer desa: {e}")
        return None

# ==============================================================================
# 5. FUNGSI EKSTRAKSI DATA SATELIT (OPTIMIZED VERSION)
# ==============================================================================

def fetch_satellite_data_single(row_data, start_dates, end_date):
    """
    Fetch satellite data untuk satu desa
    Fungsi ini dirancang untuk dijalankan secara paralel
    """
    idx = row_data['index']
    lat = row_data['lat']
    lon = row_data['lon']
    nama = row_data['nama_desa']
    
    result = {
        'index': idx,
        'LST': 30.0,
        'NDVI': 0.5,
        'Rain': 100.0,
        'LST_Date': 'N/A',
        'NDVI_Date': 'N/A',
        'Rain_Date': 'N/A'
    }
    
    try:
        # Buat geometri EE dari centroid
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(500)  # Buffer 500m untuk sampling
        
        # ===== 1. LAND SURFACE TEMPERATURE (LST) =====
        try:
            lst_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
                .filterDate(start_dates['lst'], end_date) \
                .filterBounds(point) \
                .select('LST_Day_1km')
            
            if lst_collection.size().getInfo() > 0:
                lst_image = lst_collection.mean()
                lst_value = lst_image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=1000,
                    maxPixels=1e9
                ).getInfo()
                
                if 'LST_Day_1km' in lst_value and lst_value['LST_Day_1km'] is not None:
                    lst_celsius = (lst_value['LST_Day_1km'] * 0.02) - 273.15
                    result['LST'] = round(lst_celsius, 1)
                    
                    latest_lst = lst_collection.sort('system:time_start', False).first()
                    lst_date = datetime.fromtimestamp(
                        latest_lst.get('system:time_start').getInfo() / 1000
                    ).strftime('%Y-%m-%d')
                    result['LST_Date'] = lst_date
        except Exception as e:
            pass
        
        # ===== 2. NDVI =====
        try:
            ndvi_collection = ee.ImageCollection('MODIS/061/MOD13Q1') \
                .filterDate(start_dates['ndvi'], end_date) \
                .filterBounds(point) \
                .select('NDVI')
            
            if ndvi_collection.size().getInfo() > 0:
                ndvi_image = ndvi_collection.mean()
                ndvi_value = ndvi_image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=250,
                    maxPixels=1e9
                ).getInfo()
                
                if 'NDVI' in ndvi_value and ndvi_value['NDVI'] is not None:
                    ndvi_normalized = ndvi_value['NDVI'] / 10000.0
                    result['NDVI'] = round(ndvi_normalized, 3)
                    
                    latest_ndvi = ndvi_collection.sort('system:time_start', False).first()
                    ndvi_date = datetime.fromtimestamp(
                        latest_ndvi.get('system:time_start').getInfo() / 1000
                    ).strftime('%Y-%m-%d')
                    result['NDVI_Date'] = ndvi_date
        except Exception as e:
            pass
        
        # ===== 3. PRECIPITATION =====
        try:
            rain_collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                .filterDate(start_dates['rain'], end_date) \
                .filterBounds(point) \
                .select('precipitation')
            
            if rain_collection.size().getInfo() > 0:
                rain_image = rain_collection.sum()
                rain_value = rain_image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=5000,
                    maxPixels=1e9
                ).getInfo()
                
                if 'precipitation' in rain_value and rain_value['precipitation'] is not None:
                    result['Rain'] = round(rain_value['precipitation'], 1)
                    
                    latest_rain = rain_collection.sort('system:time_start', False).first()
                    rain_date = datetime.fromtimestamp(
                        latest_rain.get('system:time_start').getInfo() / 1000
                    ).strftime('%Y-%m-%d')
                    result['Rain_Date'] = rain_date
        except Exception as e:
            pass
            
    except Exception as e:
        pass
    
    return result

def get_satellite_data_optimized(df_base, use_parallel=True):
    """
    VERSI OPTIMIZED - Mengambil data satelit dengan processing paralel
    """
    
    if df_base is None or len(df_base) == 0:
        st.error("❌ Data desa kosong!")
        return None
    
    df = df_base.copy()
    
    # Inisialisasi kolom hasil
    df['LST'] = 30.0
    df['NDVI'] = 0.5
    df['Rain'] = 100.0
    df['LST_Date'] = "N/A"
    df['NDVI_Date'] = "N/A"
    df['Rain_Date'] = "N/A"
    
    try:
        # Tentukan periode pengambilan data
        end_date = datetime.now()
        start_dates = {
            'lst': (end_date - timedelta(days=3)).strftime('%Y-%m-%d'),
            'ndvi': (end_date - timedelta(days=16)).strftime('%Y-%m-%d'),
            'rain': (end_date - timedelta(days=30)).strftime('%Y-%m-%d')
        }
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        total_desa = len(df)
        
        # Prepare data untuk parallel processing
        df_indexed = df.reset_index()
        rows_data = df_indexed[['index', 'lat', 'lon', 'nama_desa']].to_dict('records')
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if use_parallel and total_desa > 5:
            # PARALLEL PROCESSING untuk dataset besar
            status_text.text(f"🚀 Menggunakan parallel processing ({MAX_WORKERS} workers)...")
            
            completed = 0
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(fetch_satellite_data_single, row, start_dates, end_date_str): row['index'] 
                    for row in rows_data
                }
                
                # Process completed tasks
                for future in as_completed(future_to_idx):
                    result = future.result()
                    idx = result['index']
                    
                    # Update dataframe
                    df.at[idx, 'LST'] = result['LST']
                    df.at[idx, 'NDVI'] = result['NDVI']
                    df.at[idx, 'Rain'] = result['Rain']
                    df.at[idx, 'LST_Date'] = result['LST_Date']
                    df.at[idx, 'NDVI_Date'] = result['NDVI_Date']
                    df.at[idx, 'Rain_Date'] = result['Rain_Date']
                    
                    completed += 1
                    progress = completed / total_desa
                    progress_bar.progress(progress)
                    status_text.text(f"📡 Progress: {completed}/{total_desa} desa ({progress*100:.1f}%)")
        else:
            # SEQUENTIAL PROCESSING untuk dataset kecil
            status_text.text("📡 Mengambil data satelit (sequential)...")
            for i, row in enumerate(rows_data):
                result = fetch_satellite_data_single(row, start_dates, end_date_str)
                idx = result['index']
                
                df.at[idx, 'LST'] = result['LST']
                df.at[idx, 'NDVI'] = result['NDVI']
                df.at[idx, 'Rain'] = result['Rain']
                df.at[idx, 'LST_Date'] = result['LST_Date']
                df.at[idx, 'NDVI_Date'] = result['NDVI_Date']
                df.at[idx, 'Rain_Date'] = result['Rain_Date']
                
                progress = (i + 1) / total_desa
                progress_bar.progress(progress)
                status_text.text(f"📡 Progress: {i+1}/{total_desa} desa")
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Data satelit berhasil diambil untuk {total_desa} desa!")
        
        # Tampilkan statistik
        st.info(f"""
        📊 **Ringkasan Data Satelit:**
        - Suhu (LST): {df['LST'].min():.1f}°C - {df['LST'].max():.1f}°C (Rata-rata: {df['LST'].mean():.1f}°C)
        - NDVI: {df['NDVI'].min():.3f} - {df['NDVI'].max():.3f} (Rata-rata: {df['NDVI'].mean():.3f})
        - Hujan: {df['Rain'].min():.1f}mm - {df['Rain'].max():.1f}mm (Rata-rata: {df['Rain'].mean():.1f}mm)
        """)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error dalam pengambilan data satelit: {e}")
        st.info("💡 Menggunakan data simulasi untuk testing...")
        
        # Generate data simulasi
        np.random.seed(42)
        df['LST'] = np.random.uniform(28, 38, len(df))
        df['NDVI'] = np.random.uniform(0.2, 0.8, len(df))
        df['Rain'] = np.random.uniform(10, 300, len(df))
        df['LST_Date'] = datetime.now().strftime('%Y-%m-%d')
        df['NDVI_Date'] = datetime.now().strftime('%Y-%m-%d')
        df['Rain_Date'] = datetime.now().strftime('%Y-%m-%d')
        
        return df

# ==============================================================================
# 6. PREDIKSI MENGGUNAKAN MODEL KNN
# ==============================================================================

def predict_with_knn_model(df, model):
    """
    Menggunakan model KNN untuk prediksi risiko kebakaran
    """
    if model is None:
        st.warning("⚠️ Model tidak tersedia, menggunakan metode heuristik")
        return predict_heuristic(df)
    
    try:
        # Konstanta
        EPS = 0.01
        
        # Ensure numeric
        df['LST'] = pd.to_numeric(df['LST'], errors='coerce').fillna(30.0)
        df['NDVI'] = pd.to_numeric(df['NDVI'], errors='coerce').fillna(0.5).clip(0, 1)
        df['Rain'] = pd.to_numeric(df['Rain'], errors='coerce').fillna(100.0).clip(lower=0)
        
        # Hitung Rain_Log
        df['Rain_Log'] = np.log1p(df['Rain'])
        
        # PHYSICS FEATURES (sama seperti training)
        df['X1_Fuel_Dryness'] = (df['LST'] * (1 - df['NDVI'])) * (df['Rain_Log'] + EPS)
        df['X2_Thermal_Kinetic'] = df['LST'] ** 2
        df['X3_Hydro_Stress'] = df['LST'] * (df['Rain_Log'] + EPS)
        
        # Clean data
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Prepare features untuk prediksi
        X_features = df[['X1_Fuel_Dryness', 'X2_Thermal_Kinetic', 'X3_Hydro_Stress']]
        
        # PREDIKSI MENGGUNAKAN MODEL KNN
        st.info("🤖 Menggunakan Model KNN untuk prediksi...")
        predictions = model.predict(X_features)
        probabilities = model.predict_proba(X_features)
        
        # Ambil probabilitas kelas 1 (kebakaran)
        df['fire_prob_raw'] = probabilities[:, 1]
        
        # Convert to percentage
        df['prob_pct'] = (df['fire_prob_raw'] * 100).round(1)
        
        # Klasifikasi prediksi
        df['prediction'] = predictions
        
        # Penentuan level berdasarkan probabilitas
        def get_level(prob):
            if prob > 70: return "TINGGI", [255, 0, 0]
            elif prob > 40: return "SEDANG", [255, 165, 0]
            return "RENDAH", [0, 128, 0]
        
        res = df['prob_pct'].apply(get_level)
        df['level'] = [x[0] for x in res]
        df['color'] = [x[1] for x in res]
        
        # Status kekeringan
        def get_dry_status(rain):
            if pd.isna(rain): return "NO DATA"
            if rain < 10: return "SANGAT KERING"
            elif rain < 50: return "KERING"
            elif rain < 100: return "NORMAL"
            return "BASAH"
        
        df['status_kekeringan'] = df['Rain'].apply(get_dry_status)
        
        # Tampilkan info model
        st.success(f"✅ Prediksi KNN selesai! Desa dengan prediksi KEBAKARAN: {predictions.sum()}/{len(predictions)}")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error dalam prediksi model: {e}")
        st.warning("⚠️ Fallback ke metode heuristik")
        return predict_heuristic(df)

def predict_heuristic(df):
    """
    FALLBACK: Metode heuristik jika model tidak tersedia
    """
    EPS = 1e-6
    
    # Pre-processing
    df['Rain'] = pd.to_numeric(df['Rain'], errors='coerce').fillna(100.0).clip(lower=0)
    df['LST'] = pd.to_numeric(df['LST'], errors='coerce').fillna(30.0)
    df['NDVI'] = pd.to_numeric(df['NDVI'], errors='coerce').fillna(0.5).clip(0, 1)
    df['Rain_Log'] = np.log1p(df['Rain'])
    
    # Physics features
    df['X1_Fuel_Dryness'] = (df['LST'] * (1 - df['NDVI'])) * (df['Rain_Log'] + EPS)
    df['X2_Thermal_Kinetic'] = df['LST'] ** 2
    df['X3_Hydro_Stress'] = df['LST'] * (df['Rain_Log'] + EPS)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Normalisasi
    min_x2, max_x2 = 400, 1600
    norm_thermal = ((df['X2_Thermal_Kinetic'] - min_x2) / (max_x2 - min_x2)).clip(0, 1)
    norm_hydro = (df['X3_Hydro_Stress'] / 200).clip(0, 1)
    norm_fuel = (1 - df['NDVI']).clip(0, 1)
    
    # Heuristic risk score
    risk_raw = (0.6 * norm_thermal) + (0.2 * norm_fuel) - (0.5 * norm_hydro)
    risk_score = risk_raw.clip(0, 1)
    df['prob_pct'] = (risk_score * 100).round(1)
    
    # Level
    def get_level(p):
        if p > 70: return "TINGGI", [255, 0, 0]
        elif p > 40: return "SEDANG", [255, 165, 0]
        return "RENDAH", [0, 128, 0]
    
    res = df['prob_pct'].apply(get_level)
    df['level'] = [x[0] for x in res]
    df['color'] = [x[1] for x in res]
    
    # Status kekeringan
    def get_dry_status(rain):
        if pd.isna(rain): return "NO DATA"
        if rain < 10: return "SANGAT KERING"
        elif rain < 50: return "KERING"
        elif rain < 100: return "NORMAL"
        return "BASAH"
    
    df['status_kekeringan'] = df['Rain'].apply(get_dry_status)
    
    return df

# ==============================================================================
# 7. DASHBOARD UTAMA
# ==============================================================================

def main():
    st.set_page_config(page_title="RIAU FIRE COMMAND CENTER", layout="wide")
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1041/1041891.png", width=70)
        st.title("🎛️ PANEL KONTROL")
        
        st.markdown("---")
        
        # Cek koneksi GEE
        gee_connected = init_ee()
        if gee_connected:
            st.success("🛰️ GEE SATELIT: ONLINE")
        else:
            st.error("GEE OFFLINE (Autentikasi belum valid)")
            st.warning("Isi secret GEE atau jalankan `earthengine authenticate` jika mode lokal.")
        
        # Load model
        knn_model = load_knn_model()
        
        if knn_model is not None:
            st.success("🤖 MODEL KNN: LOADED")
        else:
            st.warning("⚠️ MODEL: HEURISTIC MODE")
        
        st.markdown("---")
        
        # Tombol refresh
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 REFRESH DATA"):
                st.cache_data.clear()
                if 'data_monitor' in st.session_state:
                    del st.session_state['data_monitor']
                st.rerun()
        
        with col2:
            if st.button("🗑️ CLEAR CACHE"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Cache dibersihkan!")
        
        st.markdown("---")
        st.markdown("### ℹ️ Info Sumber Data")
        st.info("""
        **1. SUHU (LST)**
        - Sumber: MODIS Terra MOD11A1
        - Update: Harian
        - Resolusi: 1 km
        
        **2. VEGETASI (NDVI)**
        - Sumber: MODIS MOD13Q1
        - Update: 16 Hari
        - Resolusi: 250 m
        
        **3. HUJAN (CHIRPS)**
        - Sumber: CHIRPS Daily
        - Update: Harian
        - Resolusi: 5 km
        
        **Klasifikasi Kekeringan:**
        - < 10mm = Sangat Kering
        - 10-50mm = Kering
        - 50-100mm = Normal
        - > 100mm = Basah
        """)
        
        st.markdown("---")
        st.markdown("### 🤖 Info Model")
        if knn_model is not None:
            st.success("""
            **Model:** K-Nearest Neighbors
            **Features:** 3 (Physics-based)
            - X1: Fuel Dryness
            - X2: Thermal Kinetic
            - X3: Hydro Stress
            """)
        else:
            st.warning("""
            **Mode:** Heuristic Fallback
            Menggunakan formula fisika
            tanpa machine learning
            """)

    # --- HEADER ---
    st.title("🔥 RIAU FIRE COMMAND CENTER (RFCC)")
    st.markdown("### 🤖 Powered by Machine Learning & Physics-Informed Features")
    
    # --- LOAD DATA ---
    df_base = load_data()
    if df_base is None:
        st.error("❌ Gagal memuat data desa!")
        st.stop()
    
    # --- GET SATELLITE DATA & PREDICT ---
    if 'data_monitor' not in st.session_state:
        # Load model
        knn_model = load_knn_model()
        
        if not gee_connected:
            st.error("❌ Tidak dapat mengambil data satelit karena GEE offline!")
            st.info("💡 Menggunakan data simulasi...")
            df_sat = get_satellite_data_optimized(df_base, use_parallel=False)
        else:
            with st.spinner("🛰️ Mengambil data satelit dari Google Earth Engine..."):
                df_sat = get_satellite_data_optimized(df_base, use_parallel=True)
        
        # Prediksi menggunakan model atau heuristic
        with st.spinner("🤖 Melakukan prediksi risiko kebakaran..."):
            st.session_state.data_monitor = predict_with_knn_model(df_sat, knn_model)
    
    df = st.session_state.data_monitor
    
    # --- INFO DATA ---
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown(f"""
        📅 **Tanggal Data Satelit:**
        - **Suhu (LST):** {df['LST_Date'].iloc[0]}
        - **Vegetasi (NDVI):** {df['NDVI_Date'].iloc[0]}
        - **Hujan (CHIRPS):** {df['Rain_Date'].iloc[0]}
        """)
    
    with col_info2:
        st.markdown(f"""
        📍 **Cakupan Wilayah:**
        - **Total Desa:** {len(df)}
        - **Kabupaten:** {df['kabupaten'].nunique()}
        """)
    
    with col_info3:
        if 'prediction' in df.columns:
            fire_pred = df['prediction'].sum()
            st.markdown(f"""
            🤖 **Prediksi Model KNN:**
            - **Risiko Kebakaran:** {fire_pred} desa
            - **Aman:** {len(df) - fire_pred} desa
            """)
        else:
            st.markdown("""
            ⚙️ **Mode Heuristic**
            Menggunakan formula fisika
            """)

    # ==============================================================================
    # BAGIAN 1: PETA & STATISTIK
    # ==============================================================================
    
    st.markdown("---")
    col_map, col_stat = st.columns([2, 1])
    
    # Logika Highlight
    view_state = pdk.ViewState(latitude=0.5, longitude=101.5, zoom=7.5, pitch=0)
    selected_desa_name = None

    if 'selection' in st.session_state and st.session_state.selection.get("selection", {}).get("rows"):
        if 'df_sorted_display' in st.session_state:
            idx = st.session_state.selection['selection']['rows'][0]
            if idx < len(st.session_state.df_sorted_display):
                sel_row = st.session_state.df_sorted_display.iloc[idx]
                selected_desa_name = sel_row['nama_desa']
                view_state = pdk.ViewState(
                    latitude=sel_row['lat'], 
                    longitude=sel_row['lon'], 
                    zoom=11.5, 
                    pitch=0
                )
                st.toast(f"📍 Menyorot: {selected_desa_name}")

    # PREPARE GEOJSON
    geojson_base = {"type": "FeatureCollection", "features": []}
    geojson_highlight = {"type": "FeatureCollection", "features": []}

    for _, row in df.iterrows():
        props = {
            "nama": row['nama_desa'],
            "kab": row['kabupaten'],
            "level": row['level'],
            "prob": row['prob_pct'],
            "color": row['color'],
            "kering": row['status_kekeringan']
        }
        geom = shapely.geometry.mapping(row['geometry'])
        
        feature = {"type": "Feature", "geometry": geom, "properties": props}
        geojson_base["features"].append(feature)
        
        if selected_desa_name and row['nama_desa'] == selected_desa_name:
            geojson_highlight["features"].append(feature)

    # LAYERS
    layers = []
    layers.append(pdk.Layer(
        "GeoJsonLayer",
        data=geojson_base,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.color",
        get_line_color=[0, 0, 0],
        get_line_width=100,
        line_width_min_pixels=2,
        opacity=0.7,
        auto_highlight=True
    ))
    
    if len(geojson_highlight["features"]) > 0:
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            data=geojson_highlight,
            stroked=True,
            filled=False,
            get_line_color=[255, 255, 0],
            get_line_width=500,
            line_width_min_pixels=5,
        ))

    with col_map:
        st.subheader("🗺️ Peta Risiko Kebakaran")
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"html": "<b>{nama}</b> ({kab})<br>Risiko: {level} ({prob}%)<br>Kekeringan: {kering}"},
            map_style="mapbox://styles/mapbox/light-v10"
        ))

    # --- STATISTIK ---
    with col_stat:
        st.subheader("📊 Analisis Risiko")
        
        # Pie Chart
        risk_counts = df['level'].value_counts().reset_index()
        risk_counts.columns = ['Status', 'Jumlah']
        
        color_scale = alt.Scale(
            domain=['TINGGI', 'SEDANG', 'RENDAH'],
            range=['#FF0000', '#FFA500', '#008000']
        )
        
        donut = alt.Chart(risk_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("Jumlah", stack=True),
            color=alt.Color("Status", scale=color_scale, legend=alt.Legend(title="Status Risiko")),
            tooltip=["Status", "Jumlah"],
            order=alt.Order("Status", sort="descending")
        ).properties(height=250)
        
        st.altair_chart(donut, use_container_width=True)
        
        # Metrik
        high_count = len(df[df['level'] == 'TINGGI'])
        st.metric("🔥 Desa Risiko Tinggi", high_count, f"{(high_count/len(df)*100):.1f}%")
        
        dry_count = len(df[(df['status_kekeringan'] == 'KERING') | (df['status_kekeringan'] == 'SANGAT KERING')])
        st.metric("💧 Desa Waspada Kekeringan", dry_count, f"{(dry_count/len(df)*100):.1f}%")

    # ==============================================================================
    # BAGIAN 2: MONITORING INDICATORS
    # ==============================================================================
    
    st.markdown("---")
    st.markdown("### 📊 Real-Time Monitoring Indicators")

    df['Rain'] = pd.to_numeric(df['Rain'], errors='coerce').fillna(0.0)
    df['LST'] = pd.to_numeric(df['LST'], errors='coerce')
    
    total_hotspots = len(df[df['level'] == 'TINGGI'])
    
    max_temp_val = df['LST'].max() 
    if pd.isna(max_temp_val):
        max_temp_val = 0.0
        max_temp_desa = "-"
    else:
        idx_max = df['LST'].idxmax()
        max_temp_desa = df.loc[idx_max, 'nama_desa']
    
    avg_rain = df['Rain'].mean()
    if pd.isna(avg_rain): avg_rain = 0.0
    
    avg_risk = df['prob_pct'].mean()
    if pd.isna(avg_risk): avg_risk = 0.0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🔥 Desa Risiko Tinggi", 
            value=f"{total_hotspots}", 
            delta="Status Siaga" if total_hotspots > 0 else "Aman",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            label="🌡️ Suhu Tertinggi", 
            value=f"{max_temp_val:.1f}°C", 
            delta=f"di {max_temp_desa[:15]}...", 
            delta_color="inverse"
        )

    with col3:
        status_hujan = "Kering" if avg_rain < 60 else "Basah"
        st.metric(
            label="💧 Rata-rata Hujan", 
            value=f"{avg_rain:.1f} mm", 
            delta=status_hujan,
            delta_color="normal" if status_hujan == "Basah" else "inverse"
        )

    with col4:
        st.metric(
            label="📈 Rata-rata Risiko", 
            value=f"{avg_risk:.1f}%", 
            delta="Waspada" if avg_risk > 50 else "Stabil",
            delta_color="inverse" if avg_risk > 50 else "normal"
        )

    # ==============================================================================
    # BAGIAN 3: VISUALISASI PRIORITAS & VALIDASI
    # ==============================================================================
    
    st.markdown("---")
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("🚨 Prioritas Penanganan (Top 10 Desa Rawan)")
        
        top_10_risk = df.sort_values(by='prob_pct', ascending=False).head(10)
        
        chart_priority = alt.Chart(top_10_risk).mark_bar().encode(
            x=alt.X('prob_pct', title='Tingkat Risiko (%)', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('nama_desa', sort='-x', title='Nama Desa'),
            color=alt.Color('level', 
                            scale=alt.Scale(domain=['TINGGI', 'SEDANG', 'RENDAH'], 
                                          range=['#FF0000', '#FFA500', '#008000']),
                            legend=None),
            tooltip=['nama_desa', 'kabupaten', 'LST', 'Rain', 'prob_pct', 'level']
        ).properties(height=350)
        
        st.altair_chart(chart_priority, use_container_width=True)
        st.caption("ℹ️ *Grafik menunjukkan desa prioritas untuk patroli/pemadaman.*")

    with c2:
        st.subheader("🌡️ Validasi Fisika")
        st.markdown("**Korelasi Suhu vs Risiko**")
        
        chart_physics = alt.Chart(df.sample(min(100, len(df)))).mark_circle(size=80).encode(
            x=alt.X('LST:Q', title='Suhu (°C)', scale=alt.Scale(zero=False)),
            y=alt.Y('prob_pct:Q', title='Risiko (%)'),
            color=alt.Color('level:N', 
                           scale=alt.Scale(domain=['TINGGI', 'SEDANG', 'RENDAH'],
                                          range=['#FF0000', '#FFA500', '#008000']),
                           legend=None),
            tooltip=['nama_desa', 'LST', 'Rain', 'NDVI', 'prob_pct', 'level']
        ).properties(height=300).interactive()
        
        st.altair_chart(chart_physics, use_container_width=True)
        
        # Wilayah paling rawan
        if 'kabupaten' in df.columns:
            avg_risk_kab = df.groupby('kabupaten')['prob_pct'].mean().idxmax()
            val_risk_kab = df.groupby('kabupaten')['prob_pct'].mean().max()
            st.info(f"📍 **Wilayah Paling Rawan:**\nKab. **{avg_risk_kab}** ({val_risk_kab:.1f}%)")

    # ==============================================================================
    # BAGIAN 4: SORT & FILTER
    # ==============================================================================
    
    st.markdown("---")
    st.markdown("### 🔃 Filter & Urutan Data")
    
    col_sort_1, col_sort_2, col_sort_3 = st.columns(3)
    
    with col_sort_1:
        sort_by = st.selectbox(
            "Urutkan Berdasarkan:",
            ["Tingkat Risiko (Probabilitas)", "Nama Desa", "Suhu (LST)", "Curah Hujan (Rain)", "NDVI"],
            index=0
        )
        
    with col_sort_2:
        sort_order = st.radio(
            "Arah Urutan:",
            ["Descending (Besar-Kecil)", "Ascending (Kecil-Besar)"],
            horizontal=True
        )
    
    with col_sort_3:
        # Filter by level
        level_filter = st.multiselect(
            "Filter Status Risiko:",
            options=['TINGGI', 'SEDANG', 'RENDAH'],
            default=['TINGGI', 'SEDANG', 'RENDAH']
        )

    # Apply filter
    df_filtered = df[df['level'].isin(level_filter)].copy()
    
    # Apply sorting
    is_ascending = True if "Ascending" in sort_order else False
    
    if sort_by == "Nama Desa":
        df_sorted = df_filtered.sort_values(by="nama_desa", ascending=is_ascending)
    elif sort_by == "Tingkat Risiko (Probabilitas)":
        df_sorted = df_filtered.sort_values(by="prob_pct", ascending=is_ascending)
    elif sort_by == "Suhu (LST)":
        df_sorted = df_filtered.sort_values(by="LST", ascending=is_ascending)
    elif sort_by == "Curah Hujan (Rain)":
        df_sorted = df_filtered.sort_values(by="Rain", ascending=is_ascending)
    else:  # NDVI
        df_sorted = df_filtered.sort_values(by="NDVI", ascending=is_ascending)
    
    df_sorted = df_sorted.reset_index(drop=True)
    st.session_state.df_sorted_display = df_sorted

    # Info filter
    st.info(f"📊 Menampilkan {len(df_sorted)} dari {len(df)} desa")

    # ==============================================================================
    # BAGIAN 5: TABEL DATA
    # ==============================================================================
    
    st.subheader("📂 Data Desa Lengkap")
    
    # Prepare display columns
    display_cols = ['nama_desa', 'kabupaten', 'level', 'prob_pct', 'LST', 'Rain', 'NDVI', 'status_kekeringan']
    if 'prediction' in df_sorted.columns:
        display_cols.insert(4, 'prediction')
    
    df_table = df_sorted[display_cols]
    
    column_config = {
        "nama_desa": st.column_config.TextColumn("Nama Desa", width="medium"),
        "kabupaten": st.column_config.TextColumn("Kabupaten", width="medium"),
        "level": st.column_config.TextColumn("Status Risiko", width="small"),
        "prob_pct": st.column_config.ProgressColumn(
            "Tingkat Risiko", 
            format="%.1f%%", 
            min_value=0, 
            max_value=100,
            width="medium"
        ),
        "LST": st.column_config.NumberColumn("Suhu (°C)", format="%.1f", width="small"),
        "Rain": st.column_config.NumberColumn("Hujan 30hr (mm)", format="%.1f", width="small"),
        "NDVI": st.column_config.NumberColumn("NDVI", format="%.3f", width="small"),
        "status_kekeringan": st.column_config.TextColumn("Status Kekeringan", width="medium")
    }
    
    if 'prediction' in df_sorted.columns:
        column_config["prediction"] = st.column_config.NumberColumn(
            "Prediksi KNN", 
            help="0=Aman, 1=Bahaya",
            width="small"
        )
    
    st.dataframe(
        df_table,
        column_config=column_config,
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        key="selection",
        height=400
    )

    # ==============================================================================
    # BAGIAN 6: DETAIL PHYSICS FEATURES (OPTIONAL)
    # ==============================================================================
    
    with st.expander("🔬 Lihat Detail Physics Features"):
        st.markdown("### Physics-Informed Features")
        st.markdown("""
        Model menggunakan 3 fitur berbasis fisika:
        - **X1 (Fuel Dryness)**: LST × (1-NDVI) × ln(Rain+1)
        - **X2 (Thermal Kinetic)**: LST²
        - **X3 (Hydro Stress)**: LST × ln(Rain+1)
        """)
        
        physics_cols = ['nama_desa', 'X1_Fuel_Dryness', 'X2_Thermal_Kinetic', 'X3_Hydro_Stress']
        if all(col in df_sorted.columns for col in physics_cols):
            st.dataframe(
                df_sorted[physics_cols].head(20),
                column_config={
                    "nama_desa": "Nama Desa",
                    "X1_Fuel_Dryness": st.column_config.NumberColumn("X1: Fuel Dryness", format="%.2f"),
                    "X2_Thermal_Kinetic": st.column_config.NumberColumn("X2: Thermal Kinetic", format="%.2f"),
                    "X3_Hydro_Stress": st.column_config.NumberColumn("X3: Hydro Stress", format="%.2f")
                },
                use_container_width=True,
                height=300
            )

    # ==============================================================================
    # BAGIAN 7: REKOMENDASI
    # ==============================================================================
    
    st.markdown("---")
    st.subheader("🛡️ REKOMENDASI TINDAKAN & MITIGASI")
    
    col_high, col_med, col_low = st.columns(3)
    
    with col_high:
        with st.expander("🚨 TINGKAT TINGGI", expanded=True):
            st.error("**STATUS: BAHAYA EKSTREM**")
            st.markdown("""
            **Tindakan Segera:**
            1. 🚨 **Aktivasi Sirine** di posko desa
            2. 👨‍🚒 **Mobilisasi RPK** ke titik panas
            3. 🚁 **Water Bombing** (koordinasi BPBD)
            4. 💧 **Pompa Tekanan Tinggi** siaga
            5. 🏃 **Evakuasi Warga** rentan
            6. 🌊 **Sekat Basah** intensif
            7. 🛸 **Patroli Drone** termal
            8. 🚫 **Larang Total** pembakaran
            """)
            
    with col_med:
        with st.expander("⚠️ TINGKAT SEDANG", expanded=True):
            st.warning("**STATUS: WASPADA**")
            st.markdown("""
            **Tindakan Preventif:**
            1. 👮 **Patroli Rutin** 2x sehari
            2. 💧 **Cek Sumber Air** kanal/sumur
            3. 🧹 **Bersihkan Sekat** bakar
            4. 📢 **Sosialisasi** door-to-door
            5. 🚩 **Bendera Kuning** di kantor desa
            6. 🧯 **Siaga Alat** di posko
            7. 🌦️ **Pantau Cuaca** 6 jam sekali
            8. 📞 **Lapor Cepat** ke Satgas
            """)
            
    with col_low:
        with st.expander("✅ TINGKAT RENDAH", expanded=True):
            st.success("**STATUS: AMAN**")
            st.markdown("""
            **Tindakan Pemeliharaan:**
            1. 📚 **Edukasi PLTB** berkelanjutan
            2. 🌊 **Canal Blocking** maintenance
            3. 🌱 **Revegetasi** area bekas bakar
            4. 🎓 **Pelatihan MPA** simulasi
            5. 🔧 **Maintenance** pompa/selang
            6. 🗺️ **Update Peta** rawan tahunan
            7. 💚 **Jaga Gambut** tetap lembab
            8. 🤝 **Forum Desa** koordinasi
            """)
    
    # ==============================================================================
    # FOOTER
    # ==============================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>🔥 RIAU FIRE COMMAND CENTER (RFCC)</strong></p>
        <p>Powered by Machine Learning, Google Earth Engine & Physics-Informed Features</p>
        <p>© 2025 | Data: MODIS, CHIRPS | Model: KNN with Physics Features</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
