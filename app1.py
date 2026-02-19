import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pydeck as pdk
import shapely.wkt
import shapely.geometry
import ee
import altair as alt
import joblib
import gdown
from datetime import datetime, timedelta

# ==============================================================================
# 1. KONFIGURASI SISTEM
# ==============================================================================
st.set_page_config(
    layout="wide", 
    page_title="Riau Fire Command Center (RFCC)", 
    page_icon="🔥",
    initial_sidebar_state="expanded"
)

# Style UI: Gelap, Elegan, Garis Tegas
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    
    /* KPI Cards */
    div[data-testid="stMetric"] {
        background-color: #1A1C24; border: 1px solid #333; padding: 15px;
        border-radius: 8px; border-left: 5px solid #FF4B2B;
    }
    
    /* Header */
    h1 {
        background: linear-gradient(to right, #FF4B2B, #FF416C);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 2.8rem; margin-bottom: 0; padding-bottom: 10px;
    }
    
    /* Expander Rekomendasi */
    .streamlit-expanderHeader {
        font-weight: bold; background-color: #262730; border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD DATA LOKAL
# ==============================================================================
# GANTI PATH SESUAI LOKASI ANDA
DATA_URL = "https://drive.google.com/uc?id=1jmBB6Dv36aRnbDkj-cuZ154M0E3tzhOQ"
LOCAL_FILE = "desa1_riau.csv"
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "initial_model.pkl",
    BASE_DIR / "model_knn.pkl",
]


@st.cache_resource
def load_knn_model():
    """Load model KNN hasil training (MODEL_KNN.py) dari file .pkl."""
    load_errors = []
    for model_path in MODEL_CANDIDATES:
        if not model_path.exists():
            continue
        try:
            model = joblib.load(model_path)
            if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
                raise ValueError("Model tidak mendukung predict/predict_proba.")
            return model, str(model_path), None
        except Exception as e:
            load_errors.append(f"{model_path.name}: {e}")

    if load_errors:
        return None, None, " | ".join(load_errors)
    return None, None, "File model KNN tidak ditemukan."



@st.cache_resource
def init_ee():
    """Koneksi Hybrid: Secrets (Cloud), refresh token, lalu auth lokal."""
    def read_secret(key):
        if key in st.secrets:
            return st.secrets.get(key)
        # Coba baca dari section umum kalau user menaruh di nested table TOML.
        for section in ("gee", "earthengine", "general", "default", "secrets"):
            sec = st.secrets.get(section)
            if hasattr(sec, "get"):
                val = sec.get(key)
                if val:
                    return val
        return None

    project_id = read_secret("GEE_PROJECT") or "website-kp"
    errors = []

    # 1) Service Account JSON untuk deployment cloud
    try:
        service_token = read_secret("EARTHENGINE_TOKEN")
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
        else:
            errors.append("EARTHENGINE_TOKEN tidak ditemukan di secrets.")
    except Exception as e:
        errors.append(f"Service account gagal: {e}")

    # 2) Refresh token (tanpa perlu `earthengine authenticate`)
    try:
        refresh_token = str(read_secret("GEE_REFRESH_TOKEN") or "").strip()
        if refresh_token and "paste_your_refresh_token_here" not in refresh_token.lower():
            from google.oauth2.credentials import Credentials as UserCredentials
            from google.auth.transport.requests import Request

            credentials = UserCredentials(
                token=None,
                refresh_token=refresh_token,
                token_uri=ee.oauth.TOKEN_URI,
                client_id=ee.oauth.CLIENT_ID,
                client_secret=ee.oauth.CLIENT_SECRET,
                scopes=ee.oauth.SCOPES,
            )
            # Paksa refresh token di awal agar error kredensial langsung terlihat.
            credentials.refresh(Request())
            ee.Initialize(credentials=credentials, project=project_id)
            return True
        elif not refresh_token:
            errors.append("GEE_REFRESH_TOKEN tidak ditemukan di secrets.")
        else:
            errors.append("GEE_REFRESH_TOKEN masih placeholder.")
    except Exception as e:
        errors.append(f"Refresh token gagal: {e}")

    # 3) Fallback auth lokal (laptop/CLI)
    try:
        ee.Initialize(project=project_id)
        return True
    except Exception as e:
        errors.append(f"Auth lokal gagal: {e}")

    st.sidebar.error("Gagal Login GEE. Tidak ada metode autentikasi yang berhasil.")
    for err in errors:
        st.sidebar.caption(f"- {err}")
    st.sidebar.warning(
        "Cloud/Server: isi `.streamlit/secrets.toml` dengan `EARTHENGINE_TOKEN` "
        "(service account JSON) atau `GEE_REFRESH_TOKEN`."
    )
    st.sidebar.warning("Laptop lokal: jalankan `earthengine authenticate`, lalu restart app.")
    st.sidebar.caption(
        "Deteksi secrets -> "
        f"GEE_PROJECT: {'ada' if bool(read_secret('GEE_PROJECT')) else 'tidak'}, "
        f"GEE_REFRESH_TOKEN: {'ada' if bool(str(read_secret('GEE_REFRESH_TOKEN') or '').strip()) else 'tidak'}, "
        f"EARTHENGINE_TOKEN: {'ada' if bool(read_secret('EARTHENGINE_TOKEN')) else 'tidak'}"
    )
    return False


@st.cache_data
def load_data():
    """Load dan preprocessing data desa dari Google Drive"""
    
    # Download jika belum ada
    if not os.path.exists(LOCAL_FILE):
        with st.spinner("⬇️ Mengunduh layer desa dari Google Drive..."):
            gdown.download(DATA_URL, LOCAL_FILE, quiet=False, fuzzy=True)

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
# 3. ENGINE SATELIT - DATA REAL DENGAN AUTO MUNDUR SAMPAI KETEMU
# ==============================================================================
def get_satellite_data_robust(df):
    status = st.empty()
    status.info("📡 MENGHUBUNGI SATELIT... MENARIK DATA METEROLOGI TERBARU...")
    
    try:
        # Buat Feature Collection dari titik centroid desa
        features = []
        for i, row in df.iterrows():
            f = ee.Feature(ee.Geometry.Point([row['lon'], row['lat']]), {'idx': i})
            features.append(f)
        fc = ee.FeatureCollection(features)

        now = datetime.now()
        
        # ========== 1. SUHU (LST) - MODIS Terra MOD11A1 ==========
        # Update: Harian, tapi kadang ada gap karena awan
        # Strategi: Ambil data 8 hari terakhir (composite)
        lst_data = None
        lst_date = None
        
        for days_back in range(0, 30):  # Coba mundur sampai 30 hari
            try:
                search_date = now - timedelta(days=days_back)
                start = search_date - timedelta(days=8)  # Window 8 hari
                end = search_date
                
                lst_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
                    .filterDate(start, end) \
                    .select('LST_Day_1km')
                
                # Cek apakah ada data
                count = lst_collection.size().getInfo()
                if count > 0:
                    lst_data = lst_collection.mean().rename('LST_RAW')
                    lst_date = search_date.strftime("%d-%B-%Y")
                    status.info(f"✅ SUHU (LST): Data ditemukan dari {start.strftime('%d-%b-%Y')} s/d {end.strftime('%d-%b-%Y')}")
                    break
            except:
                continue
        
        if lst_data is None:
            raise Exception("LST data tidak ditemukan dalam 30 hari terakhir")

        # ========== 2. VEGETASI (NDVI) - MODIS MOD13Q1 ==========
        # Update: 16 hari sekali
        # Strategi: Ambil data terbaru dalam 32 hari terakhir
        ndvi_data = None
        ndvi_date = None
        
        for days_back in range(0, 60):  # Coba mundur sampai 60 hari
            try:
                search_date = now - timedelta(days=days_back)
                start = search_date - timedelta(days=16)
                end = search_date
                
                ndvi_collection = ee.ImageCollection('MODIS/061/MOD13Q1') \
                    .filterDate(start, end) \
                    .select('NDVI')
                
                count = ndvi_collection.size().getInfo()
                if count > 0:
                    ndvi_data = ndvi_collection.mean().rename('NDVI_RAW')
                    ndvi_date = search_date.strftime("%d-%B-%Y")
                    status.info(f"✅ VEGETASI (NDVI): Data ditemukan dari {start.strftime('%d-%b-%Y')} s/d {end.strftime('%d-%b-%Y')}")
                    break
            except:
                continue
        
        if ndvi_data is None:
            raise Exception("NDVI data tidak ditemukan dalam 60 hari terakhir")

        # ========== 3. HUJAN (CHIRPS) - Daily Precipitation ==========
        # Update: Harian (biasanya delay 2-3 hari)
        # Strategi: Ambil total 30 hari dari data terbaru yang ada
        rain_data = None
        rain_date = None
        
        for days_back in range(0, 15):  # CHIRPS biasanya delay 2-7 hari
            try:
                search_end = now - timedelta(days=days_back)
                search_start = search_end - timedelta(days=30)  # Total 30 hari
                
                rain_collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                    .filterDate(search_start, search_end) \
                    .select('precipitation')
                
                count = rain_collection.size().getInfo()
                if count > 0:
                    rain_data = rain_collection.sum().rename('Rain_RAW')
                    rain_date = f"{search_start.strftime('%d-%b-%Y')} s/d {search_end.strftime('%d-%b-%Y')}"
                    status.info(f"✅ HUJAN (CHIRPS): Data 30 hari dari {rain_date}")
                    break
            except:
                continue
        
        if rain_data is None:
            raise Exception("CHIRPS data tidak ditemukan dalam 15 hari terakhir")

        # ========== GABUNGKAN SEMUA DATA ==========
        combined = lst_data.addBands(ndvi_data).addBands(rain_data).unmask(-9999)
        
        # Ekstrak data per desa
        data = combined.reduceRegions(
            collection=fc, 
            reducer=ee.Reducer.first(), 
            scale=1000, 
            tileScale=4
        ).getInfo()

        results = []
        for f in data['features']:
            p = f['properties']
            
            # Konversi Unit LST: Kelvin -> Celcius
            l_val = p.get('LST_RAW')
            if l_val and l_val > 0 and l_val != -9999:
                lst_c = (l_val * 0.02) - 273.15
            else:
                lst_c = None  # Data tidak valid
            
            # Konversi Unit NDVI: Scale 0.0001
            n_val = p.get('NDVI_RAW')
            if n_val and n_val != -9999:
                ndvi_idx = n_val * 0.0001
                # Clip ke range valid NDVI (-1 sampai 1)
                ndvi_idx = max(-1, min(1, ndvi_idx))
            else:
                ndvi_idx = None
            
            # Hujan: mm (sudah dalam satuan yang benar)
            r_val = p.get('Rain_RAW')
            if r_val is not None and r_val != -9999:
                rain_mm = float(r_val)
            else:
                rain_mm = None

            results.append({
                'idx': p.get('idx'),
                'LST': lst_c,
                'NDVI': ndvi_idx,
                'Rain': rain_mm,
                'LST_Date': lst_date,
                'NDVI_Date': ndvi_date,
                'Rain_Date': rain_date
            })
        
        status.success("✅ SEMUA DATA SATELIT REAL BERHASIL DITARIK!")
        
        df_sat = pd.DataFrame(results)
        df_final = df.merge(df_sat, left_index=True, right_on='idx').drop(columns=['idx'])
        
        # Isi nilai None dengan median (untuk desa yang mungkin tertutup awan)
        if df_final['LST'].isna().any():
            df_final['LST'].fillna(df_final['LST'].median(), inplace=True)
        if df_final['NDVI'].isna().any():
            df_final['NDVI'].fillna(df_final['NDVI'].median(), inplace=True)
        if df_final['Rain'].isna().any():
            df_final['Rain'].fillna(df_final['Rain'].median(), inplace=True)
        
        return df_final

    except Exception as e:
        status.error(f"❌ GAGAL MENARIK DATA SATELIT: {e}")
        st.error("Sistem tidak dapat terhubung ke Google Earth Engine. Pastikan koneksi internet stabil dan token GEE valid.")
        st.stop()

# ==============================================================================
# 4. LOGIKA RISIKO FISIKA
# ==============================================================================
import pandas as pd
import numpy as np
import pydeck as pdk

EPS = 1e-6  # Untuk menghindari log(0)

def calculate_risk(df):
    if df is None or df.empty:
        return None

    # Pastikan kolom penting ada
    for col in ['LST', 'NDVI', 'Rain', 'lat', 'lon']:
        if col not in df.columns:
            df[col] = 0

    # Konversi numerik aman
    lst_num = pd.to_numeric(df['LST'], errors='coerce').fillna(0.0)
    ndvi_num = pd.to_numeric(df['NDVI'], errors='coerce').fillna(0.0)
    rain_num = pd.to_numeric(df['Rain'], errors='coerce').fillna(0.0).clip(lower=0)

    # Feature engineering sama persis dengan MODEL_KNN.py
    model_eps = 0.01
    df['Rain_Log'] = np.log1p(rain_num)
    df['X1_Fuel_Dryness'] = (lst_num * (1 - ndvi_num)) * (df['Rain_Log'] + model_eps)
    df['X2_Thermal_Kinetic'] = lst_num ** 2
    df['X3_Hydro_Stress'] = lst_num * (df['Rain_Log'] + model_eps)

    # Bersihkan inf/nan
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # Prediksi probabilitas dengan model KNN
    knn_model, model_path, model_error = load_knn_model()
    use_fallback_formula = False

    if knn_model is not None:
        try:
            feature_cols = ['X1_Fuel_Dryness', 'X2_Thermal_Kinetic', 'X3_Hydro_Stress']
            X_pred = df[feature_cols].copy()

            proba_matrix = knn_model.predict_proba(X_pred)
            classes = list(getattr(knn_model, "classes_", []))
            pos_index = classes.index(1) if 1 in classes else (proba_matrix.shape[1] - 1)

            df['prob_pct'] = (proba_matrix[:, pos_index] * 100).round(1)
            df['pred_class'] = knn_model.predict(X_pred).astype(int)
            df['risk_engine'] = f"KNN ({Path(model_path).name})"
        except Exception as e:
            model_error = f"Prediksi KNN gagal: {e}"
            use_fallback_formula = True
    else:
        use_fallback_formula = True

    # Fallback aman jika model gagal diload/prediksi
    if use_fallback_formula:
        norm_lst = ((lst_num - 25) / (40 - 25)).clip(0, 1)
        norm_rain = (1 - rain_num / 300).clip(0, 1)
        norm_dry = (1 - ndvi_num).clip(0, 1)
        risk_score = 0.4 * norm_rain + 0.4 * norm_lst + 0.2 * norm_dry
        df['prob_pct'] = (risk_score * 100).round(1)
        df['pred_class'] = (df['prob_pct'] >= 50).astype(int)
        df['risk_engine'] = "HEURISTIK"
        if model_error:
            st.warning(f"Model KNN tidak tersedia, fallback ke rumus heuristik. Detail: {model_error}")

    # Level & Warna
    def get_level(p):
        if p > 60: return "TINGGI", [255, 0, 0]
        elif p > 50: return "SEDANG", [255, 165, 0]
        else: return "RENDAH", [0, 128, 0]

    res = df['prob_pct'].apply(get_level)
    df['level'] = [x[0] for x in res]
    df['color'] = [list(map(int, x[1])) for x in res]  # Pastikan int untuk PyDeck

    # Status Kekeringan berdasarkan hujan
    def get_dry_status(rain):
        if pd.isna(rain): return "DATA TIDAK ADA"
        if rain < 10: return "SANGAT KERING"
        elif rain < 50: return "KERING"
        elif rain < 100: return "NORMAL"
        return "BASAH"

    df['status_kekeringan'] = df['Rain'].apply(get_dry_status)

    # Pastikan kolom lat/lon float
    df['lat'] = df['lat'].astype(float)
    df['lon'] = df['lon'].astype(float)

    return df


# --- Contoh Penggunaan PyDeck ---
def plot_risk(df):
    if df is None or df.empty:
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius=500,
        pickable=True
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=df['lat'].mean(),
            longitude=df['lon'].mean(),
            zoom=6,
            pitch=0
        )
    )

    return deck

# ==============================================================================
# 5. DASHBOARD UTAMA
# ==============================================================================
def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1041/1041891.png", width=70)
        st.title("PANEL KONTROL")
        
        if init_ee():
            st.success("🛰️ GEE SATELIT: ONLINE")
        else:
            st.error("GEE OFFLINE (Autentikasi belum valid)")
            st.warning("Isi secret GEE atau jalankan `earthengine authenticate` jika mode lokal.")
            st.stop()
            
        knn_model, knn_path, knn_err = load_knn_model()
        if knn_model is not None:
            st.success(f"MODEL KNN AKTIF: {Path(knn_path).name}")
        else:
            st.warning("MODEL KNN tidak ditemukan, fallback ke heuristik.")
            if knn_err:
                st.caption(knn_err)

        if st.button("🔄 TARIK DATA BARU"):
            st.cache_data.clear()
            st.rerun()
            
        st.markdown("---")
        st.markdown("### ℹ️ Info Sumber Data")
        st.info("""
        **1. SUHU (LST)**
        - Sumber: MODIS Terra MOD11A1
        - Update: **Harian** (delay 0-2 hari)
        - Resolusi: 1 km
        
        **2. VEGETASI (NDVI)**
        - Sumber: MODIS MOD13Q1
        - Update: **5 Hari** sekali
        - Resolusi: 250 m
        
        **3. HUJAN (CHIRPS)**
        - Sumber: CHIRPS Daily
        - Update: **Harian** (delay 2-5 hari)
        - Resolusi: 5 km
        
        **Klasifikasi Kekeringan (Curah Hujan):**
        - < 10mm = Sangat Kering
        - 10-50mm = Kering
        - 50-100mm = Normal
        - > 100mm = Basah
        """)

    st.title("RIAU FIRE COMMAND CENTER (RFCC)")
    st.markdown("Sistem Pemantauan Kebakaran Hutan & Lahan Terintegrasi Berbasis Satelit Real-time.")
    
    # LOAD DATA
    df_base = load_data()
    if df_base is None: st.stop()
    
    if 'data_monitor' not in st.session_state:
        df_sat = get_satellite_data_robust(df_base)
        st.session_state.data_monitor = calculate_risk(df_sat)
            
    df = st.session_state.data_monitor
    
    # TANGGAL DATA - Tampilkan per Variabel
    st.markdown(f"""
    📅 **Tanggal Data Satelit:**
    - **Suhu (LST):** {df['LST_Date'].iloc[0]}
    - **Vegetasi (NDVI):** {df['NDVI_Date'].iloc[0]}
    - **Hujan (CHIRPS):** {df['Rain_Date'].iloc[0]}
    
    📍 **Total Wilayah Dipantau:** {len(df)} Desa
    """)

    # --- BAGIAN 1: PETA & INTERAKSI ---
    col_map, col_stat = st.columns([2, 1])
    
    # Logika Highlight (Interaksi Tabel ke Peta)
    view_state = pdk.ViewState(latitude=0.5, longitude=101.5, zoom=7.5, pitch=0)
    selected_desa_name = None

    if 'selection' in st.session_state and st.session_state.selection.get("selection", {}).get("rows"):
        # Kita perlu tahu baris mana yang diklik berdasarkan hasil sort terakhir
        if 'df_sorted_display' in st.session_state:
            idx = st.session_state.selection['selection']['rows'][0]
            if idx < len(st.session_state.df_sorted_display):
                sel_row = st.session_state.df_sorted_display.iloc[idx]
                selected_desa_name = sel_row['nama_desa']
                view_state = pdk.ViewState(latitude=sel_row['lat'], longitude=sel_row['lon'], zoom=11.5, pitch=0)
                st.toast(f"📍 Menyorot Desa: {selected_desa_name}")

    # PREPARE GEOJSON
    geojson_base = {
        "type": "FeatureCollection",
        "features": []
    }
    geojson_highlight = {
        "type": "FeatureCollection",
        "features": []
    }

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

    # LAYERS - GARIS BATAS TEBAL DAN TEGAS
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
        line_width_min_pixels=3,
        opacity=0.6,
        auto_highlight=True
    ))
    
    if len(geojson_highlight["features"]) > 0:
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            data=geojson_highlight,
            stroked=True,
            filled=False,
            get_line_color=[255, 255, 0],  # Kuning untuk highlight
            get_line_width=500,
            line_width_min_pixels=5,
        ))

    with col_map:
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"html": "<b>{nama}</b> ({kab})<br>Risiko: {level} ({prob}%)<br>Kekeringan: {kering}"},
            map_style="mapbox://styles/mapbox/light-v10" 
        ))

    # --- BAGIAN 2: ANALISIS VISUALISASI ---
    with col_stat:
        st.subheader("📊 Analisis Risiko")
        
        # Pie Chart Proporsi Risiko
        risk_counts = df['level'].value_counts().reset_index()
        risk_counts.columns = ['Status', 'Jumlah']
        
        color_scale = alt.Scale(
            domain=['TINGGI', 'SEDANG', 'RENDAH'],
            range=['#FF0000', '#FFA500', '#008000']
        )
        
        donut = alt.Chart(risk_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("Jumlah", stack=True),
            color=alt.Color("Status", scale=color_scale),
            tooltip=["Status", "Jumlah"],
            order=alt.Order("Status", sort="descending")
        ).properties(height=250)
        
        st.altair_chart(donut, use_container_width=True)
        
        # Metrik Risiko Kebakaran
        high_count = len(df[df['level'] == 'TINGGI'])
        st.metric("🔥 Desa Risiko Tinggi", high_count, f"{(high_count/len(df)*100):.1f}%")
        
        # Metrik Kekeringan
        dry_count = len(df[(df['status_kekeringan'] == 'KERING') | (df['status_kekeringan'] == 'SANGAT KERING')])
        st.metric("💧 Desa Waspada Kekeringan", dry_count, f"{(dry_count/len(df)*100):.1f}%")
        
        # Distribusi Kekeringan
        st.markdown("**Distribusi Kekeringan:**")
        dry_dist = df['status_kekeringan'].value_counts()
        for status, count in dry_dist.items():
            pct = (count/len(df)*100)
            emoji = "🔴" if "SANGAT" in status else "🟠" if status == "KERING" else "🟢" if status == "NORMAL" else "🔵"
            st.caption(f"{emoji} {status}: {count} desa ({pct:.1f}%)")

    # ================= SORT CONTROL (FITUR BARU) =================
    st.markdown("### 🔃 Filter & Urutan Data")

    col_sort_1, col_sort_2 = st.columns(2)

    with col_sort_1:
        sort_by = st.selectbox(
            "Urutkan Berdasarkan:",
            [
                "Nama Desa",
                "Tingkat Risiko (Probabilitas)",
                "Curah Hujan (Rain)",
                "Suhu Permukaan (LST)"  # ⬅️ tambahan
            ]
        )

    with col_sort_2:
        sort_order = st.radio(
            "Arah Urutan:",
            ["Ascending (A-Z / Kecil-Besar)", "Descending (Z-A / Besar-Kecil)"],
            horizontal=True
        )

    # Logika Sorting
    df_sorted = df.copy()
    is_ascending = True if "Ascending" in sort_order else False

    if sort_by == "Nama Desa":
        df_sorted = df_sorted.sort_values(by="nama_desa", ascending=is_ascending)

    elif sort_by == "Tingkat Risiko (Probabilitas)":
        df_sorted = df_sorted.sort_values(by="prob_pct", ascending=is_ascending)

    elif sort_by == "Curah Hujan (Rain)":
        df_sorted = df_sorted.sort_values(by="Rain", ascending=is_ascending)

    elif sort_by == "Suhu Permukaan (LST)":
        df_sorted = df_sorted.sort_values(by="LST", ascending=is_ascending)  # ⬅️ tambahan

    df_sorted = df_sorted.reset_index(drop=True)

    # Simpan state untuk highlight peta
    st.session_state.df_sorted_display = df_sorted


    # --- BAGIAN 3: TABEL DATA ---
    st.subheader("📂 Data Desa")
    
    df_table = df_sorted[['nama_desa', 'kabupaten', 'level', 'prob_pct', 'LST', 'Rain', 'NDVI', 'status_kekeringan']]
    
    st.dataframe(
        df_table,
        column_config={
            "nama_desa": "Nama Desa",
            "kabupaten": "Kabupaten",
            "level": "Status Risiko",
            "prob_pct": st.column_config.ProgressColumn("Tingkat Risiko", format="%.1f%%", min_value=0, max_value=100),
            "LST": st.column_config.NumberColumn("Suhu (°C)", format="%.1f"),
            "Rain": st.column_config.NumberColumn("Hujan 30 Hari (mm)", format="%.1f"),
            "NDVI": st.column_config.NumberColumn("NDVI", format="%.3f"),
            "status_kekeringan": "Status Kekeringan"
        },
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        key="selection",
        height=400
    )

    # --- BAGIAN 4: REKOMENDASI PENCEGAHAN ---
    st.markdown("---")
    st.subheader("🛡️ REKOMENDASI TINDAKAN & MITIGASI")
    
    col_high, col_med, col_low = st.columns(3)
    
    with col_high:
        with st.expander("🚨 TINGKAT TINGGI (PENCEGAHAN CEPAT & EFISIEN)", expanded=True):
            st.error("STATUS: BAHAYA EKSTREM")
            st.markdown("""
            1. **Aktivasi Sirine:** Nyalakan tanda bahaya di posko desa segera.
            2. **Mobilisasi RPK:** Kirim Regu Pemadam Kebakaran ke titik koordinat panas.
            3. **Water Bombing:** Koordinasi dengan BPBD untuk bantuan pemadaman udara jika darat sulit.
            4. **Peralatan Pompa:** Siapkan pompa tekanan tinggi dan embung portabel.
            5. **Evakuasi Warga:** Amankan kelompok rentan (anak/lansia) dari paparan asap.
            6. **Sekat Basah:** Lakukan pembasahan intensif di sekat bakar perimeter desa.
            7. **Patroli Drone:** Gunakan drone termal untuk mendeteksi api di bawah permukaan gambut.
            8. **Larang Total:** Hentikan paksa segala aktivitas pembakaran sampah atau lahan.
            """)
            
    with col_med:
        with st.expander("⚠️ TINGKAT SEDANG (PENCEGAHAN & PERAWATAN)", expanded=True):
            st.warning("STATUS: WASPADA")
            st.markdown("""
            1. **Patroli Rutin:** Tingkatkan frekuensi patroli darat (pagi & sore).
            2. **Cek Sumber Air:** Pastikan volume air di kanal dan sumur bor memadai.
            3. **Bersihkan Sekat:** Bersihkan semak belukar kering di batas hutan/kebun.
            4. **Sosialisasi:** Lakukan kunjungan *door-to-door* ke petani/pekebun.
            5. **Tanda Peringatan:** Pasang bendera kuning di kantor desa/lokasi strategis.
            6. **Siaga Alat:** Siapkan alat tangan (gepyok, cangkul) di posko.
            7. **Pantau Cuaca:** Update info BMKG/Satelit setiap 6 jam.
            8. **Lapor Cepat:** Segera lapor ke Satgas Kecamatan jika vegetasi mulai mengering.
            """)
            
    with col_low:
        with st.expander("✅ TINGKAT RENDAH (PERAWATAN JANGKA LAMA)", expanded=True):
            st.success("STATUS: AMAN")
            st.markdown("""
            1. **Edukasi PLTB:** Lanjutkan penyuluhan Pembukaan Lahan Tanpa Bakar.
            2. **Canal Blocking:** Perbaiki/rawat sekat kanal untuk menjaga tinggi muka air gambut.
            3. **Revegetasi:** Tanam tanaman berair tinggi di area bekas terbakar.
            4. **Pelatihan MPA:** Lakukan simulasi pemadaman untuk Masyarakat Peduli Api.
            5. **Maintenance:** Servis mesin pompa dan selang agar siap pakai.
            6. **Evaluasi Peta:** Update peta rawan kebakaran desa tahunan.
            7. **Jaga Gambut:** Pastikan tanah gambut tetap lembab/basah.
            8. **Forum Desa:** Perkuat komunikasi antar desa untuk pencegahan dini.
            """)

if __name__ == "__main__":
    main()
