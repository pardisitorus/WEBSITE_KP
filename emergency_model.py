import pandas as pd
import numpy as np
import os
import joblib
import re
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Matikan warning agar terminal bersih
warnings.filterwarnings('ignore')

# --- 1. KONFIGURASI PATH ---
BASE_DIR = r"E:\WEBSITE_KP"
# Mengarah ke folder data
FILE_DATA = os.path.join(BASE_DIR, "data", "DATA_SMOTE_ENN_PRESERVATIF.csv")
FILE_MODEL_OUTPUT = os.path.join(BASE_DIR, "model_knn.pkl")

print(f"🚀 MEMULAI TRAINING MODEL (STRICT MODE)...")
print(f"📂 Mencari dataset di: {FILE_DATA}")

# --- 2. LOAD DATA ---
if not os.path.exists(FILE_DATA):
    print(f"❌ ERROR FATAL: File tidak ditemukan di: {FILE_DATA}")
    exit()

try:
    df = pd.read_csv(FILE_DATA)
    print("✅ Dataset berhasil dimuat.")
except Exception as e:
    print(f"❌ File error: {e}")
    exit()

# --- 3. CLEANING ---
def clean_numeric(x):
    if pd.isna(x): return 0.0
    x = str(x).strip()
    x = re.sub(r'[^0-9.]', '', x)
    try: return float(x)
    except: return 0.0

for col in ['LST_Max_2024_C', 'Rain_Max_2024_mm', 'NDVI_Max_2024']:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

if 'TARGET' in df.columns:
    df['TARGET'] = pd.to_numeric(df['TARGET'], errors='coerce').fillna(0).astype(int)
else:
    print("❌ ERROR: Kolom 'TARGET' tidak ditemukan.")
    exit()

# --- 4. PHYSICS FEATURES (DISESUAIKAN DENGAN TUNING ANDA: PEMBAGIAN) ---
LST = df['LST_Max_2024_C']
NDVI = df['NDVI_Max_2024']
Rain_Log = np.log1p(df['Rain_Max_2024_mm'])
EPS = 0.01

# [PENTING] Menggunakan PEMBAGIAN (/) agar sama dengan hasil Tuning
df['X1_Fuel_Dryness'] = (LST * (1 - NDVI)) / (Rain_Log + EPS) 
df['X2_Thermal_Kinetic'] = LST ** 2
df['X3_Hydro_Stress'] = LST / (Rain_Log + EPS)

df = df.replace([np.inf, -np.inf], 0).fillna(0)

features = ['X1_Fuel_Dryness', 'X2_Thermal_Kinetic', 'X3_Hydro_Stress']
X = df[features]
y = df['TARGET']

# --- 5. SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 6. DEFINISI MODEL ---
print("⚙️ Melatih Model KNN (Pipeline: Scaler + KNN Distance)...")

knn_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, weights='distance')
)

# Training
knn_model.fit(X_train, y_train)

# --- 7. EVALUASI ---
y_pred = knn_model.predict(X_test)

res = {
    'Recall': recall_score(y_test, y_pred),
    'Presisi': precision_score(y_test, y_pred),
    'Akurasi': accuracy_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
}

print("-" * 50)
print(f"HASIL EVALUASI (Harus sama dengan Tuning):")
print(f"F1-Score : {res['F1-Score']:.4f}")
print(f"Akurasi  : {res['Akurasi']:.4f}")
print("-" * 50)

# --- 8. SIMPAN MODEL ---
joblib.dump(knn_model, FILE_MODEL_OUTPUT)
print(f"✅ MODEL VALID DISIMPAN KE: {FILE_MODEL_OUTPUT}")