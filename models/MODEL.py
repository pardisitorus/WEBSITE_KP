import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import os

# Model Components
from sklearn.neighbors import KNeighborsClassifier

# Metrik & Setup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD & PHYSICS FEATURE
# =============================================================================
# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
file_path = os.path.join(data_dir, 'DATA_SMOTE_ENN_PRESERVATIF.csv')

print("🚀 MEMULAI MODELING SINGLE KNN...")

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"❌ File error: {e}")
    exit()

# Cleaning
def clean_numeric(x):
    if pd.isna(x):
        return 0.0
    x = str(x).strip()
    x = re.sub(r'[^0-9.]', '', x)
    try:
        return float(x)
    except:
        return 0.0

for col in ['LST_Max_2024_C', 'Rain_Max_2024_mm', 'NDVI_Max_2024']:
    df[col] = df[col].apply(clean_numeric)

df['TARGET'] = pd.to_numeric(df['TARGET'], errors='coerce').fillna(0).astype(int)

# Physics Features
LST = df['LST_Max_2024_C']
NDVI = df['NDVI_Max_2024']
Rain_Log = np.log1p(df['Rain_Max_2024_mm'])
EPS = 0.01

# Variabel sesuai kode asli
df['X1_Fuel_Dryness'] = (LST * (1 - NDVI)) * (Rain_Log + EPS)
df['X2_Thermal_Kinetic'] = LST ** 2
df['X3_Hydro_Stress'] = LST * (Rain_Log + EPS)
df = df.replace([np.inf, -np.inf], 0).fillna(0)

features = ['X1_Fuel_Dryness', 'X2_Thermal_Kinetic', 'X3_Hydro_Stress']
X = df[features]
y = df['TARGET']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =============================================================================
# 2. DEFINISI MODEL KNN
# =============================================================================
print("⚙️ Melatih Model KNN (Single)...")

# KNN (Si Juara Geometris) - Wajib pakai Pipeline Scaler
# Kita pakai tetangga=5 dan distance weighting (tetangga dekat lebih didengar)
knn_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5, weights='distance')
)

# Training
knn_model.fit(X_train, y_train)

# =============================================================================
# 3. EVALUASI HASIL
# =============================================================================
print("\n🥊 HASIL EVALUASI KNN")
print("-" * 85)
print(f"{'MODEL':<20} {'RECALL':<10} {'PRESISI':<10} {'AKURASI':<10} {'F1-SCORE':<10}")
print("-" * 85)

# Prediksi
y_pred = knn_model.predict(X_test)

# Simpan Metrik
res = {
    'Model': 'KNN (Single)',
    'Recall': recall_score(y_test, y_pred),
    'Presisi': precision_score(y_test, y_pred),
    'Akurasi': accuracy_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
}

print(f"{res['Model']:<20} {res['Recall']:.4f}      {res['Presisi']:.4f}      {res['Akurasi']:.4f}      {res['F1-Score']:.4f}")

# Save the model
models_dir = os.path.join(script_dir, '..', 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'initial_model.pkl')
joblib.dump(knn_model, model_path)
print(f"\n✅ Model saved to {model_path}")

# =============================================================================
# 4. VISUALISASI PERFORMA
# =============================================================================
# Visualisasi Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix KNN\nF1-Score {res['F1-Score']:.4f}")
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'))
plt.show()
