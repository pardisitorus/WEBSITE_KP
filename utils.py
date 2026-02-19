import pandas as pd
import numpy as np
import joblib
import os
import re
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from datetime import datetime

warnings.filterwarnings('ignore')

def clean_numeric(x):
    """Clean numeric values from strings"""
    if pd.isna(x):
        return 0.0
    x = str(x).strip()
    x = re.sub(r'[^0-9.]', '', x)
    try:
        return float(x)
    except:
        return 0.0

def load_training_data(file_path):
    """Load and preprocess training data"""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

    # Clean numeric columns
    for col in ['LST_Max_2024_C', 'Rain_Max_2024_mm', 'NDVI_Max_2024']:
        df[col] = df[col].apply(clean_numeric)

    df['TARGET'] = pd.to_numeric(df['TARGET'], errors='coerce').fillna(0).astype(int)

    # Physics Features
    LST = df['LST_Max_2024_C']
    NDVI = df['NDVI_Max_2024']
    Rain_Log = np.log1p(df['Rain_Max_2024_mm'])
    EPS = 0.01

    df['X1_Fuel_Dryness'] = (LST * (1 - NDVI)) * (Rain_Log + EPS)
    df['X2_Thermal_Kinetic'] = LST ** 2
    df['X3_Hydro_Stress'] = LST * (Rain_Log + EPS)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return df

def train_model(df):
    """Train KNN model and return model with metrics"""
    features = ['X1_Fuel_Dryness', 'X2_Thermal_Kinetic', 'X3_Hydro_Stress']
    X = df[features]
    y = df['TARGET']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # KNN Model
    knn_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5, weights='distance')
    )

    # Training
    knn_model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = knn_model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    return knn_model, metrics

def save_model(model, model_path, metadata=None):
    """Save model and metadata"""
    # Save model
    joblib.dump(model, model_path)

    # Save metadata
    if metadata:
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def load_model(model_path):
    """Load model from file"""
    return joblib.load(model_path)

def predict_fire_risk(model, X):
    """Predict fire risk probabilities"""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]  # Probability of positive class
    else:
        # If no predict_proba, use decision function or just predict
        return model.predict(X).astype(float)

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability > 0.7:
        return 'high'
    elif probability > 0.5:
        return 'medium'
    else:
        return 'low'

def get_prevention_recommendations(risk_level):
    """Get prevention recommendations based on risk level"""
    recommendations = {
        'high': [
            "Evacuate immediately if fire is detected in the area",
            "Clear all flammable materials within 10 meters of buildings",
            "Install and maintain firebreaks around the village",
            "Establish emergency communication systems with local authorities",
            "Conduct regular fire drills and community training",
            "Monitor weather conditions and fire danger ratings daily",
            "Prohibit open burning during high-risk periods",
            "Maintain adequate water supply for firefighting",
            "Coordinate with local fire department for rapid response",
            "Implement no-burn policies during dry seasons"
        ],
        'medium': [
            "Clear vegetation and debris around homes regularly",
            "Create defensible space by removing flammable plants",
            "Install smoke detectors and fire extinguishers",
            "Develop family emergency plans",
            "Stay informed about local fire conditions",
            "Avoid outdoor burning during windy conditions",
            "Maintain access roads for emergency vehicles",
            "Participate in community fire prevention programs",
            "Monitor for early signs of fire",
            "Keep firefighting tools readily available"
        ],
        'low': [
            "Keep grass and vegetation around homes short",
            "Clean up dead leaves and branches regularly",
            "Store flammable materials safely",
            "Be aware of fire danger ratings",
            "Report suspicious activities that could cause fires",
            "Learn basic firefighting techniques",
            "Support local fire prevention initiatives",
            "Maintain community awareness programs",
            "Keep emergency contact numbers handy",
            "Practice safe outdoor fire use when permitted"
        ]
    }

    return recommendations.get(risk_level, [])
