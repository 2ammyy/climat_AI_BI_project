# backend/training/save_trained_model_fixed.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("💾 SAVING TRAINED MODEL (LightGBM)")
print("="*60)

# Load the clean dataset
print("\n📊 Loading clean dataset...")
df = pd.read_csv('data/merged_data_clean.csv')
print(f"✅ Loaded {len(df)} rows")

# Prepare features
feature_cols = ['temp_max', 'temp_min', 'temp_avg', 'wind_speed', 'humidity']
print(f"\n🔍 Using features: {feature_cols}")

# Handle missing values
X = df[feature_cols].copy()
X = X.fillna(X.median())

# Target
y = df['danger_label']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\n📈 Target distribution:")
for i, label in enumerate(le.classes_):
    count = sum(y_encoded == i)
    print(f"  Class {label}: {count} ({count/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create mlartifacts directory
os.makedirs('mlartifacts', exist_ok=True)

# Train LightGBM (more stable)
print("\n🤖 Training LightGBM model...")
model = lgb.LGBMClassifier(
    n_estimators=100,  # Reduced for faster training
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    verbose=-1,
    force_col_wise=True  # More stable
)

# Train with progress
model.fit(
    X_train_scaled, 
    y_train,
    eval_set=[(X_test_scaled, y_test)],
    eval_metric='multi_logloss'
)

# Evaluate
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)
print(f"\n📊 Training accuracy: {train_acc:.3f}")
print(f"📊 Test accuracy: {test_acc:.3f}")

# Save everything
print("\n💾 Saving model and preprocessors...")
joblib.dump(model, 'mlartifacts/clean_data_model.pkl')
joblib.dump(scaler, 'mlartifacts/scaler.pkl')
joblib.dump(le, 'mlartifacts/label_encoder.pkl')
joblib.dump(feature_cols, 'mlartifacts/feature_columns.pkl')

print(f"✅ Model saved to: mlartifacts/clean_data_model.pkl")
print(f"✅ Scaler saved to: mlartifacts/scaler.pkl")
print(f"✅ Label encoder saved to: mlartifacts/label_encoder.pkl")
print(f"✅ Feature columns saved to: mlartifacts/feature_columns.pkl")

# Quick test
print("\n🔮 Quick test with sample data:")
test_cases = [
    {"name": "Normal day", "values": [22, 15, 18, 15, 65]},
    {"name": "Hot day", "values": [38, 24, 31, 20, 40]},
    {"name": "Windy day", "values": [24, 18, 21, 55, 70]},
    {"name": "Stormy day", "values": [26, 18, 22, 85, 90]},
]

for case in test_cases:
    features = np.array([case['values']])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    risk = le.inverse_transform([pred])[0]
    print(f"  {case['name']}: {risk} (confidence: {max(proba)*100:.1f}%)")

print("\n✅ Done!")