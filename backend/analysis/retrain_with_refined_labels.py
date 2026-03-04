# backend/analysis/retrain_with_refined_labels.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import mlflow.sklearn

# Load refined data
print("📊 Loading refined dataset...")
df = pd.read_csv('../data/merged_data_with_refined_risk.csv')

# Prepare features and target
feature_cols = ['temperature', 'wind_speed', 'precipitation', 'humidity', 'pressure']
X = df[feature_cols].fillna(df[feature_cols].mean())
y = df['refined_risk']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'lightgbm': lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    'xgboost': xgb.XGBClassifier(n_estimators=200, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=200, random_state=42)
}

best_score = 0
best_model = None
best_name = ''

print("\n🚀 Training models on refined labels...")
for name, model in models.items():
    print(f"\n📌 {name.upper()}:")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    print(f"  Train Accuracy: {train_score:.3f}")
    print(f"  Test Accuracy:  {test_score:.3f}")
    print(f"  CV Accuracy:    {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    if test_score > best_score:
        best_score = test_score
        best_model = model
        best_name = name

print(f"\n🏆 Best Model: {best_name} with {best_score:.3f} accuracy")

# Save the best model with MLflow
with mlflow.start_run(run_name=f"refined_risk_{best_name}"):
    # Log parameters
    mlflow.log_param("model_type", best_name)
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("features", feature_cols)
    
    # Log metrics
    mlflow.log_metric("test_accuracy", best_score)
    mlflow.log_metric("cv_mean", cv_scores.mean())
    
    # Log model
    mlflow.sklearn.log_model(best_model, "model")
    
    # Log label encoder and scaler
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    mlflow.log_artifact('label_encoder.pkl')
    mlflow.log_artifact('scaler.pkl')

print(f"\n✅ Model saved with MLflow run")

# Save locally as well
joblib.dump(best_model, '../mlartifacts/refined_risk_model.pkl')
joblib.dump(le, '../mlartifacts/label_encoder.pkl')
joblib.dump(scaler, '../mlartifacts/scaler.pkl')

print("✅ Model saved locally to ../mlartifacts/")