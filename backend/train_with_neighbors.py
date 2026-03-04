# backend/train_with_neighbors.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_name = "neighbor_influence_experiments"

try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

print("="*60)
print("🚀 TRAINING MODEL WITH NEIGHBOR INFLUENCE FEATURES")
print("="*60)

# Load Tunisian data
print("\n📊 Loading Tunisian weather data...")
tunisia_df = pd.read_csv('data/merged_data_clean.csv')
print(f"✅ Loaded {len(tunisia_df)} records")

# Load neighbor data
print("\n🌍 Loading neighbor country data...")
neighbor_df = pd.read_csv('data/neighbor_data/neighbor_weather_latest.csv')
print(f"✅ Loaded {len(neighbor_df)} records from {neighbor_df['country'].nunique()} countries")

# Define influence zones (same as in collector)
influence_zones = {
    'northwest': ['Jendouba', 'Beja', 'Kef', 'Siliana'],
    'northeast': ['Bizerte', 'Tunis', 'Ariana', 'Ben Arous', 'Manouba', 'Nabeul', 'Zaghouan'],
    'southwest': ['Tozeur', 'Kebili', 'Tataouine', 'Gafsa'],
    'southeast': ['Medenine', 'Gabes', 'Sfax'],
    'coastal': ['Sousse', 'Monastir', 'Mahdia', 'Sfax', 'Gabes'],
    'central': ['Kairouan', 'Kasserine', 'Sidi Bouzid']
}

# Calculate average neighbor conditions
neighbor_avgs = {}
for country in ['algeria', 'libya', 'italy', 'malta']:
    country_data = neighbor_df[neighbor_df['country'] == country]
    if len(country_data) > 0:
        neighbor_avgs[country] = {
            'temp': country_data['temperature'].mean(),
            'wind': country_data['wind_speed'].mean(),
            'humidity': country_data['humidity'].mean()
        }
        print(f"  • {country}: {neighbor_avgs[country]['temp']:.1f}°C, wind {neighbor_avgs[country]['wind']:.1f} km/h")

# Add neighbor influence features to Tunisian data
print("\n🔧 Adding neighbor influence features...")

# Base features
feature_cols = ['temp_max', 'temp_min', 'temp_avg', 'wind_speed', 'humidity']

# New neighbor features
neighbor_features = [
    'algeria_temp_influence',
    'algeria_wind_influence',
    'libya_temp_influence',
    'libya_wind_influence',
    'italy_temp_influence',
    'italy_wind_influence',
    'malta_temp_influence',
    'malta_wind_influence',
    'max_neighbor_temp',
    'max_neighbor_wind'
]

# Initialize new features
for feature in neighbor_features:
    tunisia_df[feature] = 0.0

# Apply influences based on region
for region, governorates in influence_zones.items():
    mask = tunisia_df['city'].isin(governorates)
    
    if region in ['northwest', 'central'] and 'algeria' in neighbor_avgs:
        tunisia_df.loc[mask, 'algeria_temp_influence'] = neighbor_avgs['algeria']['temp']
        tunisia_df.loc[mask, 'algeria_wind_influence'] = neighbor_avgs['algeria']['wind']
    
    if region in ['southwest', 'southeast'] and 'libya' in neighbor_avgs:
        tunisia_df.loc[mask, 'libya_temp_influence'] = neighbor_avgs['libya']['temp']
        tunisia_df.loc[mask, 'libya_wind_influence'] = neighbor_avgs['libya']['wind']
    
    if region in ['northeast', 'coastal'] and 'italy' in neighbor_avgs:
        tunisia_df.loc[mask, 'italy_temp_influence'] = neighbor_avgs['italy']['temp']
        tunisia_df.loc[mask, 'italy_wind_influence'] = neighbor_avgs['italy']['wind']
    
    if region in ['coastal', 'northeast'] and 'malta' in neighbor_avgs:
        tunisia_df.loc[mask, 'malta_temp_influence'] = neighbor_avgs['malta']['temp']
        tunisia_df.loc[mask, 'malta_wind_influence'] = neighbor_avgs['malta']['wind']

# Calculate max influences
tunisia_df['max_neighbor_temp'] = tunisia_df[[
    'algeria_temp_influence', 'libya_temp_influence', 
    'italy_temp_influence', 'malta_temp_influence'
]].max(axis=1)

tunisia_df['max_neighbor_wind'] = tunisia_df[[
    'algeria_wind_influence', 'libya_wind_influence',
    'italy_wind_influence', 'malta_wind_influence'
]].max(axis=1)

print("\n📊 Sample data with neighbor features:")
print(tunisia_df[['city'] + neighbor_features[:4]].head(10))

# Prepare features for training
all_features = feature_cols + neighbor_features
X = tunisia_df[all_features].fillna(tunisia_df[all_features].mean())
y = tunisia_df['danger_label']

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

# Train models with and without neighbor features for comparison
print("\n🤖 Training models...")

models = {
    'LightGBM (with neighbors)': lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    ),
    'LightGBM (baseline)': lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
}

results = {}

for name, model in models.items():
    with mlflow.start_run(experiment_id=experiment_id, run_name=name):
        
        print(f"\n📌 {name}:")
        
        # Select features
        if 'baseline' in name:
            X_train_sel = X_train_scaled[:, :len(feature_cols)]
            X_test_sel = X_test_scaled[:, :len(feature_cols)]
            mlflow.log_param("features_used", feature_cols)
        else:
            X_train_sel = X_train_scaled
            X_test_sel = X_test_scaled
            mlflow.log_param("features_used", all_features)
            mlflow.log_param("neighbor_countries", list(neighbor_avgs.keys()))
        
        # Train
        model.fit(X_train_sel, y_train)
        
        # Evaluate
        train_acc = model.score(X_train_sel, y_train)
        test_acc = model.score(X_test_sel, y_test)
        
        print(f"  Train Accuracy: {train_acc:.3f}")
        print(f"  Test Accuracy:  {test_acc:.3f}")
        
        # Log parameters
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        
        # Log feature importance
        if hasattr(model, 'feature_importances_'):
            if 'baseline' in name:
                importances = dict(zip(feature_cols, model.feature_importances_))
            else:
                importances = dict(zip(all_features, model.feature_importances_))
            
            for feat, imp in importances.items():
                mlflow.log_metric(f"importance_{feat}", imp)
            
            print(f"  Top features: {sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        # Log model
        mlflow.sklearn.log_model(model, f"model_{name.replace(' ', '_')}")
        
        results[name] = {
            'accuracy': test_acc,
            'model': model
        }

# Compare results
print("\n" + "="*60)
print("📊 RESULTS COMPARISON")
print("="*60)

baseline_acc = results['LightGBM (baseline)']['accuracy']
neighbor_acc = results['LightGBM (with neighbors)']['accuracy']
improvement = (neighbor_acc - baseline_acc) * 100

print(f"\n📈 Baseline Model Accuracy: {baseline_acc:.3f}")
print(f"📈 Model with Neighbors Accuracy: {neighbor_acc:.3f}")
print(f"🚀 Improvement: {improvement:+.2f}%")

# Save the best model
best_model = results['LightGBM (with neighbors)']['model']
model_path = 'mlartifacts/model_with_neighbors.pkl'
joblib.dump(best_model, model_path)
joblib.dump(scaler, 'mlartifacts/scaler_with_neighbors.pkl')
joblib.dump(le, 'mlartifacts/label_encoder_with_neighbors.pkl')
joblib.dump(all_features, 'mlartifacts/features_with_neighbors.pkl')

print(f"\n✅ Best model saved to {model_path}")

# Feature importance visualization
plt.figure(figsize=(12, 6))
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]  # Top 15 features

plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [all_features[i] for i in indices], rotation=45, ha='right')
plt.title('Feature Importances - Model with Neighbor Influence')
plt.tight_layout()
plt.savefig('eda_results/neighbor_feature_importance.png')
plt.show()

print("\n✅ Training complete! Check MLflow UI at http://localhost:5000")