# backend/training/train_on_clean_data.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("🚀 TRAINING MODEL ON CLEAN DATA")
print("="*60)

# Load the clean dataset
print("\n📊 Loading clean dataset...")
df = pd.read_csv('data/merged_data_clean.csv')
print(f"✅ Loaded {len(df)} rows")

# Prepare features
feature_cols = ['temp_max', 'temp_min', 'temp_avg', 'wind_speed', 'humidity']
available_features = [col for col in feature_cols if col in df.columns]
print(f"\n🔍 Using features: {available_features}")

# Handle missing values
X = df[available_features].copy()
X = X.fillna(X.median())

# Target variable
y = df['danger_label']

# Encode target (just in case)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\n📈 Target distribution:")
for i, label in enumerate(le.classes_):
    count = sum(y_encoded == i)
    print(f"  Class {i} ({label}): {count} ({count/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\n🤖 Training models...")

models = {
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n📌 {name}:")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Test Accuracy:  {test_acc:.3f}")
    
    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

# Find best model
best_model_name = max(results, key=lambda x: results[x]['test_acc'])
best_model = results[best_model_name]['model']
best_acc = results[best_model_name]['test_acc']

print(f"\n🏆 Best Model: {best_model_name} with {best_acc:.3f} accuracy")

# Detailed evaluation
print("\n📊 Detailed Classification Report:")
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('eda_results/confusion_matrix.png')
plt.show()

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(8, 4))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [available_features[i] for i in indices], rotation=45)
    plt.title(f'Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig('eda_results/feature_importance.png')
    plt.show()

# Save the best model
print("\n💾 Saving model...")
joblib.dump(best_model, 'mlartifacts/clean_data_model.pkl')
joblib.dump(scaler, 'mlartifacts/scaler.pkl')
joblib.dump(le, 'mlartifacts/label_encoder.pkl')

print(f"✅ Model saved to: mlartifacts/clean_data_model.pkl")
print(f"✅ Scaler saved to: mlartifacts/scaler.pkl")
print(f"✅ Label encoder saved to: mlartifacts/label_encoder.pkl")

print("\n✅ Training complete!")