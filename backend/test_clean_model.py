# backend/test_model_fixed.py

import joblib
import numpy as np
import pandas as pd

print("🔮 Testing the trained model...")

# Load model and preprocessors
try:
    model = joblib.load('mlartifacts/clean_data_model.pkl')
    scaler = joblib.load('mlartifacts/scaler.pkl')
    le = joblib.load('mlartifacts/label_encoder.pkl')
    feature_cols = joblib.load('mlartifacts/feature_columns.pkl')
    print("✅ Model loaded successfully!")
    print(f"📊 Model type: {type(model).__name__}")
    
except FileNotFoundError as e:
    print(f"❌ Error loading model: {e}")
    print("Please run save_trained_model_fixed.py first")
    exit()

# Test with different weather scenarios
test_cases = [
    {"name": "🌱 Normal Spring Day", 
     "temp_max": 22, "temp_min": 15, "temp_avg": 18, "wind_speed": 15, "humidity": 65},
    
    {"name": "☀️ Hot Summer Day", 
     "temp_max": 38, "temp_min": 24, "temp_avg": 31, "wind_speed": 20, "humidity": 40},
    
    {"name": "💨 Windy Coastal Day", 
     "temp_max": 24, "temp_min": 18, "temp_avg": 21, "wind_speed": 55, "humidity": 70},
    
    {"name": "🌧️ Heavy Rain Day", 
     "temp_max": 18, "temp_min": 12, "temp_avg": 15, "wind_speed": 45, "humidity": 85},
    
    {"name": "⛈️ Severe Storm", 
     "temp_max": 26, "temp_min": 18, "temp_avg": 22, "wind_speed": 85, "humidity": 90},
    
    {"name": "🏜️ Extreme Heat Wave", 
     "temp_max": 42, "temp_min": 28, "temp_avg": 35, "wind_speed": 30, "humidity": 20},
]

print("\n" + "="*60)
print("📊 PREDICTIONS")
print("="*60)

risk_emojis = {0: '🟢', 1: '🟡', 2: '🟠', 3: '🔴', 4: '🟣'}
risk_names = {0: 'GREEN', 1: 'YELLOW', 2: 'ORANGE', 3: 'RED', 4: 'PURPLE'}

for case in test_cases:
    # Prepare features in correct order
    features = np.array([[
        case['temp_max'],
        case['temp_min'], 
        case['temp_avg'],
        case['wind_speed'],
        case['humidity']
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    
    # Get risk level and confidence
    risk_num = int(pred)
    risk_name = risk_names[risk_num]
    confidence = max(proba) * 100
    
    # Get top 3 probabilities
    top_indices = np.argsort(proba)[-3:][::-1]
    
    print(f"\n{risk_emojis[risk_num]} {case['name']}")
    print(f"   Weather: {case['temp_max']}°C max, wind {case['wind_speed']} km/h")
    print(f"   Predicted: {risk_name} (confidence: {confidence:.1f}%)")
    print(f"   Top predictions:")
    for idx in top_indices:
        if proba[idx] > 0.01:  # Only show probabilities > 1%
            print(f"     • {risk_names[idx]}: {proba[idx]*100:.1f}%")

# Test with real data from your dataset
print("\n" + "="*60)
print("📊 TESTING WITH REAL DATA SAMPLES")
print("="*60)

# Load a few samples from your dataset
df = pd.read_csv('data/merged_data_clean.csv').sample(5)

for idx, row in df.iterrows():
    features = np.array([[
        row['temp_max'],
        row['temp_min'],
        row['temp_avg'],
        row['wind_speed'],
        row['humidity']
    ]])
    
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    actual = row['danger_label']
    
    pred_name = risk_names[int(pred)]
    actual_name = risk_names[int(actual)]
    
    match = "✅" if pred == actual else "❌"
    
    print(f"\n{match} Sample from dataset:")
    print(f"   Actual risk: {actual_name}")
    print(f"   Predicted: {pred_name}")
    print(f"   Weather: T={row['temp_max']}°C, Wind={row['wind_speed']}km/h")

print("\n✅ Test complete!")