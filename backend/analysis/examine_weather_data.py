# backend/analysis/examine_weather_data.py

import pandas as pd
import numpy as np

print("📊 Examining weather data range...")
df = pd.read_csv('data/merged_preprocessed.csv')

print(f"\n📈 Weather Statistics:")
weather_cols = ['temp_max', 'wind_speed_kmh_hist', 'precip_daily']
for col in weather_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Min: {df[col].min():.1f}")
        print(f"  Max: {df[col].max():.1f}")
        print(f"  Mean: {df[col].mean():.1f}")
        print(f"  Std: {df[col].std():.1f}")

# Check for extreme weather
print("\n🌪️ Extreme Weather Events:")
if 'temp_max' in df.columns:
    hot_days = df[df['temp_max'] > 35].shape[0]
    print(f"  Days >35°C: {hot_days}")
if 'wind_speed_kmh_hist' in df.columns:
    windy_days = df[df['wind_speed_kmh_hist'] > 50].shape[0]
    print(f"  Days with wind >50km/h: {windy_days}")
if 'precip_daily' in df.columns:
    rainy_days = df[df['precip_daily'] > 20].shape[0]
    print(f"  Days with rain >20mm: {rainy_days}")

# Check danger_label distribution by extreme weather
print("\n⚠️ Danger Labels on Extreme Weather Days:")
if hot_days > 0:
    hot_danger = df[df['temp_max'] > 35]['danger_label'].value_counts()
    print(f"  On hot days: {hot_danger.to_dict()}")
if windy_days > 0:
    windy_danger = df[df['wind_speed_kmh_hist'] > 50]['danger_label'].value_counts()
    print(f"  On windy days: {windy_danger.to_dict()}")
if rainy_days > 0:
    rainy_danger = df[df['precip_daily'] > 20]['danger_label'].value_counts()
    print(f"  On rainy days: {rainy_danger.to_dict()}")