# backend/analysis/diagnose_data.py

import pandas as pd
import numpy as np

print("="*60)
print("🔍 DATA DIAGNOSTIC")
print("="*60)

# Load data
df = pd.read_csv('data/merged_preprocessed.csv')
print(f"\n📊 Total rows: {len(df)}")
print(f"📋 Total columns: {len(df.columns)}")

# Check which columns actually have data
print("\n📈 Columns with REAL data (non-NaN values):")
columns_with_data = []
for col in df.columns:
    non_null = df[col].count()
    if non_null > 0:
        columns_with_data.append(col)
        print(f"  ✅ {col}: {non_null} non-null values")
    else:
        print(f"  ❌ {col}: ALL NULL")

# Show first few rows of data that might have values
print("\n👀 First 5 rows of data (showing columns that might have values):")
if columns_with_data:
    print(df[columns_with_data[:10]].head())
else:
    print("  No columns with data found!")

# Check for any columns related to weather
weather_keywords = ['temp', 'wind', 'precip', 'rain', 'humidity', 'pressure']
print("\n🌤️ Weather-related columns:")
for keyword in weather_keywords:
    matching_cols = [col for col in df.columns if keyword.lower() in col.lower()]
    for col in matching_cols:
        non_null = df[col].count()
        if non_null > 0:
            print(f"  ✅ {col}: {non_null} values")
        else:
            print(f"  ❌ {col}: ALL NULL")

# Check data source columns
print("\n📡 Data source columns:")
source_cols = [col for col in df.columns if 'source' in col.lower() or 'file' in col.lower()]
for col in source_cols:
    if col in df.columns:
        print(f"  {col}: {df[col].nunique()} unique values")
        print(f"    Sample: {df[col].dropna().unique()[:5]}")

print("\n" + "="*60)
print("🔧 NEXT STEPS:")
print("="*60)
print("1. Check if your CSV file is corrupted")
print("2. Verify the data loading process")
print("3. Look at the original data files before merging")
print("4. Ensure weather data was properly scraped")