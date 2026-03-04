# backend/analysis/find_real_data.py

import pandas as pd
import os
import glob

print("🔍 SEARCHING FOR REAL DATA FILES")
print("="*60)

# Search patterns
search_paths = [
    '.',  # current directory
    '..',  # parent directory
    '../data',
    '../scraped_data',
    '../historical_data',
    '../raw_data',
    'C:/Users/VICTUS/ai_bi_project'
]

found_files = []

for path in search_paths:
    try:
        # Look for CSV files
        csv_files = glob.glob(f"{path}/**/*.csv", recursive=True)
        for file in csv_files:
            size = os.path.getsize(file)
            if size > 1000:  # Only files bigger than 1KB
                found_files.append((file, size))
                print(f"📁 Found: {file} ({size:,} bytes)")
    except:
        pass

print(f"\n✅ Found {len(found_files)} potential data files")

# Check each file for actual weather data
print("\n📊 CHECKING FILE CONTENTS")
print("="*60)

for file_path, size in found_files[:10]:  # Check first 10 files
    try:
        print(f"\n📄 {os.path.basename(file_path)} ({size:,} bytes)")
        
        # Read first few rows
        df_sample = pd.read_csv(file_path, nrows=5)
        print(f"   Columns: {list(df_sample.columns)[:5]}...")
        
        # Check for numeric weather data
        weather_cols = [col for col in df_sample.columns if any(x in col.lower() 
                        for x in ['temp', 'wind', 'precip', 'rain', 'humidity'])]
        
        if weather_cols:
            print(f"   Weather columns found: {weather_cols[:3]}")
            # Show sample values
            for col in weather_cols[:2]:
                values = df_sample[col].dropna()
                if len(values) > 0:
                    print(f"   Sample {col}: {values.iloc[0]}")
                else:
                    print(f"   ⚠️ {col} has no data")
        else:
            print("   ❌ No weather columns found")
            
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")