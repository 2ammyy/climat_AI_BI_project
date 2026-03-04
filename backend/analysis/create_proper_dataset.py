# backend/analysis/create_proper_dataset.py

import pandas as pd
import glob
import os

print("="*60)
print("🌤️ CREATING PROPER WEATHER DATASET")
print("="*60)

# 1. Load historical data files
print("\n📚 Loading historical data...")
historical_files = [
    'backend/data/historical/tunisia 2016-07-14 to 2019-04-09.csv',
    'backend/data/historical/tunisia 2019-04-10 to 2021-12-31.csv',
    'backend/data/historical/tunisia 2022-01-01 to 2024-01-01.csv'
]

historical_dfs = []
for file in historical_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        print(f"  ✅ Loaded {os.path.basename(file)}: {len(df)} rows")
        historical_dfs.append(df)
    else:
        print(f"  ❌ File not found: {file}")

# Combine historical data
if historical_dfs:
    historical_df = pd.concat(historical_dfs, ignore_index=True)
    print(f"\n📊 Combined historical data: {len(historical_df)} rows")
    
    # Show sample
    print("\n📈 Historical data sample:")
    print(historical_df[['name', 'datetime', 'tempmax', 'tempmin', 'temp']].head())
else:
    print("❌ No historical data found!")
    historical_df = pd.DataFrame()

# 2. Load scraped data
print("\n📱 Loading scraped data...")
scraped_files = glob.glob('backend/data/scrapped/scrapped_data/thousand_records/by_source/*.csv')

scraped_dfs = []
for file in scraped_files:
    try:
        df = pd.read_csv(file)
        if 'temperature_c' in df.columns and df['temperature_c'].notna().any():
            print(f"  ✅ {os.path.basename(file)}: {len(df)} rows with temp data")
            scraped_dfs.append(df)
    except Exception as e:
        print(f"  ❌ Error loading {file}: {e}")

# Combine scraped data
if scraped_dfs:
    scraped_df = pd.concat(scraped_dfs, ignore_index=True)
    print(f"\n📊 Combined scraped data: {len(scraped_df)} rows")
    
    # Show sample
    print("\n📱 Scraped data sample:")
    if 'temperature_c' in scraped_df.columns:
        print(scraped_df[['city', 'temperature_c', 'wind_speed_kmh', 'condition']].head())
else:
    print("❌ No scraped data found!")
    scraped_df = pd.DataFrame()

# 3. Create unified dataset
print("\n🔄 Creating unified dataset...")

# Prepare historical data for merging
if not historical_df.empty:
    # Rename columns to match standard format
    historical_clean = historical_df.rename(columns={
        'name': 'city',
        'datetime': 'date',
        'tempmax': 'temp_max',
        'tempmin': 'temp_min',
        'temp': 'temp_avg',
        'humidity': 'humidity',
        'windspeed': 'wind_speed',
        'precip': 'precipitation'
    })
    
    # Add data source
    historical_clean['data_source'] = 'historical'
    
    # Keep only relevant columns
    keep_cols = ['city', 'date', 'temp_max', 'temp_min', 'temp_avg', 
                 'wind_speed', 'precipitation', 'humidity', 'data_source']
    historical_clean = historical_clean[[c for c in keep_cols if c in historical_clean.columns]]

# Prepare scraped data for merging
if not scraped_df.empty:
    # Rename columns
    scraped_clean = scraped_df.rename(columns={
        'city': 'city',
        'temperature_c': 'temp_avg',
        'wind_speed_kmh': 'wind_speed',
        'humidity_percent': 'humidity'
    })
    
    # Add date if not present
    if 'date' not in scraped_clean.columns:
        scraped_clean['date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    # Add data source
    scraped_clean['data_source'] = 'scraped'
    
    # Keep relevant columns
    keep_cols = ['city', 'date', 'temp_avg', 'wind_speed', 'humidity', 'data_source']
    scraped_clean = scraped_clean[[c for c in keep_cols if c in scraped_clean.columns]]

# 4. Combine and save
print("\n💾 Saving unified dataset...")

# Create final dataset
final_dfs = []
if not historical_df.empty:
    final_dfs.append(historical_clean)
if not scraped_df.empty:
    final_dfs.append(scraped_clean)

if final_dfs:
    final_df = pd.concat(final_dfs, ignore_index=True)
    
    # Generate risk levels based on weather thresholds
    print("\n⚠️ Generating risk levels...")
    
    def calculate_risk_level(row):
        risk_score = 0
        
        # Temperature risk
        if pd.notna(row.get('temp_max')):
            if row['temp_max'] > 40:
                risk_score += 4
            elif row['temp_max'] > 35:
                risk_score += 3
            elif row['temp_max'] > 30:
                risk_score += 2
            elif row['temp_max'] > 25:
                risk_score += 1
        
        # Wind risk
        if pd.notna(row.get('wind_speed')):
            if row['wind_speed'] > 80:
                risk_score += 4
            elif row['wind_speed'] > 60:
                risk_score += 3
            elif row['wind_speed'] > 40:
                risk_score += 2
            elif row['wind_speed'] > 20:
                risk_score += 1
        
        # Precipitation risk
        if pd.notna(row.get('precipitation')):
            if row['precipitation'] > 60:
                risk_score += 4
            elif row['precipitation'] > 40:
                risk_score += 3
            elif row['precipitation'] > 20:
                risk_score += 2
            elif row['precipitation'] > 10:
                risk_score += 1
        
        # Map score to risk level
        if risk_score >= 8:
            return 4  # Purple
        elif risk_score >= 6:
            return 3  # Red
        elif risk_score >= 4:
            return 2  # Orange
        elif risk_score >= 2:
            return 1  # Yellow
        else:
            return 0  # Green
    
    # Apply risk calculation
    final_df['danger_label'] = final_df.apply(calculate_risk_level, axis=1)
    
    # Risk level mapping
    risk_names = {0: 'Green', 1: 'Yellow', 2: 'Orange', 3: 'Red', 4: 'Purple'}
    final_df['risk_level'] = final_df['danger_label'].map(risk_names)
    
    # Show distribution
    print("\n📊 Risk Level Distribution:")
    risk_dist = final_df['risk_level'].value_counts()
    for risk in ['Green', 'Yellow', 'Orange', 'Red', 'Purple']:
        count = risk_dist.get(risk, 0)
        print(f"  {risk}: {count} ({count/len(final_df)*100:.1f}%)")
    
    # Save to file
    output_file = 'data/merged_data_clean.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved clean dataset to: {output_file}")
    print(f"   Total rows: {len(final_df)}")
    
else:
    print("❌ No data to save!")

print("\n✅ Done!")