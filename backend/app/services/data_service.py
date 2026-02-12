import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(historical_path, scraped_path, output_path='data/merged_preprocessed.csv'):
    print("Loading historical data...")
    
    # Special handling for broken historical CSV (1 column issue)
    try:
        # Read without header, force column count, skip bad rows
        df_historical = pd.read_csv(
            historical_path,
            header=None,                    # Ignore any bad header
            engine='python',
            on_bad_lines='skip',
            encoding='utf-8',
            sep=',',
            quoting=csv.QUOTE_NONE,
            names=range(40)                 # Force up to 40 columns (extra for safety)
        )
        
        # Keep only first 34 columns (typical for Visual Crossing daily)
        df_historical = df_historical.iloc[:, :34]
        
        # Assign column names (based on your earlier description)
        column_names = [
            'file_source', 'city', 'date', 'tempmax', 'tempmin', 'temp',
            'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity',
            'precip', 'precipprob', 'precipcover', 'preciptype', 'snow',
            'snowdepth', 'windgust', 'windspeed', 'winddir', 'sealevelpressure',
            'cloudcover', 'visibility', 'solarradiation', 'solarenergy',
            'uvindex', 'severerisk', 'sunrise', 'sunset', 'moonphase',
            'conditions', 'description', 'icon', 'stations'
        ]
        
        if len(df_historical.columns) >= len(column_names):
            df_historical.columns = column_names + [f"extra_{i}" for i in range(len(df_historical.columns) - len(column_names))]
        else:
            df_historical.columns = column_names[:len(df_historical.columns)]
        
        print("Historical columns after fix:", df_historical.columns.tolist())
        print("Historical rows:", len(df_historical))
    except Exception as e:
        print(f"Historical load failed completely: {e}")
        return None, None

    # Load scraped (already working)
    print("Loading scraped data...")
    df_scraped = pd.read_csv(
        scraped_path,
        engine='python',
        on_bad_lines='warn',
        encoding='utf-8'

    )
    print("Scraped rows:", len(df_scraped))

    # Standardize date column (now that historical has 'date')
    df_historical['date'] = pd.to_datetime(df_historical['date'], errors='coerce', dayfirst=True)
    df_scraped['date'] = pd.to_datetime(df_scraped.get('scraped_at', df_scraped.get('batch_timestamp')), errors='coerce', dayfirst=True)

    # Rename other columns to match
    df_historical.rename(columns={
        'tempmax': 'temp_max',
        'tempmin': 'temp_min',
        'precip': 'precip_daily',
        'windspeed': 'wind_speed_kmh',
        'humidity': 'humidity_percent',
        'feelslike': 'feels_like_c'
    }, inplace=True)

    df_scraped.rename(columns={
        'temperature_c': 'temp',
        'precipitation_chance': 'precipprob',
        'wind_speed_kmh': 'wind_speed_kmh'
    }, inplace=True)

    # Add source
    df_historical['data_source'] = 'historical'
    df_scraped['data_source'] = 'scraped'

    # Merge on 'date' + 'city'
    df_merged = pd.merge(df_historical, df_scraped, on=['date', 'city'], how='outer', suffixes=('_hist', '_scraped'))

    # Clean
    df_merged.dropna(subset=['date', 'city'], inplace=True)
    df_merged = df_merged.sort_values(['city', 'date'])

    # Fill missing numerical
    num_cols = df_merged.select_dtypes(include=['float64', 'int64']).columns
    df_merged[num_cols] = df_merged.groupby('city')[num_cols].transform(lambda x: x.fillna(x.mean()))

    # Feature engineering
    for col in ['temp_max', 'precip_daily', 'wind_speed_kmh']:
        if col in df_merged.columns:
            df_merged[f'{col}_lag1'] = df_merged.groupby('city')[col].shift(1)
            df_merged[f'{col}_rolling3'] = df_merged.groupby('city')[col].rolling(3).mean().reset_index(0, drop=True)

    # Proxy danger label
    def is_danger(row):
        # Get precip from all possible columns (historical or scraped)
        precip = max(
            row.get('precip_daily', 0),
            row.get('precipprob_hist', 0),
            row.get('precipprob_scraped', 0),
            0
        )
        
        # Get precip probability from scraped/historical
        precip_prob = max(
            row.get('precipprob_hist', 0),
            row.get('precipprob_scraped', 0),
            0
        )
        
        # Wind from all sources (convert mps to kmh if needed)
        wind = max(
            row.get('wind_speed_kmh_hist', 0),
            row.get('wind_speed_kmh_scraped', 0),
            row.get('wind_speed_mps', 0) * 3.6,  # m/s → km/h
            0
        )
        
        # Temp from all sources
        temp = max(
            row.get('temp_max', 0),
            row.get('temp_scraped', 0),
            row.get('temp_hist', 0),
            0
        )
        
        # Tunisia-adapted thresholds
        # - Heavy rain: >20 mm/day OR high probability (>50%) with some rain (>10 mm)
        # - Strong wind: >40 km/h
        # - Extreme heat: >38°C
        if (precip > 20) or (precip_prob > 50 and precip > 10) or (wind > 40) or (temp > 38):
            return 1
        
        return 0

    df_merged['danger_label'] = df_merged.apply(is_danger, axis=1)

    # Encode city
    le = LabelEncoder()
    df_merged['city_encoded'] = le.fit_transform(df_merged['city'])

    # Scale numerical
    scaler = StandardScaler()
    scaled_cols = [col for col in num_cols if col in df_merged.columns and col != 'danger_label']
    df_merged[scaled_cols] = scaler.fit_transform(df_merged[scaled_cols])

    # Save
    df_merged.to_csv(output_path, index=False)
    print(f"Preprocessed data saved: {output_path}")
    print(f"Final shape: {df_merged.shape}")

    # Time-based split
    df_merged = df_merged.sort_values('date')
    train_size = int(len(df_merged) * 0.8)
    train_df = df_merged.iloc[:train_size]
    test_df = df_merged.iloc[train_size:]

    return train_df, test_df

# Run in main.py or notebook
if __name__ == "__main__":
    historical_path = r"backend\data\historical\combined_historical_data.csv"
    scraped_path = r"backend\data\scrapped\scrapped_data\thousand_records\weather_1000plus_20260204_230100.csv"
    train_df, test_df = preprocess_data(historical_path, scraped_path)
    print("Preprocessing done! Train shape:", train_df.shape)