# backend/app/services/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import csv
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def perform_eda(file_path, dataset_name=None):
    """
    Perform EDA on a weather CSV file and save results to eda_results/
    """
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    output_dir = f"eda_results/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting EDA for: {dataset_name}")
    print(f"File: {file_path}")
    print(f"Output folder: {os.path.abspath(output_dir)}")
    print(f"{'='*60}\n")

    # ────────────────────────────────────────────────
    # 1. Load CSV – very robust handling
    # ────────────────────────────────────────────────
    df = None
    try:
        # Try standard load
        df = pd.read_csv(
            file_path,
            encoding='utf-8',
            low_memory=False,
            on_bad_lines='warn',
            engine='python',
            quoting=csv.QUOTE_MINIMAL,
            sep=','
        )
        print(f"Standard load OK: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"Standard load failed: {e}")
        try:
            # Fallback 1: skip bad + latin1
            df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin1', low_memory=False)
            print(f"Fallback 1 OK: {len(df)} rows kept")
        except Exception as e2:
            print(f"Fallback 1 failed: {e2}")
            try:
                # Fallback 2: force no header + fixed columns (for broken historical)
                df = pd.read_csv(file_path, header=None, engine='python', on_bad_lines='skip')
                # Keep only first 35 columns (adjust if needed)
                df = df.iloc[:, :35]
                print(f"Fallback 2 OK: forced {len(df.columns)} columns, {len(df)} rows")
            except Exception as e3:
                print(f"All loads failed: {e3}")
                df = pd.DataFrame()  # empty DF

    # Always create at least one dummy file/plot to confirm saving works
    dummy_fig = plt.figure(figsize=(5, 3))
    plt.plot([1, 2, 3], [4, 5, 6], 'o-')
    plt.title(f"Dummy Test Plot - {dataset_name}")
    dummy_path = f"{output_dir}/dummy_test_plot.png"
    plt.savefig(dummy_path, dpi=100)
    plt.close()
    print(f"→ Dummy plot created: {os.path.abspath(dummy_path)}")

    with open(f"{output_dir}/dummy_test.txt", "w", encoding="utf-8") as f:
        f.write("This is a test file to confirm folder writing works.")
    print(f"→ Dummy text file created: {os.path.abspath(output_dir + '/dummy_test.txt')}")

    if df.empty or len(df) == 0:
        print("No usable data loaded → only dummy files created.")
        return

    # ────────────────────────────────────────────────
    # 2. Basic info
    # ────────────────────────────────────────────────
    print("\nDataFrame Info:")
    df.info()

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    # ────────────────────────────────────────────────
    # 3. Missing values
    # ────────────────────────────────────────────────
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct.round(2)})
    print("\nMissing Values (top 20):")
    print(missing_df.head(20))

    missing_df.to_csv(f"{output_dir}/missing_values.csv", index=True)

    # Missing heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title(f"Missing Values Heatmap - {dataset_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/missing_heatmap.png", dpi=150)
    plt.close()

    # ────────────────────────────────────────────────
    # 4. Detect key columns
    # ────────────────────────────────────────────────
    temp_candidates = ['temp', 'tempmax', 'temperature_c', 'temp_max', 'feelslike']
    precip_candidates = ['precip', 'precip_daily', 'precipitation_chance', 'precipprob']
    wind_candidates = ['windspeed', 'wind_speed_kmh', 'windgust', 'wind_speed_mps']

    temp_col = next((c for c in temp_candidates if c in df.columns), None)
    precip_col = next((c for c in precip_candidates if c in df.columns), None)
    wind_col = next((c for c in wind_candidates if c in df.columns), None)

    key_vars = [c for c in [temp_col, precip_col, wind_col] if c is not None]
    print(f"\nDetected key weather variables: {key_vars}")

    # ────────────────────────────────────────────────
    # 5. Plots only if we have data
    # ────────────────────────────────────────────────
    for col in key_vars:
        if col in df.columns:
            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(df[col].dropna(), kde=True, bins=30, color='teal')
                plt.title(f"{dataset_name} - {col} Distribution")
                plt.savefig(f"{output_dir}/{col}_distribution.png", dpi=150)
                plt.close()
                print(f"Saved: {col}_distribution.png")
            except Exception as e:
                print(f"Failed to plot {col}: {e}")

    # Correlation heatmap if enough numbers
    num_df = df.select_dtypes(include=['float64', 'int64'])
    if len(num_df.columns) >= 2:
        plt.figure(figsize=(12, 10))
        sns.heatmap(num_df.corr(), annot=False, cmap='coolwarm', center=0)
        plt.title(f"{dataset_name} - Correlation Heatmap")
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=150)
        plt.close()
        print("Saved: correlation_heatmap.png")

    print(f"\nEDA finished for {dataset_name}")
    print(f"Check folder: {os.path.abspath(output_dir)}\n")

# ────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────

if __name__ == "__main__":
    HISTORICAL_PATH = r"backend\data\historical\combined_historical_data.csv"
    perform_eda(HISTORICAL_PATH, "historical_combined")

    SCRAPED_PATH = r"backend\data\scrapped\scrapped_data\thousand_records\weather_1000plus_20260204_230100.csv"
    perform_eda(SCRAPED_PATH, "scraped_1000plus")