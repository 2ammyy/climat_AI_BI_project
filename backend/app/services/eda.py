# backend/app/services/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import csv
from datetime import datetime

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def perform_eda(file_path, dataset_name=None):
    """
    Perform full EDA on a weather CSV file and save results to eda_results/
    """
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create output directory
    output_dir = f"eda_results/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Starting EDA for: {dataset_name}")
    print(f"File: {file_path}")
    print(f"{'='*50}\n")

    # ────────────────────────────────────────────────
    # 1. Load CSV with maximum tolerance
    # ────────────────────────────────────────────────
    df = None
    try:
        # Most tolerant settings
        df = pd.read_csv(
            file_path,
            encoding='utf-8',
            engine='python',
            quoting=csv.QUOTE_NONE,           # Do not assume quotes → helps with unquoted commas
            on_bad_lines='warn',              # Print warning instead of crash
            sep=',',
            low_memory=False,
            doublequote=False,
            escapechar='\\'
        )
        print(f"Success! Loaded {len(df)} rows and {len(df.columns)} columns.")
    except Exception as e:
        print(f"Primary load failed: {e}")
        print("Trying fallback: skip bad lines + latin1 encoding...")
        try:
            df = pd.read_csv(
                file_path,
                on_bad_lines='skip',
                encoding='latin1',            # Fallback for weird characters
                low_memory=False
            )
            print(f"Fallback load successful: {len(df)} rows kept.")
        except Exception as e2:
            print(f"Complete load failure: {e2}")
            return

    if df is None or len(df) == 0:
        print("No data loaded. Check file path or content.")
        return

    # ────────────────────────────────────────────────
    # 2. Basic information
    # ────────────────────────────────────────────────
    print("\nDataFrame Info:")
    df.info()

    print("\nFirst 5 rows (raw):")
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
    # 4. Summary statistics
    # ────────────────────────────────────────────────
    print("\nNumerical Summary:")
    print(df.describe())

    print("\nCategorical / Object Summary:")
    print(df.describe(include=['object', 'category']))

    # ────────────────────────────────────────────────
    # 5. Detect datetime column (more robust)
    # ────────────────────────────────────────────────
    possible_time_cols = [col for col in df.columns if any(k in col.lower() for k in ['date', 'time', 'timestamp', 'scraped_at', 'batch_timestamp', 'generated_at'])]
    datetime_col = None

    if possible_time_cols:
        for col in possible_time_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                if df[col].notna().sum() > len(df) * 0.5:  # at least 50% valid dates
                    datetime_col = col
                    print(f"\nSelected datetime column: {datetime_col} ({df[col].notna().sum()} valid dates)")
                    break
            except:
                pass

    # ────────────────────────────────────────────────
    # 6. Key weather variables (flexible name matching)
    # ────────────────────────────────────────────────
    temp_candidates = ['temp', 'tempmax', 'temperature_c', 'temp_max', 'feelslike', 'feels_like_c']
    precip_candidates = ['precip', 'precip_daily', 'precipitation_chance', 'precipprob']
    wind_candidates = ['windspeed', 'wind_speed_kmh', 'windgust', 'wind_speed_mps']

    temp_col = next((c for c in temp_candidates if c in df.columns), None)
    precip_col = next((c for c in precip_candidates if c in df.columns), None)
    wind_col = next((c for c in wind_candidates if c in df.columns), None)

    key_vars = [c for c in [temp_col, precip_col, wind_col] if c is not None]
    print(f"\nDetected key variables: {key_vars}")

    # ────────────────────────────────────────────────
    # 7. Distributions (histograms + KDE)
    # ────────────────────────────────────────────────
    for col in key_vars:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True, bins=40, color='teal')
            plt.title(f"{dataset_name} - {col} Distribution")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{output_dir}/{col}_distribution.png", dpi=150)
            plt.close()

    # ────────────────────────────────────────────────
    # 8. Boxplots for outliers
    # ────────────────────────────────────────────────
    for col in key_vars:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=df[col], color='lightblue')
            plt.title(f"{dataset_name} - {col} Boxplot (Outliers)")
            plt.savefig(f"{output_dir}/{col}_boxplot.png", dpi=150)
            plt.close()

    # ────────────────────────────────────────────────
    # 9. Time series plots (monthly average)
    # ────────────────────────────────────────────────
    if datetime_col and pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df.set_index(datetime_col, inplace=True)

        for col in key_vars:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                try:
                    monthly = df[col].resample('ME').mean()  # 'ME' = month-end (pandas 2.2+)
                    if len(monthly) > 1:
                        plt.figure(figsize=(14, 6))
                        monthly.plot(marker='o', linestyle='-', color='darkorange')
                        plt.title(f"{dataset_name} - Monthly Average {col}")
                        plt.ylabel(col)
                        plt.grid(True, alpha=0.3)
                        plt.savefig(f"{output_dir}/{col}_monthly_timeseries.png", dpi=150)
                        plt.close()
                except Exception as e:
                    print(f"Time series plot skipped for {col}: {e}")

        df.reset_index(inplace=True)
    else:
        print("No valid datetime column found → skipping time series plots")

    # ────────────────────────────────────────────────
    # 10. Correlation heatmap (numerical only)
    # ────────────────────────────────────────────────
    num_df = df.select_dtypes(include=['float64', 'int64'])
    if len(num_df.columns) >= 2:
        plt.figure(figsize=(12, 10))
        corr = num_df.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, linewidths=0.5, fmt=".2f")
        plt.title(f"{dataset_name} - Correlation Heatmap (Numerical)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=150)
        plt.close()
    else:
        print("Not enough numerical columns for correlation heatmap")

    print(f"\nEDA finished for {dataset_name}!")
    print(f"Results saved in: {os.path.abspath(output_dir)}\n")

# ────────────────────────────────────────────────
# Run for your specific files
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # Historical
    HISTORICAL_PATH = r"backend\data\historical\combined_historical_data.csv"
    perform_eda(HISTORICAL_PATH, dataset_name="historical_combined")

    # Scraped
    SCRAPED_PATH = r"backend\data\scrapped\scrapped_data\thousand_records\weather_1000plus_20260204_230100.csv"
    perform_eda(SCRAPED_PATH, dataset_name="scraped_1000plus")