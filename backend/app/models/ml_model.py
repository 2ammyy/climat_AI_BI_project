
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.prophet
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from prophet import Prophet
import warnings
import os
from pathlib import Path
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────
PREPROCESSED_PATH = Path("data/merged_preprocessed.csv")

# ────────────────────────────────────────────────
# Training & Comparison
# ────────────────────────────────────────────────
def train_and_compare():
    if not PREPROCESSED_PATH.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {PREPROCESSED_PATH}")

    print("Loading preprocessed data...")
    df = pd.read_csv(PREPROCESSED_PATH)

    # ────────────────────────────────────────────────
    # Features & target (dynamic based on available columns)
    # ────────────────────────────────────────────────
    possible_features = [
        'temp_max', 'temp_min', 'temp_hist', 'temp_scraped', 'feels_like_c_hist', 'feels_like_c_scraped',
        'humidity_percent_hist', 'humidity_percent_scraped',
        'precip_daily', 'precipprob_hist', 'precipprob_scraped',
        'wind_speed_kmh_hist', 'wind_speed_kmh_scraped', 'wind_speed_mps',
        'city_encoded', 'month', 'temp_max_lag1', 'precip_daily_lag1'
    ]

    # Only keep features that exist in the DataFrame
    features = [f for f in possible_features if f in df.columns]

    if not features:
        raise ValueError("No valid features found in DataFrame. Available: " + str(df.columns.tolist()))

    target = 'danger_label'

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {df.columns.tolist()}")

    print(f"Using features: {features}")
    print(f"Target: {target}")

    # Time-based split
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    results = {}
    best_f1 = -1
    best_model_name = None
    best_run_id = None

    with mlflow.start_run(run_name="WeatherGuard_Comparison_v2"):
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("test_size", len(test_df))
        mlflow.log_param("features_used", features)

        # Model 1: XGBoost
        print("\nTraining XGBoost...")
        xgb = XGBClassifier(n_estimators=150,learning_rate=0.08,max_depth=6,random_state=42,scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)
        results['XGBoost'] = {'accuracy': acc_xgb, 'f1': f1_xgb}
        mlflow.log_metrics({"accuracy_xgb": acc_xgb, "f1_xgb": f1_xgb})
        mlflow.sklearn.log_model(xgb, "xgboost_model")
        mlflow.log_text(classification_report(y_test, y_pred_xgb), "xgboost_report.txt")

        print(f"XGBoost → Acc: {acc_xgb:.4f} | F1: {f1_xgb:.4f}")

        if f1_xgb > best_f1:
            best_f1 = f1_xgb
            best_model_name = "XGBoost"
            best_run_id = mlflow.active_run().info.run_id

        # Model 2: RandomForest
        print("\nTraining RandomForest...")
        rf = RandomForestClassifier(n_estimators=150,max_depth=10,random_state=42,class_weight='balanced_subsample')
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
        results['RandomForest'] = {'accuracy': acc_rf, 'f1': f1_rf}
        mlflow.log_metrics({"accuracy_rf": acc_rf, "f1_rf": f1_rf})
        mlflow.sklearn.log_model(rf, "randomforest_model")
        mlflow.log_text(classification_report(y_test, y_pred_rf), "randomforest_report.txt")

        print(f"RandomForest → Acc: {acc_rf:.4f} | F1: {f1_rf:.4f}")

        if f1_rf > best_f1:
            best_f1 = f1_rf
            best_model_name = "RandomForest"
            best_run_id = mlflow.active_run().info.run_id

        # Model 3: Prophet (time-series forecast on precip_daily → derive danger)
        print("\nTraining Prophet...")
        prophet_df = train_df[['date', 'precip_daily']].dropna().rename(columns={'date': 'ds', 'precip_daily': 'y'})

        if len(prophet_df) < 2:
            print("Prophet skipped: not enough valid data (need at least 2 rows)")
            acc_prophet = 0
            f1_prophet = 0
        else:
            prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            prophet.fit(prophet_df)

            future = prophet.make_future_dataframe(periods=len(test_df))
            forecast = prophet.predict(future)
            forecast_danger = (forecast['yhat'].iloc[-len(test_df):] > 60).astype(int)
            acc_prophet = accuracy_score(y_test, forecast_danger)
            f1_prophet = f1_score(y_test, forecast_danger, zero_division=0)

        results['Prophet'] = {'accuracy': acc_prophet, 'f1': f1_prophet}
        mlflow.log_metrics({"accuracy_prophet": acc_prophet, "f1_prophet": f1_prophet})


    try:
        mlflow.sklearn.log_model(xgb, "xgboost_model")
        print("XGBoost model logged")
    except Exception as e:
        print(f"Failed to log XGBoost model: {e}")

    try:
        mlflow.sklearn.log_model(rf, "randomforest_model")
        print("RandomForest model logged")
    except Exception as e:
        print(f"Failed to log RandomForest model: {e}")

    try:
        mlflow.prophet.log_model(prophet, "prophet_model")
        print("Prophet model logged")
    except Exception as e:
        print(f"Failed to log Prophet model: {e}")

    # ────────────────────────────────────────────────
    # Summary Table
    # ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Model Comparison Results")
    print("-"*60)
    print(f"{'Model':<15} {'Accuracy':<12} {'F1-Score':<12} {'Winner'}")
    print("-"*60)
    for model, scores in results.items():
        is_best = "★ BEST" if model == best_model_name else ""
        print(f"{model:<15} {scores['accuracy']:.4f}      {scores['f1']:.4f}      {is_best}")
    print("="*60)

    print(f"\nBest model: {best_model_name} (F1 = {best_f1:.4f})")
    print(f"Run ID: {best_run_id}")
    print(f"View in MLflow: http://localhost:5000/#/experiments/0/runs/{best_run_id}")

    return results, best_model_name, best_run_id

# ────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        results, best_model, best_run = train_and_compare()
    except Exception as e:
        print(f"Training failed: {e}")