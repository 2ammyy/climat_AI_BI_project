import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import mlflow
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=UserWarning)

PREPROCESSED_PATH = Path("data/merged_preprocessed.csv")
mlflow.set_tracking_uri("http://localhost:5000")

# ────────────────────────────────────────────────
# 1. Data Loading & Proxy Label Generation
# ────────────────────────────────────────────────

def load_and_prepare_data():
    if not PREPROCESSED_PATH.exists():
        raise FileNotFoundError(f"File not found: {PREPROCESSED_PATH}")

    df = pd.read_csv(PREPROCESSED_PATH)
    df['danger_label'] = df.get('danger_label', 0).astype(int)
    y = df['danger_label'].values

    print("\n" + "═"*70)
    print(" DATASET SUMMARY ".center(70))
    print(f"Rows: {len(df):,}   Original positives: {y.sum()} ({y.mean():.4%})")

    if y.sum() == 0:
        print("No positives → generating proxy labels from weather description...")
        rain_indicators = ['rain', 'shower', 'thunderstorm', 'drizzle', 'heavy']
        mask = (
            df['weather_main'].eq('Rain') |
            df['weather_description'].str.contains('|'.join(rain_indicators), case=False, na=False)
        )
        df['danger_label'] = mask.astype(int)
        y = df['danger_label'].values
        print(f"→ Generated {y.sum()} positives ({y.mean():.4%})")

    if y.sum() < 10:
        raise ValueError("Too few positives — cannot train meaningfully")

    pos_rate = y.mean()
    print("═"*70 + "\n")
    return df, y, pos_rate


# ────────────────────────────────────────────────
# 2. Feature Preparation with Fallbacks
# ────────────────────────────────────────────────

def prepare_features(df):
    candidates = [
        'temp_max', 'temp_min', 'temp_hist',
        'humidity_percent_hist', 'precip_daily',
        'wind_speed_kmh_hist', 'city_encoded',
        'temp_max_lag1', 'precip_daily_lag1'
    ]

    fallback = {
        'temp_max': 'temp_scraped',
        'temp_min': 'temp_scraped',
        'humidity_percent_hist': 'humidity_percent_scraped',
        'wind_speed_kmh_hist': lambda d: d['wind_speed_mps'] * 3.6 if 'wind_speed_mps' in d else None,
    }

    features = []
    for c in candidates:
        if c in df.columns and df[c].notna().sum() > 20:
            features.append(c)
        elif c in fallback:
            fb = fallback[c]
            if isinstance(fb, str) and fb in df.columns and df[fb].notna().sum() > 20:
                df[f"{c}_fb"] = df[fb]
                features.append(f"{c}_fb")
            elif callable(fb):
                try:
                    series = fb(df)
                    if series.notna().sum() > 20:
                        df[f"{c}_fb"] = series
                        features.append(f"{c}_fb")
                except:
                    pass

    if not features:
        raise ValueError("No usable features found")

    print(f"Features used ({len(features)}): {', '.join(features)}")

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    return X, features


# ────────────────────────────────────────────────
# 3. Model Evaluation (safe mean calculation)
# ────────────────────────────────────────────────

def evaluate_model(X, y, model_class, params, name, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    metrics = {'acc': [], 'f1': [], 'mcc': []}

    print(f"\n→ {name}")

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        pos_te = y_te.sum()
        if y_tr.sum() == 0 or pos_te < 2:  # skip if too few test positives
            print(f"  Fold {fold} skipped (train+ {y_tr.sum()}, test+ {pos_te})")
            continue

        model = model_class(**params)
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_te, y_pred)

        metrics['acc'].append(acc)
        metrics['f1'].append(f1)
        metrics['mcc'].append(mcc)

        print(f"  Fold {fold}  test+ {pos_te:2d}/{len(y_te)}   acc {acc:.4f}  f1 {f1:.4f}  mcc {mcc:+.4f}")

    if not metrics['f1']:
        print(f"  → {name} : no valid folds")
        return {'mean_acc': 0.0, 'mean_f1': 0.0, 'mean_mcc': 0.0, 'std_f1': 0.0}

    means = {k: np.mean(v) for k, v in metrics.items() if v}
    std_f1 = np.std(metrics['f1']) if metrics['f1'] else 0.0

    print(f"  → {name} CV summary ({len(metrics['f1'])} folds)")
    print(f"     acc {means.get('acc', 0.0):.4f}   f1 {means.get('f1', 0.0):.4f}   mcc {means.get('mcc', 0.0):+.4f}")

    return {
        'mean_acc': means.get('acc', 0.0),
        'mean_f1': means.get('f1', 0.0),
        'mean_mcc': means.get('mcc', 0.0),
        'std_f1': std_f1
    }


# ────────────────────────────────────────────────
# 4. Main Training Logic
# ────────────────────────────────────────────────

def run_training():
    df, y, pos_rate = load_and_prepare_data()
    X, features = prepare_features(df)

    scale_pos = min(max(1.0 / pos_rate, 1.0), 30.0)

    models_config = {
        "RandomForest": {
            "cls": RandomForestClassifier,
            "params": {
                "n_estimators": 80,
                "max_depth": 3,
                "min_samples_leaf": 20,
                "min_samples_split": 25,
                "max_features": 0.4,
                "class_weight": "balanced_subsample",
                "random_state": 42,
                "n_jobs": -1
            }
        },
        "XGBoost": {
            "cls": XGBClassifier,
            "params": {
                "n_estimators": 30,
                "max_depth": 2,
                "learning_rate": 0.04,
                "subsample": 0.6,
                "colsample_bytree": 0.6,
                "scale_pos_weight": scale_pos,
                "reg_alpha": 3.0,
                "reg_lambda": 3.0,
                "min_child_weight": 8,
                "random_state": 42,
                "eval_metric": "logloss",
                "verbosity": 0
            }
        },
        "LightGBM": {
            "cls": LGBMClassifier,
            "params": {
                "n_estimators": 40,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.65,
                "colsample_bytree": 0.65,
                "is_unbalance": True,
                "reg_alpha": 2.0,
                "reg_lambda": 2.0,
                "min_data_in_leaf": 25,
                "min_sum_hessian_in_leaf": 12,
                "random_state": 42,
                "verbose": -1
            }
        }
    }

    results = {}

    with mlflow.start_run(run_name=f"WeatherGuard_3M_pos{pos_rate:.3f}"):
        mlflow.log_param("n_samples", len(y))
        mlflow.log_param("pos_rate", pos_rate)
        mlflow.log_param("n_positives", y.sum())
        mlflow.log_param("features", features)

        # Dummy baselines
        for strat in ["most_frequent", "stratified"]:
            res = evaluate_model(X, y, DummyClassifier, {"strategy": strat}, f"Dummy {strat}")
            mlflow.log_metric(f"dummy_{strat}_f1", res['mean_f1'])

        # Train real models
        for name, cfg in models_config.items():
            res = evaluate_model(X, y, cfg["cls"], cfg["params"], name)
            results[name] = res

            prefix = name.lower()[:3]
            mlflow.log_metrics({
                f"{prefix}_mean_acc": res['mean_acc'],
                f"{prefix}_mean_f1": res['mean_f1'],
                f"{prefix}_mean_mcc": res['mean_mcc'],
                f"{prefix}_std_f1": res['std_f1']
            })

        # Winner selection
        winner_name = max(results, key=lambda k: results[k]['mean_f1'])
        winner_res = results[winner_name]

        mlflow.log_param("best_model", winner_name)
        mlflow.log_metric("best_f1", winner_res['mean_f1'])
        mlflow.log_metric("best_mcc", winner_res['mean_mcc'])

        # Feature importances for winner (if supported)
        if winner_name in models_config:
            cfg = models_config[winner_name]
            model = cfg["cls"](**cfg["params"])
            model.fit(X, y)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                imp_dict = dict(zip(features, importances))
                mlflow.log_dict(imp_dict, "feature_importances.json")
                print("\nFeature importances (winner model):")
                for f, imp in sorted(imp_dict.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {f:28} : {imp:.4f}")

        print("\n" + "═"*70)
        print(f" WINNER: {winner_name}")
        print(f"  F1   = {winner_res['mean_f1']:.4f} (± {winner_res['std_f1']:.4f})")
        print(f"  MCC  = {winner_res['mean_mcc']:+.4f}")
        print("═"*70)

        # Example probabilities (on last 5 samples of full X)
        print("\nExample probabilities (last 5 samples):")
        print(model.predict_proba(X.iloc[-5:]))


if __name__ == "__main__":
    try:
        run_training()
        print("\nFinished. View in MLflow: http://localhost:5000")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()