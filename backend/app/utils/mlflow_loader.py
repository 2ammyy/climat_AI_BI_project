# backend/app/utils/mlflow_loader.py
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

load_dotenv()

# Set MLflow tracking URI (your local server)
mlflow.set_tracking_uri("http://localhost:5000")

# Best run ID from MLflow UI (change this!)
BEST_RUN_ID = os.getenv("BEST_MLFLOW_RUN_ID", "your_run_id_here")  # ← paste from UI

def load_best_model():
    """
    Load the best trained model from MLflow.
    Returns: (model, model_type)
    """
    try:
        # Try XGBoost first (most likely best)
        model = mlflow.sklearn.load_model(f"runs:/{BEST_RUN_ID}/xgboost_model")
        print("Loaded XGBoost model from MLflow")
        return model, "xgboost"
    except Exception as e:
        print(f"XGBoost load failed: {e}")

    try:
        model = mlflow.sklearn.load_model(f"runs:/{BEST_RUN_ID}/randomforest_model")
        print("Loaded RandomForest model from MLflow")
        return model, "randomforest"
    except Exception as e:
        print(f"RandomForest load failed: {e}")

    try:
        model = mlflow.prophet.load_model(f"runs:/{BEST_RUN_ID}/prophet_model")
        print("Loaded Prophet model from MLflow")
        return model, "prophet"
    except Exception as e:
        print(f"Prophet load failed: {e}")

    raise RuntimeError("No model could be loaded from MLflow run " + BEST_RUN_ID)

# Load once at startup
best_model, model_type = load_best_model()
print(f"API ready - using {model_type} model from run {BEST_RUN_ID}")