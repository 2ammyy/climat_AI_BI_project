# backend/mlflow_setup.py
import mlflow
import mlflow.sklearn
from pathlib import Path

# Set up SQLite backend (better than file system)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create or get experiment for neighbor data
experiment_name = "neighbor_influence_experiments"
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=Path("./mlartifacts").absolute().as_uri()
    )
    print(f"✅ Created new experiment: {experiment_name} (ID: {experiment_id})")
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    print(f"✅ Using existing experiment: {experiment_name} (ID: {experiment_id})")

# List all your existing experiments
print("\n📊 Your MLflow Experiments:")
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"  • {exp.name}: {exp.experiment_id} (runs: {len(mlflow.search_runs(exp.experiment_id))})")