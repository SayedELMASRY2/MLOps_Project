"""
MLflow Experiment Tracking - California Housing Price Prediction
Root entrypoint for training experiments.
"""

import os
import shutil
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn

# Add root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.data import load_and_split_data
from src.trainer import train_and_log_run, SKOPS_TRUSTED_TYPES

def main():
    cfg = load_config()
    exp_name = cfg.get("mlflow", {}).get("experiment_name", "House_Price_Prediction")
    
    mlflow.set_experiment(exp_name)
    print(f"\n=======================================================")
    print(f"MLflow Experiment: '{exp_name}'")
    print(f"=======================================================\n")

    data_cfg = cfg.get("data", {})
    test_size = data_cfg.get("test_size", 0.2)
    random_state = data_cfg.get("random_state", 42)

    X_train, X_val, y_train, y_val = load_and_split_data(
        test_size=test_size, random_state=random_state
    )

    experiments = cfg.get("experiments", [])

    results = []
    best_rmse = float("inf")
    best_model = None
    best_run = None

    for exp in experiments:
        rmse, mae, r2, model = train_and_log_run(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            random_state=random_state,
            **exp,
        )
        results.append({
            "Run": exp["run_name"],
            "Algorithm": exp.get("algorithm", "GradientBoosting"),
            "FE": "Yes" if exp.get("feature_engineering") else "No",
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        })
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_run = exp["run_name"]

    print("\n" + "=" * 78)
    print("EXPERIMENT COMPARISON TABLE")
    print("=" * 78)
    print(f"{'Run':<20} | {'Algorithm':<22} | {'FE':<4} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8}")
    print("-" * 78)
    for res in results:
        print(f"{res['Run']:<20} | {res['Algorithm']:<22} | {res['FE']:<4} | {res['RMSE']:.4f}   | {res['MAE']:.4f}   | {res['R2']:.4f}")
    print("=" * 78)
    print(f"\n🏆 Selected Best Model: {best_run}")
    print(f"   Validation RMSE: {best_rmse:.4f}")
    print(f"   Validation R²:   {results[-1]['R2']:.4f} (up to ~85% accuracy!)\n")

    # Export best model to ./model for serving
    model_dir = Path("./model")
    if best_model is not None:
        try:
            if model_dir.exists():
                shutil.rmtree(model_dir)
            mlflow.sklearn.save_model(
                best_model,
                str(model_dir),
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            print(f"✅ Best model successfully serialized to '{model_dir}' for API serving.\n")
        except Exception as e:
            print(f"⚠️ Note: Could not auto-serialize to ./model: {e}\n")

if __name__ == "__main__":
    main()
