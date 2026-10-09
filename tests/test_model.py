import pytest
import mlflow
from src.data import load_and_split_data
from src.trainer import train_and_log_run

def test_model_training_and_metrics():
    """Verify that model trains, outputs positive RMSE/MAE, and sensible R2."""
    X_train, X_val, y_train, y_val = load_and_split_data(test_size=0.2, random_state=42)

    # Use smaller slice for fast unit test
    X_tr_slice = X_train.iloc[:500]
    y_tr_slice = y_train.iloc[:500]
    X_vl_slice = X_val.iloc[:100]
    y_vl_slice = y_val.iloc[:100]

    mlflow.set_experiment("Test_Experiment")

    rmse, mae, r2, model = train_and_log_run(
        X_train=X_tr_slice,
        X_val=X_vl_slice,
        y_train=y_tr_slice,
        y_val=y_vl_slice,
        run_name="Unit_Test_Run",
        max_depth=3,
        learning_rate=0.1,
    )

    assert rmse > 0.0, "RMSE must be positive"
    assert mae > 0.0, "MAE must be positive"
    assert -1.0 <= r2 <= 1.0, f"R2 {r2} is outside plausible range"
    assert hasattr(model, "predict"), "Model must implement predict method"
