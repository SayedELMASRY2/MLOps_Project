import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

from src.data import engineer_features

logger = logging.getLogger(__name__)

SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

def train_and_log_run(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    run_name: str,
    algorithm: str = "GradientBoosting",
    feature_engineering: bool = False,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    max_iter: int = 350,
    min_samples_leaf: int = 20,
    l2_regularization: float = 0.5,
    random_state: int = 42,
    **kwargs,
) -> Tuple[float, float, float, Any]:
    """
    Train a regressor and log configuration, metrics, and model artifact to MLflow.
    """
    # Apply feature engineering if requested
    if feature_engineering:
        X_train_processed = engineer_features(X_train)
        X_val_processed = engineer_features(X_val)
    else:
        X_train_processed = X_train
        X_val_processed = X_val

    with mlflow.start_run(run_name=run_name):
        logger.info(f"Training '{run_name}' using {algorithm} (FE={feature_engineering})...")

        if algorithm == "HistGradientBoosting":
            model = HistGradientBoostingRegressor(
                max_iter=max_iter,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                random_state=random_state,
            )
            mlflow.log_param("algorithm", "HistGradientBoosting")
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)
            mlflow.log_param("l2_regularization", l2_regularization)
        else:
            model = GradientBoostingRegressor(
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            mlflow.log_param("algorithm", "GradientBoosting")
            mlflow.log_param("max_depth", max_depth)

        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("feature_engineering", feature_engineering)
        mlflow.log_param("num_features", X_train_processed.shape[1])
        mlflow.log_param("random_state", random_state)

        model.fit(X_train_processed, y_train)
        predictions = model.predict(X_val_processed)

        rmse = float(np.sqrt(mean_squared_error(y_val, predictions)))
        mae = float(mean_absolute_error(y_val, predictions))
        r2 = float(r2_score(y_val, predictions))

        # Log evaluation metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log model artifact with trusted skops types
        mlflow.sklearn.log_model(
            model,
            "model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        logger.info(f"-> {run_name} Results: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        return rmse, mae, r2, model
