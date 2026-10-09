import os
from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load project YAML configuration with fallback defaults."""
    if not os.path.exists(config_path):
        return {
            "mlflow": {"experiment_name": "House_Price_Prediction"},
            "data": {"test_size": 0.2, "random_state": 42},
            "experiments": [
                {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
                {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
                {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
            ],
            "serving": {"host": "0.0.0.0", "port": 8000, "model_path": "./model"},
        }

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
