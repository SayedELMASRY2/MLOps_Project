import logging
from typing import Tuple
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate domain-specific features:
    - bedrooms_ratio: Ratio of bedrooms to total rooms
    - rooms_per_occupant: Average rooms per household member
    - people_per_household: Population normalized by occupancy
    - dist_sf: Euclidean distance to San Francisco metropolitan coordinates (37.77, -122.42)
    - dist_la: Euclidean distance to Los Angeles metropolitan coordinates (34.05, -118.24)
    """
    d = df.copy()
    d["bedrooms_ratio"] = d["AveBedrms"] / (d["AveRooms"] + 1e-5)
    d["rooms_per_occupant"] = d["AveRooms"] / (d["AveOccup"] + 1e-5)
    d["people_per_household"] = d["Population"] / (d["AveOccup"] + 1e-5)
    d["dist_sf"] = np.sqrt((d["Latitude"] - 37.77) ** 2 + (d["Longitude"] - (-122.42)) ** 2)
    d["dist_la"] = np.sqrt((d["Latitude"] - 34.05) ** 2 + (d["Longitude"] - (-118.24)) ** 2)
    return d

def load_and_split_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Fetch the California Housing dataset and perform deterministic train/test split.
    """
    logger.info("Fetching California Housing dataset...")
    housing = fetch_california_housing()
    
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="MedHouseVal")
    
    # Assert data integrity
    assert not X.isnull().values.any(), "Input dataset contains null values!"
    assert len(X) == len(y), "Features and target lengths mismatch!"
    
    logger.info(
        f"Splitting dataset: {(1 - test_size) * 100:.0f}% train, {test_size * 100:.0f}% validation (random_state={random_state})"
    )
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
