import pytest
from src.data import load_and_split_data, engineer_features, FEATURE_NAMES

def test_data_shapes_and_split():
    """Verify that dataset splits match 80/20 proportion and contain expected columns."""
    X_train, X_val, y_train, y_val = load_and_split_data(test_size=0.2, random_state=42)

    total_samples = len(X_train) + len(X_val)
    val_ratio = len(X_val) / total_samples

    assert 0.19 <= val_ratio <= 0.21, f"Validation ratio {val_ratio} not close to 0.2"
    assert list(X_train.columns) == FEATURE_NAMES
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)

def test_data_no_nulls():
    """Verify that there are no missing or null values in features or labels."""
    X_train, X_val, y_train, y_val = load_and_split_data(test_size=0.2, random_state=42)

    assert not X_train.isnull().any().any(), "X_train has nulls"
    assert not X_val.isnull().any().any(), "X_val has nulls"
    assert not y_train.isnull().any(), "y_train has nulls"
    assert not y_val.isnull().any(), "y_val has nulls"

def test_data_reproducibility():
    """Verify that random_state produces identical splits."""
    X1, _, _, _ = load_and_split_data(test_size=0.2, random_state=42)
    X2, _, _, _ = load_and_split_data(test_size=0.2, random_state=42)

    assert X1.equals(X2), "Data split is not deterministic across identical random_state"

def test_feature_engineering():
    """Verify that engineer_features adds 5 engineered columns without NaN."""
    X_train, _, _, _ = load_and_split_data(test_size=0.2, random_state=42)
    X_fe = engineer_features(X_train.iloc[:50])

    expected_new_cols = [
        "bedrooms_ratio",
        "rooms_per_occupant",
        "people_per_household",
        "dist_sf",
        "dist_la",
    ]
    for col in expected_new_cols:
        assert col in X_fe.columns, f"Missing feature: {col}"
    assert X_fe.shape[1] == len(FEATURE_NAMES) + 5
    assert not X_fe.isnull().any().any(), "Engineered features have nulls"
