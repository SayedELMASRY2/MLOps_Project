import pytest
from fastapi.testclient import TestClient
from src.serving import app

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    """Test health check returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_predict_success(client):
    """Test valid prediction request returns formatted price and 200 OK."""
    payload = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"].startswith("$")
    assert data["raw_value"] > 0
    assert data["currency"] == "USD"

def test_predict_validation_error(client):
    """Test invalid payload (missing fields) returns 422 Unprocessable Entity."""
    invalid_payload = {
        "MedInc": 8.3252,
        # Missing required fields
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
