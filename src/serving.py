import os
import logging
from contextlib import asynccontextmanager
from typing import Optional
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import mlflow.sklearn

from src.data import engineer_features

logger = logging.getLogger(__name__)

# Global model state
_model = None

class HousingFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")

class PredictionResponse(BaseModel):
    prediction: str
    raw_value: float
    currency: str = "USD"
    features_used: int

def get_model():
    return _model

def set_model(model):
    global _model
    _model = model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global _model
    model_path = os.getenv("MODEL_PATH", "./model")
    if os.path.exists(model_path):
        try:
            _model = mlflow.sklearn.load_model(model_path)
            logger.info(f"Model successfully loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load model from {model_path}: {e}")
    else:
        logger.warning(f"Model path {model_path} not found.")
    yield
    # Shutdown
    _model = None

def create_app() -> FastAPI:
    app = FastAPI(
        title="California Housing Price Prediction API",
        description="Production MLOps API for serving California Housing model.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["Monitoring"])
    def health_check():
        return {
            "status": "healthy",
            "model_loaded": _model is not None,
            "version": "1.0.0",
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
    def predict(features: HousingFeatures):
        if _model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not currently loaded. Please check service health.",
            )
        try:
            data_dict = features.model_dump() if hasattr(features, "model_dump") else features.dict()
            df = pd.DataFrame([data_dict])

            # Apply automatic feature engineering if model requires 13 features
            expected_features = getattr(_model, "n_features_in_", 8)
            if expected_features == 13 or expected_features > 8:
                df = engineer_features(df)

            raw_prediction = float(_model.predict(df)[0])
            actual_price = raw_prediction * 100000.0

            return PredictionResponse(
                prediction=f"${actual_price:,.2f}",
                raw_value=round(actual_price, 2),
                currency="USD",
                features_used=df.shape[1],
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Inference error: {str(exc)}")

    return app

app = create_app()
