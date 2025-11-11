"""
FastAPI service for model serving.
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ml_framework.serving.predictor import Predictor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HubSpot Customer Conversion Prediction API",
    description="Predict customer conversion probability",
    version="1.0.0",
)


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================


class CompanyFeatures(BaseModel):
    """Input features for a single company."""

    id: int
    ALEXA_RANK: float = Field(..., gt=0, description="Alexa rank (must be positive)")
    EMPLOYEE_RANGE: str = Field(..., description="Employee count range")
    INDUSTRY: str = Field(..., description="Company industry")
    total_actions: float = Field(..., ge=0, description="Total actions (non-negative)")
    total_users: float = Field(..., ge=0, description="Total users (non-negative)")
    days_active: float = Field(..., ge=0, description="Days active (non-negative)")
    activity_frequency: float = Field(..., ge=0, description="Activity frequency (non-negative)")

    @field_validator("EMPLOYEE_RANGE")
    @classmethod
    def validate_employee_range(cls, v):
        """Validate employee range categories."""
        valid_ranges = [
            "1",
            "2 to 5",
            "6 to 10",
            "11 to 25",
            "26 to 50",
            "51 to 200",
            "201 to 1000",
            "1001 to 10000",
            "10,001 or more",
        ]
        if v not in valid_ranges:
            raise ValueError(f"Invalid EMPLOYEE_RANGE. Must be one of: {valid_ranges}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": 123,
                "ALEXA_RANK": 50000,
                "EMPLOYEE_RANGE": "26 to 50",
                "INDUSTRY": "COMPUTER_SOFTWARE",
                "total_actions": 150,
                "total_users": 5,
                "days_active": 30,
                "activity_frequency": 5.0,
            }
        }
    }


class PredictionRequest(BaseModel):
    """Batch prediction request."""

    companies: List[CompanyFeatures]


class PredictionResponse(BaseModel):
    """Prediction response."""

    company_id: int
    conversion_probability: float
    prediction: int  # 0 or 1
    confidence: str  # "low", "medium", "high"


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictionResponse]
    model_version: str


# ============================================================
# HELPER FUNCTION
# ============================================================


def get_latest_artifact_dir():
    """Find the most recent artifact directory."""
    import os
    from pathlib import Path

    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        logger.warning(f"Artifacts directory not found: {artifacts_dir.absolute()}")
        return None

    # Get all experiment directories (exclude hidden dirs)
    exp_dirs = [
        d
        for d in artifacts_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and d.name not in ["logs", "data_quality"]
        and (d / "models").exists()
    ]

    if not exp_dirs:
        logger.warning("No experiment directories found in artifacts/")
        return None

    # Sort by modification time, get latest
    latest = max(exp_dirs, key=os.path.getmtime)
    logger.info(f"Found latest artifact directory: {latest}")
    return latest


# ============================================================
# MODEL SERVICE CLASS
# ============================================================


class ModelService:
    """Wrapper for predictor to match API interface."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.model_version = "1.0.0"

    def predict(self, features_df):
        """Make predictions using predictor."""
        results = self.predictor.predict(features_df, return_proba=True)
        return {
            "predictions": results["prediction"].tolist(),
            "probabilities": results["probability"].tolist(),
        }

    @staticmethod
    def get_confidence(probability):
        """Get confidence level from probability."""
        if probability < 0.3 or probability > 0.7:
            return "high"
        elif probability < 0.4 or probability > 0.6:
            return "medium"
        else:
            return "low"


# ============================================================
# INITIALIZE MODEL SERVICE
# ============================================================

# Add src to path if needed
current_dir = Path(__file__).parent.parent.parent.parent
if (current_dir / "src").exists():
    sys.path.insert(0, str(current_dir / "src"))

model_service = None  # Initialize as None

# Always print to console
print("\n" + "=" * 80)
print("INITIALIZING MODEL SERVICE")
print("=" * 80)

try:
    logger.info("=" * 60)
    logger.info("INITIALIZING MODEL SERVICE")
    logger.info("=" * 60)

    print("🔍 Looking for latest model directory...")
    latest_dir = get_latest_artifact_dir()
    print(f"Found: {latest_dir}")

    if latest_dir:
        logger.info(f"Loading model from: {latest_dir}")
        print(f"Loading model from: {latest_dir}")

        predictor = Predictor.from_artifact_dir(str(latest_dir))
        print("Predictor loaded")

        model_service = ModelService(predictor)
        logger.info(f"Model loaded successfully!")
        logger.info(f"Version: {model_service.model_version}")
        logger.info("=" * 60)

        print(f"MODEL LOADED SUCCESSFULLY!")
        print(f"Model Version: {model_service.model_version}")
    else:
        logger.error("No trained models found in artifacts/")
        logger.error("   Please train a model first!")
        logger.error("=" * 60)

        print("NO MODEL FOUND!")
        print("Check artifacts/ directory for trained models")

except Exception as e:
    logger.error("=" * 60)
    logger.error(f"FAILED TO LOAD MODEL: {e}")
    logger.error("=" * 60)

    print(f"FAILED TO LOAD MODEL!")
    print(f"Error: {e}")

    import traceback

    logger.error(traceback.format_exc())
    traceback.print_exc()
    model_service = None

# Log final status
if model_service is None:
    logger.warning("API will start but predictions will not work!")
    print("API WILL START WITHOUT MODEL")
else:
    logger.info("Model service ready for predictions")
    print("MODEL SERVICE READY FOR PREDICTIONS")

print("=" * 80 + "\n")


# ============================================================
# API ENDPOINTS
# ============================================================


@app.get("/")
def root():
    """Root endpoint - basic health check."""
    return {
        "status": "healthy",
        "service": "HubSpot Customer Conversion API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Detailed health check endpoint."""
    return {
        "status": "healthy" if model_service else "unhealthy",
        "model_loaded": model_service is not None,
        "model_version": model_service.model_version if model_service else None,
    }


@app.post("/predict", response_model=BatchPredictionResponse)
def predict(request: PredictionRequest):
    """
    Make batch predictions for multiple companies.

    Args:
        request: Batch prediction request with list of companies

    Returns:
        Batch prediction response with predictions for all companies
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")

    # Convert to DataFrame
    features_df = pd.DataFrame([company.model_dump() for company in request.companies])

    company_ids = features_df["id"].tolist()
    features_df = features_df.drop("id", axis=1)

    # Make predictions
    try:
        results = model_service.predict(features_df)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Format response
    predictions = []
    for company_id, pred, prob in zip(
        company_ids, results["predictions"], results["probabilities"]
    ):
        predictions.append(
            PredictionResponse(
                company_id=company_id,
                conversion_probability=round(prob, 4),
                prediction=int(pred),
                confidence=model_service.get_confidence(prob),
            )
        )

    return BatchPredictionResponse(
        predictions=predictions, model_version=model_service.model_version
    )


@app.post("/predict/single", response_model=PredictionResponse)
def predict_single(company: CompanyFeatures):
    """
    Make prediction for a single company.

    Args:
        company: Company features

    Returns:
        Prediction response for the company
    """
    request = PredictionRequest(companies=[company])
    response = predict(request)
    return response.predictions[0]
