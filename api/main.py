import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone
import json

import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from joblib import load
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path configuration
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
KMEANS_PATH = ARTIFACTS_DIR / "kmeans_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


# Pydantic models with validation
class CustomerInput(BaseModel):
    """Customer input data for segmentation."""
    age: int = Field(..., ge=0, le=150, description="Customer age in years")
    annual_income: int = Field(..., ge=0, description="Annual income in currency units")
    spending_score: int = Field(..., ge=0, le=100, description="Spending score (0-100)")

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "annual_income": 75000,
                "spending_score": 62
            }
        }


class SegmentResponse(BaseModel):
    """Response model for segment prediction."""
    segment_id: int = Field(..., description="Predicted segment ID (0-4)")
    segment_name: str = Field(..., description="Human-readable segment name")
    confidence_score: Optional[float] = Field(None, description="Prediction confidence (distance to centroid)")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_type: str
    n_clusters: int
    features: list
    model_loaded: bool
    artifacts_path: str
    metadata: Optional[Dict] = None


# Segment mapping
SEGMENT_NAME: Dict[int, str] = {
    0: "Low Income, Low Spending",
    1: "Average Customer",
    2: "High Income, Low Spending",
    3: "High Spending, Low Income",
    4: "High Income, High Spending",
}


# Model container class for dependency injection
class ModelContainer:
    """Container for ML models and scalers."""

    def __init__(self):
        self.kmeans_model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.metadata: Optional[Dict] = None
        self.loaded: bool = False

    def load_models(self) -> None:
        """Load models from artifacts directory."""
        if not KMEANS_PATH.exists() or not SCALER_PATH.exists():
            raise RuntimeError(
                f"Artifacts not found. Expected '{KMEANS_PATH}' and '{SCALER_PATH}'. "
                "Run training script first: python train.py --input <data.csv>"
            )

        logger.info(f"Loading K-Means model from {KMEANS_PATH}")
        self.kmeans_model = load(KMEANS_PATH)

        logger.info(f"Loading StandardScaler from {SCALER_PATH}")
        self.scaler = load(SCALER_PATH)

        # Load metadata if available
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded model metadata: {self.metadata}")

        self.loaded = True
        logger.info("Models loaded successfully")

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Predict segment for given features.

        Args:
            features: Input features array

        Returns:
            Tuple of (segment_id, confidence_score)
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded")

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict cluster
        segment_id = int(self.kmeans_model.predict(features_scaled)[0])

        # Calculate confidence (inverse of distance to centroid)
        distances = self.kmeans_model.transform(features_scaled)[0]
        confidence = float(1 / (1 + distances[segment_id]))

        return segment_id, confidence


# Global model container
model_container = ModelContainer()


# Dependency injection
def get_model_container() -> ModelContainer:
    """Dependency to get the model container."""
    if not model_container.loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Service unavailable."
        )
    return model_container


# FastAPI app initialization
app = FastAPI(
    title="Customer Segmentation API",
    version="2.0.0",
    description="Production-ready ML API for customer segmentation using K-Means clustering",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.on_event("startup")
async def startup_event() -> None:
    """Load models on application startup."""
    try:
        model_container.load_models()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to load models on startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on application shutdown."""
    logger.info("Application shutting down")


@app.get("/", tags=["Health"])
def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Customer Segmentation API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """
    Health check endpoint for monitoring and load balancers.
    """
    return HealthResponse(
        status="healthy" if model_container.loaded else "unhealthy",
        model_loaded=model_container.loaded,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info(container: ModelContainer = Depends(get_model_container)) -> ModelInfoResponse:
    """
    Get information about the loaded model.
    """
    return ModelInfoResponse(
        model_type="K-Means Clustering",
        n_clusters=container.kmeans_model.n_clusters if container.kmeans_model else 0,
        features=["age", "annual_income", "spending_score"],
        model_loaded=container.loaded,
        artifacts_path=str(ARTIFACTS_DIR),
        metadata=container.metadata
    )


@app.post("/predict", response_model=SegmentResponse, tags=["Prediction"])
def predict_segment(
    payload: CustomerInput,
    container: ModelContainer = Depends(get_model_container)
) -> SegmentResponse:
    """
    Predict customer segment based on input features.

    Args:
        payload: Customer input data (age, annual_income, spending_score)

    Returns:
        Predicted segment ID, name, and confidence score

    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info(f"Prediction request: {payload.dict()}")

        # Prepare features
        features = np.array(
            [[payload.age, payload.annual_income, payload.spending_score]],
            dtype=float
        )

        # Get prediction
        segment_id, confidence = container.predict(features)
        segment_name = SEGMENT_NAME.get(segment_id, f"Segment {segment_id}")

        logger.info(f"Prediction result: segment_id={segment_id}, confidence={confidence:.4f}")

        return SegmentResponse(
            segment_id=segment_id,
            segment_name=segment_name,
            confidence_score=round(confidence, 4)
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Backwards compatibility - keep old endpoint
@app.post("/get_segment", response_model=SegmentResponse, tags=["Prediction"], deprecated=True)
def get_segment_deprecated(
    payload: CustomerInput,
    container: ModelContainer = Depends(get_model_container)
) -> SegmentResponse:
    """
    Legacy endpoint for segment prediction (deprecated, use /predict instead).
    """
    logger.warning("Deprecated endpoint /get_segment called, use /predict instead")
    return predict_segment(payload, container)
