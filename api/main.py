"""FastAPI application for customer segmentation inference."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel, ConfigDict, Field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.features import BASE_FEATURES, FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
KMEANS_PATH = ARTIFACTS_DIR / "kmeans_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_ENGINEER_PATH = ARTIFACTS_DIR / "feature_engineer.pkl"
SEGMENT_CATALOG_PATH = ARTIFACTS_DIR / "segment_catalog.json"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


class CustomerInput(BaseModel):
    """Single customer request payload."""

    model_config = ConfigDict(
        extra="ignore",
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "age": 35,
                "annual_income": 75000,
                "spending_score": 62,
            }
        },
    )

    age: int = Field(..., ge=0, le=150, description="Customer age in years")
    annual_income: int = Field(..., ge=0, description="Annual income in currency units")
    spending_score: int = Field(..., ge=0, le=100, description="Spending score from 0 to 100")


class SegmentResponse(BaseModel):
    """Detailed prediction response."""

    model_config = ConfigDict(protected_namespaces=())

    segment_id: int
    segment_name: str
    confidence_score: Optional[float] = None
    segment_description: str
    recommended_actions: List[str] = Field(default_factory=list)
    key_drivers: List[str] = Field(default_factory=list)
    input_flags: List[str] = Field(default_factory=list)


class BatchPredictionRequest(BaseModel):
    """Batch scoring request."""

    model_config = ConfigDict(protected_namespaces=())

    customers: List[CustomerInput] = Field(..., min_length=1, max_length=250)


class BatchPredictionResponse(BaseModel):
    """Batch scoring response."""

    model_config = ConfigDict(protected_namespaces=())

    request_count: int
    segment_counts: Dict[str, int]
    predictions: List[SegmentResponse]


class HealthResponse(BaseModel):
    """Health endpoint response."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    timestamp: str


class SegmentSummary(BaseModel):
    """Business summary for a trained segment."""

    model_config = ConfigDict(protected_namespaces=())

    segment_id: int
    segment_name: str
    segment_description: str
    recommended_actions: List[str]
    customer_share: float
    size: int
    avg_age: float
    avg_annual_income: float
    avg_spending_score: float


class SegmentCatalogResponse(BaseModel):
    """Segment listing response."""

    model_config = ConfigDict(protected_namespaces=())

    segments: List[SegmentSummary]


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    n_clusters: int
    features: List[str]
    model_loaded: bool
    artifacts_path: str
    version: str
    evaluation_metrics: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


def _parse_allowed_origins() -> List[str]:
    raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["*"]


class ModelContainer:
    """Lazy-loaded store for trained artifacts."""

    def __init__(self) -> None:
        self.kmeans_model: KMeans | None = None
        self.scaler: StandardScaler | None = None
        self.feature_engineer: FeatureEngineer | None = None
        self.metadata: Dict[str, Any] | None = None
        self.segment_catalog: Dict[int, Dict[str, Any]] = {}
        self.loaded = False

    def load_models(self) -> None:
        """Load training artifacts if they are available."""
        required_paths = [
            KMEANS_PATH,
            SCALER_PATH,
            FEATURE_ENGINEER_PATH,
            SEGMENT_CATALOG_PATH,
            METADATA_PATH,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise RuntimeError(
                "Missing required artifacts. Train the model first. Missing files: "
                + ", ".join(missing)
            )

        self.kmeans_model = load(KMEANS_PATH)
        self.scaler = load(SCALER_PATH)
        self.feature_engineer = load(FEATURE_ENGINEER_PATH)
        if self.feature_engineer.scaler is None:
            self.feature_engineer.scaler = self.scaler

        self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        catalog_entries = json.loads(SEGMENT_CATALOG_PATH.read_text(encoding="utf-8"))
        self.segment_catalog = {
            int(entry["segment_id"]): entry for entry in catalog_entries
        }
        self.loaded = True
        logger.info("Loaded customer segmentation artifacts from %s", ARTIFACTS_DIR)

    def _raw_frame(self, payloads: List[CustomerInput]) -> pd.DataFrame:
        return pd.DataFrame([payload.model_dump() for payload in payloads], columns=BASE_FEATURES)

    def _key_drivers(self, row: Dict[str, float], segment: Dict[str, Any]) -> List[str]:
        medians = (self.metadata or {}).get("data_statistics", {}).get("medians", {})
        driver_candidates: List[tuple[float, str]] = []

        labels = {
            "age": "Age",
            "annual_income": "Annual income",
            "spending_score": "Spending score",
        }
        for column, label in labels.items():
            baseline = medians.get(column, 0.0)
            value = float(row[column])
            if baseline:
                delta = ((value - baseline) / baseline) * 100.0
                direction = "above" if delta >= 0 else "below"
                statement = f"{label} is {abs(delta):.0f}% {direction} the portfolio median."
                driver_candidates.append((abs(delta), statement))
            else:
                delta = abs(value - baseline)
                driver_candidates.append((delta, f"{label} is materially different from the training baseline."))

        if segment.get("income_level") == "high":
            driver_candidates.append((15.0, "The profile aligns with one of the portfolio's high-income tiers."))
        if segment.get("spending_level") == "high":
            driver_candidates.append((15.0, "Spending behavior maps to a high-intent customer segment."))
        if segment.get("spending_level") == "low":
            driver_candidates.append((10.0, "Spending behavior skews conservative relative to the portfolio."))

        driver_candidates.sort(key=lambda item: item[0], reverse=True)
        return [statement for _, statement in driver_candidates[:3]]

    def _input_flags(self, row: Dict[str, float]) -> List[str]:
        bounds = (self.metadata or {}).get("data_statistics", {}).get("bounds", {})
        labels = {
            "age": "Age",
            "annual_income": "Annual income",
            "spending_score": "Spending score",
        }
        flags: List[str] = []

        for column, label in labels.items():
            column_bounds = bounds.get(column)
            if not column_bounds:
                continue

            value = float(row[column])
            if value < column_bounds["min"] or value > column_bounds["max"]:
                flags.append(
                    f"{label} sits outside the training range "
                    f"({column_bounds['min']} to {column_bounds['max']})."
                )

        return flags

    def predict(self, payloads: List[CustomerInput]) -> List[SegmentResponse]:
        """Score one or more customers."""
        if not self.loaded or self.kmeans_model is None or self.feature_engineer is None:
            raise RuntimeError("Models are not loaded.")

        raw_frame = self._raw_frame(payloads)
        _, features_scaled = self.feature_engineer.transform_for_inference(raw_frame)
        segment_ids = self.kmeans_model.predict(features_scaled).astype(int)
        distances = self.kmeans_model.transform(features_scaled)

        responses: List[SegmentResponse] = []
        for index, segment_id in enumerate(segment_ids):
            segment = self.segment_catalog.get(
                segment_id,
                {
                    "segment_name": f"Segment {segment_id}",
                    "segment_description": "Uncatalogued segment.",
                    "recommended_actions": [],
                },
            )
            confidence = float(1.0 / (1.0 + distances[index, segment_id]))
            raw_row = raw_frame.iloc[index].to_dict()
            responses.append(
                SegmentResponse(
                    segment_id=segment_id,
                    segment_name=segment["segment_name"],
                    confidence_score=round(confidence, 4),
                    segment_description=segment["segment_description"],
                    recommended_actions=segment.get("recommended_actions", []),
                    key_drivers=self._key_drivers(raw_row, segment),
                    input_flags=self._input_flags(raw_row),
                )
            )

        return responses


model_container = ModelContainer()


def get_model_container() -> ModelContainer:
    """FastAPI dependency for loaded model access."""
    if not model_container.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Train the service artifacts first.")
    return model_container


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model_container.load_models()
    except Exception as exc:  # pragma: no cover - exercised in runtime startup failures
        logger.error("Service started without loaded artifacts: %s", exc)
    yield


app = FastAPI(
    title="Customer Segmentation API",
    version="3.0.0",
    description="Portfolio-grade ML inference service for customer segmentation.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Overview"])
def root() -> Dict[str, Any]:
    """Basic service summary."""
    return {
        "message": "Customer Segmentation API",
        "version": app.version,
        "docs": "/docs",
        "model_loaded": model_container.loaded,
        "artifacts_path": str(ARTIFACTS_DIR),
    }


@app.get("/health", response_model=HealthResponse, tags=["Overview"])
def health_check() -> HealthResponse:
    """Health probe for dashboards and orchestration."""
    metadata = model_container.metadata or {}
    return HealthResponse(
        status="healthy" if model_container.loaded else "unhealthy",
        model_loaded=model_container.loaded,
        model_version=metadata.get("model_version"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info(container: ModelContainer = Depends(get_model_container)) -> ModelInfoResponse:
    """Return the trained model contract and evaluation summary."""
    metadata = container.metadata or {}
    feature_summary = metadata.get("feature_summary", {})
    return ModelInfoResponse(
        model_type=metadata.get("model_type", "KMeans"),
        n_clusters=int(metadata.get("hyperparameters", {}).get("n_clusters", 0)),
        features=feature_summary.get("output_features", BASE_FEATURES),
        model_loaded=container.loaded,
        artifacts_path=str(ARTIFACTS_DIR),
        version=metadata.get("model_version", app.version),
        evaluation_metrics=metadata.get("evaluation_metrics", {}),
        metadata=metadata,
    )


@app.get("/segments", response_model=SegmentCatalogResponse, tags=["Model"])
def list_segments(container: ModelContainer = Depends(get_model_container)) -> SegmentCatalogResponse:
    """List business-friendly segment summaries."""
    segments = [SegmentSummary(**segment) for segment in container.segment_catalog.values()]
    segments.sort(key=lambda item: item.segment_id)
    return SegmentCatalogResponse(segments=segments)


@app.post("/predict", response_model=SegmentResponse, tags=["Prediction"])
def predict_segment(
    payload: CustomerInput,
    container: ModelContainer = Depends(get_model_container),
) -> SegmentResponse:
    """Predict the most likely customer segment for one payload."""
    try:
        return container.predict([payload])[0]
    except ValueError as exc:
        logger.error("Validation error during prediction: %s", exc)
        raise HTTPException(status_code=422, detail=f"Invalid input: {exc}") from exc
    except Exception as exc:  # pragma: no cover - unexpected runtime path
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(
    payload: BatchPredictionRequest,
    container: ModelContainer = Depends(get_model_container),
) -> BatchPredictionResponse:
    """Predict segments for a batch of customers."""
    try:
        predictions = container.predict(payload.customers)
    except ValueError as exc:
        logger.error("Validation error during batch prediction: %s", exc)
        raise HTTPException(status_code=422, detail=f"Invalid input: {exc}") from exc
    except Exception as exc:  # pragma: no cover - unexpected runtime path
        logger.error("Batch prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    segment_counts: Dict[str, int] = {}
    for prediction in predictions:
        key = str(prediction.segment_id)
        segment_counts[key] = segment_counts.get(key, 0) + 1

    return BatchPredictionResponse(
        request_count=len(predictions),
        segment_counts=segment_counts,
        predictions=predictions,
    )


@app.post("/get_segment", response_model=SegmentResponse, tags=["Prediction"], deprecated=True)
def get_segment_deprecated(
    payload: CustomerInput,
    container: ModelContainer = Depends(get_model_container),
) -> SegmentResponse:
    """Backward-compatible prediction endpoint."""
    logger.warning("Deprecated endpoint /get_segment called, use /predict instead.")
    return predict_segment(payload, container)
