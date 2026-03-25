"""Shared feature engineering for customer segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_FEATURES: List[str] = ["age", "annual_income", "spending_score"]


@dataclass
class FeatureSummary:
    """Small metadata payload for the engineered feature set."""

    base_features: List[str]
    derived_features: List[str]
    output_features: List[str]


class FeatureEngineer:
    """Create deterministic, explainable features for clustering."""

    def __init__(self) -> None:
        self.scaler: StandardScaler | None = None
        self.feature_columns: List[str] = []

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [column for column in BASE_FEATURES if column not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required feature columns: {missing}. Expected {BASE_FEATURES}."
            )

    def build_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a model-ready feature frame from raw customer inputs."""
        self._validate(df)

        frame = df[BASE_FEATURES].astype(float).copy()
        age = frame["age"]
        income = frame["annual_income"]
        spending = frame["spending_score"]

        safe_age = age.clip(lower=18.0)
        income_k = income / 1_000.0
        spending_ratio = spending / 100.0

        feature_frame = pd.DataFrame(
            {
                "age": age,
                "annual_income": income,
                "spending_score": spending,
                "income_log": np.log1p(income),
                "income_to_age": income / safe_age,
                "spending_ratio": spending_ratio,
                "spending_per_income_k": spending / income_k.clip(lower=1.0),
                "affluence_signal": np.sqrt(income.clip(lower=1.0)) * spending_ratio,
                "frugality_signal": np.sqrt(income.clip(lower=1.0)) * (1.0 - spending_ratio),
                "age_spend_interaction": age * spending_ratio,
                "premium_gap": income_k - spending,
            },
            index=frame.index,
        )

        if not self.feature_columns:
            self.feature_columns = list(feature_frame.columns)

        return feature_frame[self.feature_columns]

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Fit the scaler and transform the input frame."""
        feature_frame = self.build_feature_frame(df)
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(feature_frame)
        return feature_frame, scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw customers into engineered features."""
        feature_frame = self.build_feature_frame(df)
        if self.feature_columns:
            return feature_frame[self.feature_columns]
        return feature_frame

    def transform_for_inference(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Transform and scale a batch of inference requests."""
        if self.scaler is None or not self.feature_columns:
            raise ValueError("FeatureEngineer is not fitted. Train the model before inference.")

        feature_frame = self.transform(df)
        scaled = self.scaler.transform(feature_frame[self.feature_columns])
        return feature_frame, scaled

    def summary(self) -> Dict[str, Any]:
        """Serialize feature metadata."""
        derived_features = [column for column in self.feature_columns if column not in BASE_FEATURES]
        payload = FeatureSummary(
            base_features=BASE_FEATURES.copy(),
            derived_features=derived_features,
            output_features=self.feature_columns.copy(),
        )
        return asdict(payload)


def calculate_feature_importance(model: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
    """Estimate feature importance from cluster-center dispersion."""
    if not hasattr(model, "cluster_centers_"):
        return []

    centroid_spread = np.std(np.asarray(model.cluster_centers_), axis=0)
    importance_rows: List[Dict[str, Any]] = [
        {"feature": feature_name, "importance": round(float(score), 6)}
        for feature_name, score in zip(feature_names, centroid_spread)
    ]
    return sorted(importance_rows, key=lambda row: row["importance"], reverse=True)
