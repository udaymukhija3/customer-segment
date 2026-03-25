"""Data quality checks for customer segmentation training data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    """Structured report for dataset validation."""

    timestamp: str
    n_samples: int
    n_features: int
    passed: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _column_stats(series: pd.Series) -> Dict[str, float]:
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=0)),
    }


def validate_customer_dataframe(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    min_samples: int = 10,
) -> DataQualityReport:
    """Validate training data and collect dataset statistics."""
    errors: List[str] = []
    warnings: List[str] = []

    timestamp = datetime.now(timezone.utc).isoformat()
    n_samples = len(df)
    n_features = len(df.columns)

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return DataQualityReport(
            timestamp=timestamp,
            n_samples=n_samples,
            n_features=n_features,
            passed=False,
            errors=errors,
            warnings=warnings,
            statistics={},
        )

    numeric_frame = df[list(required_columns)].apply(pd.to_numeric, errors="coerce")
    invalid_mask = numeric_frame.isna() & ~df[list(required_columns)].isna()
    if invalid_mask.any().any():
        first_invalid_row = int(np.where(invalid_mask.any(axis=1))[0][0]) + 1
        errors.append(f"Found non-numeric values in row {first_invalid_row}.")

    if numeric_frame.isna().any().any():
        errors.append("Dataset contains missing values in required columns.")

    if np.isinf(numeric_frame.to_numpy(dtype=float)).any():
        errors.append("Dataset contains infinite values.")

    if n_samples == 0:
        errors.append("Dataset is empty.")
    elif n_samples < min_samples:
        errors.append(
            f"Dataset is too small for reliable clustering: {n_samples} rows, minimum {min_samples}."
        )

    constraints = {
        "age": {"min": 0.0, "max": 150.0},
        "annual_income": {"min": 0.0},
        "spending_score": {"min": 0.0, "max": 100.0},
    }

    for column, rule in constraints.items():
        if column not in numeric_frame:
            continue
        series = numeric_frame[column]
        if "min" in rule and (series < rule["min"]).any():
            errors.append(f"Column '{column}' contains values below {rule['min']}.")
        if "max" in rule and (series > rule["max"]).any():
            errors.append(f"Column '{column}' contains values above {rule['max']}.")

    duplicate_count = int(df.duplicated(subset=list(required_columns)).sum())
    if duplicate_count:
        warnings.append(f"Found {duplicate_count} duplicated customer records.")

    statistics: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_features": int(len(required_columns)),
        "feature_stats": {
            column: _column_stats(numeric_frame[column]) for column in required_columns
        },
        "bounds": {
            column: {
                "min": float(numeric_frame[column].min()),
                "max": float(numeric_frame[column].max()),
            }
            for column in required_columns
        },
        "medians": {
            column: float(numeric_frame[column].median()) for column in required_columns
        },
        "duplicate_count": duplicate_count,
    }

    return DataQualityReport(
        timestamp=timestamp,
        n_samples=n_samples,
        n_features=n_features,
        passed=not errors,
        errors=errors,
        warnings=warnings,
        statistics=statistics,
    )


def save_quality_report(report: DataQualityReport, path: str | Path) -> None:
    """Persist a data quality report to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
