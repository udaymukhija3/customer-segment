"""Production-style training pipeline for customer segmentation."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import KMeans

from src.data_quality import (
    DataQualityReport,
    save_quality_report,
    validate_customer_dataframe,
)
from src.evaluation import (
    ModelCandidate,
    build_segment_catalog,
    collect_clustering_metrics,
    rank_candidates,
    score_candidate,
)
from src.features import BASE_FEATURES, FeatureEngineer, calculate_feature_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = BASE_FEATURES.copy()
DEFAULT_MIN_CLUSTERS = 3
DEFAULT_MAX_CLUSTERS = 8


def validate_data(data: np.ndarray) -> Dict[str, Any]:
    """Validate raw numpy training data and return descriptive statistics."""
    array = np.asarray(data, dtype=np.float64)

    if array.size == 0:
        raise ValueError("Dataset is empty")
    if array.ndim != 2 or array.shape[1] != len(REQUIRED_COLUMNS):
        raise ValueError(
            f"Dataset must be a 2D array with {len(REQUIRED_COLUMNS)} features: {REQUIRED_COLUMNS}"
        )
    if np.isnan(array).any():
        raise ValueError("Dataset contains NaN values")
    if np.isinf(array).any():
        raise ValueError("Dataset contains infinite values")
    if len(array) < 10:
        raise ValueError(
            f"Dataset too small ({len(array)} rows). Need at least 10 samples for clustering."
        )

    return {
        "n_samples": int(len(array)),
        "n_features": int(array.shape[1]),
        "feature_stats": {
            "age": {
                "min": float(array[:, 0].min()),
                "max": float(array[:, 0].max()),
                "mean": float(array[:, 0].mean()),
                "std": float(array[:, 0].std()),
            },
            "annual_income": {
                "min": float(array[:, 1].min()),
                "max": float(array[:, 1].max()),
                "mean": float(array[:, 1].mean()),
                "std": float(array[:, 1].std()),
            },
            "spending_score": {
                "min": float(array[:, 2].min()),
                "max": float(array[:, 2].max()),
                "mean": float(array[:, 2].mean()),
                "std": float(array[:, 2].std()),
            },
        },
    }


def read_dataset(csv_path: Path) -> np.ndarray:
    """Read the required columns from a CSV into a numpy array."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. Expected columns: {REQUIRED_COLUMNS}"
        )

    numeric = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_rows = numeric.isna().any(axis=1)
    if invalid_rows.any():
        first_invalid = int(np.where(invalid_rows.to_numpy())[0][0])
        row_number = first_invalid + 1
        row_payload = df.iloc[first_invalid].to_dict()
        raise ValueError(f"Invalid numeric value in row {row_number}: {row_payload}")

    return numeric.to_numpy(dtype=np.float64)


def load_customer_frame(csv_path: Path) -> pd.DataFrame:
    """Load a customer CSV into a normalized dataframe."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. Expected columns: {REQUIRED_COLUMNS}"
        )

    numeric = df[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_rows = numeric.isna().any(axis=1)
    if invalid_rows.any():
        first_invalid = int(np.where(invalid_rows.to_numpy())[0][0])
        row_number = first_invalid + 1
        row_payload = df.iloc[first_invalid].to_dict()
        raise ValueError(f"Invalid numeric value in row {row_number}: {row_payload}")

    normalized = numeric.copy()
    if "customer_id" in df.columns:
        normalized.insert(0, "customer_id", df["customer_id"].astype(str))
    else:
        generated_ids = [f"CUST-{index:05d}" for index in range(1, len(df) + 1)]
        normalized.insert(0, "customer_id", generated_ids)

    return normalized


def evaluate_model(
    kmeans: KMeans,
    features_scaled: np.ndarray,
    _: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate a fitted KMeans model."""
    return collect_clustering_metrics(kmeans, features_scaled)


def select_best_model(
    features_scaled: np.ndarray,
    random_state: int,
    stability_runs: int,
    n_clusters: Optional[int],
    min_clusters: int,
    max_clusters: int,
) -> tuple[KMeans, ModelCandidate, List[Dict[str, Any]]]:
    """Train and rank clustering candidates."""
    if n_clusters is not None:
        candidate_values = [n_clusters]
    else:
        upper_bound = min(max_clusters, len(features_scaled) - 1)
        lower_bound = max(2, min_clusters)
        candidate_values = list(range(lower_bound, upper_bound + 1))

    if not candidate_values:
        raise ValueError(
            "Unable to create model candidates. Increase the dataset size or lower the cluster range."
        )

    candidate_records: List[tuple[KMeans, ModelCandidate]] = []
    for cluster_count in candidate_values:
        logger.info("Training candidate model with %s clusters", cluster_count)
        model = KMeans(
            n_clusters=cluster_count,
            n_init=25,
            random_state=random_state,
        )
        model.fit(features_scaled)
        candidate = score_candidate(
            model=model,
            features_scaled=features_scaled,
            random_state=random_state,
            stability_runs=stability_runs,
        )
        candidate_records.append((model, candidate))

    candidate_records.sort(key=lambda item: item[1].composite_score, reverse=True)
    best_model, best_candidate = candidate_records[0]
    leaderboard = rank_candidates([candidate for _, candidate in candidate_records])
    return best_model, best_candidate, leaderboard


def build_metadata(
    input_path: Path,
    artifacts_dir: Path,
    quality_report: DataQualityReport,
    candidate: ModelCandidate,
    leaderboard: List[Dict[str, Any]],
    feature_engineer: FeatureEngineer,
    feature_importance: List[Dict[str, float]],
    segment_catalog: List[Dict[str, Any]],
    random_state: int,
    stability_runs: int,
    selection_mode: str,
    min_clusters: int,
    max_clusters: int,
) -> Dict[str, Any]:
    """Assemble training metadata for downstream consumers."""
    return {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "model_type": "KMeans",
        "model_version": "3.0.0",
        "base_features": REQUIRED_COLUMNS,
        "feature_summary": feature_engineer.summary(),
        "hyperparameters": {
            "n_clusters": candidate.n_clusters,
            "n_init": 25,
            "random_state": random_state,
            "selection_mode": selection_mode,
            "min_clusters": min_clusters,
            "max_clusters": max_clusters,
            "stability_runs": stability_runs,
        },
        "data_statistics": quality_report.statistics,
        "data_quality": quality_report.to_dict(),
        "evaluation_metrics": candidate.to_dict(),
        "leaderboard": leaderboard,
        "feature_importance": feature_importance,
        "segment_catalog": segment_catalog,
        "training_config": {
            "input_file": str(input_path),
            "artifacts_dir": str(artifacts_dir),
        },
        "artifact_manifest": [
            "kmeans_model.pkl",
            "scaler.pkl",
            "feature_engineer.pkl",
            "segment_catalog.json",
            "candidate_leaderboard.json",
            "data_quality_report.json",
            "model_metadata.json",
        ],
    }


def run_training(
    input_path: Path,
    artifacts_dir: Path,
    n_clusters: Optional[int] = None,
    min_clusters: int = DEFAULT_MIN_CLUSTERS,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    random_state: int = 42,
    stability_runs: int = 6,
) -> Dict[str, Any]:
    """Run the full training pipeline and save artifacts."""
    logger.info("=" * 72)
    logger.info("Starting customer segmentation training")
    logger.info("=" * 72)

    customers = load_customer_frame(input_path)
    quality_report = validate_customer_dataframe(
        customers[REQUIRED_COLUMNS],
        required_columns=REQUIRED_COLUMNS,
        min_samples=max(10, max_clusters + 2),
    )
    if not quality_report.passed:
        for error in quality_report.errors:
            logger.error(error)
        raise ValueError("Training data failed validation.")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_quality_report(quality_report, artifacts_dir / "data_quality_report.json")

    feature_engineer = FeatureEngineer()
    _, features_scaled = feature_engineer.fit_transform(customers[REQUIRED_COLUMNS])

    best_model, best_candidate, leaderboard = select_best_model(
        features_scaled=features_scaled,
        random_state=random_state,
        stability_runs=stability_runs,
        n_clusters=n_clusters,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )

    segment_catalog = build_segment_catalog(customers[REQUIRED_COLUMNS], best_model.labels_)
    feature_importance = calculate_feature_importance(best_model, feature_engineer.feature_columns)

    dump(best_model, artifacts_dir / "kmeans_model.pkl")
    dump(feature_engineer.scaler, artifacts_dir / "scaler.pkl")
    dump(feature_engineer, artifacts_dir / "feature_engineer.pkl")

    (artifacts_dir / "segment_catalog.json").write_text(
        json.dumps(segment_catalog, indent=2),
        encoding="utf-8",
    )
    (artifacts_dir / "candidate_leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2),
        encoding="utf-8",
    )

    metadata = build_metadata(
        input_path=input_path,
        artifacts_dir=artifacts_dir,
        quality_report=quality_report,
        candidate=best_candidate,
        leaderboard=leaderboard,
        feature_engineer=feature_engineer,
        feature_importance=feature_importance,
        segment_catalog=segment_catalog,
        random_state=random_state,
        stability_runs=stability_runs,
        selection_mode="fixed" if n_clusters is not None else "auto-search",
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )
    (artifacts_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Selected %s clusters with composite score %.4f",
        best_candidate.n_clusters,
        best_candidate.composite_score,
    )
    logger.info("Silhouette score: %.4f", best_candidate.silhouette_score)
    logger.info("Average confidence: %.4f", best_candidate.average_confidence)
    logger.info("Artifacts saved to %s", artifacts_dir)

    print("\n" + "=" * 72)
    print("TRAINING SUMMARY")
    print("=" * 72)
    print(f"Samples trained: {quality_report.statistics['n_samples']}")
    print(f"Selected clusters: {best_candidate.n_clusters}")
    print(f"Selection mode: {'fixed' if n_clusters is not None else 'auto-search'}")
    print(f"Silhouette score: {best_candidate.silhouette_score:.4f}")
    print(f"Stability (ARI): {best_candidate.stability_ari:.4f}")
    print(f"Average confidence: {best_candidate.average_confidence:.4f}")
    print(f"Cluster distribution: {best_candidate.cluster_sizes}")
    print(f"\nArtifacts saved to: {artifacts_dir}/")
    print("  - kmeans_model.pkl")
    print("  - scaler.pkl")
    print("  - feature_engineer.pkl")
    print("  - segment_catalog.json")
    print("  - candidate_leaderboard.json")
    print("  - data_quality_report.json")
    print("  - model_metadata.json")
    print("=" * 72 + "\n")

    return metadata


def parse_args() -> argparse.Namespace:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train a production-style KMeans customer segmentation model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a CSV with age, annual_income, and spending_score columns.",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where training artifacts will be written.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Force a fixed number of clusters instead of auto-selecting the best model.",
    )
    parser.add_argument(
        "--min_clusters",
        type=int,
        default=DEFAULT_MIN_CLUSTERS,
        help="Minimum clusters to consider during auto-search.",
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=DEFAULT_MAX_CLUSTERS,
        help="Maximum clusters to consider during auto-search.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducible clustering runs.",
    )
    parser.add_argument(
        "--stability_runs",
        type=int,
        default=6,
        help="Number of repeated fits used to estimate clustering stability.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    if args.n_clusters is None and args.min_clusters > args.max_clusters:
        raise ValueError("--min_clusters must be less than or equal to --max_clusters")

    run_training(
        input_path=args.input,
        artifacts_dir=args.artifacts_dir,
        n_clusters=args.n_clusters,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        random_state=args.random_state,
        stability_runs=args.stability_runs,
    )


if __name__ == "__main__":
    main()
