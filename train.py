import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = ["age", "annual_income", "spending_score"]


def validate_data(data: np.ndarray) -> Dict[str, Any]:
    """
    Validate training data and return statistics.

    Args:
        data: Numpy array of features

    Returns:
        Dictionary containing validation results and statistics

    Raises:
        ValueError: If data fails validation
    """
    if len(data) == 0:
        raise ValueError("Dataset is empty")

    if len(data) < 10:
        raise ValueError(f"Dataset too small ({len(data)} rows). Need at least 10 samples for clustering.")

    # Check for NaN or inf values
    if np.isnan(data).any():
        raise ValueError("Dataset contains NaN values")

    if np.isinf(data).any():
        raise ValueError("Dataset contains infinite values")

    # Calculate statistics
    stats = {
        "n_samples": int(len(data)),
        "n_features": int(data.shape[1]),
        "feature_stats": {
            "age": {
                "min": float(data[:, 0].min()),
                "max": float(data[:, 0].max()),
                "mean": float(data[:, 0].mean()),
                "std": float(data[:, 0].std())
            },
            "annual_income": {
                "min": float(data[:, 1].min()),
                "max": float(data[:, 1].max()),
                "mean": float(data[:, 1].mean()),
                "std": float(data[:, 1].std())
            },
            "spending_score": {
                "min": float(data[:, 2].min()),
                "max": float(data[:, 2].max()),
                "mean": float(data[:, 2].mean()),
                "std": float(data[:, 2].std())
            }
        }
    }

    # Validate ranges
    if data[:, 0].min() < 0 or data[:, 0].max() > 150:
        logger.warning(f"Age values outside expected range [0, 150]: [{data[:, 0].min()}, {data[:, 0].max()}]")

    if data[:, 1].min() < 0:
        logger.warning(f"Negative annual income values found: min={data[:, 1].min()}")

    if data[:, 2].min() < 0 or data[:, 2].max() > 100:
        logger.warning(f"Spending score outside expected range [0, 100]: [{data[:, 2].min()}, {data[:, 2].max()}]")

    logger.info(f"Data validation passed: {stats['n_samples']} samples, {stats['n_features']} features")

    return stats


def read_dataset(csv_path: Path) -> np.ndarray:
    """
    Read and parse CSV dataset.

    Args:
        csv_path: Path to CSV file

    Returns:
        Numpy array of features

    Raises:
        ValueError: If CSV format is invalid or missing required columns
    """
    logger.info(f"Reading dataset from {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Expected columns: {REQUIRED_COLUMNS}"
            )

        rows: List[List[float]] = []
        for idx, row in enumerate(reader, start=1):
            try:
                rows.append([
                    float(row["age"]),
                    float(row["annual_income"]),
                    float(row["spending_score"]),
                ])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid numeric value in row {idx}: {row}"
                ) from e

    logger.info(f"Successfully read {len(rows)} rows")
    return np.asarray(rows, dtype=np.float64)


def evaluate_model(
    kmeans: KMeans,
    X_scaled: np.ndarray,
    X_original: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate clustering model performance.

    Args:
        kmeans: Trained KMeans model
        X_scaled: Scaled features used for training
        X_original: Original unscaled features

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model performance...")

    labels = kmeans.labels_

    # Calculate clustering metrics
    metrics = {
        "inertia": float(kmeans.inertia_),
        "silhouette_score": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X_scaled, labels))
    }

    # Calculate cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(cluster): int(count) for cluster, count in zip(unique, counts)}
    metrics["cluster_sizes"] = cluster_sizes

    logger.info(f"Inertia: {metrics['inertia']:.2f}")
    logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better, range [-1, 1])")
    logger.info(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f} (lower is better)")
    logger.info(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f} (higher is better)")
    logger.info(f"Cluster sizes: {cluster_sizes}")

    return metrics


def save_metadata(
    artifacts_dir: Path,
    data_stats: Dict[str, Any],
    metrics: Dict[str, float],
    args: argparse.Namespace
) -> None:
    """
    Save training metadata to JSON file.

    Args:
        artifacts_dir: Directory to save metadata
        data_stats: Data validation statistics
        metrics: Model evaluation metrics
        args: Training arguments
    """
    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "model_type": "KMeans",
        "model_version": "1.0.0",
        "hyperparameters": {
            "n_clusters": 5,
            "n_init": 10,
            "random_state": 42
        },
        "features": REQUIRED_COLUMNS,
        "data_statistics": data_stats,
        "evaluation_metrics": metrics,
        "training_config": {
            "input_file": str(args.input),
            "artifacts_dir": str(args.artifacts_dir)
        }
    }

    metadata_path = artifacts_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")


def main() -> None:
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train KMeans (k=5) clustering model for customer segmentation."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV with columns: age, annual_income, spending_score"
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=str(Path("artifacts")),
        help="Directory to save model artifacts (default: ./artifacts)"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of clusters (default: 5)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Starting Customer Segmentation Model Training")
    logger.info("="*60)

    # Validate input file
    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # Read and validate data
    X = read_dataset(csv_path)
    data_stats = validate_data(X)

    # Feature scaling
    logger.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KMeans
    logger.info(f"Training KMeans with {args.n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        n_init=10,
        random_state=args.random_state,
        verbose=0
    )
    kmeans.fit(X_scaled)
    logger.info("Training complete")

    # Evaluate model
    metrics = evaluate_model(kmeans, X_scaled, X)

    # Save artifacts
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    kmeans_path = artifacts_dir / "kmeans_model.pkl"
    scaler_path = artifacts_dir / "scaler.pkl"

    logger.info(f"Saving model to {kmeans_path}")
    dump(kmeans, kmeans_path)

    logger.info(f"Saving scaler to {scaler_path}")
    dump(scaler, scaler_path)

    # Save metadata
    save_metadata(artifacts_dir, data_stats, metrics, args)

    logger.info("="*60)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Artifacts saved to: {artifacts_dir}")
    logger.info("="*60)

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Samples trained: {data_stats['n_samples']}")
    print(f"Number of clusters: {args.n_clusters}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Cluster distribution: {metrics['cluster_sizes']}")
    print(f"\nArtifacts saved to: {artifacts_dir}/")
    print(f"  - kmeans_model.pkl")
    print(f"  - scaler.pkl")
    print(f"  - model_metadata.json")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
