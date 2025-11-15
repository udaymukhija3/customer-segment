"""Pytest configuration and fixtures."""
import json
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def sample_data() -> np.ndarray:
    """Generate sample customer data for testing."""
    np.random.seed(42)
    # Create 50 samples with 3 features
    data = np.array([
        [25, 50000, 45],
        [30, 60000, 50],
        [35, 70000, 55],
        [40, 80000, 60],
        [45, 90000, 65],
        [22, 45000, 40],
        [28, 55000, 48],
        [32, 65000, 52],
        [38, 75000, 58],
        [42, 85000, 62],
        [50, 100000, 70],
        [55, 110000, 75],
        [60, 120000, 80],
        [65, 130000, 85],
        [70, 140000, 90],
        [23, 48000, 42],
        [27, 52000, 46],
        [31, 62000, 51],
        [36, 72000, 56],
        [41, 82000, 61],
        [19, 40000, 35],
        [21, 42000, 38],
        [24, 46000, 41],
        [26, 50000, 44],
        [29, 58000, 49],
        [33, 68000, 53],
        [37, 74000, 57],
        [43, 86000, 63],
        [47, 94000, 67],
        [52, 104000, 72],
        [20, 41000, 36],
        [34, 69000, 54],
        [39, 79000, 59],
        [44, 89000, 64],
        [49, 99000, 69],
        [54, 109000, 74],
        [59, 119000, 79],
        [64, 129000, 84],
        [69, 139000, 89],
        [18, 38000, 33],
        [48, 96000, 68],
        [53, 106000, 73],
        [58, 116000, 78],
        [63, 126000, 83],
        [68, 136000, 88],
        [25, 51000, 45],
        [30, 61000, 50],
        [35, 71000, 55],
        [40, 81000, 60],
        [45, 91000, 65],
    ], dtype=np.float64)
    return data


@pytest.fixture
def sample_csv(tmp_path: Path, sample_data: np.ndarray) -> Path:
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "test_customers.csv"
    with open(csv_path, 'w') as f:
        f.write("age,annual_income,spending_score\n")
        for row in sample_data:
            f.write(f"{int(row[0])},{int(row[1])},{int(row[2])}\n")
    return csv_path


@pytest.fixture
def invalid_csv(tmp_path: Path) -> Path:
    """Create a CSV with invalid data."""
    csv_path = tmp_path / "invalid.csv"
    with open(csv_path, 'w') as f:
        f.write("age,annual_income,spending_score\n")
        f.write("25,50000,abc\n")  # Invalid value
    return csv_path


@pytest.fixture
def missing_columns_csv(tmp_path: Path) -> Path:
    """Create a CSV with missing columns."""
    csv_path = tmp_path / "missing_cols.csv"
    with open(csv_path, 'w') as f:
        f.write("age,annual_income\n")  # Missing spending_score
        f.write("25,50000\n")
    return csv_path


@pytest.fixture
def trained_artifacts(tmp_path: Path, sample_data: np.ndarray) -> Path:
    """Create trained model artifacts for testing."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    # Train a simple model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_data)

    kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
    kmeans.fit(X_scaled)

    # Save artifacts
    dump(kmeans, artifacts_dir / "kmeans_model.pkl")
    dump(scaler, artifacts_dir / "scaler.pkl")

    # Save metadata
    metadata = {
        "training_date": "2024-01-01T00:00:00",
        "model_type": "KMeans",
        "model_version": "1.0.0",
        "hyperparameters": {
            "n_clusters": 5,
            "n_init": 10,
            "random_state": 42
        },
        "features": ["age", "annual_income", "spending_score"],
        "data_statistics": {
            "n_samples": len(sample_data),
            "n_features": 3
        }
    }

    with open(artifacts_dir / "model_metadata.json", 'w') as f:
        json.dump(metadata, f)

    return artifacts_dir


@pytest.fixture
def api_client(trained_artifacts: Path, monkeypatch) -> Generator[TestClient, None, None]:
    """Create a FastAPI test client with loaded models."""
    # Import here to avoid circular imports
    from api import main

    # Monkeypatch the artifacts directory
    monkeypatch.setattr(main, "ARTIFACTS_DIR", trained_artifacts)
    monkeypatch.setattr(main, "KMEANS_PATH", trained_artifacts / "kmeans_model.pkl")
    monkeypatch.setattr(main, "SCALER_PATH", trained_artifacts / "scaler.pkl")
    monkeypatch.setattr(main, "METADATA_PATH", trained_artifacts / "model_metadata.json")

    # Reload the model container
    main.model_container = main.ModelContainer()
    main.model_container.load_models()

    # Create test client
    from fastapi.testclient import TestClient
    client = TestClient(main.app)

    yield client

    # Cleanup
    main.model_container = main.ModelContainer()
