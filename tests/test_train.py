"""Unit tests for training script."""
import json
from pathlib import Path

import numpy as np
import pytest
from joblib import load

import train


class TestDataValidation:
    """Tests for data validation."""

    def test_validate_data_success(self, sample_data):
        """Test successful data validation."""
        stats = train.validate_data(sample_data)

        assert stats["n_samples"] == 50
        assert stats["n_features"] == 3
        assert "age" in stats["feature_stats"]
        assert "annual_income" in stats["feature_stats"]
        assert "spending_score" in stats["feature_stats"]

    def test_validate_empty_data(self):
        """Test validation fails on empty data."""
        with pytest.raises(ValueError, match="Dataset is empty"):
            train.validate_data(np.array([]))

    def test_validate_small_dataset(self):
        """Test validation fails on too small dataset."""
        small_data = np.array([[25, 50000, 45], [30, 60000, 50]])
        with pytest.raises(ValueError, match="Dataset too small"):
            train.validate_data(small_data)

    def test_validate_nan_values(self):
        """Test validation fails on NaN values."""
        data_with_nan = np.array([[25, 50000, 45], [30, np.nan, 50]])
        with pytest.raises(ValueError, match="NaN values"):
            train.validate_data(data_with_nan)

    def test_validate_inf_values(self):
        """Test validation fails on infinite values."""
        data_with_inf = np.array([[25, 50000, 45], [30, np.inf, 50]])
        with pytest.raises(ValueError, match="infinite values"):
            train.validate_data(data_with_inf)


class TestReadDataset:
    """Tests for CSV reading."""

    def test_read_valid_csv(self, sample_csv):
        """Test reading valid CSV file."""
        data = train.read_dataset(sample_csv)

        assert isinstance(data, np.ndarray)
        assert data.shape == (50, 3)
        assert data.dtype == np.float64

    def test_read_missing_columns(self, missing_columns_csv):
        """Test reading CSV with missing required columns."""
        with pytest.raises(ValueError, match="missing required columns"):
            train.read_dataset(missing_columns_csv)

    def test_read_invalid_values(self, invalid_csv):
        """Test reading CSV with invalid numeric values."""
        with pytest.raises(ValueError, match="Invalid numeric value"):
            train.read_dataset(invalid_csv)

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading non-existent file."""
        nonexistent = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            train.read_dataset(nonexistent)


class TestModelEvaluation:
    """Tests for model evaluation."""

    def test_evaluate_model(self, sample_data):
        """Test model evaluation metrics."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sample_data)

        kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
        kmeans.fit(X_scaled)

        metrics = train.evaluate_model(kmeans, X_scaled, sample_data)

        assert "inertia" in metrics
        assert "silhouette_score" in metrics
        assert "davies_bouldin_score" in metrics
        assert "calinski_harabasz_score" in metrics
        assert "cluster_sizes" in metrics

        # Check metric ranges
        assert metrics["inertia"] > 0
        assert -1 <= metrics["silhouette_score"] <= 1
        assert metrics["davies_bouldin_score"] >= 0
        assert metrics["calinski_harabasz_score"] > 0

        # Check cluster sizes sum to total samples
        assert sum(metrics["cluster_sizes"].values()) == len(sample_data)


class TestTrainingPipeline:
    """Integration tests for complete training pipeline."""

    def test_full_training_pipeline(self, sample_csv, tmp_path):
        """Test complete training pipeline end-to-end."""
        artifacts_dir = tmp_path / "artifacts"

        # Run training
        import sys
        sys.argv = [
            "train.py",
            "--input", str(sample_csv),
            "--artifacts_dir", str(artifacts_dir)
        ]

        train.main()

        # Check artifacts were created
        assert (artifacts_dir / "kmeans_model.pkl").exists()
        assert (artifacts_dir / "scaler.pkl").exists()
        assert (artifacts_dir / "model_metadata.json").exists()

        # Load and verify model
        kmeans = load(artifacts_dir / "kmeans_model.pkl")
        assert kmeans.n_clusters == 5

        scaler = load(artifacts_dir / "scaler.pkl")
        assert scaler is not None

        # Verify metadata
        with open(artifacts_dir / "model_metadata.json") as f:
            metadata = json.load(f)

        assert metadata["model_type"] == "KMeans"
        assert metadata["hyperparameters"]["n_clusters"] == 5
        assert metadata["data_statistics"]["n_samples"] == 50

    def test_training_with_custom_clusters(self, sample_csv, tmp_path):
        """Test training with custom number of clusters."""
        artifacts_dir = tmp_path / "artifacts"

        import sys
        sys.argv = [
            "train.py",
            "--input", str(sample_csv),
            "--artifacts_dir", str(artifacts_dir),
            "--n_clusters", "3"
        ]

        train.main()

        kmeans = load(artifacts_dir / "kmeans_model.pkl")
        assert kmeans.n_clusters == 3
