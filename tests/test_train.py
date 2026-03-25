"""Tests for the upgraded training pipeline."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from joblib import load

import train


class TestDataValidation:
    """Tests for numpy validation helpers."""

    def test_validate_data_success(self, sample_data: np.ndarray) -> None:
        stats = train.validate_data(sample_data)

        assert stats["n_samples"] == 50
        assert stats["n_features"] == 3
        assert "age" in stats["feature_stats"]
        assert "annual_income" in stats["feature_stats"]
        assert "spending_score" in stats["feature_stats"]

    def test_validate_empty_data(self) -> None:
        with pytest.raises(ValueError, match="Dataset is empty"):
            train.validate_data(np.array([]))

    def test_validate_small_dataset(self) -> None:
        small_data = np.array([[25, 50000, 45], [30, 60000, 50]])
        with pytest.raises(ValueError, match="Dataset too small"):
            train.validate_data(small_data)

    def test_validate_nan_values(self) -> None:
        data_with_nan = np.array([[25, 50000, 45], [30, np.nan, 50]])
        with pytest.raises(ValueError, match="NaN values"):
            train.validate_data(data_with_nan)

    def test_validate_inf_values(self) -> None:
        data_with_inf = np.array([[25, 50000, 45], [30, np.inf, 50]])
        with pytest.raises(ValueError, match="infinite values"):
            train.validate_data(data_with_inf)


class TestReadDataset:
    """Tests for CSV loading."""

    def test_read_valid_csv(self, sample_csv) -> None:
        data = train.read_dataset(sample_csv)

        assert isinstance(data, np.ndarray)
        assert data.shape == (50, 3)
        assert data.dtype == np.float64

    def test_read_missing_columns(self, missing_columns_csv) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            train.read_dataset(missing_columns_csv)

    def test_read_invalid_values(self, invalid_csv) -> None:
        with pytest.raises(ValueError, match="Invalid numeric value"):
            train.read_dataset(invalid_csv)

    def test_read_nonexistent_file(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            train.read_dataset(tmp_path / "nonexistent.csv")


class TestModelEvaluation:
    """Tests for training evaluation output."""

    def test_evaluate_model(self, sample_data: np.ndarray) -> None:
        from sklearn.cluster import KMeans

        feature_engineer = train.FeatureEngineer()
        _, scaled = feature_engineer.fit_transform(
            train.pd.DataFrame(sample_data, columns=train.REQUIRED_COLUMNS)
        )
        kmeans = KMeans(n_clusters=5, n_init=25, random_state=42)
        kmeans.fit(scaled)

        metrics = train.evaluate_model(kmeans, scaled, sample_data)

        assert "inertia" in metrics
        assert "silhouette_score" in metrics
        assert "davies_bouldin_score" in metrics
        assert "calinski_harabasz_score" in metrics
        assert "cluster_sizes" in metrics
        assert "average_confidence" in metrics

        assert metrics["inertia"] > 0
        assert -1 <= metrics["silhouette_score"] <= 1
        assert metrics["davies_bouldin_score"] >= 0
        assert metrics["calinski_harabasz_score"] > 0
        assert 0 <= metrics["average_confidence"] <= 1
        assert sum(metrics["cluster_sizes"].values()) == len(sample_data)


class TestTrainingPipeline:
    """Integration tests for end-to-end training."""

    def test_full_training_pipeline(self, sample_csv, tmp_path) -> None:
        artifacts_dir = tmp_path / "artifacts"

        sys.argv = [
            "train.py",
            "--input",
            str(sample_csv),
            "--artifacts_dir",
            str(artifacts_dir),
            "--n_clusters",
            "5",
        ]
        train.main()

        assert (artifacts_dir / "kmeans_model.pkl").exists()
        assert (artifacts_dir / "scaler.pkl").exists()
        assert (artifacts_dir / "feature_engineer.pkl").exists()
        assert (artifacts_dir / "segment_catalog.json").exists()
        assert (artifacts_dir / "model_metadata.json").exists()

        kmeans = load(artifacts_dir / "kmeans_model.pkl")
        assert kmeans.n_clusters == 5
        assert load(artifacts_dir / "scaler.pkl") is not None
        assert load(artifacts_dir / "feature_engineer.pkl") is not None

        metadata = json.loads((artifacts_dir / "model_metadata.json").read_text(encoding="utf-8"))
        assert metadata["model_type"] == "KMeans"
        assert metadata["hyperparameters"]["n_clusters"] == 5
        assert metadata["data_statistics"]["n_samples"] == 50
        assert metadata["hyperparameters"]["selection_mode"] == "fixed"
        assert len(metadata["segment_catalog"]) == 5

    def test_training_with_custom_clusters(self, sample_csv, tmp_path) -> None:
        artifacts_dir = tmp_path / "artifacts"
        metadata = train.run_training(
            input_path=sample_csv,
            artifacts_dir=artifacts_dir,
            n_clusters=3,
            random_state=42,
            stability_runs=3,
        )

        kmeans = load(artifacts_dir / "kmeans_model.pkl")
        assert kmeans.n_clusters == 3
        assert metadata["hyperparameters"]["n_clusters"] == 3

    def test_training_auto_selects_clusters(self, sample_csv, tmp_path) -> None:
        artifacts_dir = tmp_path / "artifacts"
        metadata = train.run_training(
            input_path=sample_csv,
            artifacts_dir=artifacts_dir,
            random_state=42,
            stability_runs=3,
            min_clusters=3,
            max_clusters=5,
        )

        assert metadata["hyperparameters"]["selection_mode"] == "auto-search"
        assert 3 <= metadata["hyperparameters"]["n_clusters"] <= 5
        assert len(metadata["leaderboard"]) == 3
