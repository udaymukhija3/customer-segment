"""Integration tests for FastAPI application."""
import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, api_client):
        """Test root endpoint returns basic info."""
        response = api_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "2.0.0"

    def test_health_check(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "timestamp" in data


class TestModelInfo:
    """Tests for model information endpoint."""

    def test_model_info(self, api_client):
        """Test model info endpoint returns correct information."""
        response = api_client.get("/model/info")

        assert response.status_code == 200
        data = response.json()

        assert data["model_type"] == "K-Means Clustering"
        assert data["n_clusters"] == 5
        assert data["features"] == ["age", "annual_income", "spending_score"]
        assert data["model_loaded"] is True
        assert "metadata" in data


class TestPredictionEndpoint:
    """Tests for segment prediction endpoint."""

    def test_predict_valid_input(self, api_client):
        """Test prediction with valid input."""
        payload = {
            "age": 35,
            "annual_income": 75000,
            "spending_score": 62
        }

        response = api_client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert "segment_id" in data
        assert "segment_name" in data
        assert "confidence_score" in data

        assert 0 <= data["segment_id"] <= 4
        assert isinstance(data["segment_name"], str)
        assert 0 <= data["confidence_score"] <= 1

    def test_predict_multiple_scenarios(self, api_client):
        """Test predictions for different customer profiles."""
        test_cases = [
            {"age": 25, "annual_income": 45000, "spending_score": 40},
            {"age": 50, "annual_income": 120000, "spending_score": 80},
            {"age": 65, "annual_income": 30000, "spending_score": 20},
            {"age": 22, "annual_income": 150000, "spending_score": 95},
        ]

        for payload in test_cases:
            response = api_client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert 0 <= data["segment_id"] <= 4

    def test_predict_boundary_values(self, api_client):
        """Test predictions with boundary values."""
        # Minimum values
        payload = {"age": 0, "annual_income": 0, "spending_score": 0}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200

        # Maximum values
        payload = {"age": 150, "annual_income": 1000000, "spending_score": 100}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_invalid_age(self, api_client):
        """Test prediction with invalid age."""
        payload = {
            "age": -5,  # Negative age
            "annual_income": 75000,
            "spending_score": 62
        }

        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_spending_score(self, api_client):
        """Test prediction with invalid spending score."""
        payload = {
            "age": 35,
            "annual_income": 75000,
            "spending_score": 150  # Over 100
        }

        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_field(self, api_client):
        """Test prediction with missing required field."""
        payload = {
            "age": 35,
            "annual_income": 75000
            # Missing spending_score
        }

        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_type(self, api_client):
        """Test prediction with invalid data type."""
        payload = {
            "age": "thirty-five",  # String instead of int
            "annual_income": 75000,
            "spending_score": 62
        }

        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_extra_fields(self, api_client):
        """Test prediction with extra fields (should be ignored)."""
        payload = {
            "age": 35,
            "annual_income": 75000,
            "spending_score": 62,
            "extra_field": "ignored"
        }

        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200


class TestDeprecatedEndpoint:
    """Tests for deprecated /get_segment endpoint."""

    def test_deprecated_endpoint_still_works(self, api_client):
        """Test that deprecated endpoint still functions."""
        payload = {
            "age": 35,
            "annual_income": 75000,
            "spending_score": 62
        }

        response = api_client.post("/get_segment", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "segment_id" in data
        assert "segment_name" in data

    def test_deprecated_endpoint_returns_same_as_predict(self, api_client):
        """Test that deprecated endpoint returns same result as /predict."""
        payload = {
            "age": 35,
            "annual_income": 75000,
            "spending_score": 62
        }

        response_old = api_client.post("/get_segment", json=payload)
        response_new = api_client.post("/predict", json=payload)

        assert response_old.status_code == 200
        assert response_new.status_code == 200
        assert response_old.json()["segment_id"] == response_new.json()["segment_id"]


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_swagger_docs_available(self, api_client):
        """Test that Swagger UI is accessible."""
        response = api_client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, api_client):
        """Test that ReDoc is accessible."""
        response = api_client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self, api_client):
        """Test that OpenAPI schema is available."""
        response = api_client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]
