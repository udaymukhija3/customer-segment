"""Integration tests for the FastAPI application."""

from __future__ import annotations


class TestHealthEndpoints:
    """Tests for service health and overview routes."""

    def test_root_endpoint(self, api_client) -> None:
        response = api_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Customer Segmentation API"
        assert data["version"] == "3.0.0"
        assert data["model_loaded"] is True

    def test_health_check(self, api_client) -> None:
        response = api_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_version"] == "3.0.0"
        assert "timestamp" in data


class TestModelInfo:
    """Tests for model metadata endpoints."""

    def test_model_info(self, api_client) -> None:
        response = api_client.get("/model/info")

        assert response.status_code == 200
        data = response.json()

        assert data["model_type"] == "KMeans"
        assert data["n_clusters"] == 5
        assert data["model_loaded"] is True
        assert "evaluation_metrics" in data
        assert "metadata" in data
        assert len(data["features"]) >= 3

    def test_segments_endpoint(self, api_client) -> None:
        response = api_client.get("/segments")

        assert response.status_code == 200
        data = response.json()
        assert "segments" in data
        assert len(data["segments"]) == 5
        assert {"segment_id", "segment_name", "recommended_actions"} <= set(data["segments"][0])


class TestPredictionEndpoint:
    """Tests for prediction routes."""

    def test_predict_valid_input(self, api_client) -> None:
        payload = {
            "age": 35,
            "annual_income": 75000,
            "spending_score": 62,
        }

        response = api_client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert 0 <= data["segment_id"] <= 4
        assert isinstance(data["segment_name"], str)
        assert 0 <= data["confidence_score"] <= 1
        assert isinstance(data["recommended_actions"], list)
        assert isinstance(data["key_drivers"], list)

    def test_predict_multiple_scenarios(self, api_client) -> None:
        test_cases = [
            {"age": 25, "annual_income": 45000, "spending_score": 40},
            {"age": 50, "annual_income": 120000, "spending_score": 80},
            {"age": 65, "annual_income": 30000, "spending_score": 20},
            {"age": 22, "annual_income": 150000, "spending_score": 95},
        ]

        for payload in test_cases:
            response = api_client.post("/predict", json=payload)
            assert response.status_code == 200
            assert 0 <= response.json()["segment_id"] <= 4

    def test_predict_boundary_values(self, api_client) -> None:
        minimum = {"age": 0, "annual_income": 0, "spending_score": 0}
        maximum = {"age": 150, "annual_income": 1000000, "spending_score": 100}

        assert api_client.post("/predict", json=minimum).status_code == 200
        response = api_client.post("/predict", json=maximum)
        assert response.status_code == 200
        assert response.json()["input_flags"]

    def test_predict_invalid_age(self, api_client) -> None:
        payload = {"age": -5, "annual_income": 75000, "spending_score": 62}
        assert api_client.post("/predict", json=payload).status_code == 422

    def test_predict_invalid_spending_score(self, api_client) -> None:
        payload = {"age": 35, "annual_income": 75000, "spending_score": 150}
        assert api_client.post("/predict", json=payload).status_code == 422

    def test_predict_missing_field(self, api_client) -> None:
        payload = {"age": 35, "annual_income": 75000}
        assert api_client.post("/predict", json=payload).status_code == 422

    def test_predict_invalid_type(self, api_client) -> None:
        payload = {"age": "thirty-five", "annual_income": 75000, "spending_score": 62}
        assert api_client.post("/predict", json=payload).status_code == 422

    def test_predict_extra_fields(self, api_client) -> None:
        payload = {
            "age": 35,
            "annual_income": 75000,
            "spending_score": 62,
            "extra_field": "ignored",
        }
        assert api_client.post("/predict", json=payload).status_code == 200

    def test_batch_prediction(self, api_client) -> None:
        payload = {
            "customers": [
                {"age": 35, "annual_income": 75000, "spending_score": 62},
                {"age": 24, "annual_income": 45000, "spending_score": 38},
                {"age": 50, "annual_income": 140000, "spending_score": 84},
            ]
        }

        response = api_client.post("/predict/batch", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["request_count"] == 3
        assert len(data["predictions"]) == 3
        assert sum(data["segment_counts"].values()) == 3


class TestDeprecatedEndpoint:
    """Tests for backward compatibility."""

    def test_deprecated_endpoint_still_works(self, api_client) -> None:
        payload = {
            "age": 35,
            "annual_income": 75000,
            "spending_score": 62,
        }

        response = api_client.post("/get_segment", json=payload)
        assert response.status_code == 200
        assert "segment_id" in response.json()

    def test_deprecated_endpoint_returns_same_as_predict(self, api_client) -> None:
        payload = {
            "age": 35,
            "annual_income": 75000,
            "spending_score": 62,
        }

        response_old = api_client.post("/get_segment", json=payload)
        response_new = api_client.post("/predict", json=payload)

        assert response_old.status_code == 200
        assert response_new.status_code == 200
        assert response_old.json()["segment_id"] == response_new.json()["segment_id"]


class TestAPIDocumentation:
    """Tests for docs endpoints."""

    def test_swagger_docs_available(self, api_client) -> None:
        assert api_client.get("/docs").status_code == 200

    def test_redoc_available(self, api_client) -> None:
        assert api_client.get("/redoc").status_code == 200

    def test_openapi_schema(self, api_client) -> None:
        response = api_client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "/predict" in schema["paths"]
        assert "/predict/batch" in schema["paths"]
        assert "/segments" in schema["paths"]
