# Project Card

I built an end-to-end customer segmentation system that combines a reproducible training pipeline with a production-style inference API. The stable path is intentionally simple and runnable: `train.py` reads a CSV, validates it, scales features, trains a K-Means model, and saves artifacts that `api/main.py` loads into a FastAPI service. Alongside that, I built a more ambitious v2 pipeline with schema validation, 24 engineered customer-behavior features, deeper evaluation, and experiment-style outputs.

Key stack: Python, scikit-learn, FastAPI, Pydantic, Docker, GitHub Actions, pytest.

Strongest verified outcomes:
- trained and saved a working model on 191 customer rows, with metrics persisted in `artifacts/model_metadata.json`
- exposed `/predict`, `/health`, and `/model/info`, including a confidence-aware prediction response
- verified the serving layer with 16 passing API tests and the training flow with a passing integration test in local validation

Suggested links:
- Read the case study
- See the demo path
- View code

