# Website Copy

## Homepage Project Card

I built an end-to-end customer segmentation system that goes beyond a notebook demo: a reproducible training pipeline, serialized model artifacts, a FastAPI inference service, Docker packaging, CI, and a browser-based demo. The repo intentionally has two layers. The stable path is `train.py -> artifacts/ -> api/main.py`, which is the fastest way to prove the system is real and runnable. Beside it, I built a more ambitious v2 pipeline with schema validation, richer feature engineering, and deeper evaluation in `src/`.

Key stack: Python, scikit-learn, FastAPI, Pydantic, Docker, GitHub Actions, pytest.

Strongest verified outcomes:
- trained and saved a working K-Means model on 191 customer rows with metrics persisted in `artifacts/model_metadata.json`
- exposed `/predict`, `/health`, and `/model/info` through a FastAPI service with confidence-aware responses
- verified the serving layer with 16 passing API tests and the training flow with a passing integration test in local validation

Link text suggestions:
- Read the case study
- See the demo path
- View code

## Highlights

- Built a real training-to-serving path: `train.py` produces `kmeans_model.pkl`, `scaler.pkl`, and `model_metadata.json`, and `api/main.py` serves them through FastAPI.
- Persisted reproducibility metadata with hyperparameters, feature names, dataset statistics, and clustering metrics in `artifacts/model_metadata.json`.
- Verified baseline model quality on the checked-in artifact: 191 samples, 5 clusters, silhouette score `0.3759`, Davies-Bouldin `1.0121`, and Calinski-Harabasz `149.7547`.
- Added confidence-aware inference by converting distance-to-centroid into a bounded `confidence_score` returned by `/predict`.
- Preserved backward compatibility by keeping deprecated `/get_segment` alive and testing that it matches `/predict`.
- Built a second-generation feature pipeline in `src/features.py` that produces 24 engineered features across RFM, behavioral, recent-activity, and tenure families.
- Added feature-quality checks for zero-variance, high-correlation, missing, infinite, and outlier-heavy columns before clustering.
- Added a stricter data-contract layer in `src/data_quality.py`; `train_v2.py` fails fast on the bundled sample data because `customer_id` is missing, which is a useful guardrail rather than a silent bad run.
- Packaged the service for deployment with a multi-stage Docker build, non-root runtime, health checks, and GitHub Actions CI plus Trivy security scanning.

## Deep Dive

<details>
<summary><strong>Data Pipeline</strong></summary>

The baseline path is intentionally simple and reliable. `train.py` reads a CSV with `age`, `annual_income`, and `spending_score`, validates emptiness, sample count, NaN/Inf values, and basic value ranges, then trains and saves artifacts. That is the demo path I would show first because it is the one that actually runs cleanly from raw data to API.

The more interesting engineering is in `train_v2.py` and `src/data_quality.py`. The advanced path adds `SchemaValidator`, `DataQualityChecker`, and structured quality reports, including completeness, validity, outlier, duplicate, and distribution checks. It currently rejects the bundled sample CSV because `customer_id` is required and missing. That is a useful portfolio detail: the repo contains a real data contract, and it already caught an integration mismatch.

Proof artifacts:
- `artifacts/model_metadata.json`
- `docs/assets/logs/train-baseline.log`
- `docs/assets/logs/train-v2-schema-failure.log`

</details>

<details>
<summary><strong>Modeling</strong></summary>

The implemented production path uses `KMeans`, which is a pragmatic choice here: lightweight artifacts, fast inference, and cluster centroids that are easier to explain than a more complex model. In the baseline script, cluster count is configurable and the pipeline persists the scaler used at training time.

The advanced path adds more interesting modeling discipline. `train_v2.py` includes challenger baselines, sweeps `k` from 3 to 8, and uses silhouette score to select the best cluster count. The feature side is also materially richer: `src/features.py` builds 24 engineered features across RFM, behavioral timing, purchase regularity, recent spend, acceleration, and tenure. That is a meaningful jump from the baseline three-column representation.

Proof artifacts:
- `docs/assets/metrics/baseline-metrics.md`
- `docs/assets/json/model_metadata.pretty.json`
- `src/features.py`
- `train_v2.py`

</details>

<details>
<summary><strong>Serving &amp; MLOps</strong></summary>

`api/main.py` wraps the model and scaler in a `ModelContainer` that loads artifacts once at startup and exposes `/predict`, `/health`, and `/model/info`. The prediction endpoint returns both a `segment_id` and a distance-derived confidence score, which makes the API more useful than a typical clustering demo that returns only a label.

Operationally, the repo is stronger than most portfolio projects. It includes a multi-stage Dockerfile, non-root runtime user, image health checks, docker-compose service health checks, a CI matrix across Python 3.10 to 3.12, Docker build validation, and Trivy security scanning. The API surface is also tested directly with `tests/test_api.py`, including docs endpoints and backward compatibility for `/get_segment`.

Proof artifacts:
- `docs/assets/json/health.json`
- `docs/assets/json/model-info.json`
- `docs/assets/screenshots/swagger-predict.png`
- `.github/workflows/ci.yml`
- `api/Dockerfile`

</details>

<details>
<summary><strong>Results &amp; Evaluation</strong></summary>

The baseline path already saves recruiter-friendly proof. The checked-in `artifacts/model_metadata.json` records 191 training rows, 5 clusters, inertia, silhouette score, Davies-Bouldin score, Calinski-Harabasz score, and cluster sizes. That makes the system auditable and easy to discuss without rerunning anything live.

The advanced evaluator in `src/evaluation.py` goes beyond the basics. It adds per-cluster silhouette analysis, cluster balance, centroid separation metrics, problematic-sample detection, and business-facing segment characterization, ranking, and lift calculations. The main caveat is that the full v2 path is not yet runnable on the bundled sample data, so I would present those as implemented evaluation capabilities in code, not as already-verified end-to-end results.

Proof artifacts:
- `artifacts/model_metadata.json`
- `docs/assets/metrics/baseline-metrics.md`
- `src/evaluation.py`
- `tests/test_api.py`

</details>

## Recruiter TL;DR

I built an end-to-end ML project that trains a customer segmentation model, saves versioned artifacts, serves predictions through FastAPI, and packages the service with Docker and CI.
The working baseline path is fully runnable from CSV to API.
The repo also includes a stricter v2 pipeline with schema validation, 24 engineered features, and richer evaluation logic.
Verified outputs include saved clustering metrics, passing API tests, and a confidence-aware inference endpoint.
It is a solid example of ML engineering judgment, not just model fitting.

