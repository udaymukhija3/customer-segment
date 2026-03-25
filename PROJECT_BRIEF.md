# Customer Segmentation ML System

## Snapshot

This project is an end-to-end customer segmentation system with a working baseline training-and-serving path and a more ambitious second-generation ML pipeline beside it.

- Baseline path: `train.py` -> `artifacts/` -> `api/main.py`
- Advanced path: `train_v2.py` + `src/` modules for data quality, feature engineering, evaluation, and experiment artifacts
- Serving: FastAPI with `/predict`, `/health`, `/model/info`, and a deprecated compatibility endpoint
- Ops: Docker, docker-compose, GitHub Actions CI, GHCR deployment workflow, and Trivy security scanning

## Why It Stands Out

1. It separates a stable shipping path from a more rigorous experimental path
   - `train.py` is the lightweight, working baseline
   - `train_v2.py` introduces schema checks, feature engineering, richer evaluation, and experiment-style outputs

2. The advanced path includes real ML engineering work, not just clustering boilerplate
   - `src/features.py` builds 24 transaction-style features across RFM, behavioral, and recent-activity families
   - `src/evaluation.py` adds cluster balance, per-cluster silhouette analysis, problematic-sample detection, and business-facing segment ranking / lift
   - `src/data_quality.py` adds schema validation, outlier checks, duplicate checks, distribution analysis, and drift-detection primitives

3. The serving layer is practical
   - `api/main.py` loads model artifacts once at startup
   - returns confidence with each prediction
   - exposes model metadata and health for debugging and operations
   - preserves `/get_segment` for backward compatibility

4. The repo shows engineering judgment, including where things still need work
   - `train_v2.py` currently fails fast on the bundled sample CSV because it requires `customer_id`
   - that is a real data-contract catch, not a hidden failure

## Verified Results

| Item | Verified value | Evidence |
|---|---|---|
| Baseline dataset size | 191 rows | `artifacts/model_metadata.json` |
| Baseline model | KMeans, 5 clusters | `artifacts/model_metadata.json` |
| Silhouette score | 0.3759 | `artifacts/model_metadata.json` |
| Davies-Bouldin score | 1.0121 | `artifacts/model_metadata.json` |
| Calinski-Harabasz score | 149.7547 | `artifacts/model_metadata.json` |
| Cluster sizes | 53, 29, 23, 40, 46 | `artifacts/model_metadata.json` |
| Advanced feature count | 24 | direct execution of `FeatureEngineer.create_feature_matrix()` |
| Local repo validation | 26 tests passed, 2 failed in baseline suite; v2 training failed fast on missing `customer_id` | local verification on 2026-03-10 |

## Key Evidence

- Training and baseline artifact generation: `train.py`
- Advanced pipeline orchestration: `train_v2.py`
- Feature engineering: `src/features.py`
- Evaluation and business diagnostics: `src/evaluation.py`
- Data contracts and drift primitives: `src/data_quality.py`
- API surface and artifact loading: `api/main.py`
- Reliability scaffolding: `tests/`, `.github/workflows/`, `api/Dockerfile`, `docker-compose.yml`

## What I'd Improve Next

- Align the advanced schema with the bundled sample data
- Move `feature_engineer.pkl` into the online inference path for true offline/online parity
- Add held-out or time-based evaluation to reduce feature leakage risk
- Wire drift detection and latency metrics into runtime monitoring
- Expand tests to cover the v2 modules directly

## Quick Demo

```bash
pip install -r api/requirements.txt
python train.py --input data/sample_customers.csv
uvicorn api.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "annual_income": 75000, "spending_score": 62}'
```

Expected baseline artifacts:

- `artifacts/kmeans_model.pkl`
- `artifacts/scaler.pkl`
- `artifacts/model_metadata.json`

