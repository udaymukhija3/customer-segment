# Customer Segmentation Studio

Portfolio-grade ML system for customer segmentation with:

- auto-selected K-Means clustering
- deterministic feature engineering shared by training and inference
- business-readable segment catalogs and recommended actions
- FastAPI inference endpoints for single and batch predictions
- a polished static frontend demo for showcasing the system live

## Why this stands out

This project is intentionally built like a small production ML product, not a notebook export. The training pipeline validates data quality, engineers reusable features, compares cluster candidates, scores model stability, and writes a metadata bundle that the API and frontend consume directly.

## Core capabilities

- `train.py` supports fixed-cluster training or auto-search across a cluster range
- `api/main.py` exposes `/predict`, `/predict/batch`, `/segments`, `/model/info`, and `/health`
- predictions return not just a segment id, but confidence, business context, key drivers, and action suggestions
- the frontend visualizes live model metrics and the trained segment atlas

## Quick start

### 1. Install dependencies

```bash
make install-dev
```

### 2. Train the model

```bash
make train
```

Or run auto-search manually:

```bash
python train.py \
  --input data/sample_customers.csv \
  --artifacts_dir artifacts \
  --min_clusters 3 \
  --max_clusters 8
```

### 3. Start the API

```bash
make serve
```

### 4. Start the demo UI

```bash
make frontend
```

Open:

- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Frontend: [http://localhost:3000](http://localhost:3000)

## API snapshot

### `POST /predict`

```json
{
  "age": 35,
  "annual_income": 75000,
  "spending_score": 62
}
```

Example response:

```json
{
  "segment_id": 2,
  "segment_name": "Premium Loyalists",
  "confidence_score": 0.8421,
  "segment_description": "High-value customers with sustained spending power.",
  "recommended_actions": [
    "Invest in concierge experiences and cross-sell bundles.",
    "Use them as a priority audience for high-margin campaigns."
  ],
  "key_drivers": [
    "Annual income is 24% above the portfolio median.",
    "Spending behavior maps to a high-intent customer segment.",
    "Spending score is 18% above the portfolio median."
  ],
  "input_flags": []
}
```

### `GET /segments`

Returns the trained business summaries for each cluster so the UI and downstream apps can use the same segment playbook.

### `POST /predict/batch`

Scores up to 250 customers in one request and returns aggregate segment counts.

## Training artifacts

After training, `artifacts/` contains:

- `kmeans_model.pkl`
- `scaler.pkl`
- `feature_engineer.pkl`
- `segment_catalog.json`
- `candidate_leaderboard.json`
- `data_quality_report.json`
- `model_metadata.json`

## Project structure

```text
customer_segment/
├── api/                  # FastAPI application
├── frontend/             # Static showcase UI
├── src/                  # Shared ML utilities
├── tests/                # Unit and integration tests
├── data/                 # Sample dataset
├── train.py              # Main training pipeline
├── train_v2.py           # Compatibility wrapper
└── README.md
```

## Quality checks

Run the project checks with:

```bash
python -m pytest -o addopts=''
flake8 api src tests train.py train_v2.py --max-line-length=120 --extend-ignore=E203,W503
mypy api/main.py src train.py train_v2.py --ignore-missing-imports
```

## Resume-friendly talking points

- designed a reusable feature engineering contract shared across offline training and online inference
- implemented automatic cluster selection using multi-metric ranking and stability scoring
- converted raw clusters into business-ready segment playbooks exposed through an API
- built a polished demo layer that visualizes both model telemetry and inference narratives
