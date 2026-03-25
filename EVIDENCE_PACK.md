# Evidence Pack

This file is a release-style checklist for capturing proof that the project is real, runnable, and worth discussing on a portfolio site.

The primary evidence path below uses the working baseline flow:

- `train.py`
- `artifacts/`
- `api/main.py`

The more advanced `train_v2.py` path is included as a secondary proof point because it currently fails fast on the bundled sample data due to a schema mismatch (`customer_id` is required, but not present in `data/sample_customers.csv`).

## 1) Demo Path

### Best 10-minute demo path

Use the baseline path. It is the most reliable end-to-end story in the repo today:

1. install dependencies
2. train the baseline model
3. save model metadata and prediction outputs into `docs/assets/`
4. start the API
5. hit `/health`, `/model/info`, and `/predict`
6. optionally open Swagger docs and the frontend for screenshots

### Terminal 1: setup + training

```bash
cd /Users/udaymukhija/customer_segment
mkdir -p docs/assets/{logs,json,metrics,tests,trees,screenshots,architecture}
python3 -m venv .venv
source .venv/bin/activate
pip install -r api/requirements.txt
python train.py --input data/sample_customers.csv 2>&1 | tee docs/assets/logs/train-baseline.log
python3 -m json.tool artifacts/model_metadata.json > docs/assets/json/model_metadata.pretty.json
python3 - <<'PY' > docs/assets/metrics/baseline-metrics.md
import json
from pathlib import Path

metadata = json.loads(Path("artifacts/model_metadata.json").read_text())
metrics = metadata["evaluation_metrics"]

print("| Metric | Value |")
print("|---|---:|")
print(f"| n_samples | {metadata['data_statistics']['n_samples']} |")
print(f"| n_clusters | {metadata['hyperparameters']['n_clusters']} |")
print(f"| inertia | {metrics['inertia']:.4f} |")
print(f"| silhouette_score | {metrics['silhouette_score']:.4f} |")
print(f"| davies_bouldin_score | {metrics['davies_bouldin_score']:.4f} |")
print(f"| calinski_harabasz_score | {metrics['calinski_harabasz_score']:.4f} |")
print(f"| cluster_sizes | `{metrics['cluster_sizes']}` |")
PY
find artifacts -maxdepth 1 -type f | sort > docs/assets/trees/artifacts-tree.txt
```

### Expected generated artifacts after training

| Step | Artifact | Location on disk |
|---|---|---|
| `python train.py ...` | trained KMeans model | `artifacts/kmeans_model.pkl` |
| `python train.py ...` | fitted scaler | `artifacts/scaler.pkl` |
| `python train.py ...` | metadata with metrics and config | `artifacts/model_metadata.json` |
| `tee` during training | shareable training log | `docs/assets/logs/train-baseline.log` |
| `python3 -m json.tool ...` | pretty metadata snapshot | `docs/assets/json/model_metadata.pretty.json` |
| metrics extraction snippet | recruiter-friendly metrics table | `docs/assets/metrics/baseline-metrics.md` |
| `find artifacts ...` | artifact folder listing | `docs/assets/trees/artifacts-tree.txt` |

### Terminal 2: start the API

```bash
cd /Users/udaymukhija/customer_segment
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Terminal 3: capture runtime evidence

```bash
cd /Users/udaymukhija/customer_segment
source .venv/bin/activate
curl -s http://localhost:8000/health | python3 -m json.tool > docs/assets/json/health.json
curl -s http://localhost:8000/model/info | python3 -m json.tool > docs/assets/json/model-info.json
printf '{\n  "age": 35,\n  "annual_income": 75000,\n  "spending_score": 62\n}\n' > docs/assets/json/predict-request.json
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @docs/assets/json/predict-request.json | python3 -m json.tool > docs/assets/json/predict-response.json
curl -s http://localhost:8000/openapi.json | python3 -m json.tool > docs/assets/json/openapi.json
```

### Optional browser steps for screenshots

Open these URLs while the API is running:

- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

Optional frontend demo:

```bash
cd /Users/udaymukhija/customer_segment/frontend
python3 -m http.server 3000
```

Then open:

- `http://localhost:3000`

### Expected generated artifacts after runtime capture

| Step | Artifact | Location on disk |
|---|---|---|
| `curl /health` | API health response | `docs/assets/json/health.json` |
| `curl /model/info` | runtime model metadata | `docs/assets/json/model-info.json` |
| `printf ...` | saved inference request | `docs/assets/json/predict-request.json` |
| `curl /predict` | saved inference response | `docs/assets/json/predict-response.json` |
| `curl /openapi.json` | saved OpenAPI schema | `docs/assets/json/openapi.json` |
| manual screenshot | Swagger UI for `/predict` | `docs/assets/screenshots/swagger-predict.png` |
| manual screenshot | frontend prediction result | `docs/assets/screenshots/frontend-demo.png` |

### Secondary proof step: schema guardrail in v2

This is worth capturing because it proves the advanced path enforces a data contract.

```bash
cd /Users/udaymukhija/customer_segment
source .venv/bin/activate
python train_v2.py --input data/sample_customers.csv 2>&1 | tee docs/assets/logs/train-v2-schema-failure.log
```

Expected result:

- the run fails immediately on missing `customer_id`
- the log is saved to `docs/assets/logs/train-v2-schema-failure.log`

This is useful evidence because it shows the advanced path fails early rather than training on malformed input.

## 2) Proof Artifacts

Capture the following 10-15 artifacts. They are all visual or easily shareable.

| Artifact | Why it matters to recruiters | How to generate | Store in repo |
|---|---|---|---|
| Successful baseline training log | Proves the model actually trains from raw CSV to artifacts | `python train.py --input data/sample_customers.csv 2>&1 | tee docs/assets/logs/train-baseline.log` | `docs/assets/logs/train-baseline.log` |
| Training log tail snippet | Makes the key metrics readable without opening the full log | `tail -n 40 docs/assets/logs/train-baseline.log > docs/assets/logs/train-baseline-tail.txt` | `docs/assets/logs/train-baseline-tail.txt` |
| Pretty-printed model metadata JSON | Shows saved hyperparameters, data stats, and metrics | `python3 -m json.tool artifacts/model_metadata.json > docs/assets/json/model_metadata.pretty.json` | `docs/assets/json/model_metadata.pretty.json` |
| Metrics table | Gives recruiters a skim-friendly summary instead of raw JSON | Run the Python snippet in the demo path to produce a markdown table | `docs/assets/metrics/baseline-metrics.md` |
| Artifact folder tree | Proves physical model files exist on disk | `find artifacts -maxdepth 1 -type f | sort > docs/assets/trees/artifacts-tree.txt` | `docs/assets/trees/artifacts-tree.txt` |
| Health response JSON | Shows the service starts and recognizes loaded artifacts | `curl -s http://localhost:8000/health | python3 -m json.tool > docs/assets/json/health.json` | `docs/assets/json/health.json` |
| Model info response JSON | Proves the API exposes runtime metadata, not just a prediction endpoint | `curl -s http://localhost:8000/model/info | python3 -m json.tool > docs/assets/json/model-info.json` | `docs/assets/json/model-info.json` |
| Sample prediction request/response | Demonstrates a real inference interaction | `printf '{\n  "age": 35,\n  "annual_income": 75000,\n  "spending_score": 62\n}\n' > docs/assets/json/predict-request.json` and `curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @docs/assets/json/predict-request.json | python3 -m json.tool > docs/assets/json/predict-response.json` | `docs/assets/json/predict-request.json`, `docs/assets/json/predict-response.json` |
| OpenAPI schema dump | Shows the API is self-documenting and machine-readable | `curl -s http://localhost:8000/openapi.json | python3 -m json.tool > docs/assets/json/openapi.json` | `docs/assets/json/openapi.json` |
| Swagger screenshot for `/predict` | Gives an immediate "this is a real service" visual | Start the API, open `http://localhost:8000/docs`, expand `/predict`, take a screenshot | `docs/assets/screenshots/swagger-predict.png` |
| Frontend screenshot with result | Good non-technical proof that the system is usable end to end | Start `python3 -m http.server 3000` in `frontend/`, open `http://localhost:3000`, submit a sample prediction, take a screenshot | `docs/assets/screenshots/frontend-demo.png` |
| Passing API test run | Strong proof of runnable behavior and endpoint coverage | `pip install -r requirements-dev.txt && pytest -q -o addopts='' tests/test_api.py | tee docs/assets/tests/api-tests.txt` | `docs/assets/tests/api-tests.txt` |
| Passing end-to-end training test | Proves training artifacts can be built in a test harness | `source .venv/bin/activate && pip install -r requirements-dev.txt && pytest -q -o addopts='' tests/test_train.py::TestTrainingPipeline::test_full_training_pipeline | tee docs/assets/tests/training-pipeline-test.txt` | `docs/assets/tests/training-pipeline-test.txt` |
| v2 schema-failure log | Useful "engineering judgment" artifact: strict schema checks catch incompatible input early | `python train_v2.py --input data/sample_customers.csv 2>&1 | tee docs/assets/logs/train-v2-schema-failure.log` | `docs/assets/logs/train-v2-schema-failure.log` |
| Architecture diagram source + screenshot | Helps recruiters understand the system in one glance | `awk '/^```mermaid/{flag=1;next}/^```/{if(flag){exit}}flag' CASE_STUDY.md > docs/assets/architecture/system-overview.mmd` then render in a Mermaid-capable preview and save a screenshot | `docs/assets/architecture/system-overview.mmd`, `docs/assets/screenshots/system-overview.png` |

## 3) Recommended `/docs/assets/` folder content list

Recommended structure:

```text
docs/assets/
├── architecture/
│   └── system-overview.mmd
├── json/
│   ├── health.json
│   ├── model-info.json
│   ├── model_metadata.pretty.json
│   ├── openapi.json
│   ├── predict-request.json
│   └── predict-response.json
├── logs/
│   ├── train-baseline.log
│   ├── train-baseline-tail.txt
│   └── train-v2-schema-failure.log
├── metrics/
│   └── baseline-metrics.md
├── screenshots/
│   ├── frontend-demo.png
│   ├── swagger-predict.png
│   └── system-overview.png
├── tests/
│   ├── api-tests.txt
│   └── training-pipeline-test.txt
└── trees/
    └── artifacts-tree.txt
```

### Minimal evidence set if you only capture 5 items

If time is tight, capture these first:

1. `docs/assets/logs/train-baseline.log`
2. `docs/assets/metrics/baseline-metrics.md`
3. `docs/assets/json/predict-response.json`
4. `docs/assets/screenshots/swagger-predict.png`
5. `docs/assets/tests/api-tests.txt`

These five are enough to prove:

- the model trains
- artifacts exist
- the API serves predictions
- the service is documented
- core API behavior is tested

