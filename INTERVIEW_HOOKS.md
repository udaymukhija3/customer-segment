# Interview Hooks

- I kept two parallel tracks in the repo, `train.py` and `train_v2.py`. The interesting question is why I preserved a stable baseline path instead of immediately replacing it.
- `train_v2.py` fails fast on the bundled sample CSV because `SchemaValidator` expects `customer_id`. That is a useful discussion about data contracts and why early failure is better than silently training on bad input.
- The advanced feature pipeline in `src/features.py` expands the representation from 3 raw inputs to 24 engineered features. I can walk through which of those features are likely to help segmentation quality and which ones create leakage risk.
- `FeatureEngineer.validate_features()` checks zero-variance and high-correlation columns before clustering. I can explain why those checks matter more in unsupervised learning than many people expect.
- `api/main.py` returns a `confidence_score` based on distance to centroid. That is a good prompt to discuss how to communicate uncertainty in clustering systems.
- I preserved `/get_segment` as a deprecated endpoint and tested parity with `/predict`. That is a concrete example of handling API evolution without breaking clients.
- `tests/conftest.py` monkeypatches artifact paths so the API tests run hermetically. I can explain why that fixture design makes the service easier to test and refactor.
- The repo includes a multi-stage Dockerfile, non-root runtime user, and health checks. That is a useful way to talk about what "production-ready" really means for small ML services.
- `.github/workflows/ci.yml` builds the Docker image, runs a Python version matrix, and performs a Trivy security scan. I can use that to talk about release confidence, not just model quality.
- `src/evaluation.py` includes per-cluster silhouette analysis, cluster balance, separation metrics, and problematic-sample detection. That opens a discussion about how to do error analysis without labels.
- `src/evaluation.py` also adds business-facing segment ranking and lift calculations. I can explain how I translated cluster IDs into something a marketing or retention team could actually use.
- The advanced path persists `feature_engineer.pkl`, but the API still serves the baseline 3-feature flow. That is a concrete offline/online parity gap and a good prompt to discuss what I would change next.

