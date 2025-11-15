# Customer Segmentation API

[![CI/CD Pipeline](https://github.com/udaymukhija3/customer_segment/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/udaymukhija3/customer_segment/actions)
[![codecov](https://codecov.io/gh/udaymukhija3/customer_segment/branch/main/graph/badge.svg)](https://codecov.io/gh/udaymukhija3/customer_segment)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, end-to-end ML system for real-time customer segmentation using K-Means clustering. Built with FastAPI, scikit-learn, and Docker.

## Features

- **Production-Ready API**: FastAPI with auto-generated OpenAPI documentation
- **Comprehensive Validation**: Input validation with Pydantic, data quality checks
- **Model Evaluation**: Silhouette score, Davies-Bouldin index, Calinski-Harabasz score
- **Health Monitoring**: Health check and model info endpoints
- **Logging**: Structured logging throughout training and inference
- **Testing**: 100% test coverage with pytest
- **CI/CD**: GitHub Actions workflows for testing, linting, and deployment
- **Docker**: Multi-stage builds, health checks, non-root user
- **Dependency Injection**: Clean architecture with FastAPI dependencies
- **Model Metadata**: Version tracking and training metrics
- **Type Safety**: Full type hints with mypy validation

## Quick Start

### Option 1: Using Make (Recommended)

```bash
# Install dependencies
make install-dev

# Train the model
make train

# Run tests
make test

# Start the API
make serve
```

### Option 2: Using Docker Compose

```bash
# Train the model first
python train.py --input data/sample_customers.csv

# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Option 3: Manual Setup

```bash
# Install dependencies
pip install -r api/requirements.txt

# Train the model
python train.py --input data/sample_customers.csv

# Run the API
uvicorn api.main:app --reload
```

## Project Structure

```
customer_segment/
├── api/
│   ├── main.py              # FastAPI application with dependency injection
│   ├── Dockerfile           # Multi-stage Docker build
│   └── requirements.txt     # Production dependencies
├── tests/
│   ├── conftest.py          # Pytest fixtures and configuration
│   ├── test_api.py          # API integration tests
│   └── test_train.py        # Training pipeline unit tests
├── data/
│   └── sample_customers.csv # Sample training data (200 rows)
├── artifacts/               # Model artifacts (generated)
│   ├── kmeans_model.pkl     # Trained K-Means model
│   ├── scaler.pkl           # StandardScaler
│   └── model_metadata.json  # Training metadata and metrics
├── .github/
│   └── workflows/
│       ├── ci.yml           # CI/CD pipeline
│       └── deploy.yml       # Deployment workflow
├── train.py                 # Model training script
├── docker-compose.yml       # Multi-service orchestration
├── Makefile                 # Development commands
├── pytest.ini               # Pytest configuration
├── requirements-dev.txt     # Development dependencies
└── README.md
```

## API Endpoints

### 🔍 Prediction

#### `POST /predict` (Primary)

Predict customer segment with confidence score.

**Request:**
```json
{
  "age": 35,
  "annual_income": 75000,
  "spending_score": 62
}
```

**Response:**
```json
{
  "segment_id": 4,
  "segment_name": "High Income, High Spending",
  "confidence_score": 0.8234
}
```

**Validation:**
- `age`: 0-150
- `annual_income`: >= 0
- `spending_score`: 0-100

#### `POST /get_segment` (Deprecated)

Legacy endpoint for backward compatibility. Use `/predict` instead.

### 💚 Health & Monitoring

#### `GET /health`

Health check for load balancers and monitoring.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T12:00:00.000000"
}
```

#### `GET /model/info`

Get model information and metadata.

**Response:**
```json
{
  "model_type": "K-Means Clustering",
  "n_clusters": 5,
  "features": ["age", "annual_income", "spending_score"],
  "model_loaded": true,
  "artifacts_path": "/app/artifacts",
  "metadata": {
    "training_date": "2024-01-15T10:30:00",
    "model_version": "1.0.0",
    "evaluation_metrics": {
      "silhouette_score": 0.4521,
      "davies_bouldin_score": 1.2341,
      "inertia": 245.67
    }
  }
}
```

### 📚 Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Training Pipeline

### Basic Training

```bash
python train.py --input data/sample_customers.csv
```

### Advanced Options

```bash
python train.py \
  --input /path/to/data.csv \
  --artifacts_dir ./my_models \
  --n_clusters 3 \
  --random_state 42
```

**Arguments:**
- `--input`: Path to CSV file (required)
- `--artifacts_dir`: Output directory (default: `./artifacts`)
- `--n_clusters`: Number of clusters (default: 5)
- `--random_state`: Random seed (default: 42)

### CSV Format

Required columns: `age`, `annual_income`, `spending_score`

```csv
age,annual_income,spending_score
25,50000,45
30,60000,50
35,70000,55
```

### Training Output

```
============================================================
TRAINING SUMMARY
============================================================
Samples trained: 200
Number of clusters: 5
Silhouette Score: 0.4521
Cluster distribution: {0: 42, 1: 38, 2: 45, 3: 35, 4: 40}

Artifacts saved to: artifacts/
  - kmeans_model.pkl
  - scaler.pkl
  - model_metadata.json
============================================================
```

## Evaluation Metrics

The training pipeline automatically computes:

- **Inertia**: Sum of squared distances to cluster centers (lower is better)
- **Silhouette Score**: -1 to 1, higher means better-defined clusters
- **Davies-Bouldin Index**: Lower values indicate better separation
- **Calinski-Harabasz Score**: Higher values indicate denser, well-separated clusters

## Customer Segments

| ID | Segment Name | Description |
|----|--------------|-------------|
| 0 | Low Income, Low Spending | Budget-conscious customers |
| 1 | Average Customer | Middle-market segment |
| 2 | High Income, Low Spending | Wealthy but conservative spenders |
| 3 | High Spending, Low Income | High propensity to spend despite income |
| 4 | High Income, High Spending | Premium customers |

## Testing

### Run All Tests

```bash
make test
# or
pytest --cov=api --cov=train --cov-report=term-missing
```

### Run Specific Tests

```bash
# API tests only
pytest tests/test_api.py -v

# Training tests only
pytest tests/test_train.py -v

# With coverage report
pytest --cov=api --cov-report=html
```

### Test Coverage

The project maintains 100% test coverage including:
- Unit tests for data validation
- Unit tests for model evaluation
- Integration tests for all API endpoints
- End-to-end training pipeline tests

## Docker Deployment

### Build Image

```bash
# Using Makefile
make docker-build

# Using Docker directly
docker build -f api/Dockerfile -t customer-segmentation-api:latest .
```

### Run Container

```bash
# Using Makefile
make docker-run

# Using Docker directly
docker run -p 8000:8000 customer-segmentation-api:latest
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Features

- **Multi-stage build**: Optimized image size
- **Non-root user**: Enhanced security
- **Health checks**: Automatic container health monitoring
- **Layer caching**: Faster builds

## Development

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
mypy api/main.py train.py --ignore-missing-imports
```

### Pre-commit Checks

The CI pipeline runs:
- `black` for code formatting
- `isort` for import sorting
- `flake8` for linting
- `mypy` for type checking
- `pytest` with coverage

## CI/CD Pipeline

### Continuous Integration

On every push and PR:
1. **Test** across Python 3.10, 3.11, 3.12
2. **Lint** with flake8, black, isort
3. **Type check** with mypy
4. **Security scan** with Trivy
5. **Docker build** and health check

### Continuous Deployment

On push to `main`:
1. Build and push Docker image to GitHub Container Registry
2. Create GitHub release for version tags
3. Run full test suite

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | 1 | Unbuffered Python output |
| `LOG_LEVEL` | INFO | Logging level |

## Performance

- **Prediction latency**: < 50ms (p95)
- **Model loading time**: < 2s
- **Docker image size**: ~200MB (multi-stage build)
- **Memory usage**: ~150MB (runtime)

## Troubleshooting

### Models not found

```
RuntimeError: Artifacts not found. Expected 'artifacts/kmeans_model.pkl'
```

**Solution**: Train the model first:
```bash
python train.py --input data/sample_customers.csv
```

### Validation errors

```
422 Unprocessable Entity: age: ensure this value is less than or equal to 150
```

**Solution**: Check input ranges:
- Age: 0-150
- Income: >= 0
- Spending score: 0-100

### Docker health check failing

```bash
# Check container logs
docker logs <container_id>

# Test health endpoint manually
curl http://localhost:8000/health
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Format code (`make format`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | FastAPI | 0.115.2 |
| ML Library | scikit-learn | 1.5.2 |
| Server | Uvicorn | 0.30.6 |
| Serialization | joblib | 1.4.2 |
| Validation | Pydantic | Built-in |
| Testing | pytest | 7.4.3 |
| Container | Docker | Latest |
| CI/CD | GitHub Actions | Latest |

## License

MIT License - see LICENSE file for details

## Author

Built by [@udaymukhija3](https://github.com/udaymukhija3)

## Acknowledgments

- FastAPI for the excellent web framework
- scikit-learn for ML algorithms
- The open-source community

---

**Need help?** Open an issue on [GitHub](https://github.com/udaymukhija3/customer_segment/issues)
