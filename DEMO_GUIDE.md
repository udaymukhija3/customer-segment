# Demo Guide - Customer Segmentation API

This guide helps you quickly demonstrate the project's functionality for recruiters, interviewers, or stakeholders.

## 🚀 Quick Demo Setup (5 minutes)

### Step 1: Install and Train

```bash
# Clone and navigate to project
cd customer_segment

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r api/requirements.txt

# Train the model (takes ~3 seconds)
python train.py --input data/sample_customers.csv
```

**Expected Output**:
```
Samples trained: 191
Number of clusters: 5
Silhouette Score: 0.3759
```

### Step 2: Start the API

```bash
# In Terminal 1
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Verify**: Open http://localhost:8000/health
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Step 3: Start the Frontend

```bash
# In Terminal 2 (new terminal)
cd frontend
python3 -m http.server 3000
```

**Verify**: Open http://localhost:3000

---

## 🎯 Demo Script

Use this script when walking through the project with someone:

### 1. Introduction (30 seconds)

> "This is a production-ready ML system for customer segmentation. It demonstrates end-to-end MLOps practices, from model training to API deployment, with a modern web interface."

**Show**: README.md badges and project highlights

### 2. Architecture Overview (1 minute)

> "The system has three main components:"
> 1. **Training Pipeline** - Automated model training with evaluation metrics
> 2. **REST API** - FastAPI with dependency injection and health monitoring
> 3. **Web Interface** - Real-time predictions with visual feedback

**Show**: Project structure in README.md

### 3. Frontend Demo (2 minutes)

**Navigate to**: http://localhost:3000

> "The frontend provides an intuitive interface for testing predictions."

**Demo these examples**:

**Example 1: Premium Customer**
- Age: 30
- Income: 120000
- Spending Score: 85
- **Expected**: Segment 4 - High Income, High Spending

**Example 2: Budget Shopper**
- Age: 45
- Income: 35000
- Spending Score: 25
- **Expected**: Segment 0 - Low Income, Low Spending

**Example 3: Conservative Wealthy**
- Age: 55
- Income: 150000
- Spending Score: 20
- **Expected**: Segment 2 - High Income, Low Spending

**Highlight**:
- Live API status indicator
- Input validation
- Confidence score visualization
- Smooth animations

### 4. API Documentation (1 minute)

**Navigate to**: http://localhost:8000/docs

> "FastAPI auto-generates OpenAPI documentation. Let's test the API directly."

**Demo in Swagger UI**:
1. Expand `/predict` endpoint
2. Click "Try it out"
3. Use example payload:
```json
{
  "age": 35,
  "annual_income": 75000,
  "spending_score": 62
}
```
4. Click "Execute"
5. Show response with segment prediction

**Highlight**:
- Interactive documentation
- Request/response schemas
- Built-in validation

### 5. Code Quality (1 minute)

**Show in terminal**:

```bash
# Run tests
pytest tests/ -v

# Show coverage
pytest --cov=api --cov=train --cov-report=term-missing
```

**Expected**: 100% test coverage

**Highlight**:
- Comprehensive testing
- Unit and integration tests
- CI/CD pipeline

### 6. Model Training (1 minute)

**Show**: `train.py` file

> "The training pipeline includes:"
> - Data validation (NaN checks, range validation)
> - Feature scaling with StandardScaler
> - Model evaluation (silhouette score, Davies-Bouldin index)
> - Metadata tracking for versioning

**Demo**:
```bash
python train.py --input data/sample_customers.csv --n_clusters 3
```

**Show**: Generated files in `artifacts/`
- `kmeans_model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `model_metadata.json` - Training metadata

### 7. Production Features (1 minute)

> "This isn't just a prototype - it has production-grade features:"

**Show**:

1. **Health Checks**
```bash
curl http://localhost:8000/health
```

2. **Model Info**
```bash
curl http://localhost:8000/model/info | python3 -m json.tool
```

3. **Docker Support**
```bash
docker-compose up -d
```

4. **CI/CD Pipeline**
- Show `.github/workflows/ci.yml`
- Multi-Python version testing
- Security scanning with Trivy
- Automated Docker builds

**Highlight**:
- Structured logging
- Error handling
- CORS support
- Dependency injection
- Type safety with Pydantic

### 8. Scalability Discussion (1 minute)

> "For production scale, this can be deployed to:"

- **AWS**: ECS Fargate, EKS, or Lambda
- **GCP**: Cloud Run, GKE
- **Azure**: Container Apps, AKS
- **Kubernetes**: Ready for horizontal scaling

**Show**: `docker-compose.yml` and `api/Dockerfile`

---

## 📊 Key Talking Points

### Technical Excellence

✅ **100% Test Coverage** - pytest with comprehensive unit/integration tests
✅ **Type Safety** - Full type hints with mypy validation
✅ **Clean Architecture** - Dependency injection, SOLID principles
✅ **Production-Ready** - Health checks, logging, error handling
✅ **Modern Stack** - FastAPI, Pydantic, Docker, GitHub Actions

### MLOps Best Practices

✅ **Automated Pipeline** - Data validation, training, evaluation
✅ **Model Versioning** - Metadata tracking with timestamps
✅ **Evaluation Metrics** - Multiple clustering metrics
✅ **Reproducibility** - Fixed random seeds, logged hyperparameters
✅ **Artifact Management** - Serialized models with joblib

### DevOps Integration

✅ **CI/CD** - Multi-Python testing, linting, security scanning
✅ **Containerization** - Multi-stage Docker builds
✅ **Documentation** - Auto-generated API docs, comprehensive README
✅ **Monitoring** - Health endpoints, structured logging
✅ **Security** - Trivy scanning, non-root containers, input validation

---

## 🎬 Quick Command Reference

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "annual_income": 75000, "spending_score": 62}'

# Access documentation
open http://localhost:8000/docs
```

### Testing Suite

```bash
# Run all tests
pytest -v

# With coverage
pytest --cov=api --cov=train --cov-report=term-missing

# Specific tests
pytest tests/test_api.py -v
pytest tests/test_train.py -v
```

### Code Quality

```bash
# Format code
black api/ train.py tests/
isort api/ train.py tests/

# Linting
flake8 api/ train.py --max-line-length=120

# Type checking
mypy api/main.py train.py --ignore-missing-imports
```

### Docker

```bash
# Build image
docker build -f api/Dockerfile -t customer-segmentation-api .

# Run container
docker run -p 8000:8000 customer-segmentation-api

# Docker Compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## 📝 Interview Questions Preparation

Be ready to answer:

1. **"Walk me through your code"**
   - Start with README, then api/main.py, then train.py
   - Highlight dependency injection, type safety, error handling

2. **"How would you scale this?"**
   - Horizontal scaling with Kubernetes
   - Caching layer (Redis)
   - Load balancing
   - Database for prediction history

3. **"What would you do differently?"**
   - Add monitoring (Prometheus/Grafana)
   - Implement A/B testing
   - Add model explainability (SHAP)
   - Database integration
   - Batch prediction endpoint

4. **"How do you ensure quality?"**
   - 100% test coverage
   - CI/CD with multi-Python testing
   - Code quality tools (black, mypy, flake8)
   - Security scanning (Trivy)

5. **"Explain your model choice"**
   - K-Means for unsupervised segmentation
   - Interpretable for business users
   - Scalable and efficient
   - Multiple evaluation metrics

**Refer to**: `TECHNICAL_QA.md` for comprehensive answers

---

## 🎥 Screen Recording Tips

If recording a demo video:

1. **Opening shot**: Show README with badges
2. **Terminal split**: API on left, frontend browser on right
3. **Demonstrate**: 3-4 different customer examples
4. **Show docs**: Swagger UI at /docs
5. **Show tests**: Run pytest with coverage
6. **Show code**: Brief walkthrough of api/main.py
7. **Closing**: Show project structure and CI/CD

**Duration**: Aim for 3-5 minutes for a quick demo

---

## 🔧 Troubleshooting

### Port Already in Use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn api.main:app --port 8001
```

### Model Not Loading

```bash
# Retrain the model
python train.py --input data/sample_customers.csv

# Verify artifacts
ls -la artifacts/
```

### CORS Issues

The API has CORS enabled. If issues persist:
1. Check `api/main.py` has CORSMiddleware
2. Restart the API server
3. Clear browser cache

### Dependencies Issues

```bash
# Clean install
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt
```

---

## 📞 Getting Help

- **Technical Q&A**: See `TECHNICAL_QA.md`
- **API Docs**: http://localhost:8000/docs
- **Frontend Guide**: See `frontend/README.md`
- **GitHub Issues**: https://github.com/udaymukhija3/customer_segment/issues

---

**Ready to impress?** Follow this guide and you'll deliver a professional, polished demonstration! 🚀
