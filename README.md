# Customer Segmentation API

A containerized FastAPI application for real-time customer segmentation using K-Means clustering.

## Overview

This ML system takes a customer's data (age, annual income, spending score) and instantly assigns them to a pre-defined segment. The API is built with FastAPI, uses scikit-learn for K-Means clustering, and is fully containerized with Docker.

## Project Structure

```
customer_segment/
├── train.py           # Training script to train and save K-Means model
├── api/
│   ├── main.py        # FastAPI application
│   ├── Dockerfile     # Container configuration
│   └── requirements.txt  # Python dependencies
├── artifacts/         # Directory for trained model artifacts
├── .gitignore
└── README.md
```

## Technology Stack

- **scikit-learn**: KMeans clustering and StandardScaler
- **fastapi**: Web framework for the API
- **uvicorn**: ASGI server
- **docker**: Containerization
- **python**: 3.10

## Getting Started

### 1. Training the Model

First, train the K-Means model using your customer dataset:

```bash
python train.py --input /path/to/your/customers.csv
```

**CSV Format Required:**
- Columns: `age`, `annual_income`, `spending_score`
- Header row must be present

**Output:**
- `artifacts/kmeans_model.pkl` - Trained K-Means model (k=5)
- `artifacts/scaler.pkl` - StandardScaler for feature normalization

### 2. Running the API Locally

```bash
# Install dependencies (optional, if not using Docker)
pip install fastapi uvicorn[standard] scikit-learn joblib numpy

# Run the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Using Docker

Build and run the containerized API:

```bash
# Build the Docker image (from project root)
docker build -f api/Dockerfile -t customer-seg-api .

# Run the container
docker run -p 8000:8000 customer-seg-api
```

## API Endpoints

### POST /get_segment

Predict customer segment based on their data.

**Request Body:**
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
  "segment_id": 3,
  "segment_name": "High Spender, Low Income"
}
```

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## ML Pipeline

1. **Data Preprocessing**: StandardScaler normalizes features (age, annual_income, spending_score)
2. **Clustering**: K-Means with k=5 clusters
3. **Prediction**: New customers are assigned to the nearest cluster

## Segment Names

The API includes a mapping of segment IDs to descriptive names:
- Segment 0, 1, 2, 4: Generic segments
- Segment 3: "High Spender, Low Income"

## Development

### Prerequisites
- Python 3.10+
- Docker (optional)

### Dependencies
See `api/requirements.txt` for complete list.

## License

MIT License

## Author

Built by udaymukhija3

