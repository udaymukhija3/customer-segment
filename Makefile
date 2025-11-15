.PHONY: help install install-dev train test lint format clean docker-build docker-run docker-compose-up docker-compose-down

help:
	@echo "Available commands:"
	@echo "  make install          - Install production dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make train           - Train the model with sample data"
	@echo "  make test            - Run tests with coverage"
	@echo "  make lint            - Run linting checks"
	@echo "  make format          - Format code with black and isort"
	@echo "  make clean           - Clean up generated files"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo "  make docker-compose-up   - Start services with docker-compose"
	@echo "  make docker-compose-down - Stop services"

install:
	pip install -r api/requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python train.py --input data/sample_customers.csv --artifacts_dir artifacts

test:
	pytest --cov=api --cov=train --cov-report=term-missing --cov-report=html

lint:
	flake8 api/ train.py --max-line-length=120 --extend-ignore=E203,W503
	black --check api/ train.py tests/
	isort --check-only api/ train.py tests/
	mypy api/main.py train.py --ignore-missing-imports

format:
	black api/ train.py tests/
	isort api/ train.py tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache htmlcov .mypy_cache .coverage coverage.xml
	rm -rf build/ dist/ *.egg-info

docker-build:
	docker build -f api/Dockerfile -t customer-segmentation-api:latest .

docker-run: docker-build
	docker run -p 8000:8000 customer-segmentation-api:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

serve:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
