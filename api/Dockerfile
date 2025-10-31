FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] scikit-learn joblib numpy

# Copy API and artifacts (expects build context at project root; use: docker build -f api/Dockerfile .)
COPY api/ /app/api/
COPY artifacts/ /app/artifacts/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


