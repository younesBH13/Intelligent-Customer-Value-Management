FROM python:3.10-slim

WORKDIR /app

# System dependencies for LightGBM / XGBoost / CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary

COPY . .

ENV PYTHONPATH=/app/src
EXPOSE 8000 8501

