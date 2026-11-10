
FROM python:3.13.9-slim

WORKDIR /app

# Install OpenMP runtime required by XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy application and trained model
COPY predict.py .
COPY xgb_model.bin .

# FastAPI port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
