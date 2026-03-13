FROM python:3.11-slim

LABEL maintainer="SentinelVision Team"
LABEL description="SentinelVision Inference Service"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sentinelvision_core/ /app/sentinelvision_core/
COPY inference_service/ /app/inference_service/
COPY monitoring/ /app/monitoring/
COPY configs/ /app/configs/
COPY model_registry/ /app/model_registry/

RUN mkdir -p /app/logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "inference_service/api.py"]
