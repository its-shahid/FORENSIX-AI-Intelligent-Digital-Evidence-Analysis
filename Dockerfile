FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LOW_CONFIDENCE_THRESHOLD=0.65 \
    MODEL_PATH=models/best_model.pth \
    PORT=8000

WORKDIR /app

# Install minimal runtime libs for Pillow/OpenMP if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY backend/ ./backend/
COPY models/ ./models/
COPY results/ ./results/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
