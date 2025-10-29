# deep-audio-fingerprinting/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for some audio libs (adjust if not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# copy requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY . /app

# Create directories for persistent storage (models, data, logs)
RUN mkdir -p /data/models /data/indices /data/uploads

EXPOSE 8000

# Use uvicorn; adapt workers if needed
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
