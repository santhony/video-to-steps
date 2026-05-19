# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# System deps: ffmpeg is the only one not already in the slim base.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code (Dockerfile builds with the .dockerignore filter below).
COPY . .

# Default to 0.0.0.0 inside the container so a port publish actually works.
# A reverse proxy is still required for any public binding; see README.
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8090
EXPOSE 8090

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]
