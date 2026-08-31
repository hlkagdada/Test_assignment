FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    pytest

# Copy task data available inside the Docker build context
COPY Data/ /app/data/

RUN mkdir -p /app/output
