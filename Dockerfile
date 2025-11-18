# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1: Build dependencies
# =============================================================================
FROM python:3.12-slim AS builder

# Install build dependencies for C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip and wheel
RUN pip install --no-cache-dir --upgrade pip wheel

# Copy and install dependencies
COPY requirements.txt ./
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# =============================================================================
# Stage 2: Runtime (minimal image without build tools)
# =============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy pre-built wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application source
COPY . .

# Default command
ENTRYPOINT ["python", "crew.py"]
