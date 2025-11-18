# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1: Build dependencies with compilation tools
# =============================================================================
FROM python:3.12-alpine AS builder

# Install build dependencies for C extensions
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    libffi-dev \
    openssl-dev \
    cargo \
    rust \
    cmake \
    make \
    linux-headers

WORKDIR /app

# Install pip and wheel
RUN pip install --no-cache-dir --upgrade pip wheel

# Copy and install dependencies
COPY requirements.txt ./
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# =============================================================================
# Stage 2: Runtime (minimal Alpine image)
# =============================================================================
FROM python:3.12-alpine

# Install only runtime dependencies (no compilers)
RUN apk add --no-cache \
    libstdc++ \
    libffi \
    openssl

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
