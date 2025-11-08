# syntax=docker/dockerfile:1

# Base image with Python runtime
FROM python:3.12-slim

# Avoid writing bytecode files and ensure output is flushed straight away
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set the working directory inside the container
WORKDIR /app

# Install Python dependencies separately to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY . .

# Default command: run the crew. The container expects activity payloads on stdin.
ENTRYPOINT ["python", "crew.py"]
