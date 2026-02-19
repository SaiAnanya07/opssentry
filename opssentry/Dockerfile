# Multi-stage Dockerfile for OpsSentry
# Stage 1: Base image with Python
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Create non-root user
RUN useradd -m -u 1000 opssentry && \
    chown -R opssentry:opssentry /app

# Copy application code
COPY --chown=opssentry:opssentry . /app/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R opssentry:opssentry /app/data /app/models /app/logs

# Switch to non-root user
USER opssentry

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Set entrypoint
ENTRYPOINT ["python"]
CMD ["app.py"]
