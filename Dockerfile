# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies to /opt/venv
COPY requirements.txt ./
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete

# Stage 2: Production
FROM python:3.11-slim

WORKDIR /app

# Install minimal runtime dependencies (unzip needed for Reflex/Bun)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        unzip \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    

# Copy Python packages from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY aws_rag_chatbot_ai/ ./aws_rag_chatbot_ai/
COPY rxconfig.py ./
COPY assets/ ./assets/

# Update PATH and environment
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose application ports
EXPOSE 3000 8000

# Default command
CMD ["reflex", "run", "--env", "prod"]