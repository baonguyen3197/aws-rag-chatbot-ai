# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder to a user-accessible location
COPY --from=builder /root/.local /opt/venv

# Copy application code
COPY aws_rag_chatbot_ai/ ./aws_rag_chatbot_ai/
COPY rxconfig.py ./
COPY assets/ ./assets/

# Create unprivileged user and set ownership
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /opt/venv

# Update PATH to include user-installed packages
ENV PATH=/opt/venv/bin:$PATH

USER appuser

# Expose application ports
EXPOSE 3000 8000

# Default command
CMD ["reflex", "run", "--env", "prod"]