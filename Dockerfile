##### Stage 0: Provide Bun (avoid Reflex downloading Bun; no unzip/curl needed)
FROM --platform=linux/amd64 oven/bun:debian AS bun

##### Stage 1: Build Python venv with deps
FROM --platform=linux/amd64 python:3.11-slim-bookworm AS builder
WORKDIR /app

COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --prefer-binary -r requirements.txt \
 && find /opt/venv -type d -name __pycache__ -exec rm -rf {} + \
 && find /opt/venv -type f -name '*.py[co]' -delete

##### Stage 2: Runtime (no apt-get to avoid OOM/network issues)
FROM --platform=linux/amd64 python:3.11-slim-bookworm
WORKDIR /app

# Copy Bun to skip Reflex auto-install (so unzip/curl not required)
COPY --from=bun /usr/local/bin/bun /usr/local/bin/bun

# Python environment
COPY --from=builder /opt/venv /opt/venv

# Application code
COPY rxconfig.py ./
COPY aws_rag_chatbot_ai/ ./aws_rag_chatbot_ai/
COPY assets/ ./assets/

ENV PATH=/opt/venv/bin:$PATH \
	PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1

EXPOSE 3000 8000

CMD ["reflex", "run", "--env", "prod", "--host", "0.0.0.0", "--port", "3000"]