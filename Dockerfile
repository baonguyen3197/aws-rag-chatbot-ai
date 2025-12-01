FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc unzip curl ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 600 -r requirements.txt

RUN curl -fsSL https://bun.sh/install | bash -s -- --no-interaction && \
    ln -s /root/.bun/bin/bun /usr/local/bin/bun

COPY rxconfig.py .
COPY aws_rag_chatbot_ai/ ./aws_rag_chatbot_ai/
COPY assets/ ./assets/

EXPOSE 3000 8000

CMD ["reflex", "run", "--env", "prod", "--host", "0.0.0.0", "--port", "3000"]