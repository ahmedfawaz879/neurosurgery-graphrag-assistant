# syntax=docker/dockerfile:1

# ---- builder: installs dependencies into a venv (build toolchain lives only here) ----
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ---- runtime: slim image, no build toolchain, starts a server ------------------------
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src ./src
COPY data ./data
COPY static ./static

# Runtime configuration -- override at deploy time (docker-compose.yml, render.yaml,
# fly.toml). Never bake OPENAI_API_KEY or any other secret into the image.
ENV OPENAI_API_KEY="" \
    QDRANT_URL="http://qdrant:6333" \
    GRAPH_BACKEND="networkx" \
    ALLOWED_ORIGIN="*"

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
