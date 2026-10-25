# syntax=docker/dockerfile:1
# The API image (pp-api): FastAPI with the agent (Phase 5): its groups, the models config and the
# review encoder that config/retrieval.yaml pins, for search_reviews' queries. Stage 1 builds the
# virtualenv with uv; stage 2 copies only the venv and the files the API reads. deploy.yml downloads
# the pinned encoder to build/encoder first, as for the jobs image.
FROM python:3.12-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# Dependencies first, so a code change doesn't invalidate this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --group api --group retrieval --group causal --group agent
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group api --group retrieval --group causal --group agent --no-editable

FROM python:3.12-slim-trixie
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION} \
    PATH="/app/.venv/bin:$PATH" \
    PP_ENCODER_DIR=/app/encoder \
    PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
# config/models.yaml (the agent's Gemini chain) and config/retrieval.yaml (the index's pin).
COPY config ./config
# The encoder must match embedder.json and embed one query, or the image isn't built.
COPY build/encoder /app/encoder
RUN python -c "from pathlib import Path; from patchpulse.retrieval.embedders import OnnxEmbedder; OnnxEmbedder(Path('/app/encoder')).embed_queries(['the game stutters'])"
# The agent imports only when a question comes: a missing dependency fails here instead.
RUN python -c "import patchpulse.agent.graph, patchpulse.api.app"
USER app
EXPOSE 8000
CMD ["uvicorn", "patchpulse.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
