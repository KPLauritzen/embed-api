# syntax=docker/dockerfile:1.7

# --- deps: resolve the locked environment (CPU-only torch) -----------------
FROM python:3.12-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# --- model: fetch the pinned revision, only the files we load --------------
# Independent of the app's code and deps, so the 2.2 GB layer is only rebuilt
# when the download script (and with it the pinned revision) changes.
FROM python:3.12-slim AS model
RUN pip install --no-cache-dir "huggingface-hub==1.33.0"
COPY scripts/download_model.py /scripts/
RUN python /scripts/download_model.py --dest /models/e5

# --- runtime ---------------------------------------------------------------
FROM python:3.12-slim AS runtime
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
# Model first: it changes least, so app/dep changes don't invalidate it.
COPY --from=model /models /models
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    # Writable even with a read-only root filesystem (/tmp is a mount).
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    # Bounds glibc arena growth from variable-shaped tensor allocations.
    MALLOC_ARENA_MAX=2 \
    EMBED_MODEL_PATH=/models/e5
USER 10001
EXPOSE 8000
CMD ["uvicorn", "embed_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
