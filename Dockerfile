# syntax=docker/dockerfile:1.7

# --- deps: the locked runtime environment ----------------------------------
# Default dependencies only: ONNX Runtime + tokenizers, no torch.
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

# --- model: download the pinned revision, quantise to int8 ONNX ------------
# Depends only on the two scripts (which pin the revision and the tooling), so
# it is rebuilt only when they change. Quantising needs ~8.5 GB of RAM and the
# 2.2 GB fp32 files; only the ~580 MB result is copied out.
FROM python:3.12-slim AS model
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never
COPY scripts/download_model.py scripts/export_onnx.py /scripts/
RUN uv run /scripts/download_model.py --dest /build/e5 \
    && uv run /scripts/export_onnx.py --src /build/e5 --dest /models/e5-int8 \
    && rm -rf /build /root/.cache

# --- runtime ---------------------------------------------------------------
FROM python:3.12-slim AS runtime
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
# Model first: it changes least, so app/dep changes don't invalidate it.
COPY --from=model /models /models
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Bounds glibc arena growth from variable-shaped allocations.
    MALLOC_ARENA_MAX=2 \
    EMBED_BACKEND=onnx \
    EMBED_MODEL_PATH=/models/e5-int8
USER 10001
EXPOSE 8000
CMD ["uvicorn", "embed_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
