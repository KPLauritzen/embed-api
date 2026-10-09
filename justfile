# Common tasks. `just` lists them.

port := "8000"
url := "http://localhost:" + port

default:
    @just --list

# Install the API and dev tools, and the pre-commit hooks
setup:
    uv sync
    uv run pre-commit install

# Build the int8 model the API serves (downloads 2.2 GB, needs ~8.5 GB RAM)
model:
    uv run scripts/export_onnx.py

# Download the fp32 model for the torch backend (2.2 GB)
model-torch:
    uv run scripts/download_model.py

# Run the API (int8 ONNX backend)
serve:
    uv run uvicorn embed_api.main:app --port {{port}}

# Run the API on the fp32 torch backend
serve-torch:
    EMBED_BACKEND=torch uv run --extra torch uvicorn embed_api.main:app --port {{port}}

# Demo scripts against a running server
demo target=url:
    uv run python examples/demo.py {{target}}
    uv run python examples/limits.py {{target}}

# Lint, format check, type check and fast tests: what CI runs
check: lint typecheck test

lint:
    uv run ruff check
    uv run ruff format --check

# Apply ruff fixes and formatting
fmt:
    uv run ruff check --fix
    uv run ruff format

typecheck:
    uv run ty check

# Fast tests (fake model)
test:
    uv run pytest

# Tests against the real models in ./models (both backends)
test-slow:
    uv run --extra torch pytest -m slow

# Retrieval quality of both backends on two Danish MTEB tasks (~10-15 min on CPU)
eval:
    uv run --extra torch --with datasets python scripts/eval_retrieval.py

# Speed and fidelity of the backends
bench:
    uv run --extra torch python scripts/benchmark.py

docker-build:
    docker build -t embed-api .

docker-run:
    docker run --rm -p {{port}}:8000 embed-api
