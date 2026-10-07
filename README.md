# embeda-api

A production-minded HTTP API for the
[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)
embedding model: 1024-dimensional, L2-normalised text embeddings for 100 languages,
on CPU.

```console
$ curl -s localhost:8000/v1/embed -H 'content-type: application/json' \
    -d '{"input": ["København er Danmarks hovedstad.", "Copenhagen is the capital of Denmark."],
         "input_type": "passage"}'
{
  "model": "intfloat/multilingual-e5-large",
  "input_type": "passage",
  "dimension": 1024,
  "embeddings": [
    {"index": 0, "embedding": [0.0411, 0.0129, ...], "tokens": 9, "truncated": false},
    {"index": 1, "embedding": [0.0337, 0.0093, ...], "tokens": 11, "truncated": false}
  ],
  "usage": {"total_tokens": 20}
}
```

Interactive Swagger docs are at **`/docs`** once the server is running.

## Quickstart

**Requirements:** ~4 GB free RAM, ~2.5 GB disk for the model, Python 3.12 and
[uv](https://docs.astral.sh/uv/). The model download is 2.2 GB; after that the server is
ready within about 10 seconds of starting.

```sh
uv sync
uv run python scripts/download_model.py          # pinned revision -> ./models/e5
EMBEDA_MODEL_PATH=models/e5 uv run uvicorn embeda_api.main:app --port 8000
# then open http://localhost:8000/docs
```

Skipping the download step also works: the model is then fetched from the Hugging Face Hub
at startup. To try the API quickly on a small machine, point it at the small sibling model:

```sh
EMBEDA_MODEL_ID=intfloat/multilingual-e5-small EMBEDA_MODEL_REVISION=main \
  uv run uvicorn embeda_api.main:app --port 8000
```

### Docker

The image bundles the model at a pinned revision, so it starts without network access
(amd64; ~4 GB).

```sh
docker build -t embeda-api .
docker run --rm -p 8000:8000 embeda-api
```

### Tests

```sh
uv run pytest              # fast suite (fake model), runs in CI
uv run pytest -m slow      # against the real model in ./models/e5
uv run ruff check && uv run ruff format --check
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/embed` | Embed one text or a list of up to 64 |
| `GET /v1/info` | Model id, revision, dimension and limits |
| `GET /health/live` | The process is up (liveness) |
| `GET /health/ready` | The model is loaded and warmed up (readiness); 503 until then |
| `GET /docs` | Swagger UI (`/openapi.json` for the schema) |

**`input_type` is required.** e5 was trained with a `query: ` / `passage: ` prefix and
quality drops without it. The server adds the prefix, so clients send raw text and cannot
forget it. Use `query` for search queries *and* for symmetric tasks such as similarity or
clustering; use `passage` for the documents being searched.

**Truncation.** The model reads at most 512 tokens, counting the prefix and special
tokens. Longer inputs are truncated rather than rejected, and each embedding reports
`tokens` and `truncated` so the caller knows.

**Errors.** Validation errors are FastAPI's standard 422 body, pointing at the offending
field (for example `["body", "input", 3]`). Everything else uses one envelope:

```json
{"error": {"code": "model_not_ready", "message": "The model is still loading. Retry shortly.",
           "request_id": "0b6f3c1e9a8d4f52"}}
```

| Status | `code` | When |
|---|---|---|
| 413 | `request_too_large` | Body over `EMBEDA_MAX_BODY_BYTES` |
| 422 | *(FastAPI validation)* | Missing/unknown `input_type`, empty or blank text, >64 inputs, >8000 chars, extra fields |
| 422 | `token_budget_exceeded` | Request needs more than `EMBEDA_MAX_TOTAL_TOKENS` |
| 500 | `internal_error` | Anything unexpected; details are only in the server log |
| 503 | `model_not_ready` / `model_load_failed` | Still loading, or loading failed |

## Configuration

All settings are environment variables with the `EMBEDA_` prefix.

| Variable | Default | |
|---|---|---|
| `EMBEDA_MODEL_PATH` | *(unset)* | Load the model from a local directory (the Docker image sets this) |
| `EMBEDA_MODEL_ID` | `intfloat/multilingual-e5-large` | Hub id, used when no path is set and reported in responses |
| `EMBEDA_MODEL_REVISION` | pinned commit | Hub revision, for reproducible embeddings |
| `EMBEDA_DEVICE` | auto | `cpu`, `cuda` or `mps` |
| `EMBEDA_NUM_THREADS` | torch default | Set to the container's CPU limit (see below) |
| `EMBEDA_ENCODE_BATCH_SIZE` | `16` | Inputs per forward pass |
| `EMBEDA_MAX_CONCURRENT_BATCHES` | `1` | Requests running inference at once |
| `EMBEDA_MAX_TOTAL_TOKENS` | `8192` | Token budget per request |
| `EMBEDA_MAX_BODY_BYTES` | `1000000` | Largest accepted body |
| `EMBEDA_LOG_LEVEL` | `INFO` | |

## Design decisions

**Inference stays off the event loop.** Encoding is CPU-bound and blocking. It runs in a
worker thread, so `/health/*` keeps answering while a large batch is being embedded. A
semaphore lets one batch run at a time (configurable): on CPU, one batch using every core
beats several batches competing for them.

**Bounded work per request.** Pydantic limits (64 inputs, 8000 characters each) only apply
after the body has been read, so a middleware rejects oversized bodies first. The limit
that actually bounds CPU time and memory is the **token budget**: 64 inputs of 512 tokens
would take over a minute on a small CPU and several GB of activations.

**Background model loading.** The model loads in a background task after the server starts.
Liveness answers immediately and readiness returns 503 until the model is warm, so an
orchestrator can tell "starting" from "broken". Loading inside startup would just refuse
connections.

**Thread count follows the CPU limit.** PyTorch sizes its thread pool from the host's
cores, not the container's CPU quota. On a 12-thread node with a 4-CPU limit it would start
12 threads and get throttled, so `EMBEDA_NUM_THREADS` is set from the limit.

**Logging.** Structured JSON on stdout, one access-log line per request with the request
id, latency, input count, token count and truncation count. **Input text is never logged**:
embedding inputs are often user data. Every response carries an `X-Request-ID` header (a
caller-supplied id is reused if it is safe), and the same id is on every log line for that
request and in every error body.

**Reproducible model.** The revision is pinned, and only the files the PyTorch path needs
are downloaded. The Hub repo also carries a `.bin` duplicate, ONNX and OpenVINO exports,
~9.5 GB in all. The model is loaded from a local path, so the container runs with
`HF_HUB_OFFLINE=1`.

**Testable without the model.** The API depends on a small `Embedder` protocol. The fast
suite uses a deterministic fake and covers the HTTP contract in under a second; a separate
`slow` suite checks the real model, comparing orderings rather than absolute scores,
because e5 similarities cluster in 0.7–1.0.

## Performance

`scripts/benchmark.py` compares PyTorch fp32 with ONNX Runtime fp32 and int8 (dynamic
quantisation for AVX2), at 4 threads, on ~25-token inputs. Fidelity is measured against
PyTorch fp32 on a small Danish/English retrieval set: mean cosine similarity between the
embeddings, and whether each query's top-ranked passage is unchanged.

| Backend | Weights | p50, 1 text | p50, 8 texts | p50, 32 texts | Texts/s | Cosine vs fp32 | Same top-1 |
|---|---|---|---|---|---|---|---|
| PyTorch fp32 | 2,240 MB | 114 ms | 494 ms | 1,842 ms | 17 | 1.0000 | 100% |
| ONNX fp32 | 2,236 MB | 58 ms | 339 ms | 1,335 ms | 24 | 1.0000 | 100% |
| ONNX int8 (AVX2) | 562 MB | 36 ms | 233 ms | 931 ms | 34 | 0.9948 | 100% |

int8 doubles throughput and quarters the model size with near-identical embeddings. The
retrieval set is small (12 query/passage pairs), so treat the fidelity numbers as a sanity
check rather than an evaluation. Numbers are from an AMD Ryzen 9 5950X; rerun with
`uv run scripts/benchmark.py --threads 4`. The script has its own inline dependencies,
because the ONNX tooling pins an older `transformers` than the API uses.

## Project layout

```
src/embeda_api/
  main.py       app factory, lifespan, routes
  embedder.py   Embedder protocol, sentence-transformers implementation, EmbeddingService
  schemas.py    request/response models (drive validation and the OpenAPI docs)
  logging.py    JSON logging, request-id/access-log and body-size middleware
  errors.py     error envelope and exception handlers
  config.py     settings from EMBEDA_* environment variables
scripts/download_model.py   fetch the pinned model revision
tests/                      fast (fake model) and slow (real model) suites
```

## Limitations and next steps

The first version deliberately leaves some things out; [IMPROVEMENTS.md](IMPROVEMENTS.md)
lists them. The main ones are authentication and rate limiting, Prometheus metrics, and
cross-request batching. For a high-traffic deployment, a purpose-built server such as
Hugging Face Text Embeddings Inference would be a strong alternative to a hand-written one.
