# embed-api

A production-minded HTTP API for the
[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)
embedding model: 1024-dimensional, L2-normalised text embeddings for 100 languages,
on CPU. By default it serves an int8-quantised ONNX version of the model: a quarter of the
size, 3.5× faster than PyTorch for a single text and 1.8× the throughput for 32, with
embeddings at cosine ≥ 0.993 to the original (mean 0.995) on our test set.

```console
$ curl -s localhost:8000/v1/embed -H 'content-type: application/json' \
    -d '{"input": ["København er Danmarks hovedstad.", "Copenhagen is the capital of Denmark."],
         "input_type": "passage"}' | jq -c '.embeddings[].embedding |= (.[:2] | map(.*1e4 | round/1e4))'
{"model":"intfloat/multilingual-e5-large","input_type":"passage","dimension":1024,
 "embeddings":[{"index":0,"embedding":[0.0435,0.0155],"tokens":9,"truncated":false},
               {"index":1,"embedding":[0.0371,0.0077],"tokens":11,"truncated":false}],
 "usage":{"total_tokens":20}}
```

(The `jq` filter shortens each 1024-number vector to two rounded numbers for display.) Interactive
Swagger docs are at **`/docs`** once the server is running.

## Quickstart

**Requirements:** Python 3.12 and [uv](https://docs.astral.sh/uv/). Building the int8 model
downloads 2.2 GB and needs ~9 GB of free RAM for about a minute; serving needs ~2.5 GB.

```sh
uv sync --no-dev                                 # the API: ONNX Runtime, no torch
uv run scripts/export_onnx.py                    # pinned revision -> int8 ./models/e5-int8
uv run --no-dev uvicorn embed_api.main:app --port 8000          # ready in ~2 s
# then open http://localhost:8000/docs, or in another shell:
uv run --no-dev python examples/demo.py http://localhost:8000   # DA/EN/DE similarity
```

(`--no-dev` matters on `uv run` too: without it, uv installs the dev group, torch included.)

The scripts carry their own dependencies (PEP 723), so the export tooling never enters
the API's environment.

**The PyTorch backend** serves the original fp32 model. It is the reference the int8 model
is tested against, and the quickest way to try a smaller e5 model. Plain `uv sync` installs
it, along with the test and lint tools:

```sh
uv sync
uv run scripts/download_model.py                 # pinned fp32 revision -> ./models/e5 (2.2 GB)
EMBED_BACKEND=torch EMBED_MODEL_PATH=models/e5 uv run uvicorn embed_api.main:app --port 8000
EMBED_BACKEND=torch EMBED_MODEL_ID=intfloat/multilingual-e5-small EMBED_MODEL_REVISION=main \
  uv run uvicorn embed_api.main:app --port 8000
```

### Docker

The image builds the int8 model from the pinned revision in a separate build stage, so the
runtime image has no torch, needs no network access, and is 0.6 GB compressed (amd64). The
build itself downloads 2.2 GB and needs ~9 GB of RAM: give Docker Desktop's VM enough.

```sh
docker build -t embed-api .
docker run --rm -p 8000:8000 embed-api
```

CI builds the image, starts it with a read-only root filesystem, checks it embeds, and only
then pushes it to GHCR.

### Tests

```sh
uv run pytest              # fast suite (fake model, <1 s), runs in CI
uv run pytest -m slow      # real models in ./models: both backends, int8 vs fp32, batch invariance
uv run ruff check && uv run ruff format --check
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/embed` | Embed one text or a list of up to 64 |
| `GET /v1/info` | Model id, revision, backend, dimension and limits |
| `GET /health/live` | The process is up (liveness); 503 if the model failed to load |
| `GET /health/ready` | The model is loaded and warmed up (readiness); 503 until then |
| `GET /docs` | Swagger UI (`/openapi.json` for the schema) |

**`input_type` is required.** e5 was trained with a `query: ` / `passage: ` prefix and
quality drops without it. The server adds the prefix, so clients send raw text and cannot
forget it. Use `query` for search queries *and* for symmetric tasks such as similarity or
clustering; use `passage` for the documents being searched.

**Truncation.** The model reads at most 512 tokens, counting the prefix and special
tokens. Longer inputs are truncated rather than rejected, and each embedding reports
`tokens` and `truncated` so the caller knows.

**Errors.** Every error, validation included, uses one envelope and carries the request id:

```json
{"error": {"code": "validation_error", "message": "Request validation failed.",
           "request_id": "6d12931dc21443d0",
           "details": [{"loc": ["body", "input", 1], "type": "value_error",
                        "msg": "Value error, must contain non-whitespace characters"}]}}
```

Validation details say where and why, but unlike FastAPI's default they do not echo the
offending input back.

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | Malformed `Content-Length` |
| 413 | `request_too_large` | Body over `EMBED_MAX_BODY_BYTES` |
| 422 | `validation_error` | Missing/unknown `input_type`, empty or blank text, >64 inputs, >8000 chars, extra fields |
| 422 | `token_budget_exceeded` | Request needs more than `EMBED_MAX_TOTAL_TOKENS` |
| 500 | `internal_error` | Anything unexpected; details are only in the server log |
| 503 | `model_not_ready` / `model_load_failed` | Still loading, or loading failed |

## Configuration

All settings are environment variables with the `EMBED_` prefix.

| Variable | Default | |
|---|---|---|
| `EMBED_BACKEND` | `onnx` | `onnx` (int8, default) or `torch` (fp32 reference) |
| `EMBED_MODEL_PATH` | `models/e5-int8` for onnx | Local model directory. For torch, unset means download `EMBED_MODEL_ID` |
| `EMBED_MODEL_ID` | `intfloat/multilingual-e5-large` | torch: Hub id and the name reported (onnx reports what the export recorded) |
| `EMBED_MODEL_REVISION` | pinned commit | torch: Hub revision |
| `EMBED_DEVICE` | auto | torch only: `cpu`, `cuda` or `mps` |
| `EMBED_NUM_THREADS` | runtime default | Inference and tokenizer threads; set to the container's CPU limit |
| `EMBED_ENCODE_BATCH_SIZE` | `16` | torch only: inputs per forward pass |
| `EMBED_MAX_CONCURRENT_BATCHES` | `1` | Requests running inference at once |
| `EMBED_MAX_TOTAL_TOKENS` | `8192` | Token budget per request |
| `EMBED_MAX_BODY_BYTES` | `1000000` | Largest accepted body |
| `EMBED_LOG_LEVEL` | `INFO` | |

## Design decisions

**Inference stays off the event loop.** Encoding is CPU-bound and blocking. It runs in a
worker thread, so `/health/*` keeps answering while a large request is being embedded. A
semaphore lets one request run inference at a time (configurable): on CPU, one request
using every core beats several competing for them. Requests queue for it without a timeout
today; a timeout with `Retry-After` is the first item in [IMPROVEMENTS.md](IMPROVEMENTS.md).

**Bounded work per request.** Pydantic limits (64 inputs, 8000 characters each) only apply
after the body has been read, so a middleware rejects oversized bodies first. The limit
that actually bounds CPU time and memory is the **token budget**: a full 8192-token request
takes ~8 s on 4 cores, and 64 × 512 tokens would take four times that.

**Background model loading.** The model loads in a background task after the server starts.
Liveness answers immediately and readiness returns 503 until the model is warm, so an
orchestrator can tell "starting" from "broken". If loading fails, liveness fails too, since
only a restart can fix that.

**int8 ONNX by default, without torch.** Quantisation happens once, at build time, from the
fp32 ONNX file the model repository ships (ONNX Runtime's dynamic quantiser, AVX2 settings;
the repository's own int8 file targets AVX-512 VNNI). At runtime the API needs only ONNX
Runtime and a tokenizer: tokenise, truncate to 512, mean-pool, normalise, the same
computation sentence-transformers does. A slow test checks the backends agree: same token
counts, cosine > 0.98. The PyTorch backend stays as the fp32 reference and for GPUs.

**One text per forward pass, for reproducible embeddings.** Dynamic quantisation picks its
activation scale from the whole input tensor. Batched, a text's int8 embedding depended on
the other texts in the request (cosine ~0.994 to itself embedded alone), which breaks
caching and deduplication by text. The ONNX backend therefore runs texts one at a time, and
a test asserts batch invariance. That costs ~10% throughput at 32 texts and saves padding
work.

**Thread count follows the CPU limit.** ONNX Runtime, PyTorch and the tokenizer size their
thread pools from the host's cores, not the container's CPU quota, and would be throttled
against a smaller limit. `EMBED_NUM_THREADS` is set from the limit and applies to all three.

**Logging.** Structured JSON on stdout, one access-log line per request with the request
id, latency, input count, token count and truncation count. **Input text is never logged**
(embedding inputs are often user data): not in access lines, not in validation errors, and
not in unexpected-error logs, which record the exception type and stack but not its message.
Tests check this against the real log output. Every response carries an `X-Request-ID`
header (a caller-supplied id is reused if it is safe), and the same id is on every log line
for that request and in every error body.

**Reproducible model.** The revision is pinned in one place, and only the files needed are
downloaded; the model repository also carries a `.bin` duplicate, ONNX and OpenVINO exports,
~9.5 GB in all. The export records the model id and revision it was built from, and
`/v1/info` reports those.

**Testable without the model.** The API depends on a small `Embedder` protocol. The fast
suite uses a deterministic fake and covers the HTTP contract in under a second; a separate
`slow` suite checks the real models, comparing orderings rather than absolute scores,
because e5 similarities cluster in 0.7–1.0.

## Performance

`scripts/benchmark.py` times each backend through the API's own embedder classes, at 4
threads, on ~30-token inputs (medians of 10 runs, AMD Ryzen 9 5950X). Fidelity is measured
against PyTorch fp32 on 12 Danish/English query/passage pairs: cosine similarity between
the embeddings, and whether each query's top-ranked passage is unchanged.

| Backend | Weights | 1 text | 8 texts | 32 texts | Texts/s | Cosine vs fp32 (mean / min) | Same top-1 |
|---|---|---|---|---|---|---|---|
| PyTorch fp32 | 2,240 MB | 115 ms | 485 ms | 1,829 ms | 17 | 1.0000 / 1.0000 | 100% |
| ONNX Runtime fp32 | 2,236 MB | 58 ms | 458 ms | 1,816 ms | 18 | 1.0000 / 1.0000 | 100% |
| **ONNX Runtime int8 (served)** | 562 MB | 33 ms | 262 ms | 1,037 ms | 31 | 0.9951 / 0.9929 | 100% |

Where the speed comes from: for a single text, ONNX Runtime alone halves latency and int8
cuts it by another ~40%. For 32 texts the fp32 ONNX row loses its edge because it runs one
text at a time (the same code path as int8), so the 1.8× throughput there is int8's. Twelve
pairs make the fidelity numbers a sanity check, not an evaluation. Rerun with
`uv run python scripts/benchmark.py --threads 4` (it downloads the fp32 ONNX file once).

## Deployment

The image runs on a two-node k3s homelab, deployed GitOps-style: CI pushes to GHCR,
argocd-image-updater picks up the new digest, ArgoCD rolls it out. The pod has a 4-CPU
limit (hence the 4 threads above), a read-only root filesystem, readiness on
`/health/ready` and liveness on `/health/live`. On that node (Ryzen 5 6600U), through the
ingress: model ready in 1.7 s, 68 ms for one text, 1.4 s for 32, 9.9 s for a full
8192-token request, 1.45 GB peak memory. The
manifests live in a separate infrastructure repository.

## Project layout

```
src/embed_api/
  main.py       app factory, lifespan, routes
  embedder.py   Embedder protocol, ONNX and PyTorch backends, EmbeddingService
  schemas.py    request/response models (drive validation and the OpenAPI docs)
  logging.py    JSON logging, request-id/access-log and body-size middleware
  errors.py     error envelope and exception handlers
  config.py     settings from EMBED_* environment variables
scripts/export_onnx.py      build the int8 ONNX model from the pinned revision
scripts/download_model.py   fetch the pinned fp32 model (torch backend); pins the revision
scripts/benchmark.py        speed and fidelity of the backends
scripts/smoke_test.sh       run the built image and check it embeds (CI)
examples/demo.py            cross-lingual similarity demo
tests/                      fast (fake model) and slow (real model) suites
```

## Limitations and next steps

The first version deliberately leaves some things out; [IMPROVEMENTS.md](IMPROVEMENTS.md)
lists them. The main ones are a queue timeout, authentication and rate limiting, and
Prometheus metrics. For a high-traffic deployment, a purpose-built server such as Hugging
Face Text Embeddings Inference would be a strong alternative to a hand-written one.
