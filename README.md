# embed-api

An HTTP API for the
[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)
embedding model. It returns 1024-dimensional, L2-normalised embeddings for text in about 100
languages and runs on CPU.

By default it serves an int8-quantised ONNX version of the model. Compared with the original
PyTorch model it is a quarter of the size and 3.5x faster for a single text, and it loses
1-3% nDCG@10 on two Danish retrieval benchmarks.

```console
$ curl -s localhost:8000/v1/embed -H 'content-type: application/json' \
    -d '{"input": ["København er Danmarks hovedstad.", "Copenhagen is the capital of Denmark."],
         "input_type": "passage"}' | jq -c '.embeddings[].embedding |= (.[:2] | map(.*1e4 | round/1e4))'
{"model":"intfloat/multilingual-e5-large","input_type":"passage","dimension":1024,
 "embeddings":[{"index":0,"embedding":[0.0435,0.0155],"tokens":9,"truncated":false},
               {"index":1,"embedding":[0.0371,0.0077],"tokens":11,"truncated":false}],
 "usage":{"total_tokens":20}}
```

The `jq` filter cuts each vector down to its first two values. Swagger docs are served at
`/docs`.

## Quickstart

You need [uv](https://docs.astral.sh/uv/), which installs Python 3.12 if it is missing.
Building the int8 model downloads 2.2 GB and needs about 8.5 GB of free RAM for a minute.
The server itself peaks at about 1.5 GB.

```sh
uv sync                                          # API and dev tools (no torch)
uv run scripts/export_onnx.py                    # builds ./models/e5-int8
uv run uvicorn embed_api.main:app --port 8000    # ready after ~2 s
```

Then open http://localhost:8000/docs, or run the examples from another shell:

```sh
uv run python examples/demo.py http://localhost:8000     # similarity across Danish, English, German
uv run python examples/limits.py http://localhost:8000   # truncation and the token budget
```

The scripts in `scripts/` declare their own dependencies (PEP 723), so the export tools are not
installed into the API's environment. The Hugging Face download prints a warning about
unauthenticated requests. It is harmless; set `HF_TOKEN` to hide it.

To run the original fp32 model instead, use the PyTorch backend. It needs the `torch` extra
and its own download:

```sh
uv run scripts/download_model.py                 # downloads ./models/e5 (2.2 GB)
EMBED_BACKEND=torch uv run --extra torch uvicorn embed_api.main:app --port 8000
```

The torch backend can also serve
[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) (text only; its
vision and audio encoders are not loaded). The server reads the model's `SearchQuery` and
`Document` prompts from its `config_sentence_transformers.json` and uses them in place of
e5's `query: ` and `passage: ` prefixes:

```sh
uv run scripts/download_model.py --model embeddinggemma-2   # ./models/embeddinggemma-2
EMBED_BACKEND=torch EMBED_MODEL_PATH=models/embeddinggemma-2 \
    uv run --extra torch uvicorn embed_api.main:app --port 8000
```

### Docker

```sh
docker build -t embed-api .
docker run --rm -p 8000:8000 embed-api
```

The int8 model is built in a separate build stage, so the final image contains neither torch
nor the fp32 weights. It is 0.6 GB compressed (amd64) and needs no network access at runtime.
The build downloads 2.2 GB and needs about 8.5 GB of RAM, which is more than Docker Desktop
gives its VM by default.

CI builds the image, starts it with a read-only root filesystem, checks that it returns an
embedding, and then pushes that image to GHCR.

### Development

Common tasks are in the [justfile](justfile). Run `just` to list them.

```sh
just setup        # uv sync and install the pre-commit hooks
just check        # ruff, ty and the fast tests (CI runs the same command)
just test-slow    # tests against the real models
```

The fast tests use a fake model and take under a second. The slow tests need both models
(`just model model-torch`) and only run locally. The pre-commit hooks run ruff, ty, a
`uv.lock` check and a guard against committing large files such as model weights. Type
checking uses [ty](https://github.com/astral-sh/ty), pinned to an exact version because it is
still pre-1.0. The scripts in `scripts/` have their own dependencies and are not type-checked.

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/embed` | Embed one text or a list of up to 64 |
| `GET /v1/info` | Model id, revision, backend, dimension and limits |
| `GET /health/live` | Liveness: 200 unless the model failed to load |
| `GET /health/ready` | Readiness: 503 until the model is loaded |
| `GET /docs` | Swagger UI (`/openapi.json` for the schema) |

`input_type` is required. e5 was trained with `query: ` and `passage: ` prefixes and gives
worse results without them, so the server adds the prefix itself. Use `query` for search
queries and for symmetric tasks such as similarity or clustering, and `passage` for the
documents being searched.

The model reads at most 512 tokens, including the prefix and two special tokens. Longer texts
are truncated, not rejected, and each embedding in the response reports how many `tokens`
were used and whether the text was `truncated`.

### Errors

Every error response has the same shape and includes the request id:

```json
{"error": {"code": "validation_error", "message": "Request validation failed.",
           "request_id": "6d12931dc21443d0",
           "details": [{"loc": ["body", "input", 1], "type": "value_error",
                        "msg": "Value error, must contain non-whitespace characters"}]}}
```

Unlike FastAPI's default, validation errors do not echo the rejected input back.

| Status | `code` | When |
|---|---|---|
| 404, 405 | `not_found`, `method_not_allowed` | Unknown path or wrong method |
| 413 | `request_too_large` | Body larger than `EMBED_MAX_BODY_BYTES` |
| 422 | `validation_error` | Missing or unknown `input_type`, empty text, more than 64 inputs or 8000 characters, unknown fields |
| 422 | `token_budget_exceeded` | Request needs more than `EMBED_MAX_TOTAL_TOKENS` tokens |
| 500 | `internal_error` | Unexpected error; details are only in the server log |
| 503 | `model_not_ready`, `model_load_failed` | Model still loading, or loading failed |
| 503 | `overloaded` | No inference slot within `EMBED_QUEUE_TIMEOUT_SECONDS`; sets `Retry-After` |

Requests that are not valid HTTP, such as a non-numeric `Content-Length`, are rejected by
uvicorn with a plain-text 400 before they reach the app.

## Configuration

All settings are environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `EMBED_BACKEND` | `onnx` | `onnx` (int8) or `torch` (fp32) |
| `EMBED_MODEL_PATH` | `models/e5-int8`, or `models/e5` for torch | Model directory |
| `EMBED_NUM_THREADS` | runtime default | Threads for inference and tokenisation; set it to the container's CPU limit |
| `EMBED_QUEUE_TIMEOUT_SECONDS` | `30` | How long a request waits for inference before `503 overloaded` |
| `EMBED_MAX_TOTAL_TOKENS` | `8192` | Token budget per request |
| `EMBED_MAX_BODY_BYTES` | `1000000` | Largest accepted request body |
| `EMBED_LOG_LEVEL` | `INFO` | Log level |

## Design decisions

### One inference at a time, off the event loop

Inference is CPU-bound, so it runs in a worker thread and the server keeps answering health
checks during a long request. Only one request runs inference at a time. On a CPU, one
request using all cores finishes sooner than several requests sharing them.

### A bounded queue

A request waits at most `EMBED_QUEUE_TIMEOUT_SECONDS` for the inference slot and then gets
`503 overloaded` with a `Retry-After` header. uvicorn does not stop a handler when its client
disconnects, so when a request reaches the slot, the server first checks that the client is
still connected and drops the request if not (logged with status 499). Otherwise a client that
times out and retries would leave its abandoned requests in the queue ahead of everyone else.
In a test where five clients sent full-budget requests and gave up after 0.3 s, a following
one-text request waited 7.9 s, the time of the one request already running. Without the check
it waited 27 s. The access log records each request's time in the queue as `queue_ms`.

### Limits on request size

Pydantic's limits (64 texts, 8000 characters each) are checked only after the whole body has
been read. A middleware therefore counts the body's bytes as they arrive and answers 413 once
`EMBED_MAX_BODY_BYTES` is exceeded, with or without a `Content-Length` header. The limit that
actually bounds CPU time and memory is the token budget. A request with the full 8192 tokens
takes about 8 s on 4 cores of the benchmark machine and 10 s on the deployment server.

### Loading the model in the background

The model loads in a background task after the server starts. `/health/live` answers at once
and `/health/ready` returns 503 until the model is loaded, so an orchestrator can tell a
starting server from a broken one. If loading fails, `/health/live` returns 503 too, because
only a restart can fix that.

### int8 ONNX by default, without torch

The int8 model is built once, at build time, by quantising the fp32 ONNX export that the
model's repository provides. It uses ONNX Runtime's dynamic quantisation with AVX2 settings.
The repository also has an int8 file, but it targets AVX-512 VNNI, which the deployment CPU
does not have. At runtime the API needs only ONNX Runtime and the tokenizer: it tokenises,
truncates to 512 tokens, averages the token vectors and normalises the result, as
sentence-transformers does. The slow tests check that the token counts match
sentence-transformers and that int8 embeddings stay above cosine 0.98 to fp32.
[Retrieval quality](#retrieval-quality) below measures what int8 costs. The PyTorch backend
remains as the fp32 reference and for GPUs.

### One text per forward pass

Dynamic quantisation chooses its activation scale from the whole input tensor. If several
texts are run as a batch, each text's embedding depends on the other texts in it: the same
text embedded alone and in a batch gave vectors with cosine 0.994. That would make the API
return different vectors for the same text, which breaks caching and deduplication. The ONNX
backend therefore runs one text at a time, and a test checks that the result does not depend
on the batch. This costs about 10% throughput at 32 texts.

### Thread counts

ONNX Runtime, PyTorch and the tokenizer size their thread pools from the host's cores, not
from the container's CPU limit, and get throttled when the limit is lower. `EMBED_NUM_THREADS`
sets all three.

### Logging

Logs are JSON lines on stdout, with one access-log line per request: request id, latency,
number of texts, tokens and truncated texts. The input text is never logged, because
embedding inputs are often user data. Validation errors do not include it, and unexpected
errors are logged with the exception type and stack trace but without the exception message,
which could contain it. Unexpected errors are caught by the app's own middleware, so they do
not reach uvicorn, which would log the message. Tests check the log output, including under a
real uvicorn server.

Every response has an `X-Request-ID` header. A caller's own id is reused if it looks safe;
otherwise the server generates one. The same id appears in every log line for the request
and in error bodies.

### Pinned model

The model revision is pinned in one place (`scripts/download_model.py`), and only the needed
files are downloaded; the full repository is about 9.5 GB. The scripts that build a model
directory write the model id and revision to `source.json`, and `/v1/info` reports those, so
it shows what was actually loaded.

### Tests without the model

The API talks to the model through a small `Embedder` protocol. The fast tests use a
deterministic fake and cover the HTTP behaviour in under a second. The slow tests use the
real models and compare orderings rather than absolute scores, because e5 similarities
mostly fall between 0.7 and 1.0.

## Performance

`scripts/benchmark.py` (`just bench`) times each backend through the API's own embedder
classes, with 4 threads and texts of about 30 tokens, and reports medians of 10 runs on an
AMD Ryzen 9 5950X. It also compares each backend's embeddings with PyTorch fp32 on 12
Danish and English query/passage pairs.

| Backend | Weights | 1 text | 8 texts | 32 texts | Texts/s | Cosine vs fp32 (mean / min) | Same top-1 |
|---|---|---|---|---|---|---|---|
| PyTorch fp32 | 2,240 MB | 115 ms | 485 ms | 1,829 ms | 17 | 1.0000 / 1.0000 | 100% |
| ONNX Runtime fp32 | 2,236 MB | 58 ms | 458 ms | 1,816 ms | 18 | 1.0000 / 1.0000 | 100% |
| ONNX Runtime int8 (served) | 562 MB | 33 ms | 262 ms | 1,037 ms | 31 | 0.9951 / 0.9929 | 100% |

For a single text, switching to ONNX Runtime halves the latency and int8 removes another 40%.
For 32 texts, ONNX fp32 is no faster than PyTorch because it also runs one text at a time, so
the 1.8x throughput at that size comes from int8. Twelve pairs are only a sanity check; the
next section measures quality properly.

### Retrieval quality

`scripts/eval_retrieval.py` (`just eval`) runs two Danish retrieval tasks from
[MTEB](https://github.com/embeddings-benchmark/mteb) with both backends. It embeds the corpus
and the queries with the e5 prefixes, ranks the corpus by cosine similarity, and computes
nDCG@10 (MTEB's main retrieval metric) and Recall@10.

| Task (queries / corpus) | Backend | nDCG@10 | nDCG@10, MTEB convention | Recall@10 | Same top-1 as fp32 |
|---|---|---|---|---|---|
| DanFEVER (3,102 / 2,524) | PyTorch fp32 | 0.8396 | 0.4087 | 0.9869 | |
| | ONNX int8 (served) | 0.8276 (-1.4%) | 0.4028 | 0.9815 | 91.5% |
| TwitterHjerne (77 / 262) | PyTorch fp32 | 0.7539 | 0.7539 | 0.8071 | |
| | ONNX int8 (served) | 0.7297 (-3.2%) | 0.7297 | 0.7693 | 83.1% |

int8 loses 1.4% nDCG@10 on DanFEVER and 3.2% on TwitterHjerne. DanFEVER is the better
estimate; TwitterHjerne has only 77 queries. If an application needs that last bit of
quality, it can run the fp32 backend with `EMBED_BACKEND=torch`.

The two nDCG columns average over different queries. The first uses only queries that have a
relevant document. MTEB averages over all queries in the relevance judgements, and 3,271
DanFEVER queries have no relevant document in the corpus, so they score 0 and halve the
result. With MTEB's convention, the fp32 scores are identical to those from the official
`mteb` package, which confirms the evaluation code.

## Deployment

The image runs on a two-node k3s cluster at home. CI pushes the image to GHCR,
argocd-image-updater notices the new digest, and ArgoCD rolls it out. The pod has a 4-CPU
limit, a read-only root filesystem, a readiness probe on `/health/ready` and a liveness probe
on `/health/live`. On that node (a Ryzen 5 6600U), measured through the ingress, the model is
ready 1.7 s after start, one text takes 68 ms, 32 texts take 1.4 s, a full 8192-token request
takes 9.9 s, and memory peaks at 1.45 GB. The Kubernetes manifests are kept in a separate
repository.

## Project layout

```
src/embed_api/
  main.py          app factory, startup, routes
  embedder.py      Embedder protocol, ONNX and PyTorch backends, EmbeddingService
  schemas.py       request and response models (validation and OpenAPI docs)
  middleware.py    JSON logging, request id and access log, body size limit
  errors.py        error responses and exception handlers
  config.py        settings from EMBED_* environment variables
scripts/
  export_onnx.py      builds the int8 model from the pinned revision
  download_model.py   downloads the fp32 model; pins the model revision
  benchmark.py        speed of the backends
  eval_retrieval.py   retrieval quality of the backends on Danish MTEB tasks
  smoke_test.sh       starts the built image and checks it returns an embedding (CI)
examples/
  demo.py             similarity across languages
  limits.py           truncation and the token budget
tests/                fast tests (fake model) and slow tests (real models)
```

## Limitations

[IMPROVEMENTS.md](IMPROVEMENTS.md) lists what this version leaves out. The main gaps are
authentication and rate limiting, metrics, and the 1-3% quality loss from int8. For a
high-traffic service, a dedicated inference server such as Hugging Face Text Embeddings
Inference is worth considering instead of a custom API.
