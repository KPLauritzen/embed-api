# embed-api

A production-minded HTTP API for the
[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)
embedding model: 1024-dimensional, L2-normalised text embeddings for 100 languages,
on CPU. By default it serves an int8-quantised ONNX version of the model: a quarter of the
size, 3.5× faster than PyTorch for a single text and 1.8× the throughput for 32, for a
1–3% drop in retrieval quality (nDCG@10 on two Danish MTEB tasks).

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
downloads 2.2 GB and needs ~8.5 GB of free RAM for about a minute; serving peaks at ~1.5 GB.

```sh
uv sync                                          # the API (ONNX Runtime, no torch) + test tools
uv run scripts/export_onnx.py                    # pinned revision -> int8 ./models/e5-int8
uv run uvicorn embed_api.main:app --port 8000    # ready in ~2 s
# then open http://localhost:8000/docs, or in another shell:
uv run python examples/demo.py http://localhost:8000     # DA/EN/DE similarity
uv run python examples/limits.py http://localhost:8000   # truncation, token budget
```

The scripts carry their own dependencies (PEP 723), so the export tooling never enters
the API's environment. Downloads print a "sending unauthenticated requests to the HF Hub"
warning; it is harmless (set `HF_TOKEN` to silence it).

**The PyTorch backend** serves the original fp32 model, the reference the int8 model is
tested against. It needs the `torch` extra:

```sh
uv run scripts/download_model.py                 # pinned fp32 revision -> ./models/e5 (2.2 GB)
EMBED_BACKEND=torch uv run --extra torch uvicorn embed_api.main:app --port 8000
```

### Docker

The image builds the int8 model from the pinned revision in a separate build stage, so the
runtime image has no torch, needs no network access, and is 0.6 GB compressed (amd64). The
build itself downloads 2.2 GB and needs ~8.5 GB of RAM: give Docker Desktop's VM enough.

```sh
docker build -t embed-api .
docker run --rm -p 8000:8000 embed-api
```

CI builds the image, starts it with a read-only root filesystem, checks it embeds, and only
then pushes that same image to GHCR.

### Development

Common tasks are in the [justfile](justfile) (`just` lists them):

```sh
just setup        # uv sync + install the pre-commit hooks
just check        # ruff lint + format check, ty type check, fast tests: what CI runs
just test-slow    # real models in ./models: both backends, int8 vs fp32, batch invariance
just serve        # also: model, model-torch, serve-torch, demo, bench, eval, docker-build, slides
```

The fast suite uses a fake model and runs in under a second. The slow suite needs both
models (`just model model-torch`) and runs locally, not in CI; CI covers the built image
with a smoke test instead. Pre-commit runs ruff, ty, a `uv.lock` consistency check and a
guard against committing large files (model weights) on every commit. Type checking uses
[ty](https://github.com/astral-sh/ty), pinned exactly because it is still pre-1.0; the
standalone scripts in `scripts/` carry their own dependencies and are not type-checked.

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

**Errors.** Every error the app produces, validation and routing included, uses one
envelope and carries the request id:

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
| 404 / 405 | `not_found` / `method_not_allowed` | Unknown path, or wrong method |
| 413 | `request_too_large` | Body over `EMBED_MAX_BODY_BYTES` |
| 422 | `validation_error` | Missing/unknown `input_type`, empty or blank text, >64 inputs, >8000 chars, extra fields |
| 422 | `token_budget_exceeded` | Request needs more than `EMBED_MAX_TOTAL_TOKENS` |
| 500 | `internal_error` | Anything unexpected; details are only in the server log |
| 503 | `model_not_ready` / `model_load_failed` | Still loading, or loading failed |
| 503 | `overloaded` | No inference capacity within `EMBED_QUEUE_TIMEOUT_SECONDS`; has `Retry-After` |

Malformed HTTP (a non-numeric `Content-Length`, say) is rejected by uvicorn itself, in plain
text, before the app sees it.

## Configuration

All settings are environment variables with the `EMBED_` prefix.

| Variable | Default | |
|---|---|---|
| `EMBED_BACKEND` | `onnx` | `onnx` (int8, default) or `torch` (fp32 reference) |
| `EMBED_MODEL_PATH` | `models/e5-int8` (onnx), `models/e5` (torch) | Model directory |
| `EMBED_NUM_THREADS` | runtime default | Inference and tokenizer threads; set to the container's CPU limit |
| `EMBED_QUEUE_TIMEOUT_SECONDS` | `30` | Longest wait for inference before `503 overloaded` |
| `EMBED_MAX_TOTAL_TOKENS` | `8192` | Token budget per request |
| `EMBED_MAX_BODY_BYTES` | `1000000` | Largest accepted body |
| `EMBED_LOG_LEVEL` | `INFO` | |

## Design decisions

**Inference stays off the event loop.** Encoding is CPU-bound and blocking. It runs in a
worker thread, so `/health/*` keeps answering while a large request is being embedded. A
semaphore lets one request run inference at a time: on CPU, one request
using every core beats several competing for them.

**Bounded queue, and abandoned requests are dropped.** A request waits at most
`EMBED_QUEUE_TIMEOUT_SECONDS` for that semaphore, then gets `503 overloaded` with a
`Retry-After` of the same length. uvicorn does not cancel a handler when
its client disconnects, so a request whose client has gone by the time it reaches the front
of the queue is dropped (logged as 499) instead of computed; otherwise timeouts and retries
pile up in front of live requests. Measured: after five clients abandon full-budget requests,
a new one-text request now waits 7.9 s (the one request already computing) instead of 27 s.
The access log records each request's `queue_ms`.

**Bounded work per request.** Pydantic limits (64 inputs, 8000 characters each) only apply
after the body has been read, so a middleware counts bytes as they arrive and stops at
`EMBED_MAX_BODY_BYTES` with 413, whether or not the client sent a Content-Length. The limit
that actually bounds CPU time and memory is the **token budget**: a full 8192-token request
takes ~8 s on 4 cores of the benchmark machine (9.9 s on the deployment node), and 64 × 512
tokens would take four times that.

**Background model loading.** The model loads in a background task after the server starts.
Liveness answers immediately and readiness returns 503 until the model is warm, so an
orchestrator can tell "starting" from "broken". If loading fails, liveness fails too, since
only a restart can fix that.

**int8 ONNX by default, without torch.** Quantisation happens once, at build time, from the
fp32 ONNX file the model repository ships (ONNX Runtime's dynamic quantiser, AVX2 settings;
the repository's own int8 file targets AVX-512 VNNI). At runtime the API needs only ONNX
Runtime and a tokenizer: tokenise, truncate to 512, mean-pool, normalise, the same
computation sentence-transformers does. Slow tests check the token counts match what
sentence-transformers feeds the model and that int8 embeddings stay above cosine 0.98 to
fp32; on Danish retrieval benchmarks int8 costs 1–3% nDCG@10 (see
[Performance](#retrieval-quality-what-int8-costs)). The PyTorch backend stays as the fp32
reference and for GPUs.

**One text per forward pass, for reproducible embeddings.** Dynamic quantisation picks its
activation scale from the whole input tensor. Batched, a text's int8 embedding depended on
the other texts in the request (measured: cosine 0.994 to itself embedded alone), which breaks
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
Unexpected exceptions are answered and logged by the app's own middleware and never reach
uvicorn, which would log the message. Tests check the real log output, including under a
real uvicorn server. Every response carries an `X-Request-ID`
header (a caller-supplied id is reused if it is safe), and the same id is on every log line
for that request and in every error body.

**Reproducible model.** The revision is pinned in one place, and only the files needed are
downloaded; the model repository also carries a `.bin` duplicate, ONNX and OpenVINO exports,
~9.5 GB in all. Both backends load a local model directory, and the script that built it
records the model id and revision in `source.json`, which `/v1/info` reports: what is
actually loaded, not what a setting claims.

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
pairs make the fidelity numbers a sanity check, not an evaluation; for that, see below. Rerun
with `just bench` (it downloads the fp32 ONNX file once).

### Retrieval quality: what int8 costs

`scripts/eval_retrieval.py` (`just eval`) runs two Danish retrieval tasks from
[MTEB](https://github.com/embeddings-benchmark/mteb) through both backends: it embeds the
corpus and queries with the e5 prefixes, ranks by cosine similarity, and scores nDCG@10
(MTEB's main retrieval metric) and Recall@10.

| Task (queries / corpus) | Backend | nDCG@10 | nDCG@10, MTEB convention | Recall@10 | Same top-1 as fp32 |
|---|---|---|---|---|---|
| DanFEVER (3,102 / 2,524) | PyTorch fp32 | 0.8396 | 0.4087 | 0.9869 | |
| | **ONNX int8 (served)** | 0.8276 (−1.4%) | 0.4028 | 0.9815 | 91.5% |
| TwitterHjerne (77 / 262) | PyTorch fp32 | 0.7539 | 0.7539 | 0.8071 | |
| | **ONNX int8 (served)** | 0.7297 (−3.2%) | 0.7297 | 0.7693 | 83.1% |

int8 costs 1–3% relative nDCG@10. DanFEVER is the larger, more reliable measurement;
TwitterHjerne's 77 queries make its gap noisy. That is the price of 3.5× lower latency
and a quarter of the size; a deployment that needs the last few percent can run the fp32
backend with one setting (`EMBED_BACKEND=torch`).

The two nDCG columns differ only in which queries are averaged. The first uses the queries
that have a relevant document. MTEB averages over every judged query, and 3,271 DanFEVER
queries are judged with no relevant document (claims whose evidence is not in the corpus),
so they score 0 and halve the number. The fp32 MTEB-convention scores match the official
`mteb` package exactly, which checks the evaluation code.

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
  middleware.py JSON logging, request id + access log, body-size limit
  errors.py     error envelope and exception handlers
  config.py     settings from EMBED_* environment variables
scripts/export_onnx.py      build the int8 ONNX model from the pinned revision
scripts/download_model.py   fetch the pinned fp32 model (torch backend); the one place the revision is pinned
scripts/benchmark.py        speed and fidelity of the backends
scripts/eval_retrieval.py   retrieval quality (nDCG@10) of both backends on Danish MTEB tasks
scripts/smoke_test.sh       run the built image and check it embeds (CI)
examples/demo.py            cross-lingual similarity demo
examples/limits.py          truncation flag and token budget demo
tests/                      fast (fake model) and slow (real model) suites
```

## Limitations and next steps

The first version deliberately leaves some things out; [IMPROVEMENTS.md](IMPROVEMENTS.md)
lists them. The main ones are authentication and rate limiting, Prometheus metrics, and
closing int8's 1–3% quality gap (calibrated static quantisation). For a high-traffic
deployment, a purpose-built server such as Hugging Face Text Embeddings Inference would be a
strong alternative to a hand-written one.
