# Design

## API details

`input_type` is required. The embedding model was trained with `query: ` and `passage: ` prefixes and gives
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
[Retrieval quality](performance.md#retrieval-quality) measures what int8 costs. The PyTorch backend
remains as the fp32 reference.

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
