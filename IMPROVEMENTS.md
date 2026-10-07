# Improvements backlog

Things deliberately left out of the first version, roughly in order of value. If time allows,
items move from here into scope.

## API behaviour
- **Queue timeout + `Retry-After`.** Today a request waits for the inference semaphore
  indefinitely. Return 503 with `Retry-After` (derived from observed batch latency) when the
  wait exceeds a budget.
- **Strict truncation mode.** `truncate: false` → 422 naming the over-long input, for callers who
  would rather fail than embed a prefix of their text.
- **OpenAI-compatible alias** (`POST /v1/embeddings`) so existing clients work unchanged.

## Operations
- **Authentication** (API keys or mTLS) and **rate limiting** per client.
- **Prometheus metrics** (`/metrics`): request latency, tokens/s, queue depth, truncation rate.
- **Multi-arch image** (arm64 for Apple Silicon / Graviton).

## Performance
- **Cross-request dynamic batching**: collect requests for a few ms and encode together. With
  the int8 backend this needs static (calibrated) quantisation first: dynamic quantisation
  makes a text's embedding depend on its batch, which is why texts run one at a time today.
- **Purpose-built server**: Hugging Face Text Embeddings Inference (TEI) or Triton.
- **GPU** support and autoscaling.
- **Embedding cache** keyed on (model revision, input_type, text hash).

## Developer experience
- Type checking (pyright/mypy) in CI.
- Task runner (justfile) for common commands.
- Pretty console logs for local development (JSON-only today).
- A larger fidelity evaluation of the int8 model (e.g. an MTEB retrieval task), beyond the
  12-pair sanity check.
