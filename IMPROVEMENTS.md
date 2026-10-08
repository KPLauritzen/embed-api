# Improvements backlog

Things deliberately left out of the first version, roughly in order of value. If time allows,
items move from here into scope.

## API behaviour
- **Stop in-flight work for a departed client.** Queued requests from disconnected clients
  are dropped, but one already computing runs to the end (up to ~10 s for a full budget).
  The ONNX backend embeds one text at a time, so it could check for a disconnect between texts.
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
- Pretty console logs for local development (JSON-only today).
- Close the int8 quality gap (1–3% nDCG@10): static quantisation calibrated on real text, or
  keep the most sensitive layers in fp32. Evaluate on more MTEB tasks and languages.
