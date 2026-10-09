# Improvements

Potential improvements that I left out:

## API

- Stop work for clients that have gone. A queued request is dropped if its client
  disconnects, but a request that is already running finishes, which can take about 10 s for
  a full token budget. Since the ONNX backend embeds one text at a time, it could check for a
  disconnect between texts.
- A strict mode (`truncate: false`) that rejects texts over 512 tokens with a 422 instead of
  truncating them.
- An OpenAI-compatible endpoint (`POST /v1/embeddings`), so existing client libraries work
  without changes.

## Operations

- Authentication (API keys or mTLS) and rate limiting per client.
- Prometheus metrics on `/metrics`: latency, tokens per second, queue length, truncation rate.
- An arm64 image, for Apple Silicon and Graviton.

## Model and performance

- Reduce the int8 quality loss (1-3% nDCG@10): calibrated static quantisation, or keeping the
  most sensitive layers in fp32. Measure on more MTEB tasks and languages.
- Batch requests from different clients together. With int8 this first needs static
  quantisation: with dynamic quantisation a text's embedding depends on the other texts in the
  batch, which is why texts currently run one at a time.
- A dedicated inference server such as Hugging Face Text Embeddings Inference or Triton.
- GPU support and autoscaling.
- A cache of embeddings, keyed on model revision, input type and a hash of the text.

## Development

- Human-readable console logs for local development. Logs are JSON only.
