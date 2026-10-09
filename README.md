# Embed API

This is an API serving the [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) embedding model.

By default it serves a quantised version of the model, but it is also possible to serve the original model.

The quantised model is a quarter of the size of the original model, serves 3.5x faster for a single text and loses only 1-3% performance on nDCG@10 on Danish retrieval benchmarks.

## Quickstart

You need [uv](https://docs.astral.sh/uv/) installed to run this locally. Alternatively you can run the API in Docker.

To build the quantised model and serve the API on port 8000:

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

Or post a request with `curl` (with some `jq` to filter the output):

```console
$ curl -s localhost:8000/v1/embed -H 'content-type: application/json' \
    -d '{"input": ["København er Danmarks hovedstad.", "Copenhagen is the capital of Denmark."],
         "input_type": "passage"}' | jq -c '.embeddings[].embedding |= (.[:2] | map(.*1e4 | round/1e4))'
{"model":"intfloat/multilingual-e5-large","input_type":"passage","dimension":1024,
 "embeddings":[{"index":0,"embedding":[0.0435,0.0155],"tokens":9,"truncated":false},
               {"index":1,"embedding":[0.0371,0.0077],"tokens":11,"truncated":false}],
 "usage":{"total_tokens":20}}
```

To serve the original fp32 model instead, download it and run the PyTorch backend:

```sh
uv run scripts/download_model.py                 # downloads ./models/e5 (2.2 GB)
EMBED_BACKEND=torch uv run --extra torch uvicorn embed_api.main:app --port 8000
```

### Docker

Build and run the Docker image:

```sh
docker build -t embed-api .
docker run --rm -p 8000:8000 embed-api
```

CI builds the image and pushes it to [GitHub Container Registry](https://github.com/KPLauritzen/embed-api/pkgs/container/embed-api) on every push to `main`:

```sh
docker run --rm -p 8000:8000 ghcr.io/kplauritzen/embed-api:main
```

### Development

Common tasks are in the [justfile](justfile). Run `just` to list them.

```sh
just setup        # uv sync and install the pre-commit hooks
just check        # ruff, ty and the fast tests (CI runs the same command)
just test-slow    # tests against the real models
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/embed` | Embed one text or a list of up to 64 |
| `GET /v1/info` | Model id, revision, backend, dimension and limits |
| `GET /health/live` | Liveness: 200 unless the model failed to load |
| `GET /health/ready` | Readiness: 503 until the model is loaded |
| `GET /docs` | Swagger UI |

[docs/design.md](docs/design.md) has the request details, error codes, configuration and the
design decisions behind the service.

## Embedding model

The model returns 1024-dimensional embeddings and supports about 100 languages.
It was trained with `query: ` and `passage: ` prefixes. The API adds the prefix for you, based on the required `input_type` field.

The model reads at most 512 tokens. Longer texts are truncated, and the response marks them with `truncated: true`.

## Performance

On an AMD Ryzen 9 5950X with 4 threads, the int8 model embeds one text in 33 ms (PyTorch fp32 takes 115 ms) and handles about 31 texts/s. On the deployment server, memory peaks at about 1.5 GB.

On Danish retrieval, nDCG@10 drops from 0.840 to 0.828 on [DanFEVER](https://huggingface.co/datasets/mteb/DanFeverRetrieval) and from 0.754 to 0.730 on [TwitterHjerne](https://huggingface.co/datasets/mteb/TwitterHjerneRetrieval).

[docs/performance.md](docs/performance.md) has the full speed benchmark and the retrieval-quality
evaluation.

## Deployment

The API runs on a home Kubernetes cluster and is reachable inside that network at https://embed.home.primdal.dev/.

When CI pushes a new image to GitHub Container Registry, `argocd-image-updater` picks up the new image and ArgoCD rolls it out.
The Kubernetes manifests are kept in a separate (private) repository.

## Limitations

[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) lists what this version leaves out. The main gaps are
authentication and rate limiting, metrics, and the 1-3% quality loss from int8. For a
high-traffic service, a dedicated inference server such as Hugging Face Text Embeddings
Inference is worth considering instead of a custom API.
