import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import anyio.to_thread
import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.responses import RedirectResponse

from embed_api import errors
from embed_api.config import Settings
from embed_api.embedder import Embedder, EmbeddingService, load_embedder
from embed_api.errors import APIError
from embed_api.middleware import (
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    configure_logging,
)
from embed_api.schemas import (
    MAX_CHARS,
    MAX_INPUTS,
    Embedding,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    InfoResponse,
    Limits,
    Usage,
)

log = structlog.get_logger(__name__)

EmbedderFactory = Callable[[Settings], Embedder]

DESCRIPTION = """
Text embeddings from [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large):
1024 dimensions, L2-normalised, for about 100 languages.

* Send the text without a prefix and set `input_type`; the server adds e5's `query: ` or
  `passage: ` prefix.
* Texts longer than 512 tokens are truncated and marked `truncated: true`.
* Every response has an `X-Request-ID` header, and error bodies contain the same id.
"""


class ModelState:
    """Holds the embedding service once the background load has finished."""

    def __init__(self) -> None:
        self.service: EmbeddingService | None = None
        self.failed = False


def _load_failed() -> APIError:
    return APIError(503, "model_load_failed", "The model failed to load; see server logs.")


def _load(factory: EmbedderFactory, settings: Settings) -> Embedder:
    embedder = factory(settings)
    embedder.embed(["query: warm-up"])  # the first call is slow; keep it out of real requests
    return embedder


async def _load_in_background(
    state: ModelState, factory: EmbedderFactory, settings: Settings
) -> None:
    log.info("model_loading", backend=settings.backend.value, path=str(settings.model_dir))
    try:
        embedder = await anyio.to_thread.run_sync(_load, factory, settings)
    except Exception:
        # Safe to log in full: loading only reads local files, never request data.
        state.failed = True
        log.exception("model_load_failed")
        return
    state.service = EmbeddingService(embedder, settings)
    log.info(
        "model_ready",
        model=embedder.model_name,
        revision=embedder.revision,
        backend=embedder.backend,
        dimension=embedder.dimension,
        max_tokens=embedder.max_tokens,
    )


def create_app(
    settings: Settings | None = None,
    embedder_factory: EmbedderFactory = load_embedder,
) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)
    state = ModelState()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # Load in the background so the server answers health checks while loading.
        task = asyncio.create_task(_load_in_background(state, embedder_factory, settings))
        yield
        task.cancel()

    app = FastAPI(
        title="embed-api",
        version="0.1.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    errors.register(app)
    # The middleware added last runs first, so the request id exists before the size check.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(RequestContextMiddleware)

    def get_service() -> EmbeddingService:
        if state.service is not None:
            return state.service
        if state.failed:
            raise _load_failed()
        raise APIError(503, "model_not_ready", "The model is still loading. Retry shortly.")

    Service = Annotated[EmbeddingService, Depends(get_service)]

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.post(
        "/v1/embed",
        tags=["embeddings"],
        summary="Embed one or more texts",
        responses=errors.EMBED_ERRORS,
    )
    async def embed(body: EmbedRequest, request: Request, service: Service) -> EmbedResponse:
        result = await service.embed(body.input, body.input_type, request.is_disconnected)
        request.state.log_fields.update(
            queue_ms=result.queue_ms,
            n_inputs=len(body.input),
            input_type=body.input_type.value,
            total_tokens=result.total_tokens,
            n_truncated=sum(result.truncated),
        )
        return EmbedResponse(
            model=service.embedder.model_name,
            input_type=body.input_type,
            dimension=service.embedder.dimension,
            embeddings=[
                Embedding(index=i, embedding=vector.tolist(), tokens=tokens, truncated=truncated)
                for i, (vector, tokens, truncated) in enumerate(
                    zip(result.vectors, result.token_counts, result.truncated, strict=True)
                )
            ],
            usage=Usage(total_tokens=result.total_tokens),
        )

    @app.get("/v1/info", tags=["meta"], summary="Model and limits", responses=errors.MODEL_ERRORS)
    def info(service: Service) -> InfoResponse:
        return InfoResponse(
            model=service.embedder.model_name,
            revision=service.embedder.revision,
            backend=service.embedder.backend,
            dimension=service.embedder.dimension,
            limits=Limits(
                max_inputs=MAX_INPUTS,
                max_chars_per_input=MAX_CHARS,
                max_tokens_per_input=service.embedder.max_tokens,
                max_total_tokens=settings.max_total_tokens,
                max_body_bytes=settings.max_body_bytes,
            ),
        )

    @app.get(
        "/health/live",
        tags=["health"],
        summary="The process is up and the model has not failed to load",
        responses=errors.LIVE_ERRORS,
    )
    def live() -> HealthResponse:
        # A failed load does not recover by itself, so ask for a restart.
        if state.failed:
            raise _load_failed()
        return HealthResponse(status="ok")

    @app.get(
        "/health/ready",
        tags=["health"],
        summary="The model is loaded",
        responses=errors.MODEL_ERRORS,
    )
    def ready(_: Service) -> HealthResponse:
        return HealthResponse(status="ready")

    return app


app = create_app()
