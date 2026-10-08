"""JSON logging, request ids, the access log, and the request-size limit.

The middlewares are plain ASGI (not BaseHTTPMiddleware) so they run in the
request's own context: values bound with structlog.contextvars appear on
every log line the request produces.
"""

import logging
import re
import sys
import time
import traceback
import uuid
from typing import Any

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from embed_api.errors import APIError, error_response

_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")

log = structlog.get_logger("embed_api.access")


_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]


def json_formatter() -> logging.Formatter:
    """Renders structlog events and plain stdlib records (uvicorn's) as JSON lines."""
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )


def configure_logging(level: str) -> None:
    structlog.configure(
        processors=[*_SHARED_PROCESSORS, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(json_formatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # uvicorn's logs go through the same JSON handler; its access log is
    # replaced by ours, which has the request id and embedding stats.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    logging.getLogger("uvicorn.access").disabled = True


class RequestContextMiddleware:
    """Request id, access log, and the last stop for unexpected exceptions.

    - A caller's X-Request-ID is reused if it looks sane (so ids can be traced
      across services), otherwise one is generated. Either way it is echoed on
      the response and bound to every log line of the request.
    - One access-log line per request. Endpoints add fields to it through
      `request.state.log_fields`. Request text is never logged.
    - An unexpected exception is logged with its type and stack but not its
      message (which can carry request data) and answered with a 500. It is
      not re-raised: Starlette would pass it on to uvicorn, which logs the
      message.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if _REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex[:16]
        scope.setdefault("state", {}).update(request_id=request_id, log_fields={})
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        status: int | None = None
        start = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", request_id.encode()),
                ]
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception as exc:
            log.error(
                "unhandled_error",
                error_type=type(exc).__qualname__,
                stack="".join(traceback.format_tb(exc.__traceback__)),
            )
            if status is None:
                response = error_response(
                    500, "internal_error", "Internal server error.", request_id
                )
                await response(scope, receive, send_with_id)
            else:
                status = 500  # the response had already started; it is cut off
        finally:
            if not scope["path"].startswith("/health"):  # probes would drown everything
                log.info(
                    "request",
                    method=scope["method"],
                    path=scope["path"],
                    status=status or 499,  # no response: the client left (nginx's 499)
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                    **scope["state"]["log_fields"],
                )


class BodySizeLimitMiddleware:
    """Rejects a request body over `max_bytes` with 413 while it is being read.

    Neither uvicorn nor FastAPI caps body size, and pydantic's limits only
    apply once the whole body is in memory. Counting bytes as they arrive
    covers both a Content-Length body and a chunked one, and stops reading at
    the limit.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                # Raised inside FastAPI's body parsing, which passes HTTP errors on
                # to the exception handlers, so it gets the usual error envelope.
                raise APIError(
                    413, "request_too_large", f"Request body exceeds {self.max_bytes} bytes."
                )
            return message

        await self.app(scope, limited_receive, send)
