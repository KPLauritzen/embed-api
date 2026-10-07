"""JSON logging, request ids, access log and the request-size limit.

Middlewares are plain ASGI (not BaseHTTPMiddleware) so they share the request's
context: values bound with structlog.contextvars appear on every log line the
request produces, including uvicorn's own.
"""

import logging
import re
import sys
import time
import uuid
from typing import Any

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from embeda_api.errors import error_response

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

access_log = structlog.get_logger("embeda_api.access")


def configure_logging(level: str) -> None:
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Route uvicorn's loggers through the same JSON formatter. Its access log is
    # replaced by RequestContextMiddleware's, which carries the request id and
    # embedding stats.
    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True


class RequestContextMiddleware:
    """Assigns a request id, echoes it as X-Request-ID and writes one access-log line.

    A caller-supplied X-Request-ID is reused if it looks sane, so ids can be
    traced across services; anything else is replaced to keep logs clean.
    Endpoints add fields to the access line via `request.state.log_fields`.
    Request text is never logged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex[:16]
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["log_fields"] = {}
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        status = 500
        start = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [(k, v) for k, v in message.get("headers", []) if k != b"x-request-id"]
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            path = scope["path"]
            if not path.startswith("/health"):  # probes would drown everything else
                access_log.info(
                    "request",
                    method=scope["method"],
                    path=path,
                    status=status,
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                    **state["log_fields"],
                )


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Rejects oversized bodies with 413 before they are buffered and parsed.

    Neither uvicorn nor FastAPI caps body size, and pydantic's limits only apply
    after the whole body is in memory. Content-Length is checked up front; a
    chunked body is counted as it streams in.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and int(declared) > self.max_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if response_started:
                raise
            await self._reject(scope, receive, send)

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = error_response(
            413,
            "request_too_large",
            f"Request body exceeds {self.max_bytes} bytes.",
            scope.get("state", {}).get("request_id"),
        )
        await response(scope, receive, send)
