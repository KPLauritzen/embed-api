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

from embed_api.errors import error_response

_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")

access_log = structlog.get_logger("embed_api.access")


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
    # One INFO line per outbound HTTP call (Hub lookups by the torch backend).
    logging.getLogger("httpx").setLevel(logging.WARNING)


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
        request_id = supplied if _REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex[:16]
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["log_fields"] = {}
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        status: int | None = None
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
        except Exception:
            status = 500  # sent by Starlette's outermost error middleware
            raise
        finally:
            if status is None:
                status = 499  # client went away before a response (nginx's convention)
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


class BodySizeLimitMiddleware:
    """Rejects oversized bodies with 413 before they are parsed.

    Neither uvicorn nor FastAPI caps body size, and pydantic's limits only apply
    after the whole body is in memory. A declared Content-Length is checked up
    front (the server enforces that the body matches it). A chunked body has no
    length, so it is read here, up to the limit, and replayed to the app.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None:
            # uvicorn rejects a malformed header itself; other servers may not.
            if not declared.isdigit():
                await self._reject(scope, receive, send, 400, "Invalid Content-Length header.")
            elif int(declared) > self.max_bytes:
                await self._reject(scope, receive, send)
            else:
                await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        status: int = 413,
        message: str | None = None,
    ) -> None:
        response = error_response(
            status,
            "request_too_large" if status == 413 else "bad_request",
            message or f"Request body exceeds {self.max_bytes} bytes.",
            scope.get("state", {}).get("request_id"),
        )
        await response(scope, receive, send)
