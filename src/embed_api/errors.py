import traceback
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException

log = structlog.get_logger(__name__)


class ErrorBody(BaseModel):
    code: str = Field(examples=["model_not_ready"])
    message: str = Field(examples=["The model is still loading. Retry shortly."])
    request_id: str | None = Field(None, examples=["0b6f3c1e9a8d4f52"])
    details: list[dict[str, Any]] | None = Field(
        None,
        description="For validation_error: one entry per problem, with `loc` "
        '(e.g. ["body", "input", 3]), `msg` and `type`.',
    )


class ErrorResponse(BaseModel):
    error: ErrorBody


class APIError(Exception):
    """An error with a stable, client-facing code."""

    def __init__(
        self, status_code: int, code: str, message: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers or {}


def error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None,
    details: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    error = ErrorBody(code=code, message=message, request_id=request_id, details=details)
    headers = dict(headers or {})
    if request_id:
        headers["X-Request-ID"] = request_id
    return JSONResponse(
        ErrorResponse(error=error).model_dump(exclude_none=True),
        status_code=status_code,
        headers=headers,
    )


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, APIError)
    return error_response(
        exc.status_code, exc.code, exc.message, _request_id(request), headers=exc.headers
    )


_HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}


async def handle_http_error(request: Request, exc: Exception) -> JSONResponse:
    # Routing errors (unknown path, wrong method) in the same envelope.
    assert isinstance(exc, HTTPException)
    return error_response(
        exc.status_code,
        _HTTP_CODES.get(exc.status_code, "http_error"),
        str(exc.detail),
        _request_id(request),
        headers=dict(exc.headers or {}),
    )


async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    # FastAPI's default body echoes each offending value back (`input`, up to
    # the 1 MB body limit) and has a different shape from every other error.
    # Keep where and why; drop the client's text.
    assert isinstance(exc, RequestValidationError)
    details = [
        {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return error_response(
        422, "validation_error", "Request validation failed.", _request_id(request), details
    )


def log_unexpected(exc: BaseException) -> None:
    """Log an unhandled exception's type and stack, but not its message.

    The message can carry request data (an embedding input in a ValueError,
    say). Called by RequestContextMiddleware, which also sends the 500: if the
    exception went on to Starlette's error middleware it would be re-raised to
    uvicorn, which logs it again with the message.
    """
    log.error(
        "unhandled_error",
        error_type=type(exc).__qualname__,
        stack="".join(traceback.format_tb(exc.__traceback__)),
    )


def register(app: FastAPI) -> None:
    app.add_exception_handler(APIError, handle_api_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(HTTPException, handle_http_error)
