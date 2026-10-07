import traceback
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None,
    details: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    error = ErrorBody(code=code, message=message, request_id=request_id, details=details)
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(
        ErrorResponse(error=error).model_dump(exclude_none=True),
        status_code=status_code,
        headers=headers,
    )


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, APIError)
    return error_response(exc.status_code, exc.code, exc.message, _request_id(request))


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


async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    # The stack goes to the log, but not the exception message: it can carry
    # request data (an embedding input in a ValueError, say). The client only
    # gets the request id to quote.
    log.error(
        "unhandled_error",
        error_type=type(exc).__qualname__,
        stack="".join(traceback.format_tb(exc.__traceback__)),
    )
    return error_response(500, "internal_error", "Internal server error.", _request_id(request))


def register(app: FastAPI) -> None:
    app.add_exception_handler(APIError, handle_api_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unexpected)
