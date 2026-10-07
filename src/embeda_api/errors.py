import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


class ErrorBody(BaseModel):
    code: str = Field(examples=["model_not_ready"])
    message: str = Field(examples=["The model is still loading. Retry shortly."])
    request_id: str | None = Field(None, examples=["0b6f3c1e9a8d4f52"])


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
    status_code: int, code: str, message: str, request_id: str | None
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message, request_id=request_id))
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(body.model_dump(), status_code=status_code, headers=headers)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, APIError)
    return error_response(exc.status_code, exc.code, exc.message, _request_id(request))


async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    # Full traceback goes to the log; the client only gets the request id to quote.
    log.exception("unhandled_error")
    return error_response(500, "internal_error", "Internal server error.", _request_id(request))


def register(app: FastAPI) -> None:
    app.add_exception_handler(APIError, handle_api_error)
    app.add_exception_handler(Exception, handle_unexpected)
