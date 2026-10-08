from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException


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


class APIError(HTTPException):
    """An HTTP error with a stable, machine-readable `code` for the error envelope."""

    def __init__(
        self, status_code: int, code: str, message: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(status_code, detail=message, headers=headers)
        self.code = code


def error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None,
    details: list[dict[str, Any]] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    error = ErrorBody(code=code, message=message, request_id=request_id, details=details)
    return JSONResponse(
        ErrorResponse(error=error).model_dump(exclude_none=True),
        status_code=status_code,
        headers=headers,
    )


def _request_id(request: Request) -> str | None:
    return request.scope.get("state", {}).get("request_id")


# Codes for the errors Starlette raises itself (unknown path, wrong method).
_HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}


async def handle_http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    code = exc.code if isinstance(exc, APIError) else _HTTP_CODES.get(exc.status_code, "http_error")
    return error_response(
        exc.status_code, code, exc.detail, _request_id(request), headers=exc.headers
    )


async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    # FastAPI's default body echoes each offending value back (up to the 1 MB
    # body limit). Keep where and why; drop the client's text.
    assert isinstance(exc, RequestValidationError)
    details = [
        {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return error_response(
        422, "validation_error", "Request validation failed.", _request_id(request), details
    )


def register(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, handle_http_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
