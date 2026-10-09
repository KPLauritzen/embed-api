"""Behaviour that only shows up under a real server.

TestClient calls the app directly, so anything uvicorn itself logs (such as an
exception that escapes the app) is invisible to it.
"""

import io
import socket
import threading
from collections.abc import Iterator

import httpx
import numpy as np
import pytest
import uvicorn

from embed_api.config import Settings
from embed_api.main import create_app
from tests.conftest import FakeEmbedder, wait_until


class ExplodingEmbedder(FakeEmbedder):
    def embed(self, texts: list[str]) -> np.ndarray:
        if any("very-private-text" in text for text in texts):
            raise ValueError(f"cannot embed {texts!r}")  # the message carries the input
        return super().embed(texts)


def _ready(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health/ready").status_code == 200
    except httpx.TransportError:
        return False


@pytest.fixture
def server_url() -> Iterator[str]:
    app = create_app(Settings(), embedder_factory=lambda _: ExplodingEmbedder())
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    # log_config=None keeps the app's JSON logging, as `uvicorn embed_api.main:app` does.
    server = uvicorn.Server(uvicorn.Config(app, log_config=None))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    wait_until(lambda: _ready(url))
    yield url
    server.should_exit = True
    thread.join(5)


def test_unexpected_error_is_answered_and_logged_without_its_message(
    server_url: str, logs: io.StringIO
) -> None:
    # Given a server whose model raises an error that contains the input (ExplodingEmbedder)
    # When
    r = httpx.post(
        f"{server_url}/v1/embed", json={"input": "very-private-text", "input_type": "query"}
    )

    # Then
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "private" not in r.text
    logged = logs.getvalue()
    assert '"unhandled_error"' in logged
    assert "very-private-text" not in logged
    assert "Exception in ASGI application" not in logged  # uvicorn's own error log
