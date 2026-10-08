"""Behaviour that only shows up under a real server.

TestClient calls the app directly, so anything uvicorn itself logs (such as an
exception that escapes the app) is invisible to it.
"""

import io
import logging
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import numpy as np
import pytest
import uvicorn

from embed_api.config import Settings
from embed_api.main import create_app
from tests.conftest import FakeEmbedder


class ExplodingEmbedder(FakeEmbedder):
    def embed(self, texts: list[str]) -> np.ndarray:
        if any("very-private-text" in text for text in texts):
            raise ValueError(f"cannot embed {texts!r}")
        return super().embed(texts)


@pytest.fixture
def server() -> Iterator[tuple[str, io.StringIO]]:
    app = create_app(Settings(), embedder_factory=lambda _: ExplodingEmbedder())
    root = logging.getLogger()
    handler = logging.StreamHandler(logs := io.StringIO())
    handler.setFormatter(root.handlers[0].formatter)
    root.addHandler(handler)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    # log_config=None keeps the app's JSON logging, as `uvicorn embed_api.main:app` does.
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    for _ in range(200):
        try:
            if httpx.get(f"{url}/health/ready").status_code == 200:
                break
        except httpx.TransportError:
            pass
        time.sleep(0.01)
    yield url, logs
    server.should_exit = True
    thread.join(5)
    root.removeHandler(handler)


def test_unexpected_error_does_not_leak_input_into_server_logs(
    server: tuple[str, io.StringIO],
) -> None:
    url, logs = server

    r = httpx.post(f"{url}/v1/embed", json={"input": "very-private-text", "input_type": "query"})

    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    logged = logs.getvalue()
    assert '"unhandled_error"' in logged  # logged once, by the app...
    assert "very-private-text" not in logged  # ...without the message carrying the input
    assert "Exception in ASGI application" not in logged  # and never re-raised to uvicorn
