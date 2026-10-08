import hashlib
import io
import logging
import time
from collections.abc import Callable, Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from embed_api.config import Settings
from embed_api.main import create_app
from embed_api.middleware import json_formatter


class FakeEmbedder:
    """Deterministic stand-in for the model: one token per word plus two special tokens."""

    model_name = "fake/e5"
    revision = "fake-revision"
    backend = "fake"
    dimension = 8
    max_tokens = 16

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def count_tokens(self, texts: list[str]) -> list[tuple[int, bool]]:
        counts = [len(text.split()) + 2 for text in texts]
        return [(min(n, self.max_tokens), n > self.max_tokens) for n in counts]

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
            vector = np.random.default_rng(seed).normal(size=self.dimension)
            rows.append(vector / np.linalg.norm(vector))
        return np.array(rows, dtype=np.float32)


def wait_until(condition: Callable[[], bool], timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition never became true")
        time.sleep(0.01)


@pytest.fixture
def settings() -> Settings:
    return Settings(max_total_tokens=512, max_body_bytes=20_000)


@pytest.fixture
def fake() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def client(settings: Settings, fake: FakeEmbedder) -> Iterator[TestClient]:
    app = create_app(settings, embedder_factory=lambda _: fake)
    with TestClient(app) as client:
        wait_until(lambda: client.get("/health/ready").status_code == 200)
        yield client


@pytest.fixture
def logs() -> Iterator[io.StringIO]:
    """Everything logged during the test, rendered as the app renders it."""
    handler = logging.StreamHandler(buffer := io.StringIO())
    handler.setFormatter(json_formatter())
    logging.getLogger().addHandler(handler)
    yield buffer
    logging.getLogger().removeHandler(handler)
