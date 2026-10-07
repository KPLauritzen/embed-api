import hashlib
import threading
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from embeda_api.config import Settings
from embeda_api.main import create_app


class FakeEmbedder:
    """Deterministic stand-in for the model: one token per word plus two special tokens."""

    model_name = "fake/e5"
    dimension = 8
    max_tokens = 16

    def __init__(self, settings: Settings | None = None) -> None:
        self.calls: list[list[str]] = []

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(text.split()) + 2 for text in texts]

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
            vector = np.random.default_rng(seed).normal(size=self.dimension)
            rows.append(vector / np.linalg.norm(vector))
        return np.array(rows, dtype=np.float32)


@pytest.fixture
def settings() -> Settings:
    return Settings(max_total_tokens=512, max_body_bytes=20_000)


@pytest.fixture
def fake() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def client(settings: Settings, fake: FakeEmbedder) -> Iterator[TestClient]:
    app = create_app(settings, embedder_factory=lambda _: fake)
    with TestClient(app, raise_server_exceptions=False) as client:
        _wait_until_ready(client)
        yield client


def _wait_until_ready(client: TestClient) -> None:
    for _ in range(200):
        if client.get("/health/ready").status_code == 200:
            return
        threading.Event().wait(0.01)
    raise AssertionError("app never became ready")
