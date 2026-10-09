"""Queueing for inference capacity: timeout, Retry-After, abandoned requests."""

import threading

import anyio
import numpy as np
import pytest

from embed_api.config import Settings
from embed_api.embedder import EmbeddingService, InputType
from embed_api.errors import APIError
from tests.conftest import FakeEmbedder

pytestmark = pytest.mark.anyio


class BlockingEmbedder(FakeEmbedder):
    """Holds the inference slot until released, like a long batch."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def embed(self, texts: list[str]) -> np.ndarray:
        self.started.set()
        self.release.wait(5)
        return super().embed(texts)


async def connected() -> bool:
    return False


async def test_waiting_longer_than_the_queue_timeout_gives_503_with_retry_after() -> None:
    # Given a request that holds the inference slot
    embedder = BlockingEmbedder()
    service = EmbeddingService(embedder, Settings(queue_timeout_seconds=0.05))

    async with anyio.create_task_group() as tg:
        tg.start_soon(service.embed, ["first"], InputType.QUERY, connected)
        await anyio.to_thread.run_sync(embedder.started.wait, 5)

        # When a second request waits longer than the queue timeout
        with pytest.raises(APIError) as caught:
            await service.embed(["second"], InputType.QUERY, connected)
        embedder.release.set()

    # Then
    assert (caught.value.status_code, caught.value.code) == (503, "overloaded")
    assert caught.value.headers is not None
    assert int(caught.value.headers["Retry-After"]) >= 1


async def test_request_from_a_disconnected_client_is_not_computed() -> None:
    # Given a client that has already disconnected
    embedder = FakeEmbedder()
    service = EmbeddingService(embedder, Settings())

    async def gone() -> bool:
        return True

    # When its request reaches the inference slot
    with pytest.raises(APIError) as caught:
        await service.embed(["hej"], InputType.QUERY, gone)

    # Then it is dropped without running the model
    assert caught.value.status_code == 499
    assert embedder.calls == []
