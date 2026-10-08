"""Queueing for inference capacity: timeout, Retry-After, abandoned requests."""

import threading

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


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_waiting_longer_than_the_queue_timeout_gives_503_with_retry_after() -> None:
    import anyio

    embedder = BlockingEmbedder()
    service = EmbeddingService(embedder, Settings(queue_timeout_seconds=0.05))

    async with anyio.create_task_group() as tg:
        tg.start_soon(service.embed, ["first"], InputType.QUERY)
        await anyio.to_thread.run_sync(embedder.started.wait, 5)

        with pytest.raises(APIError) as caught:
            await service.embed(["second"], InputType.QUERY)
        embedder.release.set()

    assert (caught.value.status_code, caught.value.code) == (503, "overloaded")
    assert int(caught.value.headers["Retry-After"]) >= 1


async def test_request_from_a_disconnected_client_is_not_computed() -> None:
    embedder = FakeEmbedder()
    service = EmbeddingService(embedder, Settings())

    async def gone() -> bool:
        return True

    with pytest.raises(APIError) as caught:
        await service.embed(["hej"], InputType.QUERY, is_disconnected=gone)

    assert caught.value.status_code == 499
    assert embedder.calls == []


async def test_queue_time_is_reported() -> None:
    service = EmbeddingService(FakeEmbedder(), Settings())

    result = await service.embed(["hej"], InputType.QUERY)

    assert result.queue_ms >= 0
