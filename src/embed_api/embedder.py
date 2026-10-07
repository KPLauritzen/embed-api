"""Model access, behind a small protocol so tests can use a fake."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

import anyio.to_thread
import numpy as np

from embed_api.errors import APIError

if TYPE_CHECKING:
    from embed_api.config import Settings


class InputType(StrEnum):
    """e5 was trained with these prefixes; leaving them out degrades quality."""

    QUERY = "query"
    PASSAGE = "passage"


class Embedder(Protocol):
    model_name: str
    dimension: int
    max_tokens: int

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Untruncated token count per text, special tokens included."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalised embeddings, one row per text. Over-long texts are truncated."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, settings: Settings) -> None:
        # Imported here so the API (and its tests) start without loading torch.
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as hf_logging

        # The tokenizer warns on every over-long input; truncation is reported
        # per input in the response instead.
        hf_logging.set_verbosity_error()
        if settings.num_threads:
            torch.set_num_threads(settings.num_threads)

        if settings.model_path:
            self._model = SentenceTransformer(settings.model_path, device=settings.device)
        else:
            self._model = SentenceTransformer(
                settings.model_id, revision=settings.model_revision, device=settings.device
            )
        self._batch_size = settings.encode_batch_size
        self.model_name = settings.model_id
        self.dimension = self._model.get_embedding_dimension()
        self.max_tokens = self._model.max_seq_length

    def count_tokens(self, texts: list[str]) -> list[int]:
        encoded = self._model.tokenizer(
            texts, add_special_tokens=True, truncation=False, return_attention_mask=False
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


@dataclass(frozen=True)
class EmbedResult:
    vectors: np.ndarray
    token_counts: list[int]
    truncated: list[bool]

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)


class EmbeddingService:
    """Adds the e5 prefix, enforces the token budget and serialises inference."""

    def __init__(self, embedder: Embedder, settings: Settings) -> None:
        self.embedder = embedder
        self._max_total_tokens = settings.max_total_tokens
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_batches)

    async def embed(self, texts: list[str], input_type: InputType) -> EmbedResult:
        prefixed = [f"{input_type.value}: {text}" for text in texts]

        # Tokenising up to 64 x 8k chars is not free, so it runs off the event loop too.
        counts = await anyio.to_thread.run_sync(self.embedder.count_tokens, prefixed)
        limit = self.embedder.max_tokens
        truncated = [n > limit for n in counts]
        used = [min(n, limit) for n in counts]
        if sum(used) > self._max_total_tokens:
            raise APIError(
                422,
                "token_budget_exceeded",
                f"Request needs {sum(used)} tokens; the limit is {self._max_total_tokens}. "
                "Split it into smaller requests.",
            )

        # The worker thread is not cancellable (anyio's default), so a client
        # disconnecting mid-inference still holds the semaphore until the batch
        # finishes. That keeps max_concurrent_batches an actual bound.
        async with self._semaphore:
            vectors = await anyio.to_thread.run_sync(self.embedder.embed, prefixed)
        return EmbedResult(vectors=vectors, token_counts=used, truncated=truncated)
