"""Model access, behind a small protocol so tests can use a fake."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import anyio.to_thread
import numpy as np

from embed_api.config import Backend, Settings
from embed_api.errors import APIError

if TYPE_CHECKING:
    from tokenizers import Tokenizer


class InputType(StrEnum):
    """e5 was trained with these prefixes; without them its embeddings are worse."""

    QUERY = "query"
    PASSAGE = "passage"


class Embedder(Protocol):
    model_name: str
    revision: str
    backend: str
    dimension: int
    max_tokens: int

    def count_tokens(self, texts: list[str]) -> list[tuple[int, bool]]:
        """Per text: tokens the model sees (special tokens included), and if it was truncated."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalised embeddings, one row per text. Over-long texts are truncated."""
        ...


def _load_tokenizer(model_dir: Path) -> tuple[Tokenizer, int]:
    """The model's fast tokenizer, truncating at the model's token limit, and that limit."""
    from tokenizers import Tokenizer

    max_tokens = json.loads((model_dir / "sentence_bert_config.json").read_text())["max_seq_length"]
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=max_tokens)
    tokenizer.no_padding()
    return tokenizer, max_tokens


def _count_tokens(tokenizer: Tokenizer, texts: list[str]) -> list[tuple[int, bool]]:
    # With truncation on, `ids` is what the model sees and `overflowing` holds what was cut.
    return [(len(e.ids), bool(e.overflowing)) for e in tokenizer.encode_batch(texts)]


E5_PREFIXES = {InputType.QUERY: "query: ", InputType.PASSAGE: "passage: "}


def read_prefixes(model_dir: Path) -> dict[InputType, str]:
    """The prefix per input type: the model's own search prompts if its
    config_sentence_transformers.json has them (EmbeddingGemma 2 does), else e5's."""
    try:
        config = json.loads((model_dir / "config_sentence_transformers.json").read_text())
        prompts = config["prompts"]
        return {InputType.QUERY: prompts["SearchQuery"], InputType.PASSAGE: prompts["Document"]}
    except (FileNotFoundError, KeyError):
        return E5_PREFIXES


def _read_source(model_dir: Path) -> tuple[str, str]:
    """Model id and revision, as recorded by the script that built the directory."""
    source = json.loads((model_dir / "source.json").read_text())
    return source["model_id"], source["revision"]


class OnnxEmbedder:
    """The int8 model from scripts/export_onnx.py, run with ONNX Runtime instead of torch.

    It does what sentence-transformers does for e5: tokenise, truncate, average the
    token vectors and normalise. Texts go through the model one at a time, because
    dynamic quantisation picks its scale from the whole input; in a batch, a text's
    embedding would depend on the other texts in it.
    """

    backend = "onnx-int8"

    def __init__(self, settings: Settings) -> None:
        import onnxruntime as ort

        model_dir = settings.model_dir
        self.model_name, self.revision = _read_source(model_dir)
        self._tokenizer, self.max_tokens = _load_tokenizer(model_dir)
        options = ort.SessionOptions()
        if settings.num_threads:
            options.intra_op_num_threads = settings.num_threads
        self._session = ort.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.dimension = self._session.get_outputs()[0].shape[-1]

    def count_tokens(self, texts: list[str]) -> list[tuple[int, bool]]:
        return _count_tokens(self._tokenizer, texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for encoding in self._tokenizer.encode_batch(texts):
            input_ids = np.array([encoding.ids], dtype=np.int64)
            [hidden] = self._session.run(
                ["last_hidden_state"],
                {"input_ids": input_ids, "attention_mask": np.ones_like(input_ids)},
            )
            # One text and no padding, so a plain mean over all tokens is the masked mean.
            pooled = np.asarray(hidden)[0].mean(axis=0)
            rows.append(pooled / np.linalg.norm(pooled))
        return np.vstack(rows)


class SentenceTransformerEmbedder:
    """The original fp32 model via sentence-transformers: the reference for the int8 model."""

    backend = "torch-fp32"

    def __init__(self, settings: Settings) -> None:
        # Imported here so the default backend (and the tests) never load torch.
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()  # progress bars are not JSON logs
        if settings.num_threads:
            torch.set_num_threads(settings.num_threads)

        model_dir = settings.model_dir
        self.model_name, self.revision = _read_source(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        # EmbeddingGemma 2 has vision and audio encoders; text only needs neither.
        text_only = {k: None for k in ("vision_config", "audio_config") if k in config}
        self._model = SentenceTransformer(
            str(model_dir),
            local_files_only=True,
            config_kwargs=text_only,
            # EmbeddingGemma 2 ships bf16 weights; fp32 is faster on most CPUs, and fp16 is unsafe.
            model_kwargs={"dtype": torch.float32},
        )
        self._tokenizer, self.max_tokens = _load_tokenizer(model_dir)  # for counting tokens
        dimension = self._model.get_embedding_dimension()
        assert dimension is not None  # always set for a sentence-embedding model
        self.dimension = dimension

    def count_tokens(self, texts: list[str]) -> list[tuple[int, bool]]:
        return _count_tokens(self._tokenizer, texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def load_embedder(settings: Settings) -> Embedder:
    if settings.num_threads:
        # The tokenizer's own thread pool is sized to the host's cores unless told otherwise.
        os.environ.setdefault("RAYON_NUM_THREADS", str(settings.num_threads))
    if settings.backend is Backend.TORCH:
        return SentenceTransformerEmbedder(settings)
    return OnnxEmbedder(settings)


@dataclass(frozen=True)
class EmbedResult:
    vectors: np.ndarray
    token_counts: list[int]
    truncated: list[bool]
    queue_ms: float

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)


class EmbeddingService:
    """Adds the model's prefix, checks the token budget, and runs one inference at a time.

    On a CPU, one request using every core finishes sooner than several sharing them,
    so requests queue for a single slot. They wait at most `queue_timeout_seconds` and
    then get 503. uvicorn does not stop a handler when its client disconnects, so a
    request is dropped if its client has gone by the time it gets the slot; otherwise
    abandoned requests would be computed ahead of live ones.
    """

    def __init__(self, embedder: Embedder, settings: Settings) -> None:
        self.embedder = embedder
        self._prefixes = read_prefixes(settings.model_dir)
        self._max_total_tokens = settings.max_total_tokens
        self._queue_timeout = settings.queue_timeout_seconds
        self._slot = asyncio.Semaphore(1)

    async def embed(
        self,
        texts: list[str],
        input_type: InputType,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> EmbedResult:
        prefixed = [self._prefixes[input_type] + text for text in texts]

        # Tokenising 64 texts of 8000 characters takes a while, so this runs in a thread too.
        counts = await anyio.to_thread.run_sync(self.embedder.count_tokens, prefixed)
        tokens = [n for n, _ in counts]
        if sum(tokens) > self._max_total_tokens:
            raise APIError(
                422,
                "token_budget_exceeded",
                f"Request needs {sum(tokens)} tokens; the limit is {self._max_total_tokens}. "
                "Split it into smaller requests.",
            )

        queued_at = time.perf_counter()
        try:
            await asyncio.wait_for(self._slot.acquire(), self._queue_timeout)
        except TimeoutError:
            raise APIError(
                503,
                "overloaded",
                f"No inference capacity within {self._queue_timeout:g}s. Retry later.",
                headers={"Retry-After": str(math.ceil(self._queue_timeout))},
            ) from None
        queue_ms = round((time.perf_counter() - queued_at) * 1000, 1)

        try:
            if is_disconnected is not None and await is_disconnected():
                raise APIError(499, "client_disconnected", "Client went away while queued.")
            # anyio does not cancel the thread, so the slot stays taken until inference
            # has finished, even if the client leaves.
            vectors = await anyio.to_thread.run_sync(self.embedder.embed, prefixed)
        finally:
            self._slot.release()
        return EmbedResult(
            vectors=vectors,
            token_counts=tokens,
            truncated=[cut for _, cut in counts],
            queue_ms=queue_ms,
        )
