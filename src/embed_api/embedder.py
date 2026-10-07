"""Model access, behind a small protocol so tests can use a fake."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import anyio.to_thread
import numpy as np

from embed_api.config import DEFAULT_ONNX_PATH, Backend, Settings
from embed_api.errors import APIError


class InputType(StrEnum):
    """e5 was trained with these prefixes; leaving them out degrades quality."""

    QUERY = "query"
    PASSAGE = "passage"


class Embedder(Protocol):
    model_name: str
    revision: str
    backend: str
    dimension: int
    max_tokens: int

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Untruncated token count per text, special tokens included."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalised embeddings, one row per text. Over-long texts are truncated."""
        ...


class OnnxEmbedder:
    """The int8 model from scripts/export_onnx.py, run with ONNX Runtime.

    Reproduces what sentence-transformers does for e5 (tokenise, truncate to
    512, mean-pool over the attention mask, L2-normalise) without torch. The
    slow test suite checks it against the torch backend.

    Texts are run through the model one at a time. Dynamic quantisation picks
    its activation scale from the whole input tensor, so in a batch each
    text's embedding would depend on the other texts in the request (cosine
    ~0.994 to itself embedded alone). One at a time, a text always gets the
    same vector, for ~5-15% less throughput (and no wasted work on padding).
    """

    def __init__(self, settings: Settings) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        path = Path(settings.model_path or DEFAULT_ONNX_PATH)
        config = json.loads((path / "sentence_bert_config.json").read_text())
        self.max_tokens = int(config["max_seq_length"])
        # Written by export_onnx.py: what the weights actually are, rather
        # than what the settings claim.
        source = json.loads((path / "source.json").read_text())

        # Two tokenizers: one untruncated, to count tokens and flag truncation,
        # and one that truncates for the model.
        self._counter = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._counter.no_truncation()
        self._counter.no_padding()
        self._tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=self.max_tokens)
        self._tokenizer.no_padding()

        options = ort.SessionOptions()
        if settings.num_threads:
            options.intra_op_num_threads = settings.num_threads
        self._session = ort.InferenceSession(
            str(path / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.model_name = source["model_id"]
        self.revision = source["revision"]
        self.backend = "onnx-int8"
        self.dimension = int(self._session.get_outputs()[0].shape[-1])

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(encoding.ids) for encoding in self._counter.encode_batch(texts)]

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for encoding in self._tokenizer.encode_batch(texts):
            # One unpadded text per run, so mean pooling over all tokens is
            # mean pooling over the attention mask.
            input_ids = np.array([encoding.ids], dtype=np.int64)
            (hidden,) = self._session.run(
                ["last_hidden_state"],
                {"input_ids": input_ids, "attention_mask": np.ones_like(input_ids)},
            )
            pooled = hidden[0].mean(axis=0)
            rows.append(pooled / np.linalg.norm(pooled))
        return np.vstack(rows).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, settings: Settings) -> None:
        # Imported here so the API (and its tests) start without loading torch.
        import torch
        from sentence_transformers import SentenceTransformer
        from tokenizers import Tokenizer
        from transformers.utils import logging as hf_logging

        # The tokenizer warns on every over-long input; truncation is reported
        # per input in the response instead. Progress bars are not JSON logs.
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        if settings.num_threads:
            torch.set_num_threads(settings.num_threads)

        if settings.model_path:
            # A local directory must not fall back to (or even ask) the Hub.
            self._model = SentenceTransformer(
                settings.model_path, device=settings.device, local_files_only=True
            )
        else:
            self._model = SentenceTransformer(
                settings.model_id, revision=settings.model_revision, device=settings.device
            )
        self._batch_size = settings.encode_batch_size
        self.model_name = settings.model_id
        self.revision = settings.model_revision
        self.backend = "torch-fp32"
        self.dimension = self._model.get_embedding_dimension()
        self.max_tokens = self._model.max_seq_length
        # Counting gets its own tokenizer. Token counting runs outside the
        # inference semaphore, and the transformers tokenizer reconfigures its
        # shared backend's truncation on every call, so sharing one let a
        # concurrent count switch truncation off mid-encode (>512 tokens -> 500).
        self._counter = Tokenizer.from_str(self._model.tokenizer.backend_tokenizer.to_str())
        self._counter.no_truncation()
        self._counter.no_padding()

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(encoding.ids) for encoding in self._counter.encode_batch(texts)]

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def load_embedder(settings: Settings) -> Embedder:
    if settings.num_threads:
        # The tokenizers library has its own thread pool, sized to the host's
        # cores unless told otherwise; it is created on first use.
        os.environ.setdefault("RAYON_NUM_THREADS", str(settings.num_threads))
    if settings.backend is Backend.TORCH:
        return SentenceTransformerEmbedder(settings)
    return OnnxEmbedder(settings)


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
