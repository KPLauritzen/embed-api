"""Checks against the real model. Run with `uv run pytest -m slow`.

Set EMBED_MODEL_PATH to a downloaded model (scripts/download_model.py) to
avoid fetching from the Hub. e5 similarities cluster in 0.7-1.0, so the
assertions compare orderings rather than absolute thresholds.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from embed_api.config import Settings

pytestmark = pytest.mark.slow

DEFAULT_PATH = Path(__file__).parent.parent / "models" / "e5"


@pytest.fixture(scope="module")
def model():
    from embed_api.embedder import SentenceTransformerEmbedder

    path = os.environ.get("EMBED_MODEL_PATH") or (DEFAULT_PATH if DEFAULT_PATH.exists() else None)
    return SentenceTransformerEmbedder(Settings(model_path=str(path) if path else None))


def test_shape_and_norm(model) -> None:
    vectors = model.embed(["query: hej", "passage: verden"])

    assert vectors.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_query_prefers_relevant_passage(model) -> None:
    query, relevant, irrelevant = model.embed(
        [
            "query: hvad er hovedstaden i Danmark?",
            "passage: København er Danmarks hovedstad og største by.",
            "passage: Bananer er en god kilde til kalium.",
        ]
    )

    assert query @ relevant > query @ irrelevant


def test_danish_and_english_paraphrases_are_close(model) -> None:
    danish, english, unrelated = model.embed(
        [
            "query: Katten sover på sofaen.",
            "query: The cat is sleeping on the couch.",
            "query: Renten steg med et halvt procentpoint.",
        ]
    )

    assert danish @ english > danish @ unrelated


def test_token_count_includes_prefix_and_special_tokens(model) -> None:
    [with_prefix] = model.count_tokens(["query: hej"])
    [bare] = model.count_tokens(["hej"])

    assert with_prefix > bare >= 3  # <s> hej </s>
    assert model.max_tokens == 512
