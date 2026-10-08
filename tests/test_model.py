"""Tests against the real models. Run with `just test-slow`.

They need models/e5 (scripts/download_model.py) and models/e5-int8
(scripts/export_onnx.py); a backend whose model is missing is skipped. e5
similarities mostly fall between 0.7 and 1.0, so the tests compare orderings
rather than absolute scores.
"""

import numpy as np
import pytest

from embed_api.config import Backend, Settings
from embed_api.embedder import Embedder, SentenceTransformerEmbedder, load_embedder

pytestmark = pytest.mark.slow

TEXTS = [
    "query: hvad er hovedstaden i Danmark?",
    "passage: København er Danmarks hovedstad og største by.",
    "passage: Bananer er en god kilde til kalium.",
    "query: The cat is sleeping on the couch.",
    "passage: " + "lang tekst " * 400,  # truncated at 512 tokens
]


def _load(backend: Backend) -> Embedder:
    settings = Settings(backend=backend)
    if not settings.model_dir.exists():
        pytest.skip(f"{settings.model_dir} not found; see the README")
    return load_embedder(settings)


@pytest.fixture(scope="module", params=list(Backend), ids=lambda b: b.value)
def model(request: pytest.FixtureRequest) -> Embedder:
    return _load(request.param)


def test_shape_and_norm(model: Embedder) -> None:
    vectors = model.embed(["query: hej", "passage: verden"])

    assert vectors.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_query_prefers_relevant_passage(model: Embedder) -> None:
    query, relevant, irrelevant = model.embed(TEXTS[:3])

    assert query @ relevant > query @ irrelevant


def test_danish_and_english_paraphrases_are_close(model: Embedder) -> None:
    danish, english, unrelated = model.embed(
        [
            "query: Katten sover på sofaen.",
            "query: The cat is sleeping on the couch.",
            "query: Renten steg med et halvt procentpoint.",
        ]
    )

    assert danish @ english > danish @ unrelated


def test_token_count_includes_prefix_and_special_tokens(model: Embedder) -> None:
    [(with_prefix, _)] = model.count_tokens(["query: hej"])
    [(bare, _)] = model.count_tokens(["hej"])

    assert with_prefix > bare >= 3  # <s> hej </s>
    assert model.count_tokens([TEXTS[4]]) == [(512, True)]


def test_counts_match_what_sentence_transformers_feeds_the_model() -> None:
    torch_model = _load(Backend.TORCH)
    assert isinstance(torch_model, SentenceTransformerEmbedder)

    ours = [n for n, _ in torch_model.count_tokens(TEXTS)]
    theirs = torch_model._model.tokenize(TEXTS)["attention_mask"].sum(dim=1).tolist()
    assert ours == theirs


def test_onnx_int8_matches_torch_fp32() -> None:
    torch_model, onnx_model = _load(Backend.TORCH), _load(Backend.ONNX)

    cosines = np.sum(torch_model.embed(TEXTS) * onnx_model.embed(TEXTS), axis=1)
    assert cosines.min() > 0.98, cosines


def test_embedding_does_not_depend_on_the_rest_of_the_batch(model: Embedder) -> None:
    # Dynamic int8 quantisation scales activations per input tensor, so a
    # batched run would let one text's vector depend on its neighbours.
    alone = model.embed([TEXTS[0]])[0]
    batched = model.embed([TEXTS[0], TEXTS[2], TEXTS[4]])[0]

    assert alone @ batched > 0.99999
